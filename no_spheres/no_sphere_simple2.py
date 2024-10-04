
import torch
from math import comb
import dataclasses
#import matplotlib.pyplot as plt
import itertools
import numpy as np
from sympy import Matrix

#from tqdm.auto import tqdm
import time

@dataclasses.dataclass
class NoSphereSimple:
    """

    We store current_constructions as a torch tensor.
     0 indicates point we can take.
     1 indicates point we have taken.
    -1 indicates point we can't take.

    (We try to be mindful of GPU memory, and hence try to allocate a big block once and reuse it,
    rather than haphazard creation and deletion of smaller blocks.
    We also try to use int8's whenever we can.)

    We also store:
    1) current counts (i.e. how many 1s in current_constructions), for convenience only;
    2) 1-tuples, 2-tuples and 3-tuples;
    3) 4-tuples of points added so far.

    Some motivation for this:

    Clearly we need some variant of 3), otherwise we'd have to recompute everything every time.

    [We do many many cases of "is this the sphere with centre and radisu blah already present?",
    so it seems preferable to implement this with a hash table.]

    In order to check whether a possible new addition doesn't lie on an existing sphere, we can check that
    the sphere it forms with all existing 3-tuples is not already on our list.

    In order to do this we need to have a list of all current 3-tuples.

    Now when we add a new point, either we recompute 3-tuples, or we know all 2-tuples.
    Now when we add a new point, either we recompute 2-tuples, or we know all 1-tuples.

    Hence we store 1), 2) and 3).

    """
    batch_size: int                      # batch size
    grid_size: int                       # grid size
    max_points: int                      # max number of points we can add
    device: str = 'cpu'                  # where we do our work


    def __post_init__(self):
        self.N = self.grid_size
        assert self.grid_size < 64
        assert self.max_points < 64

        N = self.N
        self.current_constructions = torch.zeros((self.batch_size, N, N, N),dtype=torch.int8,device=self.device)
        self.current_counts = torch.zeros((self.batch_size,),dtype=torch.int8,device=self.device)

        # It is surely better to keep track of 1, 2, 3, 4-tuples of points
        # rather than generating them on the fly:

        self.tuples = {}                  # dictionary of tuples of current points
        for k in range(1,5):
            self.tuples[k] = -1 * torch.ones((self.batch_size,comb(self.max_points,k),k,3),dtype=torch.int8,device=self.device)

        self.null_tensor = torch.tensor([-1,-1,-1],dtype=torch.int8,device=self.device)

    def add_points(self,points,verbose=True):

        """
        A batched version of adding a single point to an a construction.

        Takes a tensor of shape (B,3) and updates:
        current_constructions, current_counts, tuples

        We need to allow adding of "null-point" so that points are only added to some constructions.
        This is done by sending the null vector (typically [-1,-1,-1])

        For example, if batch_size = 2 then

        NoSphereSimple.add_points(torch.tensor([[0,0,0],[-1,-1,-1]]))

        adds [0,0,0] to the first construction, and does nothing to the second.

        """

        points = points.to(torch.int8)
        points_int = points.to(torch.int) # torch wants ints for indexing arrays

        assert points.shape[0] == self.batch_size
        assert torch.max(self.current_counts).item() < self.max_points
        assert (-1 <= points).all() and (points < self.max_points).all()

        non_null_mask = (points != self.null_tensor).any(dim=1)                # shape (B,)

        # we don't want to add any points which are already present
        tuples = self.tuples[1].squeeze(2)                                     # shape (B, k, 3)
        matches = (tuples == points.unsqueeze(1)).all(dim=-1).any(dim=1)       # shape (B,)
        non_silly_matches = torch.logical_and(matches,non_null_mask)           # shape (B,)
#        if non_silly_matches.any():
#            print("Encountered points which are already present.")
#            print(f"Indices {torch.nonzero(matches).tolist()} are already present.")

        distinct_point_mask = ~non_silly_matches                               # shape (B,)

        # tensor for remembering which indices we touched.
        added_point = torch.zeros(self.batch_size,dtype=torch.int8,device=self.device)

        for cur_count in torch.unique(self.current_counts).tolist():

            count_mask = (self.current_counts == cur_count)

            relevant_indices = count_mask & non_null_mask & distinct_point_mask

            batch_insertions = torch.arange(self.batch_size,device=self.device)[relevant_indices] # shape (B',)

            if batch_insertions.size(0) == 0: continue

            # update current_constructions
            current_points = points_int[batch_insertions]
            self.current_constructions[batch_insertions,current_points[:,0],current_points[:,1],current_points[:,2]] = 1

            # update tuples
            unsqueezed_points = points[batch_insertions].unsqueeze(1).unsqueeze(2) ## shape (B, 1, 1, 3)

            if cur_count >=3:
                expanded_points = unsqueezed_points.expand(-1,comb(cur_count,3),1,3)
                fill_from = comb(cur_count,4)
                fill_to = comb(cur_count,4) + comb(cur_count,3)
                self.tuples[4][batch_insertions,fill_from:fill_to,:,:] = torch.cat((self.tuples[3][batch_insertions,:comb(cur_count,3)], expanded_points), dim=2)

            if cur_count >=2:
                expanded_points = unsqueezed_points.expand(-1,comb(cur_count,2),1,3)
                fill_from = comb(cur_count,3)
                fill_to = comb(cur_count,3) + comb(cur_count,2)
                self.tuples[3][batch_insertions,fill_from:fill_to,:,:] = torch.cat((self.tuples[2][batch_insertions,:comb(cur_count,2)], expanded_points), dim=2)

            if cur_count >=1:
                expanded_points = unsqueezed_points.expand(-1,comb(cur_count,1),1,3)
                fill_from = comb(cur_count,2)
                fill_to = comb(cur_count,2) + comb(cur_count,1)
                self.tuples[2][batch_insertions,fill_from:fill_to,:,:] = torch.cat((self.tuples[1][batch_insertions,:comb(cur_count,1)], expanded_points), dim=2)

            self.tuples[1][batch_insertions,cur_count:cur_count+1,:,:] = unsqueezed_points

            # remember that we have changed this batch index
            added_point[batch_insertions] += 1

        self.current_counts += added_point

    def validate(self):
        """
        We run a few consistency checks.

        This is only for debugging. We don't care about speed.

        """

        OK = True

        # check current counts
        construction_counts = torch.sum(torch.sum(torch.sum(self.current_constructions==1,dim=-1),dim=-1),dim=-1)
        if not (construction_counts == self.current_counts).all():
            print("current counts appears to be off")
            OK = False

        # check tuples for consistency
        for b in range(self.batch_size):
            for k in range(1,5):
                if not (self.tuples[k][b,:comb(self.current_counts[b],k)] >= 0).all():
                    print(f"Unexpected negative in {k}-tuples for batch {b}.")
                    OK = False
                if not (self.tuples[k][b,comb(self.current_counts[b],k):] == -1).all():
                    print(f"Unexpected positive in {k}-tuples for batch {b}.")
                    OK = False

        for b in range(self.batch_size):
            if self.current_counts[b] > 0:
                x = self.tuples[1][b][:comb(self.current_counts[b],1)].view(-1,3).to(torch.int)
                if not (self.current_constructions[b][x[:,0],x[:,1],x[:,2]] == 1).all():
                    print(f"1-tuples and current constructions mismatch.")
                    OK = False

        return OK

    def possible_additions(self,shuffle=False):
        """

        Return a tensor of shape (B, k, 3) indicating coordinates of possible additions.

        (Here k is the maximum over all batches of allowable points added.)

        Nothing mathematical happens here. This is just an exercise is manipulating
        torch tensors without using for loops.

        """

        possible_additions = (self.current_constructions == 0).nonzero(as_tuple=False).to(torch.int) # shape (batch_size, 4)

        if shuffle:
            indices = torch.randperm(possible_additions.size(0))
            possible_additions = possible_additions[indices]
            sorted_indices = torch.argsort(possible_additions[:,0])
            possible_additions = possible_additions[sorted_indices]

        if possible_additions.shape[0] > 0:
            max_newpoints = torch.max(torch.bincount(possible_additions[:,0])) #
        else:
            return torch.empty((self.batch_size,0,3),device=self.device) # none of our configurations allow additions of points.

        unique_non_zero_first_coords, nonzero_counts = torch.unique(possible_additions[:,0], return_counts=True)

        max_count = nonzero_counts.max().item()

        nonzero_counts = nonzero_counts.to(torch.int)

        counts = torch.zeros((self.batch_size,),dtype=torch.int,device=self.device)
        counts[unique_non_zero_first_coords] = nonzero_counts # a tensor of shape (B,) giving counts

        mask = torch.arange(max_count,device=self.device).expand(self.batch_size, max_count) < counts.unsqueeze(1)
        result_tensor = -1 * torch.ones((self.batch_size, max_count, 3),dtype=torch.int8,device=self.device)
        result_tensor[mask] = possible_additions[:, 1:].to(torch.int8)

        return result_tensor

    def check_new_points(self,new_points,verbose=False):
        """

        new_points is a tensor of shape (B,k,3).

        We return a boolean array of shape (B,k) indicating which points can be added.

        When we discover that one of new_points would add spheres,
        we update self.current_constructions with -1s to reflect this knowledge.
        (I.e. the checked point which would violate no-spheres becomes forbidden.)

        """

        def printv(x):
            if verbose: print(x)

        # for remembering which points are addable:
        good_bools = torch.zeros(new_points.shape[:2],dtype=torch.bool,device=self.device)

        non_null_mask = (new_points != self.null_tensor).any(dim=-1)                # shape (B,k)

        for cur_count in torch.unique(self.current_counts):

            batch_insertions = torch.arange(self.batch_size,device=self.device)[self.current_counts == cur_count]

            if cur_count < 4:
                printv(f"Current count < 4, no conditions to check.")
                good_bools[batch_insertions] == non_null_mask[batch_insertions]

            l = comb(cur_count,4)
            Bp, k, _ = new_points[batch_insertions].shape

            printv(f"Addding {k} points per batch, need to check {l} condition(s).")

            # self.tuples[4] is of shape (B, l, 4, 3]
            four_tuples = self.tuples[4][batch_insertions, :l].unsqueeze(1)             # shape (Bp, 1, l, 4, 3)
            trimmed_new_points = new_points[batch_insertions].unsqueeze(2).unsqueeze(2) # shape (Bp, k, 1, 1, 3)

            four_tuples = four_tuples.expand(Bp,k,l,4,3)
            unsqueezed_new_points = trimmed_new_points.expand(Bp,k,l,1,3)
            big_tensor = torch.cat((four_tuples,unsqueezed_new_points),dim=3)           # shape (Bp, k, l, 5, 3)

            sum_squares = torch.sum(big_tensor**2, dim=-1).unsqueeze(-1)                # shape (Bp, k, l, 5, 1)
            ones = torch.ones((Bp,k,l,5,1),device=self.device)                          # shape (Bp, k, l, 5, 1)

            even_bigger_tensor = torch.cat((big_tensor,sum_squares,ones),dim=-1)        # shape (Bp, k, l, 5, 5)

            dets = torch.linalg.det(even_bigger_tensor)                                 # shape (Bp, k, l)

            coplanar_or_spherical = (torch.abs(dets) < 0.1).any(dim=-1)                 # shape (Bp, k)

            batch_indices = (batch_insertions.unsqueeze(1)).expand(Bp,k)                # shape (Bp, k)


    #        trimmed_new_points = trimmed_new_points.squeeze(2).squeeze(2)
    #        bad_points = trimmed_new_points[coplanar_or_spherical].to(torch.int)

    #        print(f"bad_points.shape={bad_points.shape}")

            bad_batch_indices = batch_indices[coplanar_or_spherical]       # shape (blah,)

            # update current_constructions:
            bad_bools = torch.zeros(new_points.shape[:-1],dtype=torch.bool,device=self.device)
            bad_bools[batch_insertions] = coplanar_or_spherical
            bad_bools = torch.logical_and(bad_bools,non_null_mask)

            bad_batch_indices = torch.arange(self.batch_size,device=self.device).unsqueeze(1).expand(-1,k)[bad_bools]
            bad_points = new_points[bad_bools].to(torch.int)

    #        print("bbi,bp:",torch.cat((bad_batch_indices.unsqueeze(1),bad_points),dim=1))
            self.current_constructions[bad_batch_indices,bad_points[:,0],bad_points[:,1],bad_points[:,2]] = -1

            # finally return positions where addition _is_ possible

 #           good_bools = torch.zeros(new_points.shape[:-1],dtype=torch.bool,device=self.device)
            good_bools[batch_insertions] = torch.logical_not(coplanar_or_spherical)

        good_bools = torch.logical_and(good_bools,non_null_mask)

        return good_bools

    def propose_additions_unbatched(self):

        all_possible_additions = self.possible_additions(shuffle=True)
        possible = self.check_new_points(all_possible_additions)

        batch_fill = torch.arange(self.batch_size,device=self.device).unsqueeze(1).expand(possible.shape)

        proposed_additions = -1 * torch.ones((self.batch_size,3),dtype=torch.int8)

        if possible.any():
            indices = torch.cat((torch.tensor([0],device=self.device), (torch.diff(batch_fill[possible]) != 0).nonzero(as_tuple=True)[0] + 1))
            proposed_additions[batch_fill[possible][indices]] = all_possible_additions[possible][indices]

        return proposed_additions

    def propose_additions_batched(self):
        """Propose additions by batching over new possibilities"""

        current_proposals = -1*torch.ones((self.batch_size,3),dtype=torch.int8,device=self.device)

        all_possible_additions = self.possible_additions(shuffle=True)

        if all_possible_additions.shape[1] == 0:
            return current_proposals

        live_batches = (all_possible_additions[:,0] != self.null_tensor).any(dim=-1) # shape (B,)

        sB = 10

        for k in range(0,all_possible_additions.shape[1],sB):

            possible = self.check_new_points(all_possible_additions[:,k:k+sB]) # shape (B,sB)
            batch_fill = torch.arange(self.batch_size,device=self.device).unsqueeze(1).expand(possible.shape)

            if possible.any():
                indices = torch.cat((torch.tensor([0],device=self.device), (torch.diff(batch_fill[possible]) != 0).nonzero(as_tuple=True)[0] + 1))
                current_proposals[batch_fill[possible][indices]] = all_possible_additions[:,k:k+sB][possible][indices]

            successful_batches = possible.any(dim=-1)                          # shape B

            new_successes = torch.logical_and(live_batches,successful_batches).nonzero()

#            print(f"current_proposals[new_successes].shape={current_proposals[new_successes].shape}")

            live_batches[new_successes] = False

            if not live_batches.any():
                break

        return current_proposals

    def saturate(self):
        """
        Complete all constructions randomly until addition of any more points is impossible.
        """

        for _ in range(self.max_points):
            pa = self.propose_additions_batched()
            self.add_points(pa)

    def try_to_add_points(self,points):
        """
        points is a tensor of shape (B,3)

        we add each

        """
        can_add = self.check_new_points(points.unsqueeze(1)).squeeze(1)   # points.unsqueeze(1).shape = (B,1,3)
        points[~can_add] = self.null_tensor
        self.add_points(points)
