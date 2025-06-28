import dataclasses
from collections import Counter
import torch
import time

@dataclasses.dataclass
class NoThreeInLine:
    """
    No-three-in-line.
    """
    batch_size: int
    grid_size: int  
    max_points: int
    device: str = "cpu"
    
    def __post_init__(self):
        N = self.N = self.grid_size
        assert self.grid_size < 64, "Grid size must fit in int8 coordinate."
        assert self.max_points < 2*N + 1, "max_points should be less than or equal to 2*N + 1."

        # self.current_constructions: An (N, N) grid for each batch item.
        # Value `1` marks a placed point, `0` is empty, `-1` is forbidden. This is for fast
        # checking of whether a specific coordinate is occupied.
        self.current_constructions = torch.zeros(
            (self.batch_size, N, N),
            dtype=torch.int8,
            device=self.device,
        )

        # self.points_list: A list of coordinates for each batch item.
        # Stores up to `max_points` (x, y) pairs. Unused slots are `[-1, -1]`.
        # This is for efficient retrieval of all points to check for collinearity.
        self.points_list = torch.full(
            (self.batch_size, self.max_points, 2),
            -1,
            dtype=torch.int8,  # N can be < 64
            device=self.device,
        )

        # self.current_counts: Tracks the number of points in each construction.
        self.current_counts = torch.zeros(
            (self.batch_size,),
            dtype=torch.int8,  # max_points can be < 2*N
            device=self.device,
        )

        # self.null_tensor: A placeholder for an invalid or unused point.
        self.null_tensor = torch.tensor(
            [-1, -1],
            dtype=torch.int8,
            device=self.device,
        )

        # cache for pair indices used in collinearity checks
        self._pair_cache = {}

    def add_points(self, points, verbose=True):
        """
        Add points to constructions, updating internal state.
        
        Flow:
        1. Identify valid points (non-null, within bounds, and on available squares).
        2. Group batches by their current number of points.
        3. In a vectorized way, for each group, update both the grid and point list.
        4. Increment point counts for batches where a point was added.
        
        Args:
            points: tensor of shape (batch_size, 2) with (x, y) coordinates.
            verbose: whether to print addition info (unused in this implementation).
        """
        points = points.to(dtype=torch.int8, device=self.device)
        points_int = points.to(dtype=torch.int)  # For indexing

        # Create a mask for non-null points (i.e., rows that really add something)
        non_null_mask = (points != self.null_tensor).any(dim=-1)

        # Assertions for safety
        assert points.shape[0] == self.batch_size
        if non_null_mask.any():
            assert torch.max(self.current_counts[non_null_mask]).item() < self.max_points
        assert (-1 <= points).all() and (points < self.N).all()

        # Get indices for all batches and for only non-null ones
        batch_indices_all = torch.arange(self.batch_size, device=self.device)
        valid_batch_indices = batch_indices_all[non_null_mask]
        
        # If no valid points, exit early
        if valid_batch_indices.numel() == 0:
            return

        # Check occupancy for valid points
        coords_to_check = points_int[non_null_mask]
        occupancy_status = self.current_constructions[valid_batch_indices, coords_to_check[:, 0], coords_to_check[:, 1]]
        is_available = (occupancy_status == 0)

        # Final mask of batches where a point should be added
        final_add_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        # Place `True` at the indices of batches that had a valid point and an available square
        final_add_mask[valid_batch_indices[is_available]] = True

        added_point_counts = torch.zeros_like(self.current_counts)

        # Loop over unique counts for batched updates
        for cur_count in torch.unique(self.current_counts).tolist():
            count_mask = (self.current_counts == cur_count)
            # Find the intersection of points to add and batches with the current count
            batch_insertions_mask = count_mask & final_add_mask
            
            if not batch_insertions_mask.any():
                continue

            # Get the indices and coordinates of points to add for this group
            batch_insertion_indices = batch_indices_all[batch_insertions_mask]
            points_to_add = points[batch_insertions_mask]
            points_to_add_int = points_int[batch_insertions_mask]

            # Update construction grid
            self.current_constructions[batch_insertion_indices, points_to_add_int[:, 0], points_to_add_int[:, 1]] = 1
            
            # Update points list
            self.points_list[batch_insertion_indices, cur_count] = points_to_add
            
            # Mark that we added a point to these batches
            added_point_counts[batch_insertion_indices] = 1

        # Update the total counts
        self.current_counts += added_point_counts

    def check_new_points(self, new_points, verbose=False):
        """
        Check which points can be added without creating three-in-a-line.
        
        This implementation is fully vectorized to process all constructions in parallel.
        
        Args:
            new_points: tensor of shape (batch_size, num_candidates, 2) with candidate points.
            verbose: whether to print collinearity info.
        Returns: 
            A boolean mask of shape (batch_size, num_candidates).
        """
        
        num_candidates = new_points.shape[1]

        # 1. Basic validation (vectorized)
        # Check for null points
        good_bools = (new_points != self.null_tensor).any(dim=-1)

        # Check bounds
        in_bounds = (new_points >= 0).all(dim=-1) & (new_points < self.N).all(dim=-1)
        good_bools &= in_bounds
        
        # Check occupancy
        # Create indices for gathering
        batch_indices_int = torch.arange(self.batch_size, device=self.device).unsqueeze(1).expand(-1, num_candidates)
        
        # We only check valid points to avoid index errors
        points_to_check = new_points[good_bools].to(torch.int)
        batch_indices_to_check = batch_indices_int[good_bools]
        
        # Gather occupancy status
        occupancy_status = self.current_constructions[batch_indices_to_check, points_to_check[:, 0], points_to_check[:, 1]]
        
        # Create a full occupancy mask and update it
        is_occupied = torch.zeros_like(good_bools)
        is_occupied[good_bools] = (occupancy_status != 0)
        good_bools &= ~is_occupied

        # 2. Collinearity check (vectorized)
        # Only iterate over counts that actually occur (>1)
        unique_counts = torch.unique(self.current_counts)

        for cur_count in unique_counts.tolist():
            if cur_count < 2:
                continue
            # Find all constructions that currently have `cur_count` points
            batch_mask = (self.current_counts == cur_count)
            if not batch_mask.any():
                continue

            # Get the indices of the relevant batches
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]
            
            # Get existing points and candidates for these batches
            existing_points = self.points_list[batch_indices] # (num_batches, max_points, 2)
            candidates = new_points[batch_indices]            # (num_batches, num_candidates, 2)
            
            # Retrieve or compute cached pair indices for this cur_count
            if cur_count not in self._pair_cache:
                self._pair_cache[cur_count] = torch.combinations(torch.arange(cur_count), r=2)
            pair_indices = self._pair_cache[cur_count].to(self.device)  # (num_pairs, 2)
            p1s = existing_points[:, pair_indices[:, 0]] # (num_batches, num_pairs, 2)
            p2s = existing_points[:, pair_indices[:, 1]] # (num_batches, num_pairs, 2)

            # Reshape for broadcasting
            # p1s/p2s: (num_batches, num_pairs, 1, 2)
            # candidates: (num_batches, 1, num_candidates, 2)
            p1s = p1s.unsqueeze(2)
            p2s = p2s.unsqueeze(2)
            p3s = candidates.unsqueeze(1)

            # Extract coordinates for broadcasting
            x1, y1 = p1s[..., 0], p1s[..., 1] # (num_batches, num_pairs, 1)
            x2, y2 = p2s[..., 0], p2s[..., 1] # (num_batches, num_pairs, 1)
            x3, y3 = p3s[..., 0], p3s[..., 1] # (num_batches, 1, num_candidates)

            # Perform the collinearity check for all pairs and candidates at once
            # Result shape: (num_batches, num_pairs, num_candidates)
            collinearity_check = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            
            # A candidate is invalid if ANY pair is collinear with it
            is_collinear = (collinearity_check == 0)
            invalid_candidates = is_collinear.any(dim=1) # (num_batches, num_candidates)
            
            # Update the good_bools mask for the relevant batches
            good_bools[batch_indices] &= ~invalid_candidates
            
            # Update forbidden positions
            forbidden_points = new_points[batch_indices][invalid_candidates].to(torch.int)
            forbidden_batch_indices = batch_indices.unsqueeze(1).expand(-1, num_candidates)[invalid_candidates]
            
            if forbidden_points.numel() > 0:
                self.current_constructions[forbidden_batch_indices, forbidden_points[:, 0], forbidden_points[:, 1]] = -1

        return good_bools


    def possible_additions(self, shuffle=False):
        """
        Return a tensor of shape (batch_size, k, 2) indicating coordinates of possible additions.
        (Here k is the maximum over all batches of allowable points added.)
        
        Args:
            shuffle: whether to randomize order of candidates
        Returns:
            tensor of shape (batch_size, max_candidates, 2) with coordinates
        """
        
        possible_additions = (self.current_constructions == 0).nonzero(as_tuple=False).to(torch.int)

        if shuffle:
            indices = torch.randperm(possible_additions.size(0))
            possible_additions = possible_additions[indices]
            sorted_indices = torch.argsort(possible_additions[:,0])
            possible_additions = possible_additions[sorted_indices]

        if possible_additions.shape[0] > 0:
            max_newpoints = torch.max(torch.bincount(possible_additions[:,0])) 
        else:
            return torch.empty((self.batch_size,0,2),device=self.device)

        unique_non_zero_first_coords, nonzero_counts = torch.unique(possible_additions[:,0], return_counts=True)

        max_count = nonzero_counts.max().item()

        nonzero_counts = nonzero_counts.to(torch.int)

        counts = torch.zeros((self.batch_size,),dtype=torch.int,device=self.device)
        counts[unique_non_zero_first_coords] = nonzero_counts

        mask = torch.arange(max_count,device=self.device).expand(self.batch_size, max_count) < counts.unsqueeze(1)
        result_tensor = -1 * torch.ones((self.batch_size, max_count, 2),dtype=torch.int8,device=self.device)
        result_tensor[mask] = possible_additions[:, 1:].to(torch.int8)

        return result_tensor

    def propose_additions_batched(self):
        """Propose additions by batching over new possibilities"""
        current_proposals = -1*torch.ones((self.batch_size,2),dtype=torch.int8,device=self.device)

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

            live_batches[new_successes] = False

            if not live_batches.any():
                break

        return current_proposals
    
    @torch.no_grad()
    def saturate(self):
        """
        Complete all constructions randomly until addition of any more points is impossible.
        """
        for _ in range(self.max_points):
            pa = self.propose_additions_batched()
            self.add_points(pa)

    def try_to_add_points(self, points):
        """
        points is a tensor of shape (batch_size, 2)
        we add each point if it satisfies constraints
        """
        can_add = self.check_new_points(points.unsqueeze(1)).squeeze(1)
        points[~can_add] = self.null_tensor
        self.add_points(points)
    

def print_grid(construction_grid, title="Construction"):
    """Prints a 2D text representation of a single construction."""
    grid_list = construction_grid.cpu().tolist()
    N = len(grid_list)
    
    print(f"\n{title}")
    for r in range(N):
        row_str = []
        for c in range(N):
            if grid_list[r][c] == 1:
                row_str.append('X')
            else:
                row_str.append('.')
        print(' '.join(row_str))


if __name__ == "__main__":

    N = 12
    solver = NoThreeInLine(batch_size=1000000, grid_size=N, max_points=2*N)
    t0 = time.time()
    solver.saturate()
    t1 = time.time()
    print(f"Saturation took {t1-t0:.2f} seconds.")

    t2 = time.time()
    print("Test completed!")
    print(f"Total time: {t2-t0:.2f} seconds.")
    
    # print("Final point counts per batch:", solver.current_counts.tolist())
    print("Max points found in a construction:", torch.max(solver.current_counts).item())
    print("Total constructions:", solver.batch_size)

    points_counter = Counter(solver.current_counts.tolist())
    print(points_counter)



    # Print all constructions in the batch
    if solver.batch_size > 0:
        # Find top 5 constructions by point count
        top_indices = torch.topk(solver.current_counts, min(1, solver.batch_size)).indices
        
        print("\n--- Top Final Constructions ---")
        for i, idx in enumerate(top_indices):
            construction = solver.current_constructions[idx]
            num_points = solver.current_counts[idx].item()
            print_grid(
                construction,
                title=f"\nTop Construction #{i+1} (found {num_points} points on a {solver.N}x{solver.N} grid)"
            )