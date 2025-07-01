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
            dtype=torch.int8,  # max_points can be <= 2*N
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

    def add_points(self, points):
        """
        Add points to constructions, updating internal state.
        
        Flow:
        1. Identify valid points (non-null, within bounds, and on available squares).
        2. Group batches by their current number of points.
        3. In a vectorized way, for each group, update both the grid and point list.
        4. Increment point counts for batches where a point was added.
        
        Args:
            points: tensor of shape (batch_size, 2) with (x, y) coordinates.
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

    def check_new_points(self, new_points):
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

    @torch.no_grad()
    def greedy_saturate(self):
        """
        Complete all constructions greedily until addition of any more points is impossible.
        """
        for i in range(self.batch_size):
            current_grid = self.current_constructions[i].clone()

            while True:
                # Find all empty spaces on the current grid
                possible_points = self.available_spaces(current_grid)

                if possible_points.shape[0] == 0:
                    break  # No more empty spaces

                # Set up the inputs for best_grid: one grid for each possible point
                num_candidates = possible_points.shape[0]
                expanded_grids = current_grid.unsqueeze(0).repeat(num_candidates, 1, 1)

                # Find the best grid after trying all possible single-point additions
                best_next_grid = self.best_grid(expanded_grids, possible_points)

                # Check if we made progress
                if (best_next_grid == 1).sum() > (current_grid == 1).sum():
                    current_grid = best_next_grid
                    # After adding a point, update the grid to mark any newly formed forbidden squares
                    # We unsqueeze to add a temporary batch dimension for the vectorized function
                    current_grid = self.update_forbidden_squares(current_grid.unsqueeze(0)).squeeze(0)
                else:
                    # No valid point could be added to improve the score, so we're done
                    break
            
            # Update the final construction and its point count
            self.current_constructions[i] = current_grid
            self.current_counts[i] = (current_grid == 1).sum().item()
            

    def update_forbidden_squares(self, grids):
        """
        For a batch of grids, this function updates each grid by marking newly 
        forbidden squares with -1. This is the batched version.
        """
        point_counts = (grids == 1).sum(dim=(1, 2))
        
        unique_counts = torch.unique(point_counts)

        for count in unique_counts:
            if count < 2:
                continue
                
            # Identify all grids in the batch that have `count` points
            count_mask = (point_counts == count)
            group_indices = count_mask.nonzero(as_tuple=True)[0]
            group_grids = grids[group_indices]
            
            # This part is tricky to vectorize fully because the number of empty
            # squares can differ for each grid in the group. We iterate through 
            # the grids within the group, which is still much better than iterating 
            # through the entire batch.
            
            # Get existing points for all grids in the group at once
            existing_nz = (group_grids == 1).nonzero(as_tuple=False)
            if existing_nz.numel() == 0: continue
            
            # Reshape to (num_group_grids, count, 2)
            existing_points = existing_nz[:, 1:].reshape(group_grids.shape[0], count.item(), 2)

            # Generate pairs of points; this is the same for every grid in the group
            pair_indices = torch.combinations(torch.arange(count.item(), device=self.device), r=2)
            p1s = existing_points[:, pair_indices[:, 0]]  # (num_group_grids, num_pairs, 2)
            p2s = existing_points[:, pair_indices[:, 1]]  # (num_group_grids, num_pairs, 2)
            
            # Iterate through the group to handle varying numbers of empty squares
            for i, original_grid_idx in enumerate(group_indices):
                grid_p1s = p1s[i]  # (num_pairs, 2)
                grid_p2s = p2s[i]  # (num_pairs, 2)
                
                empty_squares = (grids[original_grid_idx] == 0).nonzero(as_tuple=False)
                if empty_squares.shape[0] == 0:
                    continue
                
                # Reshape for broadcasting
                p3s = empty_squares.unsqueeze(0)  # (1, num_empty, 2)
                
                x1 = grid_p1s[:, 0].unsqueeze(1)
                y1 = grid_p1s[:, 1].unsqueeze(1)
                x2 = grid_p2s[:, 0].unsqueeze(1)
                y2 = grid_p2s[:, 1].unsqueeze(1)
                x3 = p3s[..., 0]
                y3 = p3s[..., 1]
                
                # Check for collinearity
                collinearity_check = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
                is_collinear = (collinearity_check == 0).any(dim=0)
                forbidden_points = empty_squares[is_collinear]
                
                if forbidden_points.numel() > 0:
                    grids[original_grid_idx, forbidden_points[:, 0], forbidden_points[:, 1]] = -1
                    
        return grids

    def add_points_to_grid(self, grids, points):
        """
        Add points to grids simultaneously where point i is added to grid i.
        
        Args:
            grids: tensor of shape (batch_size, N, N) 
            points: tensor of shape (batch_size, 2) with coordinates to add
        
        Returns:
            new_grids: tensor of shape (batch_size, N, N) with points added where valid
        """
        points = points.to(torch.int)
        
        # 1. Initial Filtering (null, bounds, occupancy)
        non_null_mask = (points != self.null_tensor).any(dim=-1)
        if not non_null_mask.any(): return grids
        
        batch_indices = torch.arange(grids.shape[0], device=self.device)[non_null_mask]
        valid_points = points[non_null_mask]
        valid_grids = grids[non_null_mask]
        
        x, y = valid_points[:, 0], valid_points[:, 1]
        in_bounds = (x >= 0) & (x < self.N) & (y >= 0) & (y < self.N)
        if not in_bounds.any(): return grids
        
        available = (valid_grids[torch.arange(valid_grids.shape[0]), x, y] == 0)
        final_mask = in_bounds & available
        if not final_mask.any(): return grids

        # Apply final mask to get the set of grids and points we'll actually check
        batch_indices = batch_indices[final_mask]
        valid_points = valid_points[final_mask]
        valid_grids = valid_grids[final_mask]
        
        # 2. Group by point count and check collinearity
        point_counts = (valid_grids == 1).sum(dim=(1, 2))
        can_add_mask = torch.zeros_like(point_counts, dtype=torch.bool)

        unique_counts = torch.unique(point_counts)
        for count in unique_counts:
            # Find all grids in our valid set that have `count` points
            count_mask = (point_counts == count)
            current_indices = count_mask.nonzero(as_tuple=True)[0]
            
            # Case 1: Less than 2 points, no collinearity possible
            if count < 2:
                can_add_mask[current_indices] = True
                continue

            # Case 2: Check for collinearity
            group_grids = valid_grids[current_indices]
            group_candidates = valid_points[current_indices]

            # Efficiently extract existing points for the entire group
            nz = (group_grids == 1).nonzero(as_tuple=False)
            existing_points = nz[:, 1:].reshape(group_grids.shape[0], count.item(), 2)
            
            # Vectorized collinearity check (adapted from check_new_points)
            pair_indices = torch.combinations(torch.arange(count.item()), r=2)
            p1s = existing_points[:, pair_indices[:, 0]]
            p2s = existing_points[:, pair_indices[:, 1]]
            p3s = group_candidates.unsqueeze(1) # a.k.a. the candidate points

            x1, y1 = p1s[..., 0], p1s[..., 1]
            x2, y2 = p2s[..., 0], p2s[..., 1]
            x3, y3 = p3s[..., 0], p3s[..., 1]

            collinearity_check = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            
            # A candidate is invalid if ANY pair is collinear
            invalid_candidates = (collinearity_check == 0).any(dim=1)
            can_add_mask[current_indices] = ~invalid_candidates

        # 3. Add all valid points to their grids in a single operation
        if can_add_mask.any():
            points_to_add = valid_points[can_add_mask]
            indices_to_update = batch_indices[can_add_mask]
            grids[indices_to_update, points_to_add[:, 0], points_to_add[:, 1]] = 1
            
        return grids

    def available_spaces(self, grid):
        """
        Return the coordinates of all empty (value==0) squares in a single grid.
        """
        return (grid == 0).nonzero(as_tuple=False).to(torch.int)

    def best_grid(self, grids, points):
        """
        Given `grids` (a stack of identical grids) and `points` (distinct 
        candidate points), this function returns the grid that is "best" after
        adding its point. The "best" grid is the one with the most available
        (0) squares remaining after the point and all newly-forbidden squares
        have been accounted for.
        If multiple grids are tied for the best score, one is chosen randomly.
        """
        # 1. Get the resulting grids after attempting to add each point
        updated_grids = self.add_points_to_grid(grids, points)

        # 2. For each of these new grids, update their forbidden squares
        updated_grids = self.update_forbidden_squares(updated_grids)
        
        # 3. Score each grid by the number of available spaces left
        scores = (updated_grids == 0).sum(dim=(1, 2))
        
        # 4. Find all indices of grids with the highest score
        max_score = torch.max(scores)
        best_indices = (scores == max_score).nonzero(as_tuple=True)[0]
        
        # 5. Randomly choose one from the best candidates
        random_choice_idx = torch.randint(0, best_indices.shape[0], (1,)).item()
        best_idx = best_indices[random_choice_idx]
        
        # 6. Return the chosen best grid
        return updated_grids[best_idx]

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

    N = 10
    solver = NoThreeInLine(batch_size=150, grid_size=N, max_points=2*N)
    t0 = time.time()
    solver.greedy_saturate()
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