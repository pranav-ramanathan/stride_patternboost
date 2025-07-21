import os
import logging
import heapq


class TopPool:
    """Maintain a fixed-capacity min-heap of the highest-scoring constructions."""

    def __init__(self, capacity: int, grid_size: int, logger: logging.Logger | None = None):
        self.capacity = capacity
        self.heap: list[tuple[int, str]] = []  # (score, token_string)
        self.perfect_score = 2 * grid_size
        self.logger = logger
        self.added_count = 0  # count of constructions added or replaced

    # internal ---------------------------------------------------------------
    def _push(self, score: int, token_string: str):
        heapq.heappush(self.heap, (score, token_string))

    def _pop(self):
        score, token_string = heapq.heappop(self.heap)
        return score, token_string

    # public -----------------------------------------------------------------
    def add(self, score: int, token_string: str):
        """
        Attempt to add construction; keep only if it improves pool.
        The heap has a fixed capacity.
        """
        # The special case for expanding the heap for perfect constructions has been removed
        # to prevent bugs related to invalid grids being classified as "perfect".
        # The heap now has a strictly fixed capacity.

        # Add new or replace worst if better; count additions without per-item logs
        if len(self.heap) < self.capacity:
            self._push(score, token_string)
            self.added_count += 1
        elif score > self.heap[0][0]:  # replace worst
            self._pop()
            self._push(score, token_string)
            self.added_count += 1

    def build_from_file(self, path: str):
        if not os.path.isfile(path):
            return
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Correctly calculate score by counting tokens, not "V"s
                score = len(line.split(','))

                # Defensively skip adding constructions that are too large,
                # in case of a corrupted input file.
                if score > self.perfect_score * 2: # Allow some buffer over perfect
                    continue

                self.add(score, line)

    def dump_to_file(self, path: str):
        # Summary of additions
        if self.logger:
            self.logger.info(f"Added {self.added_count} best constructions to pool (capacity={self.capacity}).")
        # Dump heap to file
        with open(path, "w") as f:
            for _score, token_str in self.heap:
                f.write(token_str + "\n")

    def __len__(self):
        return len(self.heap)

