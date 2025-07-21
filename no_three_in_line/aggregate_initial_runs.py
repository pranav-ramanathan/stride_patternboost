import os
import argparse
import logging
from collections import Counter

from rich.logging import RichHandler
from utils.top_pool import TopPool

def get_parser():
    parser = argparse.ArgumentParser(description="Aggregate multiple initial generation runs into a single top pool.")
    parser.add_argument("--grid_size", "-N", type=int, required=True, help="The grid size (e.g., 11).")
    parser.add_argument("--pool_capacity", "-c", type=int, required=True, help="The target capacity of the final training pool.")
    parser.add_argument("--dump_path", type=str, required=True, help="Base dump path for the experiment (e.g., 'training/N11_2').")
    parser.add_argument("--num_jobs", type=int, required=True, help="Number of initial generation jobs to aggregate.")
    parser.add_argument("--output_file", "-o", type=str, default=None, help="Optional: Path for the final aggregated file. If not provided, it's generated automatically based on dump_path and grid_size.")
    return parser

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
    return logger

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = setup_logging()

    logger.info("Initializing Top Pool...")
    pool = TopPool(capacity=args.pool_capacity, grid_size=args.grid_size, logger=logger)
    
    base_path = os.path.join(args.dump_path, "training_sets")
    logger.info(f"Looking for job outputs in: {base_path}")

    input_files = []
    for i in range(1, args.num_jobs + 1):
        file_path = os.path.join(base_path, f"N{args.grid_size}_gen0_job{i}.txt")
        if os.path.isfile(file_path):
            input_files.append(file_path)
        else:
            logger.warning(f"File not found, skipping: {file_path}")

    if not input_files:
        logger.error("No input files found for the specified jobs. Exiting.")
        exit(1)

    logger.info(f"Found {len(input_files)} files to aggregate.")

    for file_path in input_files:
        logger.info(f"Processing file: {file_path}")
        pool.build_from_file(file_path)

    logger.info(f"Aggregation complete. Final pool size: {len(pool)}")
    if pool:
        logger.info(f"Worst score in pool: {pool.heap[0][0]}")

        # Calculate and log the score distribution
        score_counts = Counter(score for score, _ in pool.heap)
        sorted_counts = sorted(score_counts.items())
        logger.info(f"Final score distribution: {Counter(dict(sorted_counts))}")

    output_file = args.output_file
    if not output_file:
        output_dir = os.path.join(args.dump_path, "training_sets")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"N{args.grid_size}_gen0.txt")

    logger.info(f"Dumping aggregated pool to: {output_file}")
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    pool.dump_to_file(output_file)

    logger.info("Done.") 