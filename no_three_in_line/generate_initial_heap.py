import os, time, argparse
import pprint, logging
from collections import Counter
import torch
import numpy as np

from rich.logging import RichHandler

from no_three_in_line import NoThreeInLine


def get_parser():
    parser = argparse.ArgumentParser('Generate initial training data (no symmetries) for no-three-in-line')

    parser.add_argument('--grid_size', type=int, default=6, help='Grid size for 2D no-three-in-line (creates NxN grid)')
    parser.add_argument('--batch_size', type=int, default=500, help='Generate and process samples in batches of this size')
    parser.add_argument('--max_points', type=int, default=18, help='Max points which can be added to a construction')
    parser.add_argument('--target_training_size', type=int, default=20000, help='Number of examples to aim for')
    parser.add_argument('--keep_best_fraction', type=float, default=1.0, help='Percentage of good constructions to keep from each batch')

    parser.add_argument('--seed', type=int, default=-1, help='Random seed; -1 for random')
    parser.add_argument('--dump_path', type=str, default='dump_path', help='Experiment dump directory (inside training/)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use: auto|cpu|cuda|mps')
    parser.add_argument('--job_id', type=str, default='', help='Optional job ID for parallel runs (creates separate files)')

    return parser


if __name__ == '__main__':
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()

    # Always put output in <project-root>/training/<dump_path>
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.dump_path = os.path.join(project_root, "training", os.path.basename(args.dump_path))

    # ---------------------------------------------------------------------
    # Logging setup
    # ---------------------------------------------------------------------
    log_prefix = args.dump_path + "/"
    os.makedirs(log_prefix, exist_ok=True)
    training_dir = os.path.join(log_prefix, 'training_sets')
    os.makedirs(training_dir, exist_ok=True)

    log_suffix = f"_job{args.job_id}" if args.job_id else ""

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear existing handlers

    fh = logging.FileHandler(os.path.join(log_prefix, f"initial-generation{log_suffix}.log"))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.addHandler(RichHandler(rich_tracebacks=True, show_path=False))

    # ---------------------------------------------------------------------
    # Device & seed
    # ---------------------------------------------------------------------
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
    logger.info(f"Using device: {args.device}")

    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"seed: {args.seed}")

    with open(os.path.join(log_prefix, 'args.txt'), 'w') as f:
        f.write(pprint.pformat(vars(args)))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------------------------
    # Coordinate mapping tensors
    # ---------------------------------------------------------------------
    N = args.grid_size
    logger.info(f"Creating token maps for {N}x{N} grid...")
    token_encoding = torch.arange(N * N).view(N, N)  # (x, y) -> token_id
    logger.info("Token maps created successfully.")

    # ---------------------------------------------------------------------
    # Determine output file paths
    # ---------------------------------------------------------------------
    initial_gen = 0
    base_filename = f"N{N}_gen{initial_gen}"
    if args.job_id:
        base_filename += f"_job{args.job_id}"
    output_file = os.path.join(training_dir, base_filename + '.txt')

    if os.path.isfile(output_file):
        logger.info(f"Initial generation already exists at {output_file}")
        logger.info("Delete the file if you want to regenerate it.")
        exit(0)

    # ---------------------------------------------------------------------
    # Generation loop
    # ---------------------------------------------------------------------
    logger.info("Generating 0th generation of training data (no symmetries)...")

    best_constructions_per_batch = int(args.batch_size * args.keep_best_fraction)
    total_batches = int(np.ceil(args.target_training_size / best_constructions_per_batch))

    constructions_log = []
    t0 = time.time()

    for batch_idx in range(total_batches):
        logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")

        nti = NoThreeInLine(
            batch_size=args.batch_size,
            grid_size=args.grid_size,
            max_points=args.max_points,
            device=args.device,
        )
        nti.greedy_saturate_batched()

        # Sort by number of points and keep the best fraction
        x = torch.argsort(nti.current_counts, descending=True)
        top_constructions = ((nti.current_constructions[x[:best_constructions_per_batch]] == 1) * 1).cpu()

        constructions_log += nti.current_counts.int().tolist()

        # Convert constructions to token sequences and save
        with open(output_file, 'a') as f:
            for construction in top_constructions:
                indices = torch.nonzero(construction)
                if indices.numel() == 0:
                    continue
                encoding = token_encoding[indices[:, 0], indices[:, 1]]
                sorted_encoding = torch.sort(encoding)[0]
                encoding_string = ','.join([f'V{tok}' for tok in sorted_encoding])
                f.write(encoding_string + '\n')

    # ---------------------------------------------------------------------
    # Logging summary
    # ---------------------------------------------------------------------
    generation_time = time.time() - t0
    logger.info(f"Generated {len(constructions_log)} constructions (raw).")
    logger.info(f"Distribution of counts = {Counter(constructions_log)}")

    with open(output_file, 'r') as f:
        num_lines = sum(1 for _ in f)
    logger.info(f"Saved {num_lines} training examples (no symmetries) to {output_file}")
    logger.info(f"Generation took {generation_time:.2f} seconds.")

    end_time = time.time()
    logger.info(f"Total runtime: {end_time - start_time:.2f} seconds")

    if args.job_id:
        logger.info(f"Parallel job {args.job_id} completed! You can now concatenate job files if you ran multiple jobs.")
    else:
        logger.info("Ready for training! Run gw_train_heap.py with the same --dump_path and --grid_size") 