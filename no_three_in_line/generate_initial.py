import os, time, argparse
import pprint, logging
from collections import Counter
import torch
import numpy as np

from rich.logging import RichHandler

from no_three_in_line import NoThreeInLine

def get_parser():
    parser = argparse.ArgumentParser('Generate initial training data for no-three-in-line')

    parser.add_argument('--grid_size', type=int, default=6, help='Grid size for 2D no-three-in-line (creates NxN grid)')
    parser.add_argument('--batch_size', type=int, default=500, help='Generate and process samples in batches of this size')
    parser.add_argument('--max_points', type=int, default=18, help='max points which can be added to construction')
    parser.add_argument('--target_training_size', type=int, default=20000, help='number of examples to aim for (before symmetrization)')
    parser.add_argument('--keep_best_fraction', type=float, default=0.1, help='Percentage of good constructions to keep')
    parser.add_argument('--symmetrize', default=True, action=argparse.BooleanOptionalAction, help='symmetrize constructions, set to --no-symmetrize to disable')

    parser.add_argument('--seed', type=int, default=-1, help="seed")
    parser.add_argument("--dump_path", type=str, default="dump_path", help="Experiment dump path")
    parser.add_argument("--device", type=str, default="auto", help="device to use for compute: auto|cpu|cuda|mps")
    parser.add_argument("--job_id", type=str, default="", help="Job ID for parallel execution (optional, creates separate output files)")

    return parser

def generate_2d_symmetries(construction):
    """Generate unique symmetries of 2D constructions
    
    Generates up to 8 symmetries of a 2D construction, but only yields unique ones.
    For highly symmetric constructions, this may yield fewer than 8 results.
    
    Args:
        construction: tensor representing 2D point placement
    Yields:
        unique transformed versions of the construction
    """
    # Generate all 8 possible symmetries
    transformations = []
    
    # 4 rotations of the original
    for k in range(4):
        transformations.append(torch.rot90(construction, k, [0, 1]))
    
    # 4 rotations of a single flip
    flipped = torch.flip(construction, [0])
    for k in range(4):
        transformations.append(torch.rot90(flipped, k, [0, 1]))
    
    # Only yield unique transformations
    for transformed in transformations:
        yield transformed

if __name__ == '__main__':
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    
    # Always put output in training/<dump_path>
    args.dump_path = f"training/{os.path.basename(args.dump_path)}"

    # Setup logging and directories
    log_prefix = args.dump_path + "/"
    if not os.path.exists(log_prefix):
        os.makedirs(log_prefix)
    training_dir = log_prefix + 'training_sets'
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    # Determine log suffix for parallel jobs
    log_suffix = f"_job{args.job_id}" if args.job_id else ""

    # Configure logging with both console and file output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Configure file handler
    fh = logging.FileHandler(log_prefix + f"initial-generation{log_suffix}.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Configure console handler using rich
    logger.addHandler(RichHandler(rich_tracebacks=True, show_path=False))

    # Set device with proper MPS support
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    
    logger.info(f"Using device: {args.device}")
        
    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"seed: {args.seed}")

    # print args to file
    with open(log_prefix + 'args.txt', 'w') as f:
        logger.info(pprint.pformat(args.__dict__))

    # system inits
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # ========================================================================
    # Create coordinate mapping system for 2D problem
    # ======================================================================== 
    N = args.grid_size
    logger.info(f"Creating token maps for {N}x{N} grid...")
    # token_encoding: (x, y) -> token_id
    token_encoding = torch.arange(N*N).view(N, N)
    # token_decoding: token_id -> (x, y)
    coords = torch.stack(
        torch.meshgrid(torch.arange(N), torch.arange(N), indexing='xy'), 
        -1
    )
    token_decoding = coords.view(-1, 2) # Reshape from (N, N, 2) to (N*N, 2)
    logger.info("Token maps created successfully.")
    
    # Check if initial generation already exists
    initial_gen = 0
    training_path_root = args.dump_path + f'/training_sets/N{N}_gen{initial_gen}'
    
    # Add job_id to filename if provided (for parallel execution)
    if args.job_id:
        training_path = training_path_root + f'_job{args.job_id}.txt'
        training_path_unpermuted = training_path_root + f'_job{args.job_id}_unpermuted.txt'
    else:
        training_path = training_path_root + '.txt'
        training_path_unpermuted = training_path_root + '_unpermuted.txt'
    
    if os.path.isfile(training_path):
        logger.info(f"Initial generation already exists at {training_path}")
        logger.info("Delete the file if you want to regenerate it.")
        exit(0)
    
    # Generate initial training data
    logger.info("Generating 0th generation of training data...")

    constructions_log = []
    best_constructions_per_batch = int(args.batch_size * args.keep_best_fraction)
    
    t0 = time.time()

    for batch_idx in range(int(args.target_training_size / best_constructions_per_batch)):
        logger.info(f"Processing batch {batch_idx + 1}/{int(args.target_training_size / best_constructions_per_batch)}")
        
        # Create initial random data
        nothreeinline = NoThreeInLine(
            batch_size=args.batch_size,
            grid_size=args.grid_size,
            max_points=args.max_points,
            device=args.device
        )
        nothreeinline.greedy_saturate_batched()

        # sort according to number of points
        x = torch.argsort(nothreeinline.current_counts, descending=True)
        top_constructions = ((nothreeinline.current_constructions[x[0:best_constructions_per_batch]] == 1) * 1).cpu()

        constructions_log += nothreeinline.current_counts.int().tolist()

        # Convert constructions to token sequences and save
        out_string = ''
        out_string_unpermuted = ''
        if args.symmetrize:
            for construction in top_constructions:
                # Save unpermuted version first
                indices = torch.nonzero(construction)
                if indices.numel() > 0:
                    encoding = token_encoding[indices[:, 0], indices[:, 1]]
                    sorted_encoding = torch.sort(encoding)[0]
                    encoding_string = ','.join([f'V{tok}' for tok in sorted_encoding]) + '\n'
                    out_string_unpermuted += encoding_string

                for symmetric_construction in generate_2d_symmetries(construction):
                    indices = torch.nonzero(symmetric_construction)
                    if indices.numel() > 0:
                        encoding = token_encoding[indices[:, 0], indices[:, 1]]
                        sorted_encoding = torch.sort(encoding)[0]
                        encoding_string = ','.join([f'V{tok}' for tok in sorted_encoding]) + '\n'
                        out_string += encoding_string
        else:
            for construction in top_constructions:
                indices = torch.nonzero(construction)
                if indices.numel() > 0:
                    encoding = token_encoding[indices[:, 0], indices[:, 1]]
                    sorted_encoding = torch.sort(encoding)[0]
                    encoding_string = ','.join([f'V{tok}' for tok in sorted_encoding]) + '\n'
                    out_string += encoding_string

        with open(training_path, 'a') as f:
            f.write(out_string)

        if args.symmetrize:
            with open(training_path_unpermuted, 'a') as f:
                f.write(out_string_unpermuted)

    if args.device == "cuda":
        logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
    elif args.device == "mps":
        logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB")
    
    generation_time = time.time() - t0
    logger.info(f"Generated {len(constructions_log)} constructions.")
    logger.info(f"Generation took {generation_time:.2f} seconds.")
    logger.info(f"Distribution of counts = {Counter(constructions_log)}")
    logger.info(f"Training data saved to {training_path}")
    
    # Count lines in the output file
    with open(training_path, 'r') as f:
        num_lines = sum(1 for _ in f)
    logger.info(f"Generated {num_lines} training examples (after symmetrization)")
    
    end_time = time.time()
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    
    if args.job_id:
        logger.info(f"Parallel job {args.job_id} completed!")
        logger.info(f"To combine with other parallel jobs, concatenate all job files:")
        logger.info(f"cat {training_path_root}_job*.txt > {training_path_root}.txt")
        if args.symmetrize:
            logger.info(f"cat {training_path_root}_job*_unpermuted.txt > {training_path_root}_unpermuted.txt")
        logger.info(f"Then run gw_train.py with the same --dump_path and --grid_size")
    else:
        logger.info(f"Ready for training! Run gw_train.py with the same --dump_path and --grid_size") 