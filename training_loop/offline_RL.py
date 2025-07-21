import os, time, argparse, logging, random, re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

from models import (
    ModelConfig,
    InfiniteDataLoader,
)
from utils import TopPool
from models import DecisionTransformer

from .training_tools import (  # <- small helper module you can create to hold shared funcs
    create_datasets_from_file,
    decode_and_update_pool,
    generate_samples_dt,
)

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser("Decision‑Transformer SFT for No‑Three‑In‑Line")

    # problem setup ----------------------------------------------------------
    parser.add_argument("--grid_size", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=18)

    # data / pool ------------------------------------------------------------
    parser.add_argument("--target_training_size", type=int, default=20000,
                        help="Desired size of the pooled training set (capacity of heap)")

    # generation / selection -------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--keep_best_fraction", type=float, default=0.1)
    parser.add_argument("--symmetrize", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate all 8 symmetries of selected constructions (default on)")

    # model & optimisation ---------------------------------------------------
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_embd2", type=int, default=16, help="ignored for DT but kept for compatibility")

    parser.add_argument("--nn_batch_size", "-b", type=int, default=64)
    parser.add_argument("--learning_rate", "-l", type=float, default=5e-4)
    parser.add_argument("--weight_decay", "-w", type=float, default=0.01)
    parser.add_argument("--num_workers", "-n", type=int, default=8)

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--use_reward_weighting", action=argparse.BooleanOptionalAction, default=True,
                        help="Use reward-weighted loss for Decision Transformer training")

    # sampling ---------------------------------------------------------------
    parser.add_argument("--gen_batch_size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)

    # system / paths ---------------------------------------------------------
    parser.add_argument("--dump_path", type=str, default="dump_path")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=-1)

    return parser

# -----------------------------------------------------------------------------
# Logging / Device helpers (identical to original SFT) -------------------------
# -----------------------------------------------------------------------------

def setup_logging(dump_path):
    os.makedirs(dump_path, exist_ok=True)
    training_dir = os.path.join(dump_path, "training_sets")
    os.makedirs(training_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(os.path.join(dump_path, "training.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
    return logger

def set_device(args, logger):
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    logger.info(f"Using device: {args.device}")

# -----------------------------------------------------------------------------
# Utility – compute RTG tensor on the fly -------------------------------------
# -----------------------------------------------------------------------------

def make_rtg_tensor(batch_size: int, seq_len: int, max_points: int, device: str):
    """Return-to-go = how many more valid points we want to place.
    
    For no-three-in-line, RTG should represent the target number of points
    we want to achieve, decreasing as we place valid points.
    
    This creates RTG values that start at max_points and decrease linearly.
    """
    # Create decreasing RTG values: [max_points, max_points-1, ..., 1, 0, 0, ...]
    rtg_values = torch.arange(seq_len, device=device)
    rtg = (max_points - rtg_values).clamp(min=0).float()
    rtg = rtg.unsqueeze(0).repeat(batch_size, 1)
    
    return rtg

def make_rewards_from_rtg(rtg: torch.FloatTensor, max_points: int) -> torch.FloatTensor:
    """Convert RTG values to rewards for loss weighting.
    
    For the no-three-in-line problem, we want to weight learning from 
    trajectories that achieve higher scores. We can weight based on the
    remaining target points - higher RTG means we're aiming for more points.
    
    Args:
        rtg: Return-to-go tensor of shape (B, T)
        max_points: Maximum possible points in the problem
        
    Returns:
        rewards: Reward weights of shape (B, T) in [0, 1] range
    """
    # Normalize RTG to [0, 1] range
    normalized_rtg = rtg / max_points
    
    # Weight by RTG: higher RTG (aiming for more points) gets higher weight
    # Use square root to not over-emphasize the differences
    rewards = torch.pow(normalized_rtg, 0.5)
    
    # Alternative: you could also weight by trajectory quality
    # This would require knowing the final score of each trajectory
    
    return rewards

def evaluate_dt(model, dataset, device, batch_size=50, max_batches=None, 
                make_rtg=None, use_reward_weighting=False, max_points=30, num_workers=0):
    """Custom evaluation function for Decision Transformer that handles RTG values."""
    from torch.utils.data import DataLoader
    
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    losses = []
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
                
            batch = [t.to(device) for t in batch]
            X, Y = batch
            
            # Create RTG tensor
            if make_rtg is not None:
                rtg = make_rtg(X.size(0), X.size(1), max_points, device)
            else:
                rtg = make_rtg_tensor(X.size(0), X.size(1), max_points, device)
            
            # Create rewards for evaluation (use same loss function as training)
            if use_reward_weighting:
                rewards = make_rewards_from_rtg(rtg, max_points)
            else:
                rewards = None
            
            # Forward pass
            logits, _, loss = model(X, rtg, targets=Y, rewards=rewards)
            losses.append(loss.item())
    
    mean_loss = torch.tensor(losses).mean().item()
    model.train()  # reset model back to training mode
    return mean_loss

# -----------------------------------------------------------------------------
# (The dataset loading, symmetrization, pool logic, etc. remain identical to
# the original script – paste or import them here to keep file compact.)
# For brevity, we assume `create_datasets_from_file`, `generate_samples`,
# `decode_and_update_pool`, etc. are imported from the original SFT module or
# duplicated verbatim below. ---------------------------------------------------
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Main ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    start = time.time()
    args = get_parser().parse_args()

    # --- housekeeping ------------------------------------------------------
    args.dump_path = os.path.join("training", args.dump_path)
    logger = setup_logging(args.dump_path)
    set_device(args, logger)

    # --- seeding -----------------------------------------------------------
    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"seed: {args.seed}")

    # --- token maps --------------------------------------------------------
    N = args.grid_size
    logger.info("Creating token maps...")
    token_encoding = torch.arange(N * N).view(N, N)
    coords = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(N), indexing="xy"), -1)
    token_decoding = coords.view(-1, 2)

    # --- determine initial generation -------------------------------------
    ts_dir = os.path.join(args.dump_path, "training_sets")
    gen_re = re.compile(rf"N{N}_gen(\d+)\.txt$")
    gens = [int(m.group(1)) for fn in os.listdir(ts_dir) if (m := gen_re.match(fn))]
    if not gens:
        logger.error("No generation files found. Run generate_initial.py first.")
        exit(1)
    initial_gen = max(gens)
    logger.info(f"Resuming from generation {initial_gen}")

    # --- build initial TopPool -------------------------------------------
    pool = TopPool(args.target_training_size, args.grid_size)
    init_file = os.path.join(ts_dir, f"N{N}_gen{initial_gen}.txt")
    pool.build_from_file(init_file)
    logger.info(f"Initial pool size: {len(pool)}")

    best_loss = None
    end_gen = initial_gen + args.max_epochs

    for gen in range(initial_gen, end_gen):
        logger.info(f"=========== Start of generation {gen + 1} ===========")

        # --- dataset ------------------------------------------------------
        input_gen_file = os.path.join(ts_dir, f"N{N}_gen{gen}.txt")
        train_ds, test_ds = create_datasets_from_file(input_gen_file, args, token_decoding, token_encoding, logger)

        cfg = ModelConfig(
            vocab_size=train_ds.get_vocab_size(),
            block_size=train_ds.get_output_length(),
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
        )

        model = DecisionTransformer(vocab_size=cfg.vocab_size, config=cfg).to(args.device)
        
        # Load from SFT model.pt FIRST, then check for existing DT model
# Load from existing DT model FIRST, then fall back to SFT model
        sft_model_path = os.path.join(args.dump_path, "model.pt")
        dt_model_path = os.path.join(args.dump_path, "model_dt.pt")

        if os.path.isfile(dt_model_path):
            # Resume from existing DT model (prioritize this)
            model.load_state_dict(torch.load(dt_model_path, map_location=args.device))
            logger.info("Resumed existing Decision Transformer model.")
            
        elif os.path.isfile(sft_model_path):
            # Load SFT model weights and convert to DT format (only if no DT model exists)
            sft_state_dict = torch.load(sft_model_path, map_location=args.device)
            
            # Extract transformer weights from SFT model
            dt_state_dict = {}
            for key, value in sft_state_dict.items():
                if key.startswith('transformer.'):
                    dt_state_dict[key] = value
                elif key.startswith('lm_head.'):
                    dt_state_dict[key] = value
            
            # Load compatible weights into DT model
            model.load_state_dict(dt_state_dict, strict=False)
            logger.info("Loaded SFT model weights into Decision Transformer.")
            
        else:
            logger.info("No existing model found. Training Decision Transformer from scratch.")

        # Set initial best_loss only once at the very beginning
        if best_loss is None:
            te_loss = evaluate_dt(model, test_ds, args.device, batch_size=50, max_batches=20,
                            make_rtg=make_rtg_tensor, use_reward_weighting=args.use_reward_weighting, 
                            max_points=args.max_points, num_workers=args.num_workers)
            best_loss = te_loss
            logger.info(f"Initial best_loss set to {te_loss:.4f}")
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # Disable pin_memory for MPS to avoid warnings
        use_pin_memory = args.device not in ["mps"]
        loader = InfiniteDataLoader(train_ds, batch_size=args.nn_batch_size, pin_memory=use_pin_memory, num_workers=args.num_workers)

        # Log training configuration
        logger.info(f"Training with reward weighting: {args.use_reward_weighting}")
        if args.use_reward_weighting:
            logger.info("Using reward-weighted loss to focus on high-reward trajectories")
        else:
            logger.info("Using standard cross-entropy loss")

        # --- training -----------------------------------------------------
        for step in range(args.max_steps + 1):
            X, Y = [t.to(args.device) for t in next(loader)]
            rtg = make_rtg_tensor(X.size(0), X.size(1), args.max_points, args.device)
            
            # Use reward weighting if enabled
            if args.use_reward_weighting:
                rewards = make_rewards_from_rtg(rtg, args.max_points)
            else:
                rewards = None
                
            logits, _, loss = model(X, rtg, targets=Y, rewards=rewards)
            model.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if step % 100 == 0:
                logger.info(f"step {step} | loss {loss.item():.4f}")
            if step and step % 500 == 0:
                tr_loss = evaluate_dt(model, train_ds, args.device, batch_size=50, max_batches=10,
                                   make_rtg=make_rtg_tensor, use_reward_weighting=args.use_reward_weighting, 
                                   max_points=args.max_points, num_workers=args.num_workers)
                te_loss = evaluate_dt(model, test_ds, args.device, batch_size=50, max_batches=10,
                                   make_rtg=make_rtg_tensor, use_reward_weighting=args.use_reward_weighting, 
                                   max_points=args.max_points, num_workers=args.num_workers)
                logger.info(f"step {step} train {tr_loss:.4f} test {te_loss:.4f}")
                if best_loss is None or te_loss < best_loss:
                    logger.info("best test loss improved; saving model to model_dt.pt")
                    torch.save(model.state_dict(), dt_model_path)
                    best_loss = te_loss
        loader.shutdown()

        # --- sampling -----------------------------------------------------
        total_to_generate = int(args.target_training_size * (1 / args.keep_best_fraction))
        all_samples = []
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn()) as prog:
            t = prog.add_task("Generating samples", total=total_to_generate)
            done = 0
            while done < total_to_generate:
                n = min(args.gen_batch_size, total_to_generate - done)
                model.eval()
                new_samples = generate_samples_dt(model, train_ds, args, n, args.max_points)
                model.train()
                all_samples.extend(new_samples)
                done += n
                prog.update(t, advance=n)

        # --- decode & update pool ----------------------------------------
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.synchronize()  # MPS doesn't have empty_cache()
        decode_and_update_pool(args, token_decoding, token_encoding, gen + 1, logger, pool, all_samples)
        logger.info(f"=========== End of generation {gen + 1} ===========")

    logger.info("All generations completed!")
    logger.info(f"Total time: {time.time() - start:.2f} seconds")