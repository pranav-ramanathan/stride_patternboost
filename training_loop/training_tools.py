import os, time, argparse, logging, random
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

from rich.logging import RichHandler

from models import (
    CharDataset,
    generate,
)

from no_three_in_line import NoThreeInLine
from utils import TopPool

# --------------------------- Utility functions ------------------------------
def setup_logging(dump_path):
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
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

def construction_to_string(construction_tensor: torch.Tensor, token_encoding: torch.Tensor) -> str:
    """Converts a construction tensor to its canonical token string."""
    indices = torch.nonzero(construction_tensor == 1)
    if indices.numel() == 0:
        return ""
    encoding = token_encoding[indices[:, 0], indices[:, 1]]
    sorted_encoding = torch.sort(encoding)[0]
    return ','.join([f'V{tok}' for tok in sorted_encoding.tolist()])

def generate_symmetries(construction):
    for k in range(4):
        yield torch.rot90(construction, k, [0, 1])
    flipped = torch.flip(construction, [0])
    for k in range(4):
        yield torch.rot90(flipped, k, [0, 1])

def get_canonical_form(construction_tensor: torch.Tensor, token_encoding: torch.Tensor) -> str | None:
    """
    Generates all 8 symmetries of a construction and returns the token string
    of the lexicographically smallest symmetry.
    """
    canonical_string = None
    for sym_con in generate_symmetries(construction_tensor):
        token_str = construction_to_string(sym_con, token_encoding)
        if token_str:
            if canonical_string is None or token_str < canonical_string:
                canonical_string = token_str
    return canonical_string

def create_datasets_from_file(input_file: str, args: argparse.Namespace, 
                                          token_decoding: torch.Tensor, token_encoding: torch.Tensor, 
                                          logger: logging.Logger):
    """
    Loads constructions from a file and creates train/test datasets.
    If args.symmetrize is True, it generates all symmetries on the fly.
    """
    # Load top constructions up to target_training_size using TopPool (pick best by score)
    pool = TopPool(capacity=args.target_training_size, grid_size=args.grid_size, logger=logger)
    pool.build_from_file(input_file)
    lines = [token_str for _, token_str in pool.heap]
    logger.info(f"Selected {len(lines)} top constructions using TopPool (capacity={args.target_training_size}).")

    if args.symmetrize:
        words = []
        for line in lines:
            tokens = [int(t[1:]) for t in line.split(',')]
            construction_tensor = torch.zeros((args.grid_size, args.grid_size), dtype=torch.int8)
            for token_num in tokens:
                if token_num < len(token_decoding):
                    coords = token_decoding[token_num]
                    construction_tensor[coords[0], coords[1]] = 1
            
            for sym_con in generate_symmetries(construction_tensor):
                indices = torch.nonzero(sym_con)
                if indices.numel() > 0:
                    encoding = token_encoding[indices[:, 0], indices[:, 1]]
                    sorted_encoding_list = sorted(encoding.tolist())
                    words.append([f'V{tok}' for tok in sorted_encoding_list])
        logger.info(f"Loaded {len(lines)} constructions and generated {len(words)} symmetrized examples.")
    else:
        words = [line.split(',') for line in lines]
        logger.info(f"Loaded {len(words)} constructions (no symmetrization).")


    chars = [f"V{i}" for i in range(args.grid_size ** 2)]

    random.shuffle(words)
    split_idx = int(0.9 * len(words))
    train_words = words[:split_idx]
    test_words = words[split_idx:]
    
    train_ds = CharDataset(train_words, chars, args.max_points)
    test_ds = CharDataset(test_words, chars, args.max_points)
    
    logger.info(f"Created dataset ({len(train_words)} train, {len(test_words)} test).")
    
    return train_ds, test_ds

def generate_samples(model, train_dataset, args, num_samples):
    """Generates samples from the model and returns them as a list of token lists."""
    top_k = args.top_k if args.top_k != -1 else None

    X_init = torch.zeros(num_samples, 1, dtype=torch.long).to(args.device)
    steps = train_dataset.get_output_length() - 1
    X_samp = generate(model, X_init, steps, temperature=args.temperature, top_k=top_k, do_sample=True).cpu()

    all_tokens = []
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()
        crop = row.index(0) if 0 in row else len(row)
        # Decode to token strings like "V12", then strip "V" and convert to int
        decoded_str = train_dataset.decode(row[:crop])
        tokens = [int(t[1:]) for t in decoded_str.split(',') if t]
        if tokens:
            all_tokens.append(tokens)
    return all_tokens

def generate_samples_dt(model, train_dataset, args, num_samples, max_points):
    """Generates samples from a Decision Transformer model and returns them as a list of token lists."""
    # For Decision Transformer, we need to provide RTG (return-to-go) values
    # Start with maximum RTG (max_points) and generate tokens
    X_init = torch.zeros(num_samples, 1, dtype=torch.long).to(args.device)
    rtg_init = torch.full((num_samples, 1), max_points, dtype=torch.float).to(args.device)
    
    steps = train_dataset.get_output_length() - 1
    
    # Use the Decision Transformer's generate method with proper RTG updates
    with torch.no_grad():
        model.eval()
        tokens = X_init
        rtg = rtg_init
        
        for _ in range(steps):
            # Crop context if necessary
            tokens_cond = tokens[:, -model.block_size:] if tokens.size(1) > model.block_size else tokens
            rtg_cond = rtg[:, -model.block_size:] if rtg.size(1) > model.block_size else rtg
            
            # Get next token
            logits, _, _ = model(tokens_cond, rtg_cond)
            next_token_logits = logits[:, -1, :]
            
            # Sample or take argmax
            if hasattr(args, 'temperature') and args.temperature > 0:
                next_token_logits = next_token_logits / args.temperature
                if hasattr(args, 'top_k') and args.top_k > 0:
                    v, _ = torch.topk(next_token_logits, args.top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Update RTG: decrease by 1 for valid tokens, stay same for padding (token 0)
            next_rtg = torch.where(next_token == 0, rtg[:, -1:], torch.clamp(rtg[:, -1:] - 1, min=0))
            rtg = torch.cat([rtg, next_rtg], dim=1)
            
            # Stop if all sequences hit the end token
            if (next_token == 0).all():
                break
    
    X_samp = tokens.cpu()
    
    all_tokens = []
    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist()  # Skip the initial token
        crop = row.index(0) if 0 in row else len(row)
        # Decode to token strings like "V12", then strip "V" and convert to int
        decoded_str = train_dataset.decode(row[:crop])
        tokens = [int(t[1:]) for t in decoded_str.split(',') if t]
        if tokens:
            all_tokens.append(tokens)
    return all_tokens

def format_grid(construction_tensor: torch.Tensor) -> str:
    """Formats a 2D tensor construction into a printable grid string."""
    grid_str = "\n"
    for r in range(construction_tensor.shape[0]):
        row_str = " ".join(["X" if construction_tensor[r, c] == 1 else "." for c in range(construction_tensor.shape[1])])
        grid_str += row_str + "\n"
    return grid_str

# --------------------------- Core: decode & update pool ----------------------
def decode_and_update_pool(args, token_decoding, token_encoding, generation, logger, pool: TopPool, sampled_tokens: list):
    N = args.grid_size

    # Force a sync before we start processing, to make sure any previous GPU work is complete
    if args.device == "mps":
        torch.mps.synchronize()

    decode_start_time = time.time()
    logger.info(f"{len(sampled_tokens)} samples received to process.")

    hist_pre = torch.zeros(args.max_points + 1, dtype=torch.int64)
    hist_post = torch.zeros(args.max_points + 1, dtype=torch.int64)
    total_pre, total_post = 0, 0

    best_pre_sat_score = -1
    best_pre_sat_construction = None

    for b in range(0, len(sampled_tokens), args.batch_size):
        cur_bs = min(args.batch_size, len(sampled_tokens) - b)
        # Force NoThreeInLine to run on CPU
        nti_device = "cpu"
        nti = NoThreeInLine(cur_bs, N, args.max_points, device=nti_device)
        batch = sampled_tokens[b : b + cur_bs]

        max_len = max(len(seq) for seq in batch) if batch else 0
        for i in range(max_len):
            pts = -1 * torch.ones((cur_bs, 2), dtype=torch.int8, device=nti_device)
            for j in range(cur_bs):
                if i < len(batch[j]):
                    tok_num = batch[j][i]
                    if tok_num < len(token_decoding):
                        pts[j] = token_decoding[tok_num]
            nti.try_to_add_points(pts)

        # Check for a new best pre-saturation score in the current batch
        if nti.current_counts.numel() > 0:
            current_max_score, current_max_idx = torch.max(nti.current_counts, dim=0)
            if current_max_score > best_pre_sat_score:
                best_pre_sat_score = current_max_score
                best_pre_sat_construction = nti.current_constructions[current_max_idx].cpu()

        total_pre += torch.sum(nti.current_counts).item()
        hist_pre.add_(torch.bincount(nti.current_counts.cpu(), minlength=args.max_points + 1))

        nti.saturate()

        total_post += torch.sum(nti.current_counts).item()
        hist_post.add_(torch.bincount(nti.current_counts.cpu(), minlength=args.max_points + 1))

        # select best fraction ------------------------------------------------
        x = torch.argsort(nti.current_counts, descending=True)
        best_k = int(cur_bs * args.keep_best_fraction)
        counts_sorted = nti.current_counts[x]
        constructions_sorted = nti.current_constructions[x]

        for idx in range(best_k):
            score = int(counts_sorted[idx])
            
            if score > args.max_points:
                continue

            construction = constructions_sorted[idx] # Already on CPU

            if args.symmetrize:
                canonical_form = get_canonical_form(construction, token_encoding)
                if canonical_form:
                    pool.add(score, canonical_form)
            else:
                token_str = construction_to_string(construction, token_encoding)
                if token_str:
                    pool.add(score, token_str)

    logger.info(f"Main processing loop took {time.time() - decode_start_time:.2f} seconds.")

    if best_pre_sat_construction is not None:
        grid_viz = format_grid(best_pre_sat_construction.cpu())
        logger.info(f"Best pre-saturation grid from transformer (score: {best_pre_sat_score}):{grid_viz}")

    # write full pool to new training file -----------------------------------
    training_path = os.path.join(args.dump_path, "training_sets", f"N{N}_gen{generation}.txt")
    pool.dump_to_file(training_path)

    logger.info(f"Generation {generation-1} -> {generation}")
    logger.info(f"Pool size: {len(pool)}  |  Worst score in pool: {pool.heap[0][0] if pool.heap else 'N/A'}")
    logger.info(f"NN points: {total_pre}, Saturation added: {total_post - total_pre}")

    post_processing_start_time = time.time()
    all_counts_pre_ctr = Counter({i: int(v) for i, v in enumerate(hist_pre.tolist()) if v != 0})
    all_counts_post_ctr = Counter({i: int(v) for i, v in enumerate(hist_post.tolist()) if v != 0})
    logger.info(f"Score distribution before saturation: {all_counts_pre_ctr}")
    logger.info(f"Score distribution after  saturation: {all_counts_post_ctr}")
    logger.info(f"Counter creation took {time.time() - post_processing_start_time:.2f} seconds.")

    # histogram --------------------------------------------------------------
    hist_start_time = time.time()
    scores = np.arange(len(hist_pre))
    pre_counts = hist_pre.numpy()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.plot(scores, pre_counts, "-o", label="Transformer Generated", color="#3498db", linewidth=2.5, markersize=7)
    plt.yscale("symlog", linthresh=1)
    perfect = 2 * N
    ax.axvline(perfect, ls=":", color="#2ecc71", label=f"Perfect ({perfect})", linewidth=2.5)
    
    # Add text annotations for number of perfect grids
    if perfect < len(pre_counts):
        num_perfect_pre = int(pre_counts[perfect])
        if num_perfect_pre > 0:
            ax.text(perfect + 0.1, num_perfect_pre, f'{num_perfect_pre}',
                    color='#3498db', va='center', ha='left', fontsize=10, weight='bold')

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.legend()
    plt.title(f"Transformer Generated Score Distribution (Generation {generation})")
    plt.xlabel("Score (# points)")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    hist_path = os.path.join(args.dump_path, f"score_distribution_gen{generation}.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    logger.info(f"Histogram saved to {hist_path}")
    logger.info(f"Histogram generation took {time.time() - hist_start_time:.2f} seconds.") 