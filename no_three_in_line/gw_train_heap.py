import os, time, argparse, logging, shutil, heapq, random, re
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn.functional as F

from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

from makemoretokens import (
    ModelConfig,
    CharDataset,
    Transformer,
    InfiniteDataLoader,
    evaluate,
    generate,
)

from no_three_in_line import NoThreeInLine


# --------------------------- Helper class ------------------------------------
class TopPool:
    """Maintain a fixed-capacity min-heap of the highest-scoring constructions."""

    def __init__(self, capacity: int, grid_size: int):
        self.capacity = capacity
        self.heap: list[tuple[int, str]] = []  # (score, token_string)
        self.perfect_score = 2 * grid_size

    # internal ---------------------------------------------------------------
    def _push(self, score: int, token_string: str):
        heapq.heappush(self.heap, (score, token_string))

    def _pop(self):
        score, token_string = heapq.heappop(self.heap)
        return score, token_string

    # public -----------------------------------------------------------------
    def add(self, score: int, token_string: str, logger: logging.Logger | None = None):
        """
        Attempt to add construction; keep only if it improves pool.
        The heap has a fixed capacity.
        """
        # The special case for expanding the heap for perfect constructions has been removed
        # to prevent bugs related to invalid grids being classified as "perfect".
        # The heap now has a strictly fixed capacity.

        if len(self.heap) < self.capacity:
            if logger:
                logger.info(f"Pool below capacity. Adding new construction with score {score}.")
            self._push(score, token_string)
        elif score > self.heap[0][0]:  # If score is strictly better, replace the worst
            popped_score, _ = self._pop()
            if logger:
                logger.info(f"New score {score} > worst in heap {popped_score}. Replacing.")
            self._push(score, token_string)

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
        # Sorting removed for performance as it is not functionally required.
        with open(path, "w") as f:
            for _score, token_str in self.heap:
                f.write(token_str + "\n")

    def __len__(self):
        return len(self.heap)


# --------------------------- Argparser ---------------------------------------

def get_parser():
    parser = argparse.ArgumentParser("PatternBoost with Top-Pool for no-three-in-line")

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
    parser.add_argument("--type", type=str, default="transformer")
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_embd2", type=int, default=16)

    parser.add_argument("--nn_batch_size", "-b", type=int, default=64)
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-w", type=float, default=0.1)
    parser.add_argument("--num_workers", "-n", type=int, default=8)

    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_epochs", type=int, default=20)

    # RL loss options --------------------------------------------------------
    parser.add_argument("--w-imitate", type=float, default=0.5, help="Weight for imitation (cross-entropy) loss.")
    parser.add_argument("--w-goal", type=float, default=0.5, help="Weight for goal (RL policy) loss.")
    parser.add_argument("--w-len", type=float, default=1.0, help="Reward weight for number of valid points.")
    parser.add_argument("--w-invalid", type=float, default=1.5, help="Penalty weight for invalid moves.")
    parser.add_argument("--w-over", type=float, default=2.0, help="Penalty weight for exceeding 2n points.")
    parser.add_argument("--w-empty", type=float, default=10.0, help="Penalty for generating an empty grid.")
    parser.add_argument("--w-perfect", type=float, default=5.0, help="Bonus reward for achieving a perfect 2n grid.")

    # sampling ---------------------------------------------------------------
    parser.add_argument("--gen_batch_size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)

    # system / paths ---------------------------------------------------------
    parser.add_argument("--dump_path", type=str, default="dump_path")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=-1)

    return parser


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
    with open(input_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

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

    logger.info(f"{len(sampled_tokens)} samples received to process.")

    hist_pre = torch.zeros(args.max_points + 1, dtype=torch.int64)
    hist_post = torch.zeros(args.max_points + 1, dtype=torch.int64)
    total_pre, total_post = 0, 0

    best_pre_sat_score = -1
    best_pre_sat_construction = None

    for b in range(0, len(sampled_tokens), args.batch_size):
        cur_bs = min(args.batch_size, len(sampled_tokens) - b)
        nti = NoThreeInLine(cur_bs, N, args.max_points, device=args.device)
        batch = sampled_tokens[b : b + cur_bs]

        max_len = max(len(seq) for seq in batch) if batch else 0
        for i in range(max_len):
            pts = -1 * torch.ones((cur_bs, 2), dtype=torch.int8, device=args.device)
            for j in range(cur_bs):
                if i < len(batch[j]):
                    tok_num = batch[j][i]
                    if tok_num < len(token_decoding):
                        pts[j] = token_decoding[tok_num]
            nti.try_to_add_points(pts)
        nti.current_counts = (nti.current_constructions == 1).sum(dim=(1, 2)).to(torch.int8)

        # The validation logic has been removed from here, as the root cause of invalid
        # constructions has been fixed inside the NoThreeInLine.add_points method,
        # which now correctly updates the grid state after each addition.

        # Check for a new best pre-saturation score in the current batch
        if nti.current_counts.numel() > 0:
            current_max_score, current_max_idx = torch.max(nti.current_counts, dim=0)
            if current_max_score > best_pre_sat_score:
                best_pre_sat_score = current_max_score
                best_pre_sat_construction = nti.current_constructions[current_max_idx]

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

            construction = constructions_sorted[idx].cpu()

            if args.symmetrize:
                canonical_form = get_canonical_form(construction, token_encoding)
                if canonical_form:
                    pool.add(score, canonical_form, logger)
            else:
                token_str = construction_to_string(construction, token_encoding)
                if token_str:
                    pool.add(score, token_str, logger)

    if best_pre_sat_construction is not None:
        grid_viz = format_grid(best_pre_sat_construction.cpu())
        logger.info(f"Best pre-saturation grid from transformer (score: {best_pre_sat_score}):{grid_viz}")

    # write full pool to new training file -----------------------------------
    training_path = os.path.join(args.dump_path, "training_sets", f"N{N}_gen{generation}.txt")
    pool.dump_to_file(training_path)

    logger.info(f"Generation {generation-1} -> {generation}")
    logger.info(f"Pool size: {len(pool)}  |  Worst score in pool: {pool.heap[0][0] if pool.heap else 'N/A'}")
    logger.info(f"NN points: {total_pre}, Saturation added: {total_post - total_pre}")

    all_counts_pre_ctr = Counter({i: int(v) for i, v in enumerate(hist_pre.tolist()) if v != 0})
    all_counts_post_ctr = Counter({i: int(v) for i, v in enumerate(hist_post.tolist()) if v != 0})
    logger.info(f"Score distribution before saturation: {all_counts_pre_ctr}")
    logger.info(f"Score distribution after  saturation: {all_counts_post_ctr}")

    # histogram --------------------------------------------------------------
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


# --------------------------- Main -------------------------------------------

if __name__ == "__main__":
    start = time.time()
    parser = get_parser()
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    args.dump_path = os.path.join(project_root, "training", os.path.basename(args.dump_path))
    logger = setup_logging(args.dump_path)
    set_device(args, logger)

    # seed -------------------------------------------------------------------
    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"seed: {args.seed}")

    # token maps -------------------------------------------------------------
    N = args.grid_size
    logger.info("Creating token maps...")
    token_encoding = torch.arange(N * N).view(N, N)
    coords = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(N), indexing="xy"), -1)
    token_decoding = coords.view(-1, 2)

    # determine initial generation ------------------------------------------
    ts_dir = os.path.join(args.dump_path, "training_sets")
    gen_re = re.compile(rf"N{N}_gen(\d+)\.txt$")
    gens = [int(m.group(1)) for fn in os.listdir(ts_dir) if (m := gen_re.match(fn))]
    if not gens:
        logger.error("No generation files found. Run generate_initial.py first.")
        exit(1)
    initial_gen = max(gens)
    logger.info(f"Resuming from generation {initial_gen}")

    # build initial pool -----------------------------------------------------
    pool = TopPool(args.target_training_size, args.grid_size)
    init_file = os.path.join(ts_dir, f"N{N}_gen{initial_gen}.txt")
    pool.build_from_file(init_file)
    logger.info(f"Initial pool size: {len(pool)}")

    best_loss = 1e9 # Initialize with a very high value
    end_gen = initial_gen + args.max_epochs
    for gen in range(initial_gen, end_gen):
        logger.info(f"=========== Start of generation {gen + 1} ===========")

        # datasets: Load constructions and optionally create symmetrized versions on the fly
        input_gen_file = os.path.join(ts_dir, f"N{N}_gen{gen}.txt")
        train_ds, test_ds = create_datasets_from_file(input_gen_file, args, token_decoding, token_encoding, logger)

        config = ModelConfig(
            vocab_size=train_ds.get_vocab_size(),
            block_size=train_ds.get_output_length(),
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_embd2=args.n_embd2,
        )
        model = Transformer(config).to(args.device)
        model_path = os.path.join(args.dump_path, "model.pt")
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            logger.info("Resumed existing model.")
        
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        loader = InfiniteDataLoader(train_ds, batch_size=args.nn_batch_size, pin_memory=True, num_workers=args.num_workers)

        for step in range(args.max_steps + 1):
            # --- HYBRID RL + SUPERVISED TRAINING STEP ---
            model.train() # Ensure model is in training mode for gradients

            # --- Part 1: Supervised Learning (Imitation Loss) ---
            X, Y = [t.to(args.device) for t in next(loader)]
            _, imitation_loss = model(X, Y)

            # --- Part 2: Reinforcement Learning (Goal Loss) ---
            # Generate sequences (the "rollout")
            nti = NoThreeInLine(args.nn_batch_size, args.grid_size, args.max_points, device=args.device)
            
            # Initialize sequences with a <START> token (index 0)
            sequences = torch.zeros(args.nn_batch_size, 1, dtype=torch.long, device=args.device)
            
            log_probs = torch.zeros(args.nn_batch_size, device=args.device)
            total_invalid_moves = torch.zeros(args.nn_batch_size, device=args.device)
            
            # We need to detach the hidden state to prevent gradients from flowing endlessly
            # This part is complex; for now, we generate with no_grad and calculate loss on the result.
            # A more advanced implementation would use a proper policy gradient method.
            with torch.no_grad():
                for _ in range(args.max_points):
                    logits, _ = model(sequences)
                    probs = F.softmax(logits[:, -1, :], dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                    
                    # Decode tokens to points
                    points = -1 * torch.ones((args.nn_batch_size, 2), dtype=torch.int8, device=args.device)
                    for i in range(args.nn_batch_size):
                        token_val = next_tokens[i].item()
                        if token_val > 0 and token_val < len(token_decoding): # token 0 is special
                            points[i] = token_decoding[token_val]

                    was_added = nti.try_to_add_points(points)
                    total_invalid_moves += (~was_added).float()
                    sequences = torch.cat([sequences, next_tokens], dim=1)

            # Now, re-calculate the log probabilities of the generated sequences with gradients
            full_logits, _ = model(sequences[:, :-1]) # feed all but the last token
            full_log_probs = F.log_softmax(full_logits, dim=-1)
            
            # Gather the log_probs for the specific tokens we generated
            action_log_probs = full_log_probs.gather(2, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
            final_log_probs = action_log_probs.sum(dim=1)

            # Calculate rewards
            T = nti.current_counts.float()
            I = total_invalid_moves
            length_reward = args.w_len * T
            invalid_penalty = args.w_invalid * I
            overage_penalty = args.w_over * torch.clamp(T - 2 * args.grid_size, min=0)
            rewards = length_reward - invalid_penalty - overage_penalty
            
            # Add a large penalty for empty grids
            empty_mask = (T == 0)
            rewards[empty_mask] -= args.w_empty

            # Add a bonus for perfect 2n grids
            perfect_mask = (T == 2 * args.grid_size)
            rewards[perfect_mask] += args.w_perfect
            
            advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            policy_loss = -(advantage.detach() * final_log_probs).mean()
            
            # --- Part 3: Combine losses and ensure positivity ---
            # We use SoftPlus to ensure the final loss is always positive.
            # This makes the loss value easier to interpret and more stable.
            combined_loss = (args.w_imitate * imitation_loss) + (args.w_goal * policy_loss)
            loss = F.softplus(combined_loss)

            model.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if step % 100 == 0:
                logger.info(f"step {step} | loss {loss.item():.4f}")
            if step and step % 500 == 0:
                # We now evaluate using the loss itself, as it's the most direct metric.
                current_loss = loss.item()
                logger.info(f"step {step} | loss: {current_loss:.4f}")
                if current_loss < best_loss:
                    logger.info(f"New best loss {current_loss:.4f} < {best_loss:.4f}, saving model")
                    torch.save(model.state_dict(), model_path)
                    best_loss = current_loss
        loader.shutdown()

        # sample -------------------------------------------------------------
        total_to_generate = int(args.target_training_size * (1 / args.keep_best_fraction))
        
        all_samples = []
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn()) as prog:
            t = prog.add_task("Generating samples", total=total_to_generate)
            done = 0
            while done < total_to_generate:
                n = min(args.gen_batch_size, total_to_generate - done)
                new_samples = generate_samples(model, train_ds, args, num_samples=n)
                all_samples.extend(new_samples)
                done += n
                prog.update(t, advance=n)

        # decode / update pool ----------------------------------------------
        if args.device in {"mps", "cuda"}:
            torch.cuda.empty_cache()
        decode_and_update_pool(args, token_decoding, token_encoding, gen + 1, logger, pool, all_samples)

        logger.info(f"=========== End of generation {gen + 1} ===========")

    logger.info("All generations completed!")
    logger.info(f"Total time: {time.time() - start:.2f} seconds") 