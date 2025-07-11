import os, time, argparse
import pprint, logging
from collections import Counter
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import random
import matplotlib.ticker as mticker
import re

from rich.logging import RichHandler
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from makemoretokens import (
    ModelConfig, 
    CharDataset, 
    Transformer, Bigram, MLP, RNN, BoW, 
    InfiniteDataLoader, 
    evaluate,
    generate,
)

from no_three_in_line import NoThreeInLine

def get_parser():
    parser = argparse.ArgumentParser('PatternBoost for no-three-in-line')

    parser.add_argument('--grid_size', type=int, default=6, help='Grid size for 2D no-three-in-line (creates NxN grid)')
    parser.add_argument('--batch_size', type=int, default=500, help='Generate and process samples in batches of this size')
    parser.add_argument('--max_points', type=int, default=18, help='max points which can be added to construction')
    parser.add_argument('--target_training_size', type=int, default=20000, help='number of examples to aim for (before symmetrization)')
    parser.add_argument('--keep_best_fraction', type=float, default=0.1, help='Percentage of good constructions to keep')
    parser.add_argument('--symmetrize', default=True, action=argparse.BooleanOptionalAction, help='symmetrize constructions, set to --no-symmetrize to disable')

    # Makemore / Neural network params (matching your working version)
    parser.add_argument('--num-workers', '-n', type=int, default=8, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=5000, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--max_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--seed', type=int, default=-1, help="seed")
    
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=16, help="number of feature channels elsewhere in the model")
    
    # optimization
    parser.add_argument('--nn-batch-size', '-b', type=int, default=64, help="batch size during neural network optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.1, help="weight decay")

    parser.add_argument('--gen_batch_size', type=int, default=10, help="batch size for generation from transformer")
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
    
    # path and system
    parser.add_argument("--dump_path", type=str, default="dump_path", help="Experiment dump path")
    parser.add_argument("--device", type=str, default="auto", help="device to use for compute: auto|cpu|cuda|mps")

    return parser

def create_datasets(input_file, args):
    """Set up datasets from a .txt file consisting of tokens like V0,V1,..."""
    
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] 
    words = [w for w in words if w]  # remove empty lines
    words = [w.split(",") for w in words]  # Split "V1,V2,V3" into ["V1","V2","V3"]
    
    # Create vocabulary
    chars = sorted(list(set([i for word in words for i in word])), key=lambda x: int(x[1:]))
    
    N_squared = args.grid_size**2
    forced_chars = ['V'+str(i) for i in range(N_squared)]
    assert set(chars).issubset(set(forced_chars)), f"It looks like force_tokens={N_squared} is too small."
    chars = forced_chars
    
    # Create train/test split
    train_words = words[:int(0.9*len(words))]
    test_words = words[int(0.9*len(words)):]
    
    train_dataset = CharDataset(train_words, chars, args.max_points)
    test_dataset = CharDataset(test_words, chars, args.max_points)
    
    return train_dataset, test_dataset

def write_samples(model, train_dataset, args, num=10, new_file=False):
    """samples from the model and writes them to file"""
    if args.type == 'bigram':
        samples = []
        for _ in range(num):
            out = []
            context = [0]  # starts with special start character
            while True:
                logits = model(torch.tensor([context], dtype=torch.long, device=args.device))
                probs = F.softmax(logits, dim=-1)
                ix = torch.multinomial(probs, num_samples=1, generator=torch.Generator(device=args.device).manual_seed(random.randint(0, 10000))).item()
                context = [ix]
                if ix == 0:
                    break
                out.append(ix)
            samples.append(train_dataset.decode(out))
    else:
        X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
        top_k = args.top_k if args.top_k != -1 else None
        steps = train_dataset.get_output_length() - 1
        X_samp = generate(model, X_init, steps, temperature=args.temperature, top_k=top_k, do_sample=True).to('cpu')
        
        samples = []
        for i in range(X_samp.size(0)):
            row = X_samp[i, 1:].tolist()  # crop out the first token
            crop_index = row.index(0) if 0 in row else len(row)
            row = row[:crop_index]
            word_samp = train_dataset.decode(row)
            samples.append(word_samp)
    
    out_file = args.dump_path + '/out.txt'
    with open(out_file, "a" if not new_file else "w") as file:
        for word in samples:
            file.write(word + "\n")

def generate_2d_symmetries(construction):
    """Generate unique symmetries of 2D constructions"""
    transformations = []
    
    for k in range(4):
        transformations.append(torch.rot90(construction, k, [0, 1]))
    
    flipped = torch.flip(construction, [0])
    for k in range(4):
        transformations.append(torch.rot90(flipped, k, [0, 1]))
    
    for transformed in transformations:
        yield transformed

def decode_and_fix(args, token_decoding, token_encoding, generation, logger):
    """Core algorithm: Read samples, convert to constructions, test, and save the best."""
    N = args.grid_size
    
    with open(args.dump_path + '/out.txt', 'r') as file:
        sampled_tokens = [
            [int(item[1:]) for item in line.strip().split(',')]
            for line in file if line.strip()
        ]

    logger.info(f"{len(sampled_tokens)} samples decoded.")

    out_string = ""
    total_pre_sat, total_post_sat = 0, 0
    hist_pre = torch.zeros(args.max_points + 1, dtype=torch.int64)
    hist_post = torch.zeros(args.max_points + 1, dtype=torch.int64)
    
    for b in range(0, len(sampled_tokens), args.batch_size):
        cur_batch_size = min(args.batch_size, len(sampled_tokens)-b)
        
        nothreeinline = NoThreeInLine(
            batch_size=cur_batch_size, 
            grid_size=N, 
            max_points=args.max_points, 
            device=args.device
        )
        
        current_batch = sampled_tokens[b:b+cur_batch_size]
        
        max_length = max(len(seq) for seq in current_batch) if current_batch else 0
        for i in range(max_length):
            points = -1 * torch.ones((cur_batch_size, 2), dtype=torch.int8, device=args.device)
            for j in range(cur_batch_size):
                if i < len(current_batch[j]):
                    token_num = current_batch[j][i]
                    if token_num < len(token_decoding):
                        points[j] = token_decoding[token_num]
            nothreeinline.try_to_add_points(points)
        
        nothreeinline.current_counts = (nothreeinline.current_constructions == 1).sum(dim=(1, 2)).to(torch.int8)

        total_pre_sat += torch.sum(nothreeinline.current_counts).item()
        hist_pre.add_(torch.bincount(nothreeinline.current_counts.cpu(), minlength=args.max_points + 1))

        nothreeinline.saturate()
        total_post_sat += torch.sum(nothreeinline.current_counts).item()
        hist_post.add_(torch.bincount(nothreeinline.current_counts.cpu(), minlength=args.max_points + 1))
        
        x = torch.argsort(nothreeinline.current_counts, descending=True)
        best_per_batch = int(cur_batch_size * args.keep_best_fraction)
        
        top_constructions = ((nothreeinline.current_constructions[x[0:best_per_batch]] == 1) * 1).cpu()
        
        if args.symmetrize:
            for construction in top_constructions:
                for sym_con in generate_2d_symmetries(construction):
                    indices = torch.nonzero(sym_con)
                    if indices.numel() > 0:
                        encoding = token_encoding[indices[:, 0], indices[:, 1]]
                        sorted_encoding = tuple(sorted(encoding.tolist()))
                        out_string += ','.join([f'V{tok}' for tok in sorted_encoding]) + '\n'
        else:
            for construction in top_constructions:
                indices = torch.nonzero(construction)
                if indices.numel() > 0:
                    encoding = token_encoding[indices[:, 0], indices[:, 1]]
                    out_string += ','.join([f'V{tok}' for tok in encoding]) + '\n'
    
    training_path = args.dump_path + f"/training_sets/N{N}_gen{generation}.txt"
    with open(training_path, 'w') as f:
        f.write(out_string)
    
    logger.info(f"Generation {generation-1} -> {generation}")
    logger.info(f"Max points: {torch.max(nothreeinline.current_counts).item() if nothreeinline.current_counts.numel() > 0 else 0}")
    logger.info(f"NN points: {total_pre_sat}, Saturation added: {total_post_sat - total_pre_sat}")
    logger.info(f"Training data saved to {training_path}")

    all_counts_pre_ctr = Counter({i:int(v) for i,v in enumerate(hist_pre.tolist()) if v != 0})
    all_counts_post_ctr = Counter({i:int(v) for i,v in enumerate(hist_post.tolist()) if v != 0})
    
    plt.style.use('seaborn-v0_8-whitegrid')
    scores = np.arange(max(len(hist_pre), len(hist_post)))

    pre_counts = np.pad(hist_pre.tolist(), (0, len(scores) - len(hist_pre)), 'constant')
    post_counts = np.pad(hist_post.tolist(), (0, len(scores) - len(hist_post)), 'constant')

    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    
    plt.plot(scores, pre_counts, color='#3498db', marker='o', linestyle='--', linewidth=1.5, markersize=5, label='Pre-Saturation')
    plt.plot(scores, post_counts, color='#e74c3c', marker='o', linestyle='-', linewidth=2.5, markersize=7, label='Post-Saturation')

    plt.yscale('symlog', linthresh=1)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    
    perfect_score = 2 * args.grid_size
    plt.xlim(-0.5, perfect_score + 0.5)
    
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, min_n_ticks=10))
    plt.xticks(rotation=0)

    plt.axvline(x=perfect_score, color='#2ecc71', linestyle=':', linewidth=2.5, label=f'Perfect Score ({perfect_score})')
    
    if perfect_score < len(post_counts):
        num_perfect = post_counts[perfect_score]
        if num_perfect > 0:
            ax.text(perfect_score, num_perfect, f' {int(num_perfect)} Found', 
                    color='#27ae60', va='center', ha='left', fontsize=12, weight='bold')

    plt.title(f'Score Distribution Comparison (Generation {generation})', fontsize=18, pad=20)
    plt.xlabel('Score (Number of Points)', fontsize=14, labelpad=15)
    plt.ylabel('Number of Constructions (Log Scale)', fontsize=14, labelpad=15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = plt.legend(frameon=True, loc='upper left')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')

    plt.tight_layout()

    histogram_path = os.path.join(args.dump_path, f'score_distribution_gen{generation}.png')
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Score distribution histogram saved to {histogram_path}")

    logger.info(f"score distribution before saturation: {all_counts_pre_ctr}")
    logger.info(f"score distribution after  saturation: {all_counts_post_ctr}")

    shutil.os.remove(args.dump_path + '/out.txt')

def setup_logging(dump_path):
    log_prefix = dump_path + "/"
    if not os.path.exists(log_prefix):
        os.makedirs(log_prefix)
    training_dir = os.path.join(log_prefix, 'training_sets')
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(os.path.join(log_prefix, "training.log"))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
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

if __name__ == '__main__':
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    
    args.dump_path = f"training/{os.path.basename(args.dump_path)}"
    logger = setup_logging(args.dump_path)
    set_device(args, logger)

    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"seed: {args.seed}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    N = args.grid_size
    logger.info(f"Creating token maps for {N}x{N} grid...")
    token_encoding = torch.arange(N*N).view(N, N)
    coords = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(N), indexing='xy'), -1)
    token_decoding = coords.view(-1, 2)
    logger.info("Token maps created successfully.")
    
    # Find initial generation by listing existing generation files
    training_sets_dir = os.path.join(args.dump_path, 'training_sets')
    try:
        filenames = os.listdir(training_sets_dir)
    except FileNotFoundError:
        logger.error(f"Training sets directory not found: {training_sets_dir}")
        exit(1)
    gen_pattern = re.compile(rf'N{N}_gen(\d+)\.txt$')
    gens = [int(m.group(1)) for fn in filenames if (m := gen_pattern.match(fn))]
    if not gens:
        logger.error(f"No generation files found in {training_sets_dir}. Please run generate_initial.py first.")
        exit(1)
    initial_gen = max(gens)
    logger.info(f"Resuming training from generation: {initial_gen}")

    # Determine end generation: at least initial_gen+1, or args.max_epochs, whichever is larger
    end_gen = initial_gen + args.max_epochs
    logger.info(f"Will run through generation {end_gen - 1} (range {initial_gen} to {end_gen - 1})")

    # Main training loop
    for generation in range(initial_gen, end_gen):
        # Log the generation number that will be produced
        logger.info(f"============ Start of generation {generation + 1} ============")
        
        input_file = os.path.join(args.dump_path, f"training_sets/N{N}_gen{generation}.txt")
        train_dataset, test_dataset = create_datasets(input_file, args)
        
        config = ModelConfig(vocab_size=train_dataset.get_vocab_size(), 
                             block_size=train_dataset.get_output_length(),
                             n_layer=args.n_layer, n_head=args.n_head,
                             n_embd=args.n_embd, n_embd2=args.n_embd2)
    
        model = Transformer(config)
        model.to(args.device)
        logger.info(f"model #params: {sum(p.numel() for p in model.parameters())}")
        
        model_path = os.path.join(args.dump_path, "model.pt")
        if os.path.isfile(model_path):
            logger.info("resuming from existing model")
            model.load_state_dict(torch.load(model_path))

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.nn_batch_size, pin_memory=True, num_workers=args.num_workers)

        best_loss = None
        for step in range(args.max_steps + 1):
            t0 = time.time()
            batch = next(batch_loader)
            X, Y = [t.to(args.device) for t in batch]

            logits, loss = model(X, Y)

            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                logger.info(f"step {step} | loss {loss.item():.4f} | time {(time.time()-t0)*1000:.2f}ms")

            if step > 0 and step % 500 == 0:
                train_loss = evaluate(model, train_dataset, args.device, max_batches=10)
                test_loss = evaluate(model, test_dataset, args.device, max_batches=10)
                logger.info(f"step {step} train loss: {train_loss} test loss: {test_loss}")
                if best_loss is None or test_loss < best_loss:
                    logger.info(f"test loss {test_loss} is best so far, saving model")
                    torch.save(model.state_dict(), model_path)
                    best_loss = test_loss
        
        batch_loader.shutdown()

        logger.info('generating new samples...')
        todo = int(args.target_training_size * (1/args.keep_best_fraction))
        write_samples(model, train_dataset, args, num=0, new_file=True)
        # Ensure sample_batch_size is defined
        sample_batch_size = args.gen_batch_size
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Generating samples...", total=todo)
            generated_count = 0
            while generated_count < todo:
                num_to_gen = min(sample_batch_size, todo - generated_count)
                write_samples(model, train_dataset, args, num=num_to_gen)
                generated_count += num_to_gen
                progress.update(task, advance=num_to_gen)

        logger.info('decoding and fixing')
        if args.device in ["mps", "cuda"]:
            torch.cuda.empty_cache()

        # Always decode and fix (generate histogram) after each trained generation
        decode_and_fix(args, token_decoding, token_encoding, generation + 1, logger)
        
        logger.info(f"============ End of generation {generation+ 1} ============")
    
    logger.info("All generations completed!")
    end_time = time.time()
    logger.info(f"Total time: {end_time - start_time:.2f} seconds") 