#!/usr/bin/env python3
"""
PatternBoost skeleton for no-three-in-line problem.

TODO: Replace all sphere-specific logic with no-three-in-line constraint.
Most of the file can stay the same - only geometric constraint changes.
"""

import os, sys, time, math, argparse
from dataclasses import dataclass
from typing import List
import pprint, logging
from collections import Counter
import itertools, shutil

from makemoretokens import ModelConfig, CharDataset, Transformer, Bigram, MLP, RNN, BoW, InfiniteDataLoader, evaluate, generate, print_samples

# ============================================================================
# TODO: Import your constraint class instead of NoSphereSimple
# ============================================================================
from no_three_in_line import NoThreeInLine

def get_parser():
    parser = argparse.ArgumentParser('PatternBoost for no-three-in-line')

    # ========================================================================
    # TODO: Adjust default parameters for no-three-in-line problem
    # ========================================================================
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
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--local_rank", type=int, default=-1, help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1, help="Master port (for multi-node SLURM jobs)")

    parser.add_argument("--cpu", default=False, action=argparse.BooleanOptionalAction, help="run on cpu only")
    parser.add_argument("--device", type=str, default="auto", help="device to use for compute: auto|cpu|cuda|mps")
    
    # debug
    parser.add_argument("--debug_slurm", default=True, action=argparse.BooleanOptionalAction, help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", default=True, action=argparse.BooleanOptionalAction, help="Enable all debug flags")

    return parser

def create_datasets(input_file, force_tokens=-1):
    """Set up datasets from a .txt file consisting of tokens like V0,V1,...
    
    Flow: 
    1. Read lines of comma-separated tokens: "V1,V5,V12"
    2. Split into individual tokens: ["V1", "V5", "V12"] 
    3. Create vocabulary from all unique tokens
    4. Create train/test split
    5. Return CharDataset objects for neural network training
    """
    
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] 
    words = [w for w in words if w]  # remove empty lines
    words = [w.split(",") for w in words]  # Split "V1,V2,V3" into ["V1","V2","V3"]
    
    # Create vocabulary
    chars = sorted(list(set([i for word in words for i in word])), key=lambda x: int(x[1:]))
    
    if force_tokens >= 0:
        forced_chars = ['V'+str(i) for i in range(force_tokens)]
        assert set(chars).issubset(set(forced_chars)), f"It looks like {force_tokens=} is too small."
        chars = forced_chars
    
    # Create train/test split
    train_words = words[:int(0.9*len(words))]
    test_words = words[int(0.9*len(words)):]
    
    train_dataset = CharDataset(train_words, chars, args.max_points)
    test_dataset = CharDataset(test_words, chars, args.max_points)
    
    return train_dataset, test_dataset

def write_samples(model, train_dataset, num=10, new_file=False, use_logger=False):
    """ samples from the model and writes them to file 
    
    TODO: This function uses undefined variables (model, train_dataset, args)
    These will be defined in the main training loop where this function is called
    
    Flow:
    1. Use trained neural network to generate token sequences
    2. Start with <START> token, generate until <END> token or max length
    3. Convert token numbers back to "V1,V5,V12" format
    4. Write sequences to out.txt for decode_and_fix() to process
    """
    # TODO: These variables need to be defined in calling scope:
    # - model: trained neural network
    # - train_dataset: dataset with decode() method  
    # - args: command line arguments
    
    if args.type == 'bigram':
        # TODO: Handle bigram model sampling
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
            row = X_samp[i, 1:].tolist()  # crop out the first token, which is always the special start token
            crop_index = row.index(0) if 0 in row else len(row)  # crop out everything starting from the first end token
            row = row[:crop_index]
            word_samp = train_dataset.decode(row)
            samples.append(word_samp)
    
    out_file = args.dump_path + '/out.txt'
    # logger.info(f"Writing {len(samples)} samples to {out_file}")
    with open(out_file, "a" if not new_file else "w") as file:
        for word in samples:
            file.write(word + "\n")

def generate_2d_symmetries(construction):
    """Generate unique symmetries of 2D constructions
    
    Generates up to 8 symmetries of a 2D construction, but only yields unique ones.
    For highly symmetric constructions, this may yield fewer than 8 results.
    
    Args:
        construction: tensor representing 2D point placement
    Yields:
        unique transformed versions of the construction
    """
    seen_constructions = set()
    
    # Convert to tuple for hashing (tensors aren't hashable)
    def tensor_to_tuple(tensor):
        return tuple(tensor.flatten().tolist())
    
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
        key = tensor_to_tuple(transformed)
        if key not in seen_constructions:
            seen_constructions.add(key)
            yield transformed

def decode_and_fix(args, token_decoding, token_encoding, generation):
    """
    Core algorithm: Read neural network samples, convert to geometric constructions,
    test them with no-three-in-line constraint, and save the best ones.
    
    TODO: Adapt coordinate handling for 2D problem
    Flow:
    1. Read token sequences from out.txt: ["V1,V5,V12", "V3,V8,V15", ...]
    2. Convert tokens to 2D coordinates using token_decoding mapping
    3. For each sequence, try adding points in order using constraint solver
    4. Complete constructions with saturation (greedy point addition)
    5. Keep only best constructions (most points placed)
    6. Apply 2D symmetries to increase training data
    7. Save augmented data as training set for next generation
    """
    N = args.grid_size
    
    # Read generated sequences from neural network (unchanged)
    with open(args.dump_path + '/out.txt', 'r') as file:
        sampled_tokens = []
        for line in file:
            line = line.strip()
            if line:
                numbers = [int(item[1:]) for item in line.split(',')]  # "V1,V5,V12" -> [1,5,12]
                sampled_tokens.append(numbers)

    logger.info(f"{len(sampled_tokens)} samples decoded.")
    logger.info("first few sampled sequences: %s", sampled_tokens[0:3])

    out_string = ""
    batch_idx = 0
    total_pre_sat = 0
    total_post_sat = 0
    # Efficient histogram accumulators (CPU tensors)
    hist_pre = torch.zeros(args.max_points + 1, dtype=torch.int64)
    hist_post = torch.zeros(args.max_points + 1, dtype=torch.int64)
    # timing accumulators
    times_adding_points = []  # seconds per batch for adding points
    times_saturating = []     # seconds per batch for saturate
    
    # Global deduplication across all batches
    unique_encodings = set()
    
    # Process in batches
    for b in range(0, len(sampled_tokens), args.batch_size):
        batch_idx += 1
        cur_batch_size = min(args.batch_size, len(sampled_tokens)-b)
        
        # ====================================================================
        # TODO: Use no-three-in-line constraint instead of sphere constraint
        # ====================================================================
        no_three = NoThreeInLine(
            batch_size=cur_batch_size, 
            grid_size=N, 
            max_points=args.max_points, 
            device=args.device
        )
        
        current_batch = sampled_tokens[b:b+cur_batch_size]
        
        # Try adding points suggested by neural network
        # TODO: Handle coordinate conversion from tokens to 2D positions
        max_length = max(len(seq) for seq in current_batch)
        t_add0 = time.time()
        for i in range(max_length):
            # TODO: Create coordinate tensor appropriate for 2D problem
            # Flow: 
            # 1. For each batch, get the i-th token in the sequence
            # 2. Convert token number to (x,y) coordinate using token_decoding
            # 3. Handle missing tokens (sequences of different lengths)
            # 4. Pass coordinates to no_three.try_to_add_points()
            
            # HINT: Use something like:
            # points = -1 * torch.ones((cur_batch_size, 2), dtype=torch.int8, device=args.device)
            # for j in range(cur_batch_size):
            #     if i < len(current_batch[j]):
            #         token_num = current_batch[j][i]
            #         if token_num < len(token_decoding):
            #             points[j] = token_decoding[token_num]
            # no_three.try_to_add_points(points)

            points = -1 * torch.ones((cur_batch_size, 2), dtype=torch.int8, device=args.device)
            for j in range(cur_batch_size):
                if i < len(current_batch[j]):
                    token_num = current_batch[j][i]
                    if token_num < len(token_decoding):
                        points[j] = token_decoding[token_num]
            no_three.try_to_add_points(points)
            
        t_add1 = time.time()
        times_adding_points.append(t_add1 - t_add0)
        
        # Record points added from neural network suggestions
        total_pre_sat += torch.sum(no_three.current_counts.float()).item()
        # Update histogram efficiently
        counts_cpu = no_three.current_counts.cpu().int()
        hist_pre += torch.bincount(counts_cpu, minlength=args.max_points + 1)
        
        t_sat0 = time.time()
        no_three.saturate()  # Complete constructions greedily
        t_sat1 = time.time()
        times_saturating.append(t_sat1 - t_sat0)
        
        # Record final point counts
        total_post_sat += torch.sum(no_three.current_counts.float()).item()
        counts_cpu_post = no_three.current_counts.cpu().int()
        hist_post += torch.bincount(counts_cpu_post, minlength=args.max_points + 1)
        
        # Keep only the best constructions (unchanged logic)
        x = torch.argsort(no_three.current_counts, descending=True)
        best_constructions_per_batch = int(args.batch_size * args.keep_best_fraction)
        
        # TODO: Extract the best constructions after saturation.
        #  1. Use `torch.argsort` on `no_three.current_counts` to get the indices of the best constructions.
        #  2. Select the top `best_constructions_per_batch` indices.
        #  3. Use these indices to get the corresponding construction grids from `no_three.current_constructions`.
        #  4. Ensure the resulting `top_constructions` tensor is on the CPU for further processing.
        top_constructions = ((no_three.current_constructions[x[0:best_constructions_per_batch]] == 1) * 1).cpu()
        
        # Apply symmetries to increase training data
        if args.symmetrize:
            for construction in top_constructions:
                for symmetric_construction in generate_2d_symmetries(construction):
                    indices = torch.nonzero(symmetric_construction)
                    if indices.numel() > 0:
                        encoding = token_encoding[indices[:, 0], indices[:, 1]]
                        # Sort encoding to create canonical form (same pattern = same sorted tokens)
                        sorted_encoding = tuple(sorted(encoding.tolist()))
                        
                        # Only add if we haven't seen this pattern before
                        if sorted_encoding not in unique_encodings:
                            unique_encodings.add(sorted_encoding)
                            encoding_string = ','.join([f'V{tok}' for tok in sorted_encoding]) + '\n'
                            out_string += encoding_string

        else:
            for construction in top_constructions:
                
                # indices = torch.nonzero(construction)
                # if indices.numel() > 0:
                #     encoding = token_encoding[indices[:, 0], indices[:, 1]]
                #     encoding_string = ','.join([f'V{tok}' for tok in encoding]) + '\n'
                #     out_string += encoding_string

                indices = torch.nonzero(construction)
                if indices.numel() > 0:
                    encoding = token_encoding[indices[:, 0], indices[:, 1]]
                    encoding_string = ','.join([f'V{tok}' for tok in encoding]) + '\n'
                    out_string += encoding_string
                

    
    # Save training data for next generation (unchanged)
    training_path = args.dump_path + f"/training_sets/N{N}_gen{generation}.txt"
    with open(training_path, 'a') as f:
        f.write(out_string)
    
    logger.info(f"Generation {generation-1} -> {generation}")
    logger.info(f"Neural network contributed {total_pre_sat} points")
    logger.info(f"Saturation added {total_post_sat - total_pre_sat} more points")
    logger.info(f"Total points in final constructions: {total_post_sat}")
    logger.info(f"Training data saved to {training_path}")

    # Log deduplication statistics
    if args.symmetrize:
        logger.info(f"Generated {len(unique_encodings)} unique constructions after symmetry deduplication")

    # Convert histograms to Counter for logging
    all_counts_pre_ctr = Counter({i:int(v) for i,v in enumerate(hist_pre.tolist()) if v != 0})
    all_counts_post_ctr = Counter({i:int(v) for i,v in enumerate(hist_post.tolist()) if v != 0})
    logger.info(f"score distribution before saturation: {all_counts_pre_ctr}")
    logger.info(f"score distribution after  saturation: {all_counts_post_ctr}")
    if times_adding_points:
        logger.info(f"adding_average_time={sum(times_adding_points)/len(times_adding_points):.2f}s, saturating_average_time={sum(times_saturating)/len(times_saturating):.2f}s")

    # Clean up out.txt file (matching no_spheres behavior)
    shutil.os.remove(args.dump_path + '/out.txt')

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    import numpy as np
    import random
    
    # Parse arguments (unchanged)
    parser = get_parser()
    args = parser.parse_args()
    
    # Setup logging and directories (matching your working version)
    log_prefix = args.dump_path + "/"
    if not os.path.exists(log_prefix):
        os.makedirs(log_prefix)
    training_dir = log_prefix + 'training_sets'
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    # Configure logging with both console and file output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(log_prefix + 'program-exp.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Set device with proper MPS support
    if args.device == "auto":
        if args.cpu:
            args.device = "cpu"
        elif torch.backends.mps.is_available():
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
    
    # Find initial generation (like your working version)
    for i in range(args.max_epochs):
        if not os.path.isfile(log_prefix + f"training_sets/N{N}_gen{i}.txt"):
            break
    initial_gen = i
    
    if initial_gen > 0:
        initial_gen = initial_gen - 1  # first index for which we have training data
    else:
        # Generate initial training data (like your working version)
        logger.info("Generating 0th generation of training data...")

        training_path_root = args.dump_path + f'/training_sets/N{N}_gen{initial_gen}'
        training_path = training_path_root + '.txt'

        constructions_log = []
        best_constructions_per_batch = int(args.batch_size * args.keep_best_fraction)
        
        t0 = time.time()

        for _ in range(int(args.target_training_size / best_constructions_per_batch)):
            # Create initial random data
            no_three = NoThreeInLine(
                batch_size=args.batch_size,
                grid_size=args.grid_size,
                max_points=args.max_points,
                device=args.device
            )
            no_three.saturate()

            # sort according to number of points
            x = torch.argsort(no_three.current_counts, descending=True)
            top_constructions = ((no_three.current_constructions[x[0:best_constructions_per_batch]] == 1) * 1).cpu()

            constructions_log += no_three.current_counts.int().tolist()

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
                with open(training_path_root + '_unpermuted.txt', 'a') as f:
                    f.write(out_string_unpermuted)

        if args.device == "cuda":
            logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB")
        logger.info(f"Generated {len(constructions_log)} constructions.")
        logger.info(f"Generation took {time.time()-t0:.2f} seconds.")
        logger.info(f"Distribution of counts = {Counter(constructions_log)}")

    assert os.path.isfile(log_prefix + f"training_sets/N{N}_gen{initial_gen}.txt")

        

    logger.info(f"initializing at generation: {initial_gen}")
    input_file = args.dump_path + f"/training_sets/N{N}_gen{initial_gen}.txt"
    train_dataset, test_dataset = create_datasets(input_file, force_tokens=N**2)
    vocab_size = train_dataset.get_vocab_size()
    block_size = args.max_points + 1
    logger.info(f"dataset determined that: {vocab_size=}, {block_size=}")

    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                    n_layer=args.n_layer, n_head=args.n_head,
                    n_embd=args.n_embd, n_embd2=args.n_embd2)
    
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn')
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru')
    elif args.type == 'bow':
        model = BoW(config)
    else:
        logger.error(f'model type {args.type} is not recognized')
    model.to(args.device)
    logger.info(f"model #params: {sum(p.numel() for p in model.parameters())}")
    model_path = os.path.join(args.dump_path, "model.pt")
    if os.path.isfile(model_path): # Note: if we sample-only then we also assume we are resuming
        logger.info("resuming from existing model")
        model.load_state_dict(torch.load(model_path))
    # ========================================================================
    # MAIN TRAINING LOOP (matching your working version structure)
    # ========================================================================
    for generation in range(initial_gen, args.max_epochs):
        logger.info(f"============ Start of generation {generation} ============")
        if args.device == "cuda":
            logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB")

        input_file = args.dump_path + f"/training_sets/N{N}_gen{generation}.txt"
        train_dataset, test_dataset = create_datasets(input_file,force_tokens=N**2)
        vocab_size = train_dataset.get_vocab_size()
        block_size = args.max_points + 1
        logger.info(f"dataset determined that: {vocab_size=}, {block_size=}")

        config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                    n_layer=args.n_layer, n_head=args.n_head,
                    n_embd=args.n_embd, n_embd2=args.n_embd2)
    
        if args.type == 'transformer': model = Transformer(config)
        elif args.type == 'bigram': model = Bigram(config)
        elif args.type == 'mlp': model = MLP(config)
        elif args.type == 'rnn': model = RNN(config, cell_type='rnn')
        elif args.type == 'gru': model = RNN(config, cell_type='gru')
        elif args.type == 'bow': model = BoW(config)
        else: logger.error(f'model type {args.type} is not recognized')
        
        model.to(args.device)
        logger.info(f"model #params: {sum(p.numel() for p in model.parameters())}")
        model_path = os.path.join(args.dump_path, "model.pt")
        if os.path.isfile(model_path):
            logger.info("resuming from existing model")
            model.load_state_dict(torch.load(model_path))

        logger.info(f"training on {input_file}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.nn_batch_size, pin_memory=True, num_workers=args.num_workers)

        # training loop
        best_loss = None
        step = 0
        t_training = time.time()
        while True:
            t0 = time.time()

            # get the next batch, ship to device, and unpack it to input and target
            batch = batch_loader.next()
            batch = [t.to(args.device) for t in batch]
            X, Y = batch

            # feed into the model
            logits, loss = model(X, Y)

            # calculate the gradient, update the weights
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if args.device == "cuda": torch.cuda.synchronize()
            elif args.device == "mps": torch.mps.synchronize()
            t1 = time.time()

            # logging
            if step % 100 == 0:
                logger.info(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

            # evaluate the model
            if step > 0 and step % 500 == 0:
                train_loss = evaluate(model, train_dataset, args.device, batch_size=100, max_batches=10)
                test_loss  = evaluate(model, test_dataset,  args.device, batch_size=100, max_batches=10)
                logger.info(f"step {step} train loss: {train_loss} test loss: {test_loss}")
                # save the model to disk if it has improved
                if best_loss is None or test_loss < best_loss:
                    out_path = os.path.join(args.dump_path, "model.pt")
                    logger.info(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                    torch.save(model.state_dict(), out_path)
                    best_loss = test_loss
                    
            step += 1
            if args.max_steps >= 0 and step >= args.max_steps:
                break
        logger.info(f"training took {time.time()-t_training:.2f} seconds")
        if args.device == "cuda":
            logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB")

        t_generating = time.time()
        logger.info('generating new samples...')
        sample_batch_size = args.gen_batch_size
        todo = int(args.target_training_size * (1/args.keep_best_fraction))
        
        # Clear previous samples and generate new ones
        write_samples(model, train_dataset, num=0, new_file=True)
        generated_count = 0
        while generated_count < todo:
            num_to_gen = min(sample_batch_size, todo - generated_count)
            write_samples(model, train_dataset, num=num_to_gen)
            generated_count += num_to_gen

        logger.info(f"generation took {time.time()-t_generating:.2f} seconds")
        logger.info('decoding and fixing')
        
        if args.device in ["mps", "cuda"]:
            torch.mps.empty_cache() if args.device == "mps" else torch.cuda.empty_cache()
            
        if generation < args.max_epochs:
            decode_and_fix(args, token_decoding=token_decoding, token_encoding=token_encoding, generation=generation+1)
        
        if args.device in ["mps", "cuda"]:
            mem_allocated = torch.mps.current_allocated_memory() if args.device == "mps" else torch.cuda.memory_allocated(0)
            logger.info(f"Memory allocated: {mem_allocated/(1024*1024):.2f}MB")
        logger.info(f"============ End of generation {generation} ============")
    
    logger.info("All generations completed!")