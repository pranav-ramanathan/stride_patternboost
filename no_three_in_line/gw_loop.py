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
    parser.add_argument('--grid_size', type=int, default=6, help='Grid size (creates NxN 2D grid)')  
    # TODO: Consider smaller grid_size for 2D (e.g., 8-12) since we have fewer positions
    
    parser.add_argument('--batch_size', type=int, default=500, help='Generate and process samples in batches')
    
    parser.add_argument('--max_points', type=int, default=18, help='max points which can be added')  
    # TODO: Consider larger max_points for 2D - might be able to place more before hitting constraint
    
    parser.add_argument('--target_training_size', type=int, default=20000, help='number of examples to aim for')
    parser.add_argument('--keep_best_fraction', type=float, default=0.1, help='Percentage of good constructions to keep')
    parser.add_argument('--symmetrize', default=True, action=argparse.BooleanOptionalAction, help='symmetrize constructions')
    
    # Neural network parameters (unchanged)
    parser.add_argument('--max-steps', type=int, default=5000, help="max optimization steps")
    parser.add_argument('--type', type=str, default='transformer', choices=['transformer', 'bigram', 'mlp', 'rnn', 'bow'], help="model type")
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.01, help="weight decay")
    parser.add_argument('--batch-size', '-b', type=int, default=64, help="batch size for training neural network")
    parser.add_argument('--sequence-length', '-s', type=int, default=128, help="sequence length")
    parser.add_argument('--device', type=str, default='cpu', help="device")
    parser.add_argument('--warmup-iters', type=int, default=100, help="linear learning rate warmup")
    parser.add_argument('--sample-every', type=int, default=100, help="how often to sample")
    parser.add_argument('--dump_path', type=str, default='./logs', help="model path")
    
    # Generation parameters (unchanged)
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature for sampling")
    parser.add_argument('--top_k', type=int, default=-1, help="top-k sampling")
    
    # Evolution parameters (unchanged)
    parser.add_argument('--max_epochs', type=int, default=10, help="number of generations to run")
    parser.add_argument('--initial_gen', type=int, default=0, help="generation to start from")
    
    return parser

def create_datasets(input_file, force_tokens=-1):
    """Set up datasets from a .txt file consisting of tokens like V0,V1,...
    
    TODO: No changes needed - this handles token sequences generically
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
    
    train_dataset = CharDataset(train_words, chars, args.sequence_length)
    test_dataset = CharDataset(test_words, chars, args.sequence_length)
    
    return train_dataset, test_dataset

def write_samples(
    model,
    train_dataset,
    num=10,
    new_file=False,
    use_logger=False,
):
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
    logger.info(f"Writing {len(samples)} samples to {out_file}")
    with open(out_file, "a" if not new_file else "w") as file:
        for word in samples:
            file.write(word + "\n")

def generate_2d_symmetries(construction):
    """Generate symmetries of 2D constructions
    
    TODO: Implement 2D symmetry generation
    Flow:
    1. For 2D NxN grid, we have 8 total symmetries:
       - 4 rotations: 0°, 90°, 180°, 270°
       - 4 reflections: horizontal flip, vertical flip, diagonal flips
    2. Apply each transformation to input construction
    3. Yield each transformed version
    4. This increases training data diversity (1 good construction → 8 training examples)
    
    Args:
        construction: tensor representing 2D point placement
    Yields:
        transformed versions of the construction
    """
    # TODO: Implement 2D transformations
    # For now, just yield original (no symmetries)
    yield construction

def decode_and_fix(args, token_decoding, generation):
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
        for i in range(max_length):
            # TODO: Create coordinate tensor appropriate for 2D problem
            # Flow: 
            # 1. For each batch, get the i-th token in the sequence
            # 2. Convert token number to (x,y) coordinate using token_decoding
            # 3. Handle missing tokens (sequences of different lengths)
            # 4. Pass coordinates to no_three.try_to_add_points()
            raise NotImplementedError("TODO: Implement token-to-coordinate conversion for 2D")
        
        # Record points added from neural network suggestions
        total_pre_sat += torch.sum(no_three.current_counts.float()).item()
        
        # TODO: Make sure your constraint class implements saturate()
        no_three.saturate()  # Complete constructions greedily
        
        # Record final point counts
        total_post_sat += torch.sum(no_three.current_counts.float()).item()
        
        # Keep only the best constructions (unchanged logic)
        x = torch.argsort(no_three.current_counts, descending=True)
        best_constructions_per_batch = int(args.batch_size * args.keep_best_fraction)
        
        # TODO: Handle extraction of best constructions for 2D case
        # Flow: Get top constructions based on point count ranking
        raise NotImplementedError("TODO: Extract best 2D constructions")
        
        logger.info(f"Batch {batch_idx}: keeping {len(top_constructions)} best constructions")
        
        # Apply symmetries to increase training data
        if args.symmetrize:
            # TODO: Use 2D symmetries instead of 3D
            for construction in top_constructions:
                for symmetric_construction in generate_2d_symmetries(construction):
                    # TODO: Convert 2D construction back to token sequence
                    # Flow:
                    # 1. Find all positions where points are placed
                    # 2. Convert (x,y) positions back to token numbers
                    # 3. Create comma-separated string: "V1,V5,V12"
                    # 4. Add to training data string
                    raise NotImplementedError("TODO: Convert 2D construction to token sequence")
        else:
            for construction in top_constructions:
                # TODO: Convert construction to token sequence without symmetries
                raise NotImplementedError("TODO: Convert construction to tokens")
    
    # Save training data for next generation (unchanged)
    training_path = args.dump_path + f"/training_sets/N{N}_gen{generation}.txt"
    with open(training_path, 'a') as f:
        f.write(out_string)
    
    logger.info(f"Generation {generation-1} -> {generation}")
    logger.info(f"Neural network contributed {total_pre_sat} points")
    logger.info(f"Saturation added {total_post_sat - total_pre_sat} more points")
    logger.info(f"Total points in final constructions: {total_post_sat}")
    logger.info(f"Training data saved to {training_path}")

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F
    import random
    
    # Parse arguments (unchanged)
    parser = get_parser()
    args = parser.parse_args()
    
    # Setup logging (unchanged)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    
    # Create output directories (unchanged)
    os.makedirs(args.dump_path, exist_ok=True)
    os.makedirs(args.dump_path + "/training_sets", exist_ok=True)
    
    # ========================================================================
    # TODO: Create coordinate mapping system for 2D problem
    # ======================================================================== 
    N = args.grid_size
    
    # TODO: Design token ↔ coordinate mapping for 2D grid
    # Flow:
    # 1. We have NxN positions in 2D grid: (0,0), (0,1), ..., (N-1,N-1)
    # 2. Need bijection between positions and token numbers 0, 1, 2, ..., N²-1
    # 3. Example mapping: token_number = x * N + y
    # 4. Reverse mapping: x = token_number // N, y = token_number % N
    # 5. Create token_encoding: (x,y) → token_number
    # 6. Create token_decoding: token_number → (x,y)
    raise NotImplementedError("TODO: Implement 2D coordinate ↔ token mapping system")
    
    # ========================================================================
    # MAIN TRAINING LOOP (mostly unchanged)
    # ========================================================================
    for generation in range(args.initial_gen, args.max_epochs + 1):
        logger.info(f"=== GENERATION {generation} ===")
        
        # Phase 1: Train neural network on current training data
        input_file = args.dump_path + f"/training_sets/N{N}_gen{generation}.txt"
        
        if not os.path.exists(input_file):
            logger.info(f"Creating initial random training data: {input_file}")
            
            # TODO: Create initial random training data using your constraint class
            # Flow:
            # 1. Create constraint solver instance
            # 2. Call saturate() to generate random valid constructions
            # 3. Convert constructions to token sequences
            # 4. Save to file for neural network training
            initial_solver = NoThreeInLine(
                batch_size=args.batch_size, 
                grid_size=N, 
                max_points=args.max_points, 
                device=args.device
            )
            # TODO: Implement the rest of initial data generation
            raise NotImplementedError("TODO: Generate initial 2D training data")
        
        # Create datasets and train neural network (unchanged)
        # TODO: Update token count for 2D problem (N*N instead of N*N*N)
        train_dataset, test_dataset = create_datasets(input_file, force_tokens=N*N)
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Vocabulary size: {train_dataset.vocab_size}")
        
        # Initialize model (unchanged)
        model_config = ModelConfig(
            vocab_size=train_dataset.vocab_size,
            sequence_length=train_dataset.get_output_length(),
            device=args.device,
            type=args.type
        )
        
        if args.type == 'transformer':
            model = Transformer(model_config)
        elif args.type == 'bigram':
            model = Bigram(model_config)
        elif args.type == 'mlp':
            model = MLP(model_config)
        elif args.type == 'rnn':
            model = RNN(model_config)
        elif args.type == 'bow':
            model = BoW(model_config)
        
        model = model.to(args.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Train the model (unchanged)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, device=args.device)
        
        step = 0
        while step < args.max_steps:
            batch = batch_loader.next()
            X, Y = batch
            
            # Forward pass
            logits, loss = model(X, Y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            if step % args.sample_every == 0:
                logger.info(f"Step {step}, Loss: {loss.item():.4f}")
            
            step += 1
        
        logger.info(f"Training completed. Final loss: {loss.item():.4f}")
        
        # Phase 2: Generate new sequences using trained model (unchanged)
        logger.info("Generating new sequences...")
        todo = int(args.target_training_size * 1/args.keep_best_fraction)
        sample_batch_size = min(todo, 1000)
        
        # Clear previous samples
        write_samples(model, train_dataset, num=0, new_file=True)
        
        while sample_batch_size < todo:
            write_samples(model, train_dataset, num=sample_batch_size)
            todo = todo - sample_batch_size
        
        # Phase 3: Test sequences and create next generation training data
        if generation < args.max_epochs:
            # TODO: token_decoding needs to be defined before this call
            decode_and_fix(args, token_decoding=token_decoding, generation=generation+1)
        
        logger.info(f"Generation {generation} completed")
    
    logger.info("All generations completed!")