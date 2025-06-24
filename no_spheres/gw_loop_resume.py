import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List
import pprint
import logging
from collections import Counter
import itertools
import shutil

# from utils import bool_flag, initialize_exp

from makemoretokens import ModelConfig, CharDataset, Transformer, Bigram, MLP, RNN, BoW, InfiniteDataLoader, evaluate, generate, print_samples # type: ignore

import numpy as np
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger

from no_sphere_simple2 import NoSphereSimple

logger=getLogger()

def get_parser():
    parser = argparse.ArgumentParser('PatternBoost for no_spheres - with resume capability')

    # Resume training parameters
    parser.add_argument('--resume_model_path', type=str, default=None, help='Path to the model checkpoint to resume from')
    parser.add_argument('--resume_generation', type=int, default=None, help='Generation number to resume from (if not specified, will auto-detect)')
    parser.add_argument('--resume_training_data_path', type=str, default=None, help='Path to the training data directory to resume from (if different from dump_path)')

    # local search parameters
    parser.add_argument('--grid_size',type=int,default=6,help='Grid size for no_spheres')
    parser.add_argument('--batch_size', type=int, default=500, help='Generate and process samples in batches of this size (reduce if GPU crashes with OOM)')
    parser.add_argument('--max_points',type=int, default=18, help='max points which can be added to construction')
    parser.add_argument('--target_training_size', type=int, default=20000, help='number of examples to aim for (before symmetrization)')
    parser.add_argument('--keep_best_fraction',type=float,default=0.1,help='Percentage of good constructions to keep')
    parser.add_argument('--symmetrize', default=True, action=argparse.BooleanOptionalAction,help='symmetrize constructions, set to --no-symmetrize to disable')

    # Makemore params
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
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.1, help="weight decay")

#    parser.add_argument('--max-output-length', type=int, default=120, help="maximum output length")
    parser.add_argument('--gen_batch_size', type=int, default=10, help="batch size for generation from transformer (reduce if GPU crashes with OOM when sampling)")
#    parser.add_argument('--n_samples_from_transformer',type=int,default=500,help='number to sample from transformer during each epoch')
    parser.add_argument('--temperature', type=float, default=1.0, help="temperature")
    
    # path and ports
    parser.add_argument("--dump_path", type=str, default="dump_path", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",help="Experiment ID")
    parser.add_argument("--local_rank", type=int, default=-1,help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,help="Master port (for multi-node SLURM jobs)")

    parser.add_argument("--cpu", default=False, action=argparse.BooleanOptionalAction,help="run on cpu only")
    parser.add_argument("--device", type=str, default="auto", help="device to use for compute: auto|cpu|cuda|mps")
# debug
    parser.add_argument("--debug_slurm", default=True, action=argparse.BooleanOptionalAction,help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", default=True, action=argparse.BooleanOptionalAction,help="Enable all debug flags")

#    parser.add_argument("--cpu", type=bool_flag, default="false",help="run on cpu only")
# debug
#    parser.add_argument("--debug_slurm", type=bool_flag, default=False,help="Debug multi-GPU / multi-node within a SLURM job")
#    parser.add_argument("--debug", help="Enable all debug flags",action="store_true")

    return parser


def create_datasets(input_file,force_tokens=-1):
    """Set up datasets from a .txt file consistin of tokens like V0,V1,...
       When we set force_tokens we fix the number of tokens which should occur.
       For example, if the data set looks like
       V0,V0
       V2,V2
       without force_tokens our chars would be 'V0' and 'V2'.
       With force_tokens=4 it would be 'V0', 'V1', 'V2', 'V3'.
       """

    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    words = [w.split(",") for w in words]

    # maybe a tad hacky: we sort our dataset so that it is ordered V1, V2, .... V10, V11 ....
    chars = sorted(list(set([i for word in words for i in word])), key=lambda x: int(x[1:]))

    if force_tokens >= 0:
        forced_chars = ['V'+str(i) for i in range(force_tokens)]
        assert set(chars).issubset(set(forced_chars)), f"It looks like {force_tokens=} is too small."

    max_word_length = max(len(w) for w in words)
    logger.info(f"number of examples in the dataset: {len(words)}")
    logger.info(f"max word length: {max_word_length}")
    logger.info(f"number of unique characters in the vocabulary: {len(chars)}")
#    logger.info("vocabulary:")
#    logger.info(chars)
    assert max_word_length <= args.max_points, f'block size too large {max_word_length} vs {args.max_output_length}'
        
    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples

    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    logger.info(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")
    
    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, args.max_points)
    test_dataset = CharDataset(test_words, chars, args.max_points)

    return train_dataset, test_dataset


def write_samples(num=10, new_file=False, use_logger=False):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - 1 # -1 because we already start with <START> token (index 0)
    X_samp = generate(model, X_init, steps, temperature = args.temperature, top_k=top_k, do_sample=True).to('cpu')
    #logger.info(f"generated")
    n_samp =0
    max_samp=0
    sum_samp=0
    samples = []
#    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        samples.append(word_samp)
    for s in samples:
        n_samp +=1
        sum_samp += len(s)
        max_samp = max(max_samp, len(s))
    out_file = args.dump_path + "/out.txt"
    #if use_logger:
        #logger.info("decoded")
        # logger.info(f"Printing {len(samples)} samples to {out_file}.")
    #else: 
        # print(f"Printing {len(samples)} samples to {out_file}.")
    if not new_file:
        with open(out_file, "a") as file:
            for word in samples:
                file.write(word)
                file.write("\n")
    else:
        with open(out_file, "w") as file:
            for word in samples:
                file.write(word)
                file.write("\n")
    #logger.info("printed")
    return n_samp, sum_samp, max_samp

def permute_and_flip_generator(X):
    permutations = list(itertools.permutations([1, 2, 3]))
    signs_combinations = list(itertools.product([1, -1], repeat=3))
    for p in permutations:
        for signs in signs_combinations:
            # Permute the axes according to p
            permuted_X = X.permute(0, p[0], p[1], p[2])
            # Flip the axes where the sign is -1
            for i, sign in enumerate(signs):
                if sign == -1:
                    permuted_X = torch.flip(permuted_X, dims=[i + 1])
            yield permuted_X

def decode_and_fix(args,token_decoding,generation):
    """read samples from transformer, and use as seeds for local search"""


    with open(args.dump_path + '/out.txt', 'r') as file:
        # Initialize an empty list to store the lists
        sampled_tokens = []

        for line in file:
            # Strip any whitespace characters (including newlines) from the ends
            line = line.strip()
            if line: # make sure line is non-empty
                # Split the line by commas and remove the 'V' prefix, then convert to integers
                numbers = [int(item[1:]) for item in line.split(',')]
                # Append the list of integers to the list of lists
                sampled_tokens.append(numbers)
    
    logger.info(f"{len(sampled_tokens)} samples decoded.")
    logger.info("first few sampled sequences: %s", sampled_tokens[0:3])
    logger.info(f"average length of sampled sequences: {torch.mean(torch.tensor([len(x) for x in sampled_tokens],dtype=torch.float)).item():.2f}")

    N = args.grid_size

    training_path_root = args.dump_path + f'/training_sets/N{N}_gen{generation}'
    training_path = training_path_root + '.txt'

    # it is good to have a record of the samples processed in each round
    shutil.copy(args.dump_path + '/out.txt', training_path_root + '_unprocessed.txt')

    total_pre_sat = 0
    total_post_sat = 0
    best_score = 0
    all_counts_pre = []
    all_counts_post = []
    t0 = time.time()

    times_adding_points = []
    times_saturating = []
    for b in range(0,len(sampled_tokens),args.batch_size):
        
        t_batch_start = time.time()

        cur_batch_size = min(args.batch_size,len(sampled_tokens)-b)
        no_sphere = NoSphereSimple(batch_size=cur_batch_size,grid_size=N,max_points=args.max_points,device=args.device)
        current_batch = sampled_tokens[b:b+cur_batch_size]
        max_length = max(len(seq) for seq in current_batch)
        for i in range(max_length):
            x = -1 * torch.ones((cur_batch_size,3),dtype=torch.int8,device=args.device)
            for j in range(cur_batch_size):
                if i < len(current_batch[j]):
                    x[j] = token_decoding[current_batch[j][i]]
            no_sphere.try_to_add_points(x)

        t_batch_mid = time.time()
        times_adding_points.append(t_batch_mid-t_batch_start)

        total_pre_sat += torch.sum(no_sphere.current_counts.float()).item()
        all_counts_pre += no_sphere.current_counts.tolist()

        no_sphere.saturate()

        times_saturating.append(time.time()-t_batch_mid)

        total_post_sat += torch.sum(no_sphere.current_counts.float()).item()
        all_counts_post += no_sphere.current_counts.tolist()
        best_score = max(best_score, torch.max(no_sphere.current_counts.float()).item())

#        logger.info("Processed a batch: mean before / after saturation, max:")
#        logger.info(f" {mean_pre_saturation:.3f}, {torch.mean(no_sphere.current_counts.float()):.3f}, {torch.max(no_sphere.current_counts)}")

        x = torch.argsort(no_sphere.current_counts,descending=True)

        best_constructions_per_batch = int(args.batch_size * args.keep_best_fraction)

        top_constructions = ((no_sphere.current_constructions[x[0:best_constructions_per_batch]]==1)*1).cpu()

        all_counts_post += no_sphere.current_counts.tolist()

        out_string = ''
        if args.symmetrize:
            for permuted_constructions in permute_and_flip_generator(top_constructions):
                for construction in permuted_constructions:
                    indices = torch.nonzero(construction)
                    encoding = token_encoding[indices[:,0],indices[:,1],indices[:,2]]
                    encoding_string = ','.join([f'V{tok}' for tok in encoding]) + '\n'
                    out_string += encoding_string
            out_string_unpermuted = ''
            for construction in top_constructions:
                indices = torch.nonzero(construction)
                encoding = token_encoding[indices[:,0],indices[:,1],indices[:,2]]
                encoding_string = ','.join([f'V{tok}' for tok in encoding]) + '\n'
                out_string_unpermuted += encoding_string
            
        else:
            for construction in top_constructions:
                indices = torch.nonzero(construction)
                encoding = token_encoding[indices[:,0],indices[:,1],indices[:,2]]
                encoding_string = ','.join([f'V{tok}' for tok in encoding]) + '\n'
                out_string += encoding_string

        with open(training_path,'a') as f:
        #    print(f"Writing {top_constructions.shape[0]} constructions to {training_path}.")
            f.write(out_string)

        if args.symmetrize:
            with open(training_path_root + '_unpermuted.txt','a') as f:
                f.write(out_string_unpermuted)
#        torch.cuda.empty_cache()

    pre_saturation_average = total_pre_sat/len(sampled_tokens)
    post_saturation_average = total_post_sat/len(sampled_tokens)
    logger.info(f"finished decoding and fixing")
    logger.info(f"time taken = {time.time()-t0:.2f} seconds")
    
    adding_average_time = sum(times_adding_points)/len(times_adding_points)
    saturating_average_time = sum(times_saturating)/len(times_saturating)

    logger.info(f"{adding_average_time=:.2f},{saturating_average_time=:.2f} (seconds)")
    logger.info(f"{pre_saturation_average=:.2f},{post_saturation_average=:.2f}, {best_score=}")
    logger.info(f"score distribution before saturation: {Counter(all_counts_pre)}")
    logger.info(f"score distribution after saturation: {Counter(all_counts_post)}")    

    # it is good to have a record of the samples processed in each round
    shutil.os.remove(args.dump_path + '/out.txt')


def detect_resume_generation(training_data_path, grid_size):
    """Auto-detect the latest generation available for resuming"""
    max_gen = -1
    # Look for any generation files, not necessarily continuous from 0
    for i in range(100):  # reasonable upper bound
        if os.path.isfile(os.path.join(training_data_path, f"training_sets/N{grid_size}_gen{i}.txt")):
            max_gen = i
    return max_gen


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Validate resume arguments
    if args.resume_model_path is not None:
        if not os.path.isfile(args.resume_model_path):
            raise ValueError(f"Resume model path does not exist: {args.resume_model_path}")
        logger.info(f"Resuming from model: {args.resume_model_path}")
    
    log_prefix = args.dump_path + "/"

    if not os.path.exists(log_prefix):
        os.makedirs(log_prefix)
    training_dir = log_prefix + 'training_sets'
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    # Configure logging with both console and file output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a file handler
    fh = logging.FileHandler(log_prefix + 'program-exp.log')
    fh.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Add both handlers to the logger
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

    N = args.grid_size

    token_encoding = torch.arange(N**3).view((N,N,N))
    # token decoder
    a_indices, b_indices, c_indices = torch.meshgrid(torch.arange(N), torch.arange(N), torch.arange(N), indexing='ij')
    token_decoding = torch.stack([a_indices, b_indices, c_indices], dim=-1).view(-1, 3)

    # Determine the generation to start from
    if args.resume_model_path is not None:
        # Resume mode
        training_data_path = args.resume_training_data_path if args.resume_training_data_path else args.dump_path
        
        if args.resume_generation is not None:
            initial_gen = args.resume_generation
            logger.info(f"Resuming from explicitly specified generation: {initial_gen}")
        else:
            # Auto-detect the latest generation
            initial_gen = detect_resume_generation(training_data_path, N)
            if initial_gen == -1:
                raise ValueError(f"No training data found in {training_data_path}/training_sets/ for grid size {N}")
            logger.info(f"Auto-detected resuming from generation: {initial_gen}")
        
        # Check if training data exists
        resume_training_file = os.path.join(training_data_path, f"training_sets/N{N}_gen{initial_gen}.txt")
        if not os.path.isfile(resume_training_file):
            raise ValueError(f"Training data not found: {resume_training_file}")
        
        logger.info(f"Using training data from: {resume_training_file}")
        
    else:
        # Standard mode - find existing generation or start from 0
        for i in range(args.max_epochs):
            if not os.path.isfile(log_prefix+f"training_sets/N{N}_gen{i}.txt"):
                break
        initial_gen = i
        
        if initial_gen > 0:
            initial_gen = initial_gen - 1 # first index for which we have training data
        else:
            # Generate 0th generation training data
            logger.info("Generating 0th generation of training data...")

            training_path_root = args.dump_path + f'/training_sets/N{N}_gen{initial_gen}'
            training_path = training_path_root + '.txt'

            constructions_log = []
            best_constructions_per_batch = int(args.batch_size * args.keep_best_fraction)
            
            t0 = time.time()

            for _ in range(int(args.target_training_size/best_constructions_per_batch)):
                # generate some constructions
                no_sphere = NoSphereSimple(batch_size=args.batch_size,grid_size=args.grid_size,max_points=args.max_points,device=args.device)
                no_sphere.saturate()

                # sort according to number of points
                x = torch.argsort(no_sphere.current_counts,descending=True)
                top_constructions = ((no_sphere.current_constructions[x[0:best_constructions_per_batch]]==1)*1).cpu()

                constructions_log += no_sphere.current_counts.int().tolist()

                out_string = ''
                if args.symmetrize:
                    for permuted_constructions in permute_and_flip_generator(top_constructions):
                        for construction in permuted_constructions:
                            indices = torch.nonzero(construction)
                            encoding = token_encoding[indices[:,0],indices[:,1],indices[:,2]]
                            encoding_string = ','.join([f'V{tok}' for tok in encoding]) + '\n'
                            out_string += encoding_string
                    out_string_unpermuted = ''
                    for construction in top_constructions:
                        indices = torch.nonzero(construction)
                        encoding = token_encoding[indices[:,0],indices[:,1],indices[:,2]]
                        encoding_string = ','.join([f'V{tok}' for tok in encoding]) + '\n'
                        out_string_unpermuted += encoding_string
                    
                else:
                    for construction in top_constructions:
                        indices = torch.nonzero(construction)
                        encoding = token_encoding[indices[:,0],indices[:,1],indices[:,2]]
                        encoding_string = ','.join([f'V{tok}' for tok in encoding]) + '\n'
                        out_string += encoding_string

                with open(training_path,'a') as f:
                #    print(f"Writing {top_constructions.shape[0]} constructions to {training_path}.")
                    f.write(out_string)

                if args.symmetrize:
                    with open(training_path_root + '_unpermuted.txt','a') as f:
                        f.write(out_string_unpermuted)

        #        torch.cuda.empty_cache()
            if args.device == "cuda":
                logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
            elif args.device == "mps":
                logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB")
            logger.info(f"Generated {len(constructions_log)} constructions.")
            logger.info(f"Generation took {time.time()-t0:.2f} seconds.")
            logger.info(f"Distribution of counts = {Counter(constructions_log)}")

    # Now we should have training data available
    input_file = (args.resume_training_data_path if args.resume_training_data_path else args.dump_path) + f"/training_sets/N{N}_gen{initial_gen}.txt"
    assert os.path.isfile(input_file), f"Training file not found: {input_file}"

    logger.info(f"initializing at generation: {initial_gen}")
    train_dataset, test_dataset = create_datasets(input_file,force_tokens=N**3)
    vocab_size = train_dataset.get_vocab_size()
    block_size = args.max_points + 1
    logger.info(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
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
    
    # Load the model
    if args.resume_model_path is not None:
        logger.info(f"Loading model from: {args.resume_model_path}")
        model.load_state_dict(torch.load(args.resume_model_path, map_location=args.device))
    else:
        # Check for model in current dump_path
        model_path = os.path.join(args.dump_path, "model.pt")
        if os.path.isfile(model_path):
            logger.info(f"Loading existing model from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=args.device))

    # When resuming, we want the output to start from generation 0, regardless of the resume generation
    resume_offset = initial_gen if args.resume_model_path is not None else 0
    
    for generation_idx in range(args.max_epochs + 1):
        actual_generation = initial_gen + generation_idx if args.resume_model_path is not None else generation_idx
        output_generation = generation_idx  # Always start output numbering from 0
        
        logger.info(f"============ Start of generation {output_generation} ============")
        if args.resume_model_path is not None and generation_idx == 0:
            logger.info(f"Resuming from original generation {actual_generation}")
        
        if args.device == "cuda":
            logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB")

        # For the initial generation when resuming, use the resume training data path
        if generation_idx == 0 and args.resume_training_data_path is not None:
            input_file = args.resume_training_data_path + f"/training_sets/N{N}_gen{actual_generation}.txt"
        else:
            input_file = args.dump_path + f"/training_sets/N{N}_gen{output_generation}.txt"
        train_dataset, test_dataset = create_datasets(input_file,force_tokens=N**3)

        logger.info(f"training on {input_file}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

        # init dataloader
        batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

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

            # wait for all GPU work to finish then calculate iteration time taken
            if args.device == "cuda":
                torch.cuda.synchronize()
            elif args.device == "mps":
                torch.mps.synchronize()
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
#                print_samples(num=10)
                    
            step += 1
            # termination conditions
            if args.max_steps >= 0 and step >= args.max_steps:
                break
        logger.info(f"training took {time.time()-t_training:.2f} seconds")
        if args.device == "cuda":
            logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB")

        t_generating = time.time()
        logger.info('generating')
        sample_batch_size =args.gen_batch_size # reduce this if GPU crashes, increase it if sampling is slow
        todo = int(args.target_training_size * 1/args.keep_best_fraction)
        tot_n = 0
        tot_sum = 0
        tot_max = 0

        while sample_batch_size < todo:
            n, sm, mx = write_samples(num=sample_batch_size)
            tot_n+=n
            tot_sum+=sm
            tot_max = max(tot_max,mx)
            todo = todo - sample_batch_size
        n, sm, mx = write_samples(num=todo)
        tot_n+=n
        tot_sum+=sm
        tot_max = max(tot_max,mx)
        logger.info(f"distribution of sample length strings: average: {tot_sum/tot_n if tot_n != 0 else 0} max: {tot_max}")
        logger.info(f"generation took {time.time()-t_generating:.2f} seconds")
        if args.device == "cuda":
            logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB")

        logger.info('decoding and fixing')
        
        # Clear GPU memory before decoding phase
        if args.device == "mps":
            torch.mps.empty_cache()
        elif args.device == "cuda":
            torch.cuda.empty_cache()
            
        decode_and_fix(args,token_decoding=token_decoding,generation=output_generation+1)
        if args.device == "cuda":
            logger.info(f"Memory allocated:  {torch.cuda.memory_allocated(0)/(1024*1024):.2f}MB, reserved: {torch.cuda.memory_reserved(0)/(1024*1024):.2f}MB")
        elif args.device == "mps":
            logger.info(f"Memory allocated:  {torch.mps.current_allocated_memory()/(1024*1024):.2f}MB")
        logger.info(f"============ End of generation {output_generation} ============") 