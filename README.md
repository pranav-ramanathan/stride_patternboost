# PatternBoost: Constructions in Mathematics with a Little Help from AI

This repository contains the code for **PatternBoost**, an algorithm that alternates between local search and transformer-based global pattern learning to find new constructions in mathematics, particularly in extremal combinatorics. 

## Overview
PatternBoost consists of two iterative phases:
1. **Local phase**: A classical search algorithm optimizes mathematical constructions.
2. **Global phase**: A transformer neural network is trained on the best constructions from the local phase, generating new seeds for the next iteration.

The project's goal is to provide mathematicians with an accessible tool that balances simplicity and performance without requiring deep machine learning expertise.

## Installation

### Prerequisites
- Python 3.12+
- Julia 1.8+

### Setup
1. Clone the repository:
```
git clone https://github.com/zawagner22/transformers_math_experiments.git
```
2. Navigate to the directory and install the necessary libraries for python and julia

## Usage
1. Pick a problem from `search_fc.jl`. You can create new problems based on the examples provided.
2. Configure your parameters in `fc_loop.py`.
3. Run PatternBoost:
```
python fc_loop.py
```

## Contributing
Feel free to explore other problems or propose extensions to the PatternBoost algorithm!

## Fixed-timeout run results

The four-seed fixed-timeout results used in the manuscript are committed under
[results/fixed_timeout_seed0_3/](results/fixed_timeout_seed0_3/). The protocol
covers grid sizes n=10..19, seeds 0..3, and a 120-minute timeout per
method/grid/seed run.

For the PatternBoost results:

- raw/patternboost_seed*_part*.csv contains the source summaries. Seed 0 has
  two source parts.
- aggregate.csv contains the cross-method summary; inspect the pb_* columns
  for this repository.
- long.csv contains one normalized record per method, seed, and grid size.
- manifest.json records source paths and SHA-256 hashes.
- validation.md records the validation checks and known timeout anomaly.

For example:

~~~bash
column -s, -t < results/fixed_timeout_seed0_3/aggregate.csv
less results/fixed_timeout_seed0_3/raw/patternboost_seed1_part1.csv
~~~

## Apple Silicon Support

This codebase has been adapted to work on Apple Silicon (M1/M2/M3) Macs using Metal Performance Shaders (MPS). 

### Device Auto-Detection

The code automatically detects the best available device:
1. **MPS** (Metal Performance Shaders) - Apple Silicon GPU
2. **CUDA** - NVIDIA GPU  
3. **CPU** - Fallback option

### Usage on Apple Silicon

#### Basic Usage
```bash
# Auto-detect device (will use MPS on Apple Silicon)
python no_spheres/gw_loop.py --grid_size 6 --dump_path ./experiment_output

# Explicitly use MPS
python no_spheres/gw_loop.py --device mps --grid_size 6 --dump_path ./experiment_output

# Force CPU usage
python no_spheres/gw_loop.py --cpu --grid_size 6 --dump_path ./experiment_output
```

#### Prerequisites for Apple Silicon

1. **Install PyTorch with MPS support**:
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Verify MPS availability**:
   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   print(f"MPS built: {torch.backends.mps.is_built()}")
   ```

### Performance Notes

- **Memory**: Apple Silicon unified memory is shared between CPU and GPU
- **Batch Size**: You may be able to use larger batch sizes than CUDA due to unified memory
- **Speed**: MPS provides significant speedup over CPU for neural network operations

### Key Changes for Apple Silicon

1. **Device Detection**: Automatic MPS detection and fallback
2. **Memory Monitoring**: Device-agnostic memory reporting
3. **Synchronization**: Proper MPS synchronization support
4. **Random Seeds**: MPS-specific random seed initialization

The code maintains full backward compatibility with CUDA and CPU devices.
