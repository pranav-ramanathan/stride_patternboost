# PatternBoost: Constructions in Mathematics with a Little Help from AI

This repository contains the code for **PatternBoost**, an algorithm that alternates between local search and transformer-based global pattern learning to find new constructions in mathematics, particularly in extremal combinatorics. 

## Overview
PatternBoost consists of two iterative phases:
1. **Local phase**: A classical search algorithm optimizes mathematical constructions.
2. **Global phase**: A transformer neural network is trained on the best constructions from the local phase, generating new seeds for the next iteration.

The project’s goal is to provide mathematicians with an accessible tool that balances simplicity and performance without requiring deep machine learning expertise.

## Installation

### Prerequisites
- Python 3.8+
- Julia 1.6+

### Setup
1. Clone the repository:
```
git clone https://github.com/zawagner22/transformers_math_experiments.git
```
2. Navigate to the directory and install dependencies for python and julia

## Usage
1. Pick a problem from 'search_fc.jl'. You can create new problems based on the examples provided.
2. Configure your parameters in 'fc_loop.py'.
3. Run PatternBoost:
```
python fc_loop.py
```

## Contributing
Feel free to explore other problems or propose extensions to the PatternBoost algorithm!