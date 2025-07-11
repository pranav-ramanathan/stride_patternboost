#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 30
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G
#$ -N aggregate_initial
#$ -o logs/N20/aggregate_initial.log

module load python

cd stride_patternboost

source .venv/bin/activate

uv run no_three_in_line/aggregate_initial.py \
    --grid_size 20 \
    --dump_path "training/N20" \
    --num_jobs 10