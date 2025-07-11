#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 30
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G
#$ -N no_three_in_line_training
#$ -o logs/N20/

module load python

cd stride_patternboost

source .venv/bin/activate

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $JOB_ID"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

# Set OMP threads for CPU-only execution
export OMP_NUM_THREADS=$NSLOTS

# Run the training job
echo "Starting no-three-in-line training..."

# Single line command to avoid line continuation issues
uv run no_three_in_line/gw_train_heap.py --device cpu --grid_size 20 --max_points 40 --target_training_size 500 --max_steps 1000 --num_workers 30 --dump_path "./N20" --n_layer 8 --n_head 8 --n_embd 256 --n_embd2 64 --nn_batch_size 128 --learning_rate 5e-4 --weight_decay 0.05 --temperature 1.2 --top_k 100 --keep_best_fraction 0.05 --max_epochs 1

echo "Job completed at: $(date)" 