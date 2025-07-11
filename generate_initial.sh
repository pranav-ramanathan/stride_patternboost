#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 30
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G
#$ -N generate_initial_array
#$ -t 1-10
#$ -tc 10
#$ -o logs/N20/

cd stride_patternboost

module load python

# Activate virtual environment
source .venv/bin/activate

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $JOB_ID"
echo "Task ID: $SGE_TASK_ID"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Grid size: 20"
echo "Max points: 40"
echo "Target training size: 500"
echo "Job ID: $SGE_TASK_ID"

# Set OMP threads for CPU-only execution
export OMP_NUM_THREADS=$NSLOTS

# Run the initial generation with job-specific parameters
echo "Starting initial generation for task $SGE_TASK_ID..."

# Build command as single line to avoid line continuation issues
uv run no_three_in_line/generate_initial_heap.py --device cpu --grid_size 20 --max_points 40 --target_training_size 500 --keep_best_fraction 0.1 --dump_path "N20" --job_id $SGE_TASK_ID

echo "Task $SGE_TASK_ID completed at: $(date)"