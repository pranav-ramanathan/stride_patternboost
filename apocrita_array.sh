#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 30
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G
#$ -N generate_initial_array
#$ -t 1-5
#$ -tc 5
#$ -o logs/

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

# Set OMP threads for CPU-only execution
export OMP_NUM_THREADS=$NSLOTS

# Run the initial generation with job-specific parameters
# Total target: 2000, split across 5 jobs = 400 per job
echo "Starting initial generation for task $SGE_TASK_ID..."
uv run no_three_in_line/generate_initial.py \
    --grid_size 20 \
    --target_training_size 400 \
    --batch_size 1000 \
    --dump_path "./N20" \
    --job_id "$SGE_TASK_ID" \
    --device cpu

echo "Task $SGE_TASK_ID completed at: $(date)"