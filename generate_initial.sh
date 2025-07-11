#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 30
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G
#$ -N generate_initial_array
#$ -t 1-10
#$ -tc 10
#$ -o logs/N20/generate_initial_$SGE_TASK_ID.log

# Parse command line arguments


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
echo "Pool capacity: 500"
echo "Number of jobs: 10"

# Set OMP threads for CPU-only execution
export OMP_NUM_THREADS=$NSLOTS

# Run the initial generation with job-specific parameters
echo "Starting initial generation for task $SGE_TASK_ID..."
uv run no_three_in_line/generate_initial_heap.py \          
    --grid_size 20 \
    --pool_capacity 500 \
    --dump_path "training/N20" \
    --num_jobs 10     

echo "Task $SGE_TASK_ID completed at: $(date)"