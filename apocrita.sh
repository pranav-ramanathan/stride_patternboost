#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 30
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G
#$ -N no_three_in_line_training
#$ -o job_output.log

module load python

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
python no_three_in_line/gw_loop.py \
    --device cpu \
    --grid_size 20 \
    --max_points 40 \
    --target_training_size 2000 \
    --max-steps 2000 \
    --learning-rate 0.00001 \
    --keep_best_fraction 0.1 \
    --max_epochs 20 \
    --num-workers 30 \
    --dump_path "./N20"

echo "Job completed at: $(date)" 