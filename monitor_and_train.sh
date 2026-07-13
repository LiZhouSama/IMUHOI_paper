#!/bin/bash

# Monitor training process and start next training when done
WATCH_PID=1437469
SCRIPT_DIR="/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI"
CONDA_BASE="/home/l/anaconda3"

# Check if process is still running
if ps -p $WATCH_PID > /dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Process $WATCH_PID is still running"
    exit 0
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Process $WATCH_PID has finished. Starting next training..."

    # Change to working directory
    cd "$SCRIPT_DIR"

    # Activate conda environment with explicit path
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate SAGE

    # Verify environment is active
    echo "Using Python: $(which python)"
    echo "Current directory: $(pwd)"

    python train/rnn/train_object_trans.py \
        --vc_ckpt outputs/IMUHOI_RNN_2/velocity_contact_vc_boundary_zero_06211816 \
        --joint_train \
        --ablate_vc_boundary \
        --batch_size 40 \
        --epochs 100 \
        --lr 0.0002

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Next training completed or failed"
fi
