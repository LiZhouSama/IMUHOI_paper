#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv,noheader
exec env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=10 /home/l/anaconda3/envs/SAGE/bin/python scripts/run_behave_paper_aug_20261019.py "${1:-run}"
