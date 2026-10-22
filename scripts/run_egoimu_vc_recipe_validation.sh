#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
/usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv,noheader
exec env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/l/anaconda3/envs/SAGE/bin/python scripts/validate_egoimu_vc_recipe.py run
