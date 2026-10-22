#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv,noheader
exec env PYTHONUNBUFFERED=1 /home/l/anaconda3/envs/SAGE/bin/python scripts/run_omomo_egoimu_split.py "$@"
