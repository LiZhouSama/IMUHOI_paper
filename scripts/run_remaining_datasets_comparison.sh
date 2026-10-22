#!/usr/bin/env bash
# Launch/resume the six formal pipelines in the WSL-native SAGE environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
exec env PYTHONUNBUFFERED=1 /home/l/anaconda3/envs/SAGE/bin/python \
  scripts/run_remaining_datasets_comparison.py "$@"
