#!/usr/bin/env bash
# Finalize the 2x2 table only after every dataset/crossed cell finishes testing.
set -euo pipefail
cd /mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
R=outputs/IMUHOI++/RNN_cano
runs=(
  vc_ot_cross_ablation_omomo_v4_vc_imuhoi_ot_4heads
  vc_ot_cross_ablation_omomo_imuhoi_v2_armrot_biclose
  vc_ot_cross_ablation_behave_v4_vc_imuhoi_ot_4heads
  vc_ot_cross_ablation_behave_imuhoi_v2_armrot_biclose
)
while true; do
  complete=0
  for run in "${runs[@]}"; do
    manifest="$R/$run/experiment_manifest.json"
    if [[ -f "$manifest" ]] && grep -q '"status": "failed"' "$manifest"; then
      echo "Detected a failed run in $manifest; fix/resume it before finalizing."
      exit 1
    fi
    [[ -f "$R/$run/summary.md" ]] && complete=$((complete + 1))
  done
  if [[ "$complete" -eq "${#runs[@]}" ]]; then
    nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu --format=csv,noheader
    /home/l/anaconda3/envs/SAGE/bin/python scripts/summarize_vc_ot_cross_ablation.py
    exit 0
  fi
  sleep 60
done