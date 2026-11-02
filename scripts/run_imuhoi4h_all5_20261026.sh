#!/usr/bin/env bash
# Four-branch IMUHOI (ObjectTrans imuhoi_4heads) on the five datasets.
#
# Each dataset reuses the HumanPose and VC-IMUHOI checkpoints of its
# three-branch IMUHOI run (all5_v5ar_vs_imuhoi3b_20261012 sources) and the same
# recipe / shared overrides / step-scaled schedule; only ObjectTrans + the
# joint phase are retrained (variant vc_imuhoi_ot_4heads of
# run_behave_full_pipeline_trust.py).  Evaluation = the runner's online eval
# (Noitom noise seed 42, 7/3 protocol, per-joint rotation errors).
#
# WSL stalls instead of raising OOM near 16 GB: a job starts only while the used
# GPU memory is below MAX_USED_MIB, and the next launch waits SETTLE_S seconds.
#
# Usage: scripts/run_imuhoi4h_all5_20261026.sh [max parallel, default 2] [max used MiB, default 9000]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MAX_JOBS="${1:-2}"
MAX_USED_MIB="${2:-9000}"
SETTLE_S=240
PY=/home/l/anaconda3/envs/SAGE/bin/python
R=outputs/IMUHOI++/RNN_cano
OUT=$R/imuhoi4h_all5_20261026
LOG="$OUT/queue.log"
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1; }

# dataset -> three-branch experiment root (source of config, overrides, HP and VC)
declare -A SRC=(
    [imhd]=$R/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b
    [hodome]=$R/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b
    [pahoi]=$R/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b
    [omomo]=$R/imuhoi3b_20261012_omomo_unified
    [behave]=$R/imuhoi3b_20261008_behave
)
ORDER=(imhd hodome pahoi omomo behave)  # longest OT trainings first

launch() {
    local ds="$1" src="${SRC[$1]}"
    # Prints: source config, HP ckpt, VC ckpt, then one --set KEY=VALUE per shared override.
    mapfile -t spec < <("$PY" - "$src/experiment_manifest.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["source_config"])
print(m["stages"]["human_pose"]["checkpoint"])
print(m["stages"]["velocity_contact"]["checkpoint"])
for key, value in m["shared_overrides"].items():
    print(f"{key}={json.dumps(value)}")
EOF
)
    local sets=()
    for item in "${spec[@]:3}"; do sets+=(--set "$item"); done
    "$PY" scripts/run_behave_full_pipeline_trust.py --cfg "${spec[0]}" --experiment_root "$OUT/$ds" \
        --dataset "$ds" --vc_arch imuhoi --variants vc_imuhoi_ot_4heads --run_tag imuhoi4h_20261026 \
        --seed 42 --num_workers 2 --eval_batch_size 48 --max_parallel 1 --schedule_scale auto \
        --skip_hp "${spec[1]}" --skip_vc "${spec[2]}" "${sets[@]}" --resume --check_gpu_before_stage
}

log "queue start (max $MAX_JOBS jobs, launch below $MAX_USED_MIB MiB)"
for ds in "${ORDER[@]}"; do
    if grep -q '"status": "complete"' "$OUT/$ds/experiment_manifest.json" 2>/dev/null; then
        log "skip $ds (complete)"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ] || [ "$(gpu_used)" -ge "$MAX_USED_MIB" ]; do
        sleep 30
    done
    log "start $ds (GPU used $(gpu_used) MiB)"
    ( launch "$ds" > "$OUT/$ds.log" 2>&1; log "end $ds rc=$?" ) &
    sleep "$SETTLE_S"
done
wait
log "queue finished"
