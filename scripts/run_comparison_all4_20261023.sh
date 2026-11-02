#!/usr/bin/env bash
# Queue the protocol-v3 comparison baselines on BEHAVE, HODome, PAHOI and IMHD
# (single-dataset training and testing, scripts/run_comparison_chain_20261023.sh).
# Re-running resumes unfinished trainings from last.pt and skips finished runs.
#
# WSL does not raise CUDA OOM near the 16 GB limit; every process stalls in
# WDDM paging instead (2026-10-24: five parallel runs froze for 8 h).  A job is
# therefore started only while the used GPU memory stays below a threshold,
# and the next launch waits until the previous one has allocated its memory.
#
# Usage: scripts/run_comparison_all4_20261023.sh [max parallel, default 4] [max used MiB, default 11000]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MAX_JOBS="${1:-4}"
MAX_USED_MIB="${2:-11000}"
SETTLE_S=300
ROOT=outputs/Comparisons/all4_cano_20261023
LOG="$ROOT/queue.log"
mkdir -p "$ROOT"
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

QUEUE=(
    "behave globalpose" "hodome globalpose" "behave transpose" "hodome dip18"
    "hodome transpose" "pahoi tip" "pahoi globalpose" "imhd tip" "imhd globalpose"
    "pahoi dip18" "pahoi transpose" "imhd dip18" "imhd transpose"
    "behave tip" "hodome tip" "behave dip18"
)
done_run() {  # finished runs end their chain.log with the last evaluation
    local ds="$1" method="$2" last
    last=$(tail -n 1 "$ROOT/$ds/$method/chain.log" 2>/dev/null)
    if [ "$method" = dip18 ]; then [[ "$last" == *"(GT root translation)"* ]]; else [[ "$last" == *"full test evaluated" ]]; fi
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1; }

log "queue start (max $MAX_JOBS jobs, launch below $MAX_USED_MIB MiB)"
for item in "${QUEUE[@]}"; do
    read -r ds method <<< "$item"
    if done_run "$ds" "$method"; then
        log "skip $ds $method (finished)"
        continue
    fi
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ] || [ "$(gpu_used)" -ge "$MAX_USED_MIB" ]; do
        sleep 30
    done
    log "start $ds $method (GPU used $(gpu_used) MiB)"
    ( DATASET="$ds" scripts/run_comparison_chain_20261023.sh "$method"; echo "$(date '+%F %T') end $ds $method rc=$?" >> "$ROOT/$ds/$method/queue_end.log" ) &
    sleep "$SETTLE_S"
done
wait
log "queue finished"
