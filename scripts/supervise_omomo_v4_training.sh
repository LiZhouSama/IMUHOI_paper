#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI"
PYTHON="/home/l/anaconda3/envs/SAGE/bin/python"
CONFIG="$PROJECT_ROOT/configs/IMUHOI_train_rnn.yaml"
HP_CKPT="$PROJECT_ROOT/outputs/IMUHOI++/RNN/omomo_baseline_v3_new_hp_20260823/hp/human_pose_08230110_omomo_newhp_20260823/best.pt"
REFERENCE_ROOT="$PROJECT_ROOT/outputs/IMUHOI++/RNN/omomo_baseline_v3_new_hp_20260823"
OUTPUT_ROOT="${IMUHOI_V4_OUTPUT_ROOT:-$PROJECT_ROOT/outputs/IMUHOI++/RNN/omomo_v4_pose_prior_20260823}"
GPU="${IMUHOI_V4_GPU:-0}"
NUM_WORKERS="${IMUHOI_V4_NUM_WORKERS:-4}"
CHECK_INTERVAL="${IMUHOI_V4_CHECK_INTERVAL:-300}"
SUPERVISOR_LOG="$OUTPUT_ROOT/supervisor.log"
LOCK_FILE="$OUTPUT_ROOT/supervisor.lock"
PID_FILE="$OUTPUT_ROOT/supervisor.pid"

mkdir -p "$OUTPUT_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf '[%s] another v4 supervisor is already active\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    exit 0
fi

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SUPERVISOR_LOG"
}

if [[ ! -f "$HP_CKPT" ]]; then
    log "missing HumanPose checkpoint: $HP_CKPT"
    exit 1
fi
if [[ ! -f "$PYTHON" ]]; then
    log "missing SAGE Python: $PYTHON"
    exit 1
fi

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp

log "checking GPU before starting v4 pipeline"
nvidia-smi | tee -a "$SUPERVISOR_LOG"
log "starting VC v4 -> OT -> joint_train -> eval_batch_online"

"$PYTHON" scripts/run_velocity_contact_ablation_experiments.py \
    --config "$CONFIG" \
    --hp_ckpt "$HP_CKPT" \
    --output_root "$OUTPUT_ROOT" \
    --only 08_v4_supervised_pose_prior \
    --gpu "$GPU" \
    --num_workers "$NUM_WORKERS" \
    --eval_online_lookahead 5 \
    > >(tee -a "$SUPERVISOR_LOG") 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$PID_FILE"
log "training PID: $TRAIN_PID"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    log "training is still running; recording GPU and manifest status"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader | tee -a "$SUPERVISOR_LOG"
    if [[ -f "$OUTPUT_ROOT/manifest.json" ]]; then
        tail -n 24 "$OUTPUT_ROOT/manifest.json" | tee -a "$SUPERVISOR_LOG"
    fi
    sleep "$CHECK_INTERVAL"
done

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
rm -f "$PID_FILE"
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    log "v4 pipeline failed with exit code $TRAIN_STATUS"
    exit "$TRAIN_STATUS"
fi

log "checking GPU before comparison"
nvidia-smi | tee -a "$SUPERVISOR_LOG"
log "building baseline/v3/v4 comparison"
"$PYTHON" scripts/compare_omomo_v4_results.py \
    --reference_root "$REFERENCE_ROOT" \
    --v4_root "$OUTPUT_ROOT" \
    | tee -a "$SUPERVISOR_LOG"
log "v4 pipeline and comparison completed"