#!/usr/bin/env bash
# Train one comparison baseline on one canonical dataset, then evaluate it online.
#
# Protocol v3 "paper" (2026-10-23): each method follows its paper/released
# recipe, but all of them read the main pipeline's stored IMU stream with its
# fixed per-sequence Noitom noise (seed 42).  OMOMO uses the canonical random
# split chosen by the user (train 4705, 5% held out for checkpoint selection,
# test 1173).  The other datasets (DATASET=behave|imhd|pahoi|hodome) train on
# the same segments as the main pipeline and are tested on its full-sequence
# test sets; their epoch budgets are scaled so every run takes the optimizer
# steps of the OMOMO run (--schedule_scale auto).  Contact-palm HOI metric.
# Every evaluation uses the method's online runner (DIP/TransPose 20 past + 5
# future frames, TIP 40-frame autoregression with its 11-frame acceleration
# filter, GlobalPose causal recurrent state followed by the released
# per-frame physics optimizer).  DIP has no translation estimator and is also
# evaluated with the ground-truth root translation.
#
# Usage: [DATASET=behave] scripts/run_comparison_chain_20261023.sh <method> [extra train args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

METHOD="$1"
shift
DATASET="${DATASET:-omomo}"
if [ "$DATASET" = omomo ]; then
    ROOT="${COMPARISON_ROOT:-outputs/Comparisons/omomo_cano_20261023}"
    dataset_args=()
else
    ROOT="${COMPARISON_ROOT:-outputs/Comparisons/all4_cano_20261023/$DATASET}"
    dataset_args=(--dataset "$DATASET" --schedule_scale auto)
fi
PY=/home/l/anaconda3/envs/SAGE/bin/python
RUN="$ROOT/$METHOD"
mkdir -p "$RUN"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

stamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }
echo "$(stamp) start $METHOD $DATASET $*" >> "$RUN/chain.log"

if [ ! -f "$RUN/train.done" ]; then
    resume=()
    if [ -f "$RUN/last.pt" ]; then
        resume=(--resume "$RUN/last.pt")
    fi
    "$PY" -m Comparisons.train_comparison \
        --method "$METHOD" \
        --cfg configs/IMUHOI_train_rnn.yaml \
        --save_dir "$ROOT" \
        --run_name "$METHOD" \
        --protocol paper \
        --num_workers 4 \
        --no-pin_memory \
        "${dataset_args[@]}" \
        "${resume[@]}" \
        "$@" >> "$RUN/train.log" 2>&1
    touch "$RUN/train.done"
    echo "$(stamp) trained" >> "$RUN/chain.log"
fi

# selected.pt holds the method's published model-selection rule (DIP: early
# stopping; TIP/GlobalPose: final state of the fixed schedule; TransPose:
# per-network validation minimum), see PAPER_TRAINING_DEFAULTS.
CKPT="$RUN/selected.pt"
EVAL=("$PY" -m Comparisons.eval_comparison
    --method "$METHOD"
    --checkpoint "$CKPT"
    --dataset "$DATASET" --split test
    --simulate_imu_noise --simulate_imu_noise_seed 42)

# Small online evaluation whose predictions are rendered and inspected before
# the full test-set numbers are trusted.
"${EVAL[@]}" \
    --max_sequences "${VIS_SEQUENCES:-24}" \
    --output_json "$RUN/eval_vis_subset.json" \
    --save_predictions "$RUN/predictions_vis" > "$RUN/eval_vis_subset.log" 2>&1
echo "$(stamp) vis subset evaluated" >> "$RUN/chain.log"

"${EVAL[@]}" \
    --output_json "$RUN/eval_${DATASET}_test_online.json" > "$RUN/eval_full.log" 2>&1
echo "$(stamp) full test evaluated" >> "$RUN/chain.log"

if [ "$METHOD" = dip18 ]; then
    "${EVAL[@]}" --gt_root_trans \
        --output_json "$RUN/eval_${DATASET}_test_online_gtroot.json" > "$RUN/eval_full_gtroot.log" 2>&1
    echo "$(stamp) full test evaluated (GT root translation)" >> "$RUN/chain.log"
fi
