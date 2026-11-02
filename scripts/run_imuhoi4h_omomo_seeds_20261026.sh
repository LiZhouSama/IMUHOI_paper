#!/usr/bin/env bash
# Seed repeats of the four-branch IMUHOI on OMOMO (seed 42 = imuhoi4h_all5_20261026/omomo).
#
# Same recipe, HP and VC-IMUHOI checkpoints as the all5 run; only the training
# seed differs (the training noise realization follows it).  Evaluation stays on
# seed 42 (--eval_seed) so every repeat is scored on the same noisy test stream.
# Afterwards scripts/summarize_imuhoi4h_omomo_seeds_20261026.py adds the gate
# diagnostics and writes a separate table (not merged into the method table).
#
# Usage: scripts/run_imuhoi4h_omomo_seeds_20261026.sh [seed ...]   (default 43 44, run in parallel)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=/home/l/anaconda3/envs/SAGE/bin/python
R=outputs/IMUHOI++/RNN_cano
OUT=$R/imuhoi4h_omomo_seeds_20261026
SRC=$R/imuhoi3b_20261012_omomo_unified
SEEDS=("${@:-43 44}")
read -r -a SEEDS <<< "${SEEDS[*]}"
mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*" >> "$OUT/queue.log"; }

mapfile -t spec < <("$PY" - "$SRC/experiment_manifest.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["source_config"])
print(m["stages"]["human_pose"]["checkpoint"])
print(m["stages"]["velocity_contact"]["checkpoint"])
for key, value in m["shared_overrides"].items():
    print(f"{key}={json.dumps(value)}")
EOF
)
sets=()
for item in "${spec[@]:3}"; do sets+=(--set "$item"); done

for seed in "${SEEDS[@]}"; do
    log "start seed $seed"
    (
        "$PY" scripts/run_behave_full_pipeline_trust.py --cfg "${spec[0]}" --experiment_root "$OUT/seed$seed" \
            --dataset omomo --vc_arch imuhoi --variants vc_imuhoi_ot_4heads --run_tag "imuhoi4h_s${seed}_20261026" \
            --seed "$seed" --eval_seed 42 --num_workers 2 --eval_batch_size 48 --max_parallel 1 --schedule_scale auto \
            --skip_hp "${spec[1]}" --skip_vc "${spec[2]}" "${sets[@]}" --resume --check_gpu_before_stage \
            > "$OUT/seed$seed.log" 2>&1
        log "end seed $seed rc=$?"
    ) &
    sleep 240
done
wait
log "training finished; summarizing"
"$PY" scripts/summarize_imuhoi4h_omomo_seeds_20261026.py >> "$OUT/queue.log" 2>&1
log "queue finished rc=$?"
