#!/bin/bash
# HP plug-in experiment (2026-11-05): IMUHOI++'s VC + OT on top of another method's human pose.
# One job = (dataset, comparison method) on one GPU:
#   0. external-HP cache of the method (scripts/hp_plugin/build_hp_cache.py): its online pose and
#      root on every train / validation / test sequence, plus a re-check of its published eval;
#   1. VC v5ar retrained from scratch on that HP (OMOMO / BEHAVE: the published v5ar recipe, as the
#      `full` row of outputs/IMUHOI++/RNN_cano/vc_ablation_omomo_20261030/run_ablation.sh);
#   2. OT v2_armrot_biclose + joint stage retrained, online evals of the best and final checkpoints
#      with per-sequence records (unified protocol: diff-3 velocity init; HP smoothing does not apply
#      to an external HP).  IMHD / PAHOI / HODome run steps 1-2 as one orchestrator call from the
#      published recipe.yaml, exactly as their published pipelines.
# The HP is frozen everywhere; its "checkpoint" is the cache manifest, and the configs carry
# external_hp_method / external_hp_cache_dir (model/rnn/external_human_pose.py).
#
# Usage: run_plugin.sh <omomo|behave|imhd|pahoi|hodome> <dip18|tip|transpose|globalpose> <gpu index>
# Env:   IMUHOI_PYTHON    interpreter of every stage (default: python on PATH)
#        HP_PLUGIN_ROOT   output root (default outputs/IMUHOI++/RNN_cano/hp_plugin_20261105)
#        PHYSICS_WORKERS  CPU processes of the GlobalPose physics during the cache build (default 12)
# Re-running a job resumes it: finished stages are skipped, interrupted trainings resume from last.pt.
cd "$(dirname "$0")/../.." || exit 1
ds=$1; m=$2; gpu=$3
if [ -z "$ds" ] || [ -z "$m" ] || [ -z "$gpu" ]; then
  echo "usage: $0 <dataset> <method> <gpu>"; exit 2
fi
PY=${IMUHOI_PYTHON:-$(command -v python)}
export IMUHOI_PYTHON=$PY  # read by scripts/run_behave_full_pipeline_trust.py for its stages
R=outputs/IMUHOI++/RNN_cano
case "$m" in
  dip18|tip|transpose|globalpose) ;;
  *) echo "unknown method $m"; exit 2 ;;
esac
RECIPE=""
case "$ds" in
  omomo)  VCCFG=$R/vc_v5_20260930/vc_config_omomo_v5ar.yaml; EPOCHS=300; DATACFG=$VCCFG ;;
  behave) VCCFG=$R/vc_v5_20260930/vc_config_behave_v5ar.yaml; EPOCHS=986; DATACFG=$VCCFG ;;
  imhd|pahoi|hodome) RECIPE=$R/remaining3_v5ar_vs_imuhoi3b_20261010/recipe.yaml; DATACFG=$RECIPE ;;
  *) echo "unknown dataset $ds"; exit 2 ;;
esac
ROOT=${HP_PLUGIN_ROOT:-$R/hp_plugin_20261105}
CACHE=$ROOT/cache/$ds/$m
JOB=$ROOT/$ds/$m
mkdir -p "$JOB"
log() { echo "[$(date +%FT%T)] $ds/$m gpu$gpu: $*" | tee -a "$JOB/job.log"; }
EXT=(--set "external_hp_method=$m" --set "external_hp_cache_dir=$CACHE")
log "start (python $PY, root $ROOT)"

# ---- 0. external-HP cache ---------------------------------------------------------------------
if [ ! -s "$CACHE/manifest.json" ]; then
  log "cache build"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/hp_plugin/build_hp_cache.py --dataset "$ds" --method "$m" \
    --cache_dir "$CACHE" --cfg "$DATACFG" --python "$PY" --physics_workers "${PHYSICS_WORKERS:-12}" \
    >> "$JOB/cache.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -s "$CACHE/manifest.json" ] || { log "cache FAILED (rc=$rc, see $JOB/cache.log)"; exit 1; }
fi
log "cache ready: $(grep -o '"passed": [a-z]*' "$CACHE/manifest.json" | head -1)"

OTROOT=$JOB/pipeline
PIPE=$OTROOT/v2_armrot_biclose
if [ -n "$RECIPE" ]; then
  # ---- 1+2. VC, OT + joint and evals in one orchestrator run (the published recipe) ---------------
  for attempt in 1 2 3; do
    [ -s "$PIPE/eval_${ds}_best.json" ] && [ -s "$PIPE/eval_${ds}_final.json" ] && break
    log "pipeline attempt $attempt (recipe=$RECIPE)"
    $PY scripts/run_behave_full_pipeline_trust.py --cfg "$RECIPE" --experiment_root "$OTROOT" \
      --dataset "$ds" --vc_arch v5 --variants v2_armrot_biclose --run_tag "hpplug_${m}" --seed 42 --gpu "$gpu" \
      --num_workers 2 --eval_batch_size 48 --max_parallel 1 --schedule_scale auto --resume --eval_records \
      --check_gpu_before_stage --skip_hp "$CACHE/manifest.json" "${EXT[@]}" >> "$OTROOT.launcher.log" 2>&1
    log "pipeline attempt $attempt exit $?"
  done
else
  # ---- 1. VC ------------------------------------------------------------------------------------
  CFG=$JOB/vc_config.yaml
  if [ ! -f "$CFG" ]; then
    $PY - "$VCCFG" "$CFG" "external_hp_method=$m" "external_hp_cache_dir=$CACHE" <<'EOF'
import re, sys
src, dst, *sets = sys.argv[1:]
text = open(src).read()
for item in sets:  # replace a top-level key in place, else append it
    key, value = item.split("=", 1)
    line = f"{key}: {value}"
    text, n = re.subn(rf"^{re.escape(key)}:.*$", line, text, flags=re.M)
    if n == 0:
        text = text.rstrip("\n") + "\n" + line + "\n"
open(dst, "w").write(text)
EOF
  fi
  SUFFIX=${ds}_vc_hpplug_${m}
  for attempt in 1 2 3 4; do
    stage=$(ls -d "$JOB"/vc/velocity_contact_v5_*_"${SUFFIX}" 2>/dev/null | tail -1)
    [ -n "$stage" ] && [ -f "$stage/final.pt" ] && break
    resume=()
    [ -n "$stage" ] && [ -f "$stage/last.pt" ] && resume=(--resume_dir "$stage")
    log "VC attempt $attempt ${resume[*]}"
    CUDA_VISIBLE_DEVICES=$gpu $PY train/rnn/train_velocity_contact.py --cfg "$CFG" --save_dir "$JOB/vc" \
      --run_suffix "$SUFFIX" --seed 42 --train_datasets "$ds" --num_workers 2 --simulate_imu_noise \
      --epochs "$EPOCHS" --velocity_contact_arch v5 --hp_ckpt "$CACHE/manifest.json" "${resume[@]}" \
      >> "$JOB/vc.log" 2>&1
    log "VC attempt $attempt exit $?"
  done
  VC=$(ls "$JOB"/vc/velocity_contact_v5_*_"${SUFFIX}"/best.pt 2>/dev/null | tail -1)
  [ -f "$VC" ] || { log "VC FAILED (see $JOB/vc.log)"; exit 1; }

  # ---- 2. OT + joint + evals ----------------------------------------------------------------------
  for attempt in 1 2 3; do
    [ -s "$PIPE/eval_${ds}_best.json" ] && [ -s "$PIPE/eval_${ds}_final.json" ] && break
    log "pipeline attempt $attempt (vc=$VC)"
    $PY scripts/run_behave_full_pipeline_trust.py --dataset "$ds" --experiment_root "$OTROOT" \
      --run_tag "hpplug_${m}" --vc_arch v5 --skip_hp "$CACHE/manifest.json" --skip_vc "$VC" --gpu "$gpu" \
      --max_parallel 1 --max_attempts 3 --resume --eval_records --check_gpu_before_stage \
      --variants v2_armrot_biclose --set v2_fk_filter_alpha=0.35 --set v2_fk_filter_beta=0.05 \
      --set v5_heading_normalize_imu=false --set v5_init_noise_prob=0.5 --set v5_obj_vel_init_noise_std=0.2 \
      --set v5_hand_vel_init_noise_std=0.2 --set v5_contact_init_flip_prob=0.0 \
      --set human_pose_smoothing_half_width=7 --set online_obj_vel_init_diff_frames=3 \
      "${EXT[@]}" >> "$OTROOT.launcher.log" 2>&1
    log "pipeline attempt $attempt exit $?"
  done
fi
[ -s "$PIPE/eval_${ds}_best.json" ] || { log "pipeline INCOMPLETE (see $OTROOT.launcher.log, $OTROOT/orchestrator.log)"; exit 1; }
log "DONE $($PY - "$PIPE/eval_${ds}_best.json" "$ds" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))["results"][sys.argv[2]]
print(f"obj {r['obj_trans_err_fusion_refine']:.2f} HOI {r['hoi_err_fusion']:.2f} MPJPE {r['mpjpe']:.2f} root {r['root_trans_err']:.2f}")
EOF
)"
