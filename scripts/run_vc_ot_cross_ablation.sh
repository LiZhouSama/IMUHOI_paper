#!/usr/bin/env bash
# Train the two missing crossed VC x OT cells on each canonical dataset.
# Source each OT recipe from its original run; reuse the same domain-specific HP
# and VC stage-2 checkpoints.  --schedule_scale 1 avoids scaling an already
# scaled BEHAVE config a second time.  Outputs can be resumed safely.
set -euo pipefail
cd /mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
PY=/home/l/anaconda3/envs/SAGE/bin/python
R=outputs/IMUHOI++/RNN_cano
TAG=vc_ot_cross_ablation
run_one() {
  local ds="$1" vc_arch="$2" variant="$3" cfg="$4" hp="$5" vc="$6"
  local root="$R/${TAG}_${ds}_${vc_arch}_${variant}"
  echo "=== $ds: VC=$vc_arch OT=$variant ==="
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_behave_full_pipeline_trust.py \
    --cfg "$cfg" --schedule_scale 1 --experiment_root "$root" --resume \
    --run_tag "$TAG" --dataset "$ds" --vc_arch "$vc_arch" \
    --variants "$variant" --skip_hp "$hp" --skip_vc "$vc" \
    --seed 42 --gpu 0 --num_workers 2 --eval_batch_size 48 --max_parallel 1
}
HPO="$R/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt"
HPB="$R/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/best.pt"
V4O="$R/relabel_20260918_omomo/vc_v4/velocity_contact_v4_09182114_omomo_vc_v4_20260918/best.pt"
V4B="$R/relabel_20260918_behave/vc_v4/velocity_contact_v4_09182116_behave_vc_v4_20260918/best.pt"
IMO="$R/relabel_20260920_omomo_imuhoi/vc_v4/velocity_contact_IMUHOI_09210023_omomo_vc_imuhoi_20260920/best.pt"
IMB="$R/relabel_20260920_behave_imuhoi/vc_imuhoi/velocity_contact_IMUHOI_09210116_behave_vc_imuhoi_20260920/best.pt"
CFG_4O="$R/relabel_20260920_omomo_imuhoi/vc_imuhoi_ot_4heads/config.yaml"
CFG_4B="$R/relabel_20260922_behave_hpsmooth_imuhoi/vc_imuhoi_ot_4heads/config.yaml"
CFG_V2O="$R/v2_armrot_biclose_20260923_omomo/v2_armrot_biclose/config.yaml"
CFG_V2B="$R/v2_armrot_biclose_20260923_behave/v2_armrot_biclose/config.yaml"
# Restrict to one requested cell (e.g. "omomo:v4") for a resumed run.
requested="${1:-all}"
if [[ "$requested" == all || "$requested" == omomo:v4 ]]; then
  run_one omomo v4 vc_imuhoi_ot_4heads "$CFG_4O" "$HPO" "$V4O"
fi
if [[ "$requested" == all || "$requested" == omomo:imuhoi ]]; then
  run_one omomo imuhoi v2_armrot_biclose "$CFG_V2O" "$HPO" "$IMO"
fi
if [[ "$requested" == all || "$requested" == behave:v4 ]]; then
  run_one behave v4 vc_imuhoi_ot_4heads "$CFG_4B" "$HPB" "$V4B"
fi
if [[ "$requested" == all || "$requested" == behave:imuhoi ]]; then
  run_one behave imuhoi v2_armrot_biclose "$CFG_V2B" "$HPB" "$IMB"
fi

