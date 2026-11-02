#!/usr/bin/env bash
# A1 contact-event timing (2026-10-28): per-frame VC probabilities (hand contact, object motion, endpoint and
# anticipation heads) at every published online frame, unified protocol (HP smoothing 7, diff-3 velocity init)
# for every run.  The original three-branch IMUHOI is reused from vc_probe_pilot_20261027 (same protocol).
#   v5ar      IMUHOI++ VC (pose prior + coupling + init-robust) + OT v2_armrot_biclose
#   v4        pose-prior VC (no coupling, no init perturbation) + OT v2_armrot_biclose
#   imuhoivc  IMUHOI VC + OT v2_armrot_biclose (VC fine-tuned in our joint stage)
set -u
ROOT=/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
PY=/home/l/anaconda3/envs/SAGE/bin/python
RC=$ROOT/outputs/IMUHOI++/RNN_cano
OUT=$RC/vc_event_timing_20261028
cd "$ROOT"

COMMON="--simulate_imu_noise --simulate_imu_noise_seed 42 --seed 42 --num_workers 2 --online_batch_size 48 \
--human_pose_smoothing_half_width 7 --online_obj_vel_init_diff_frames 3"
HP_OMOMO=$RC/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt
HP_BEHAVE=$RC/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/best.pt

run() {  # name, dataset, pipeline dir, run subdir, hp ckpt, vc arch
    local name=$1 ds=$2 pipe=$3 sub=$4 hp=$5 arch=$6
    echo "[$(date '+%F %T')] START $name"
    $PY scripts/vc_probe/dump_vc_features.py --feature_dir "$OUT/$name/probs" --probs_only -- \
        --config "$pipe/config.yaml" --dataset "$ds" --hp_ckpt "$hp" \
        --vc_ckpt "$pipe/$sub/best_velocity_contact.pt" --ot_ckpt "$pipe/$sub/best_object_trans.pt" \
        --velocity_contact_arch "$arch" --object_trans_arch v2 $COMMON \
        --output_json "$OUT/$name/eval.json" > "$OUT/$name.log" 2>&1
    echo "[$(date '+%F %T')] END $name rc=$?"
}

run omomo_v5ar omomo $RC/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose \
    object_trans_v2_10020305_omomo_ot_v2_armrot_biclose_20261002u $HP_OMOMO v5
run omomo_v4 omomo $RC/v2_armrot_biclose_20260923_omomo/v2_armrot_biclose \
    object_trans_v2_vc_v4_09232346_omomo_ot_v2_armrot_biclose_20260923 $HP_OMOMO v4
run omomo_imuhoivc omomo $RC/vc_ot_cross_ablation_omomo_imuhoi_v2_armrot_biclose/v2_armrot_biclose \
    object_trans_v2_vc_IMUHOI_09300313_omomo_ot_v2_armrot_biclose_vc_ot_cross_ablation $HP_OMOMO imuhoi
run behave_v5ar behave $RC/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose \
    object_trans_v2_09301506_behave_ot_v2_armrot_biclose_20260930 $HP_BEHAVE v5
run behave_v4 behave $RC/v2_armrot_biclose_20260923_behave/v2_armrot_biclose \
    object_trans_v2_vc_v4_09240321_behave_ot_v2_armrot_biclose_20260923 $HP_BEHAVE v4
run behave_imuhoivc behave $RC/vc_ot_cross_ablation_behave_imuhoi_v2_armrot_biclose/v2_armrot_biclose \
    object_trans_v2_vc_IMUHOI_09300537_behave_ot_v2_armrot_biclose_vc_ot_cross_ablation $HP_BEHAVE imuhoi

# Compare with the pipelines' stored evals (OMOMO v4 / imuhoivc were stored under the old 0/0 protocol).
$PY - <<'EOF'
import json, math
RC = "/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI/outputs/IMUHOI++/RNN_cano"
pairs = {
    "omomo_v5ar": ("omomo", "vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/eval_omomo_best.json"),
    "omomo_v4": ("omomo", "v2_armrot_biclose_20260923_omomo/v2_armrot_biclose/eval_omomo_best.json"),
    "omomo_imuhoivc": ("omomo", "vc_ot_cross_ablation_omomo_imuhoi_v2_armrot_biclose/v2_armrot_biclose/eval_omomo_best.json"),
    "behave_v5ar": ("behave", "vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/eval_behave_best.json"),
    "behave_v4": ("behave", "v2_armrot_biclose_20260923_behave/v2_armrot_biclose/eval_behave_best.json"),
    "behave_imuhoivc": ("behave", "vc_ot_cross_ablation_behave_imuhoi_v2_armrot_biclose/v2_armrot_biclose/eval_behave_best.json"),
}
keys = ("obj_trans_err_fusion_refine", "hoi_err_fusion", "mpjpe", "root_trans_err", "contact_f1_lhand",
        "contact_f1_rhand", "contact_f1_obj", "obj_jitter")
for name, (ds, ref) in pairs.items():
    try:
        a = json.load(open(f"{RC}/vc_event_timing_20261028/{name}/eval.json"))["results"][ds]
    except Exception as exc:
        print(name, "MISSING", exc); continue
    stored = json.load(open(f"{RC}/{ref}"))
    b = stored["results"][ds]
    proto = (stored.get("human_pose_smoothing_half_width"), stored.get("online_obj_vel_init_diff_frames"))
    diffs = {k: (round(a[k], 4), round(b[k], 4)) for k in keys if k in a and k in b and a[k] != b[k]}
    print(f"{name}: stored protocol {proto}; " + ("identical" if not diffs else f"now vs stored {diffs}"))
EOF
echo "[$(date '+%F %T')] ALL DONE"
