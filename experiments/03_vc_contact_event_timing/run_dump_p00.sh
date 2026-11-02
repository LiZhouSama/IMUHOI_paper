#!/usr/bin/env bash
# Sensitivity check: OMOMO v4 and IMUHOI-VC pipelines were trained/selected under the old 0/0 protocol
# (no HP smoothing, VC's own output as velocity init); dump them again under that native protocol.
set -u
ROOT=/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
PY=/home/l/anaconda3/envs/SAGE/bin/python
RC=$ROOT/outputs/IMUHOI++/RNN_cano
OUT=$RC/vc_event_timing_20261028
cd "$ROOT"
COMMON="--simulate_imu_noise --simulate_imu_noise_seed 42 --seed 42 --num_workers 2 --online_batch_size 48 \
--human_pose_smoothing_half_width 0 --online_obj_vel_init_diff_frames 0"
HP=$RC/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt
run() {  # name, pipeline dir, run subdir, vc arch
    local name=$1 pipe=$2 sub=$3 arch=$4
    echo "[$(date '+%F %T')] START $name"
    $PY scripts/vc_probe/dump_vc_features.py --feature_dir "$OUT/$name/probs" --probs_only -- \
        --config "$pipe/config.yaml" --dataset omomo --hp_ckpt "$HP" \
        --vc_ckpt "$pipe/$sub/best_velocity_contact.pt" --ot_ckpt "$pipe/$sub/best_object_trans.pt" \
        --velocity_contact_arch "$arch" --object_trans_arch v2 $COMMON \
        --output_json "$OUT/$name/eval.json" > "$OUT/$name.log" 2>&1
    echo "[$(date '+%F %T')] END $name rc=$?"
}
run omomo_v4_p00 $RC/v2_armrot_biclose_20260923_omomo/v2_armrot_biclose \
    object_trans_v2_vc_v4_09232346_omomo_ot_v2_armrot_biclose_20260923 v4
run omomo_imuhoivc_p00 $RC/vc_ot_cross_ablation_omomo_imuhoi_v2_armrot_biclose/v2_armrot_biclose \
    object_trans_v2_vc_IMUHOI_09300313_omomo_ot_v2_armrot_biclose_vc_ot_cross_ablation imuhoi
$PY - <<'PYEOF'
import json
RC = "/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI/outputs/IMUHOI++/RNN_cano"
for name, ref in (("omomo_v4_p00", "v2_armrot_biclose_20260923_omomo/v2_armrot_biclose/eval_omomo_best.json"),
                  ("omomo_imuhoivc_p00", "vc_ot_cross_ablation_omomo_imuhoi_v2_armrot_biclose/v2_armrot_biclose/eval_omomo_best.json")):
    a = json.load(open(f"{RC}/vc_event_timing_20261028/{name}/eval.json"))["results"]["omomo"]
    b = json.load(open(f"{RC}/{ref}"))["results"]["omomo"]
    keys = ("obj_trans_err_fusion_refine", "hoi_err_fusion", "contact_f1_lhand", "contact_f1_rhand", "contact_f1_obj")
    d = {k: (round(a[k], 4), round(b[k], 4)) for k in keys if a[k] != b[k]}
    print(name, "identical to stored 0/0 eval" if not d else f"now vs stored {d}")
PYEOF
echo "[$(date '+%F %T')] ALL DONE"
