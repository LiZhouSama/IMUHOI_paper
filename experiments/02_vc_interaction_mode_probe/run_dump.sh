#!/usr/bin/env bash
# VC probing pilot (2026-10-27): dump VC internal features at every published online frame for the
# OMOMO / BEHAVE best pipelines of IMUHOI++ (VC-v5ar + OT v2_armrot_biclose) and the three-branch IMUHOI.
# Each run is the pipeline's own eval command (orchestrator logs) through scripts/vc_probe/dump_vc_features.py;
# its eval JSON must reproduce the pipeline's stored eval_<ds>_best.json.
set -u
ROOT=/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
PY=/home/l/anaconda3/envs/SAGE/bin/python
RC=$ROOT/outputs/IMUHOI++/RNN_cano
OUT=$RC/vc_probe_pilot_20261027
cd "$ROOT"

COMMON="--simulate_imu_noise --simulate_imu_noise_seed 42 --seed 42 --num_workers 2 --online_batch_size 48"

run() {  # name, extra dump flags, eval args...
    local name=$1; shift
    local dumpflags=$1; shift
    echo "[$(date '+%F %T')] START $name"
    $PY scripts/vc_probe/dump_vc_features.py --feature_dir "$OUT/$name/features" $dumpflags -- "$@" \
        --output_json "$OUT/$name/eval.json" > "$OUT/$name.log" 2>&1
    echo "[$(date '+%F %T')] END $name rc=$?"
}

# OMOMO, IMUHOI++
U=$RC/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose
J=$U/object_trans_v2_10020305_omomo_ot_v2_armrot_biclose_20261002u
run omomo_ours --random_control --config $U/config.yaml --dataset omomo \
    --hp_ckpt $RC/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt \
    --vc_ckpt $J/best_velocity_contact.pt --ot_ckpt $J/best_object_trans.pt \
    --velocity_contact_arch v5 --object_trans_arch v2 $COMMON

# OMOMO, IMUHOI three-branch
U=$RC/imuhoi3b_20261012_omomo_unified/vc_imuhoi_ot_3branch
J=$U/object_trans_IMUHOI_vc_IMUHOI_10121941_omomo_ot_vc_imuhoi_ot_3branch_20261012u
run omomo_imuhoi "" --config $U/config.yaml --dataset omomo \
    --hp_ckpt $J/best_human_pose.pt \
    --vc_ckpt $RC/imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008/best.pt \
    --ot_ckpt $J/best_object_trans.pt \
    --velocity_contact_arch imuhoi --object_trans_arch imuhoi $COMMON

# BEHAVE, IMUHOI++
U=$RC/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose
J=$U/object_trans_v2_09301506_behave_ot_v2_armrot_biclose_20260930
run behave_ours --random_control --config $U/config.yaml --dataset behave \
    --hp_ckpt $RC/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/best.pt \
    --vc_ckpt $J/best_velocity_contact.pt --ot_ckpt $J/best_object_trans.pt \
    --velocity_contact_arch v5 --object_trans_arch v2 $COMMON

# BEHAVE, IMUHOI three-branch
U=$RC/imuhoi3b_20261008_behave/vc_imuhoi_ot_3branch
J=$U/object_trans_IMUHOI_vc_IMUHOI_10082127_behave_ot_vc_imuhoi_ot_3branch_20261008
run behave_imuhoi "" --config $U/config.yaml --dataset behave \
    --hp_ckpt $J/best_human_pose.pt \
    --vc_ckpt $RC/imuhoi3b_20261008_behave/vc_imuhoi/velocity_contact_IMUHOI_10082052_behave_vc_imuhoi_20261008/best.pt \
    --ot_ckpt $J/best_object_trans.pt \
    --velocity_contact_arch imuhoi --object_trans_arch imuhoi $COMMON

# Reproduction check against the stored pipeline evals.
$PY - <<'EOF'
import json, math
RC = "/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI/outputs/IMUHOI++/RNN_cano"
pairs = {
    "omomo_ours": ("omomo", "vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/eval_omomo_best.json"),
    "omomo_imuhoi": ("omomo", "imuhoi3b_20261012_omomo_unified/vc_imuhoi_ot_3branch/eval_omomo_best.json"),
    "behave_ours": ("behave", "vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/eval_behave_best.json"),
    "behave_imuhoi": ("behave", "imuhoi3b_20261008_behave/vc_imuhoi_ot_3branch/eval_behave_best.json"),
}
for name, (ds, ref) in pairs.items():
    try:
        a = json.load(open(f"{RC}/vc_probe_pilot_20261027/{name}/eval.json"))["results"][ds]
    except Exception as exc:
        print(name, "MISSING", exc); continue
    b = json.load(open(f"{RC}/{ref}"))["results"][ds]
    keys = [k for k in ("obj_trans_err_fusion_refine", "hoi_err_fusion", "mpjpe", "root_trans_err",
                        "contact_f1_lhand", "contact_f1_rhand", "obj_jitter") if k in a]
    diffs = {k: (a[k], b.get(k)) for k in keys if not (a[k] == b.get(k) or (isinstance(a[k], float) and math.isnan(a[k])))}
    print(name, "identical" if not diffs else f"DIFFERS {diffs}")
EOF
echo "[$(date '+%F %T')] ALL DONE"
