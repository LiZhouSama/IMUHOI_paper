#!/usr/bin/env bash
# A2 (2026-10-28): body-part knockouts and integrated gradients of the IMUHOI++ VC (v5ar) under its native 7/3
# protocol. Per dataset: one reference run (means of every pose input, capture of the windows that publish a GT
# hand-contact onset/offset, IG on them), then six knockout runs that replace inputs by those means.
set -u
ROOT=/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
PY=/home/l/anaconda3/envs/SAGE/bin/python
RC=$ROOT/outputs/IMUHOI++/RNN_cano
OUT=$RC/vc_knockout_20261028
cd "$ROOT"
COMMON="--simulate_imu_noise --simulate_imu_noise_seed 42 --seed 42 --num_workers 2 --online_batch_size 48 \
--human_pose_smoothing_half_width 7 --online_obj_vel_init_diff_frames 3 --velocity_contact_arch v5 --object_trans_arch v2"

args_for() {  # dataset -> eval arguments of the v5ar best pipeline
    if [ "$1" = omomo ]; then
        local U=$RC/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose
        local J=$U/object_trans_v2_10020305_omomo_ot_v2_armrot_biclose_20261002u
        echo "--config $U/config.yaml --dataset omomo --hp_ckpt $RC/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt \
--vc_ckpt $J/best_velocity_contact.pt --ot_ckpt $J/best_object_trans.pt $COMMON"
    else
        local U=$RC/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose
        local J=$U/object_trans_v2_09301506_behave_ot_v2_armrot_biclose_20260930
        echo "--config $U/config.yaml --dataset behave --hp_ckpt $RC/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/best.pt \
--vc_ckpt $J/best_velocity_contact.pt --ot_ckpt $J/best_object_trans.pt $COMMON"
    fi
}

for ds in omomo behave; do
    EV=$(args_for $ds)
    echo "[$(date '+%F %T')] START ${ds}_ref"
    $PY scripts/vc_probe/dump_vc_knockout.py --probs_dir $OUT/${ds}_ref/probs --stats_out $OUT/${ds}_means.pt \
        --capture_out $OUT/${ds}_windows.pt --ig_out $OUT/${ds}_ig.pt -- $EV --output_json $OUT/${ds}_ref/eval.json \
        > $OUT/${ds}_ref.log 2>&1
    echo "[$(date '+%F %T')] END ${ds}_ref rc=$? $(grep -h '\[A2\] integrated' $OUT/${ds}_ref.log | cut -c1-40)"
    for k in left_arm right_arm torso legs pose_prior hp_all; do
        echo "[$(date '+%F %T')] START ${ds}_$k"
        $PY scripts/vc_probe/dump_vc_knockout.py --probs_dir $OUT/${ds}_$k/probs --knockout $k --means $OUT/${ds}_means.pt \
            -- $EV --output_json $OUT/${ds}_$k/eval.json > $OUT/${ds}_$k.log 2>&1
        echo "[$(date '+%F %T')] END ${ds}_$k rc=$?"
    done
done
# The reference runs must reproduce the stored v5ar evals (knockouts off).
$PY - <<'EOF'
import json
RC = "/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI/outputs/IMUHOI++/RNN_cano"
for ds, ref in (("omomo", "vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/eval_omomo_best.json"),
                ("behave", "vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/eval_behave_best.json")):
    a = json.load(open(f"{RC}/vc_knockout_20261028/{ds}_ref/eval.json"))["results"][ds]
    b = json.load(open(f"{RC}/{ref}"))["results"][ds]
    keys = ("obj_trans_err_fusion_refine", "hoi_err_fusion", "contact_f1_lhand", "contact_f1_rhand", "contact_f1_obj")
    d = {k: (a[k], b[k]) for k in keys if a[k] != b[k]}
    print(ds, "reference identical to stored eval" if not d else f"reference DIFFERS {d}")
EOF
echo "[$(date '+%F %T')] ALL DONE"
