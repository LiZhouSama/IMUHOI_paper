#!/usr/bin/env bash
# A3 (2026-10-28): downstream sensitivity of the object branch to contact errors. GT contact (hands + object motion)
# replaces the VC contact output inside the online loop, delayed by K frames or corrupted by flicker, for our
# pipeline (VC-v5ar + OT v2_armrot_biclose) and the original three-branch IMUHOI; both under their native 7/3
# protocol and with identical corrupted contact. Starts after the A2 chain so that only one GPU job runs at a time.
set -u
ROOT=/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
PY=/home/l/anaconda3/envs/SAGE/bin/python
RC=$ROOT/outputs/IMUHOI++/RNN_cano
OUT=$RC/vc_contact_oracle_20261028
cd "$ROOT"
until grep -q "ALL DONE" $RC/vc_knockout_20261028/run_a2.log 2>/dev/null; do sleep 60; done

COMMON="--simulate_imu_noise --simulate_imu_noise_seed 42 --seed 42 --num_workers 2 --online_batch_size 48 \
--human_pose_smoothing_half_width 7 --online_obj_vel_init_diff_frames 3"
args_for() {  # dataset pipeline
    local ds=$1 pipe=$2 U J
    if [ $pipe = ours ] && [ $ds = omomo ]; then
        U=$RC/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose; J=$U/object_trans_v2_10020305_omomo_ot_v2_armrot_biclose_20261002u
        echo "--config $U/config.yaml --dataset omomo --hp_ckpt $RC/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt \
--vc_ckpt $J/best_velocity_contact.pt --ot_ckpt $J/best_object_trans.pt --velocity_contact_arch v5 --object_trans_arch v2"
    elif [ $pipe = ours ]; then
        U=$RC/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose; J=$U/object_trans_v2_09301506_behave_ot_v2_armrot_biclose_20260930
        echo "--config $U/config.yaml --dataset behave --hp_ckpt $RC/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/best.pt \
--vc_ckpt $J/best_velocity_contact.pt --ot_ckpt $J/best_object_trans.pt --velocity_contact_arch v5 --object_trans_arch v2"
    elif [ $ds = omomo ]; then
        U=$RC/imuhoi3b_20261012_omomo_unified/vc_imuhoi_ot_3branch; J=$U/object_trans_IMUHOI_vc_IMUHOI_10121941_omomo_ot_vc_imuhoi_ot_3branch_20261012u
        echo "--config $U/config.yaml --dataset omomo --hp_ckpt $J/best_human_pose.pt \
--vc_ckpt $RC/imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008/best.pt \
--ot_ckpt $J/best_object_trans.pt --velocity_contact_arch imuhoi --object_trans_arch imuhoi"
    else
        U=$RC/imuhoi3b_20261008_behave/vc_imuhoi_ot_3branch; J=$U/object_trans_IMUHOI_vc_IMUHOI_10082127_behave_ot_vc_imuhoi_ot_3branch_20261008
        echo "--config $U/config.yaml --dataset behave --hp_ckpt $J/best_human_pose.pt \
--vc_ckpt $RC/imuhoi3b_20261008_behave/vc_imuhoi/velocity_contact_IMUHOI_10082052_behave_vc_imuhoi_20261008/best.pt \
--ot_ckpt $J/best_object_trans.pt --velocity_contact_arch imuhoi --object_trans_arch imuhoi"
    fi
}
run() {  # dataset pipeline condition-name shift flip
    local ds=$1 pipe=$2 name=$3 shift=$4 flip=$5 tag=${1}_${2}_${3}
    echo "[$(date '+%F %T')] START $tag"
    $PY scripts/vc_probe/dump_contact_oracle.py --flip_frac $flip -- $(args_for $ds $pipe) $COMMON \
        --oracle_vc_contact all --oracle_vc_shift $shift --output_json $OUT/$tag.json \
        --sequence_records_json $OUT/${tag}_records.json > $OUT/$tag.log 2>&1
    echo "[$(date '+%F %T')] END $tag rc=$? $(grep -h '\[A3\]' $OUT/$tag.log | cut -c1-80)"
}
for ds in behave omomo; do
    for cond in s0:0:0 s-5:-5:0 s5:5:0 s-10:-10:0 s10:10:0 s-15:-15:0 s15:15:0 f05:0:0.05 f10:0:0.10 f20:0:0.20; do
        name=${cond%%:*}; rest=${cond#*:}; shift=${rest%%:*}; flip=${rest#*:}
        for pipe in ours imuhoi3b; do
            run $ds $pipe $name $shift $flip
        done
    done
done
echo "[$(date '+%F %T')] ALL DONE"
