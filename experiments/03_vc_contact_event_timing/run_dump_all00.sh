#!/usr/bin/env bash
# Protocol sensitivity, part 2: every remaining model under the old 0/0 protocol (no HP smoothing, VC's own
# previous output as velocity init), so that all four models can be compared under 7/3 and under 0/0.
set -u
ROOT=/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
PY=/home/l/anaconda3/envs/SAGE/bin/python
RC=$ROOT/outputs/IMUHOI++/RNN_cano
OUT=$RC/vc_event_timing_20261028
cd "$ROOT"
COMMON="--simulate_imu_noise --simulate_imu_noise_seed 42 --seed 42 --num_workers 2 --online_batch_size 48 \
--human_pose_smoothing_half_width 0 --online_obj_vel_init_diff_frames 0"
HP_OMOMO=$RC/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt
HP_BEHAVE=$RC/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/best.pt

run() {  # name, dataset, config, hp, vc, ot, vc arch, ot arch
    local name=$1 ds=$2 cfg=$3 hp=$4 vc=$5 ot=$6 varch=$7 oarch=$8
    echo "[$(date '+%F %T')] START $name"
    $PY scripts/vc_probe/dump_vc_features.py --feature_dir "$OUT/$name/probs" --probs_only -- \
        --config "$cfg" --dataset "$ds" --hp_ckpt "$hp" --vc_ckpt "$vc" --ot_ckpt "$ot" \
        --velocity_contact_arch "$varch" --object_trans_arch "$oarch" $COMMON \
        --output_json "$OUT/$name/eval.json" > "$OUT/$name.log" 2>&1
    echo "[$(date '+%F %T')] END $name rc=$?"
}

U=$RC/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose; J=$U/object_trans_v2_10020305_omomo_ot_v2_armrot_biclose_20261002u
run omomo_v5ar_p00 omomo $U/config.yaml $HP_OMOMO $J/best_velocity_contact.pt $J/best_object_trans.pt v5 v2
U=$RC/imuhoi3b_20261012_omomo_unified/vc_imuhoi_ot_3branch; J=$U/object_trans_IMUHOI_vc_IMUHOI_10121941_omomo_ot_vc_imuhoi_ot_3branch_20261012u
run omomo_imuhoi3b_p00 omomo $U/config.yaml $J/best_human_pose.pt \
    $RC/imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008/best.pt \
    $J/best_object_trans.pt imuhoi imuhoi
U=$RC/vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose; J=$U/object_trans_v2_09301506_behave_ot_v2_armrot_biclose_20260930
run behave_v5ar_p00 behave $U/config.yaml $HP_BEHAVE $J/best_velocity_contact.pt $J/best_object_trans.pt v5 v2
U=$RC/v2_armrot_biclose_20260923_behave/v2_armrot_biclose; J=$U/object_trans_v2_vc_v4_09240321_behave_ot_v2_armrot_biclose_20260923
run behave_v4_p00 behave $U/config.yaml $HP_BEHAVE $J/best_velocity_contact.pt $J/best_object_trans.pt v4 v2
U=$RC/vc_ot_cross_ablation_behave_imuhoi_v2_armrot_biclose/v2_armrot_biclose; J=$U/object_trans_v2_vc_IMUHOI_09300537_behave_ot_v2_armrot_biclose_vc_ot_cross_ablation
run behave_imuhoivc_p00 behave $U/config.yaml $HP_BEHAVE $J/best_velocity_contact.pt $J/best_object_trans.pt imuhoi v2
U=$RC/imuhoi3b_20261008_behave/vc_imuhoi_ot_3branch; J=$U/object_trans_IMUHOI_vc_IMUHOI_10082127_behave_ot_vc_imuhoi_ot_3branch_20261008
run behave_imuhoi3b_p00 behave $U/config.yaml $J/best_human_pose.pt \
    $RC/imuhoi3b_20261008_behave/vc_imuhoi/velocity_contact_IMUHOI_10082052_behave_vc_imuhoi_20261008/best.pt \
    $J/best_object_trans.pt imuhoi imuhoi
echo "[$(date '+%F %T')] ALL DONE"
