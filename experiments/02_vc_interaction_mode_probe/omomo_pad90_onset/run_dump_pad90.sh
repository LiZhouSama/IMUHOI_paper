#!/usr/bin/env bash
# OMOMO test sequences not in contact at frame 1, with 90 static frames prepended (make_padded_testset.py):
# features of IMUHOI++ (with anticipation head outputs) and IMUHOI, same pipelines and 7/3 protocol as run_dump.sh;
# then a noise-free junction check on sequences whose first contact starts at real frame >= 90.
set -u
ROOT=/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
PY=/home/l/anaconda3/envs/SAGE/bin/python
RC=$ROOT/outputs/IMUHOI++/RNN_cano
OUT=$RC/vc_probe_pilot_20261027
cd "$ROOT"
PAD=$OUT/omomo_pad90_data/test
SEQ_PAD=$(for f in $PAD/*.pt; do printf -- "--sequence_path %s " "$f"; done)
U=$RC/vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose; J=$U/object_trans_v2_10020305_omomo_ot_v2_armrot_biclose_20261002u
OURS="--config $U/config.yaml --dataset omomo --hp_ckpt $RC/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt \
--vc_ckpt $J/best_velocity_contact.pt --ot_ckpt $J/best_object_trans.pt --velocity_contact_arch v5 --object_trans_arch v2 \
--seed 42 --num_workers 2 --online_batch_size 48"
U=$RC/imuhoi3b_20261012_omomo_unified/vc_imuhoi_ot_3branch; J=$U/object_trans_IMUHOI_vc_IMUHOI_10121941_omomo_ot_vc_imuhoi_ot_3branch_20261012u
IMU="--config $U/config.yaml --dataset omomo --hp_ckpt $J/best_human_pose.pt \
--vc_ckpt $RC/imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008/best.pt \
--ot_ckpt $J/best_object_trans.pt --velocity_contact_arch imuhoi --object_trans_arch imuhoi --seed 42 --num_workers 2 --online_batch_size 48"
NOISE="--simulate_imu_noise --simulate_imu_noise_seed 42"

echo "[$(date '+%F %T')] START pad90_ours"
$PY scripts/vc_probe/dump_vc_features.py --feature_dir $OUT/omomo_pad90_ours/features --random_control -- $OURS $NOISE \
    --output_json $OUT/omomo_pad90_ours/eval.json $SEQ_PAD > $OUT/omomo_pad90_ours.log 2>&1
echo "[$(date '+%F %T')] END pad90_ours rc=$?"
echo "[$(date '+%F %T')] START pad90_imuhoi"
$PY scripts/vc_probe/dump_vc_features.py --feature_dir $OUT/omomo_pad90_imuhoi/features -- $IMU $NOISE \
    --output_json $OUT/omomo_pad90_imuhoi/eval.json $SEQ_PAD > $OUT/omomo_pad90_imuhoi.log 2>&1
echo "[$(date '+%F %T')] END pad90_imuhoi rc=$?"

# Junction check (no IMU noise, so both runs see identical inputs on the real frames).
LATE=$($PY - <<'PYEOF'
import json
segs = json.load(open("/mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI/outputs/IMUHOI++/RNN_cano/vc_probe_pilot_20261027/labels_omomo_test.json"))
first = {}
for s in segs:
    first[s["file"]] = min(first.get(s["file"], 10 ** 9), s["a"])
print(" ".join(f for f, a in sorted(first.items()) if a >= 90))
PYEOF
)
SEQ_O=""; SEQ_P=""
for f in $LATE; do SEQ_O="$SEQ_O --sequence_path $ROOT/process/IMUHOI++cano/processed_split_data_OMOMO_full/test/$f.pt"; SEQ_P="$SEQ_P --sequence_path $PAD/$f.pt"; done
echo "[$(date '+%F %T')] START junction check on $(echo $LATE | wc -w) sequences"
$PY scripts/vc_probe/dump_vc_features.py --feature_dir $OUT/omomo_junction_orig/probs --probs_only -- $OURS \
    --output_json $OUT/omomo_junction_orig/eval.json $SEQ_O > $OUT/omomo_junction_orig.log 2>&1
$PY scripts/vc_probe/dump_vc_features.py --feature_dir $OUT/omomo_junction_pad/probs --probs_only -- $OURS \
    --output_json $OUT/omomo_junction_pad/eval.json $SEQ_P > $OUT/omomo_junction_pad.log 2>&1
echo "[$(date '+%F %T')] ALL DONE"
