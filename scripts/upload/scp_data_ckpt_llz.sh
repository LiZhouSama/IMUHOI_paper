#!/bin/bash
# Data list + checkpoint list -> llz@202.120.59.70:/disk2/mmzhou/IMUHOI (same layout as the local repo root).
# Run from the local repo root. One SSH connection is reused (password asked once).
set -e
cd /mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
R=llz@202.120.59.70
D=/disk2/mmzhou/IMUHOI
O="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m"

# 1. parent directories
# ssh $O $R "mkdir -p \
#   $D/outputs/IMUHOI++/RNN_cano \
#   $D/outputs/IMUHOI++/RNN_cano/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911 \
#   $D/outputs/IMUHOI++/RNN_cano/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010 \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/hp/human_pose_10101550_hodome_hp_remaining3_20261010 \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/vc_v5/velocity_contact_v5_10101621_hodome_vc_v5_remaining3_20261010 \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/hp/human_pose_10100114_imhd_hp_remaining3_20261010 \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/vc_v5/velocity_contact_v5_10100229_imhd_vc_v5_remaining3_20261010 \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/hp/human_pose_10100950_pahoi_hp_remaining3_20261010 \
#   $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/vc_v5/velocity_contact_v5_10101041_pahoi_vc_v5_remaining3_20261010 \
#   $D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930 \
#   $D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/behave_v5ar/velocity_contact_v5_09301147_behave_vc_v5ar_20260930 \
#   $D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/omomo_v5ar/velocity_contact_v5_09301324_omomo_vc_v5ar_20260930 \
#   $D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/protocol_1002 \
#   $D/process/IMUHOI++cano/processed_seg_data_BEHAVE_full \
#   $D/process/IMUHOI++cano/processed_seg_data_HODOME_full \
#   $D/process/IMUHOI++cano/processed_seg_data_IMHD_full \
#   $D/process/IMUHOI++cano/processed_split_data_BEHAVE \
#   $D/process/IMUHOI++cano/processed_split_data_HODOME \
#   $D/process/IMUHOI++cano/processed_split_data_IMHD \
#   $D/process/IMUHOI++cano/processed_split_data_OMOMO_full \
#   $D/process/IMUHOI++cano/processed_split_data_PAHOI_full \
# "

# 2. files and directories
# scp $O -rp process/IMUHOI++cano/processed_split_data_OMOMO_full/train $R:$D/process/IMUHOI++cano/processed_split_data_OMOMO_full/
# scp $O -rp process/IMUHOI++cano/processed_split_data_OMOMO_full/test $R:$D/process/IMUHOI++cano/processed_split_data_OMOMO_full/
# scp $O -rp process/IMUHOI++cano/processed_seg_data_BEHAVE_full/train $R:$D/process/IMUHOI++cano/processed_seg_data_BEHAVE_full/
# scp $O -rp process/IMUHOI++cano/processed_seg_data_BEHAVE_full/test $R:$D/process/IMUHOI++cano/processed_seg_data_BEHAVE_full/
# scp $O -rp process/IMUHOI++cano/processed_split_data_BEHAVE/test $R:$D/process/IMUHOI++cano/processed_split_data_BEHAVE/
# scp $O -rp process/IMUHOI++cano/processed_seg_data_IMHD_full/train $R:$D/process/IMUHOI++cano/processed_seg_data_IMHD_full/
# scp $O -rp process/IMUHOI++cano/processed_seg_data_IMHD_full/test $R:$D/process/IMUHOI++cano/processed_seg_data_IMHD_full/
# scp $O -rp process/IMUHOI++cano/processed_split_data_IMHD/test $R:$D/process/IMUHOI++cano/processed_split_data_IMHD/
# scp $O -rp process/IMUHOI++cano/processed_split_data_PAHOI_full/train $R:$D/process/IMUHOI++cano/processed_split_data_PAHOI_full/
# scp $O -rp process/IMUHOI++cano/processed_split_data_PAHOI_full/test $R:$D/process/IMUHOI++cano/processed_split_data_PAHOI_full/
# scp $O -rp process/IMUHOI++cano/processed_seg_data_HODOME_full/train $R:$D/process/IMUHOI++cano/processed_seg_data_HODOME_full/
# scp $O -rp process/IMUHOI++cano/processed_seg_data_HODOME_full/test $R:$D/process/IMUHOI++cano/processed_seg_data_HODOME_full/
# scp $O -rp process/IMUHOI++cano/processed_split_data_HODOME/test $R:$D/process/IMUHOI++cano/processed_split_data_HODOME/
scp $O -rp outputs/IMUHOI++/RNN_cano/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/
scp $O -rp outputs/IMUHOI++/RNN_cano/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/omomo_cano_20260907/hp/human_pose_09072119_omomo_cano/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930/omomo_v5ar/velocity_contact_v5_09301324_omomo_vc_v5ar_20260930/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/omomo_v5ar/velocity_contact_v5_09301324_omomo_vc_v5ar_20260930/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930/omomo_v5ar/velocity_contact_v5_09301324_omomo_vc_v5ar_20260930/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/omomo_v5ar/velocity_contact_v5_09301324_omomo_vc_v5ar_20260930/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930/vc_config_omomo_v5ar.yaml $R:$D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_omomo_v5ar_unified $R:$D/outputs/IMUHOI++/RNN_cano/
scp $O -rp outputs/IMUHOI++/RNN_cano/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/
scp $O -rp outputs/IMUHOI++/RNN_cano/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/behave_full_trust_20260911/hp/human_pose_09111505_behave_hp_20260911/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930/behave_v5ar/velocity_contact_v5_09301147_behave_vc_v5ar_20260930/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/behave_v5ar/velocity_contact_v5_09301147_behave_vc_v5ar_20260930/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930/behave_v5ar/velocity_contact_v5_09301147_behave_vc_v5ar_20260930/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/behave_v5ar/velocity_contact_v5_09301147_behave_vc_v5ar_20260930/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930/vc_config_behave_v5ar.yaml $R:$D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_behave_v5ar $R:$D/outputs/IMUHOI++/RNN_cano/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930/train_vc.sh $R:$D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930/run_ot.sh $R:$D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/
scp $O -rp outputs/IMUHOI++/RNN_cano/vc_v5_20260930/protocol_1002/run_ot_omomo_v5ar_unified.sh $R:$D/outputs/IMUHOI++/RNN_cano/vc_v5_20260930/protocol_1002/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/recipe.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/provenance.json $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/summary.md $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/experiment_manifest.json $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/hp_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/vc_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/shared_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/summary.md $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/hp/human_pose_10100114_imhd_hp_remaining3_20261010/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/hp/human_pose_10100114_imhd_hp_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/hp/human_pose_10100114_imhd_hp_remaining3_20261010/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/hp/human_pose_10100114_imhd_hp_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/vc_v5/velocity_contact_v5_10100229_imhd_vc_v5_remaining3_20261010/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/vc_v5/velocity_contact_v5_10100229_imhd_vc_v5_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/vc_v5/velocity_contact_v5_10100229_imhd_vc_v5_remaining3_20261010/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/vc_v5/velocity_contact_v5_10100229_imhd_vc_v5_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/v2_armrot_biclose $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/experiment_manifest.json $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/hp_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/vc_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/shared_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/summary.md $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/hp/human_pose_10100950_pahoi_hp_remaining3_20261010/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/hp/human_pose_10100950_pahoi_hp_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/hp/human_pose_10100950_pahoi_hp_remaining3_20261010/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/hp/human_pose_10100950_pahoi_hp_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/vc_v5/velocity_contact_v5_10101041_pahoi_vc_v5_remaining3_20261010/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/vc_v5/velocity_contact_v5_10101041_pahoi_vc_v5_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/vc_v5/velocity_contact_v5_10101041_pahoi_vc_v5_remaining3_20261010/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/vc_v5/velocity_contact_v5_10101041_pahoi_vc_v5_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/v2_armrot_biclose $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/experiment_manifest.json $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/hp_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/vc_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/shared_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/summary.md $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/hp/human_pose_10101550_hodome_hp_remaining3_20261010/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/hp/human_pose_10101550_hodome_hp_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/hp/human_pose_10101550_hodome_hp_remaining3_20261010/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/hp/human_pose_10101550_hodome_hp_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/vc_v5/velocity_contact_v5_10101621_hodome_vc_v5_remaining3_20261010/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/vc_v5/velocity_contact_v5_10101621_hodome_vc_v5_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/vc_v5/velocity_contact_v5_10101621_hodome_vc_v5_remaining3_20261010/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/vc_v5/velocity_contact_v5_10101621_hodome_vc_v5_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/v2_armrot_biclose $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/proposed/
