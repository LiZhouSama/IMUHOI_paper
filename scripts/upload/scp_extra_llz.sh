#!/bin/bash
# Supplement to scp_data_ckpt_llz.sh: the IMU-HOI 3-branch baseline (filelist_ckpt_imuhoi3b.txt), the OT ablation
# plan and the upload manifest / verifier -> llz@202.120.59.70:/disk2/mmzhou/IMUHOI (same layout as the repo root).
set -e
cd /mnt/d/a_WORK/Projects/PhD/tasks/IMUHOI
R=llz@202.120.59.70
D=/disk2/mmzhou/IMUHOI
O="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-%r@%h:%p -o ControlPersist=30m"

# 1. parent directories
ssh $O $R "mkdir -p \
  $D/docs \
  $D/outputs/IMUHOI++/RNN_cano \
  $D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave \
  $D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/vc_imuhoi/velocity_contact_IMUHOI_10082052_behave_vc_imuhoi_20261008 \
  $D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008 \
  $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b \
  $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101856_hodome_vc_imuhoi_remaining3_20261010 \
  $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b \
  $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10100654_imhd_vc_imuhoi_remaining3_20261010 \
  $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b \
  $D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101407_pahoi_vc_imuhoi_remaining3_20261010 \
  $D/scripts \
"

# 2. files and directories
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_omomo/vc_imuhoi/velocity_contact_IMUHOI_10081945_omomo_vc_imuhoi_20261008/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261012_omomo_unified $R:$D/outputs/IMUHOI++/RNN_cano/
scp $O -rp outputs/IMUHOI++/RNN_cano/chain_imuhoi3b_20261012_omomo_unified.sh $R:$D/outputs/IMUHOI++/RNN_cano/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/experiment_manifest.json $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/hp_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/vc_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/shared_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/summary.md $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/vc_imuhoi/velocity_contact_IMUHOI_10082052_behave_vc_imuhoi_20261008/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/vc_imuhoi/velocity_contact_IMUHOI_10082052_behave_vc_imuhoi_20261008/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/vc_imuhoi/velocity_contact_IMUHOI_10082052_behave_vc_imuhoi_20261008/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/vc_imuhoi/velocity_contact_IMUHOI_10082052_behave_vc_imuhoi_20261008/
scp $O -rp outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/vc_imuhoi_ot_3branch $R:$D/outputs/IMUHOI++/RNN_cano/imuhoi3b_20261008_behave/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/experiment_manifest.json $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/hp_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/vc_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/shared_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/summary.md $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10100654_imhd_vc_imuhoi_remaining3_20261010/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10100654_imhd_vc_imuhoi_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10100654_imhd_vc_imuhoi_remaining3_20261010/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10100654_imhd_vc_imuhoi_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/vc_imuhoi_ot_3branch $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/imhd/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/experiment_manifest.json $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/hp_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/vc_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/shared_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/summary.md $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101407_pahoi_vc_imuhoi_remaining3_20261010/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101407_pahoi_vc_imuhoi_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101407_pahoi_vc_imuhoi_remaining3_20261010/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101407_pahoi_vc_imuhoi_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/vc_imuhoi_ot_3branch $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/pahoi/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/experiment_manifest.json $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/hp_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/vc_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/shared_config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/summary.md $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101856_hodome_vc_imuhoi_remaining3_20261010/best.pt $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101856_hodome_vc_imuhoi_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101856_hodome_vc_imuhoi_remaining3_20261010/config.yaml $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/vc_imuhoi/velocity_contact_IMUHOI_10101856_hodome_vc_imuhoi_remaining3_20261010/
scp $O -rp outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/vc_imuhoi_ot_3branch $R:$D/outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010/hodome/imuhoi3b/
scp $O -rp docs/ot_ablation_plan_20261102.md $R:$D/docs/
scp $O -rp scripts/upload $R:$D/scripts/
