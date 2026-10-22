"""Record the effective BEHAVE protocol before training starts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu,write_json,now
from scripts.run_behave_paper_aug_20261019 import OUT,read,verify
check_gpu(0)
import yaml
from train.rnn.train_object_trans import _joint_trainable_modules
protocol=read(OUT/'protocol.json');verify(protocol,full=True)
configs={k:yaml.safe_load(Path(p).read_text()) for k,p in protocol['configs'].items()}
for name,c in configs.items():
    assert c['train_datasets']==['behave'] and c['windows_per_sequence']==1
    assert c['train_data_window']==c['test_data_window']==120
    assert c['seed']==10 and c['epochs']==300
    assert c['eval_hoi_protocol']=='egoimu'
    assert all(c[k]==.1 for k in ('imu_acc_noise_std','imu_rot_noise_std','obj_imu_acc_noise_std','obj_imu_rot_noise_std'))
    assert c['loss_weights']['obj_vel']==c['loss_weights']['hand_vel']==10
    assert c['v5_init_noise_prob']==.5 and c['v5_obj_vel_init_noise_std']==c['v5_hand_vel_init_noise_std']==.2
    assert c['body_model_path']=='datasets/smpl_models/smplh/neutral/model.npz'
vc=configs['proposed_vc'];joint=configs['proposed_joint']
assert vc['velocity_contact_arch']=='v5' and vc['vc_staged_training'] and vc['vc_velocity_stage_epochs']==120
assert joint['object_trans_arch']=='v2' and joint['object_trans_forearm_obj_rotation'] and joint['object_trans_bimanual_closure']
assert _joint_trainable_modules('v2',True)==(False,True)
audit=read(OUT/'data_audit.json')
assert audit['counts']['test']['all']==387 and audit['max_pose_rotation_6d_error']<1e-4
from dataset.dataset_IMUHOI import IMUDataset
full=IMUDataset(data_dir=str(OUT/'data/eval'),window_size=120,debug=False,min_obj_contact_frames=0,full_sequence=True,require_canonical_skeleton=True,resolve_bimanual_contact_conflicts=False)
assert len(full)==340
write_json(OUT/'recipe_audit.json',{'status':'passed','updated_at':now(),'data_counts':audit['counts'],'frame_alignment_max_error':audit['max_pose_rotation_6d_error'],'original_common_loss_weights_match':True,'original_checkpoint_reuse':False,'model':'vc-v5ar + ot-v2armbic','augmentation':{'probability':.5,'hand_object_std_mps':.2},'vc_curriculum':{'velocity_epochs':120,'contact_epochs':180},'joint_trainable':['vc','ot'],'joint_frozen':['hp'],'common_loss_weights':joint['loss_weights'],'budget':protocol['budget'],'evaluation_members':340,'canonical_object_preprocessing_max_difference_m':audit['max_object_trajectory_error_m'],'limitations':protocol['remaining_differences']})
print('BEHAVE recipe/data audit passed')

