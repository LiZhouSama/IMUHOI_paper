"""Remove derivative caches before freezing the rebuilt paper crops."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu,write_json,sha,now
from scripts.run_omomo_original_recipe_20261018 import read
OUT=ROOT/'outputs/IMUHOI++/paper_recipe_aug_seed10_behave_20261019'
check_gpu(0)
import torch
assert not (OUT/'protocol.json').exists(), 'Dataset is already frozen'
audit=read(OUT/'data_audit.json'); inventory=read(OUT/'data/inventory.json')
keys=('human_imu','obj_imu','ori_root_reduced','root_vel','obj_vel','sensor_vel_root','sensor_vel_glb','obj_rot_6d')
for split,rows in inventory.items():
    for row in rows:
        path=OUT/'data'/split/row['name'];data=torch.load(path,map_location='cpu')
        for key in keys:data.pop(key,None)
        torch.save(data,path);stat=path.stat();row.update(sha256=sha(path),stat=[stat.st_size,stat.st_mtime_ns])
audit.update(removed_derivative_caches=list(keys),derivative_policy='Recompute after crop, including complete test crops; original ordering',max_pose_rotation_6d_error=max(r['pose_rotation_6d_max_error'] for r in audit['rows']),finalized_at=now())
write_json(OUT/'data/inventory.json',inventory);write_json(OUT/'data_audit.json',audit)
print({'counts':audit['counts'],'max_pose_rotation_error':audit['max_pose_rotation_6d_error'],'max_object_preprocessing_difference_m':audit['max_object_trajectory_error_m']})
