"""Select original full-evaluation members without filtering canonical labels again."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu,write_json,now
from scripts.run_omomo_original_recipe_20261018 import read
OUT=ROOT/'outputs/IMUHOI++/paper_recipe_aug_seed10_behave_20261019'
check_gpu(0)
audit=read(OUT/'data_audit.json');inventory=read(OUT/'data/inventory.json')
names=sorted(r['name'] for r in audit['rows'] if r['split']=='test' and r['frames']>=1 and r['original_contact_frames']>=10)
assert set(names)==set(audit['members']['test']) and len(names)==340
folder=OUT/'data/eval';folder.mkdir(parents=True,exist_ok=True)
for name in names:
    path=folder/name
    if not path.exists():os.link(OUT/'data/test'/name,path)
assert {p.name for p in folder.glob('*.pt')}==set(names)
inventory['eval']=[r for r in inventory['test'] if r['name'] in names]
write_json(OUT/'data/inventory.json',inventory)
write_json(OUT/'evaluation_members.json',{'created_at':now(),'primary_count':340,'raw_test_count':387,'members':names,'excluded_original_low_motion':[r['name'] for r in audit['rows'] if r['split']=='test' and r['name'] not in names],'reason':'EgoIMU evaluator full_sequence=True retains original IMUDataset min_obj_contact_frames=10 default; do not re-filter current labels','directory':str(folder)})
print('Original full-evaluation membership: 340/387')
