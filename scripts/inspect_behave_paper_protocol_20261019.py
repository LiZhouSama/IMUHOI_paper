from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu
check_gpu(0)
import torch,yaml
from scripts.run_omomo_original_recipe_20261018 import original_config
EGO=ROOT.parent/'EgoIMU'
for split in ('train','test'):
    old=EGO/'process/processed_seg_data_BEHAVE'/split
    new=ROOT/'process/IMUHOI++cano/processed_seg_data_BEHAVE_full'/split
    a={p.name for p in old.glob('*.pt')}; b={p.name for p in new.glob('*.pt')}
    print(json.dumps({'split':split,'old':len(a),'new':len(b),'missing':sorted(a-b)[:30],'extra':sorted(b-a)[:30]}))
    for label,p in [('old',next(old.glob('*.pt'))),('new',next(new.glob('*.pt')))]:
        d=torch.load(p,map_location='cpu')
        print(label,p.name,[(k,list(v.shape) if torch.is_tensor(v) else str(v)[:140]) for k,v in d.items()])
class Loader(yaml.SafeLoader): pass
Loader.add_constructor('tag:yaml.org,2002:python/object/new:easydict.EasyDict',lambda l,n:l.construct_mapping(n,deep=True)['dictitems'])
saved=yaml.load((EGO/'outputs/IMUHOI/behave/training_config.yaml').read_text(),Loader=Loader)
print('ORIGINAL',json.dumps({k:saved[k] for k in ('seed','lr','weight_decay','train','test','staged_training') if k in saved}))
