#!/usr/bin/env python3
"""Gate weights and per-candidate HOI on GT-contact hand-frames, binned by frame index."""
import sys, numpy as np, torch
BINS=[(0,60),(60,120),(120,240),(240,10**9)]
def run(name,path):
    recs=torch.load(path,map_location='cpu')['records']
    acc={b:{'w':[], 'wown':[], 'hoi_pub':[], 'hoi_fk':[], 'hoi_imu':[], 'hoi_static':[]} for b in BINS}
    for r in recs:
        if not r.get('has_object',True): continue
        n=int(r['publish_len']); f=np.arange(n)
        w=r['gating_weights'][:n].float().numpy()
        gt=r['gt_obj_trans'][:n].double().numpy(); gpalm=r.get('gt_palm_pos_eval',r['gt_palm_pos'])[:n].double().numpy()
        palm=r['pred_palm_glb_pos'][:n].double().numpy()
        pub=r['pred_obj_trans_published'][:n].double().numpy()
        cands={'imu':r['pred_imu_obj_trans'][:n].double().numpy(),'static':r['pred_static_obj_trans'][:n].double().numpy()}
        fk=[r['pred_lhand_obj_trans'][:n].double().numpy(), r['pred_rhand_obj_trans'][:n].double().numpy()]
        for h,key in enumerate(('gt_lhand_contact','gt_rhand_contact')):
            m=r[key][:n].numpy().astype(bool)
            gvec=gt-gpalm[:,h]
            for b in BINS:
                sel=m&(f>=b[0])&(f<b[1])
                if not sel.any(): continue
                a=acc[b]
                a['w'].append(w[sel]); a['wown'].append(w[sel,h])
                a['hoi_pub'].append(np.linalg.norm((pub-palm[:,h])-gvec,axis=-1)[sel]*100)
                a['hoi_fk'].append(np.linalg.norm((fk[h]-palm[:,h])-gvec,axis=-1)[sel]*100)
                for k,c in cands.items(): a['hoi_'+k].append(np.linalg.norm((c-palm[:,h])-gvec,axis=-1)[sel]*100)
    print(f"\n== {name}")
    print("| bin | w FK_L FK_R IMU static | w(own FK) | HOI pub | HOI ownFK | HOI imu | HOI static |")
    for b in BINS:
        a=acc[b]; W=np.concatenate(a['w']).mean(0)
        print(f"| {b[0]}-{'' if b[1]>1e8 else b[1]} | {' '.join(f'{x:.3f}' for x in W)} | {np.concatenate(a['wown']).mean():.3f} | "
              + " | ".join(f"{np.concatenate(a[k]).mean():.2f}" for k in ('hoi_pub','hoi_fk','hoi_imu','hoi_static'))+" |")
for arg in sys.argv[1:]:
    n,p=arg.split('=',1); run(n,p)
