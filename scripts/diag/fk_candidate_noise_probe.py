#!/usr/bin/env python3
"""Per-frame noise of the FK candidate on GT-contact frames: second difference (same unit as obj_jitter),
for the FK candidate, the predicted palm, the published object and the GT object; plus bias-vs-noise split of the FK error."""
import sys, numpy as np, torch
def run(name,path):
    recs=torch.load(path,map_location='cpu')['records']; acc={k:[] for k in ('fk','palm','pub','gt','bone','err_total','err_smooth','len')}
    for r in recs:
        n=int(r['publish_len'])
        if not r.get('has_object',True) or n<5: continue
        gt=r['gt_obj_trans'][:n].double().numpy(); palm=r['pred_palm_glb_pos'][:n].double().numpy()
        hs=r.get('stage1_pred_palm_glb_pos',r['pred_palm_glb_pos'])[:n].double().numpy(); pub=r['pred_obj_trans_published'][:n].double().numpy()
        g=r.get('gt_palm_pos_eval',r['gt_palm_pos'])[:n].double().numpy()
        for h,(s,key) in enumerate((('l','gt_lhand_contact'),('r','gt_rhand_contact'))):
            m=r[key][:n].numpy().astype(bool); c=m[2:]&m[1:-1]&m[:-2]
            if not c.any(): continue
            bone=r[f'pred_{s}hand_obj_trans'][:n].double().numpy()-hs[:,h]; fk=palm[:,h]+bone
            sd=lambda x: np.linalg.norm(x[2:]-2*x[1:-1]+x[:-2],axis=-1)[c]*1000
            for k,x in (('fk',fk),('palm',palm[:,h]),('pub',pub),('gt',gt),('bone',bone)): acc[k].append(sd(x))
            e=(fk-palm[:,h])-(gt-g[:,h])   # bone error vector
            # smooth part = 15-frame centred moving average of the error vector; noise part = remainder
            ker=np.ones(15)/15; es=np.stack([np.convolve(e[:,d],ker,mode='same') for d in range(3)],-1)
            acc['err_total'].append(np.linalg.norm(e[m],axis=-1)*100); acc['err_smooth'].append(np.linalg.norm((e-es)[m],axis=-1)*100)
            acc['len'].append(np.linalg.norm((gt-g[:,h])[m],axis=-1)*100)
    c=lambda k: np.concatenate(acc[k]).mean()
    print(f"| {name} | {c('fk'):.1f} | {c('bone'):.1f} | {c('palm'):.1f} | {c('pub'):.1f} | {c('gt'):.1f} | {c('err_total'):.2f} | {c('err_smooth'):.2f} | {c('len'):.1f} |")
print("| model | FK cand jitter | bone jitter | pred palm jitter | published jitter | GT obj jitter | FK HOI | FK HOI high-freq part (<0.5 s) | GT hand→obj distance cm |\n|---|---|---|---|---|---|---|---|---|")
for a in sys.argv[1:]: run(*a.split('=',1))
