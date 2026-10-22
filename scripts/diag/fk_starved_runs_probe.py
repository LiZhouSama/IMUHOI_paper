#!/usr/bin/env python3
"""Characterise long FK-starved runs (total FK gate share < 0.05 for >= 60 frames) on GT-contact hand-frames.

    python scripts/diag/fk_starved_runs_probe.py ours_dump.pt reference_dump.pt   # reference supplies the other VC contact probs
"""
import sys, numpy as np, torch
from collections import defaultdict
ours=torch.load(sys.argv[1],map_location='cpu')['records']; ref={r['seq_name']:r for r in torch.load(sys.argv[2],map_location='cpu')['records']}
rows=defaultdict(list); per_obj=defaultdict(lambda:[0,0])
for r in ours:
    n=int(r['publish_len'])
    if not r.get('has_object',True) or n==0: continue
    o=ref.get(r['seq_name'])
    w=r['gating_weights'][:n].double().numpy(); k=w[:,0]+w[:,1]; st=k<0.05
    run=np.zeros(n)
    for t in range(n): run[t]=(run[t-1]+1 if t>0 else 1) if st[t] else 0
    pc=r['pred_hand_contact_prob'][:n].double().numpy(); pco=o['pred_hand_contact_prob'][:n].double().numpy() if o is not None else None
    gt=r['gt_obj_trans'][:n].double().numpy(); gspeed=np.zeros(n); gspeed[1:]=np.linalg.norm(gt[1:]-gt[:-1],axis=-1)*30*100
    pub=r['pred_obj_trans_published'][:n].double().numpy(); pspeed=np.zeros(n); pspeed[1:]=np.linalg.norm(pub[1:]-pub[:-1],axis=-1)*30*100
    gm=r['gt_obj_contact'][:n].numpy().astype(bool)
    for h,key in enumerate(('gt_lhand_contact','gt_rhand_contact')):
        m=r[key][:n].numpy().astype(bool)
        for nm,sel in (('long_starved',m&(run>=60)),('not_starved',m&(run==0))):
            if not sel.any(): continue
            rows[nm].append(np.stack([pc[sel,h],pc[sel,2],w[sel,2],w[sel,3],gspeed[sel],pspeed[sel],gm[sel].astype(float),
                                      pco[sel,h] if pco is not None else np.full(sel.sum(),np.nan), pco[sel,2] if pco is not None else np.full(sel.sum(),np.nan)],-1))
        per_obj[r['obj_name']][0]+=int((m&(run>=60)).sum()); per_obj[r['obj_name']][1]+=int(m.sum())
print("cols: p_hand(ours VC) | p_move(ours VC) | w_IMU | w_static | GT obj speed cm/s | published speed cm/s | GT obj-move label | p_hand(IMUHOI VC) | p_move(IMUHOI VC)")
for nm,v in rows.items():
    a=np.concatenate(v); print(f"{nm:13s} n={len(a):6d} | "+" | ".join(f"{x:.3f}" for x in np.nanmean(a,0)))
    print(f"{'':13s} p_hand(ours)>0.5: {100*(a[:,0]>0.5).mean():.1f}%   p_move(ours)>0.5: {100*(a[:,1]>0.5).mean():.1f}%   p_hand(IMUHOI)>0.5: {100*(a[:,7]>0.5).mean():.1f}%   p_move(IMUHOI)>0.5: {100*(a[:,8]>0.5).mean():.1f}%")
print("long-starved contact frames by object (share of that object's contact frames):")
for kname,(a,b) in sorted(per_obj.items(), key=lambda kv:-kv[1][0])[:8]: print(f"  {kname:18s} {a:6d} / {b:6d} = {100*a/max(b,1):.1f}%")
