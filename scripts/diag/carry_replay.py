#!/usr/bin/env python3
"""Open-loop replay: replace the IMU/static pull with a rigid hand-carry candidate on predicted-contact frames.
carry_h(t) = h_t + R_t R_{t-1}^T (p_{t-1} - h_{t-1});  contacting hands averaged."""
import sys, numpy as np, torch
from pytorch3d.transforms import rotation_6d_to_matrix
BINS=[(0,60),(60,120),(120,240),(240,10**9)]
def publish(fk, dimu, w, p0, hands, R, pc, mode, thr=0.5):
    prev=p0.copy(); out=np.zeros_like(fk[0]); n=len(w)
    for t in range(n):
        imu=prev+dimu[t]
        base=w[t,0]*fk[0][t]+w[t,1]*fk[1][t]
        if mode=='asis' or t==0:
            cur=base+w[t,2]*imu+w[t,3]*prev
        else:
            Rrel=R[t]@R[t-1].T
            carries=[hands[t,h]+Rrel@(prev-hands[t-1,h]) for h in (0,1)]
            pl,pr=pc[t,0],pc[t,1]
            if mode=='carry_hard':
                sel=[h for h,p in ((0,pl),(1,pr)) if p>thr]
                if sel:
                    c=np.mean([carries[h] for h in sel],0); cur=base+(w[t,2]+w[t,3])*c
                else: cur=base+w[t,2]*imu+w[t,3]*prev
            else:  # carry_soft: contact-prob-weighted carry vs the model's own imu/static mix
                a=max(pl,pr); s=max(pl+pr,1e-6)
                c=(pl*carries[0]+pr*carries[1])/s
                other=(w[t,2]*imu+w[t,3]*prev)/max(w[t,2]+w[t,3],1e-8)
                cur=base+(w[t,2]+w[t,3])*(a*c+(1-a)*other)
        out[t]=cur; prev=cur
    return out
def run(name,path):
    recs=torch.load(path,map_location='cpu')['records']; acc={}
    for r in recs:
        n=int(r['publish_len'])
        if not r.get('has_object',True) or n==0: continue
        hs=r.get('stage1_pred_palm_glb_pos',r['pred_palm_glb_pos'])[:n].double().numpy(); hr=r['pred_palm_glb_pos'][:n].double().numpy()
        fk=[hr[:,h]+(r[f'pred_{s}hand_obj_trans'][:n].double().numpy()-hs[:,h]) for h,s in ((0,'l'),(1,'r'))]
        dimu=r['pred_imu_obj_trans'][:n].double().numpy()-r['pred_static_obj_trans'][:n].double().numpy()
        w=r['gating_weights'][:n].double().numpy(); pc=r['pred_hand_contact_prob'][:n].double().numpy()
        R=rotation_6d_to_matrix(r['gt_obj_imu'][:n,3:9].double()).numpy()
        gt=r['gt_obj_trans'][:n].double().numpy(); g=r.get('gt_palm_pos_eval',r['gt_palm_pos'])[:n].double().numpy()
        cl=r['gt_lhand_contact'][:n].numpy().astype(bool); cr=r['gt_rhand_contact'][:n].numpy().astype(bool); f=np.arange(n)
        p0=r['pred_static_obj_trans'][0].double().numpy()
        for mode in ('asis','carry_hard','carry_soft'):
            p=publish(fk,dimu,w,p0,hr,R,pc,mode)
            oe=np.linalg.norm(p-gt,axis=-1)*100
            if n>=3: acc.setdefault((mode,'jit'),[]).append(np.linalg.norm(p[2:]-2*p[1:-1]+p[:-2],axis=-1).mean()*1000)
            for b in ['all']+BINS:
                sel=np.ones(n,bool) if b=='all' else (f>=b[0])&(f<b[1])
                acc.setdefault((mode,'obj',b),[]).append(oe[sel])
                for h,m in ((0,cl),(1,cr)):
                    mm=m&sel
                    if mm.any(): acc.setdefault((mode,'hoi',b),[]).append(np.linalg.norm((p[mm]-hr[mm,h])-(gt[mm]-g[mm,h]),axis=-1)*100)
    print(f"\n== {name}")
    print("| policy | jitter | all obj/HOI | 0-59 | 60-119 | 120-239 | 240+ |")
    for mode in ('asis','carry_hard','carry_soft'):
        cells=[f"{np.concatenate(acc[(mode,'obj',b)]).mean():.2f} / {np.concatenate(acc[(mode,'hoi',b)]).mean():.2f}" for b in ['all']+BINS]
        print(f"| {mode} | {np.mean(acc[(mode,'jit')]):.2f} | "+" | ".join(cells)+" |")
for a in sys.argv[1:]:
    n,p=a.split('=',1); run(n,p)
