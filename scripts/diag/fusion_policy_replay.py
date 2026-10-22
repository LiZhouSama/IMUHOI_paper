#!/usr/bin/env python3
"""Open-loop replay of the causal 4-way publication under alternative gate policies, binned by frame index.
Candidates are the dumped ones (their window-init coupling to the original published track is kept as-is)."""
import sys, numpy as np, torch
BINS=[(0,60),(60,120),(120,240),(240,10**9)]
def publish(fk_l, fk_r, dimu, w, p0):
    prev=p0.copy(); out=np.zeros_like(fk_l)
    for t in range(len(w)):
        cur=w[t,0]*fk_l[t]+w[t,1]*fk_r[t]+w[t,2]*(prev+dimu[t])+w[t,3]*prev
        out[t]=cur; prev=cur
    return out
def policies(w, pc):
    out={'asis':w}
    s=(w[:,0]+w[:,1]).clip(1e-8); f=np.zeros_like(w); f[:,0]=w[:,0]/s; f[:,1]=w[:,1]/s; out['fk_only']=f
    # IMUHOI-style contact prior (beta = temperature -> weights ∝ w * prior)
    pl,pr,pm=pc[:,0],pc[:,1],pc[:,2]
    plc=np.clip(pl/np.clip(pm,1e-6,None),0,1); prc=np.clip(pr/np.clip(pm,1e-6,None),0,1)
    prior=np.stack([pl,pr,pm*(1-plc)*(1-prc),1-pm],-1).clip(1e-6); prior/=prior.sum(-1,keepdims=True)
    q=w*prior; out['x_prior']=q/q.sum(-1,keepdims=True)
    # hard hand anchoring: when a hand's predicted contact >0.5, move all IMU/static weight onto the contacting FK(s)
    c=np.stack([pl>0.5,pr>0.5],-1); any_c=c.any(-1)
    h=w.copy(); nc=c.sum(-1).clip(1,None)
    for k in (0,1): h[:,k]=np.where(any_c, np.where(c[:,k], 1.0/nc, 0.0), w[:,k])
    h[:,2]=np.where(any_c,0,w[:,2]); h[:,3]=np.where(any_c,0,w[:,3]); out['hard_anchor']=h
    # soft anchoring: a = max contact prob goes to the FKs in proportion to (pl, pr); the rest keeps the model's mix
    a=np.maximum(pl,pr); sh=(pl+pr).clip(1e-6); rest=w.copy(); rs=rest.sum(-1,keepdims=True).clip(1e-8); rest=rest/rs
    sft=(1-a)[:,None]*rest; sft[:,0]+=a*pl/sh; sft[:,1]+=a*pr/sh; out['soft_anchor']=sft
    return out
def run(name,path):
    recs=torch.load(path,map_location='cpu')['records']
    acc={}
    for r in recs:
        n=int(r['publish_len'])
        if not r.get('has_object',True) or n==0: continue
        hs=r.get('stage1_pred_palm_glb_pos',r['pred_palm_glb_pos'])[:n].double().numpy(); hr=r['pred_palm_glb_pos'][:n].double().numpy()
        fk=[hr[:,h]+(r[f'pred_{s}hand_obj_trans'][:n].double().numpy()-hs[:,h]) for h,s in ((0,'l'),(1,'r'))]
        dimu=r['pred_imu_obj_trans'][:n].double().numpy()-r['pred_static_obj_trans'][:n].double().numpy()
        w=r['gating_weights'][:n].double().numpy(); pc=r['pred_hand_contact_prob'][:n].double().numpy()
        gt=r['gt_obj_trans'][:n].double().numpy(); g=r.get('gt_palm_pos_eval',r['gt_palm_pos'])[:n].double().numpy()
        cl=r['gt_lhand_contact'][:n].numpy().astype(bool); cr=r['gt_rhand_contact'][:n].numpy().astype(bool); f=np.arange(n)
        p0=r['pred_static_obj_trans'][0].double().numpy()
        for pol,ww in policies(w,pc).items():
            p=publish(fk[0],fk[1],dimu,ww,p0)
            if pol=='asis':
                acc.setdefault('sanity',[]).append(np.abs(p-r['pred_obj_trans_published'][:n].double().numpy()).max()*100)
            oe=np.linalg.norm(p-gt,axis=-1)*100
            for b in BINS:
                sel=(f>=b[0])&(f<b[1])
                acc.setdefault((pol,'obj',b),[]).append(oe[sel])
                for h,m in ((0,cl),(1,cr)):
                    mm=m&sel
                    if mm.any(): acc.setdefault((pol,'hoi',b),[]).append(np.linalg.norm((p[mm]-hr[mm,h])-(gt[mm]-g[mm,h]),axis=-1)*100)
            acc.setdefault((pol,'obj','all'),[]).append(oe)
            if n>=3: acc.setdefault((pol,'jit'),[]).append(np.linalg.norm(p[2:]-2*p[1:-1]+p[:-2],axis=-1).mean()*1000)
            e=[np.linalg.norm((p[m]-hr[m,h])-(gt[m]-g[m,h]),axis=-1)*100 for h,m in ((0,cl),(1,cr)) if m.any()]
            if e: acc.setdefault((pol,'hoi','all'),[]).append(np.concatenate(e))
    print(f"\n== {name}  (asis replay max dev from published: {np.max(acc['sanity']):.3f} cm)")
    print("| policy | jitter | all obj/HOI | 0-59 | 60-119 | 120-239 | 240+ |")
    for pol in ('asis','fk_only','x_prior','hard_anchor','soft_anchor'):
        cells=[]
        for b in ['all']+BINS:
            cells.append(f"{np.concatenate(acc[(pol,'obj',b)]).mean():.2f} / {np.concatenate(acc[(pol,'hoi',b)]).mean():.2f}")
        print(f"| {pol} | {np.mean(acc[(pol,'jit')]):.2f} | "+" | ".join(cells)+" |")
for a in sys.argv[1:]:
    n,p=a.split('=',1); run(n,p)
