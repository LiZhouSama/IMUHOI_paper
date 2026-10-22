#!/usr/bin/env python3
"""Is the long-horizon gap the IMU branch's velocity?  Replay our causal publication with the IMU step
taken from (a) our VC velocity [as-is], (b) the IMUHOI VC velocity (same test sequences), (c) GT velocity.
Also report per-model velocity error / bias on all frames by frame bin."""
import sys, numpy as np, torch
BINS=[(0,60),(60,120),(120,240),(240,10**9)]
DT=1/30
def publish(fk,dimu,w,p0):
    prev=p0.copy(); out=np.zeros_like(fk[0])
    for t in range(len(w)):
        cur=w[t,0]*fk[0][t]+w[t,1]*fk[1][t]+w[t,2]*(prev+dimu[t])+w[t,3]*prev; out[t]=cur; prev=cur
    return out
ours=sys.argv[1]; other=sys.argv[2]
ref={r['seq_name']:r for r in torch.load(other,map_location='cpu')['records']}
for name,path in (('ours',ours),):
    recs=torch.load(path,map_location='cpu')['records']; acc={}
    for r in recs:
        n=int(r['publish_len'])
        if not r.get('has_object',True) or n==0 or r['seq_name'] not in ref: continue
        o=ref[r['seq_name']]
        hs=r.get('stage1_pred_palm_glb_pos',r['pred_palm_glb_pos'])[:n].double().numpy(); hr=r['pred_palm_glb_pos'][:n].double().numpy()
        fk=[hr[:,h]+(r[f'pred_{s}hand_obj_trans'][:n].double().numpy()-hs[:,h]) for h,s in ((0,'l'),(1,'r'))]
        d_ours=r['pred_imu_obj_trans'][:n].double().numpy()-r['pred_static_obj_trans'][:n].double().numpy()
        gt=r['gt_obj_trans'][:n].double().numpy()
        v_ours=r['pred_obj_vel'][:n].double().numpy(); v_oth=o['pred_obj_vel'][:n].double().numpy()
        # sanity: IMU step == v*dt ?
        acc.setdefault('step_dev',[]).append(np.abs(d_ours-v_ours*DT).max()*100)
        d_gt=np.zeros_like(gt); d_gt[1:]=gt[1:]-gt[:-1]
        w=r['gating_weights'][:n].double().numpy(); p0=r['pred_static_obj_trans'][0].double().numpy()
        g=r.get('gt_palm_pos_eval',r['gt_palm_pos'])[:n].double().numpy()
        cl=r['gt_lhand_contact'][:n].numpy().astype(bool); cr=r['gt_rhand_contact'][:n].numpy().astype(bool); f=np.arange(n)
        vgt=d_gt/DT
        for vn,v in (('ours_vc',v_ours),('imuhoi_vc',v_oth)):
            for b in BINS:
                sel=(f>=b[0])&(f<b[1])
                acc.setdefault((vn,'verr',b),[]).append(np.linalg.norm(v[sel]-vgt[sel],axis=-1)*100)
                acc.setdefault((vn,'vbias',b),[]).append((v[sel]-vgt[sel]))
        for mode,d in (('asis',d_ours),('imuhoi_vel',v_oth*DT),('gt_vel',d_gt)):
            p=publish(fk,d,w,p0); oe=np.linalg.norm(p-gt,axis=-1)*100
            acc.setdefault((mode,'jit'),[]).append(np.linalg.norm(p[2:]-2*p[1:-1]+p[:-2],axis=-1).mean()*1000)
            for b in ['all']+BINS:
                sel=np.ones(n,bool) if b=='all' else (f>=b[0])&(f<b[1])
                acc.setdefault((mode,'obj',b),[]).append(oe[sel])
                for h,m in ((0,cl),(1,cr)):
                    mm=m&sel
                    if mm.any(): acc.setdefault((mode,'hoi',b),[]).append(np.linalg.norm((p[mm]-hr[mm,h])-(gt[mm]-g[mm,h]),axis=-1)*100)
    print(f"== {path.split('/')[-1]}  IMU-step vs v*dt max dev {np.max(acc['step_dev']):.4f} cm")
    print("| IMU step from | jitter | all obj/HOI | 0-59 | 60-119 | 120-239 | 240+ |")
    for mode in ('asis','imuhoi_vel','gt_vel'):
        cells=[f"{np.concatenate(acc[(mode,'obj',b)]).mean():.2f} / {np.concatenate(acc[(mode,'hoi',b)]).mean():.2f}" for b in ['all']+BINS]
        print(f"| {mode} | {np.mean(acc[(mode,'jit')]):.2f} | "+" | ".join(cells)+" |")
    print("velocity error |v-v_gt| cm/s  (mean bias norm cm/s) by bin")
    for vn in ('ours_vc','imuhoi_vc'):
        print(vn, " | ".join(f"{np.concatenate(acc[(vn,'verr',b)]).mean():.2f} ({np.linalg.norm(np.concatenate(acc[(vn,'vbias',b)]).mean(0))*100:.2f})" for b in BINS))
