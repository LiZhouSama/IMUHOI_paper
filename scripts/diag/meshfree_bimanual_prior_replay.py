"""Train-side geometry priors and mesh-free correction replay on recorded predictions.

Only category, predicted palms/contact, and recorded IMU enter the correction.
GT objects/hands/contacts enter training targets or evaluation, never inference.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
DATA={"OMOMO":"processed_split_data_OMOMO_full", "BEHAVE":"processed_split_data_BEHAVE"}
TEST={"OMOMO":"outputs/IMUHOI++/RNN_cano/relabel_20260918_omomo/dumps/cur_hoi_p800_omomo_online.pt",
      "BEHAVE":"outputs/IMUHOI++/RNN_cano/behave_hpsmooth_vcinit_20260922/dump_v2_retrained_final.pt"}
CAL={"OMOMO":"outputs/analysis/meshfree_bimanual_20260930/omomo_train_calibration.pt",
     "BEHAVE":"outputs/IMUHOI++/RNN_cano/behave_hpsmooth_vcinit_20260922/v2_train_calibration32.pt"}
RODS={"OMOMO":{"mop","floorlamp","clothesstand","tripod","vacuum"},"BEHAVE":{"boxlong","yogamat"}}


def array(v):
    return np.asarray(v,dtype=np.float64)


def rot6(x):
    a,b=x[...,:3],x[...,3:6]
    a=a/np.linalg.norm(a,axis=-1,keepdims=True).clip(1e-10)
    b=b-(a*b).sum(-1,keepdims=True)*a
    b=b/np.linalg.norm(b,axis=-1,keepdims=True).clip(1e-10)
    return np.stack((a,b,np.cross(a,b)),axis=-2)


def features(hands,obj_imu,human_imu):
    rot=rot6(obj_imu[:,3:9])
    root=rot6(human_imu[:,0,3:9])
    arms=root[:,None]@rot6(human_imu[:,[4,5],3:9])
    rel=rot.transpose(0,2,1)[:,None]@arms
    d=np.einsum("tji,tj->ti",rot,hands[:,1]-hands[:,0])
    x=np.concatenate((d,np.linalg.norm(d,axis=-1,keepdims=True),rel[:,:,:2,:].reshape(len(d),12),rot[:,1,:]),-1)
    return x,rot


def csv_write(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def geom_stats(s):
    v=np.abs(s)*100
    return {"frames":len(v),"mean_cm":float(v.mean()),"median_cm":float(np.median(v)),
            "p90_cm":float(np.percentile(v,90)),"p95_cm":float(np.percentile(v,95)),
            **{f"within_{t}cm_pct":float((v<=t).mean()*100) for t in (2,5,10,20)}}


def load_training(dataset,out,cal_files):
    cache=out/f"{dataset}_training_geometry.pt"
    if cache.exists():
        return torch.load(cache,map_location="cpu",weights_only=False)
    roots=ROOT/"process/IMUHOI++cano"/DATA[dataset]
    records=[]
    paths=sorted((roots/"train").glob("*.pt"))
    for i,p in enumerate(paths):
        d=torch.load(p,map_location="cpu",weights_only=False)
        mask=np.asarray(d["lhand_contact"],bool)&np.asarray(d["rhand_contact"],bool)
        if not mask.any(): continue
        hands=array(d["position_global_full_gt_world"])[mask][:,[22,23]]
        x,R=features(hands,array(d["obj_imu"])[mask],array(d["human_imu"])[mask])
        c=np.einsum("tji,tj->ti",R,array(d["obj_trans"])[mask].reshape(-1,3)-hands.mean(1))
        source_rot=array(d["obj_rot"])[mask]
        np.testing.assert_allclose(R,source_rot,atol=2e-5,rtol=2e-5)
        name=str(d["obj_name"])
        split="cal" if p.name in cal_files else ("validation" if int(hashlib.sha256(p.name.encode()).hexdigest()[:8],16)%5==0 else "fit")
        records.append({"file":p.name,"object":name,"split":split,"x":x.astype(np.float32),"c":c.astype(np.float32)})
        if (i+1)%1000==0: print(dataset,"training geometry",i+1,len(paths),flush=True)
    torch.save(records,cache)
    return records


class Prior(nn.Module):
    def __init__(self,names):
        super().__init__()
        self.embedding=nn.Embedding(len(names),12)
        self.net=nn.Sequential(nn.Linear(31,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3))
    def forward(self,x,k):
        return self.net(torch.cat((x,self.embedding(k)),dim=-1))


def fit_prior(records,names,out,dataset,device):
    rng=np.random.RandomState(930)
    idx={n:i for i,n in enumerate(names)}
    packs={s:[] for s in ("fit","validation")}
    for r in records:
        if r["split"] not in packs: continue
        # Bound the dominance of long static holds without using test information.
        ii=np.linspace(0,len(r["x"])-1,min(200,len(r["x"])),dtype=int)
        packs[r["split"]].append((r["x"][ii],r["c"][ii],np.full(len(ii),idx[r["object"]])))
    def combine(s):
        return [np.concatenate([r[j] for r in packs[s]]) for j in range(3)]
    xf,yf,kf=combine("fit"); xv,yv,kv=combine("validation")
    xm=xf.mean(0);xs=xf.std(0).clip(.05)
    net=Prior(names).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.002)
    X=torch.tensor(xf,device=device);Y=torch.tensor(yf,device=device);K=torch.tensor(kf,device=device,dtype=torch.long)
    VX=torch.tensor((xv-xm)/xs,device=device);VY=torch.tensor(yv,device=device);VK=torch.tensor(kv,device=device,dtype=torch.long)
    M=torch.tensor(xm,device=device);S=torch.tensor(xs,device=device)
    best=1e9;best_state=None;history=[]
    for epoch in range(65):
        net.train();ids=torch.randperm(len(X),device=device)
        for start in range(0,len(ids),2048):
            ii=ids[start:start+2048]
            x=X[ii].clone();y=Y[ii].clone()
            # Span perturbations emulate modest differential palm noise.
            x[:,:3]+=torch.randn_like(x[:,:3])*.025
            x[:,3]=x[:,:3].norm(dim=-1)
            x[:,4:]+=torch.randn_like(x[:,4:])*.02
            pred=net((x-M)/S,K[ii])
            loss=torch.nn.functional.smooth_l1_loss(pred,y,beta=.05)
            opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            val=(net(VX,VK)-VY).norm(dim=-1).mean().item()
        history.append(val)
        if val<best:
            best=val;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};best_epoch=epoch
        if epoch%15==0: print(dataset,"prior epoch",epoch,"heldout GT cm",round(val*100,3),flush=True)
    net.load_state_dict(best_state);net.eval()
    torch.save({"state_dict":best_state,"names":names,"x_mean":xm,"x_std":xs,"best_epoch":best_epoch,"validation_mean_cm":best*100},out/f"{dataset}_prior.pt")
    return net,xm,xs,{"best_epoch":best_epoch,"validation_cm":best*100,"history_m":history,"fit_frames":len(xf),"validation_frames":len(xv)}


def make_statistics(records,names):
    stats={}
    for name in names:
        # Priors only from model fit split; calibration/validation never enter fitting.
        rr=[r for r in records if r["split"]=="fit" and r["object"]==name]
        c=np.concatenate([r["c"] for r in rr]);x=np.concatenate([r["x"] for r in rr])
        n=x[:,:3]/np.linalg.norm(x[:,:3],axis=-1,keepdims=True).clip(1e-8)
        signed=(c*n).sum(-1)
        cov=np.cov(c.T)+np.eye(3)*1e-4
        stats[name]={"c_mean":c.mean(0),"c_cov":cov,"signed_mean":float(signed.mean()),"signed_q05":float(np.percentile(signed,5)),"signed_q95":float(np.percentile(signed,95)),
                     "plane_eligible":bool(np.median(np.abs(signed))<=.06 and np.percentile(np.abs(signed),90)<=.20),"plane":geom_stats(signed)}
    return stats


def load_dump(path,dataset):
    data=torch.load(path,map_location="cpu",weights_only=False)
    rr=[]
    for r in data["records"]:
        n=int(r["publish_len"])
        if not n or not r.get("has_object",True): continue
        h=array(r["pred_palm_glb_pos"][:n]);p=array(r["pred_obj_trans_published"][:n])
        x,R=features(h,array(r["gt_obj_imu"][:n]),array(r["gt_human_imu"][:n]))
        gt=array(r["gt_obj_trans"][:n]);gh=array(r.get("gt_palm_pos_eval",r["gt_palm_pos"])[:n])
        contact=np.stack([np.asarray(r["gt_lhand_contact"][:n],bool),np.asarray(r["gt_rhand_contact"][:n],bool)],-1)
        cp=array(r["pred_hand_contact_prob"][:n,:2])
        c=np.einsum("tji,tj->ti",R,p-h.mean(1))
        age=[];v=0
        for active in (cp>=.5).all(-1):
            v=v+1 if active else 0;age.append(v)
        rr.append({"file":r["seq_file"],"object":r["obj_name"],"p":p,"h":h,"x":x,"R":R,"c":c,"cp":cp,"gt":gt,"gh":gh,"contact":contact,"age":np.array(age)})
    return rr,data


def smooth_correction(correction,cp):
    # A causal ramp/low-pass. Only the previous correction persists after release.
    out=np.zeros_like(correction);state=np.zeros(3)
    conf=np.clip((cp.min(-1)-.5)/.4,0,1)
    for t in range(len(out)):
        target=correction[t]*conf[t]
        state=.8*state+.2*target
        out[t]=state
    return out


def candidates(records,stats,net,xm,xs,names,device):
    ids={n:i for i,n in enumerate(names)}
    for r in records:
        name=r["object"];s=stats[name];c=r["c"];d=r["x"][:,:3]
        norm=np.linalg.norm(d,axis=-1);normal=d/norm[:,None].clip(.05)
        reliable=norm>=.10
        signed=(c*normal).sum(-1)
        with torch.no_grad():
            q=[]
            for start in range(0,len(c),2048):
                tx=torch.tensor((r["x"][start:start+2048]-xm)/xs,dtype=torch.float32,device=device)
                kk=torch.full((len(tx),),ids[name],dtype=torch.long,device=device)
                q.append(net(tx,kk).cpu().numpy())
        prior=np.concatenate(q)
        corrected_scalar=(prior*normal).sum(-1)
        # covariance MAP, 10 cm isotropic baseline uncertainty; blend calibrated later.
        W=.10**2*np.linalg.inv(s["c_cov"]+np.eye(3)*.10**2)
        choices={
            "hard_plane":-signed[:,None]*normal*s["plane_eligible"],
            "slab_q90":(np.clip(signed,s["signed_q05"],s["signed_q95"])-signed)[:,None]*normal,
            "category_plane_mean":(s["signed_mean"]-signed)[:,None]*normal*s["plane_eligible"],
            "category_covariance":(s["c_mean"]-c)@W.T,
            "learned_plane":(corrected_scalar-signed)[:,None]*normal,
            "learned_c":prior-c,
        }
        r["c_prior"]=prior
        r["delta"]={}
        for method,dc in choices.items():
            dc=dc*reliable[:,None]
            # Cap proposed world displacement at 20 cm before smoothing/gain.
            if method!="hard_plane":
                dc=dc*np.minimum(1,.2/np.linalg.norm(dc,axis=-1).clip(1e-9))[:,None]
            world=np.einsum("tij,tj->ti",r["R"],dc)
            r["delta"][method]=smooth_correction(world,r["cp"])
        # Deliberately naive comparator: unsmoothed full projection, predicted gate.
        dc=-signed[:,None]*normal*s["plane_eligible"]*reliable[:,None]
        r["delta"]["naive_hard_plane"]=np.einsum("tij,tj->ti",r["R"],dc)*(r["cp"]>=.5).all(-1)[:,None]


def measure(records,method="baseline",alpha=0.,per_object=False):
    seq=[];bi=[];bi_hoi=[];bi_reg=[];nonbi=[];per={}
    for r in records:
        p=r["p"]+(r["delta"][method]*alpha if method!="baseline" else 0)
        error=np.linalg.norm(p-r["gt"],axis=-1)*100
        rel=(p[:,None]-r["h"])-(r["gt"][:,None]-r["gh"])
        hoi=np.linalg.norm(rel,axis=-1)*100
        mask=r["contact"].all(-1)
        row={"file":r["file"],"object":r["object"],"frames":len(p),"bi_frames":int(mask.sum()),"obj_cm":float(error.mean()),
             "hoi_cm":float(hoi[r["contact"]].mean()) if r["contact"].any() else None,
             "acc_mm_frame2":float(np.linalg.norm(np.diff(p,n=2,axis=0),axis=-1).mean()*1000) if len(p)>2 else 0,
             "step_p99_cm":float(np.percentile(np.linalg.norm(np.diff(p,axis=0),axis=-1),99)*100),
             "bi_obj_cm":float(error[mask].mean()) if mask.any() else None,
             "bi_hoi_cm":float(hoi[mask].mean()) if mask.any() else None}
        seq.append(row);bi.extend(error[mask]);bi_hoi.extend(hoi[mask].mean(-1));nonbi.extend(error[~mask])
        if r["object"] not in RODS["OMOMO"]|RODS["BEHAVE"]:bi_reg.extend(error[mask])
    def mean(vals):
        arr=[v for v in vals if v is not None];return float(np.mean(arr)) if arr else None
    result={"sequences":len(seq),"obj_seq_cm":mean(r["obj_cm"] for r in seq),"hoi_seq_cm":mean(r["hoi_cm"] for r in seq),
            "bi_frames":len(bi),"bi_obj_frame_cm":mean(bi),"bi_hoi_frame_cm":mean(bi_hoi),"ordinary_bi_obj_frame_cm":mean(bi_reg),
            "nonbi_obj_frame_cm":mean(nonbi),"bi_obj_p95_cm":float(np.percentile(bi,95)),
            "acc_seq_mm_frame2":mean(r["acc_mm_frame2"] for r in seq),"step_p99_seq_cm":mean(r["step_p99_cm"] for r in seq)}
    return result,seq


def tune(records,method,baseline):
    trials=[]
    for alpha in (0.,.10,.25,.50,1.):
        m,_=measure(records,method,alpha)
        trials.append({"alpha":alpha,**m})
    # Select on sequence-weighted absolute object error; modest motion guardrails.
    eligible=[r for r in trials if r["acc_seq_mm_frame2"]<=baseline["acc_seq_mm_frame2"]*1.10 and r["hoi_seq_cm"]<=baseline["hoi_seq_cm"]*1.01]
    selected=min(eligible,key=lambda r:r["obj_seq_cm"]) if eligible else trials[0]
    return selected["alpha"],trials


def residual_features(r,stats,names):
    # Correct systematic predicted-input error, rather than assuming GT palm geometry.
    one=np.zeros((len(r["p"]),len(names)));one[:,names.index(r["object"])]=1
    return np.concatenate((r["x"],r["c"],r["c_prior"]-r["c"],r["cp"],np.minimum(r["age"],150)[:,None]/150,one),-1)


def fit_residual(fit_records,val_records,test_records,stats,names,out,dataset):
    # A small ridge baseline on predicted inputs; never use test targets in fitting.
    X=[];Y=[]
    for r in fit_records:
        ix=np.flatnonzero((r["cp"]>=.5).all(-1))[::2]
        if not len(ix):continue
        X.append(residual_features(r,stats,names)[ix])
        Y.append(np.einsum("tji,tj->ti",r["R"],r["gt"]-r["p"])[ix])
    x=np.concatenate(X);y=np.concatenate(Y)
    mu=x.mean(0);std=x.std(0).clip(.05)
    xn=(x-mu)/std;xn=np.c_[np.ones(len(xn)),xn]
    reg=np.eye(xn.shape[1])*len(xn)*.03;reg[0,0]=0
    w=np.linalg.solve(xn.T@xn+reg,xn.T@y)
    for r in fit_records+val_records+test_records:
        xx=(residual_features(r,stats,names)-mu)/std
        dc=np.c_[np.ones(len(xx)),xx]@w
        dc*=np.minimum(1,.15/np.linalg.norm(dc,axis=-1).clip(1e-9))[:,None]
        world=np.einsum("tij,tj->ti",r["R"],dc)
        r["delta"]["predicted_input_ridge"]=smooth_correction(world,r["cp"])
    np.savez(out/f"{dataset}_residual_ridge.npz",mu=mu,std=std,weights=w,names=np.array(names))
    return {"fit_sequences":len(fit_records),"validation_sequences":len(val_records),"fit_frames":len(x)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/analysis/meshfree_bimanual_20260930")
    parser.add_argument("--reuse-model",action="store_true")
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(930);np.random.seed(930)
    device=torch.device("cuda:0")
    all_results={};all_rows=[];obj_rows=[]
    for ds in DATA:
        cal,cal_dump=load_dump(ROOT/CAL[ds],ds)
        test,test_dump=load_dump(ROOT/TEST[ds],ds)
        train_files={p.name for p in (ROOT/"process/IMUHOI++cano"/DATA[ds]/"train").glob("*.pt")}
        assert {r["file"] for r in cal}<=train_files
        assert not {r["file"] for r in test}&train_files
        cp=cal_dump["provenance"]["checkpoints"];tp=test_dump["provenance"]["checkpoints"]
        assert {k:v["sha256"] for k,v in cp.items()}=={k:v["sha256"] for k,v in tp.items()}, "Checkpoint mismatch"
        records=load_training(ds,out,{r["file"] for r in cal})
        names=sorted({r["object"] for r in records})
        stats=make_statistics(records,names)
        if args.reuse_model and (out/f"{ds}_prior.pt").exists():
            model=torch.load(out/f"{ds}_prior.pt",map_location="cpu",weights_only=False)
            net=Prior(names).to(device);net.load_state_dict(model["state_dict"]);net.eval();xm=model["x_mean"];xs=model["x_std"]
            history={"reused":True,"validation_cm":model["validation_mean_cm"]}
        else:net,xm,xs,history=fit_prior(records,names,out,ds,device)
        candidates(cal+test,stats,net,xm,xs,names,device)
        # One third of each object's calibration sequences for gain validation.
        fit_cal=[];val_cal=[]
        for name in names:
            group=sorted([r for r in cal if r["object"]==name],key=lambda r:hashlib.sha256(r["file"].encode()).hexdigest())
            if len(group)>=2:
                val_cal.extend(group[::3]);fit_cal.extend([r for i,r in enumerate(group) if i%3])
            else:val_cal.extend(group)
        residual_meta=fit_residual(fit_cal,val_cal,test,stats,names,out,ds)
        base_cal,_=measure(val_cal);base_test,seq=measure(test)
        selected={};tuning={};results={"baseline":base_test};comparators={}
        for method in cal[0]["delta"]:
            alpha,trials=tune(val_cal,method,base_cal)
            selected[method]=alpha;tuning[method]=trials
            m,seq=measure(test,method,alpha)
            results[method]={"alpha":alpha,**m}
            all_rows.extend({"dataset":ds,"method":method,"alpha":alpha,**r} for r in seq)
            if method in ("hard_plane","naive_hard_plane"):
                comparators[method+"_forced100"]=measure(test,method,1.)[0]
            for name in names:
                subset=[r for r in test if r["object"]==name]
                if not subset or not any(r["contact"].all(-1).any() for r in subset):continue
                a,_=measure(subset,method,alpha);b,_=measure(subset)
                obj_rows.append({"dataset":ds,"object":name,"method":method,"alpha":alpha,"baseline_bi_obj_cm":b["bi_obj_frame_cm"],"baseline_obj_seq_cm":b["obj_seq_cm"],**a})
        base_seq=measure(test)[1]
        all_rows.extend({"dataset":ds,"method":"baseline","alpha":0,**r} for r in base_seq)
        meta={"training":history,"residual_calibration":residual_meta,"selected_alpha":selected,"tuning":tuning,"results":results,"forced_diagnostics":comparators,
              "stats":{n:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in s.items()} for n,s in stats.items()},
              "validation_files":[r["file"] for r in val_cal],"residual_fit_files":[r["file"] for r in fit_cal],
              "test_cache":TEST[ds],"calibration_cache":CAL[ds],"test_frames":sum(len(r["p"]) for r in test)}
        all_results[ds]=meta
        (out/"results.json").write_text(json.dumps(all_results,indent=2,allow_nan=False),encoding="utf-8")
        print(ds,"RESULTS",json.dumps(results),flush=True)
    csv_write(out/"per_sequence_results.csv",all_rows)
    csv_write(out/"per_object_results.csv",obj_rows)
    csv_write(out/"overall_results.csv",[{"dataset":ds,"method":method,**r} for ds,d in all_results.items() for method,r in d["results"].items()])
    print("Saved",out,flush=True)


if __name__=="__main__":main()

