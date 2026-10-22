"""Additional mesh-free ablations: no category, empirical Bayes, sequence uncertainty.

Run after meshfree_bimanual_prior_replay.py. No test-derived tuning.
"""
from __future__ import annotations
import csv,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meshfree_bimanual_prior_replay as core

OUT=core.ROOT/"outputs/analysis/meshfree_bimanual_20260930"


def fit_bayes(cal,stats):
    data=defaultdict(list)
    for r in cal:
        mask=(r["cp"]>=.5).all(-1)
        mid=r["h"].mean(1)
        truth=np.einsum("tji,tj->ti",r["R"],r["gt"]-mid)
        err=r["c"]-truth
        data[r["object"]].append(err[mask])
    allerr=np.concatenate([a for parts in data.values() for a in parts if len(a)])
    noise=np.cov(allerr.T)+np.eye(3)*.0004
    result={}
    for name,s in stats.items():
        # Pseudo-prior target should account for common hand error.
        # Retain geometric mean; calibration only sets measurement uncertainty.
        gain=noise@np.linalg.inv(noise+s["c_cov"])
        result[name]=gain
    return result,noise


def aggregate_delta_ci(base_rows,rows):
    base={r["file"]:r for r in base_rows}
    delta=np.array([r["obj_cm"]-base[r["file"]]["obj_cm"] for r in rows])
    rng=np.random.RandomState(930)
    means=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(3000)])
    return {"mean_delta_cm":float(delta.mean()),"paired_sequence_bootstrap95_cm":np.percentile(means,[2.5,97.5]).tolist(),"sequences_improved_pct":float((delta<0).mean()*100)}


def diagnose(records,dataset):
    pools=defaultdict(list)
    rows=[]
    for r in records:
        h=r["h"];p=r["p"];gt=r["gt"];gh=r["gh"]
        normal=h[:,1]-h[:,0];span=np.linalg.norm(normal,axis=-1)
        normal/=span[:,None].clip(1e-8)
        gnormal=gh[:,1]-gh[:,0];gnormal/=np.linalg.norm(gnormal,axis=-1,keepdims=True).clip(1e-8)
        signed_gt=((gt-gh.mean(1))*gnormal).sum(-1)
        signed_pred=((p-h.mean(1))*normal).sum(-1)
        signed_gt_predhands=((gt-h.mean(1))*normal).sum(-1)
        object_error=p-gt
        along=(object_error*normal).sum(-1)
        tangent=np.linalg.norm(object_error-along[:,None]*normal,axis=-1)
        miderr=h.mean(1)-gh.mean(1)
        object_cm=np.linalg.norm(object_error,axis=-1)*100
        age=r["age"]
        for name,mask in (("all_bimanual",r["contact"].all(-1)),("ordinary_bimanual",r["contact"].all(-1)&(r["object"] not in core.RODS[dataset])),("predicted_bimanual",(r["cp"]>=.5).all(-1))):
            pools[(name,"base_obj")].append(object_cm[mask])
            pools[(name,"ideal_normal_only_obj")].append(tangent[mask]*100)
            pools[(name,"normal_error")].append(np.abs(along[mask])*100)
            pools[(name,"GT_plane_distance")].append(np.abs(signed_gt[mask])*100)
            pools[(name,"predicted_plane_distance")].append(np.abs(signed_pred[mask])*100)
            pools[(name,"GT_object_predicted_plane_distance")].append(np.abs(signed_gt_predhands[mask])*100)
            pools[(name,"midpoint_error")].append(np.linalg.norm(miderr[mask],axis=-1)*100)
            pools[(name,"normal_error_sq")].append(along[mask]**2)
            pools[(name,"full_error_sq")].append((object_error[mask]**2).sum(-1))
        for subset,mask in (("true_bimanual",r["contact"].all(-1)),("false_bimanual",(r["cp"]>=.5).all(-1)&~r["contact"].all(-1)),("late_bimanual",r["contact"].all(-1)&(age>=60))):
            if mask.any():rows.append({"file":r["file"],"object":r["object"],"subset":subset,"frames":int(mask.sum()),"object_cm":float(object_cm[mask].mean())})
    out={}
    for subset in ("all_bimanual","ordinary_bimanual","predicted_bimanual"):
        out[subset]={}
        for metric in ("base_obj","ideal_normal_only_obj","normal_error","GT_plane_distance","predicted_plane_distance","GT_object_predicted_plane_distance","midpoint_error"):
            v=np.concatenate(pools[subset,metric]);out[subset][metric]={"frames":len(v),"mean_cm":float(v.mean()),"median_cm":float(np.median(v)),"p90_cm":float(np.percentile(v,90))}
        out[subset]["normal_fraction_squared_object_error"]=float(np.concatenate(pools[subset,"normal_error_sq"]).sum()/np.concatenate(pools[subset,"full_error_sq"]).sum())
    return out,rows


def main():
    torch.set_num_threads(2);device=torch.device("cuda:0")
    original=json.loads((OUT/"results.json").read_text())
    output={};per_rows=[];selected_results=[]
    for ds in core.DATA:
        torch.manual_seed(930);np.random.seed(930)
        cal,_=core.load_dump(core.ROOT/core.CAL[ds],ds);test,_=core.load_dump(core.ROOT/core.TEST[ds],ds)
        records=core.load_training(ds,OUT,{r["file"] for r in cal});names=sorted({r["object"] for r in records})
        stats=core.make_statistics(records,names)
        m=torch.load(OUT/f"{ds}_prior.pt",map_location="cpu",weights_only=False)
        net=core.Prior(names).to(device);net.load_state_dict(m["state_dict"]);net.eval()
        core.candidates(cal+test,stats,net,m["x_mean"],m["x_std"],names,device)
        val_names=set(original[ds]["validation_files"])
        fit=[r for r in cal if r["file"] not in val_names];val=[r for r in cal if r["file"] in val_names]
        core.fit_residual(fit,val,test,stats,names,OUT,ds)
        gains,noise=fit_bayes(fit,stats)
        # Universal student; identical observations but a single constant category token.
        anon=[{**r,"object":"unknown"} for r in records]
        unet,um,us,history=core.fit_prior(anon,["unknown"],OUT,ds+"_no_identity",device)
        for r in cal+test:
            with torch.no_grad():
                xx=torch.tensor((r["x"]-um)/us,dtype=torch.float32,device=device)
                k=torch.zeros(len(xx),dtype=torch.long,device=device)
                universal=unet(xx,k).cpu().numpy()
            d=r["x"][:,:3];span=np.linalg.norm(d,axis=-1);n=d/span[:,None].clip(1e-8)
            sg=(r["c"]*n).sum(-1)
            variants={"no_identity_soft_plane":-sg[:,None]*n,
                      "no_identity_student":universal-r["c"],
                      "category_empirical_bayes":(stats[r["object"]]["c_mean"]-r["c"])@gains[r["object"]].T}
            for method,dc in variants.items():
                dc*=np.minimum(1,.20/np.linalg.norm(dc,axis=-1).clip(1e-9))[:,None]
                dc*= (span>=.10)[:,None]
                r["delta"][method]=core.smooth_correction(np.einsum("tij,tj->ti",r["R"],dc),r["cp"])
        base_val,_=core.measure(val);base_test,base_seq=core.measure(test)
        results={"baseline":base_test};trials={};cis={};selected={}
        for method in cal[0]["delta"]:
            alpha,trial=core.tune(val,method,base_val)
            value,seq=core.measure(test,method,alpha)
            results[method]={"alpha":alpha,**value};trials[method]=trial
            cis[method]=aggregate_delta_ci(base_seq,seq)
            per_rows.extend({"dataset":ds,"method":method,"alpha":alpha,**r} for r in seq)
        # Selection among deployable candidates is made on validation only.
        best_method=min(results.keys(),key=lambda method:base_val["obj_seq_cm"] if method=="baseline" else next(t["obj_seq_cm"] for t in trials[method] if t["alpha"]==results[method]["alpha"]))
        diag,diag_rows=diagnose(test,ds)
        output[ds]={"results":results,"training_only_selected_method":best_method,"selection_tuning":trials,"paired_sequence_ci":cis,"diagnostics":diag,"no_identity_student_training":history,"calibration_error_covariance_m2":noise.tolist()}
        selected_results.append({"dataset":ds,"method":best_method,**results[best_method]})
        print(ds,"SELECTED",best_method,json.dumps(results[best_method]),flush=True)
        print(ds,"EXTRA",json.dumps({k:v for k,v in results.items() if k.startswith("no_identity") or k=="category_empirical_bayes"}),flush=True)
        (OUT/"extended_results.json").write_text(json.dumps(output,indent=2,allow_nan=False),encoding="utf-8")
    core.csv_write(OUT/"extended_per_sequence.csv",per_rows)
    core.csv_write(OUT/"train_selected_results.csv",selected_results)


if __name__=="__main__":main()
