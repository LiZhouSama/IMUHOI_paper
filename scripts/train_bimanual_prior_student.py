"""Train and evaluate causal bimanual correction students without test meshes.

Frozen HP/VC/OT predictions provide inputs. Subject-object groups in the original
training split determine fit/validation/calibration; the original test is held out.
"""
from __future__ import annotations
import argparse,csv,copy,hashlib,json,random,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.rnn.bimanual_prior_student import BimanualPriorStudent,causal_filter

VARIANTS={
 "frame_category":dict(temporal=False,use_category=True,mesh_weight=0.),
 "temporal_unknown":dict(temporal=True,use_category=False,mesh_weight=0.),
 "temporal_category":dict(temporal=True,use_category=True,mesh_weight=0.),
 "temporal_mesh_aux":dict(temporal=True,use_category=True,mesh_weight=.10),
}


def json_write(path,data):
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8");temp.replace(path)


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def make_model(package,device):
    settings=package["settings"]
    model=BimanualPriorStudent(settings["feature_dim"],len(package["names"]),
        temporal=settings["temporal"],use_category=settings["use_category"],hidden_dim=settings["hidden_dim"],max_correction=settings["max_correction"]).to(device)
    if "state_dict" in package:model.load_state_dict(package["state_dict"])
    return model


def collate_windows(records,rng,names,mean,std,device,batch_size=48,length=128):
    keys=("x","R","cp","p","h","c","gt","gh","contact","residual_target","geometry_target","patch_target","surface_target","mesh_confidence")
    batches={k:[] for k in keys};valid=[];cats=[];startframe=[]
    for _ in range(batch_size):
        r=records[int(rng.integers(len(records)))];n=len(r["p"])
        # Sample most windows around predicted or true bimanual events, retain negatives.
        candidates=torch.where(r["contact"].all(-1)|(r["cp"][:,:2].amin(-1)>.4))[0]
        if len(candidates) and rng.random()<.8:
            pivot=int(candidates[int(rng.integers(len(candidates)))])
            start=max(0,min(n-length,pivot-int(rng.integers(32,length))))
        else:start=int(rng.integers(max(1,n-length+1)))
        take=min(length,n-start)
        for k in keys:
            v=r[k][start:start+take]
            pad=torch.zeros((length-take,*v.shape[1:]),dtype=v.dtype)
            batches[k].append(torch.cat((v,pad)))
        mask=torch.arange(length)<take
        if start>0:mask[:24]=False
        valid.append(mask);cats.append(names.index(r["object"]));startframe.append(start)
    result={k:torch.stack(v).to(device) for k,v in batches.items()}
    result["x"]=(result["x"]-mean)/std
    result["valid"]=torch.stack(valid).to(device)
    result["cat"]=torch.tensor(cats,device=device,dtype=torch.long)
    result["start"]=torch.tensor(startframe,device=device)
    return result


def train_loss(model,b,mesh_weight):
    pred=model(b["x"],b["cat"])
    world=torch.einsum("btij,btj->bti",b["R"],pred["delta"])
    applied,_=causal_filter(world,b["cp"],pred["variance"].detach(),uncertainty=False)
    valid=b["valid"]
    contact=b["contact"].all(-1)&valid
    predicted_active=(b["cp"][...,:2].amin(-1)>.35)&valid
    weights=valid.float()*(.1+.9*(contact|predicted_active).float())
    err=b["p"]+applied-b["gt"]
    # Absolute object error has priority; HOI penalizes worsening relative geometry.
    obj=(F.smooth_l1_loss(err,torch.zeros_like(err),beta=.04,reduction="none").sum(-1)*weights).sum()/weights.sum().clamp_min(1)
    rel=err.unsqueeze(-2)-(b["h"]-b["gh"])
    mask=b["contact"]&valid.unsqueeze(-1)
    hoi=(F.smooth_l1_loss(rel,torch.zeros_like(rel),beta=.04,reduction="none").sum(-1)*mask).sum()/mask.sum().clamp_min(1)
    geom=(F.smooth_l1_loss(pred["geometry"],b["geometry_target"],beta=.05,reduction="none").sum(-1)*contact).sum()/contact.sum().clamp_min(1)
    residual_error=(pred["delta"].detach()-b["residual_target"]).square().clamp_max(.25)
    variance=pred["variance"]
    var_loss=((torch.log(variance)+residual_error/variance).mean(-1)*predicted_active).sum()/predicted_active.sum().clamp_min(1)
    smooth_mask=valid[:,1:]&valid[:,:-1]
    smooth=((applied[:,1:]-applied[:,:-1]).square().sum(-1)*smooth_mask).sum()/smooth_mask.sum().clamp_min(1)
    regularizer=(applied.square().sum(-1)*valid).sum()/valid.sum().clamp_min(1)
    loss=obj+.20*hoi+.05*geom+.002*var_loss+.5*smooth+.02*regularizer
    aux=loss.detach()*0
    if mesh_weight:
        logits=pred["patch_logits"].reshape(-1,8)
        ce=F.cross_entropy(logits,b["patch_target"].reshape(-1),reduction="none").reshape(*contact.shape,2)
        w=b["mesh_confidence"]*contact.unsqueeze(-1)
        patch=(ce*w).sum()/w.sum().clamp_min(1)
        surf=pred["surface"].reshape(*contact.shape,4,3)
        gt=b["surface_target"].reshape(*contact.shape,4,3)
        point=(F.smooth_l1_loss(surf[...,:2,:],gt[...,:2,:],beta=.05,reduction="none").sum(-1)*w).sum()/w.sum().clamp_min(1)
        normal=((1-F.cosine_similarity(surf[...,2:,:],gt[...,2:,:],dim=-1))*w).sum()/w.sum().clamp_min(1)
        aux=.02*patch+.2*point+.01*normal
        loss=loss+mesh_weight*aux
    return loss,{"obj":float(obj.detach()),"hoi":float(hoi.detach()),"geom":float(geom.detach()),"mesh":float(aux.detach())}


@torch.inference_mode()
def predict(model,records,names,mean,std,device,batch_size=20):
    model.eval();out={}
    order=sorted(records,key=lambda r:len(r["p"]),reverse=True)
    for start in range(0,len(order),batch_size):
        rr=order[start:start+batch_size];n=max(len(r["p"]) for r in rr)
        xs=torch.zeros(len(rr),n,len(mean),device=device)
        ids=torch.tensor([names.index(r["object"]) for r in rr],device=device,dtype=torch.long)
        for i,r in enumerate(rr):xs[i,:len(r["p"])]=r["x"].to(device)
        xs=(xs-mean)/std
        # Chunk hidden state without resetting it. Never read future features.
        state=None;deltas=[];vars=[]
        for left in range(0,n,256):
            result=model(xs[:,left:left+256],ids,state);state=result["state"]
            deltas.append(result["delta"].cpu());vars.append(result["variance"].cpu())
        dd=torch.cat(deltas,1);vv=torch.cat(vars,1)
        for i,r in enumerate(rr):
            length=len(r["p"]);local=dd[i,:length]
            world=torch.einsum("tij,tj->ti",r["R"],local)
            out[r["file"]]={"world":world.numpy(),"variance":vv[i,:length].numpy()}
    return out


def apply_numpy(value,r,gain,uncertainty=False,carry=False):
    if gain==0:return np.asarray(r["p"])
    key=f"{int(uncertainty)}_{int(carry)}"
    cached=value.setdefault("filtered",{})
    if key not in cached:
        delta=value["world"];var=value["variance"]
        weight=np.clip((np.asarray(r["cp"])[:,:2].min(-1)-.35)/.55,0,1)
        if uncertainty:weight*=.01/(.01+var.mean(-1))
        weight[0]=0.  # Preserve the original known initialization frame.
        state=np.zeros(3);track=np.zeros_like(delta)
        for i in range(len(track)):
            if carry:state=state+.25*weight[i]*(delta[i]-state)
            else:state=.75*state+.25*weight[i]*delta[i]
            track[i]=state
        cached[key]=track
    return np.asarray(r["p"])+cached[key]*gain


def score(records,outputs=None,gain=0.,uncertainty=False,carry=False):
    rows=[];pools=defaultdict(list)
    for r in records:
        base=np.asarray(r["p"]);p=base if outputs is None else apply_numpy(outputs[r["file"]],r,gain,uncertainty,carry)
        gt=np.asarray(r["gt"]);gh=np.asarray(r["gh"]);h=np.asarray(r["h"]);contact=np.asarray(r["contact"],bool);bi=contact.all(-1)
        obj=np.linalg.norm(p-gt,axis=-1)*100
        hoi=np.linalg.norm((p[:,None]-h)-(gt[:,None]-gh),axis=-1)*100
        acc=np.linalg.norm(np.diff(p,n=2,axis=0),axis=-1)*1000
        row={"file":r["file"],"object":r["object"],"group":r["group"],"frames":len(p),"bi_frames":int(bi.sum()),
             "object_cm":float(obj.mean()),"hoi_cm":float(hoi[contact].mean()) if contact.any() else None,
             "acc_mm_frame2":float(acc.mean()),"bi_object_cm":float(obj[bi].mean()) if bi.any() else None,
             "bi_hoi_cm":float(hoi[bi].mean()) if bi.any() else None,"correction_mean_cm":float(np.linalg.norm(p-base,axis=-1).mean()*100)}
        rows.append(row);pools["bi_object"].append(obj[bi]);pools["bi_hoi"].append(hoi[bi].mean(-1));pools["nonbi_object"].append(obj[~bi])
    mean=lambda key:float(np.mean([r[key] for r in rows if r[key] is not None]))
    out={"sequences":len(rows),"frames":sum(r["frames"] for r in rows),"bi_frames":sum(r["bi_frames"] for r in rows),
         "object_cm":mean("object_cm"),"hoi_cm":mean("hoi_cm"),"acc_mm_frame2":mean("acc_mm_frame2"),"correction_mean_cm":mean("correction_mean_cm")}
    for key,parts in pools.items():
        values=np.concatenate(parts)
        out[key+"_cm"]=float(values.mean()) if len(values) else None
        if key=="bi_object":out["bi_object_p95_cm"]=float(np.percentile(values,95)) if len(values) else None
    return out,rows


def choose(records,outputs,baseline):
    results=[]
    for uncertainty in (False,True):
        for carry in (False,True):
            for gain in (0.,.1,.25,.5,.75,1.):
                m,_=score(records,outputs,gain,uncertainty,carry)
                result={"gain":gain,"uncertainty":uncertainty,"carry":carry,**m}
                result["eligible"]=(m["hoi_cm"]<=baseline["hoi_cm"]+.02 and m["acc_mm_frame2"]<=baseline["acc_mm_frame2"]*1.03 and m["bi_object_p95_cm"]<=baseline["bi_object_p95_cm"]+.2)
                results.append(result)
    valid=[r for r in results if r["eligible"]]
    selected=min(valid,key=lambda r:(r["object_cm"],r["gain"]))
    return selected,results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"outputs/experiments/meshfree_temporal_20261001")
    parser.add_argument("--dataset",required=True,choices=("OMOMO","BEHAVE"))
    parser.add_argument("--variant",required=True,choices=tuple(VARIANTS))
    parser.add_argument("--seed",type=int,default=1001)
    parser.add_argument("--epochs",type=int,default=60)
    parser.add_argument("--steps",type=int,default=48)
    parser.add_argument("--evaluate-only",action="store_true")
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device=torch.device("cuda:0");root=args.root;run=root/args.dataset/args.variant/f"seed{args.seed}";run.mkdir(parents=True,exist_ok=True)
    package=torch.load(root/f"{args.dataset}_train_prepared.pt",map_location="cpu",weights_only=False)
    all_records=package["records"];names=package["names"]
    fit=[r for r in all_records if r["partition"]=="fit"]
    valid=[r for r in all_records if r["partition"]=="validation"]
    calibration=[r for r in all_records if r["partition"]=="calibration"]
    assert not {r["group"] for r in fit}&{r["group"] for r in valid+calibration}
    assert not {r["group"] for r in valid}&{r["group"] for r in calibration}
    config=VARIANTS[args.variant]
    settings={**config,"feature_dim":package["feature_dim"],"hidden_dim":64,"max_correction":.25}
    features=torch.cat([r["x"] for r in fit])
    mean=features.mean(0).to(device);std=features.std(0).clamp_min(.03).to(device);del features
    checkpoint={"settings":settings,"names":names,"mean":mean.cpu(),"std":std.cpu(),"dataset":args.dataset,"variant":args.variant,"seed":args.seed,
                "source_checkpoints":package["source_provenance"]["checkpoints"],"split_file":str(root/f"{args.dataset}_student_split.json")}
    model=make_model(checkpoint,device)
    if not args.evaluate_only:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.002)
        rng=np.random.default_rng(args.seed)
        baseline,_=score(valid)
        history=[];best=float("inf");best_epoch=0;start_time=time.time()
        print(args.dataset,args.variant,args.seed,"fit/val/cal",len(fit),len(valid),len(calibration),"baseline",json.dumps(baseline),flush=True)
        # Balanced random windows. Validation always traverses complete sequences.
        for epoch in range(1,args.epochs+1):
            model.train();losses=[]
            for step in range(args.steps):
                batch=collate_windows(fit,rng,names,mean,std,device)
                loss,terms=train_loss(model,batch,settings["mesh_weight"])
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach()))
            if epoch%3==0 or epoch==1:
                prediction=predict(model,valid,names,mean,std,device)
                scores,_=score(valid,prediction,gain=.5)
                objective=scores["object_cm"]+.2*max(0,scores["hoi_cm"]-baseline["hoi_cm"])+.1*max(0,scores["acc_mm_frame2"]-baseline["acc_mm_frame2"])
                if objective<best:
                    best=objective;best_epoch=epoch
                    saved={**checkpoint,"state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"epoch":epoch,"validation":scores}
                    torch.save(saved,run/"best.pt")
                history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"objective":objective,"elapsed_seconds":time.time()-start_time,**scores})
                json_write(run/"history.json",history)
                print("epoch",epoch,"obj",round(scores["object_cm"],4),"hoi",round(scores["hoi_cm"],4),"best",best_epoch,flush=True)
                if epoch-best_epoch>=18 and epoch>=30:break
            if epoch in (25,45):
                for g in optimizer.param_groups:g["lr"]*=.5
    trained=torch.load(run/"best.pt",map_location="cpu",weights_only=False);model=make_model(trained,device);model.eval()
    mean=trained["mean"].to(device);std=trained["std"].to(device)
    cal_base,_=score(calibration);cal_prediction=predict(model,calibration,names,mean,std,device)
    chosen,trials=choose(calibration,cal_prediction,cal_base)
    json_write(run/"calibration.json",{"baseline":cal_base,"selected":chosen,"trials":trials,"files":[r["file"] for r in calibration]})
    # Test opened after checkpoint and gain are fixed.
    test_package=torch.load(root/f"{args.dataset}_test_prepared.pt",map_location="cpu",weights_only=False)
    train_ckpt=package["source_provenance"]["checkpoints"];test_ckpt=test_package["source_provenance"]["checkpoints"]
    assert {k:v["sha256"] for k,v in train_ckpt.items()}=={k:v["sha256"] for k,v in test_ckpt.items()}
    test=test_package["records"]
    assert not {r["file"] for r in test}&{r["file"] for r in all_records}
    assert all("surface_target" not in r and "patch_target" not in r for r in test)
    prediction=predict(model,test,names,mean,std,device)
    baseline,base_rows=score(test)
    selected={k:chosen[k] for k in ("gain","uncertainty","carry")}
    actual,rows=score(test,prediction,**selected)
    # Store raw output tensors so inference decisions and later audits are reproducible.
    torch.save({"prediction":prediction,"selected":selected,"model":str(run/"best.pt")},run/"test_student_outputs.pt")
    report={"dataset":args.dataset,"variant":args.variant,"seed":args.seed,"best_epoch":trained["epoch"],"selected":selected,"baseline":baseline,"result":actual,
            "fit_sequences":len(fit),"validation_sequences":len(valid),"calibration_sequences":len(calibration),"mesh_used_at_test":False}
    json_write(run/"result.json",report);csv_write(run/"per_sequence.csv",rows);csv_write(run/"baseline_per_sequence.csv",base_rows)
    print("FINAL",json.dumps(report),flush=True)


if __name__=="__main__":main()


