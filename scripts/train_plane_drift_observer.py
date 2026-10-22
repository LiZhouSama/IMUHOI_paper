"""Full-sequence training/validation of persistent bimanual plane drift observers."""
from __future__ import annotations
import argparse,csv,json,random,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.rnn.plane_drift_observer import PlaneDriftObserver

VARIANTS=("vector_state","plane_state","plane_predicted_band","plane_gt_band")


def save_json(path,data):
    temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fit_bands(fit,names):
    result={}
    for name in names:
        values=[]
        for r in fit:
            if r["object"]!=name:continue
            normal=r["gh"][:,1]-r["gh"][:,0]
            normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            distance=((r["gt"]-r["gh"].mean(1))*normal).sum(-1).abs()
            values.append(distance[r["contact"].all(-1)])
        v=torch.cat(values).numpy()
        median=float(np.median(v));q90=float(np.percentile(v,90))
        result[name]={"median_m":median,"q90_m":q90,"tolerance_m":max(.02,q90),"eligible":median<=.06 and q90<=.20,"frames":len(v)}
    return result


def batch_sequences(records,names,bands,mean,std,device):
    keys=("x","R","cp","p","h","gt","gh","contact")
    n=max(len(r["p"]) for r in records);out={}
    for key in keys:
        first=records[0][key]
        value=torch.zeros((len(records),n,*first.shape[1:]),dtype=first.dtype)
        for i,r in enumerate(records):value[i,:len(r[key])]=r[key]
        out[key]=value.to(device)
    out["x"]=(out["x"]-mean)/std
    out["valid"]=torch.arange(n,device=device)[None]<torch.tensor([len(r["p"]) for r in records],device=device)[:,None]
    out["cat"]=torch.tensor([names.index(r["object"]) for r in records],device=device,dtype=torch.long)
    out["eligible"]=torch.tensor([bands[r["object"]]["eligible"] for r in records],device=device,dtype=torch.bool)
    out["tolerance"]=torch.tensor([bands[r["object"]]["tolerance_m"] for r in records],device=device)
    return out


def slice_batch(b,start,end):
    return {k:(v[:,start:end] if k not in ("cat","eligible","tolerance") else v) for k,v in b.items()}


def detach_state(state):
    return state[0].detach(),(state[1][0].detach(),state[1][1].detach())


def forward(model,b,state=None):
    return model(b["x"],b["cat"],b["h"],b["p"],b["R"],b["cp"],b["eligible"],b["valid"],state)


def objective(model,b,o,variant,plane_weight=.5):
    valid=b["valid"]
    bi=b["contact"].all(-1)&valid&b["eligible"][:,None]
    pred=b["p"]+o["correction"]
    diff=pred-b["gt"]
    weights=valid.float()*(.25+.75*b["eligible"][:,None].float())
    obj=(F.smooth_l1_loss(diff,torch.zeros_like(diff),beta=.05,reduction="none").sum(-1)*weights).sum()/weights.sum().clamp_min(1)
    rel=diff[:,:,None]-(b["h"]-b["gh"])
    hm=b["contact"]&valid[:,:,None]&b["eligible"][:,None,None]
    hoi=(F.smooth_l1_loss(rel,torch.zeros_like(rel),beta=.05,reduction="none").sum(-1)*hm).sum()/hm.sum().clamp_min(1)
    if model.vector_mode:
        target=torch.einsum("btji,btj->bti",b["R"],b["gt"]-b["p"])
        measurement=F.smooth_l1_loss(o["target"],target,beta=.05,reduction="none").sum(-1)
    else:
        target=((b["gt"]-b["h"].mean(-2))*o["normal"]).sum(-1)
        measurement=F.smooth_l1_loss(o["target"],target,beta=.05,reduction="none")
    measurement=(measurement*bi).sum()/bi.sum().clamp_min(1)
    smoothmask=valid[:,1:]&valid[:,:-1]
    speed=(o["correction"][:,1:]-o["correction"][:,:-1]).square().sum(-1)
    smooth=(speed*smoothmask).sum()/smoothmask.sum().clamp_min(1)
    plane=obj*0
    if variant in ("plane_predicted_band","plane_gt_band"):
        if variant=="plane_gt_band":
            direction=b["gh"][:,:,1]-b["gh"][:,:,0]
            normal=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-6);mid=b["gh"].mean(-2)
        else:normal=o["normal"];mid=b["h"].mean(-2)
        signed=((pred-mid)*normal).sum(-1)
        excess=(signed.abs()-b["tolerance"][:,None]).clamp_min(0)
        plane=(F.smooth_l1_loss(excess,torch.zeros_like(excess),beta=.05,reduction="none")*bi).sum()/bi.sum().clamp_min(1)
    regular=(o["correction"].square().sum(-1)*valid).sum()/valid.sum().clamp_min(1)
    loss=obj+.20*hoi+.05*measurement+plane_weight*plane+.2*smooth+.005*regular
    return loss,{"object":float(obj.detach()),"hoi":float(hoi.detach()),"plane":float(plane.detach()),"measurement":float(measurement.detach())}


@torch.inference_mode()
def predict(model,records,names,bands,mean,std,device,chunk=256):
    model.eval();outputs={}
    order=sorted(records,key=lambda r:len(r["p"]),reverse=True)
    for start in range(0,len(order),32):
        rr=order[start:start+32];b=batch_sequences(rr,names,bands,mean,std,device);state=None;pred=[];info=[]
        for offset in range(0,b["x"].shape[1],chunk):
            batch=slice_batch(b,offset,offset+chunk);o=forward(model,batch,state);state=o["state"]
            pred.append(o["correction"].cpu());info.append(o["information"].cpu())
        track=torch.cat(pred,1);mat=torch.cat(info,1)
        for i,r in enumerate(rr):
            n=len(r["p"]);outputs[r["file"]]={"correction":track[i,:n].numpy(),"information":mat[i,:n].numpy()}
    return outputs


def analytic(records,bands,memory=90.,ridge=.5,slab=True):
    from model.rnn.plane_drift_observer import plane_accumulate
    outputs={}
    for r in records:
        p=r["p"];h=r["h"];n=h[:,1]-h[:,0];span=n.norm(dim=-1);n=n/span[:,None].clamp_min(1e-6)
        signed=((p-h.mean(1))*n).sum(-1)
        tol=bands[r["object"]]["tolerance_m"]
        measurement=(signed.clamp(-tol,tol)-signed) if slab else -signed
        weight=((r["cp"][:,:2].amin(-1)-.35)/.55).clamp(0,1)*(span>=.12)*bands[r["object"]]["eligible"]
        weight[0]=0;decay=torch.exp(-weight/memory)
        state=None;out=[];information=[]
        for j in range(0,len(p),256):
            c,state,a=plane_accumulate(n[None,j:j+256],measurement[None,j:j+256],weight[None,j:j+256],decay[None,j:j+256],state,ridge)
            out.append(c[0]);information.append(a[0])
        outputs[r["file"]]={"correction":torch.cat(out).numpy(),"information":torch.cat(information).numpy()}
    return outputs


def score(records,bands,outputs=None,gain=0.,per_sequence=False):
    rows=[];pool=defaultdict(list)
    for r in records:
        base=np.asarray(r["p"]);p=base if outputs is None or gain==0 else base+outputs[r["file"]]["correction"]*gain
        gt=np.asarray(r["gt"]);h=np.asarray(r["h"]);gh=np.asarray(r["gh"]);contact=np.asarray(r["contact"],bool);bi=contact.all(-1)
        error=np.linalg.norm(p-gt,axis=-1)*100
        rel=(p[:,None]-h)-(gt[:,None]-gh);hoi=np.linalg.norm(rel,axis=-1)*100
        n=len(p);third=max(1,n//3);eligible=bands[r["object"]]["eligible"]
        row={"file":r["file"],"group":r["group"],"object":r["object"],"seconds":n/30,"frames":n,"eligible":eligible,
             "obj_cm":float(error.mean()),"hoi_cm":float(hoi[contact].mean()) if contact.any() else None,
             "last_third_obj_cm":float(error[-third:].mean()),"first_third_obj_cm":float(error[:third].mean()),
             "late_minus_early_cm":float(error[-third:].mean()-error[:third].mean()),
             "endpoint_drift_cm":float(np.linalg.norm((p[-1]-gt[-1])-(p[0]-gt[0]))*100),
             "acc_mm_frame2":float(np.linalg.norm(np.diff(p,n=2,axis=0),axis=-1).mean()*1000),
             "bi_obj_cm":float(error[bi].mean()) if bi.any() else None,
             "bi_hoi_cm":float(hoi[bi].mean()) if bi.any() else None,
             "after30_obj_cm":float(error[900:].mean()) if n>900 else None,
             "after60_obj_cm":float(error[1800:].mean()) if n>1800 else None}
        rows.append(row)
        for key,values in (("bi_obj",error[bi]),("bi_hoi",hoi[bi].mean(-1)),("nonbi_obj",error[~bi])):pool[key].append(values)
        if eligible:
            pool["ordinary_obj"].append(error);pool["ordinary_bi_obj"].append(error[bi])
    result={"sequences":len(rows),"frames":sum(r["frames"] for r in rows)}
    for key in ("obj_cm","hoi_cm","last_third_obj_cm","first_third_obj_cm","late_minus_early_cm","endpoint_drift_cm","acc_mm_frame2","after30_obj_cm","after60_obj_cm"):
        vals=[r[key] for r in rows if r[key] is not None];result[key]=float(np.mean(vals)) if vals else None
    for key,parts in pool.items():
        v=np.concatenate(parts);result[key+"_cm"]=float(v.mean()) if len(v) else None
        if key=="bi_obj":result["bi_obj_p95_cm"]=float(np.percentile(v,95))
    result["after30_sequences"]=sum(r["after30_obj_cm"] is not None for r in rows);result["after60_sequences"]=sum(r["after60_obj_cm"] is not None for r in rows)
    return result,rows


def calibration(records,bands,outputs,base):
    trials=[]
    for gain in (0.,.1,.25,.5,.75,1.):
        metrics,_=score(records,bands,outputs,gain)
        eligible=metrics["hoi_cm"]<=base["hoi_cm"]+.05 and metrics["acc_mm_frame2"]<=base["acc_mm_frame2"]*1.05 and metrics["bi_obj_p95_cm"]<=base["bi_obj_p95_cm"]+.3
        trials.append({"gain":gain,"eligible":eligible,"selection_score":metrics["obj_cm"]+.5*metrics["last_third_obj_cm"],**metrics})
    return min([r for r in trials if r["eligible"]],key=lambda r:(r["selection_score"],r["gain"])),trials


def create_model(config,feature_dim,names,variant,device):
    return PlaneDriftObserver(feature_dim,len(names),vector_mode=variant=="vector_state",hidden_dim=config["hidden_dim"],memory_frames=config["memory_frames"],ridge=config["ridge"]).to(device)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--dataset",required=True,choices=("OMOMO","BEHAVE"));ap.add_argument("--variant",required=True,choices=VARIANTS+("analytic",));ap.add_argument("--seed",type=int,default=1101);ap.add_argument("--epochs",type=int);a=ap.parse_args()
    config=yaml.safe_load((ROOT/"configs/experiments/plane_drift_20261001.yaml").read_text());out=ROOT/config["output_root"];source=ROOT/config["source_root"];out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);rng=np.random.default_rng(a.seed);device=torch.device("cuda:0")
    dataset=torch.load(source/f"{a.dataset}_train_prepared.pt",map_location="cpu",weights_only=False);names=dataset["names"]
    fit=[r for r in dataset["records"] if r["partition"]=="fit"];val=[r for r in dataset["records"] if r["partition"]=="validation"];cal=[r for r in dataset["records"] if r["partition"]=="calibration"]
    bands=fit_bands(fit,names);save_json(out/f"{a.dataset}_plane_bands.json",bands)
    run=out/a.dataset/a.variant/f"seed{a.seed}";run.mkdir(parents=True,exist_ok=True)
    if a.variant=="analytic":
        results={};test=torch.load(source/f"{a.dataset}_test_prepared.pt",map_location="cpu",weights_only=False)["records"]
        base_cal,_=score(cal,bands);base,base_rows=score(test,bands)
        for method,slab in (("zero_plane",False),("thick_plane",True)):
            cp=analytic(cal,bands,config["memory_frames"],config["ridge"],slab);chosen,trials=calibration(cal,bands,cp,base_cal)
            pred=analytic(test,bands,config["memory_frames"],config["ridge"],slab);metrics,rows=score(test,bands,pred,chosen["gain"])
            results[method]={"gain":chosen["gain"],"baseline":base,"result":metrics,"calibration":trials};write_csv(run/f"{method}_per_sequence.csv",rows)
        write_csv(run/"baseline_per_sequence.csv",base_rows);save_json(run/"result.json",results);print("ANALYTIC",a.dataset,json.dumps(results),flush=True);return
    xx=torch.cat([r["x"] for r in fit]);mean=xx.mean(0).to(device);std=xx.std(0).clamp_min(.03).to(device);del xx
    model=create_model(config,dataset["feature_dim"],names,a.variant,device);optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=.002)
    baseline_val,_=score(val,bands);history=[];best=float("inf");best_epoch=0;updates=0;frames_seen=0;start_time=time.time()
    print(a.dataset,a.variant,a.seed,"FULL_EPOCH fit",len(fit),sum(len(r["p"]) for r in fit),"baseline",json.dumps(baseline_val),flush=True)
    epochs=a.epochs or config["epochs"]
    for epoch in range(1,epochs+1):
        model.train();epoch_frames=0;epoch_sequences=0;losses=[]
        # Each original fit sequence appears exactly once, retaining its chronology.
        ordered=sorted(fit,key=lambda r:len(r["p"]));groups=[ordered[i:i+config["batch_sequences"]] for i in range(0,len(ordered),config["batch_sequences"])];rng.shuffle(groups)
        for rr in groups:
            b=batch_sequences(rr,names,bands,mean,std,device);state=None
            for offset in range(0,b["x"].shape[1],config["chunk_frames"]):
                chunk=slice_batch(b,offset,offset+config["chunk_frames"])
                result=forward(model,chunk,state);loss,terms=objective(model,chunk,result,a.variant,config["plane_loss_weight"])
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                state=detach_state(result["state"]);updates+=1
                epoch_frames+=int(chunk["valid"].sum());losses.append(float(loss.detach()))
            epoch_sequences+=len(rr)
        assert epoch_sequences==len(fit) and epoch_frames==sum(len(r["p"]) for r in fit)
        frames_seen+=epoch_frames
        predictions=predict(model,val,names,bands,mean,std,device,config["chunk_frames"])
        scores,_=score(val,bands,predictions,.5)
        selection=scores["obj_cm"]+.5*scores["last_third_obj_cm"]+.3*max(0,scores["hoi_cm"]-baseline_val["hoi_cm"])+.1*max(0,scores["acc_mm_frame2"]-baseline_val["acc_mm_frame2"])
        if selection<best:
            best=selection;best_epoch=epoch
            checkpoint={"state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"variant":a.variant,"dataset":a.dataset,"seed":a.seed,"epoch":epoch,"optimizer_updates":updates,"fit_frames_seen":frames_seen,"config":config,"names":names,"feature_dim":dataset["feature_dim"],"mean":mean.cpu(),"std":std.cpu(),"bands":bands,"validation":scores}
            torch.save(checkpoint,run/"best.pt")
        history.append({"epoch":epoch,"optimizer_updates":updates,"epoch_sequences":epoch_sequences,"epoch_frames":epoch_frames,"fit_frames_seen":frames_seen,"train_loss":float(np.mean(losses)),"elapsed_seconds":time.time()-start_time,"selection_score":selection,**scores})
        save_json(run/"history.json",history)
        print("epoch",epoch,"updates",updates,"frames",epoch_frames,"obj",round(scores["obj_cm"],4),"late",round(scores["last_third_obj_cm"],4),"hoi",round(scores["hoi_cm"],4),"best",best_epoch,flush=True)
        if epoch in (15,30):
            for group in optimizer.param_groups:group["lr"]*=.5
        if epoch>=config["min_epochs"] and epoch-best_epoch>=config["patience_epochs"]:break
    saved=torch.load(run/"best.pt",map_location="cpu",weights_only=False);model.load_state_dict(saved["state_dict"])
    predictions_cal=predict(model,cal,names,bands,mean,std,device);baseline_cal,_=score(cal,bands);selected,trials=calibration(cal,bands,predictions_cal,baseline_cal)
    save_json(run/"calibration.json",{"selected":selected,"trials":trials})
    test=torch.load(source/f"{a.dataset}_test_prepared.pt",map_location="cpu",weights_only=False)["records"]
    assert not {r["file"] for r in test}&{r["file"] for r in dataset["records"]}
    predictions_test=predict(model,test,names,bands,mean,std,device);base,base_rows=score(test,bands);actual,rows=score(test,bands,predictions_test,selected["gain"])
    torch.save({"predictions":predictions_test,"gain":selected["gain"]},run/"test_predictions.pt")
    # All-time observability is a diagnostic, never an input from GT.
    eig=[]
    for r in test:
        if bands[r["object"]]["eligible"]:
            mat=predictions_test[r["file"]]["information"][-1];eig.append(np.linalg.eigvalsh(mat).tolist())
    report={"dataset":a.dataset,"variant":a.variant,"seed":a.seed,"best_epoch":saved["epoch"],"stopped_epoch":history[-1]["epoch"],"optimizer_updates":updates,"fit_frames_seen":frames_seen,"selected_gain":selected["gain"],"baseline":base,"result":actual,"final_information_eigenvalues":eig,"full_sequence_epochs":True,"mesh_at_test":False}
    save_json(run/"result.json",report);write_csv(run/"per_sequence.csv",rows);write_csv(run/"baseline_per_sequence.csv",base_rows)
    print("FINAL",json.dumps({k:v for k,v in report.items() if k!="final_information_eigenvalues"}),flush=True)


if __name__=="__main__":main()
