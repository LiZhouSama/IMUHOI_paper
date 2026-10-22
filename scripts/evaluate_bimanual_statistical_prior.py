"""Fit the statistical category-offset comparator on the locked student fit split."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_bimanual_prior_student import score,choose,json_write,csv_write


def candidates(records,priors):
    predictions={}
    for r in records:
        p=priors[r["object"]];cov=np.asarray(p["covariance"]);mu=np.asarray(p["mean"])
        gain=.01*np.linalg.inv(cov+np.eye(3)*.01)
        delta=(mu-np.asarray(r["c"]))@gain.T
        delta*=np.minimum(1,.2/np.linalg.norm(delta,axis=-1).clip(1e-9))[:,None]
        world=np.einsum("tij,tj->ti",np.asarray(r["R"]),delta)
        predictions[r["file"]]={"world":world,"variance":np.broadcast_to(np.diag(cov),(len(world),3)).copy()}
    return predictions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset",required=True,choices=("OMOMO","BEHAVE"));p.add_argument("--root",type=Path,default=ROOT/"outputs/experiments/meshfree_temporal_20261001");a=p.parse_args()
    torch.set_num_threads(2);data=torch.load(a.root/f"{a.dataset}_train_prepared.pt",map_location="cpu",weights_only=False)
    fit=[r for r in data["records"] if r["partition"]=="fit"];cal=[r for r in data["records"] if r["partition"]=="calibration"]
    priors={}
    for name in data["names"]:
        c=np.concatenate([np.asarray(r["geometry_target"])[np.asarray(r["contact"]).all(-1)] for r in fit if r["object"]==name])
        if len(c)<2:raise ValueError((name,"insufficient fit observations"))
        priors[name]={"mean":c.mean(0).tolist(),"covariance":(np.cov(c.T)+np.eye(3)*.0001).tolist(),"fit_frames":len(c)}
    base,_=score(cal);prediction=candidates(cal,priors);selection,trials=choose(cal,prediction,base)
    test=torch.load(a.root/f"{a.dataset}_test_prepared.pt",map_location="cpu",weights_only=False)["records"]
    pred=candidates(test,priors);base,base_rows=score(test);selected={k:selection[k] for k in ("gain","uncertainty","carry")};actual,rows=score(test,pred,**selected)
    out=a.root/a.dataset/"category_covariance";out.mkdir(parents=True,exist_ok=True)
    json_write(out/"result.json",{"dataset":a.dataset,"selected":selected,"baseline":base,"result":actual,"priors":priors,"calibration_trials":trials})
    csv_write(out/"per_sequence.csv",rows);csv_write(out/"baseline_per_sequence.csv",base_rows)
    print(a.dataset,selected,actual,flush=True)


if __name__=="__main__":main()
