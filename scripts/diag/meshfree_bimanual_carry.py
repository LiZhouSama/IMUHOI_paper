"""Evaluate causal carry of a mesh-free correction through contact gaps.

This is output-track replay, not rerunning model feedback. Gains selected only on
training calibration. GT is used solely for scoring and calibration targets.
"""
from __future__ import annotations
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import meshfree_bimanual_prior_replay as core
from meshfree_bimanual_ablation import aggregate_delta_ci
OUT=core.ROOT/"outputs/analysis/meshfree_bimanual_20260930"


def carry(target,cp,normal=None):
    state=np.zeros(3);out=np.zeros_like(target)
    confidence=np.clip((cp.min(-1)-.5)/.4,0,1)
    for t in range(len(target)):
        innovation=target[t]-state
        if normal is not None:
            innovation=np.dot(innovation,normal[t])*normal[t]
        state=state+.2*confidence[t]*innovation
        out[t]=state
    return out


def main():
    torch.set_num_threads(2);device=torch.device("cuda:0")
    previous=json.loads((OUT/"results.json").read_text());output={};rows=[]
    for ds in core.DATA:
        cal,_=core.load_dump(core.ROOT/core.CAL[ds],ds);test,_=core.load_dump(core.ROOT/core.TEST[ds],ds)
        records=core.load_training(ds,OUT,{r["file"] for r in cal});names=sorted({r["object"] for r in records});st=core.make_statistics(records,names)
        model=torch.load(OUT/f"{ds}_prior.pt",map_location="cpu",weights_only=False)
        net=core.Prior(names).to(device);net.load_state_dict(model["state_dict"]);net.eval()
        core.candidates(cal+test,st,net,model["x_mean"],model["x_std"],names,device)
        val_ids=set(previous[ds]["validation_files"]);calval=[r for r in cal if r["file"] in val_ids]
        for r in cal+test:
            s=st[r["object"]];c=r["c"];d=r["x"][:,:3];span=np.linalg.norm(d,axis=-1);n=d/span[:,None].clip(1e-8)
            W=.01*np.linalg.inv(s["c_cov"]+np.eye(3)*.01)
            choices={"carry_covariance":(s["c_mean"]-c)@W.T,"carry_learned":r["c_prior"]-c,
                     "carry_learned_normal":((r["c_prior"]-c)*n).sum(-1)[:,None]*n}
            for method,dc in choices.items():
                dc*=np.minimum(1,.20/np.linalg.norm(dc,axis=-1).clip(1e-9))[:,None]
                world=np.einsum("tij,tj->ti",r["R"],dc)
                cp=r["cp"].copy();cp[span<.10]=0
                nw=np.einsum("tij,tj->ti",r["R"],n) if method.endswith("normal") else None
                r["delta"][method]=carry(world,cp,nw)
        baseline,_=core.measure(calval);base_test,base_seq=core.measure(test)
        results={"baseline":base_test};tuning={};ci={}
        for method in choices:
            alpha,trial=core.tune(calval,method,baseline)
            result,seq=core.measure(test,method,alpha)
            results[method]={"alpha":alpha,**result};tuning[method]=trial;ci[method]=aggregate_delta_ci(base_seq,seq)
            rows.extend({"dataset":ds,"method":method,"alpha":alpha,**r} for r in seq)
        output[ds]={"results":results,"tuning":tuning,"paired_sequence_ci":ci}
        print(ds,json.dumps(results),flush=True)
    (OUT/"carry_results.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    core.csv_write(OUT/"carry_per_sequence.csv",rows)


if __name__=="__main__":main()
