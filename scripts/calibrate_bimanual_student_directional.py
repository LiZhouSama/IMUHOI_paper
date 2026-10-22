"""Calibration-only comparison of directional vs scalar uncertainty weighting.

Reuses trained student weights; model parameters and test decisions stay fixed
before the test output is evaluated. No geometry is read at inference.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_bimanual_prior_student import make_model,predict,score,choose,json_write,csv_write


def directional(outputs,records):
    result={}
    for r in records:
        value=outputs[r["file"]];R=np.asarray(r["R"])
        local=np.einsum("tji,tj->ti",R,value["world"])
        local*=.01/(.01+value["variance"])
        result[r["file"]]={"world":np.einsum("tij,tj->ti",R,local),"variance":np.zeros_like(value["variance"])}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT/"outputs/experiments/meshfree_temporal_20261001");p.add_argument("--dataset",required=True,choices=("OMOMO","BEHAVE"));args=p.parse_args()
    torch.set_num_threads(2);device=torch.device("cuda:0");root=args.root
    train=torch.load(root/f"{args.dataset}_train_prepared.pt",map_location="cpu",weights_only=False)
    cal=[r for r in train["records"] if r["partition"]=="calibration"]
    test=torch.load(root/f"{args.dataset}_test_prepared.pt",map_location="cpu",weights_only=False)["records"]
    for checkpoint in sorted((root/args.dataset).glob("*/seed*/best.pt")):
        run=checkpoint.parent
        if (run/"directional_result.json").exists():continue
        package=torch.load(checkpoint,map_location="cpu",weights_only=False)
        model=make_model(package,device);model.eval();names=package["names"];mu=package["mean"].to(device);std=package["std"].to(device)
        outputs=predict(model,cal,names,mu,std,device)
        new=directional(outputs,cal);baseline,_=score(cal);chosen,trials=choose(cal,new,baseline)
        selection={k:chosen[k] for k in ("gain","uncertainty","carry")}
        json_write(run/"directional_calibration.json",{"selected":chosen,"trials":trials})
        original=torch.load(run/"test_student_outputs.pt",map_location="cpu",weights_only=False)["prediction"]
        test_prediction=directional(original,test);base,base_rows=score(test);actual,rows=score(test,test_prediction,**selection)
        report={"dataset":args.dataset,"variant":package["variant"],"seed":package["seed"],"selected":selection,"baseline":base,"result":actual}
        json_write(run/"directional_result.json",report);csv_write(run/"directional_per_sequence.csv",rows)
        print("DIRECTIONAL",json.dumps(report),flush=True)


if __name__=="__main__":main()
