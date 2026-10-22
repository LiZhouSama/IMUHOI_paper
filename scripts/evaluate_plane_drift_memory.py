"""Calibration-only memory sweep and non-deployable GT-plane diagnostic."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_plane_drift_observer import fit_bands,score,calibration,analytic,create_model,predict,save_json,write_csv


def oracle(records,bands,mode,memory=300.,ridge=.5):
    # Explicitly diagnostic: GT hands/contact supplied only to compute a non-deployable reference, not an optimal upper bound.
    from model.rnn.plane_drift_observer import plane_accumulate
    outputs={}
    for r in records:
        h=r["gh"] if mode=="true_hands_plane" else r["h"]
        direction=h[:,1]-h[:,0];span=direction.norm(dim=-1);n=direction/span[:,None].clamp_min(1e-6)
        signed=((r["p"]-h.mean(1))*n).sum(-1)
        if mode=="true_offset":target=((r["gt"]-h.mean(1))*n).sum(-1)
        else:target=signed*0
        y=(target-signed).clamp(-.5,.5)
        w=r["contact"].all(-1).float()*(span>=.12)*bands[r["object"]]["eligible"];w[0]=0
        decay=torch.exp(-w/memory);state=None;pieces=[];information=[]
        for offset in range(0,len(h),256):
            c,state,a=plane_accumulate(n[None,offset:offset+256],y[None,offset:offset+256],w[None,offset:offset+256],decay[None,offset:offset+256],state,ridge)
            pieces.append(c[0]);information.append(a[0])
        outputs[r["file"]]={"correction":torch.cat(pieces).numpy(),"information":torch.cat(information).numpy()}
    return outputs


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--dataset",required=True,choices=("OMOMO","BEHAVE"));ap.add_argument("--trained",action="store_true");args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/"configs/experiments/plane_drift_20261001.yaml").read_text());root=ROOT/cfg["output_root"];source=ROOT/cfg["source_root"]
    torch.set_num_threads(2);device=torch.device("cuda:0")
    package=torch.load(source/f"{args.dataset}_train_prepared.pt",map_location="cpu",weights_only=False);fit=[r for r in package["records"] if r["partition"]=="fit"];cal=[r for r in package["records"] if r["partition"]=="calibration"]
    bands=fit_bands(fit,package["names"]);base_cal,_=score(cal,bands)
    output={}
    if not args.trained:
        for method,slab in (("zero_plane",False),("thick_plane",True)):
            all_trials=[]
            for memory in (15.,90.,300.,900.):
                pp=analytic(cal,bands,memory,cfg["ridge"],slab);selection,trials=calibration(cal,bands,pp,base_cal)
                all_trials.extend({"memory_frames":memory,**t} for t in trials)
            chosen=min((t for t in all_trials if t["eligible"]),key=lambda t:(t["selection_score"],t["gain"]))
            save_json(root/f"{args.dataset}_{method}_memory_calibration.json",{"selected":chosen,"trials":all_trials})
            test=torch.load(source/f"{args.dataset}_test_prepared.pt",map_location="cpu",weights_only=False)["records"]
            pp=analytic(test,bands,chosen["memory_frames"],cfg["ridge"],slab);result,rows=score(test,bands,pp,chosen["gain"])
            base,base_rows=score(test,bands);output[method]={"selected":chosen,"baseline":base,"result":result}
            write_csv(root/f"{args.dataset}_{method}_memory_per_sequence.csv",rows)
        # Fixed gain/memory oracle diagnostics, never used to select deployable models.
        for mode in ("true_hands_plane","true_offset"):
            pp=oracle(test,bands,mode)
            result,rows=score(test,bands,pp,1.)
            output["ORACLE_"+mode]={"baseline":base,"result":result,"deployable":False,"memory_frames":300.,"gain":1.}
            write_csv(root/f"{args.dataset}_ORACLE_{mode}_per_sequence.csv",rows)
        save_json(root/f"{args.dataset}_analytic_memory_and_oracle.json",output)
    else:
        for checkpoint in sorted((root/args.dataset).glob("*/seed*/best.pt")):
            run=checkpoint.parent
            if not (run/"result.json").exists() or (run/"memory_result.json").exists():continue
            saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
            model=create_model(saved["config"],saved["feature_dim"],saved["names"],saved["variant"],device);model.load_state_dict(saved["state_dict"])
            mean=saved["mean"].to(device);std=saved["std"].to(device);trials=[]
            for memory in (15.,90.,300.,900.):
                model.memory_frames=memory
                pp=predict(model,cal,saved["names"],bands,mean,std,device);selected,tt=calibration(cal,bands,pp,base_cal)
                trials.extend({"memory_frames":memory,**t} for t in tt)
            chosen=min((t for t in trials if t["eligible"]),key=lambda t:(t["selection_score"],t["gain"]))
            save_json(run/"memory_calibration.json",{"selected":chosen,"trials":trials})
            test=torch.load(source/f"{args.dataset}_test_prepared.pt",map_location="cpu",weights_only=False)["records"]
            model.memory_frames=chosen["memory_frames"];pp=predict(model,test,saved["names"],bands,mean,std,device)
            result,rows=score(test,bands,pp,chosen["gain"]);base,_=score(test,bands)
            report={"dataset":args.dataset,"variant":saved["variant"],"seed":saved["seed"],"selected":chosen,"baseline":base,"result":result}
            save_json(run/"memory_result.json",report);write_csv(run/"memory_per_sequence.csv",rows)
            print("MEMORY",args.dataset,saved["variant"],saved["seed"],chosen["memory_frames"],chosen["gain"],json.dumps(result),flush=True)
    print("DONE",args.dataset,args.trained,flush=True)


if __name__=="__main__":main()

