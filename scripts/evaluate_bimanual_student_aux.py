"""Check whether the training-only mesh targets were actually learned on validation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_bimanual_prior_student import make_model,json_write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT/"outputs/experiments/meshfree_temporal_20261001");p.add_argument("--dataset",required=True,choices=("OMOMO","BEHAVE"));a=p.parse_args()
    torch.set_num_threads(2);device=torch.device("cuda:0")
    payload=torch.load(a.root/f"{a.dataset}_train_prepared.pt",map_location="cpu",weights_only=False)
    records=sorted([r for r in payload["records"] if r["partition"]=="validation"],key=lambda r:len(r["p"]),reverse=True)
    # Class/hand constants expose how much auxiliary accuracy is due to imbalance.
    constants={}
    for name in payload["names"]:
        counts=torch.zeros(2,8);points=torch.zeros(2,3);normals=torch.zeros(2,3);mass=torch.zeros(2)
        for r in payload["records"]:
            if r["partition"]!="fit" or r["object"]!=name:continue
            w=r["mesh_confidence"]*r["contact"].all(-1)[:,None]
            v=r["surface_target"].reshape(-1,4,3)
            for hand in (0,1):counts[hand].scatter_add_(0,r["patch_target"][:,hand],w[:,hand])
            points+=(v[:,:2]*w[:,:,None]).sum(0);normals+=(v[:,2:]*w[:,:,None]).sum(0);mass+=w.sum(0)
        constants[name]=(counts.argmax(-1),points/mass[:,None].clamp_min(1e-6),normals/mass[:,None].clamp_min(1e-6))
    bc=bp=bn=bm=0.
    for r in records:
        patch,points,normals=constants[r["object"]];w=r["mesh_confidence"]*r["contact"].all(-1)[:,None];gt=r["surface_target"].reshape(-1,4,3)
        bm+=float(w.sum());bc+=float(((patch[None]==r["patch_target"])*w).sum())
        bp+=float(((points[None]-gt[:,:2]).norm(dim=-1)*w).sum());bn+=float((F.cosine_similarity(normals[None],gt[:,2:],dim=-1)*w).sum())
    results=[{"dataset":a.dataset,"variant":"class_hand_constant","seed":None,"weighted_patch_accuracy_pct":100*bc/bm,"surface_point_error_cm":100*bp/bm,"normal_cosine":bn/bm}]
    for variant in ("temporal_category","temporal_mesh_aux"):
        for path in sorted((a.root/a.dataset/variant).glob("seed*/best.pt")):
            if not (path.parent/"result.json").exists():continue
            package=torch.load(path,map_location="cpu",weights_only=False);m=make_model(package,device).eval()
            mean=package["mean"].to(device);std=package["std"].to(device);names=package["names"]
            count=correct=point=cosine=0.
            with torch.inference_mode():
                for start in range(0,len(records),20):
                    rr=records[start:start+20];n=max(len(r["p"]) for r in rr)
                    x=torch.zeros((len(rr),n,len(mean)),device=device)
                    for i,r in enumerate(rr):x[i,:len(r["p"])]=r["x"].to(device)
                    output=m((x-mean)/std,torch.tensor([names.index(r["object"]) for r in rr],device=device))
                    patches=output["patch_logits"].argmax(-1).cpu();surfaces=output["surface"].cpu().reshape(len(rr),n,4,3)
                    for i,r in enumerate(rr):
                        length=len(r["p"]);w=r["mesh_confidence"]*r["contact"].all(-1)[:,None]
                        gt=r["surface_target"].reshape(length,4,3);s=surfaces[i,:length]
                        count+=float(w.sum());correct+=float(((patches[i,:length]==r["patch_target"])*w).sum())
                        point+=float(((s[:,:2]-gt[:,:2]).norm(dim=-1)*w).sum())
                        cosine+=float((F.cosine_similarity(s[:,2:],gt[:,2:],dim=-1)*w).sum())
            results.append({"dataset":a.dataset,"variant":variant,"seed":package["seed"],"weighted_patch_accuracy_pct":100*correct/count,"surface_point_error_cm":100*point/count,"normal_cosine":cosine/count})
    json_write(a.root/f"{a.dataset}_aux_validation.json",results)
    print(json.dumps(results,indent=2))


if __name__=="__main__":main()

