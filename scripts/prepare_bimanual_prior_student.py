"""Prepare frozen prediction inputs and training-only mesh auxiliary targets."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.rnn.contact_residual import build_contact_residual_features

DATA={"OMOMO":"processed_split_data_OMOMO_full","BEHAVE":"processed_split_data_BEHAVE"}


def json_write(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")


def partition(rows):
    groups=defaultdict(set)
    for r in rows:groups[r["object"]].add(r["group"])
    mapping={}
    for name,keys in sorted(groups.items()):
        keys=sorted(keys,key=lambda k:hashlib.sha256(("1001:"+k).encode()).hexdigest())
        n=len(keys);nv=max(1,round(n*.15));nc=max(1,round(n*.15))
        if n<4:raise ValueError((name,n,"Not enough source groups"))
        for i,k in enumerate(keys):mapping[k]="validation" if i<nv else "calibration" if i<nv+nc else "fit"
    for r in rows:r["partition"]=mapping[r["group"]]
    return rows


def observed(record):
    f=build_contact_residual_features(record)
    h=f["pred_palm"];r=f["obj_rotation"];p=f["base_obj"]
    c=torch.einsum("tji,tj->ti",r,p-h.mean(1))
    d=torch.einsum("tji,tj->ti",r,h[:,1]-h[:,0])
    span=d.norm(dim=-1)
    support=f["contact_prob"][:,:2].amin(-1)
    age=[];counter=0
    for yes in support>=.5:
        counter=counter+1 if yes else 0;age.append(min(counter,150)/150)
    extra=torch.stack((span,(c*d).sum(-1)/span.clamp_min(.05),support,torch.tensor(age)),dim=-1)
    return {"x":torch.cat((f["features"],extra),-1).float(),"R":r.float(),"cp":f["contact_prob"].float(),
            "p":p.float(),"h":h.float(),"c":c.float()}


def mesh_geometry(ds,name):
    import trimesh
    from scipy.spatial import cKDTree
    from sklearn.cluster import KMeans
    if ds=="OMOMO":path=ROOT/"datasets/OMOMO/captured_objects"/f"{name}_cleaned_simplified.obj"
    else:
        folder=ROOT/"datasets/BEHAVE/objects"/name
        paths=sorted(folder.glob("*.ply"))+sorted(folder.glob("*.obj"))
        path=paths[0]
    mesh=trimesh.load_mesh(path,process=False)
    if isinstance(mesh,trimesh.Scene):mesh=trimesh.util.concatenate(tuple(mesh.geometry.values()))
    pts,faces=trimesh.sample.sample_surface(mesh,12000,seed=1001)
    pts-=np.asarray(mesh.vertices).mean(0)
    normals=np.asarray(mesh.face_normals)[faces]
    spread=float(np.sqrt((pts**2).sum(-1).mean()))
    clusters=KMeans(n_clusters=8,n_init=5,random_state=1001).fit_predict(pts/max(spread,1e-5))
    return {"points":pts,"normals":normals,"clusters":clusters,"tree":cKDTree(pts),"path":str(path)}


def mesh_targets(source,n,geometry):
    scale=np.asarray(source["obj_scale"],float)[:n].reshape(-1)
    rot=np.asarray(source["obj_rot"],float)[:n]
    palms=np.asarray(source["position_global_full_gt_world"],float)[:n,[22,23]]
    obj=np.asarray(source["obj_trans"],float)[:n].reshape(-1,3)
    local=np.einsum("tji,thj->thi",rot,palms-obj[:,None])
    queries=local/np.maximum(scale[:,None,None],1e-6)
    distance,indices=geometry["tree"].query(queries.reshape(-1,3))
    points=geometry["points"][indices].reshape(n,2,3)*scale[:,None,None]
    normals=geometry["normals"][indices].reshape(n,2,3)
    patch=geometry["clusters"][indices].reshape(n,2)
    distance=distance.reshape(n,2)*scale[:,None]
    confidence=np.exp(-.5*(distance/.07)**2)*(scale[:,None]>1e-6)
    return torch.tensor(patch,dtype=torch.long),torch.tensor(np.concatenate((points.reshape(n,6),normals.reshape(n,6)),-1),dtype=torch.float32),torch.tensor(confidence,dtype=torch.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"outputs/experiments/meshfree_temporal_20261001")
    parser.add_argument("--dataset",choices=tuple(DATA),required=True)
    parser.add_argument("--split",choices=("train","test"),required=True)
    args=parser.parse_args();torch.set_num_threads(2);out=args.root;ds=args.dataset
    payload=torch.load(out/f"{ds}_{args.split}.pt",map_location="cpu",weights_only=False)
    assert payload["split"]==args.split
    record_list=payload["records"]
    rows=[]
    if args.split=="train":
        for record in record_list:
            path=ROOT/"process/IMUHOI++cano"/DATA[ds]/"train"/record["seq_file"]
            source=torch.load(path,map_location="cpu",weights_only=False)
            original=str(source["seq_name"])
            group=original.rsplit("_",1)[0]
            rows.append({"file":record["seq_file"],"object":record["obj_name"],"source_name":original,"group":group})
        partition(rows);json_write(out/f"{ds}_student_split.json",rows)
    else:
        # Source-name metadata is used only for clustered uncertainty reporting.
        metadata_path=ROOT/"outputs/analysis/bimanual_midplane_20260928/per_sequence.csv"
        source_names={}
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8-sig") as handle:
                source_names={r["file"]:r["seq_name"] for r in csv.DictReader(handle) if r["dataset"]==ds}
        for record in record_list:
            original=source_names.get(record["seq_file"],record["seq_file"])
            rows.append({"file":record["seq_file"],"object":record["obj_name"],"partition":"test","group":original.rsplit("_",1)[0]})
    membership={r["file"]:r for r in rows}
    prepared=[];geometry={}
    for i,record in enumerate(record_list):
        n=int(record["publish_len"])
        if n<3:continue
        r=observed(record);meta=membership[record["seq_file"]]
        r.update(meta)
        r["gt"]=record["gt_obj_trans"][:n].float()
        r["gh"]=record.get("gt_palm_pos_eval",record["gt_palm_pos"])[:n].float()
        r["contact"]=torch.stack((record["gt_lhand_contact"][:n],record["gt_rhand_contact"][:n]),-1).bool()
        r["residual_target"]=torch.einsum("tji,tj->ti",r["R"],r["gt"]-r["p"])
        r["geometry_target"]=torch.einsum("tji,tj->ti",r["R"],r["gt"]-r["gh"].mean(1))
        if args.split=="train":
            source_path=ROOT/"process/IMUHOI++cano"/DATA[ds]/"train"/r["file"]
            source=torch.load(source_path,map_location="cpu",weights_only=False)
            np.testing.assert_allclose(r["gt"].numpy(),np.asarray(source["obj_trans"])[:n],atol=1e-6)
            if r["object"] not in geometry:geometry[r["object"]]=mesh_geometry(ds,r["object"])
            r["patch_target"],r["surface_target"],r["mesh_confidence"]=mesh_targets(source,n,geometry[r["object"]])
        for key,v in r.items():
            if isinstance(v,torch.Tensor) and not torch.isfinite(v).all():raise ValueError((r["file"],key))
        prepared.append(r)
        if (i+1)%500==0:print(ds,args.split,"prepared",i+1,len(record_list),flush=True)
    package={"records":prepared,"source_provenance":payload.get("provenance",{}),"names":sorted({r["object"] for r in rows}),
             "mesh_paths_training_only":{k:v["path"] for k,v in geometry.items()},"feature_dim":int(prepared[0]["x"].shape[-1]),"split":args.split}
    torch.save(package,out/f"{ds}_{args.split}_prepared.pt")
    counts={s:sum(r["partition"]==s for r in prepared) for s in ("fit","validation","calibration","test")}
    print(ds,args.split,"DONE",counts,"frames",sum(len(r["p"]) for r in prepared),flush=True)
    json_write(out/f"{ds}_{args.split}_prepared_summary.json",{"counts":counts,"frames":sum(len(r["p"]) for r in prepared),"names":package["names"],"feature_dim":package["feature_dim"],"mesh_target_split":args.split if geometry else None})


if __name__=="__main__":main()

