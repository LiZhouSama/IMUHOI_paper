"""GT-conditioned diagnostic of object-axis and hand-pair offset priors.

Descriptive geometry and train-only lookup baselines; not a model evaluation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import trimesh

DATASETS = {"OMOMO": "processed_split_data_OMOMO_full", "BEHAVE": "processed_split_data_BEHAVE"}


def shape(dataset, name):
    if dataset == "OMOMO":
        path = Path("datasets/OMOMO/captured_objects") / f"{name}_cleaned_simplified.obj"
    else:
        folder = Path("datasets/BEHAVE/objects") / name
        paths = sorted(list(folder.glob("*.ply")) + list(folder.glob("*.obj")),
                       key=lambda p: ("f1000" not in p.stem, len(p.name), p.name))
        path = paths[0]
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    # Matches preprocessing's object reference origin.
    vertex_mean = np.asarray(mesh.vertices).mean(0)
    np.random.seed(20260929)
    pts, _ = trimesh.sample.sample_surface(mesh, 20000, seed=20260929)
    pts -= vertex_mean
    centre = pts.mean(0)
    centered = pts-centre
    val, vec = np.linalg.eigh(centered.T @ centered / len(pts))
    return vec[:, -1].copy(), centre, float(np.sqrt(val[-1]/val[-2]))


def dist_summary(x, prefix):
    if not len(x):
        return {f"{prefix}_{key}": None for key in ("mean_cm", "median_cm", "p90_cm", "within5_pct", "within10_pct")}
    cm = np.asarray(x)*100
    return {f"{prefix}_mean_cm": float(cm.mean()), f"{prefix}_median_cm": float(np.median(cm)),
            f"{prefix}_p90_cm": float(np.percentile(cm, 90)),
            f"{prefix}_within5_pct": float(np.mean(cm <= 5)*100),
            f"{prefix}_within10_pct": float(np.mean(cm <= 10)*100)}


def knn(train_x, train_c, test_x, test_c, device):
    if not len(test_x):
        return np.array([]), np.array([]), np.array([])
    # Cap train support uniformly for bounded memory and reproducible cost.
    rng = np.random.RandomState(20260929)
    idx = rng.choice(len(train_x), min(12000, len(train_x)), replace=False)
    tx = torch.as_tensor(train_x[idx], dtype=torch.float32, device=device)
    tc = torch.as_tensor(train_c[idx], dtype=torch.float32, device=device)
    qx = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    errs, ambiguity, support = [], [], []
    for start in range(0, len(qx), 256):
        distances, indices = torch.cdist(qx[start:start+256], tx).topk(min(20, len(tx)), largest=False)
        neighbors = tc[indices]
        pred = neighbors.mean(1)
        truth = torch.as_tensor(test_c[start:start+256], dtype=torch.float32, device=device)
        errs.append((pred-truth).norm(dim=-1).cpu().numpy())
        ambiguity.append(((neighbors-pred[:,None])**2).sum(-1).mean(-1).sqrt().cpu().numpy())
        support.append(distances[:,0].cpu().numpy())
    return np.concatenate(errs), np.concatenate(ambiguity), np.concatenate(support)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/shape_relative_prior_20260929"))
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device=torch.device("cuda:0")
    pools=defaultdict(lambda: defaultdict(list))
    shapes={}
    for dataset, directory in DATASETS.items():
        root=Path("process/IMUHOI++cano")/directory
        sets={s:{p.name for p in (root/s).glob("*.pt")} for s in ("train","test")}
        assert not sets["train"] & sets["test"]
        paths=sorted((root/"unified").glob("*.pt"))
        for i,path in enumerate(paths):
            data=torch.load(path,map_location="cpu",weights_only=False)
            mask=np.asarray(data["lhand_contact"],dtype=bool)&np.asarray(data["rhand_contact"],dtype=bool)
            if not mask.any():
                continue
            split=next((s for s in sets if path.name in sets[s]), None)
            if split is None:
                raise ValueError(f"Unassigned bimanual sequence: {path}")
            name=str(data["obj_name"])
            key=(dataset,name)
            if key not in shapes:
                shapes[key]=shape(dataset,name)
            pos=np.asarray(data["position_global_full_gt_world"],dtype=np.float64)[mask][:,[22,23]]
            obj=np.asarray(data["obj_trans"],dtype=np.float64)[mask].reshape(-1,3)
            rot=np.asarray(data["obj_rot"],dtype=np.float64)[mask]
            mid=pos.mean(1)
            c=np.einsum("tji,tj->ti",rot,obj-mid)
            d=np.einsum("tji,tj->ti",rot,pos[:,1]-pos[:,0])
            # Forearm orientation features are GT proxies for available arm IMUs.
            global_rot=np.asarray(data["rotation_global"],dtype=np.float64)[mask][:,[18,19]]
            rel_rot=np.einsum("tji,thjk->thik",rot,global_rot)
            arm=rel_rot[...,:2].reshape(len(c),-1)
            scale=np.asarray(data["obj_scale"],dtype=np.float64)[mask].reshape(-1)
            assert np.max(np.abs(np.einsum("tji,tjk->tik", rot, rot)-np.eye(3))) < 1e-5, path
            span=np.linalg.norm(d,axis=-1)
            valid=np.isfinite(c).all(-1)&np.isfinite(d).all(-1)&(span>1e-6)
            assert valid.all(), path
            # All entries in the same frame are in meters; only directions use mesh coordinates.
            pools[key][split].append(np.column_stack((c,d,arm,scale)))
            if (i+1)%500==0:
                print(dataset, i+1, len(paths), flush=True)
    observed_counts = {ds: sum(len(a) for (dataset, _), splits in pools.items() if dataset == ds for arrays in splits.values() for a in arrays) for ds in DATASETS}
    assert observed_counts == {"OMOMO": 376998, "BEHAVE": 136579}, observed_counts
    rows=[]
    summaries={}
    for (dataset,name), splits in sorted(pools.items()):
        train=np.concatenate(splits["train"])
        test=np.concatenate(splits["test"]) if splits["test"] else np.empty((0, 19))
        allv=np.concatenate((train,test))
        axis,center,elongation=shapes[(dataset,name)]
        c=allv[:,:3]
        d=allv[:,3:6]
        # Midpoint-to-object offset decomposed along the object's intrinsic axis.
        axial=c@axis
        radial=c-axial[:,None]*axis
        geom=(-c-allv[:,-1,None]*center)
        geom-= (geom@axis)[:,None]*axis
        train_c,test_c=train[:,:3],test[:,:3]
        mean=train_c.mean(0)
        radial_test=test_c-(test_c@axis)[:,None]*axis
        radial_mean=mean-(mean@axis)*axis
        plane=np.abs(np.sum(c*d,axis=-1))/np.linalg.norm(d,axis=-1)
        line=np.sqrt(np.maximum((c*c).sum(-1)-plane*plane,0))
        covariance=np.cov(train_c.T)
        evals,evecs=np.linalg.eigh(covariance)
        cos=np.abs(d@axis)/np.linalg.norm(d,axis=-1)
        alignment=np.degrees(np.arccos(cos.clip(0,1)))
        sample=np.linspace(0,len(test)-1,min(4000,len(test)),dtype=int)
        # Geometry-only and geometry + object-relative forearm orientation.
        err_d,amb_d,near_d=knn(train[:,3:6],train_c,test[sample,3:6],test_c[sample],device)
        train_feat=np.concatenate((train[:,3:6],train[:,6:18]*0.1),-1)
        test_feat=np.concatenate((test[sample,3:6],test[sample,6:18]*0.1),-1)
        err_arm,amb_arm,_=knn(train_feat,train_c,test_feat,test_c[sample],device)
        row={"dataset":dataset,"object":name,"train_frames":len(train),"test_frames":len(test),
             "lookup_test_sample_frames":len(sample),"elongation_sd_ratio":elongation,
             "hand_line_to_object_axis_angle_median_deg":float(np.median(alignment)),
             "axial_offset_std_cm":float(np.std(axial)*100),
             "covariance_std_1_cm":float(np.sqrt(evals[-1])*100),
             "covariance_std_2_cm":float(np.sqrt(evals[-2])*100),
             "covariance_std_3_cm":float(np.sqrt(evals[-3])*100),
             "covariance_mainaxis_vs_mesh_axis_deg":float(np.degrees(np.arccos(np.abs(evecs[:,-1]@axis).clip(0,1)))),
             "pair_geometry_neighbor_offset_spread_median_cm":float(np.median(amb_d)*100) if len(amb_d) else None,
             "pair_geometry_nearest_support_median_cm":float(np.median(near_d)*100) if len(near_d) else None,
             **dist_summary(line,"all_hand_line"),
             **dist_summary(np.linalg.norm(radial,axis=-1),"all_object_origin_axis"),
             **dist_summary(np.linalg.norm(geom,axis=-1),"all_mesh_surface_centroid_axis"),
             **dist_summary(np.linalg.norm(radial_test-radial_mean,axis=-1),"test_trainmean_radial_residual"),
             **dist_summary(np.abs(test_c@axis-mean@axis),"test_trainmean_axial_residual"),
             **dist_summary(np.linalg.norm(test_c[sample]-mean,axis=-1),"test_object_mean_offset_error"),
             **dist_summary(err_d,"test_gt_pair_knn_error"),
             **dist_summary(err_arm,"test_gt_pair_arm_knn_error")}
        rows.append(row)
        summaries[f"{dataset}/{name}"]={"mean_c_object_m":mean.tolist(),"cov_c_object_m2":covariance.tolist(),"mesh_axis":axis.tolist(),"mesh_surface_centroid":center.tolist()}
        print(dataset,name,json.dumps(row),flush=True)
    fields=list(rows[0])
    with (args.output/"per_object.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (args.output/"priors.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    meta={"data_root":"process/IMUHOI++cano","definitions":{"c":"R_obj.T @ (obj-midpoint)","d":"R_obj.T @ (right-left)"},
          "conditioning":"GT hands, GT object rotation, and GT forearm rotations, known object identity; not predicted-input performance",
          "lookup":"20-NN mean, at most 12000 train frames/object; <=4000 test frames/object; original train/test folders; orientation matrix first 2 columns weighted 0.1 m", "all_frames":sum(r["train_frames"]+r["test_frames"] for r in rows)}
    (args.output/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("Saved", args.output,flush=True)


if __name__=="__main__":
    main()


