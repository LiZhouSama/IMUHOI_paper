#!/usr/bin/env python3
"""GT-only oracle: how much does the object mesh constrain the hand-object geometry?

Measured on the canonical OMOMO test split with GT joints / object pose / contacts:
  A. palm-anchor -> mesh-surface distance on contact frames (how tight is the
     'bone tip lies on the surface' constraint)
  B. object support: lowest mesh vertex height on static vs moving frames
     (is 'object rests on the floor' an observable constraint on obj y)
  C. contact-point clustering per object in the canonical frame (is the grasp
     location a low-entropy, learnable target)
  D. bone-vector ladder with mesh: frozen first-frame bone (L0) vs. the same
     with its tip snapped to the surface, and a 'nearest cluster centre' prior.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import trimesh
from mesh_nn import SurfaceNN, load_mesh as _load_mesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "process/IMUHOI++cano/processed_split_data_OMOMO_full/test")
MESH_DIR = os.path.join(ROOT, "datasets/OMOMO/captured_objects")
OUT = os.path.join(ROOT, "outputs/analysis/mesh_probe_20260907", "mesh_oracle_gt.json")
PALM = (22, 23)
MIN_SEG = 10
STRIDE = 3  # subsample contact frames for the proximity queries


def load_mesh(name):
    path = os.path.join(MESH_DIR, f"{name}_cleaned_simplified.obj")
    m = trimesh.load_mesh(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    v = np.asarray(m.vertices, dtype=np.float64)
    v = v - v.mean(axis=0)  # preprocess centres at the vertex centroid
    return trimesh.Trimesh(vertices=v, faces=np.asarray(m.faces), process=False)


def segments(mask):
    idx = torch.where(mask)[0].tolist()
    out = []
    if not idx:
        return out
    s = p = idx[0]
    for i in idx[1:]:
        if i != p + 1:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return out


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.pt")))
    meshes = {}
    queries = {}
    # per-object accumulators
    surf_d = defaultdict(list)          # palm->surface distance (m) on contact frames
    cpoints = defaultdict(list)         # closest surface point, canonical metres
    cframes = defaultdict(list)         # (file idx, hand, t) for later use
    floor_static = defaultdict(list)    # min vertex y on static frames
    floor_moving = defaultdict(list)    # min vertex y on moving frames
    objy_static = defaultdict(list)
    bone_L0 = defaultdict(list)         # frozen first-frame bone error (cm), frame-weighted
    bone_L0_snap = defaultdict(list)    # same bone, tip snapped to surface at onset
    bone_len = defaultdict(list)
    n_contact_frames = 0
    per_seq = []
    for fi, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if not d.get("has_object", False):
            continue
        name = d["obj_name"]
        if name not in meshes:
            meshes[name] = _load_mesh(name)
            queries[name] = SurfaceNN(meshes[name])
        mesh, pq = meshes[name], queries[name]
        R = d["obj_rot"].double()            # [T,3,3]
        p = d["obj_trans"].double()          # [T,3]
        s = d["obj_scale"].double()          # [T]
        J = d["position_global_full_gt_world"].double()
        h = J[:, list(PALM)]                 # [T,2,3]
        lc, rc = d["lhand_contact"].bool(), d["rhand_contact"].bool()
        moving = d["obj_contact"].bool()
        T = R.shape[0]

        # ---- B. support / floor statistics (all frames) ----
        min_y = pq.min_y_world(R, s, p).cpu().double()
        floor_static[name].extend(min_y[~moving].tolist())
        floor_moving[name].extend(min_y[moving].tolist())
        objy_static[name].extend(p[~moving, 1].tolist())

        # ---- A/C/D. contact geometry ----
        for hand, cm in enumerate((lc, rc)):
            segs = [(a, b) for a, b in segments(cm) if b - a + 1 >= MIN_SEG]
            for a, b in segs:
                t = torch.arange(a, b + 1)
                # bone tip in canonical (unscaled) mesh frame: R^T (h - p) / s
                tip = torch.einsum("tji,tj->ti", R[t], h[t, hand] - p[t]) / s[t].unsqueeze(-1)
                sub = t[::STRIDE] - a
                closest, dist = pq.query(tip[sub])
                closest, dist = closest.cpu().double().numpy(), dist.cpu().double().numpy()
                dist_m = dist * s[t[sub]].numpy()
                surf_d[name].extend(dist_m.tolist())
                cpoints[name].extend((closest * s[t[sub]].numpy()[:, None]).tolist())
                n_contact_frames += len(sub)
                # bone ladder: frozen first-frame bone in object frame (metres)
                c = torch.einsum("tji,tj->ti", R[t], p[t] - h[t, hand])  # [n,3]
                c0 = c[0]
                err0 = (c - c0).norm(dim=-1) * 100.0
                bone_L0[name].extend(err0.tolist())
                bone_len[name].extend(c.norm(dim=-1).tolist())
                # snap the frozen tip (-c0, canonical unscaled) to the surface once
                cl0, _ = pq.query((-c0 / s[a]).unsqueeze(0))
                c0_snap = -cl0[0].cpu().double() * s[a]
                bone_L0_snap[name].extend(((c - c0_snap).norm(dim=-1) * 100.0).tolist())
        per_seq.append(name)
        if fi % 100 == 0:
            print(f"{fi}/{len(files)} contact frames so far {n_contact_frames}", flush=True)

    def q(x, qs=(0.1, 0.5, 0.9)):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return [None] * len(qs)
        return [float(v) for v in np.quantile(x, qs)]

    report = {"objects": {}, "all": {}}
    all_surf, all_L0, all_L0s = [], [], []
    for name in sorted(meshes):
        sd = np.asarray(surf_d[name])
        fs = np.asarray(floor_static[name])
        fm = np.asarray(floor_moving[name])
        pts = np.asarray(cpoints[name])
        # C. k-means on contact points (k=6), within-cluster rms vs total rms
        kinfo = {}
        if len(pts) > 50:
            rng = np.random.default_rng(0)
            for k in (1, 3, 6, 10):
                cent = pts[rng.choice(len(pts), k, replace=False)]
                for _ in range(30):
                    dmat = ((pts[:, None] - cent[None]) ** 2).sum(-1)
                    lab = dmat.argmin(1)
                    for j in range(k):
                        if (lab == j).any():
                            cent[j] = pts[lab == j].mean(0)
                dmin = np.sqrt(((pts[:, None] - cent[None]) ** 2).sum(-1).min(1))
                kinfo[f"k{k}_rms_cm"] = float(np.sqrt((dmin ** 2).mean()) * 100)
                kinfo[f"k{k}_median_cm"] = float(np.median(dmin) * 100)
        report["objects"][name] = {
            "n_contact_samples": int(sd.size),
            "palm_surface_dist_cm_q10_50_90": [v * 100 for v in q(sd)],
            "palm_surface_dist_cm_mean": float(sd.mean() * 100) if sd.size else None,
            "frac_within_3cm": float((sd < 0.03).mean()) if sd.size else None,
            "frac_within_5cm": float((sd < 0.05).mean()) if sd.size else None,
            "bone_len_cm_q10_50_90": [v * 100 for v in q(bone_len[name])],
            "floor_min_y_static_cm_q10_50_90": [v * 100 for v in q(fs)],
            "floor_min_y_moving_cm_q10_50_90": [v * 100 for v in q(fm)],
            "frac_static_resting_2cm": float((np.abs(fs) < 0.02).mean()) if fs.size else None,
            "frac_moving_resting_2cm": float((np.abs(fm) < 0.02).mean()) if fm.size else None,
            "frac_static_resting_5cm": float((np.abs(fs) < 0.05).mean()) if fs.size else None,
            "frac_moving_resting_5cm": float((np.abs(fm) < 0.05).mean()) if fm.size else None,
            "obj_y_static_std_cm": float(np.std(objy_static[name]) * 100) if objy_static[name] else None,
            "n_static_frames": int(fs.size),
            "n_moving_frames": int(fm.size),
            "bone_L0_cm_mean": float(np.mean(bone_L0[name])),
            "bone_L0_snap_cm_mean": float(np.mean(bone_L0_snap[name])),
            "contact_point_clusters": kinfo,
        }
        all_surf.extend(sd.tolist())
        all_L0.extend(bone_L0[name])
        all_L0s.extend(bone_L0_snap[name])
    all_fs = np.concatenate([np.asarray(v) for v in floor_static.values()])
    all_fm = np.concatenate([np.asarray(v) for v in floor_moving.values()])
    report["all"] = {
        "n_contact_samples": len(all_surf),
        "palm_surface_dist_cm_q10_50_90": [v * 100 for v in q(all_surf)],
        "palm_surface_dist_cm_mean": float(np.mean(all_surf) * 100),
        "frac_within_3cm": float((np.asarray(all_surf) < 0.03).mean()),
        "frac_within_5cm": float((np.asarray(all_surf) < 0.05).mean()),
        "frac_static_resting_2cm": float((np.abs(all_fs) < 0.02).mean()),
        "frac_moving_resting_2cm": float((np.abs(all_fm) < 0.02).mean()),
        "frac_static_resting_5cm": float((np.abs(all_fs) < 0.05).mean()),
        "frac_moving_resting_5cm": float((np.abs(all_fm) < 0.05).mean()),
        "floor_min_y_static_cm_q10_50_90": [v * 100 for v in q(all_fs)],
        "floor_min_y_moving_cm_q10_50_90": [v * 100 for v in q(all_fm)],
        "bone_L0_cm_mean": float(np.mean(all_L0)),
        "bone_L0_snap_cm_mean": float(np.mean(all_L0s)),
        "n_sequences": len(per_seq),
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report["all"], indent=2))
    for name, r in report["objects"].items():
        print(
            f"{name:13s} surf q50={r['palm_surface_dist_cm_q10_50_90'][1]:.1f} q90={r['palm_surface_dist_cm_q10_50_90'][2]:.1f} "
            f"<3cm={r['frac_within_3cm']:.2f} | floor static q50={r['floor_min_y_static_cm_q10_50_90'][1]:.1f} "
            f"rest2cm s={r['frac_static_resting_2cm']:.2f} m={r['frac_moving_resting_2cm']:.2f} | "
            f"L0={r['bone_L0_cm_mean']:.2f} L0snap={r['bone_L0_snap_cm_mean']:.2f} | "
            f"k1={r['contact_point_clusters'].get('k1_rms_cm', 0):.1f} k6={r['contact_point_clusters'].get('k6_rms_cm', 0):.1f}"
        )


if __name__ == "__main__":
    main()
