#!/usr/bin/env python3
"""Can a per-object grasp-anchor set (learned from the mesh + train contacts)
denoise an onset bone vector that carries the pipeline's hand error?

For every object: K-means anchors on TRAIN contact tips (canonical metres).
On TEST contact tips add isotropic noise sigma (mimicking the onset bone
error) and compare
  raw          |noisy - tip|
  surface      snap the noisy tip to the nearest mesh surface point (hard projection)
  anchorK      snap to the nearest of K anchors
  oracleK      nearest anchor to the TRUE tip (quantisation floor)
frame-weighted, cm.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_nn import SurfaceNN, load_mesh  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TRAIN = os.path.join(ROOT, "process/IMUHOI++cano/processed_split_data_OMOMO_full/train")
TEST = os.path.join(ROOT, "process/IMUHOI++cano/processed_split_data_OMOMO_full/test")
OUT = os.path.join(ROOT, "outputs/analysis/mesh_probe_20260907", "grasp_anchor_probe.json")
PALM = (22, 23)
STRIDE = 5
SIGMAS = (0.04, 0.08, 0.12)
KS = (4, 8, 16, 32)


def tips_of(split):
    out = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(split, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if not d.get("has_object", False):
            continue
        R = d["obj_rot"].double()
        p = d["obj_trans"].double()
        s = d["obj_scale"].double()
        h = d["position_global_full_gt_world"].double()[:, list(PALM)]
        for k, cm in enumerate((d["lhand_contact"].bool(), d["rhand_contact"].bool())):
            t = torch.where(cm)[0][::STRIDE]
            if t.numel() == 0:
                continue
            tip = torch.einsum("tji,tj->ti", R[t], h[t, k] - p[t])  # metres, canonical (scaled)
            out[d["obj_name"]].append((tip.numpy(), s[t].numpy()))
    return {k: (np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for k, v in out.items()}


def kmeans(x, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    return c


def main():
    tr = tips_of(TRAIN)
    te = tips_of(TEST)
    rng = np.random.default_rng(0)
    rep = {}
    agg = defaultdict(list)
    for name in sorted(te):
        x_tr, _ = tr[name]
        x_te, s_te = te[name]
        nn = SurfaceNN(load_mesh(name))
        anchors = {k: kmeans(x_tr, min(k, len(x_tr) // 2)) for k in KS}
        r = {"n_train": int(len(x_tr)), "n_test": int(len(x_te))}
        for k, c in anchors.items():
            r[f"oracle{k}"] = float(np.sqrt(((x_te[:, None] - c[None]) ** 2).sum(-1).min(1)).mean() * 100)
        for sg in SIGMAS:
            noisy = x_te + rng.normal(0, sg, x_te.shape)
            raw = np.linalg.norm(noisy - x_te, axis=-1)
            cl, _ = nn.query(torch.from_numpy(noisy / s_te[:, None]))
            surf = np.linalg.norm(cl.cpu().numpy() * s_te[:, None] - x_te, axis=-1)
            row = {"raw": float(raw.mean() * 100), "surface": float(surf.mean() * 100)}
            for k, c in anchors.items():
                idx = ((noisy[:, None] - c[None]) ** 2).sum(-1).argmin(1)
                row[f"anchor{k}"] = float(np.linalg.norm(c[idx] - x_te, axis=-1).mean() * 100)
            r[f"sigma{int(sg*100)}"] = row
            for key, val in row.items():
                agg[(int(sg * 100), key)].append((val, len(x_te)))
        rep[name] = r
        print(name, json.dumps(r["sigma8"]), "oracle8", round(r["oracle8"], 2), flush=True)
    overall = {}
    for (sg, key), vals in agg.items():
        w = np.array([n for _, n in vals], dtype=np.float64)
        v = np.array([x for x, _ in vals])
        overall[f"sigma{sg}_{key}"] = float((v * w).sum() / w.sum())
    rep["_overall_frame_weighted"] = overall
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
