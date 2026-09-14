#!/usr/bin/env python3
"""Is the grasp region on the mesh predictable from test-time-observable context?

Per object: K anchors = K-means of TRAIN contact tips (canonical metres).
One sample per GT contact segment (onset frame t0), features observable at test time:
  gravity direction in the object frame (R_obj^T e_y)            3   [object IMU]
  R_obj^T R_forearm (6D)                                          6   [object + forearm IMU]
  palm relative to root, heading-normalised                       3   [HP]
  palm height, root height                                        2   [HP]
  hand id, other hand in contact                                  2   [VC]
Target: anchor id of the segment-mean tip.  Multinomial logistic regression per
object, fit on train, evaluated on test:
  accuracy vs majority class, and the tip error (cm) when the bone is set to the
  predicted anchor, vs. the object-level mean tip (k=1) and the oracle anchor.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TRAIN = os.path.join(ROOT, "process/IMUHOI++cano/processed_split_data_OMOMO_full/train")
TEST = os.path.join(ROOT, "process/IMUHOI++cano/processed_split_data_OMOMO_full/test")
OUT = os.path.join(ROOT, "outputs/analysis/mesh_probe_20260907", "grasp_region_probe.json")
PALM = (22, 23)
FORE = (18, 19)
MIN_SEG = 10
K = 8


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


def heading_yaw(Rroot):
    a = np.arctan2(Rroot[0, 2] - Rroot[2, 0], Rroot[0, 0] + Rroot[2, 2])
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def samples_of(split):
    out = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(split, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if not d.get("has_object", False):
            continue
        R = d["obj_rot"].double().numpy()
        p = d["obj_trans"].double().numpy()
        J = d["position_global_full_gt_world"].double().numpy()
        Rg = d["rotation_global"].double().numpy()
        cms = (d["lhand_contact"].bool(), d["rhand_contact"].bool())
        for k in range(2):
            for a, b in segments(cms[k]):
                if b - a + 1 < MIN_SEG:
                    continue
                t = np.arange(a, b + 1)
                tips = np.einsum("tji,tj->ti", R[t], J[t, PALM[k]] - p[t])
                tip_mean = tips.mean(0)
                t0 = a
                g_obj = R[t0].T @ np.array([0.0, 1.0, 0.0])
                rel = R[t0].T @ Rg[t0, FORE[k]]
                yaw = heading_yaw(Rg[t0, 0])
                palm_rel = yaw.T @ (J[t0, PALM[k]] - J[t0, 0])
                other = float(cms[1 - k][t0])
                feat = np.concatenate((g_obj, rel[:, :2].T.reshape(-1), palm_rel, [J[t0, PALM[k], 1], J[t0, 0, 1], float(k), other]))
                out[d["obj_name"]].append((feat, tip_mean, tips[0]))
    return {n: (np.stack([f for f, _, _ in v]), np.stack([m for _, m, _ in v]), np.stack([o for _, _, o in v])) for n, v in out.items()}


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
    tr = samples_of(TRAIN)
    te = samples_of(TEST)
    rep = {}
    tot = defaultdict(float)
    n_tot = 0
    for name in sorted(te):
        Xtr, Mtr, _ = tr[name]
        Xte, Mte, Ote = te[name]
        k = min(K, len(Xtr) // 4)
        C = kmeans(Mtr, k)
        ytr = ((Mtr[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        yte = ((Mte[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
        pred = clf.predict(sc.transform(Xte))
        acc = float((pred == yte).mean())
        maj = float((yte == np.bincount(ytr).argmax()).mean())
        err_pred = float(np.linalg.norm(C[pred] - Mte, axis=-1).mean() * 100)
        err_oracle = float(np.linalg.norm(C[yte] - Mte, axis=-1).mean() * 100)
        err_mean = float(np.linalg.norm(Mtr.mean(0) - Mte, axis=-1).mean() * 100)
        err_first = float(np.linalg.norm(Ote - Mte, axis=-1).mean() * 100)  # frozen first-frame tip vs segment mean
        rep[name] = dict(n_train=int(len(Xtr)), n_test=int(len(Xte)), k=int(k), acc=acc, majority=maj,
                         tip_err_pred_anchor=err_pred, tip_err_oracle_anchor=err_oracle, tip_err_object_mean=err_mean,
                         tip_err_first_frame=err_first)
        for key in ("acc", "majority", "tip_err_pred_anchor", "tip_err_oracle_anchor", "tip_err_object_mean", "tip_err_first_frame"):
            tot[key] += rep[name][key] * len(Xte)
        n_tot += len(Xte)
        print(f"{name:13s} n={len(Xte):4d} k={k} acc={acc:.2f} maj={maj:.2f} | tip err: pred-anchor={err_pred:.1f} oracle-anchor={err_oracle:.1f} obj-mean={err_mean:.1f} first-frame={err_first:.1f}", flush=True)
    rep["_overall"] = {key: val / n_tot for key, val in tot.items()}
    print(json.dumps(rep["_overall"], indent=2))
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=2)


if __name__ == "__main__":
    main()
