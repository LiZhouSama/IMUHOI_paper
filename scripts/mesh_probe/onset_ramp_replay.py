#!/usr/bin/env python3
"""Replay a soft FK hand-over at contact onsets on an online dump (publisher level).

At each hysteresis onset of hand k (t0), the FK weight of that hand is scaled by
ramp(t) = min(1, (t - t0 + 1) / K) for K frames and the removed weight is given to
the static branch, so the object leaves its (well known) pre-onset position
gradually instead of jumping onto a 7-8 cm-error bone proposal.  Nothing else
changes.  Reports published obj / HOI / jitter for several K.  Usage:

    python scripts/mesh_probe/onset_ramp_replay.py label=dump.pt [...] --K 5 10 15 30
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.analyze_ot_dumps import hysteresis, rot6d, segments_of  # noqa: E402

ON, OFF = 0.15, 0.075


def norm(x):
    return np.linalg.norm(x, axis=-1)


def f64(t):
    return t.numpy().astype(np.float64)


def publish(h, Rb, w, d_imu, p0):
    T = h.shape[0]
    out = np.zeros((T, 3))
    prev = p0.copy()
    for t in range(T):
        p = w[t, 0] * (h[t, 0] + Rb[t, 0]) + w[t, 1] * (h[t, 1] + Rb[t, 1]) + w[t, 2] * (prev + d_imu[t]) + w[t, 3] * prev
        out[t] = p
        prev = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--K", type=int, nargs="+", default=[5, 10, 15, 30])
    args = ap.parse_args()
    for spec in args.dumps:
        label, path = spec.split("=", 1)
        recs = torch.load(path, map_location="cpu", weights_only=False)["records"]
        res = {k: {"obj": [], "hoi": [], "jit": []} for k in [0] + args.K}
        for rec in recs:
            p_gt = f64(rec["gt_obj_trans"]); h_gt = f64(rec["gt_palm_pos"]); h_r = f64(rec["pred_palm_glb_pos"])
            h_s = f64(rec.get("ot_hand_positions", rec["stage1_pred_palm_glb_pos"]))
            c_gt = np.stack((rec["gt_lhand_contact"].numpy(), rec["gt_rhand_contact"].numpy()), 1).astype(bool)
            prob = f64(rec["pred_hand_contact_prob"]); w0 = f64(rec["gating_weights"])
            fk = np.stack((f64(rec["pred_lhand_obj_trans"]), f64(rec["pred_rhand_obj_trans"])), 1)
            Rb = fk - h_s
            d_imu = f64(rec["pred_imu_obj_trans"]) - f64(rec["pred_static_obj_trans"])
            p0 = f64(rec["pred_static_obj_trans"])[0]
            c_h = hysteresis(prob[:, :2], ON, OFF)
            T = h_r.shape[0]
            for K in [0] + args.K:
                w = w0.copy()
                if K > 0:
                    for k in range(2):
                        for a, b in segments_of(c_h[:, k]):
                            if a == 0:
                                continue
                            n = min(K, b - a + 1)
                            ramp = np.minimum(1.0, (np.arange(n) + 1) / K)
                            removed = w[a : a + n, k] * (1 - ramp)
                            w[a : a + n, k] -= removed
                            w[a : a + n, 3] += removed
                p = publish(h_r, Rb, w, d_imu, p0)
                res[K]["obj"].append(norm(p - p_gt).mean() * 100)
                hoi = [norm((p - h_r[:, k]) - (p_gt - h_gt[:, k]))[c_gt[:, k]] for k in range(2) if c_gt[:, k].any()]
                res[K]["hoi"].append(np.concatenate(hoi).mean() * 100 if hoi else np.nan)
                if T >= 3:
                    res[K]["jit"].append(norm(p[2:] - 2 * p[1:-1] + p[:-2]).mean() * 1000)
        print(f"== {label} ({len(recs)} seq)")
        for K, r in res.items():
            print(f"   K={K:3d}: obj {np.nanmean(r['obj']):.3f} | hoi {np.nanmean(r['hoi']):.3f} | jitter {np.nanmean(r['jit']):.3f}")


if __name__ == "__main__":
    main()
