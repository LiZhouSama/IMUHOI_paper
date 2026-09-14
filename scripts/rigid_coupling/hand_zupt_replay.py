#!/usr/bin/env python3
"""Hand ZUPT on IMU-static objects (publisher-level replay, no training).

While a hand is in (predicted) contact with an object whose IMU certifies it
static, the hand is planted in the world.  Like a foot-contact ZUPT, the whole
body is rigidly shifted so that the planted palm stays where it was when the
static contact began; the shift persists after the segment (drift accumulated
during the hold is removed for good).  The object publisher is re-run with the
shifted hands (dumped bones / gating weights reused).

Usage: python scripts/rigid_coupling/hand_zupt_replay.py label=dump.pt [...]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.analyze_ot_dumps import imu_static_detector, rot6d  # noqa: E402


def f64(t):
    return t.numpy().astype(np.float64)


def norm(x):
    return np.linalg.norm(x, axis=-1)


def replay(rec, prob_thr, min_static, ramp, veto, use_hand, oracle=False):
    p_gt = f64(rec["gt_obj_trans"]); T = p_gt.shape[0]
    palm_gt = f64(rec["gt_palm_pos"]); palm = f64(rec["pred_palm_glb_pos"])
    h = f64(rec["ot_hand_positions"]); root = f64(rec["root_trans_pred"]); root_gt = f64(rec["gt_trans"])
    c_gt = np.stack((rec["gt_lhand_contact"].numpy(), rec["gt_rhand_contact"].numpy()), 1).astype(bool)
    prob = f64(rec["pred_hand_contact_prob"])[:, :2]
    w = f64(rec["gating_weights"]).copy()
    fk = np.stack((f64(rec["pred_lhand_obj_trans"]), f64(rec["pred_rhand_obj_trans"])), 1)
    Rb = fk - h
    d_imu = f64(rec["pred_imu_obj_trans"]) - f64(rec["pred_static_obj_trans"])
    imu = f64(rec["gt_obj_imu"]); R = rot6d(rec["gt_obj_imu"][:, 3:])
    static = imu_static_detector(R, imu[:, :3])
    if oracle:
        # upper bound: GT contact labels and GT object rest (|v| < 2 cm/s)
        v = np.linalg.norm(np.gradient(p_gt, axis=0), axis=1) * 30.0
        static = v < 0.02
        prob = c_gt.astype(np.float64)
    if veto:
        w[static] = np.array([0.0, 0.0, 0.0, 1.0])
    anchor_pt = palm if use_hand == "palm" else h
    shift = np.zeros(3); target = np.zeros(3); shift_hist = np.zeros((T, 3))
    static_run = 0; planted = np.zeros(2, dtype=bool); anchor = np.zeros((2, 3)); n_seg = 0; n_frames = 0
    for t in range(T):
        static_run = static_run + 1 if static[t] else 0
        for k in range(2):
            active = static_run >= min_static and prob[t, k] > prob_thr
            if active and not planted[k]:
                anchor[k] = anchor_pt[t, k] + target; planted[k] = True; n_seg += 1
            elif not active:
                planted[k] = False
        if planted.any():
            # rigid body shift that keeps the planted palm(s) at their anchors
            corr = np.mean([anchor[k] - (anchor_pt[t, k] + target) for k in range(2) if planted[k]], 0)
            target = target + corr; n_frames += 1
        shift = shift + (target - shift) / ramp if ramp > 1 else target.copy()
        shift_hist[t] = shift
    hs = h + shift_hist[:, None, :]
    p = np.zeros((T, 3)); prev = f64(rec["pred_static_obj_trans"])[0].copy()
    for t in range(T):
        p[t] = w[t, 0] * (hs[t, 0] + Rb[t, 0]) + w[t, 1] * (hs[t, 1] + Rb[t, 1]) + w[t, 2] * (prev + d_imu[t]) + w[t, 3] * prev
        prev = p[t]
    palm_s = palm + shift_hist[:, None, :]
    obj = norm(p - p_gt).mean() * 100
    hoi = [norm((p - palm_s[:, k]) - (p_gt - palm_gt[:, k]))[c_gt[:, k]] for k in range(2) if c_gt[:, k].any()]
    hoi = np.concatenate(hoi).mean() * 100 if hoi else np.nan
    rooterr = norm(root + shift_hist - root_gt).mean() * 100
    palmerr = norm(palm_s - palm_gt).mean() * 100
    jit = norm(p[2:] - 2 * p[1:-1] + p[:-2]).mean() * 1000 if T >= 3 else np.nan
    return obj, hoi, rooterr, palmerr, jit, n_seg, n_frames / T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--prob_thr", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--min_static", type=int, default=5)
    ap.add_argument("--ramp", type=int, default=1)
    ap.add_argument("--anchor", choices=["palm", "hand"], default="palm")
    ap.add_argument("--oracle", action="store_true", help="GT contact + GT object rest (ceiling)")
    args = ap.parse_args()
    for spec in args.dumps:
        label, path = spec.split("=", 1)
        recs = torch.load(path, map_location="cpu", weights_only=False)["records"]
        print(f"== {label} ({len(recs)} seq): min_static={args.min_static} ramp={args.ramp} anchor={args.anchor}")
        configs = [("model (replayed)", 2.0, False)] + [(f"hand ZUPT thr={p}", p, False) for p in args.prob_thr] \
            + [("static veto only", 2.0, True)] + [(f"veto + hand ZUPT thr={p}", p, True) for p in args.prob_thr]
        for name, thr, veto in configs:
            res = np.array([replay(r, thr, args.min_static, args.ramp, veto, args.anchor, args.oracle) for r in recs])
            m = np.nanmean(res[:, :5], 0)
            print(f"   {name:26s} obj {m[0]:6.2f} | hoi {m[1]:6.2f} | root {m[2]:6.2f} | palm {m[3]:6.2f} | jitter {m[4]:6.2f} | segs/seq {res[:, 5].mean():.2f} | planted frames {res[:, 6].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
