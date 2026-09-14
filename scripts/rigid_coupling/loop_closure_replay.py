#!/usr/bin/env python3
"""Motion-onset loop closure replay (publisher level, no training).

Long sequences fail in the *gaps*: the human's root drifts while the object
rests, so at the next grasp the drifted hand and the (drift-free) resting object
give the OT an inconsistent hand->object geometry.  A resting object certified
static by its IMU is a world-fixed landmark, and the instant it starts moving is
an IMU-certified contact.  At that instant the discrepancy between the resting
object p_static and the FK proposal p_fk = h + R b of the hand(s) in contact is
attributed to root drift (bone-prior error is ~8 cm, root drift on long clips is
tens of cm), and the human is shifted by delta = p_static - p_fk (gain g).

Everything downstream re-uses the dumped bones and gating weights; only the
hands/root move, so the replay is conservative (the real OT would also see a
consistent init geometry and predict better bones).

Modes:  human   shift the human onto the landmark (loop closure)
        object  shift the object onto the hand (re-anchor, sanity comparison)
Usage: python scripts/rigid_coupling/loop_closure_replay.py label=dump.pt [...] --gain 1.0 0.5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.analyze_ot_dumps import hysteresis, imu_static_detector, rot6d  # noqa: E402

ON, OFF = 0.15, 0.075


def f64(t):
    return t.numpy().astype(np.float64)


def norm(x):
    return np.linalg.norm(x, axis=-1)


def replay(rec, mode, gain, min_static, contact_mode, ramp, veto=False, win=0):
    p_gt = f64(rec["gt_obj_trans"]); T = p_gt.shape[0]
    palm_gt = f64(rec["gt_palm_pos"]); palm = f64(rec["pred_palm_glb_pos"])
    h = f64(rec["ot_hand_positions"]); root = f64(rec["root_trans_pred"]); root_gt = f64(rec["gt_trans"])
    c_gt = np.stack((rec["gt_lhand_contact"].numpy(), rec["gt_rhand_contact"].numpy()), 1).astype(bool)
    prob = f64(rec["pred_hand_contact_prob"])[:, :2]
    w = f64(rec["gating_weights"])
    fk = np.stack((f64(rec["pred_lhand_obj_trans"]), f64(rec["pred_rhand_obj_trans"])), 1)
    Rb = fk - h
    d_imu = f64(rec["pred_imu_obj_trans"]) - f64(rec["pred_static_obj_trans"])
    imu = f64(rec["gt_obj_imu"]); R = rot6d(rec["gt_obj_imu"][:, 3:]) if rec["gt_obj_imu"].shape[-1] >= 9 else None
    static = imu_static_detector(R, imu[:, :3])
    c_hys = hysteresis(prob, ON, OFF)
    shift = np.zeros(3); shift_hist = np.zeros((T, 3)); target = np.zeros(3); n_closures = 0; n_onsets = 0
    w = w.copy()
    if veto:
        w[static] = np.array([0.0, 0.0, 0.0, 1.0])
    p = np.zeros((T, 3)); prev = f64(rec["pred_static_obj_trans"])[0].copy()
    static_run = 0
    for t in range(T):
        # certified motion onset: object was IMU-static for >= min_static frames and now moves
        onset = static_run >= min_static and not static[t] and t > 0
        if onset:
            n_onsets += 1
            lo, hi = max(0, t - win), min(T, t + win + 1)
            if contact_mode == "prob":
                active = (prob[lo:hi] > 0.5).any(0)
            else:
                active = c_hys[lo:hi].any(0)
            if active.any():
                p_fk = (fk[t][active] + shift).mean(0)
                delta = prev - p_fk
                if mode == "human":
                    target = target + gain * delta
                else:
                    prev = prev - gain * delta
                n_closures += 1
        # ramp the shift towards its target to avoid a step in the published tracks
        if ramp > 1:
            shift = shift + (target - shift) / ramp
        else:
            shift = target.copy()
        shift_hist[t] = shift
        hs = h[t] + shift
        p[t] = w[t, 0] * (hs[0] + Rb[t, 0]) + w[t, 1] * (hs[1] + Rb[t, 1]) + w[t, 2] * (prev + d_imu[t]) + w[t, 3] * prev
        prev = p[t]
        static_run = static_run + 1 if static[t] else 0
    palm_s = palm + shift_hist[:, None, :]
    obj = norm(p - p_gt).mean() * 100
    hoi = [norm((p - palm_s[:, k]) - (p_gt - palm_gt[:, k]))[c_gt[:, k]] for k in range(2) if c_gt[:, k].any()]
    hoi = np.concatenate(hoi).mean() * 100 if hoi else np.nan
    rooterr = norm(root + shift_hist - root_gt).mean() * 100
    palmerr = norm(palm_s - palm_gt).mean() * 100
    jit = norm(p[2:] - 2 * p[1:-1] + p[:-2]).mean() * 1000 if T >= 3 else np.nan
    return obj, hoi, rooterr, palmerr, jit, n_closures, n_onsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--gain", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--min_static", type=int, default=15, help="frames of IMU-static rest required before an onset counts")
    ap.add_argument("--contact", choices=["prob", "hysteresis"], default="prob")
    ap.add_argument("--ramp", type=int, default=1, help="frames over which the shift is applied (1 = step)")
    ap.add_argument("--veto", action="store_true", help="force the static branch whenever the object IMU says static")
    ap.add_argument("--win", type=int, default=0, help="contact may be detected within +-win frames of the onset")
    args = ap.parse_args()
    for spec in args.dumps:
        label, path = spec.split("=", 1)
        recs = torch.load(path, map_location="cpu", weights_only=False)["records"]
        print(f"== {label} ({len(recs)} seq): min_static={args.min_static} contact={args.contact} ramp={args.ramp} veto={args.veto} win={args.win}")
        configs = [("model (replayed)", "human", 0.0)] + [(f"loop closure human g={g}", "human", g) for g in args.gain] + [(f"re-anchor object g={g}", "object", g) for g in args.gain]
        for name, mode, g in configs:
            res = np.array([replay(r, mode, g, args.min_static, args.contact, args.ramp, args.veto, args.win) for r in recs])
            m = np.nanmean(res[:, :5], 0)
            print(f"   {name:28s} obj {m[0]:6.2f} | hoi {m[1]:6.2f} | root {m[2]:6.2f} | palm {m[3]:6.2f} | jitter {m[4]:6.2f} | closures/seq {res[:, 5].mean():.2f} of {res[:, 6].mean():.2f} onsets")


if __name__ == "__main__":
    main()
