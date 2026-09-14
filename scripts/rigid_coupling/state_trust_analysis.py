#!/usr/bin/env python3
"""Diagnose a state-trust ObjectTrans dump: is the confidence calibrated, and did the
long-horizon bone-error growth flatten?

Usage: python scripts/rigid_coupling/state_trust_analysis.py label=dump.pt [ref=dump.pt ...]
Dumps of non-trust models are accepted for the time-growth reference (no confidence rows).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

BINS_T = [(0, 150), (150, 300), (300, 600), (600, 1200), (1200, 10 ** 9)]
BINS_ONSET = [(0, 15), (15, 60), (60, 120), (120, 10 ** 9)]


def f64(t):
    return t.numpy().astype(np.float64)


def norm(x):
    return np.linalg.norm(x, axis=-1)


def onset_age(mask):
    age = np.full(len(mask), -1)
    run = -1
    for t, m in enumerate(mask):
        run = run + 1 if (m and run >= 0) else (0 if m else -1)
        age[t] = run
    return age


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    for spec in args.dumps:
        label, path = spec.split("=", 1)
        recs = torch.load(path, map_location="cpu", weights_only=False)["records"]
        bone, t_abs, age, conf, state_err, hoi, gateFK, prior_err = [], [], [], [], [], [], [], []
        for r in recs:
            p_gt = f64(r["gt_obj_trans"]); palm_gt = f64(r["gt_palm_pos"]); h = f64(r["ot_hand_positions"])
            pub = f64(r["pred_obj_trans_published"]); palm = f64(r["pred_palm_glb_pos"])
            fk = np.stack((f64(r["pred_lhand_obj_trans"]), f64(r["pred_rhand_obj_trans"])), 1)
            c = np.stack((r["gt_lhand_contact"].numpy(), r["gt_rhand_contact"].numpy()), 1).astype(bool)
            w = f64(r["gating_weights"])
            has_conf = "state_confidence" in r
            cf = f64(r["state_confidence"]) if has_conf else None
            prior = np.stack((f64(r["prior_lhand_obj_trans"]), f64(r["prior_rhand_obj_trans"])), 1) if "prior_lhand_obj_trans" in r else None
            T = len(p_gt)
            # state used at frame t = published position at t-1 (init at t=0 ~ GT first frame)
            st_err = np.concatenate(([0.0], norm(pub[:-1] - p_gt[:-1])))
            for k in range(2):
                m = c[:, k]
                if not m.any():
                    continue
                idx = np.nonzero(m)[0]
                bone.append(norm((fk[idx, k] - h[idx, k]) - (p_gt[idx] - palm_gt[idx, k])))
                hoi.append(norm((pub[idx] - palm[idx, k]) - (p_gt[idx] - palm_gt[idx, k])))
                t_abs.append(idx); age.append(onset_age(m)[idx]); gateFK.append(w[idx, 0] + w[idx, 1])
                state_err.append(st_err[idx])
                conf.append(cf[idx] if has_conf else np.full(len(idx), np.nan))
                prior_err.append(norm((prior[idx, k] - h[idx, k]) - (p_gt[idx] - palm_gt[idx, k])) if prior is not None else np.full(len(idx), np.nan))
        bone, t_abs, age, conf, state_err, hoi, gateFK, prior_err = map(np.concatenate, (bone, t_abs, age, conf, state_err, hoi, gateFK, prior_err))
        print(f"== {label}: {len(recs)} seq, {len(bone)} GT-contact frames | bone err {bone.mean()*100:.2f} | HOI(frame) {hoi.mean()*100:.2f} | prior-head err {np.nanmean(prior_err)*100:.2f} | mean gate FK {gateFK.mean():.2f}")
        print("   bone err by frame index : " + " | ".join(f"[{lo},{min(hi,9999)}) {bone[(t_abs>=lo)&(t_abs<hi)].mean()*100:.2f}" for lo, hi in BINS_T if ((t_abs>=lo)&(t_abs<hi)).any()))
        print("   HOI by frames since onset: " + " | ".join(f"[{lo},{min(hi,9999)}) {hoi[(age>=lo)&(age<hi)].mean()*100:.2f}" for lo, hi in BINS_ONSET if ((age>=lo)&(age<hi)).any()))
        if np.isfinite(conf).any():
            good = state_err < 0.10
            print(f"   confidence: mean {conf.mean():.3f} | when state err<10cm {conf[good].mean():.3f} (n={good.sum()}) | when >=10cm {conf[~good].mean():.3f} (n={(~good).sum()}) | corr(conf, -state err) {np.corrcoef(conf, -state_err)[0,1]:.2f}")
            edges = [0, 0.05, 0.10, 0.20, 0.40, 9]
            print("   conf by state err bin  : " + " | ".join(f"[{lo*100:.0f},{hi*100:.0f}cm) {conf[(state_err>=lo)&(state_err<hi)].mean():.2f} (n={((state_err>=lo)&(state_err<hi)).sum()})" for lo, hi in zip(edges[:-1], edges[1:]) if ((state_err>=lo)&(state_err<hi)).any()))
            q = np.quantile(conf, [0.25, 0.5, 0.75]); ce = [-1, q[0], q[1], q[2], 2]
            print("   bone err by conf quartile: " + " | ".join(f"<= {hi:.2f}: {bone[(conf>lo)&(conf<=hi)].mean()*100:.2f}" if hi < 2 else f"> {lo:.2f}: {bone[(conf>lo)].mean()*100:.2f}" for lo, hi in zip(ce[:-1], ce[1:])))


if __name__ == "__main__":
    main()
