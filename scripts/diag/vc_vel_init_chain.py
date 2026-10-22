#!/usr/bin/env python3
"""Velocity-initializer error vs VC output error in online dumps (diff-3 or own-output protocol).

The window publishing frame t starts at s = t-59 and reads its object-velocity
initializer at frame t-60: own-output protocol -> pred_obj_vel[t-60], diff-3 protocol ->
(pub[t-60] - pub[t-63]) * 10.  Reports, by band of the published frame t:
  |own init err|, |diff3 init err|, |VC output err at t|, and the least-squares slope of
  the output error vector on the init error vector of the protocol actually used
  (1 = the init error passes straight through the window, 0 = fully corrected).

    python scripts/diag/vc_vel_init_chain.py name=diff=dump.pt name=own=dump.pt ...
"""
import sys

import numpy as np
import torch

BANDS = ((60, 240), (240, 600), (600, 10**9))


def analyse(path, protocol):
    records = torch.load(path, map_location="cpu")["records"]
    acc = {b: {"own": [], "diff": [], "out": [], "ei": [], "eo": [], "vgt": []} for b in BANDS}
    for r in records:
        gt = r["gt_obj_trans"].numpy().astype(np.float64)
        pub = r["pred_obj_trans"].numpy().astype(np.float64)
        v_gt = np.concatenate((np.zeros((1, 3)), np.diff(gt, axis=0) * 30.0), 0)
        v_out = r["pred_obj_vel"].numpy().astype(np.float64)
        T = len(gt)
        if T <= 64:
            continue
        t = np.arange(63, T)
        i = t - 60
        own = v_out[i]
        diff = (pub[i] - pub[i - 3]) * 10.0
        init = diff if protocol == "diff" else own
        e_init = init - v_gt[i]
        e_out = v_out[t] - v_gt[t]
        for b in BANDS:
            m = (t >= b[0]) & (t < b[1])
            if not m.any():
                continue
            a = acc[b]
            a["own"].append(np.linalg.norm(own[m] - v_gt[i][m], axis=-1))
            a["diff"].append(np.linalg.norm(diff[m] - v_gt[i][m], axis=-1))
            a["out"].append(np.linalg.norm(e_out[m], axis=-1))
            a["ei"].append(e_init[m])
            a["eo"].append(e_out[m])
            a["vgt"].append(np.linalg.norm(v_gt[t][m], axis=-1))
    rows = {}
    for b, a in acc.items():
        if not a["out"]:
            continue
        ei, eo = np.concatenate(a["ei"]), np.concatenate(a["eo"])
        slope = float((ei * eo).sum() / max((ei * ei).sum(), 1e-12))
        corr = float((ei * eo).sum() / np.sqrt((ei * ei).sum() * (eo * eo).sum()))
        rows[b] = (np.concatenate(a["own"]).mean(), np.concatenate(a["diff"]).mean(),
                   np.concatenate(a["out"]).mean(), slope, corr, np.concatenate(a["vgt"]).mean(), len(eo))
    return rows


def main():
    print(f"{'run':24s} {'band':>9s} {'|v|':>6s} {'own_init':>8s} {'diff3_init':>10s} {'VC_out':>7s} {'slope':>6s} {'corr':>6s} {'n':>8s}")
    for arg in sys.argv[1:]:
        name, protocol, path = arg.split("=", 2)
        for b, (own, diff, out, slope, corr, vgt, n) in analyse(path, protocol).items():
            band = f"{b[0]}-{b[1] if b[1] < 10**9 else 'end'}"
            print(f"{name:24s} {band:>9s} {vgt:6.3f} {own:8.3f} {diff:10.3f} {out:7.3f} {slope:6.2f} {corr:6.2f} {n:8d}")


if __name__ == "__main__":
    main()
