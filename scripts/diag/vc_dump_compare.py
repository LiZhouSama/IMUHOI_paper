#!/usr/bin/env python3
"""Compare VelocityContact behaviour across online OT dumps (scripts/dump_ot_predictions.py).

Per dump: online object-velocity error by frame band and regime, hand-contact
F1 / FP / FN by frame band, and published object error per GT contact regime.

    python scripts/diag/vc_dump_compare.py name=dump.pt [name=dump.pt ...]
"""
from __future__ import annotations

import sys

import numpy as np
import torch

BANDS = ((0, 60), (60, 240), (240, 600), (600, 10**9))


def band_name(band):
    return f"{band[0]}-{band[1] if band[1] < 10**9 else 'end'}"


def analyse(path):
    records = torch.load(path, map_location="cpu")["records"]
    vel = {band: [[], []] for band in BANDS}
    regime_vel = {key: [[], []] for key in ("static", "hand", "nohand-moving")}
    contact = {band: np.zeros(3) for band in BANDS}
    regime_obj = {key: [] for key in ("static", "hand", "nohand-moving")}
    for record in records:
        gt = record["gt_obj_trans"].numpy().astype(np.float64)
        v_gt = np.concatenate((np.zeros((1, 3)), np.diff(gt, axis=0) * 30.0), 0)
        v_pred = record["pred_obj_vel"].numpy()
        err_obj = np.linalg.norm(record["pred_obj_trans_published"].numpy() - gt, axis=-1) * 100
        hands = np.stack((record["gt_lhand_contact"].numpy(), record["gt_rhand_contact"].numpy()), -1).astype(bool)
        moving = record["gt_obj_contact"].numpy().astype(bool)
        pred = record["pred_hand_contact_prob"].numpy()[:, :2] > 0.5
        frames = np.arange(len(gt))
        regimes = {"static": ~moving, "hand": moving & hands.any(-1), "nohand-moving": moving & ~hands.any(-1)}
        err_vel = np.linalg.norm(v_pred - v_gt, axis=-1)
        for band in BANDS:
            mask = (frames >= band[0]) & (frames < band[1])
            vel[band][0].append(err_vel[mask])
            vel[band][1].append(np.linalg.norm(v_gt[mask], axis=-1))
            g, p = hands[mask], pred[mask]
            contact[band] += ((g & p).sum(), (~g & p).sum(), (g & ~p).sum())
        for key, mask in regimes.items():
            regime_vel[key][0].append(err_vel[mask])
            regime_vel[key][1].append(np.linalg.norm(v_gt[mask], axis=-1))
            regime_obj[key].append(err_obj[mask])
    out = {}
    for band in BANDS:
        e, m = np.concatenate(vel[band][0]), np.concatenate(vel[band][1])
        tp, fp, fn = contact[band]
        n = len(e)
        out[f"vel_err {band_name(band)}"] = e.mean() if n else float("nan")
        out[f"hand F1 {band_name(band)}"] = 2 * tp / max(2 * tp + fp + fn, 1)
    for key in regime_vel:
        out[f"vel_err {key}"] = np.concatenate(regime_vel[key][0]).mean()
        out[f"obj_err {key}"] = np.concatenate(regime_obj[key]).mean()
    out["obj_err all"] = np.concatenate([np.linalg.norm(r["pred_obj_trans_published"].numpy() - r["gt_obj_trans"].numpy(), axis=-1) for r in records]).mean() * 100
    return out


def main():
    runs = dict(arg.split("=", 1) for arg in sys.argv[1:])
    results = {name: analyse(path) for name, path in runs.items()}
    keys = list(next(iter(results.values())))
    print(f"{'metric':28s} " + " ".join(f"{name:>10s}" for name in results))
    for key in keys:
        print(f"{key:28s} " + " ".join(f"{results[name][key]:10.3f}" for name in results))


if __name__ == "__main__":
    main()
