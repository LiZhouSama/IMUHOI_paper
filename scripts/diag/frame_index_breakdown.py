#!/usr/bin/env python3
"""Published object / HOI error binned by absolute frame index within the sequence.

Reads OT prediction dumps (scripts/dump_ot_predictions.py, online mode) and reports,
per model, the mean published object error and the metric-aligned HOI error
(GT-contact hand-frames) for frame-index bins.  The first bin (0-59) is the
cold-start window of the online protocol (GT init, no published prefix); later
bins run from the model's own published prefix.

    python scripts/diag/frame_index_breakdown.py name=path.pt [name=path.pt ...]
"""
import sys

import numpy as np
import torch

BINS = [(0, 60), (60, 120), (120, 240), (240, 10 ** 9)]


def load(path):
    return torch.load(path, map_location="cpu")["records"]


def per_frame(records):
    """Return arrays: frame index, object error (cm), HOI error per hand-frame (cm) with frame index."""
    idx, obj, hoi_idx, hoi = [], [], [], []
    for r in records:
        if not r.get("has_object", True):
            continue
        n = int(r["publish_len"])
        gt = np.asarray(r["gt_obj_trans"][:n], dtype=np.float64)
        pr = np.asarray(r["pred_obj_trans_published"][:n], dtype=np.float64)
        palm = np.asarray(r["pred_palm_glb_pos"][:n], dtype=np.float64)
        gt_palm = np.asarray(r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n], dtype=np.float64)
        frames = np.arange(n)
        idx.append(frames)
        obj.append(np.linalg.norm(pr - gt, axis=-1) * 100.0)
        for h, key in enumerate(("gt_lhand_contact", "gt_rhand_contact")):
            m = np.asarray(r[key][:n]).astype(bool)
            if not m.any():
                continue
            rel_pred = pr[m] - palm[m, h]
            rel_gt = gt[m] - gt_palm[m, h]
            hoi.append(np.linalg.norm(rel_pred - rel_gt, axis=-1) * 100.0)
            hoi_idx.append(frames[m])
    return np.concatenate(idx), np.concatenate(obj), np.concatenate(hoi_idx), np.concatenate(hoi)


def main():
    models = [arg.split("=", 1) for arg in sys.argv[1:]]
    stats = {}
    for name, path in models:
        idx, obj, hidx, hoi = per_frame(load(path))
        row = {"all": (obj.mean(), hoi.mean())}
        for lo, hi in BINS:
            mo = (idx >= lo) & (idx < hi)
            mh = (hidx >= lo) & (hidx < hi)
            row[f"{lo}-{hi - 1 if hi < 10 ** 8 else ''}"] = (obj[mo].mean() if mo.any() else np.nan, hoi[mh].mean() if mh.any() else np.nan, int(mo.sum()), int(mh.sum()))
        stats[name] = row
    cols = list(next(iter(stats.values())).keys())
    print("frame-index bins: object error (cm) / HOI on GT-contact hand-frames (cm)")
    print("| model | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, row in stats.items():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v[0]:.2f} / {v[1]:.2f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    first = next(iter(stats.values()))
    print("frames per bin (object / hand-frames):", {c: (first[c][2], first[c][3]) for c in cols if c != "all"})


if __name__ == "__main__":
    main()
