#!/usr/bin/env python3
"""Offline check of the online VC velocity initialiser, from OT prediction dumps.

Online, window [t-59, t+10] initialises obj_vel_net with the model's own published velocity at t-60.  Candidates for
that initialiser, scored against the GT velocity (cm/s) by frame-index bin:
  self      : published VC velocity (current protocol)
  diff_k    : k-frame backward difference of the PUBLISHED object position, (p[t] - p[t-k]) / (k dt)
Also: is the growth of the velocity error with frame index a drift, or an activity confound?  -> error binned by
GT object speed x frame index.

    python scripts/diag/vc_init_velocity_probe.py name=dump.pt [...]
"""
import sys

import numpy as np
import torch

BINS = [(0, 60), (60, 120), (120, 240), (240, 10 ** 9)]
SPEED = [(0, 2), (2, 20), (20, 60), (60, 10 ** 9)]
DT = 1.0 / 30.0


def run(name, path):
    recs = torch.load(path, map_location="cpu")["records"]
    acc, conf = {}, {}
    for r in recs:
        n = int(r["publish_len"])
        if not r.get("has_object", True) or n < 12:
            continue
        gt = r["gt_obj_trans"][:n].double().numpy()
        pub = r["pred_obj_trans_published"][:n].double().numpy()
        v = r["pred_obj_vel"][:n].double().numpy()
        vgt = np.zeros_like(gt); vgt[1:] = (gt[1:] - gt[:-1]) / DT
        f = np.arange(n)
        cands = {"self": v}
        for k in (1, 2, 3, 5, 10):
            d = np.zeros_like(pub); d[k:] = (pub[k:] - pub[:-k]) / (k * DT); cands[f"diff_{k}"] = d
        gs = np.linalg.norm(vgt, axis=-1) * 100.0
        for b in BINS:
            sel = (f >= max(b[0], 10)) & (f < b[1])
            if not sel.any():
                continue
            for key, c in cands.items():
                acc.setdefault((key, b), []).append(np.linalg.norm(c[sel] - vgt[sel], axis=-1) * 100.0)
            for s in SPEED:
                ss = sel & (gs >= s[0]) & (gs < s[1])
                if ss.any():
                    conf.setdefault((s, b), []).append(np.linalg.norm(v[ss] - vgt[ss], axis=-1) * 100.0)
    print(f"\n== {name}: initialiser error |v - v_gt| cm/s by frame index")
    print("| candidate | " + " | ".join(f"{b[0]}-{'' if b[1] > 1e8 else b[1] - 1}" for b in BINS) + " |")
    for key in ("self", "diff_1", "diff_2", "diff_3", "diff_5", "diff_10"):
        print(f"| {key} | " + " | ".join(f"{np.concatenate(acc[(key, b)]).mean():.2f}" for b in BINS) + " |")
    print("self-velocity error by GT speed (rows, cm/s) x frame index (cols); share of frames in brackets")
    total = sum(len(np.concatenate(v)) for v in conf.values())
    for s in SPEED:
        cells = []
        for b in BINS:
            x = np.concatenate(conf[(s, b)]) if (s, b) in conf else np.zeros(0)
            cells.append(f"{x.mean():.2f} [{100 * len(x) / total:.1f}%]" if len(x) else "-")
        print(f"| speed {s[0]}-{'' if s[1] > 1e8 else s[1]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run(*arg.split("=", 1))
