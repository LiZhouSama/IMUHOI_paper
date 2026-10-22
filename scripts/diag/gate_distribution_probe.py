#!/usr/bin/env python3
"""Distribution (not mean) of the fusion gate on GT-contact hand-frames.

The published object is a recursive filter: p_t = k_t * FK_t + (1 - k_t) * (p_{t-1} + step_t), so the FK weight is a
per-frame GAIN (time constant -1/ln(1-k) frames), not a trust share.  This reports, on GT-contact hand-frames:
  * quantiles of the total FK share k and the fraction of FK-starved frames (k < 0.05),
  * k and HOI binned by the predicted contact probability of that hand,
  * HOI as a function of how long the hand-frame has been FK-starved (run length of k < 0.05).

    python scripts/diag/gate_distribution_probe.py name=dump.pt [...]
"""
import sys

import numpy as np
import torch

PBINS = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]
RUNBINS = [(0, 1), (1, 15), (15, 60), (60, 10 ** 9)]


def run(name, path):
    recs = torch.load(path, map_location="cpu")["records"]
    k_all, own_all, p_all, hoi_all, starve_all = [], [], [], [], []
    for r in recs:
        n = int(r["publish_len"])
        if not r.get("has_object", True) or n == 0:
            continue
        w = r["gating_weights"][:n].double().numpy()
        k = w[:, 0] + w[:, 1]
        starved = k < 0.05
        runlen = np.zeros(n)
        for t in range(n):
            runlen[t] = (runlen[t - 1] + 1 if t > 0 else 1) if starved[t] else 0
        pc = r["pred_hand_contact_prob"][:n].double().numpy()
        gt = r["gt_obj_trans"][:n].double().numpy()
        g = r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n].double().numpy()
        palm = r["pred_palm_glb_pos"][:n].double().numpy()
        pub = r["pred_obj_trans_published"][:n].double().numpy()
        for h, key in enumerate(("gt_lhand_contact", "gt_rhand_contact")):
            m = r[key][:n].numpy().astype(bool)
            if not m.any():
                continue
            hoi = np.linalg.norm((pub - palm[:, h]) - (gt - g[:, h]), axis=-1) * 100.0
            k_all.append(k[m]); own_all.append(w[m, h]); p_all.append(pc[m, h]); hoi_all.append(hoi[m])
            starve_all.append(runlen[m])
    k, own, p, hoi, sr = map(np.concatenate, (k_all, own_all, p_all, hoi_all, starve_all))
    q = np.quantile(k, [0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"\n== {name}: {len(k)} GT-contact hand-frames, HOI {hoi.mean():.2f}")
    print(f"  FK share k: mean {k.mean():.3f} | q10/25/50/75/90 = {' / '.join(f'{x:.3f}' for x in q)} | "
          f"starved (k<0.05): {100 * (k < 0.05).mean():.1f}% | k>0.3: {100 * (k > 0.3).mean():.1f}%")
    print("  by predicted contact prob of this hand:  frames% | mean k | mean own-FK w | HOI")
    for lo, hi in PBINS:
        s = (p >= lo) & (p < hi)
        if s.any():
            print(f"    p in [{lo:.1f},{min(hi, 1.0):.1f}): {100 * s.mean():5.1f}% | {k[s].mean():.3f} | {own[s].mean():.3f} | {hoi[s].mean():.2f}")
    print("  by FK-starved run length (frames since k last >= 0.05):  frames% | HOI")
    for lo, hi in RUNBINS:
        s = (sr >= lo) & (sr < hi)
        if s.any():
            label = "not starved" if hi == 1 else f"{lo}-{'' if hi > 1e8 else hi - 1}"
            print(f"    {label:12s}: {100 * s.mean():5.1f}% | {hoi[s].mean():.2f}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run(*arg.split("=", 1))
