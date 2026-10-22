#!/usr/bin/env python3
"""What the fusion gate sees in TRAINING: 60-frame windows whose recursion starts from the GT object position.

For every non-overlapping 60-frame window of each test sequence, compare on GT-contact hand-frames (HOI, cm):
  integ_gt_init : pure VC-velocity integration started from the GT position at the window start (no FK at all)
  own_fk        : the dumped FK candidate of the contacting hand
and the same two at inference, where the recursion starts from the model's own published state:
  published     : what the online rollout actually published
A gate trained with metric/MSE losses learns a gain ~ (recursion error) / (FK error) of the TRAINING regime.

    python scripts/diag/training_regime_candidate_probe.py name=dump.pt [...]
"""
import sys

import numpy as np
import torch

WIN, DT = 60, 1.0 / 30.0
WBINS = [(0, 1), (1, 2), (2, 4), (4, 10 ** 9)]  # window index bins; window 0 is the only GT-initialised one online


def _wbin(frame):
    w = frame // WIN
    return next(i for i, (lo, hi) in enumerate(WBINS) if lo <= w < hi)


def run(name, path):
    recs = torch.load(path, map_location="cpu")["records"]
    acc = {(k, b): [] for k in ("integ", "fk", "pub", "integ_end", "verr") for b in range(len(WBINS))}
    for r in recs:
        n = int(r["publish_len"])
        if not r.get("has_object", True) or n < WIN:
            continue
        gt = r["gt_obj_trans"][:n].double().numpy()
        g = r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n].double().numpy()
        palm = r["pred_palm_glb_pos"][:n].double().numpy()
        hs = r.get("stage1_pred_palm_glb_pos", r["pred_palm_glb_pos"])[:n].double().numpy()
        v = r["pred_obj_vel"][:n].double().numpy()
        pub = r["pred_obj_trans_published"][:n].double().numpy()
        fk = [palm[:, h] + (r[f"pred_{s}hand_obj_trans"][:n].double().numpy() - hs[:, h]) for h, s in ((0, "l"), (1, "r"))]
        integ = np.zeros_like(gt)
        for s in range(0, n, WIN):
            e = min(s + WIN, n)
            integ[s:e] = gt[s] + np.cumsum(v[s:e] * DT, axis=0) - v[s] * DT
            if e - s == WIN:
                acc[("integ_end", _wbin(s))].append(np.linalg.norm(integ[e - 1] - gt[e - 1]) * 100.0)
        vgt = np.zeros_like(gt); vgt[1:] = (gt[1:] - gt[:-1]) / DT
        wb = np.array([_wbin(t) for t in range(n)])
        verr = np.linalg.norm(v - vgt, axis=-1) * 100.0
        for b in range(len(WBINS)):
            if (wb == b).any():
                acc[("verr", b)].append(verr[wb == b])
        for h, key in enumerate(("gt_lhand_contact", "gt_rhand_contact")):
            m = r[key][:n].numpy().astype(bool)
            if not m.any():
                continue
            for b in range(len(WBINS)):
                mm = m & (wb == b)
                if not mm.any():
                    continue
                tgt = gt[mm] - g[mm, h]
                for k, p in (("integ", integ), ("fk", fk[h]), ("pub", pub)):
                    acc[(k, b)].append(np.linalg.norm((p[mm] - palm[mm, h]) - tgt, axis=-1) * 100.0)
    for b, (lo, hi) in enumerate(WBINS):
        c = lambda k: np.concatenate(acc[(k, b)]).mean()  # noqa: E731
        label = f"{lo * WIN}-{'' if hi > 1e8 else hi * WIN - 1}"
        print(f"| {name} | {label} | {c('integ'):.2f} | {c('fk'):.2f} | {c('pub'):.2f} | "
              f"{np.mean(acc[('integ_end', b)]):.2f} | {c('verr'):.2f} |")


if __name__ == "__main__":
    print("| model | frames | HOI: VC-velocity integration from GT window init | HOI: own FK | HOI: published | "
          "integration drift at window end, cm | velocity error cm/s |")
    print("|---|---|---|---|---|---|---|")
    for arg in sys.argv[1:]:
        run(*arg.split("=", 1))
