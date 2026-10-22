#!/usr/bin/env python3
"""Where does the predicted palm jitter come from, and what would a lookahead-bounded smoother buy?

Second difference (mm/frame^2, the obj_jitter unit) of: GT palm, predicted palm, predicted root, root-relative palm.
Then offline smoothers applied to the predicted palm (all usable online with the 10-frame lookahead):
  ema_a        causal exponential smoothing
  ma_c(2h+1)   centred moving average with half-width h <= lookahead
  sg(w,p)      centred Savitzky-Golay
scored by palm jitter and palm position error (cm) on all frames / GT-contact frames.

    python scripts/diag/hand_jitter_probe.py name=dump.pt [...]
"""
import sys

import numpy as np
import torch
from scipy.signal import savgol_filter


def sd(x):
    return np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1) * 1000.0


def ema(x, a):
    out = x.copy()
    for t in range(1, len(x)):
        out[t] = a * x[t] + (1 - a) * out[t - 1]
    return out


def ma(x, h):
    k = np.ones(2 * h + 1) / (2 * h + 1)
    pad = np.concatenate([np.repeat(x[:1], h, 0), x, np.repeat(x[-1:], h, 0)])
    return np.stack([np.convolve(pad[:, d], k, mode="valid") for d in range(x.shape[1])], -1)


def run(name, path):
    recs = torch.load(path, map_location="cpu")["records"]
    acc = {}
    filters = {"raw": lambda x: x, "ema_0.5": lambda x: ema(x, 0.5), "ema_0.3": lambda x: ema(x, 0.3),
               "ma_c5": lambda x: ma(x, 2), "ma_c9": lambda x: ma(x, 4), "ma_c15": lambda x: ma(x, 7),
               "sg(9,2)": lambda x: savgol_filter(x, 9, 2, axis=0), "sg(15,2)": lambda x: savgol_filter(x, 15, 2, axis=0),
               "sg(21,3)": lambda x: savgol_filter(x, 21, 3, axis=0)}
    for r in recs:
        n = int(r["publish_len"])
        if n < 30:
            continue
        palm = r["pred_palm_glb_pos"][:n].double().numpy(); gpalm = r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n].double().numpy()
        root = r["root_trans_pred"][:n].double().numpy(); groot = r["gt_trans"][:n].double().numpy()
        acc.setdefault("root", []).append(sd(root)); acc.setdefault("gt_root", []).append(sd(groot))
        for h, key in enumerate(("gt_lhand_contact", "gt_rhand_contact")):
            m = r[key][:n].numpy().astype(bool)
            acc.setdefault("gt_palm", []).append(sd(gpalm[:, h])); acc.setdefault("rel", []).append(sd(palm[:, h] - root))
            acc.setdefault("gt_rel", []).append(sd(gpalm[:, h] - groot))
            for fname, fn in filters.items():
                y = fn(palm[:, h])
                acc.setdefault((fname, "jit"), []).append(sd(y))
                acc.setdefault((fname, "err"), []).append(np.linalg.norm(y - gpalm[:, h], axis=-1) * 100.0)
                if m.any():
                    acc.setdefault((fname, "err_c"), []).append(np.linalg.norm(y[m] - gpalm[m, h], axis=-1) * 100.0)
                    # hand->object vector error when the bone is perfect: isolates the palm's share of HOI noise
    c = lambda k: np.concatenate(acc[k]).mean()  # noqa: E731
    print(f"\n== {name}\n  jitter: GT palm {c('gt_palm'):.1f} | GT root {c('gt_root'):.1f} | GT root-relative palm {c('gt_rel'):.1f} | "
          f"pred root {c('root'):.1f} | pred root-relative palm {c('rel'):.1f}")
    print("  | filter | palm jitter | palm err cm (all) | palm err cm (contact) |")
    for fname in filters:
        print(f"  | {fname} | {c((fname, 'jit')):.1f} | {c((fname, 'err')):.2f} | {c((fname, 'err_c')):.2f} |")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        run(*arg.split("=", 1))
