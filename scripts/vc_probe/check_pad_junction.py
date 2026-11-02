#!/usr/bin/env python3
"""How much does a synthetic static prefix change the VC outputs on the real frames that follow it?

Compares noise-free online runs of the same sequences without (``omomo_junction_orig``) and with a PAD-frame static
prefix (``omomo_junction_pad``): published hand-contact and anticipation probabilities at the same real frame,
binned by real-frame index (= frames since the junction).

    python scripts/vc_probe/check_pad_junction.py --pad 90
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

PILOT = Path(__file__).resolve().parents[2] / "outputs/IMUHOI++/RNN_cano/vc_probe_pilot_20261027"
BINS = ((60, 90), (90, 120), (120, 150), (150, 10 ** 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, default=90)
    ap.add_argument("--out", type=Path, default=PILOT / "analysis_omomo_pad90_junction.md")
    args = ap.parse_args()
    orig_dir, pad_dir = PILOT / "omomo_junction_orig" / "probs", PILOT / "omomo_junction_pad" / "probs"
    files = sorted(p.name for p in orig_dir.glob("*.pt"))
    diffs = {b: {"p": [], "ant": []} for b in BINS}
    for f in files:
        o = torch.load(orig_dir / f, weights_only=False)
        q = torch.load(pad_dir / f, weights_only=False)
        po, pq = o["pred_hand_contact_prob"][:, :2].numpy(), q["pred_hand_contact_prob"][args.pad:, :2].numpy()
        ao, aq = o["pred_contact_anticipation_prob"].numpy(), q["pred_contact_anticipation_prob"][args.pad:].numpy()
        n = min(len(po), len(pq))
        for lo, hi in BINS:
            sl = slice(lo, min(hi, n))
            if sl.start < sl.stop:
                diffs[(lo, hi)]["p"].append(np.abs(po[sl] - pq[sl]).ravel())
                diffs[(lo, hi)]["ant"].append(np.abs(ao[sl] - aq[sl]).ravel())
    lines = [f"# Static-prefix junction check ({len(files)} OMOMO sequences whose first contact starts at real frame >= 90)",
             "", "Noise-free online runs with and without a " f"{args.pad}-frame static prefix; |difference| of the published "
             "probabilities at the same real frame (real frames < 60 are the warmup block of the original run and skipped).", "",
             "| real frames | hand contact: mean / p95 / max | anticipation: mean / p95 / max |", "|---|---|---|"]
    for (lo, hi), d in diffs.items():
        if not d["p"]:
            continue
        p, a = np.concatenate(d["p"]), np.concatenate(d["ant"])
        span = f"{lo}+" if hi > 10 ** 5 else f"{lo}-{hi - 1}"
        lines.append(f"| {span} | {p.mean():.4f} / {np.percentile(p, 95):.4f} / {p.max():.3f} | "
                     f"{a.mean():.4f} / {np.percentile(a, 95):.4f} / {a.max():.3f} |")
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
