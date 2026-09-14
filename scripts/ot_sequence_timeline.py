#!/usr/bin/env python3
"""Print a frame-level timeline of one sequence across several dumps.

    python scripts/ot_sequence_timeline.py --seq 1008 v2=dump_a.pt v3=dump_b.pt [--step 10]

Columns per dump: object error, palm error (mean of both hands), root error,
predicted left/right contact probability, static gate weight, and for v3 the
norm of the latched anchor.  GT contact is shown as L/R flags so the segment
structure is visible at a glance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_ot_dumps import Seq, norm, segments_of  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+", help="label=path.pt")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    seqs = {}
    for item in args.dumps:
        label, path = item.split("=", 1)
        payload = torch.load(path, weights_only=False)
        for rec in payload["records"]:
            if str(rec["seq_name"]) == args.seq:
                seqs[label] = Seq(rec)
                break
        else:
            raise SystemExit(f"{args.seq} not in {path}")
    labels = list(seqs)
    ref = seqs[labels[0]]
    T = min(s.T for s in seqs.values())
    end = T if args.end is None else min(args.end, T)
    print(f"seq {ref.name} obj={ref.obj} T={T}  GT segments (any hand): {ref.seg_gt}")
    print(f"   left segs {segments_of(ref.c_gt[:, 0])}\n   right segs {segments_of(ref.c_gt[:, 1])}")
    head = f"{'t':>5} {'GT':>3} "
    for lb in labels:
        head += f"| {lb:^40} "
    print(head)
    sub = f"{'':>5} {'LR':>3} "
    for lb in labels:
        sub += f"| {'err':>5} {'palm':>5} {'root':>5} {'pL':>4} {'pR':>4} {'wS':>4} {'anch':>5} {'gate':>4} "
    print(sub)
    for t in range(args.start, end, args.step):
        line = f"{t:5d} {'L' if ref.c_gt[t,0] else '.'}{'R' if ref.c_gt[t,1] else '.'}  "
        for lb in labels:
            s = seqs[lb]
            palm = norm(s.d_hand[t]).mean() * 100
            root = norm(s.d_root[t]) * 100
            anchor = ""
            if "pred_lhand_segment_anchor" in s.rec:
                a = np.stack((s.rec["pred_lhand_segment_anchor"][t].numpy(), s.rec["pred_rhand_segment_anchor"][t].numpy()))
                anchor = f"{norm(a).mean()*100:5.1f}"
            else:
                anchor = f"{np.mean([s.rec['pred_lhand_lb'][t].item(), s.rec['pred_rhand_lb'][t].item()])*100:5.1f}"
            gate = "LRIS"[int(np.argmax(s.w[t]))]
            line += f"| {s.err[t]:5.1f} {palm:5.1f} {root:5.1f} {s.prob[t,0]:4.2f} {s.prob[t,1]:4.2f} {s.w[t,3]:4.2f} {anchor:>5} {gate:>4} "
        print(line)
    print("\nper-segment summary (err at onset -> err at end | palm drift over segment | root drift over segment)")
    for st, en in ref.seg_gt:
        line = f"  seg {st:4d}-{en:4d} (len {en-st+1:4d}) "
        for lb in labels:
            s = seqs[lb]
            if en >= s.T:
                continue
            palm_drift = norm(s.d_hand[en] - s.d_hand[st]).mean() * 100
            root_drift = norm(s.d_root[en] - s.d_root[st]) * 100
            line += f"| {lb}: {s.err[st]:5.1f}->{s.err[en]:5.1f} palm {palm_drift:4.1f} root {root_drift:4.1f} "
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
