#!/usr/bin/env python3
"""Paired per-sequence comparison of two eval_batch_online --sequence_records_json files.

Reports mean deltas (B - A) of object / HOI error and per-sequence contact F1 with
95% bootstrap CIs over sequences, and the fraction of sequences that improve.

    python scripts/diag/paired_eval_compare.py A_records.json B_records.json [--dataset behave]
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def load(path, dataset):
    payload = json.load(open(path, encoding="utf-8"))["datasets"]
    records = payload[dataset] if dataset else next(iter(payload.values()))
    return {record["seq_name"]: record for record in records}


def contact_f1(record, hand):
    value = (record.get("contact_prediction") or {}).get(hand, {}).get("f1")
    return float(value) if value is not None and np.isfinite(value) else np.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--boot", type=int, default=4000)
    args = parser.parse_args()
    a, b = load(args.a, args.dataset), load(args.b, args.dataset)
    names = sorted(set(a) & set(b))
    metrics = {
        "obj (cm)": lambda r: r["obj_trans_err_fusion_published_cm"],
        "HOI (cm)": lambda r: r["hoi_err_fusion_cm"],
        "F1 left": lambda r: contact_f1(r, "left"),
        "F1 right": lambda r: contact_f1(r, "right"),
    }
    rng = np.random.default_rng(0)
    print(f"{len(names)} paired sequences (B - A)")
    for label, getter in metrics.items():
        deltas = np.array([getter(b[n]) - getter(a[n]) for n in names], dtype=np.float64)
        deltas = deltas[np.isfinite(deltas)]
        boots = [rng.choice(deltas, len(deltas)).mean() for _ in range(args.boot)]
        better = np.mean(deltas < 0) if "cm" in label else np.mean(deltas > 0)
        print(
            f"  {label:9s} mean {deltas.mean():+.3f}  95% CI [{np.percentile(boots, 2.5):+.3f}, "
            f"{np.percentile(boots, 97.5):+.3f}]  B better on {better:.2f} (n={len(deltas)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
