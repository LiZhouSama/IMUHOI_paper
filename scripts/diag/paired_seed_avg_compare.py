#!/usr/bin/env python3
"""Seed-averaged paired comparison of two methods from eval_batch_online --sequence_records_json files.

Each method contributes one records file per training seed; every sequence's
metric is first averaged over the seeds of that method, then the per-sequence
differences (B - A) are bootstrapped over sequences.

    python scripts/diag/paired_seed_avg_compare.py --a a_seed42.json a_seed43.json \
        --b b_seed42.json b_seed43.json [--dataset behave] [--boot 4000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def load(path, dataset):
    payload = json.load(open(path, encoding="utf-8"))["datasets"]
    records = payload[dataset] if dataset else next(iter(payload.values()))
    return {record["seq_name"]: record for record in records}


def metric(record, key):
    if key == "obj":
        value = record.get("obj_trans_err_fusion_published_cm")
    elif key == "hoi":
        value = record.get("hoi_err_fusion_cm")
    else:
        hand = {"f1_l": "left", "f1_r": "right"}[key]
        value = ((record.get("contact_prediction") or {}).get(hand) or {}).get("f1")
    return float(value) if value is not None and np.isfinite(value) else np.nan


def seed_average(paths, dataset, key):
    tables = [load(path, dataset) for path in paths]
    names = set.intersection(*(set(table) for table in tables))
    out = {}
    for name in names:
        values = np.array([metric(table[name], key) for table in tables])
        values = values[np.isfinite(values)]
        out[name] = values.mean() if len(values) else np.nan  # e.g. F1 of a hand that never touches
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", required=True)
    parser.add_argument("--b", nargs="+", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--boot", type=int, default=4000)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    print(f"A: {len(args.a)} seed(s), B: {len(args.b)} seed(s); B - A per sequence, seed-averaged")
    for key, label in (("obj", "obj (cm)"), ("hoi", "HOI (cm)"), ("f1_l", "F1 left"), ("f1_r", "F1 right")):
        a, b = seed_average(args.a, args.dataset, key), seed_average(args.b, args.dataset, key)
        names = sorted(set(a) & set(b))
        delta = np.array([b[n] - a[n] for n in names])
        delta = delta[np.isfinite(delta)]
        if not len(delta):
            continue
        boot = rng.choice(delta, size=(args.boot, len(delta)), replace=True).mean(1)
        lo, hi = np.percentile(boot, (2.5, 97.5))
        print(f"  {label:9s} mean {delta.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  B better on {np.mean(delta < 0 if key in ('obj', 'hoi') else delta > 0):.2f} (n={len(delta)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
