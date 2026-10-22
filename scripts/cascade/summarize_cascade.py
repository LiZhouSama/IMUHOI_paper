#!/usr/bin/env python3
"""Tables for the cascade refine experiment (scripts/cascade/run_cascade_experiment.py).

Per dataset: aggregate metrics of every evaluated model, and per-sequence paired
differences (mean and 95% bootstrap CI over sequences) for the comparisons the
experiment was built for.  Writes <root>/summary.md and prints it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROWS = (
    ("baseline_best", "current pipeline (feedback + joint), best"),
    ("baseline_final", "current pipeline (feedback + joint), final"),
    ("b0_best", "B0: OT without feedback, no joint, best"),
    ("b0_final", "B0: OT without feedback, no joint, final"),
    ("cascade_e2_best", "cascade (e2), VC-refine best"),
    ("cascade_e2_final", "cascade (e2), VC-refine final"),
    ("cascade_e1_best", "cascade e1 (re-fuse published frame only), VC-refine best"),
    ("cascade_e1_final", "cascade e1 (re-fuse published frame only), VC-refine final"),
    ("cascade_e2_identity", "ablation: e2 re-fusion, VC-refine off"),
    ("cascade_e2_final_hprefined", "ablation: e2 final + refined HP history"),
    ("cascade_e1_final_hprefined", "ablation: e1 final + refined HP history"),
    ("cascade_e2p_final", "failed first e2: window start = value published 59 frames back, final"),
)
PAIRS = (
    ("b0_best", "baseline_best", "remove feedback + joint (best)"),
    ("b0_final", "baseline_final", "remove feedback + joint (final)"),
    ("cascade_e2_best", "b0_best", "refine stages, VC-refine best (same OT)"),
    ("cascade_e2_final", "b0_best", "refine stages, VC-refine final (same OT)"),
    ("cascade_e2_best", "baseline_best", "cascade vs current pipeline (best)"),
    ("cascade_e2_final", "baseline_final", "cascade vs current pipeline (final)"),
    ("cascade_e2_identity", "b0_best", "e2 whole-window re-fusion alone"),
    ("cascade_e2_best", "cascade_e2_identity", "VC-refine alone under e2 (best)"),
    ("cascade_e2_final", "cascade_e2_identity", "VC-refine alone under e2 (final)"),
    ("cascade_e1_best", "b0_best", "e1: VC-refine best (same OT)"),
    ("cascade_e1_final", "b0_best", "e1: VC-refine final (same OT)"),
    ("cascade_e1_best", "baseline_best", "e1 cascade vs current pipeline (best)"),
    ("cascade_e1_final", "baseline_final", "e1 cascade vs current pipeline (final)"),
    ("cascade_e1_final", "cascade_e2_final", "e1 instead of e2"),
    ("cascade_e2_final_hprefined", "cascade_e2_final", "refined HP history (e2)"),
    ("cascade_e1_final_hprefined", "cascade_e1_final", "refined HP history (e1)"),
    ("cascade_e2p_final", "cascade_e2_final", "published-anchor e2p instead of carried e2"),
)


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def aggregate(evals: Path, name: str):
    main = load(evals / f"{name}.json")
    if main is None:
        return None
    results = next(iter(main["results"].values()))
    extra = load(evals / f"{name}_extra.json")
    row = dict(results)
    if extra:
        data = extra["datasets"][0]
        row["obj_jitter_published"] = data.get("obj_jitter_published")
        row["obj_jitter_raw"] = data.get("obj_jitter_raw")
        contact = data.get("contact", {})
        for tag in ("published", "first_pass"):
            if tag in contact:
                row[f"{tag}_hand_f1"] = contact[tag]["eval_thr"]["f1"]
                row[f"{tag}_hand_precision"] = contact[tag]["eval_thr"]["precision"]
                row[f"{tag}_hand_recall"] = contact[tag]["eval_thr"]["recall"]
        static_fp = data.get("static_nohand_fp_rate_thr_0.5", {})
        row["static_fp_rate"] = static_fp.get("published")
    return row


def per_sequence(evals: Path, name: str):
    """{sequence: (obj error, HOI error, published object jitter)}; jitter is NaN without the extra JSON."""
    records = load(evals / f"{name}_records.json")
    if records is None:
        return None
    rows = next(iter(records["datasets"].values()))
    extra = load(evals / f"{name}_extra.json")
    jitter = {}
    if extra:
        for r in extra["datasets"][0].get("per_sequence", []):
            jitter[Path(r["seq_path"]).name] = r.get("obj_jitter_published", float("nan"))
    return {
        Path(r["seq_path"]).name: (
            r["obj_trans_err_fusion_published_cm"],
            r["hoi_err_fusion_cm"],
            jitter.get(Path(r["seq_path"]).name, float("nan")),
        )
        for r in rows
    }


def paired(a, b, metric, resamples=10000, seed=0):
    keys = sorted(set(a) & set(b))
    diff = np.array([a[k][metric] - b[k][metric] for k in keys], dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(resamples, diff.size), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)), diff.size


def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def dataset_section(root: Path, dataset: str):
    evals = root / dataset / "evals"
    lines = [f"## {dataset.upper()}", "",
             "| model | Obj (cm) | HOI (cm) | Obj jitter pub. | Obj jitter raw | MPJPE | Root | hand F1 | hand P | hand R | static-FP rate |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, label in ROWS:
        row = aggregate(evals, name)
        if row is None:
            continue
        f1 = row.get("published_hand_f1")
        if f1 is None:
            f1 = 0.5 * (row.get("contact_f1_lhand", float("nan")) + row.get("contact_f1_rhand", float("nan")))
        first = row.get("first_pass_hand_f1")
        f1_text = fmt(f1, 4) + (f" (VC {fmt(first, 4)})" if first is not None else "")
        lines.append(
            f"| {label} | {fmt(row.get('obj_trans_err_fusion'))} | {fmt(row.get('hoi_err_fusion'))} | "
            f"{fmt(row.get('obj_jitter_published'))} | {fmt(row.get('obj_jitter_raw'))} | {fmt(row.get('mpjpe'))} | "
            f"{fmt(row.get('root_trans_err'))} | {f1_text} | {fmt(row.get('published_hand_precision'), 4)} | "
            f"{fmt(row.get('published_hand_recall'), 4)} | {fmt(row.get('static_fp_rate'), 4)} |"
        )
    lines += ["", "Paired per-sequence differences (A - B, mean [95% bootstrap CI]):", "",
              "| comparison | A | B | Obj (cm) | HOI (cm) | Obj jitter pub. (per-seq) | n |",
              "|---|---|---|---|---|---|---|"]
    for a, b, label in PAIRS:
        seq_a, seq_b = per_sequence(evals, a), per_sequence(evals, b)
        if seq_a is None or seq_b is None:
            continue
        obj = paired(seq_a, seq_b, 0)
        hoi = paired(seq_a, seq_b, 1)
        jit = paired(seq_a, seq_b, 2)
        cell = lambda r: "-" if r is None else f"{r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}]"  # noqa: E731
        lines.append(f"| {label} | {a} | {b} | {cell(obj)} | {cell(hoi)} | {cell(jit)} | {obj[3] if obj else '-'} |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/IMUHOI++/RNN_cano/cascade_refine_20261010")
    args = parser.parse_args()
    root = (PROJECT_ROOT / args.root).resolve()
    lines = ["# Cascade refine experiment", ""]
    for dataset in ("behave", "omomo"):
        if (root / dataset / "evals").is_dir():
            lines += dataset_section(root, dataset) + [""]
    text = "\n".join(lines)
    (root / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
