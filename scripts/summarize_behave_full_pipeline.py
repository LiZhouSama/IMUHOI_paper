#!/usr/bin/env python3
"""Summarize the BEHAVE full-pipeline state-trust comparison.

Reads the manifest written by ``run_behave_full_pipeline_trust.py`` and prints a
markdown table of the BEHAVE full-sequence metrics for every variant, at both the
best and the final checkpoint (long-horizon numbers move with the checkpoint, so
the final one is the comparable column).  With ``--with_omomo`` it appends the
earlier OMOMO-trained runs of the same variants for side-by-side reading: those
measure the same recipes across a domain shift, these measure them in domain.

    python scripts/summarize_behave_full_pipeline.py \
        --root outputs/IMUHOI++/RNN_cano/behave_full_trust_20260911 --with_omomo
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OMOMO_TRAINED = {
    # variant -> (OMOMO eval json, BEHAVE best eval json, BEHAVE final eval json)
    "cur_hoi_p800": (
        "outputs/IMUHOI++/RNN_cano/omomo_cano_ot_trust_20260910_c/cur_hoi_p800/eval_batch_online.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/cur_hoi_p800_c_behave_eval.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/p800_c_final_behave_eval.json",
    ),
    "cur_hoi_trust_noaug": (
        "outputs/IMUHOI++/RNN_cano/omomo_cano_ot_trust_20260910_e/cur_hoi_trust_noaug/eval_batch_online.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/cur_hoi_trust_noaug_behave_eval.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/trust_noaug_final_behave_eval.json",
    ),
    "cur_hoi_trust_mild": (
        "outputs/IMUHOI++/RNN_cano/omomo_cano_ot_trust_20260910_h/cur_hoi_trust_mild/eval_batch_online.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/cur_hoi_trust_mild_behave_eval.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/trust_mild_final_behave_eval.json",
    ),
    "cur_hoi_trust": (
        "outputs/IMUHOI++/RNN_cano/omomo_cano_ot_trust_20260910/cur_hoi_trust/eval_batch_online.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/cur_hoi_trust_behave_eval.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/trust_a_final_behave_eval.json",
    ),
    "cur_hoi_trust_perturb": (
        "outputs/IMUHOI++/RNN_cano/omomo_cano_ot_trust_20260910_d/cur_hoi_trust_perturb/eval_batch_online.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/cur_hoi_trust_perturb_behave_eval.json",
        "outputs/IMUHOI++/RNN_cano/behave_longseq_20260909/trust_perturb_final_behave_eval.json",
    ),
}
COLUMNS = (
    ("obj_trans_err_fusion", "obj"),
    ("hoi_err_fusion", "HOI"),
    ("obj_jitter", "obj jit"),
    ("mpjpe", "MPJPE"),
    ("root_trans_err", "root"),
    ("contact_f1_lhand", "F1 L"),
    ("contact_f1_rhand", "F1 R"),
)


def load(path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    if isinstance(results, dict) and len(results) == 1:
        results = next(iter(results.values()))
    return {k: v for k, v in results.items() if isinstance(v, (int, float))}


def row(name: str, tag: str, metrics: dict) -> str:
    cells = ["" if metrics.get(key) is None else f"{metrics[key]:.2f}" for key, _ in COLUMNS]
    return f"| {name} | {tag} | " + " | ".join(cells) + " |"


def table(title: str, rows) -> None:
    print(f"\n### {title}")
    print("| variant | ckpt | " + " | ".join(label for _, label in COLUMNS) + " |")
    print("|---|---|" + "---|" * len(COLUMNS))
    for line in rows:
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--with_omomo", action="store_true", help="append the OMOMO-trained runs of the same recipes")
    args = ap.parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))

    schedule = manifest.get("schedule", {})
    print(f"# BEHAVE full-pipeline state-trust comparison ({root.name})")
    print(f"\nTrained end to end on BEHAVE (HP -> VC-v4 -> ObjectTrans+joint); reported on the BEHAVE "
          f"full-length test sequences. Schedule scale {schedule.get('schedule_scale')} "
          f"({schedule.get('reference_steps_per_epoch')} -> {schedule.get('train_steps_per_epoch')} steps/epoch): "
          f"epochs {schedule.get('epochs')}, milestones {schedule.get('milestones')}, "
          f"early stopping {schedule.get('early_stopping')}.")

    rows = []
    for name, entry in manifest.get("pipelines", {}).items():
        for tag, key in (("best", "metrics_best"), ("final", "metrics_final")):
            metrics = entry.get(key) or {}
            if metrics:
                rows.append(row(name, tag, metrics))
        if entry.get("status") == "failed":
            rows.append(f"| {name} | FAILED | " + " | ".join([""] * len(COLUMNS)) + " |")
    table("BEHAVE-trained (in domain)", rows)

    if args.with_omomo:
        rows = []
        for name, (omomo_json, behave_best, behave_final) in OMOMO_TRAINED.items():
            metrics = load(behave_best)
            if metrics:
                rows.append(row(name, "best", metrics))
            metrics = load(behave_final)
            if metrics:
                rows.append(row(name, "final", metrics))
        table("OMOMO-trained, evaluated on BEHAVE (domain shift, for reference)", rows)

    stages = manifest.get("stages", {})
    print("\nShared stages: " + ", ".join(
        f"{name}={info.get('status')}" for name, info in stages.items()) or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
