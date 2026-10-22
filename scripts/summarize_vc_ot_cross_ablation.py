#!/usr/bin/env python3
"""Summarize the completed 2x2 VC x OT crossed training runs.

Only best checkpoints are compared; missing/failed cells abort rather than
silently producing a misleading partial ablation table.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/IMUHOI++/RNN_cano"
ROWS = {
    "omomo": (
        ("v4", "v2_armrot_biclose", "v2_armrot_biclose_20260923_omomo"),
        ("v4", "imuhoi_4heads", "vc_ot_cross_ablation_omomo_v4_vc_imuhoi_ot_4heads"),
        ("imuhoi", "v2_armrot_biclose", "vc_ot_cross_ablation_omomo_imuhoi_v2_armrot_biclose"),
        ("imuhoi", "imuhoi_4heads", "relabel_20260920_omomo_imuhoi"),
    ),
    "behave": (
        ("v4", "v2_armrot_biclose", "v2_armrot_biclose_20260923_behave"),
        ("v4", "imuhoi_4heads", "vc_ot_cross_ablation_behave_v4_vc_imuhoi_ot_4heads"),
        ("imuhoi", "v2_armrot_biclose", "vc_ot_cross_ablation_behave_imuhoi_v2_armrot_biclose"),
        ("imuhoi", "imuhoi_4heads", "relabel_20260922_behave_hpsmooth_imuhoi"),
    ),
}
VARIANT = {"v2_armrot_biclose": "v2_armrot_biclose", "imuhoi_4heads": "vc_imuhoi_ot_4heads"}
METRICS = (
    ("obj_trans_err_fusion_refine", "Object (cm)"),
    ("hoi_err_fusion", "HOI (cm)"),
    ("mpjpe", "MPJPE (cm)"),
    ("root_trans_err", "Root (cm)"),
    ("contact_f1_lhand", "L F1"),
    ("contact_f1_rhand", "R F1"),
    ("obj_jitter", "Obj jitter (mm/frame²)"),
)


def main() -> None:
    entries = []
    for dataset, specs in ROWS.items():
        for vc, ot, run in specs:
            manifest_path = RUNS / run / "experiment_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pipeline = manifest["pipelines"][VARIANT[ot]]
            if pipeline["status"] != "complete" or not pipeline.get("metrics_best"):
                raise RuntimeError(f"Incomplete run: {manifest_path} ({pipeline['status']})")
            if pipeline["velocity_contact_arch"] != vc or pipeline["object_trans_arch"] != (
                "v2" if ot == "v2_armrot_biclose" else ot
            ):
                raise RuntimeError(f"Architecture mismatch: {manifest_path}")
            entries.append({
                "dataset": dataset, "VC": vc, "OT": ot, "run": run,
                "checkpoint": pipeline["joint_ot_ckpt"], "config": pipeline["config"],
                **{label: pipeline["metrics_best"][key] for key, label in METRICS},
            })

    out_csv = ROOT / "outputs/analysis/vc_ot_cross_ablation.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=entries[0].keys())
        writer.writeheader()
        writer.writerows(entries)

    lines = [
        "# VC × OT crossed ablation (BEHAVE and OMOMO)", "",
        "The two diagonal baselines use their latest completed, domain-trained runs; the",
        "off-diagonal cells retrain OT and perform the corresponding joint stage, reusing",
        "the same dataset-specific pretrained HP and VC checkpoints. All rows use seed 42,",
        "the canonical splits and full-sequence batch-online test evaluation. Lower is",
        "better for errors/jitter; higher is better for contact F1.", "",
    ]
    headers = ["VC", "OT", *(label for _, label in METRICS)]
    for dataset in ROWS:
        lines.extend([f"## {dataset.upper()}", "", "| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"])
        for row in (r for r in entries if r["dataset"] == dataset):
            numbers = [f"{row[label]:.3f}" for _, label in METRICS]
            lines.append("| " + " | ".join([row["VC"], row["OT"], *numbers]) + " |")
        lines.append("")
    lines.extend([
        "## Interpretation and provenance", "",
        "This is a *pipeline-level* 2×2 crossing, not a strictly isolated architecture",
        "main-effect experiment: the original joint-training recipe fine-tunes HP for",
        "IMUHOI OT, but VC for v2 OT. BEHAVE uses HP smoothing half-width 7 and",
        "object-velocity initialization diff 3 across all four cells; OMOMO uses 0",
        "for both. The v2 rows include forearm/object rotation, bimanual closure,",
        "and the fixed-fusion FK alpha-beta filter (α=0.35, β=0.05).", "",
        "| Dataset | VC | OT | Run manifest | Selected OT checkpoint |", "|---|---|---|---|---|",
    ])
    for row in entries:
        ckpt = Path(row["checkpoint"])
        lines.append(f"| {row['dataset']} | {row['VC']} | {row['OT']} | "
                     f"`outputs/IMUHOI++/RNN_cano/{row['run']}/experiment_manifest.json` | "
                     f"`{ckpt.name}` |")
    lines.extend(["", f"Machine-readable full-precision metrics: `{out_csv.relative_to(ROOT)}`.", ""])
    doc = ROOT / "docs/vc_ot_cross_ablation.md"
    doc.write_text("\n".join(lines), encoding="utf-8")
    print(doc)
    print(out_csv)


if __name__ == "__main__":
    main()