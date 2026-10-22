#!/usr/bin/env python3
"""Five-dataset table: IMUHOI++ (VC-v5ar + OT-v2-armrot-biclose) vs original three-branch IMUHOI.

Reads the full-precision full-sequence eval JSONs of the matched runs (seed 42, unified online
protocol: HP smoothing 7 + diff-3 velocity init) and refuses to summarize if any run deviates
from the shared evaluation protocol.

    python scripts/summarize_imuhoi_pp_vs_imuhoi3b.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/IMUHOI++/RNN_cano"
R3 = "remaining3_v5ar_vs_imuhoi3b_20261010"
DEFAULT_OUT = RUNS / "all5_v5ar_vs_imuhoi3b_20261012"

# dataset -> method -> pipeline dir (holding config.yaml and eval_<ds>_{best,final}.json)
PIPELINES = {
    "omomo": {
        "imuhoi++": "vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose",
        "imuhoi3b": "imuhoi3b_20261012_omomo_unified/vc_imuhoi_ot_3branch",
    },
    "behave": {
        "imuhoi++": "vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose",
        "imuhoi3b": "imuhoi3b_20261008_behave/vc_imuhoi_ot_3branch",
    },
    "imhd": {
        "imuhoi++": f"{R3}/imhd/proposed/v2_armrot_biclose",
        "imuhoi3b": f"{R3}/imhd/imuhoi3b/vc_imuhoi_ot_3branch",
    },
    "pahoi": {
        "imuhoi++": f"{R3}/pahoi/proposed/v2_armrot_biclose",
        "imuhoi3b": f"{R3}/pahoi/imuhoi3b/vc_imuhoi_ot_3branch",
    },
    "hodome": {
        "imuhoi++": f"{R3}/hodome/proposed/v2_armrot_biclose",
        "imuhoi3b": f"{R3}/hodome/imuhoi3b/vc_imuhoi_ot_3branch",
    },
}
LABELS = {"omomo": "OMOMO", "behave": "BEHAVE", "imhd": "IMHD", "pahoi": "PAHOI", "hodome": "HODome"}
METHODS = {"imuhoi++": "IMUHOI++ (ours)", "imuhoi3b": "IMUHOI (3-branch)"}
EXPECTED_ARCH = {"imuhoi++": ("v5", "v2"), "imuhoi3b": ("imuhoi", "imuhoi")}
METRICS = (
    ("obj_trans_err_fusion_refine", "Object (cm)", "low"),
    ("hoi_err_fusion", "HOI (cm)", "low"),
    ("mpjpe", "MPJPE (cm)", "low"),
    ("root_trans_err", "Root (cm)", "low"),
    ("contact_f1_lhand", "L F1", "high"),
    ("contact_f1_rhand", "R F1", "high"),
    ("contact_f1_obj", "Obj F1", "high"),
    ("obj_jitter", "Obj jitter (mm/f²)", "low"),
)
# Every eval must match these values exactly; anything else is a protocol mismatch.
PROTOCOL = {
    "inference_mode": "batch_online",
    "resolved_online_window": 60,
    "resolved_online_lookahead": 10,
    "online_batch_size": 48,
    "human_pose_smoothing_half_width": 7,
    "online_obj_vel_init_diff_frames": 3,
    "object_trans_nocontact_lock_vertical": False,
    "eval_oracle_vc": None,
    "eval_vc_contact_init": None,
    "contact_init_frame": 0,
    "seed": 42,
    "simulate_imu_noise": True,
    "simulate_imu_noise_seed": 42,
    "eval_imu_noise": False,
}


def load_eval(path: Path, dataset: str, method: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bad = {k: payload.get(k) for k, v in PROTOCOL.items() if payload.get(k) != v}
    if bad:
        raise ValueError(f"{path}: protocol mismatch {bad}")
    arch = (payload.get("velocity_contact_arch"), payload.get("object_trans_arch"))
    if arch != EXPECTED_ARCH[method]:
        raise ValueError(f"{path}: arch {arch} != {EXPECTED_ARCH[method]}")
    results = payload["results"]
    if set(results) != {dataset}:
        raise ValueError(f"{path}: results for {sorted(results)}, expected {dataset}")
    return results[dataset]


def full_sequence_count(config_path: Path, dataset: str) -> int:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    entry = cfg["datasets"][dataset]
    test_dir = ROOT / entry.get("full_sequence_test_path", entry["test_path"])
    return sum(1 for _ in test_dir.glob("*.pt"))


def fmt(value, key) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.3f}" if "f1" in key else f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records, missing = [], []
    for ds, methods in PIPELINES.items():
        for method, rel in methods.items():
            pipe = RUNS / rel
            for tag in ("best", "final"):
                path = pipe / f"eval_{ds}_{tag}.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                result = load_eval(path, ds, method)
                records.append({
                    "dataset": ds, "method": method, "checkpoint": tag,
                    "num_full_sequences": full_sequence_count(pipe / "config.yaml", ds),
                    "eval_json": str(path.relative_to(ROOT)),
                    **{key: result.get(key) for key, _, _ in METRICS},
                })
    if missing:
        raise FileNotFoundError("Missing evals:\n" + "\n".join(missing))

    index = {(r["dataset"], r["method"], r["checkpoint"]): r for r in records}
    lines = [
        "# IMUHOI++ vs original IMUHOI (three-branch) on five datasets", "",
        "IMUHOI++ = HP -> VC-v5ar -> OT-v2 (forearm/object rotation + bimanual closure, fixed fusion) -> joint VC+OT (HP frozen).",
        "IMUHOI = HP -> VC-IMUHOI -> OT-IMUHOI with the paper's three candidates (left FK, right FK, IMU integration) -> joint HP+OT (VC frozen).", "",
        "Shared per dataset: initial HP checkpoint, canonical data, seed 42, Noitom-style per-sequence IMU noise (seed 42),",
        "optimizer-step-scaled schedule and early stopping. Evaluation (checked for every row): full-sequence batch-online,",
        "window 60 / lookahead 10, HP smoothing half-width 7, diff-3 object-velocity init, initial contact 0.", "",
        "Caveats: single training seed; `best` is selected on test-window loss (no separate validation split) and early",
        "stopping also observes it; IMUHOI's joint stage fine-tunes HP, so its MPJPE/root differ from ours.", "",
    ]
    for tag in ("best", "final"):
        header = ["Dataset", "Method"] + [label for _, label, _ in METRICS]
        lines += [f"## Checkpoint: {tag}", "", "| " + " | ".join(header) + " |",
                  "|" + "|".join("---" for _ in header) + "|"]
        for ds in PIPELINES:
            for method in METHODS:
                r = index[(ds, method, tag)]
                name = f"**{METHODS[method]}**" if method == "imuhoi++" else METHODS[method]
                lines.append("| " + " | ".join([LABELS[ds], name] + [fmt(r[k], k) for k, _, _ in METRICS]) + " |")
        for method in METHODS:
            vals = []
            for key, _, _ in METRICS:
                xs = [index[(ds, method, tag)][key] for ds in PIPELINES]
                vals.append(fmt(sum(xs) / len(xs), key) if all(x is not None and math.isfinite(x) for x in xs) else "-")
            lines.append("| " + " | ".join(["Mean (5)", METHODS[method]] + vals) + " |")
        lines += ["", f"Relative error reduction, (IMUHOI - ours) / IMUHOI ({tag}):", "",
                  "| Dataset | Object | HOI | Root |", "|---|---|---|---|"]
        for ds in PIPELINES:
            ours, base = index[(ds, "imuhoi++", tag)], index[(ds, "imuhoi3b", tag)]
            cells = [f"{100 * (base[k] - ours[k]) / base[k]:.1f}%"
                     for k in ("obj_trans_err_fusion_refine", "hoi_err_fusion", "root_trans_err")]
            lines.append("| " + " | ".join([LABELS[ds]] + cells) + " |")
        lines.append("")
    lines += ["## Sources", "", "| Dataset | Method | Full-sequence test seqs | Eval JSON (best) |", "|---|---|---|---|"]
    for ds in PIPELINES:
        for method in METHODS:
            r = index[(ds, method, "best")]
            lines.append(f"| {LABELS[ds]} | {METHODS[method]} | {r['num_full_sequences']} | `{r['eval_json']}` |")
    lines.append("")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out_dir / "metrics.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    with (args.out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
