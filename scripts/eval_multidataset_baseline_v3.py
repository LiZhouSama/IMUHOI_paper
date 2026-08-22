#!/usr/bin/env python3
"""Evaluate the five-dataset baseline/v3 pipelines and build comparison tables."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "IMUHOI_RNN_2" / "multidataset_baseline_v3_experiment_20260817_wsl"
DATASETS = ["behave", "imhd", "omomo", "pahoi", "hodome"]
PIPELINES = [
    ("baseline", "imuhoi", "multi5_baseline_wsl_20260817"),
    ("v3", "v3", "multi5_v3_wsl_20260817"),
]
METRICS = [
    ("mpjpe", "MPJPE (cm)", "lower"),
    ("mpjre_angle", "MPJRE (deg)", "lower"),
    ("root_trans_err", "Root (cm)", "lower"),
    ("jitter", "Jitter (mm/frame²)", "lower"),
    ("obj_trans_err_fusion", "Obj Fusion (cm)", "lower"),
    ("hoi_err_fusion", "HOI Fusion (cm)", "lower"),
    ("contact_f1_lhand", "Left F1", "higher"),
    ("contact_f1_rhand", "Right F1", "higher"),
    ("contact_f1_obj", "Object F1", "higher"),
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_pipeline_manifest(output_root: Path, name: str) -> tuple[dict, dict]:
    manifest_path = output_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest_path}")
    experiment = json.loads(manifest_path.read_text(encoding="utf-8"))
    if experiment.get("status") != "complete":
        raise RuntimeError(f"Experiment is not complete: {experiment.get('status')}")
    item = experiment.get("pipelines", {}).get(name)
    if not item or item.get("status") != "complete":
        raise RuntimeError(f"Pipeline {name} is not complete: {item}")
    pipeline_dir = Path(item["pipeline_dir"])
    sequential_manifest_path = pipeline_dir / "manifest.json"
    if not sequential_manifest_path.is_file():
        raise FileNotFoundError(f"Missing sequential manifest: {sequential_manifest_path}")
    sequential = json.loads(sequential_manifest_path.read_text(encoding="utf-8"))
    checkpoints = sequential.get("checkpoints", {})
    hp_ckpt = Path(checkpoints.get("hp_ckpt", ""))
    object_stage = next(
        stage
        for stage in sequential.get("stages", [])
        if stage.get("name") in {"object_trans", "object_trans_joint_train"}
    )
    object_dir = Path(object_stage["save_dir"])
    vc_ckpt = object_dir / "best_velocity_contact.pt"
    ot_ckpt = object_dir / "best_object_trans.pt"
    for path in (hp_ckpt, vc_ckpt, ot_ckpt):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {path}")
    return experiment, {
        "hp_ckpt": str(hp_ckpt),
        "vc_ckpt": str(vc_ckpt),
        "ot_ckpt": str(ot_ckpt),
        "pipeline_dir": str(pipeline_dir),
        "config": experiment.get("config"),
    }


def run_eval(name: str, arch: str, dataset: str, checkpoints: dict, output_json: Path, num_workers: int) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "eval_batch_online.py"),
        "--config",
        checkpoints["config"],
        "--dataset",
        dataset,
        "--hp_ckpt",
        checkpoints["hp_ckpt"],
        "--vc_ckpt",
        checkpoints["vc_ckpt"],
        "--ot_ckpt",
        checkpoints["ot_ckpt"],
        "--object_trans_arch",
        "current",
        "--velocity_contact_arch",
        arch,
        "--simulate_imu_noise",
        "--simulate_imu_noise_seed",
        "42",
        "--seed",
        "42",
        "--num_workers",
        str(num_workers),
        "--output_json",
        str(output_json),
    ]
    log_path = output_json.with_suffix(".log")
    print(f"[{now()}] EVAL {name}/{dataset}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {name}/{dataset}; see {log_path}")


def read_metrics(path: Path, dataset: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if dataset not in results:
        raise KeyError(f"{path} has no result for dataset {dataset}: {list(results)}")
    return {key: results[dataset].get(key) for key, _, _ in METRICS}


def finite_mean(rows: list[dict], key: str):
    values = [row["metrics"].get(key) for row in rows]
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def fmt(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NaN"
    return f"{float(value):.4f}"


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Five-dataset IMUHOI baseline/v3 evaluation",
        "",
        "Protocol: independent HP→VC→OT→joint_train pipelines, `train_datasets=[behave, imhd, omomo, pahoi, hodome]`, full-sequence test loading, `seed=42`, `simulate_imu_noise` with base seed 42, `object_trans_arch=current`, and the configured evaluation worker count.",
        "",
        "## Per-dataset results",
        "",
        "| Pipeline | Dataset | " + " | ".join(label for _, label, _ in METRICS) + " |",
        "|---|---|" + "---:|" * len(METRICS),
    ]
    for row in payload["rows"]:
        values = " | ".join(fmt(row["metrics"].get(key)) for key, _, _ in METRICS)
        lines.append(f"| {row['pipeline']} | {row['dataset']} | {values} |")

    lines.extend(["", "## Macro average across five datasets", ""])
    lines.append("| Pipeline | " + " | ".join(label for _, label, _ in METRICS) + " |")
    lines.append("|---|" + "---:|" * len(METRICS))
    for summary in payload["macro_average"]:
        values = " | ".join(fmt(summary["metrics"].get(key)) for key, _, _ in METRICS)
        lines.append(f"| {summary['pipeline']} | {values} |")
    lines.extend([
        "",
        "Lower is better for MPJPE, MPJRE, Root, Jitter, Obj Fusion, and HOI Fusion; higher is better for contact F1.",
        "",
        "Checkpoint and raw JSON paths are recorded in `multidataset_baseline_v3_eval.json`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate five-dataset baseline/v3 pipelines.")
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    output_root = args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for name, arch, _ in PIPELINES:
        _, checkpoints[name] = load_pipeline_manifest(output_root, name)

    rows = []
    raw_dir = output_root / "eval_results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, arch, _ in PIPELINES:
        for dataset in DATASETS:
            output_json = raw_dir / f"{name}_{dataset}.json"
            if not output_json.is_file():
                run_eval(name, arch, dataset, checkpoints[name], output_json, args.num_workers)
            rows.append(
                {
                    "pipeline": name,
                    "velocity_contact_arch": arch,
                    "dataset": dataset,
                    "eval_json": str(output_json),
                    "metrics": read_metrics(output_json, dataset),
                }
            )

    macro_average = []
    for name, arch, _ in PIPELINES:
        pipeline_rows = [row for row in rows if row["pipeline"] == name]
        macro_average.append(
            {
                "pipeline": name,
                "velocity_contact_arch": arch,
                "metrics": {key: finite_mean(pipeline_rows, key) for key, _, _ in METRICS},
            }
        )

    payload = {
        "generated_at": now(),
        "dataset_order": DATASETS,
        "full_sequence": True,
        "seed": 42,
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": 42,
        "object_trans_arch": "current",
        "pipelines": checkpoints,
        "rows": rows,
        "macro_average": macro_average,
    }
    payload["num_workers"] = args.num_workers
    json_path = output_root / "multidataset_baseline_v3_eval.json"
    md_path = output_root / "multidataset_baseline_v3_eval.md"
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    print(f"[{now()}] WROTE {json_path}", flush=True)
    print(f"[{now()}] WROTE {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
