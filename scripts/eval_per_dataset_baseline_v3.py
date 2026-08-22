#!/usr/bin/env python3
"""Evaluate each independently trained dataset pipeline on its own test set."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "IMUHOI++" / "RNN" / "per_dataset_baseline_v3_shared_hp_20260819"
DATASETS = ["behave", "imhd", "omomo", "pahoi", "hodome"]
PIPELINES = [("baseline", "imuhoi"), ("v3", "v3")]
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
SAGE_PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def finite_mean(rows: list[dict], key: str):
    values = [row["metrics"].get(key) for row in rows]
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def fmt(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NaN"
    return f"{float(value):.4f}"


def read_metrics(path: Path, dataset: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("results", {}).get(dataset)
    if result is None:
        raise KeyError(f"{path} has no result for {dataset}")
    return {key: result.get(key) for key, _, _ in METRICS}


def load_item(root: Path, pipeline: str, dataset: str) -> dict:
    path = root / pipeline / dataset / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing pipeline manifest: {path}")
    item = json.loads(path.read_text(encoding="utf-8"))
    if item.get("status") != "complete":
        raise RuntimeError(f"Pipeline {pipeline}/{dataset} is not complete: {item.get('status')}")
    for key in ("hp_ckpt", "joint_vc_ckpt", "joint_ot_ckpt"):
        if not Path(item["checkpoints"][key]).is_file():
            raise FileNotFoundError(f"Missing {key} for {pipeline}/{dataset}: {item['checkpoints'][key]}")
    return item


def run_eval(root: Path, pipeline: str, arch: str, dataset: str, item: dict, output_json: Path, workers: int) -> None:
    command = [
        str(SAGE_PYTHON),
        str(PROJECT_ROOT / "eval_batch_online.py"),
        "--config", item["config"],
        "--dataset", dataset,
        "--hp_ckpt", item["checkpoints"]["hp_ckpt"],
        "--vc_ckpt", item["checkpoints"]["joint_vc_ckpt"],
        "--ot_ckpt", item["checkpoints"]["joint_ot_ckpt"],
        "--object_trans_arch", "current",
        "--velocity_contact_arch", arch,
        "--simulate_imu_noise",
        "--simulate_imu_noise_seed", "42",
        "--seed", "42",
        "--num_workers", str(workers),
        "--output_json", str(output_json),
    ]
    log_path = output_json.with_suffix(".log")
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"})
    print(f"[{now()}] EVAL {pipeline}/{dataset}: {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed for {pipeline}/{dataset}; see {log_path}")


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Per-dataset IMUHOI baseline/v3 evaluation",
        "",
        "Protocol: each pipeline was trained independently on its named dataset, reused the same shared HumanPose checkpoint, and was evaluated on the corresponding full-sequence test set with `seed=42`, `simulate_imu_noise`, `object_trans_arch=current`, and `num_workers=4`.",
        "",
        "## Per-dataset results",
        "",
        "| Pipeline | Dataset | " + " | ".join(label for _, label, _ in METRICS) + " |",
        "|---|---|" + "---:|" * len(METRICS),
    ]
    for row in payload["rows"]:
        values = " | ".join(fmt(row["metrics"].get(key)) for key, _, _ in METRICS)
        lines.append(f"| {row['pipeline']} | {row['dataset']} | {values} |")
    lines.extend(["", "## Macro average across five corresponding datasets", ""])
    lines.append("| Pipeline | " + " | ".join(label for _, label, _ in METRICS) + " |")
    lines.append("|---|" + "---:|" * len(METRICS))
    for row in payload["macro_average"]:
        values = " | ".join(fmt(row["metrics"].get(key)) for key, _, _ in METRICS)
        lines.append(f"| {row['pipeline']} | {values} |")
    lines.extend(["", "Lower is better for MPJPE, MPJRE, Root, Jitter, Obj Fusion, and HOI Fusion; higher is better for contact F1.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    if args.num_workers != 4:
        raise ValueError("This evaluation requires --num_workers 4.")
    if not SAGE_PYTHON.is_file():
        raise FileNotFoundError(f"Missing WSL SAGE interpreter: {SAGE_PYTHON}")
    root = args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root
    root = root.resolve()
    manifest_path = root / "experiment_manifest.json"
    experiment = json.loads(manifest_path.read_text(encoding="utf-8"))
    if experiment.get("status") != "complete":
        raise RuntimeError(f"Training is not complete: {experiment.get('status')}")

    raw_dir = root / "eval_results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pipeline, arch in PIPELINES:
        for dataset in DATASETS:
            item = load_item(root, pipeline, dataset)
            output_json = raw_dir / f"{pipeline}_{dataset}.json"
            if not output_json.is_file():
                run_eval(root, pipeline, arch, dataset, item, output_json, args.num_workers)
            rows.append({
                "pipeline": pipeline,
                "velocity_contact_arch": arch,
                "dataset": dataset,
                "eval_json": str(output_json),
                "metrics": read_metrics(output_json, dataset),
            })

    macro_average = []
    for pipeline, arch in PIPELINES:
        pipeline_rows = [row for row in rows if row["pipeline"] == pipeline]
        macro_average.append({
            "pipeline": pipeline,
            "velocity_contact_arch": arch,
            "metrics": {key: finite_mean(pipeline_rows, key) for key, _, _ in METRICS},
        })
    payload = {
        "generated_at": now(),
        "dataset_order": DATASETS,
        "training_mode": "per_dataset",
        "full_sequence": True,
        "seed": 42,
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": 42,
        "object_trans_arch": "current",
        "num_workers": args.num_workers,
        "dataset_specific_hp": bool(experiment.get("dataset_specific_hp", False)),
        "hp_checkpoints": {
            dataset: load_item(root, "baseline", dataset)["checkpoints"]["hp_ckpt"]
            for dataset in DATASETS
        },
        "rows": rows,
        "macro_average": macro_average,
    }
    json_path = root / "per_dataset_baseline_v3_eval.json"
    md_path = root / "per_dataset_baseline_v3_eval.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"[{now()}] WROTE {json_path}", flush=True)
    print(f"[{now()}] WROTE {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
