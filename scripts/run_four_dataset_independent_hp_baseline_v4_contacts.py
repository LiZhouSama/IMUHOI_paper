#!/usr/bin/env python3
"""Train per-dataset HP and baseline/v4 VC, then evaluate contact metrics."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGE_PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
DEFAULT_CONFIG = PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml"
DEFAULT_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN2/four_dataset_independent_hp_baseline_v4_contacts_20260829"
)
DATASETS = ("behave", "imhd", "pahoi", "hodome")
PIPELINES = (("baseline", "imuhoi"), ("v4", "v4"))
BODY_MODEL_PATHS = {
    "behave": "datasets/smpl_models/smplh/male/model.npz",
    "imhd": "datasets/smpl_models/smplh/male/model.npz",
    "pahoi": "datasets/smpl_models/smplx/SMPLX_MALE.npz",
    "hodome": "datasets/smpl_models/smplh/male/model.npz",
}
CONTACT_METRICS = (
    "contact_f1_lhand",
    "contact_f1_rhand",
    "contact_f1_obj",
    "contact_precision_lhand",
    "contact_precision_rhand",
    "contact_recall_lhand",
    "contact_recall_rhand",
    "contact_f05_lhand",
    "contact_f05_rhand",
    "contact_hard_neg_fp_rate_any",
    "contact_hard_neg_fp_rate_lhand",
    "contact_hard_neg_fp_rate_rhand",
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def find_completed_stage(root: Path, prefix: str, suffix: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    candidates = [
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and directory.name.startswith(prefix)
        and suffix in directory.name
        and (directory / "best.pt").is_file()
        and (directory / "final.pt").is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def find_interrupted_stage(root: Path, prefix: str, suffix: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    candidates = [
        directory
        for directory in root.iterdir()
        if directory.is_dir()
        and directory.name.startswith(prefix)
        and suffix in directory.name
        and (directory / "last.pt").is_file()
        and not (directory / "final.pt").is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def load_contact_metrics(path: Path, dataset: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("results", {}).get(dataset)
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing {dataset} metrics in {path}")
    return {key: float(metrics[key]) for key in CONTACT_METRICS}


def finite_mean(values) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def fmt(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-parallel-datasets", type=int, choices=(1, 2), default=2)
    parser.add_argument("--hp-epochs", type=int, default=300)
    parser.add_argument("--vc-epochs", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_summary(root: Path, manifest: dict) -> None:
    rows = []
    for dataset in DATASETS:
        for pipeline, vc_arch in PIPELINES:
            item = manifest["pipelines"][f"{pipeline}/{dataset}"]
            rows.append(
                {
                    "dataset": dataset,
                    "pipeline": pipeline,
                    "velocity_contact_arch": vc_arch,
                    "hp_checkpoint": manifest["hp"][dataset]["best_checkpoint"],
                    "vc_checkpoint": item["best_checkpoint"],
                    "eval_json": item["eval_json"],
                    "sequence_count": item["sequence_count"],
                    "metrics": item["metrics"],
                }
            )

    macro = []
    for pipeline, vc_arch in PIPELINES:
        pipeline_rows = [row for row in rows if row["pipeline"] == pipeline]
        macro.append(
            {
                "pipeline": pipeline,
                "velocity_contact_arch": vc_arch,
                "metrics": {
                    key: finite_mean(row["metrics"][key] for row in pipeline_rows)
                    for key in CONTACT_METRICS
                },
            }
        )

    payload = {
        "generated_at": now(),
        "protocol": {
            "datasets": list(DATASETS),
            "dataset_specific_hp": True,
            "baseline_uses_hp": False,
            "v4_uses_dataset_hp": True,
            "object_trans_trained": False,
            "seed": 42,
            "simulate_imu_noise": True,
            "simulate_imu_noise_seed": 42,
            "online_window": 60,
            "online_lookahead": 10,
        },
        "rows": rows,
        "macro_average": macro,
    }
    json_path = root / "four_dataset_baseline_v4_contact_metrics.json"
    markdown_path = root / "four_dataset_baseline_v4_contact_metrics.md"
    write_json(json_path, payload)

    lines = [
        "# Four-dataset independent-HP baseline/v4 contact evaluation",
        "",
        "Each dataset has a freshly trained HP. v4 consumes that dataset-specific HP; "
        "the IMUHOI baseline is trained independently on the same dataset but does not "
        "consume HP by architecture. No ObjectTrans or joint-training stage is run.",
        "",
        "| Dataset | Pipeline | N | Left F1 | Right F1 | Object F1 | Left P | Right P | Left R | Right R | Hard-neg FP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['dataset']} | {row['pipeline']} | {row['sequence_count']} | "
            f"{fmt(metrics['contact_f1_lhand'])} | "
            f"{fmt(metrics['contact_f1_rhand'])} | "
            f"{fmt(metrics['contact_f1_obj'])} | "
            f"{fmt(metrics['contact_precision_lhand'])} | "
            f"{fmt(metrics['contact_precision_rhand'])} | "
            f"{fmt(metrics['contact_recall_lhand'])} | "
            f"{fmt(metrics['contact_recall_rhand'])} | "
            f"{fmt(metrics['contact_hard_neg_fp_rate_any'])} |"
        )

    lines.extend(
        [
            "",
            "## Delta: v4 - baseline",
            "",
            "Positive is better for F1/precision/recall; negative is better for hard-negative FP.",
            "",
            "| Dataset | Left F1 | Right F1 | Object F1 | Left P | Right P | Hard-neg FP |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_key = {(row["pipeline"], row["dataset"]): row for row in rows}
    for dataset in DATASETS:
        baseline = by_key[("baseline", dataset)]["metrics"]
        v4 = by_key[("v4", dataset)]["metrics"]
        lines.append(
            f"| {dataset} | "
            f"{fmt(v4['contact_f1_lhand'] - baseline['contact_f1_lhand'])} | "
            f"{fmt(v4['contact_f1_rhand'] - baseline['contact_f1_rhand'])} | "
            f"{fmt(v4['contact_f1_obj'] - baseline['contact_f1_obj'])} | "
            f"{fmt(v4['contact_precision_lhand'] - baseline['contact_precision_lhand'])} | "
            f"{fmt(v4['contact_precision_rhand'] - baseline['contact_precision_rhand'])} | "
            f"{fmt(v4['contact_hard_neg_fp_rate_any'] - baseline['contact_hard_neg_fp_rate_any'])} |"
        )

    lines.extend(
        [
            "",
            "## Macro average",
            "",
            "| Pipeline | Left F1 | Right F1 | Object F1 | Left P | Right P | Hard-neg FP |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in macro:
        metrics = row["metrics"]
        lines.append(
            f"| {row['pipeline']} | {fmt(metrics['contact_f1_lhand'])} | "
            f"{fmt(metrics['contact_f1_rhand'])} | "
            f"{fmt(metrics['contact_f1_obj'])} | "
            f"{fmt(metrics['contact_precision_lhand'])} | "
            f"{fmt(metrics['contact_precision_rhand'])} | "
            f"{fmt(metrics['contact_hard_neg_fp_rate_any'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg_path = (args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg).resolve()
    root = (
        args.experiment_root
        if args.experiment_root.is_absolute()
        else PROJECT_ROOT / args.experiment_root
    ).resolve()
    for path in (SAGE_PYTHON, cfg_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.num_workers != 4:
        raise ValueError("Use num_workers=4 for this controlled comparison.")

    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    clean_cfg = copy.deepcopy(base_cfg)
    clean_cfg.update(
        {
            "save_dir": str(root),
            "seed": args.seed,
            "simulate_imu_noise": True,
            "simulate_imu_noise_seed": args.seed,
            "num_workers": args.num_workers,
            "pretrained_modules": {
                "human_pose": None,
                "velocity_contact": None,
                "object_trans": None,
            },
        }
    )

    plan = {
        dataset: {
            "hp": {"epochs": args.hp_epochs},
            "pipelines": {
                pipeline: {"velocity_contact_arch": vc_arch, "epochs": args.vc_epochs}
                for pipeline, vc_arch in PIPELINES
            },
            "body_model_path": BODY_MODEL_PATHS[dataset],
        }
        for dataset in DATASETS
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "python": str(SAGE_PYTHON),
                    "config": str(cfg_path),
                    "experiment_root": str(root),
                    "datasets": plan,
                    "max_parallel_datasets": args.max_parallel_datasets,
                    "object_trans_trained": False,
                    "baseline_uses_hp": False,
                    "v4_uses_dataset_hp": True,
                },
                indent=2,
            )
        )
        return 0

    if root.exists() and not args.resume:
        raise FileExistsError(
            f"Experiment root already exists; use --resume or another path: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    clean_cfg_path = root / "clean_config.yaml"
    write_yaml(clean_cfg_path, clean_cfg)
    manifest_path = root / "experiment_manifest.json"
    root_log = root / "orchestrator.log"
    state_lock = threading.Lock()
    log_lock = threading.Lock()
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.resume and manifest_path.is_file()
        else {
            "status": "running",
            "phase": "parallel_training",
            "started_at": now(),
            "source_config": str(cfg_path),
            "clean_config": str(clean_cfg_path),
            "datasets": {dataset: {"status": "pending"} for dataset in DATASETS},
            "hp": {},
            "pipelines": {},
            "dataset_specific_hp": True,
            "baseline_uses_hp": False,
            "v4_uses_dataset_hp": True,
            "object_trans_trained": False,
            "seed": args.seed,
            "gpu": args.gpu,
            "max_parallel_datasets": args.max_parallel_datasets,
        }
    )
    write_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            "IMUHOI_PERSISTENT_WORKERS": "0",
            "IMUHOI_SKIP_DATALOADER_PRIME": "1",
        }
    )

    def log(message: str) -> None:
        with log_lock, root_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")

    def commit() -> None:
        with state_lock:
            write_json(manifest_path, manifest)

    def update_dataset(dataset: str, **fields) -> None:
        with state_lock:
            manifest["datasets"].setdefault(dataset, {}).update(fields)
            write_json(manifest_path, manifest)

    def update_hp(dataset: str, **fields) -> None:
        with state_lock:
            manifest["hp"].setdefault(dataset, {}).update(fields)
            write_json(manifest_path, manifest)

    def update_pipeline(key: str, **fields) -> None:
        with state_lock:
            manifest["pipelines"].setdefault(key, {}).update(fields)
            write_json(manifest_path, manifest)

    def run_stage(command, log_path: Path) -> None:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        log(f"GPU before {Path(command[1]).stem}: {gpu_result.stdout.strip()}")
        log("RUN " + " ".join(command))
        print(f"[{now()}] RUN {Path(command[1]).name} -> {log_path}", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        log(f"END returncode={result.returncode} log={log_path}")
        print(f"[{now()}] END returncode={result.returncode} log={log_path}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); see {log_path}")

    def run_dataset(dataset: str) -> Tuple[str, bool]:
        update_dataset(dataset, status="running", phase="human_pose", started_at=now())
        hp_root = root / "hp" / dataset
        hp_root.mkdir(parents=True, exist_ok=True)
        hp_suffix = f"hp_{dataset}_independent_20260829"
        try:
            hp_stage = find_completed_stage(hp_root, "human_pose", hp_suffix)
            if hp_stage is None:
                interrupted_hp = find_interrupted_stage(hp_root, "human_pose", hp_suffix)
                hp_command = [
                    str(SAGE_PYTHON),
                    str(PROJECT_ROOT / "train/rnn/train_human_pose.py"),
                    "--cfg", str(clean_cfg_path),
                    "--save_dir", str(hp_root),
                    "--run_suffix", hp_suffix,
                    "--seed", str(args.seed),
                    "--train_datasets", dataset,
                    "--num_workers", str(args.num_workers),
                    "--simulate_imu_noise",
                    "--epochs", str(args.hp_epochs),
                ]
                if interrupted_hp is not None:
                    hp_command.extend(("--resume_dir", str(interrupted_hp)))
                run_stage(hp_command, hp_root / "human_pose.log")
                hp_stage = find_completed_stage(hp_root, "human_pose", hp_suffix)
            if hp_stage is None:
                raise FileNotFoundError(f"{dataset} HP training produced no completed stage")
            hp_ckpt = hp_stage / "best.pt"
            update_hp(
                dataset,
                status="complete",
                best_checkpoint=str(hp_ckpt),
                save_dir=str(hp_stage),
                finished_at=now(),
            )

            for pipeline, vc_arch in PIPELINES:
                key = f"{pipeline}/{dataset}"
                pipeline_root = root / pipeline / dataset
                pipeline_root.mkdir(parents=True, exist_ok=True)
                suffix = f"{pipeline}_{dataset}_independent_20260829"
                update_dataset(dataset, phase=f"{pipeline}:velocity_contact")
                update_pipeline(
                    key,
                    status="running",
                    phase="velocity_contact",
                    dataset=dataset,
                    pipeline=pipeline,
                    velocity_contact_arch=vc_arch,
                    hp_checkpoint=(str(hp_ckpt) if vc_arch == "v4" else None),
                )
                vc_stage = find_completed_stage(pipeline_root, "velocity_contact", suffix)
                if vc_stage is None:
                    interrupted_vc = find_interrupted_stage(
                        pipeline_root, "velocity_contact", suffix
                    )
                    command = [
                        str(SAGE_PYTHON),
                        str(PROJECT_ROOT / "train/rnn/train_velocity_contact.py"),
                        "--cfg", str(clean_cfg_path),
                        "--save_dir", str(pipeline_root),
                        "--run_suffix", suffix,
                        "--seed", str(args.seed),
                        "--train_datasets", dataset,
                        "--num_workers", str(args.num_workers),
                        "--simulate_imu_noise",
                        "--epochs", str(args.vc_epochs),
                        "--velocity_contact_arch", vc_arch,
                    ]
                    if vc_arch == "v4":
                        command.extend(("--hp_ckpt", str(hp_ckpt)))
                    if interrupted_vc is not None:
                        command.extend(("--resume_dir", str(interrupted_vc)))
                    run_stage(command, pipeline_root / "velocity_contact.log")
                    vc_stage = find_completed_stage(pipeline_root, "velocity_contact", suffix)
                if vc_stage is None:
                    raise FileNotFoundError(f"{key} VC training produced no completed stage")
                vc_ckpt = vc_stage / "best.pt"

                output_json = pipeline_root / "eval_contacts.json"
                records_json = pipeline_root / "eval_contacts_sequences.json"
                if not output_json.is_file():
                    eval_command = [
                        str(SAGE_PYTHON),
                        str(PROJECT_ROOT / "eval_batch_online.py"),
                        "--config", str(clean_cfg_path),
                        "--dataset", dataset,
                        "--smpl_model_path", BODY_MODEL_PATHS[dataset],
                        "--hp_ckpt", str(hp_ckpt),
                        "--vc_ckpt", str(vc_ckpt),
                        "--velocity_contact_arch", vc_arch,
                        "--simulate_imu_noise",
                        "--simulate_imu_noise_seed", str(args.seed),
                        "--seed", str(args.seed),
                        "--num_workers", str(args.num_workers),
                        "--online_batch_size", str(args.eval_batch_size),
                        "--online_lookahead", "10",
                        "--output_json", str(output_json),
                        "--sequence_records_json", str(records_json),
                    ]
                    run_stage(eval_command, pipeline_root / "eval_contacts.log")
                metrics = load_contact_metrics(output_json, dataset)
                records_payload = json.loads(records_json.read_text(encoding="utf-8"))
                records = records_payload["datasets"][dataset]
                protocol = records_payload["protocol"]
                assert protocol["object_trans_checkpoint"] is None
                assert protocol["velocity_contact_arch"] == vc_arch
                assert protocol["human_pose_checkpoint"] == str(hp_ckpt)
                update_pipeline(
                    key,
                    status="complete",
                    phase="complete",
                    best_checkpoint=str(vc_ckpt),
                    save_dir=str(vc_stage),
                    eval_json=str(output_json),
                    sequence_records_json=str(records_json),
                    sequence_count=len(records),
                    metrics=metrics,
                    error=None,
                    finished_at=now(),
                )

            update_dataset(dataset, status="complete", phase="complete", error=None, finished_at=now())
            return dataset, True
        except Exception as exc:
            update_dataset(
                dataset,
                status="failed",
                phase="failed",
                error=str(exc),
                finished_at=now(),
            )
            log(f"DATASET FAILED {dataset}: {exc}")
            return dataset, False

    commit()
    with ThreadPoolExecutor(max_workers=args.max_parallel_datasets) as executor:
        futures = {executor.submit(run_dataset, dataset): dataset for dataset in DATASETS}
        results = dict(future.result() for future in as_completed(futures))

    manifest["dataset_results"] = results
    manifest["status"] = "complete" if all(results.values()) else "failed"
    manifest["phase"] = "complete" if manifest["status"] == "complete" else "dataset_failed"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)
    if manifest["status"] == "complete":
        write_summary(root, manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
