#!/usr/bin/env python3
"""Run two independent dataset pipelines concurrently on one 16 GB GPU.

Each dataset owns one HP checkpoint.  Its baseline and v3 pipelines reuse only
that checkpoint and remain sequential within the dataset.  Different datasets
run concurrently, bounded by ``--max_parallel_datasets``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "IMUHOI_train_rnn.yaml"
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "IMUHOI++" / "RNN" / "per_dataset_baseline_v3_independent_hp_20260819"
DATASETS = ["behave", "imhd", "omomo", "pahoi", "hodome"]
PIPELINES = [("baseline", "imuhoi"), ("v3", "v3")]
SAGE_PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def find_best(root: Path, prefix: str, suffix: str) -> tuple[Path, Path]:
    matches = [
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and suffix in path.name and (path / "best.pt").is_file()
    ]
    if not matches:
        raise FileNotFoundError(f"No completed {prefix} stage under {root}")
    stage_dir = max(matches, key=lambda path: path.stat().st_mtime)
    return stage_dir, stage_dir / "best.pt"


def find_joint_ot(root: Path, suffix: str) -> Path:
    matches = [
        path for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("object_trans")
        and suffix in path.name
        and (path / "best_velocity_contact.pt").is_file()
        and (path / "best_object_trans.pt").is_file()
    ]
    if not matches:
        raise FileNotFoundError(f"No completed joint OT stage under {root}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_parallel_datasets", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = (args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg).resolve()
    root = (args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root).resolve()
    if not cfg.is_file():
        raise FileNotFoundError(f"Missing config: {cfg}")
    if not SAGE_PYTHON.is_file():
        raise FileNotFoundError(f"Missing WSL SAGE interpreter: {SAGE_PYTHON}")
    if args.num_workers != 4:
        raise ValueError("This experiment requires --num_workers 4.")
    if not 1 <= args.max_parallel_datasets <= 2:
        raise ValueError("The measured 16 GB GPU profile supports at most two concurrent dataset pipelines.")

    root.mkdir(parents=True, exist_ok=True)
    root_log = root / "orchestrator.log"
    manifest_path = root / "experiment_manifest.json"
    state_lock = threading.Lock()
    log_lock = threading.Lock()
    py = str(SAGE_PYTHON)
    base_env = os.environ.copy()
    base_env.update({
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "TMPDIR": "/tmp",
        "TMP": "/tmp",
        "TEMP": "/tmp",
        "IMUHOI_PERSISTENT_WORKERS": "0",
        "IMUHOI_SKIP_DATALOADER_PRIME": "1",
    })
    manifest = {
        "status": "running",
        "phase": "parallel_training",
        "started_at": now(),
        "config": str(cfg),
        "train_datasets": DATASETS,
        "dataset_specific_hp": True,
        "hp_shared_between_baseline_v3_only_within_dataset": True,
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": args.seed,
        "seed": args.seed,
        "object_trans_arch": "current",
        "num_workers_per_pipeline": args.num_workers,
        "max_parallel_datasets": args.max_parallel_datasets,
        "python": py,
        "gpu": 0,
        "datasets": {dataset: {"status": "pending", "phase": "pending"} for dataset in DATASETS},
        "hp": {},
        "pipelines": {},
    }

    def append_log(message: str) -> None:
        with log_lock:
            root_log.parent.mkdir(parents=True, exist_ok=True)
            with root_log.open("a", encoding="utf-8") as handle:
                handle.write(f"[{now()}] {message}\n")

    def commit() -> None:
        with state_lock:
            write_json(manifest_path, manifest)

    def update_dataset(dataset: str, **fields) -> None:
        with state_lock:
            manifest["datasets"][dataset].update(fields)
            write_json(manifest_path, manifest)

    def update_hp(dataset: str, item: dict) -> None:
        with state_lock:
            manifest["hp"][dataset] = copy.deepcopy(item)
            write_json(manifest_path, manifest)

    def update_pipeline(key: str, item: dict) -> None:
        with state_lock:
            manifest["pipelines"][key] = copy.deepcopy(item)
            write_json(manifest_path, manifest)

    def run_stage(command: list[str], log_path: Path) -> None:
        append_log("RUN " + " ".join(command))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=base_env.copy(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        append_log(f"END returncode={result.returncode} log={log_path}")
        if result.returncode != 0:
            raise RuntimeError(f"Stage failed with exit code {result.returncode}: {log_path}")

    def run_dataset(dataset: str) -> bool:
        update_dataset(dataset, status="running", phase="human_pose", started_at=now())
        append_log(f"DATASET START {dataset}")
        hp_root = root / "hp" / dataset
        hp_root.mkdir(parents=True, exist_ok=True)
        hp_suffix = f"hp_{dataset}_parallel_independent_20260819"
        hp_command = [
            py, str(PROJECT_ROOT / "train" / "rnn" / "train_human_pose.py"),
            "--cfg", str(cfg), "--run_suffix", hp_suffix, "--save_dir", str(hp_root),
            "--seed", str(args.seed), "--train_datasets", dataset,
            "--num_workers", str(args.num_workers), "--simulate_imu_noise", "--epochs", "300",
        ]
        hp_item = {"status": "running", "phase": "human_pose", "dataset": dataset, "command": hp_command}
        update_hp(dataset, hp_item)
        try:
            run_stage(hp_command, hp_root / "human_pose.log")
            hp_dir, hp_ckpt = find_best(hp_root, "human_pose", hp_suffix)
            hp_item.update({
                "status": "complete", "phase": "complete", "save_dir": str(hp_dir),
                "best_ckpt": str(hp_ckpt), "finished_at": now(),
            })
            update_hp(dataset, hp_item)
            update_dataset(dataset, phase="baseline:velocity_contact", hp_ckpt=str(hp_ckpt))
            append_log(f"HP COMPLETE dataset={dataset} ckpt={hp_ckpt}")

            for pipeline, vc_arch in PIPELINES:
                pipeline_root = root / pipeline / dataset
                pipeline_root.mkdir(parents=True, exist_ok=True)
                suffix = f"{pipeline}_{dataset}_parallel_independenthp_20260819"
                key = f"{pipeline}/{dataset}"
                item = {
                    "status": "running", "phase": "velocity_contact", "pipeline": pipeline,
                    "dataset": dataset, "velocity_contact_arch": vc_arch, "object_trans_arch": "current",
                    "run_suffix": suffix, "pipeline_dir": str(pipeline_root), "config": str(cfg),
                    "train_datasets": [dataset], "hp_ckpt": str(hp_ckpt), "num_workers": args.num_workers,
                    "seed": args.seed, "simulate_imu_noise": True, "stages": [],
                    "checkpoints": {"hp_ckpt": str(hp_ckpt)},
                }
                update_pipeline(key, item)
                write_json(pipeline_root / "manifest.json", item)
                update_dataset(dataset, phase=f"{pipeline}:velocity_contact")
                vc_command = [
                    py, str(PROJECT_ROOT / "train" / "rnn" / "train_velocity_contact.py"),
                    "--cfg", str(cfg), "--run_suffix", suffix, "--save_dir", str(pipeline_root),
                    "--seed", str(args.seed), "--train_datasets", dataset,
                    "--num_workers", str(args.num_workers), "--simulate_imu_noise", "--epochs", "200",
                    "--velocity_contact_arch", vc_arch, "--hp_ckpt", str(hp_ckpt),
                ]
                run_stage(vc_command, pipeline_root / "velocity_contact.log")
                vc_dir, vc_ckpt = find_best(pipeline_root, "velocity_contact", suffix)
                item["stages"].append({
                    "name": "velocity_contact", "save_dir": str(vc_dir),
                    "best_ckpt": str(vc_ckpt), "command": vc_command,
                })
                item["checkpoints"]["vc_ckpt"] = str(vc_ckpt)
                item["phase"] = "object_trans_joint_train"
                update_pipeline(key, item)
                write_json(pipeline_root / "manifest.json", item)
                update_dataset(dataset, phase=f"{pipeline}:object_trans_joint_train")
                ot_command = [
                    py, str(PROJECT_ROOT / "train" / "rnn" / "train_object_trans.py"),
                    "--cfg", str(cfg), "--run_suffix", suffix, "--save_dir", str(pipeline_root),
                    "--seed", str(args.seed), "--train_datasets", dataset,
                    "--num_workers", str(args.num_workers), "--simulate_imu_noise", "--epochs", "300",
                    "--joint_train", "--velocity_contact_arch", vc_arch, "--object_trans_arch", "current",
                    "--hp_ckpt", str(hp_ckpt), "--vc_ckpt", str(vc_ckpt),
                ]
                run_stage(ot_command, pipeline_root / "object_trans_joint_train.log")
                ot_dir = find_joint_ot(pipeline_root, suffix)
                item["stages"].append({
                    "name": "object_trans_joint_train", "save_dir": str(ot_dir),
                    "joint_vc_ckpt": str(ot_dir / "best_velocity_contact.pt"),
                    "joint_ot_ckpt": str(ot_dir / "best_object_trans.pt"), "command": ot_command,
                })
                item["checkpoints"].update({
                    "joint_vc_ckpt": str(ot_dir / "best_velocity_contact.pt"),
                    "joint_ot_ckpt": str(ot_dir / "best_object_trans.pt"),
                })
                item.update({"status": "complete", "phase": "complete", "finished_at": now()})
                update_pipeline(key, item)
                write_json(pipeline_root / "manifest.json", item)
                append_log(f"PIPELINE COMPLETE {key}")

            update_dataset(dataset, status="complete", phase="complete", finished_at=now())
            append_log(f"DATASET COMPLETE {dataset}")
            return True
        except Exception as exc:
            hp_item.setdefault("error", str(exc))
            update_dataset(dataset, status="failed", phase="failed", error=str(exc), finished_at=now())
            append_log(f"DATASET FAIL {dataset}: {exc}")
            return False

    commit()
    append_log(
        f"START parallel independent experiment root={root} "
        f"max_parallel_datasets={args.max_parallel_datasets}"
    )
    results = {}
    with ThreadPoolExecutor(max_workers=args.max_parallel_datasets) as executor:
        futures = {executor.submit(run_dataset, dataset): dataset for dataset in DATASETS}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                results[dataset] = bool(future.result())
            except Exception as exc:
                results[dataset] = False
                update_dataset(dataset, status="failed", phase="failed", error=str(exc), finished_at=now())
                append_log(f"WORKER FAIL {dataset}: {exc}")

    with state_lock:
        manifest["finished_at"] = now()
        manifest["dataset_results"] = results
        if all(results.get(dataset, False) for dataset in DATASETS):
            manifest["status"] = "complete"
            manifest["phase"] = "training_complete_waiting_for_eval"
        else:
            manifest["status"] = "failed"
            manifest["phase"] = "one_or_more_dataset_pipelines_failed"
        write_json(manifest_path, manifest)
    append_log(f"ALL DATASET WORKERS FINISHED status={manifest['status']}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
