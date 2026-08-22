#!/usr/bin/env python3
"""Train independent baseline/v3 pipelines for five datasets with one shared HP.

The HumanPose checkpoint is supplied explicitly and is never retrained here.
Each dataset gets an independent VC -> OT(joint_train) pipeline under:

    <experiment_root>/baseline/<dataset>/
    <experiment_root>/v3/<dataset>/

The YAML is passed through unchanged.  The WSL-safe worker environment keeps
``num_workers=4`` while disabling persistent/pre-started workers, which avoids
the known large-dataset fork instability without changing the requested worker
count or AMP configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "IMUHOI_train_rnn.yaml"
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "IMUHOI++" / "RNN" / "per_dataset_baseline_v3_shared_hp_20260819"
DEFAULT_HP = PROJECT_ROOT / "outputs" / "IMUHOI++" / "RNN" / "multidataset_baseline_v3_shared_hp_20260818" / "human_pose_08180144_sharedhp_human_pose_20260818" / "best.pt"
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


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def find_best(stage_root: Path, prefix: str, run_suffix: str) -> tuple[Path, Path]:
    candidates = [
        path
        for path in stage_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and run_suffix in path.name
    ]
    candidates = [path for path in candidates if (path / "best.pt").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No completed {prefix} stage found under {stage_root}")
    save_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    return save_dir, save_dir / "best.pt"


def write_dataset_manifest(path: Path, item: dict) -> None:
    payload = dict(item)
    payload["dataset_manifest"] = str(path)
    write_json(path, payload)


def run_stage(command: list[str], log_path: Path, env: dict[str, str], orchestrator_log: Path) -> None:
    append_log(orchestrator_log, "RUN " + " ".join(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    append_log(orchestrator_log, f"END returncode={result.returncode} log={log_path}")
    if result.returncode != 0:
        raise RuntimeError(f"Stage failed with exit code {result.returncode}: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hp_ckpt", type=Path, default=DEFAULT_HP)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-dataset VC/OT stages.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg
    experiment_root = args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root
    hp_ckpt = args.hp_ckpt if args.hp_ckpt.is_absolute() else PROJECT_ROOT / args.hp_ckpt
    config_path = config_path.resolve()
    experiment_root = experiment_root.resolve()
    hp_ckpt = hp_ckpt.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not hp_ckpt.is_file():
        raise FileNotFoundError(f"Missing shared HP checkpoint: {hp_ckpt}")
    if args.num_workers != 4:
        raise ValueError("This experiment requires --num_workers 4.")
    if not SAGE_PYTHON.is_file():
        raise FileNotFoundError(f"Missing WSL SAGE interpreter: {SAGE_PYTHON}")

    experiment_root.mkdir(parents=True, exist_ok=True)
    orchestrator_log = experiment_root / "orchestrator.log"
    manifest_path = experiment_root / "experiment_manifest.json"
    python_bin = str(SAGE_PYTHON)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            "IMUHOI_PERSISTENT_WORKERS": "0",
            "IMUHOI_SKIP_DATALOADER_PRIME": "1",
        }
    )

    manifest = {
        "status": "running",
        "phase": "training",
        "started_at": now(),
        "config": str(config_path),
        "hp_ckpt": str(hp_ckpt),
        "hp_reused": True,
        "train_datasets": DATASETS,
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": args.seed,
        "seed": args.seed,
        "object_trans_arch": "current",
        "velocity_contact_arches": {"baseline": "imuhoi", "v3": "v3"},
        "num_workers": args.num_workers,
        "python": python_bin,
        "gpu": 0,
        "resume": args.resume,
        "pipelines": {},
    }
    write_json(manifest_path, manifest)
    append_log(orchestrator_log, f"START root={experiment_root} shared_hp={hp_ckpt}")

    for pipeline_name, vc_arch in PIPELINES:
        for dataset in DATASETS:
            pipeline_root = experiment_root / pipeline_name / dataset
            pipeline_root.mkdir(parents=True, exist_ok=True)
            run_suffix = f"{pipeline_name}_{dataset}_sharedhp_20260819"
            key = f"{pipeline_name}/{dataset}"
            item = {
                "status": "running",
                "phase": "velocity_contact",
                "pipeline": pipeline_name,
                "dataset": dataset,
                "velocity_contact_arch": vc_arch,
                "object_trans_arch": "current",
                "run_suffix": run_suffix,
                "pipeline_dir": str(pipeline_root),
                "config": str(config_path),
                "train_datasets": [dataset],
                "hp_ckpt": str(hp_ckpt),
                "num_workers": args.num_workers,
                "seed": args.seed,
                "simulate_imu_noise": True,
                "stages": [],
                "checkpoints": {"hp_ckpt": str(hp_ckpt)},
            }
            manifest["pipelines"][key] = item
            write_dataset_manifest(pipeline_root / "manifest.json", item)
            write_json(manifest_path, manifest)

            vc_command = [
                python_bin,
                str(PROJECT_ROOT / "train" / "rnn" / "train_velocity_contact.py"),
                "--cfg", str(config_path),
                "--run_suffix", run_suffix,
                "--save_dir", str(pipeline_root),
                "--seed", str(args.seed),
                "--train_datasets", dataset,
                "--num_workers", str(args.num_workers),
                "--simulate_imu_noise",
                "--epochs", "200",
                "--velocity_contact_arch", vc_arch,
                "--hp_ckpt", str(hp_ckpt),
            ]
            try:
                if args.resume:
                    try:
                        vc_dir, vc_ckpt = find_best(pipeline_root, "velocity_contact", run_suffix)
                        append_log(orchestrator_log, f"REUSE {key} VC {vc_ckpt}")
                    except FileNotFoundError:
                        run_stage(vc_command, pipeline_root / "velocity_contact.log", env, orchestrator_log)
                        vc_dir, vc_ckpt = find_best(pipeline_root, "velocity_contact", run_suffix)
                else:
                    run_stage(vc_command, pipeline_root / "velocity_contact.log", env, orchestrator_log)
                    vc_dir, vc_ckpt = find_best(pipeline_root, "velocity_contact", run_suffix)
                item["stages"].append({
                    "name": "velocity_contact",
                    "save_dir": str(vc_dir),
                    "best_ckpt": str(vc_ckpt),
                    "command": vc_command,
                })
                item["checkpoints"]["vc_ckpt"] = str(vc_ckpt)
                item["phase"] = "object_trans_joint_train"
                write_dataset_manifest(pipeline_root / "manifest.json", item)
                write_json(manifest_path, manifest)

                ot_command = [
                    python_bin,
                    str(PROJECT_ROOT / "train" / "rnn" / "train_object_trans.py"),
                    "--cfg", str(config_path),
                    "--run_suffix", run_suffix,
                    "--save_dir", str(pipeline_root),
                    "--seed", str(args.seed),
                    "--train_datasets", dataset,
                    "--num_workers", str(args.num_workers),
                    "--simulate_imu_noise",
                    "--epochs", "300",
                    "--joint_train",
                    "--velocity_contact_arch", vc_arch,
                    "--object_trans_arch", "current",
                    "--hp_ckpt", str(hp_ckpt),
                    "--vc_ckpt", str(vc_ckpt),
                ]
                if args.resume:
                    try:
                        ot_dir = max(
                            [path for path in pipeline_root.iterdir() if path.is_dir() and path.name.startswith("object_trans") and run_suffix in path.name and (path / "best_velocity_contact.pt").is_file() and (path / "best_object_trans.pt").is_file()],
                            key=lambda path: path.stat().st_mtime,
                        )
                        append_log(orchestrator_log, f"REUSE {key} OT {ot_dir}")
                    except ValueError:
                        run_stage(ot_command, pipeline_root / "object_trans_joint_train.log", env, orchestrator_log)
                        ot_dir = max(
                            [path for path in pipeline_root.iterdir() if path.is_dir() and path.name.startswith("object_trans") and run_suffix in path.name and (path / "best_velocity_contact.pt").is_file() and (path / "best_object_trans.pt").is_file()],
                            key=lambda path: path.stat().st_mtime,
                        )
                else:
                    run_stage(ot_command, pipeline_root / "object_trans_joint_train.log", env, orchestrator_log)
                    ot_dir = max(
                        [path for path in pipeline_root.iterdir() if path.is_dir() and path.name.startswith("object_trans") and run_suffix in path.name and (path / "best_velocity_contact.pt").is_file() and (path / "best_object_trans.pt").is_file()],
                        key=lambda path: path.stat().st_mtime,
                    )
                joint_vc_ckpt = ot_dir / "best_velocity_contact.pt"
                joint_ot_ckpt = ot_dir / "best_object_trans.pt"
                item["stages"].append({
                    "name": "object_trans_joint_train",
                    "save_dir": str(ot_dir),
                    "joint_vc_ckpt": str(joint_vc_ckpt),
                    "joint_ot_ckpt": str(joint_ot_ckpt),
                    "command": ot_command,
                })
                item["checkpoints"].update({
                    "joint_vc_ckpt": str(joint_vc_ckpt),
                    "joint_ot_ckpt": str(joint_ot_ckpt),
                })
                item["phase"] = "complete"
                item["status"] = "complete"
                item["finished_at"] = now()
                write_dataset_manifest(pipeline_root / "manifest.json", item)
                write_json(manifest_path, manifest)
                append_log(orchestrator_log, f"COMPLETE {key}")
            except Exception as exc:
                item["status"] = "failed"
                item["phase"] = "failed"
                item["error"] = str(exc)
                item["finished_at"] = now()
                manifest["status"] = "failed"
                manifest["phase"] = f"failed:{key}"
                manifest["error"] = str(exc)
                write_dataset_manifest(pipeline_root / "manifest.json", item)
                write_json(manifest_path, manifest)
                append_log(orchestrator_log, f"FAIL {key}: {exc}")
                return 1

    manifest["status"] = "complete"
    manifest["phase"] = "training_complete_waiting_for_eval"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)
    append_log(orchestrator_log, "ALL PER-DATASET BASELINE/V3 TRAINING COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
