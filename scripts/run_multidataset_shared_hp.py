#!/usr/bin/env python3
"""Train one five-dataset HP model, then run baseline/v3 VC -> OT -> joint.

The current YAML is intentionally passed through unchanged.  HumanPose is
trained once in this experiment; both VC architectures then receive that
same explicit HP checkpoint and train VC, ObjectTrans, and the automatic joint
stage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "IMUHOI_train_rnn.yaml"
DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "IMUHOI++" / "RNN" / "multidataset_baseline_v3_shared_hp_20260818"
DATASETS = ["behave", "imhd", "omomo", "pahoi", "hodome"]
HP_RUN_SUFFIX = "sharedhp_human_pose_20260818"
PIPELINES = [
    ("baseline", "imuhoi", "sharedhp_baseline_20260818"),
    ("v3", "v3", "sharedhp_v3_20260818"),
]
SAGE_PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def find_stage_dir(root: Path, prefix: str, run_suffix: str) -> Path:
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and run_suffix in path.name
    ]
    candidates = [path for path in candidates if (path / "best.pt").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No completed {prefix} stage found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def update_pipeline_manifest(item: dict) -> None:
    pipeline_root = Path(item["pipeline_dir"])
    payload = {
        "pipeline_tag": item["pipeline_tag"],
        "cfg": item["config"],
        "pipeline_dir": str(pipeline_root),
        "stages": item.get("stages", []),
        "checkpoints": item.get("checkpoints", {}),
        "status": item.get("status"),
    }
    write_json(pipeline_root / "manifest.json", payload)


def run_stage(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    orchestrator_log: Path,
) -> None:
    append_log(orchestrator_log, f"RUN {' '.join(command)}")
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hp_ckpt",
        type=Path,
        default=None,
        help="Reuse an already completed shared HP checkpoint and skip HP training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg
    experiment_root = args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root
    config_path = config_path.resolve()
    experiment_root = experiment_root.resolve()
    requested_hp_ckpt = None
    if args.hp_ckpt is not None:
        requested_hp_ckpt = args.hp_ckpt if args.hp_ckpt.is_absolute() else PROJECT_ROOT / args.hp_ckpt
        requested_hp_ckpt = requested_hp_ckpt.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if args.num_workers < 1:
        raise ValueError("This experiment requires num_workers >= 1; requested value was not 4.")

    python_bin = str(SAGE_PYTHON if SAGE_PYTHON.is_file() else Path(sys.executable))
    experiment_root.mkdir(parents=True, exist_ok=True)
    orchestrator_log = experiment_root / "orchestrator.log"
    manifest_path = experiment_root / "experiment_manifest.json"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            "CUDA_VISIBLE_DEVICES": "0",
            # Four workers are retained; disabling persistent pre-start avoids
            # the WSL fork/large-preloaded-dataset segfault seen in VC.
            "IMUHOI_PERSISTENT_WORKERS": "0",
            "IMUHOI_SKIP_DATALOADER_PRIME": "1",
        }
    )
    manifest = {
        "status": "running",
        "started_at": now(),
        "config": str(config_path),
        "hp_trained_once": True,
        "hp_reused_across_pipelines": True,
        "train_datasets": DATASETS,
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": 42,
        "seed": args.seed,
        "object_trans_arch": "current",
        "num_workers": args.num_workers,
        "python": python_bin,
        "gpu": 0,
        "pipelines": {},
    }
    write_json(manifest_path, manifest)
    append_log(orchestrator_log, f"START experiment root={experiment_root} shared HP will be trained once")

    hp_command = []
    if requested_hp_ckpt is not None:
        if not requested_hp_ckpt.is_file():
            raise FileNotFoundError(f"Requested shared HP checkpoint does not exist: {requested_hp_ckpt}")
        hp_ckpt = requested_hp_ckpt
        manifest["hp_stage"] = {
            "status": "reused",
            "phase": "complete",
            "run_suffix": HP_RUN_SUFFIX,
            "best_ckpt": str(hp_ckpt),
            "reused_at": now(),
        }
        manifest["hp_ckpt"] = str(hp_ckpt)
        write_json(manifest_path, manifest)
        append_log(orchestrator_log, f"REUSE SHARED HP {hp_ckpt}")
    else:
        hp_command = [
            python_bin,
            str(PROJECT_ROOT / "train" / "rnn" / "train_human_pose.py"),
            "--cfg", str(config_path),
            "--run_suffix", HP_RUN_SUFFIX,
            "--save_dir", str(experiment_root),
            "--seed", str(args.seed),
            "--train_datasets", *DATASETS,
            "--num_workers", str(args.num_workers),
            "--simulate_imu_noise",
            "--epochs", "300",
        ]
        manifest["hp_stage"] = {
            "status": "running",
            "phase": "human_pose",
            "run_suffix": HP_RUN_SUFFIX,
            "command": hp_command,
            "log": str(experiment_root / "human_pose.log"),
        }
        write_json(manifest_path, manifest)
    try:
        if requested_hp_ckpt is None:
            run_stage(hp_command, experiment_root / "human_pose.log", env, orchestrator_log)
            hp_dir = find_stage_dir(experiment_root, "human_pose", HP_RUN_SUFFIX)
            hp_ckpt = hp_dir / "best.pt"
            manifest["hp_stage"].update(
                {
                    "status": "complete",
                    "phase": "complete",
                    "save_dir": str(hp_dir),
                    "best_ckpt": str(hp_ckpt),
                    "finished_at": now(),
                }
            )
            manifest["hp_ckpt"] = str(hp_ckpt)
            write_json(manifest_path, manifest)
            append_log(orchestrator_log, f"SHARED HP COMPLETE {hp_ckpt}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["phase"] = "human_pose_failed"
        manifest["error"] = str(exc)
        manifest["hp_stage"]["status"] = "failed"
        manifest["hp_stage"]["error"] = str(exc)
        manifest["hp_stage"]["finished_at"] = now()
        write_json(manifest_path, manifest)
        append_log(orchestrator_log, f"FAIL human_pose: {exc}")
        return 1

    for name, vc_arch, run_suffix in PIPELINES:
        pipeline_root = experiment_root / name
        pipeline_root.mkdir(parents=True, exist_ok=True)
        item = {
            "status": "running",
            "phase": "velocity_contact",
            "velocity_contact_arch": vc_arch,
            "pipeline_tag": run_suffix,
            "pipeline_dir": str(pipeline_root),
            "config": str(config_path),
            "hp_ckpt": str(hp_ckpt),
            "hp_reused": True,
            "num_workers": args.num_workers,
            "stages": [],
            "checkpoints": {"hp_ckpt": str(hp_ckpt)},
        }
        manifest["pipelines"][name] = item
        write_json(manifest_path, manifest)
        update_pipeline_manifest(item)

        vc_command = [
            python_bin,
            str(PROJECT_ROOT / "train" / "rnn" / "train_velocity_contact.py"),
            "--cfg", str(config_path),
            "--run_suffix", run_suffix,
            "--save_dir", str(pipeline_root),
            "--seed", str(args.seed),
            "--train_datasets", *DATASETS,
            "--num_workers", str(args.num_workers),
            "--simulate_imu_noise",
            "--epochs", "200",
            "--velocity_contact_arch", vc_arch,
            "--hp_ckpt", str(hp_ckpt),
        ]
        try:
            run_stage(vc_command, pipeline_root / "velocity_contact.log", env, orchestrator_log)
            vc_dir = find_stage_dir(pipeline_root, "velocity_contact", run_suffix)
            vc_ckpt = vc_dir / "best.pt"
            item["stages"].append({
                "name": "velocity_contact",
                "save_dir": str(vc_dir),
                "best_ckpt": str(vc_ckpt),
                "command": vc_command,
            })
            item["checkpoints"]["vc_ckpt"] = str(vc_ckpt)
            item["phase"] = "object_trans_joint_train"
            write_json(manifest_path, manifest)
            update_pipeline_manifest(item)

            ot_command = [
                python_bin,
                str(PROJECT_ROOT / "train" / "rnn" / "train_object_trans.py"),
                "--cfg", str(config_path),
                "--run_suffix", run_suffix,
                "--save_dir", str(pipeline_root),
                "--seed", str(args.seed),
                "--train_datasets", *DATASETS,
                "--num_workers", str(args.num_workers),
                "--simulate_imu_noise",
                "--epochs", "300",
                "--joint_train",
                "--velocity_contact_arch", vc_arch,
                "--object_trans_arch", "current",
                "--hp_ckpt", str(hp_ckpt),
                "--vc_ckpt", str(vc_ckpt),
            ]
            run_stage(ot_command, pipeline_root / "object_trans_joint_train.log", env, orchestrator_log)
            ot_dir = find_stage_dir(pipeline_root, "object_trans", run_suffix)
            joint_vc_ckpt = ot_dir / "best_velocity_contact.pt"
            joint_ot_ckpt = ot_dir / "best_object_trans.pt"
            if not joint_vc_ckpt.is_file() or not joint_ot_ckpt.is_file():
                raise FileNotFoundError(
                    "Joint stage completed without both best_velocity_contact.pt and "
                    f"best_object_trans.pt under {ot_dir}"
                )
            item["stages"].append({
                "name": "object_trans_joint_train",
                "save_dir": str(ot_dir),
                "best_ckpt": str(ot_dir / "best.pt"),
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
            update_pipeline_manifest(item)
            write_json(manifest_path, manifest)
        except Exception as exc:
            item["status"] = "failed"
            item["phase"] = "failed"
            item["error"] = str(exc)
            item["finished_at"] = now()
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            update_pipeline_manifest(item)
            write_json(manifest_path, manifest)
            append_log(orchestrator_log, f"FAIL {name}: {exc}")
            return 1

    manifest["status"] = "complete"
    manifest["phase"] = "training_complete_waiting_for_eval"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)
    append_log(orchestrator_log, "ALL TRAINING PIPELINES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
