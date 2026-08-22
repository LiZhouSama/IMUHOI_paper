#!/usr/bin/env python3
"""Train independent HP/VC/OT pipelines for each dataset.

For every dataset, one dataset-specific HumanPose model is trained once and
then reused only by that dataset's baseline and v3 pipelines.  No checkpoint
is shared across datasets.  Outputs are organized as ``hp/<dataset>``,
``baseline/<dataset>``, and ``v3/<dataset>`` below one fresh run directory.
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


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def find_best(root: Path, prefix: str, suffix: str) -> tuple[Path, Path]:
    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and suffix in p.name and (p / "best.pt").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed {prefix} stage under {root}")
    stage_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    return stage_dir, stage_dir / "best.pt"


def find_joint_ot(root: Path, suffix: str) -> Path:
    candidates = [
        p for p in root.iterdir()
        if p.is_dir()
        and p.name.startswith("object_trans")
        and suffix in p.name
        and (p / "best_velocity_contact.pt").is_file()
        and (p / "best_object_trans.pt").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed joint OT stage under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_stage(command: list[str], log_path: Path, env: dict[str, str], root_log: Path) -> None:
    append_log(root_log, "RUN " + " ".join(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    append_log(root_log, f"END returncode={result.returncode} log={log_path}")
    if result.returncode != 0:
        raise RuntimeError(f"Stage failed with exit code {result.returncode}: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = (args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg).resolve()
    root = (args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root).resolve()
    if not cfg.is_file():
        raise FileNotFoundError(f"Missing config: {cfg}")
    if args.num_workers != 4:
        raise ValueError("This experiment requires --num_workers 4.")
    if not SAGE_PYTHON.is_file():
        raise FileNotFoundError(f"Missing WSL SAGE interpreter: {SAGE_PYTHON}")

    root.mkdir(parents=True, exist_ok=True)
    root_log = root / "orchestrator.log"
    manifest_path = root / "experiment_manifest.json"
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "TMPDIR": "/tmp",
        "TMP": "/tmp",
        "TEMP": "/tmp",
        "IMUHOI_PERSISTENT_WORKERS": "0",
        "IMUHOI_SKIP_DATALOADER_PRIME": "1",
    })
    py = str(SAGE_PYTHON)
    manifest = {
        "status": "running",
        "phase": "human_pose",
        "started_at": now(),
        "config": str(cfg),
        "train_datasets": DATASETS,
        "dataset_specific_hp": True,
        "hp_shared_between_baseline_v3_only_within_dataset": True,
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": args.seed,
        "seed": args.seed,
        "object_trans_arch": "current",
        "num_workers": args.num_workers,
        "python": py,
        "gpu": 0,
        "hp": {},
        "pipelines": {},
    }
    write_json(manifest_path, manifest)
    append_log(root_log, f"START independent per-dataset HP root={root}")

    for dataset in DATASETS:
        hp_root = root / "hp" / dataset
        hp_root.mkdir(parents=True, exist_ok=True)
        hp_suffix = f"hp_{dataset}_independent_20260819"
        hp_command = [
            py, str(PROJECT_ROOT / "train" / "rnn" / "train_human_pose.py"),
            "--cfg", str(cfg), "--run_suffix", hp_suffix, "--save_dir", str(hp_root),
            "--seed", str(args.seed), "--train_datasets", dataset,
            "--num_workers", str(args.num_workers), "--simulate_imu_noise", "--epochs", "300",
        ]
        hp_item = {"status": "running", "phase": "human_pose", "dataset": dataset, "save_dir": str(hp_root), "command": hp_command}
        manifest["hp"][dataset] = hp_item
        manifest["phase"] = f"human_pose:{dataset}"
        write_json(manifest_path, manifest)
        try:
            run_stage(hp_command, hp_root / "human_pose.log", env, root_log)
            hp_dir, hp_ckpt = find_best(hp_root, "human_pose", hp_suffix)
            hp_item.update({"status": "complete", "phase": "complete", "save_dir": str(hp_dir), "best_ckpt": str(hp_ckpt), "finished_at": now()})
            write_json(manifest_path, manifest)
            append_log(root_log, f"HP COMPLETE dataset={dataset} ckpt={hp_ckpt}")
        except Exception as exc:
            hp_item.update({"status": "failed", "phase": "failed", "error": str(exc), "finished_at": now()})
            manifest.update({"status": "failed", "phase": f"human_pose_failed:{dataset}", "error": str(exc)})
            write_json(manifest_path, manifest)
            append_log(root_log, f"FAIL HP dataset={dataset}: {exc}")
            return 1

        for pipeline, vc_arch in PIPELINES:
            pipeline_root = root / pipeline / dataset
            pipeline_root.mkdir(parents=True, exist_ok=True)
            suffix = f"{pipeline}_{dataset}_independenthp_20260819"
            key = f"{pipeline}/{dataset}"
            item = {
                "status": "running", "phase": "velocity_contact", "pipeline": pipeline, "dataset": dataset,
                "velocity_contact_arch": vc_arch, "object_trans_arch": "current", "run_suffix": suffix,
                "pipeline_dir": str(pipeline_root), "config": str(cfg), "train_datasets": [dataset],
                "hp_ckpt": str(hp_ckpt), "num_workers": args.num_workers, "seed": args.seed,
                "simulate_imu_noise": True, "stages": [], "checkpoints": {"hp_ckpt": str(hp_ckpt)},
            }
            manifest["pipelines"][key] = item
            manifest["phase"] = f"{pipeline}:{dataset}:velocity_contact"
            write_json(pipeline_root / "manifest.json", item)
            write_json(manifest_path, manifest)
            vc_command = [
                py, str(PROJECT_ROOT / "train" / "rnn" / "train_velocity_contact.py"),
                "--cfg", str(cfg), "--run_suffix", suffix, "--save_dir", str(pipeline_root),
                "--seed", str(args.seed), "--train_datasets", dataset, "--num_workers", str(args.num_workers),
                "--simulate_imu_noise", "--epochs", "200", "--velocity_contact_arch", vc_arch,
                "--hp_ckpt", str(hp_ckpt),
            ]
            try:
                run_stage(vc_command, pipeline_root / "velocity_contact.log", env, root_log)
                vc_dir, vc_ckpt = find_best(pipeline_root, "velocity_contact", suffix)
                item["stages"].append({"name": "velocity_contact", "save_dir": str(vc_dir), "best_ckpt": str(vc_ckpt), "command": vc_command})
                item["checkpoints"]["vc_ckpt"] = str(vc_ckpt)
                item["phase"] = "object_trans_joint_train"
                manifest["phase"] = f"{pipeline}:{dataset}:object_trans_joint_train"
                write_json(pipeline_root / "manifest.json", item)
                write_json(manifest_path, manifest)

                ot_command = [
                    py, str(PROJECT_ROOT / "train" / "rnn" / "train_object_trans.py"),
                    "--cfg", str(cfg), "--run_suffix", suffix, "--save_dir", str(pipeline_root),
                    "--seed", str(args.seed), "--train_datasets", dataset, "--num_workers", str(args.num_workers),
                    "--simulate_imu_noise", "--epochs", "300", "--joint_train",
                    "--velocity_contact_arch", vc_arch, "--object_trans_arch", "current",
                    "--hp_ckpt", str(hp_ckpt), "--vc_ckpt", str(vc_ckpt),
                ]
                run_stage(ot_command, pipeline_root / "object_trans_joint_train.log", env, root_log)
                ot_dir = find_joint_ot(pipeline_root, suffix)
                item["stages"].append({"name": "object_trans_joint_train", "save_dir": str(ot_dir), "joint_vc_ckpt": str(ot_dir / "best_velocity_contact.pt"), "joint_ot_ckpt": str(ot_dir / "best_object_trans.pt"), "command": ot_command})
                item["checkpoints"].update({"joint_vc_ckpt": str(ot_dir / "best_velocity_contact.pt"), "joint_ot_ckpt": str(ot_dir / "best_object_trans.pt")})
                item.update({"status": "complete", "phase": "complete", "finished_at": now()})
                write_json(pipeline_root / "manifest.json", item)
                write_json(manifest_path, manifest)
                append_log(root_log, f"PIPELINE COMPLETE {key}")
            except Exception as exc:
                item.update({"status": "failed", "phase": "failed", "error": str(exc), "finished_at": now()})
                manifest.update({"status": "failed", "phase": f"failed:{key}", "error": str(exc)})
                write_json(pipeline_root / "manifest.json", item)
                write_json(manifest_path, manifest)
                append_log(root_log, f"FAIL {key}: {exc}")
                return 1

    manifest.update({"status": "complete", "phase": "training_complete_waiting_for_eval", "finished_at": now()})
    write_json(manifest_path, manifest)
    append_log(root_log, "ALL INDEPENDENT HP/BASELINE/V3 PIPELINES COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
