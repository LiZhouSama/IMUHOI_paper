#!/usr/bin/env python3
"""Run independent five-dataset HP -> VC -> OT -> joint_train pipelines.

The two pipelines intentionally run serially because they share one GPU.  The
existing sequential runner owns the stage wiring and checkpoint validation;
this wrapper only supplies the common five-dataset/noise protocol and records
an experiment-level manifest for monitoring.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "IMUHOI_RNN_2" / "multidataset_baseline_v3_experiment_20260817_wsl"
DATASETS = ["behave", "imhd", "omomo", "pahoi", "hodome"]
PIPELINES = [
    ("baseline", "imuhoi", "multi5_baseline_wsl_20260817"),
    ("v3", "v3", "multi5_v3_wsl_20260817"),
]
NUM_WORKERS = os.environ.get("IMUHOI_NUM_WORKERS", "0")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_path = PROJECT_ROOT / "configs" / "IMUHOI_train_rnn.yaml"
    manifest_path = OUTPUT_ROOT / "experiment_manifest.json"
    manifest = {
        "status": "running",
        "started_at": now(),
        "config": str(config_path),
        "train_datasets": DATASETS,
        "simulate_imu_noise": True,
        "seed": 42,
        "object_trans_arch": "current",
        "pipelines": {},
    }
    write_json(manifest_path, manifest)

    for name, vc_arch, pipeline_tag in PIPELINES:
        pipeline_log = OUTPUT_ROOT / f"{name}_runner.log"
        pipeline_dir = PROJECT_ROOT / "outputs" / "IMUHOI_RNN_2" / f"rnn_sequential_{pipeline_tag}"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train" / "rnn" / "run_sequential_training.py"),
            "--cfg",
            str(config_path),
            "--pipeline_tag",
            pipeline_tag,
            "--seed",
            "42",
            "--train_datasets",
            *DATASETS,
            "--velocity_contact_arch",
            vc_arch,
            "--object_trans_arch",
            "current",
            "--simulate_imu_noise",
            "--num_workers",
            NUM_WORKERS,
        ]
        # Baseline HP completed before the WSL worker deadlock was observed;
        # resume it from VC. The v3 pipeline still starts from HP.
        if name == "baseline":
            hp_matches = sorted(
                pipeline_dir.glob("human_pose*multi5_baseline_wsl_20260817*/best.pt"),
                key=lambda path: path.stat().st_mtime,
            )
            if hp_matches:
                command.extend(["--start_stage", "velocity_contact"])
        manifest["pipelines"][name] = {
            "status": "running",
            "velocity_contact_arch": vc_arch,
            "pipeline_tag": pipeline_tag,
            "pipeline_dir": str(pipeline_dir),
            "command": command,
            "log": str(pipeline_log),
            "num_workers": int(NUM_WORKERS),
            "started_at": now(),
        }
        write_json(manifest_path, manifest)
        print(f"[{now()}] START {name}: {' '.join(command)}", flush=True)
        with pipeline_log.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        manifest["pipelines"][name].update(
            {
                "status": "complete" if process.returncode == 0 else "failed",
                "returncode": process.returncode,
                "finished_at": now(),
            }
        )
        write_json(manifest_path, manifest)
        print(f"[{now()}] END {name}: returncode={process.returncode}", flush=True)
        if process.returncode != 0:
            manifest["status"] = "failed"
            manifest["finished_at"] = now()
            write_json(manifest_path, manifest)
            return process.returncode

    manifest["status"] = "complete"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)
    print(f"[{now()}] ALL PIPELINES COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
