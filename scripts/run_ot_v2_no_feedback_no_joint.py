#!/usr/bin/env python3
"""Train and evaluate OT-v2 without state feedback or joint training."""

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
SOURCE_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN2/omomo_baseline_v4_ot_v2_shared_hp_20260826"
)
DEFAULT_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN2/omomo_baseline_v4_ot_v2_no_feedback_no_joint_20260829"
)
DEFAULT_CFG = PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml"
HP_CKPT = SOURCE_ROOT / (
    "hp/human_pose_08261838_omomo_rnn2_shared_hp_20260826/best.pt"
)
PIPELINES = {
    "baseline": {
        "vc_arch": "imuhoi",
        "vc_ckpt": SOURCE_ROOT / (
            "baseline/velocity_contact_IMUHOI_08261906_"
            "omomo_rnn2_baseline_ot_v2_20260826/best.pt"
        ),
        "old_ot_only_ckpt": SOURCE_ROOT / (
            "baseline/object_trans_v2_vc_IMUHOI_08261916_"
            "omomo_rnn2_baseline_ot_v2_20260826/best.pt"
        ),
    },
    "v4": {
        "vc_arch": "v4",
        "vc_ckpt": SOURCE_ROOT / (
            "v4/velocity_contact_v4_08262101_omomo_rnn2_v4_ot_v2_20260826/best.pt"
        ),
        "old_ot_only_ckpt": SOURCE_ROOT / (
            "v4/object_trans_v2_vc_v4_08262116_"
            "omomo_rnn2_v4_ot_v2_20260826/best.pt"
        ),
    },
}
DATASETS = ("omomo_full", "pahoi_full")


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


def read_metrics(path: Path, dataset: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("results", {}).get(dataset)
    if not isinstance(metrics, dict):
        raise ValueError(f"Evaluation output has no {dataset} metrics: {path}")
    return metrics


def find_ot_checkpoint(pipeline_root: Path, suffix: str) -> Optional[Path]:
    candidates = [
        directory / "best.pt"
        for directory in pipeline_root.iterdir()
        if directory.is_dir()
        and directory.name.startswith("object_trans_v2")
        and suffix in directory.name
        and (directory / "best.pt").is_file()
        and (directory / "final.pt").is_file()
    ] if pipeline_root.is_dir() else []
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, choices=(1, 2), default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = (args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg).resolve()
    root = (
        args.experiment_root
        if args.experiment_root.is_absolute()
        else PROJECT_ROOT / args.experiment_root
    ).resolve()
    required_paths = [SAGE_PYTHON, cfg_path, HP_CKPT]
    for pipeline in PIPELINES.values():
        required_paths.extend((pipeline["vc_ckpt"], pipeline["old_ot_only_ckpt"]))
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.num_workers != 4:
        raise ValueError("Use num_workers=4 for this controlled comparison.")

    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if list(base_cfg.get("train_datasets", [])) != ["omomo"]:
        raise ValueError("The source config must select train_datasets: [omomo].")
    shared_cfg = copy.deepcopy(base_cfg)
    shared_cfg.update(
        {
            "save_dir": str(root),
            "train_datasets": ["omomo"],
            "object_trans_arch": "v2",
            "object_trans_state_feedback": False,
            "joint_train": False,
            "simulate_imu_noise": True,
            "simulate_imu_noise_seed": args.seed,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "pretrained_modules": {
                "human_pose": None,
                "velocity_contact": None,
                "object_trans": None,
            },
        }
    )

    command_plan = {}
    for name, pipeline in PIPELINES.items():
        pipeline_root = root / name
        suffix = f"omomo_{name}_ot_v2_no_feedback_no_joint_20260829"
        train_command = [
            str(SAGE_PYTHON),
            str(PROJECT_ROOT / "train/rnn/train_object_trans.py"),
            "--cfg", str(pipeline_root / "config.yaml"),
            "--save_dir", str(pipeline_root),
            "--run_suffix", suffix,
            "--seed", str(args.seed),
            "--train_datasets", "omomo",
            "--num_workers", str(args.num_workers),
            "--simulate_imu_noise",
            "--epochs", str(args.epochs),
            "--velocity_contact_arch", str(pipeline["vc_arch"]),
            "--object_trans_arch", "v2",
            "--object_trans_state_feedback", "none",
            "--hp_ckpt", str(HP_CKPT),
            "--vc_ckpt", str(pipeline["vc_ckpt"]),
        ]
        command_plan[name] = {
            "suffix": suffix,
            "train": train_command,
            "controls": {
                "old_feedback_no_joint": str(pipeline["old_ot_only_ckpt"]),
                "new_no_feedback_no_joint": "<new best.pt>",
            },
        }

    if args.dry_run:
        print(
            json.dumps(
                {
                    "python": str(SAGE_PYTHON),
                    "config": str(cfg_path),
                    "experiment_root": str(root),
                    "gpu": args.gpu,
                    "max_parallel": args.max_parallel,
                    "epochs": args.epochs,
                    "joint_train": False,
                    "object_trans_state_feedback": False,
                    "pipelines": command_plan,
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
    write_yaml(root / "shared_config.yaml", shared_cfg)
    manifest_path = root / "experiment_manifest.json"
    root_log = root / "orchestrator.log"
    lock = threading.Lock()
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.resume and manifest_path.is_file()
        else {
            "status": "running",
            "phase": "parallel_ot_training",
            "started_at": now(),
            "source_config": str(cfg_path),
            "shared_hp_checkpoint": str(HP_CKPT),
            "object_trans_arch": "v2",
            "object_trans_state_feedback": False,
            "joint_train": False,
            "seed": args.seed,
            "gpu": args.gpu,
            "max_parallel": args.max_parallel,
            "pipelines": {},
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
        with lock, root_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")

    def update_pipeline(name: str, **fields) -> None:
        with lock:
            manifest["pipelines"].setdefault(name, {}).update(fields)
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

    def evaluate(
        name: str,
        pipeline_root: Path,
        variant_cfg: Path,
        vc_arch: str,
        vc_ckpt: Path,
        ot_ckpt: Path,
        feedback_mode: str,
        variant_name: str,
    ) -> Dict[str, dict]:
        metrics_by_dataset = {}
        for dataset in DATASETS:
            output_json = pipeline_root / f"eval_{variant_name}_{dataset}.json"
            records_json = pipeline_root / f"eval_{variant_name}_{dataset}_sequences.json"
            if not output_json.is_file():
                command = [
                    str(SAGE_PYTHON),
                    str(PROJECT_ROOT / "eval_batch_online.py"),
                    "--config", str(variant_cfg),
                    "--dataset", dataset,
                    "--smpl_model_path", "datasets/smpl_models/smplx/SMPLX_NEUTRAL.npz",
                    "--hp_ckpt", str(HP_CKPT),
                    "--vc_ckpt", str(vc_ckpt),
                    "--ot_ckpt", str(ot_ckpt),
                    "--velocity_contact_arch", vc_arch,
                    "--object_trans_arch", "v2",
                    "--object_trans_state_feedback", feedback_mode,
                    "--enable_ot_refine",
                    "--simulate_imu_noise",
                    "--simulate_imu_noise_seed", str(args.seed),
                    "--seed", str(args.seed),
                    "--num_workers", str(args.num_workers),
                    "--online_batch_size", str(args.eval_batch_size),
                    "--online_lookahead", "10",
                    "--output_json", str(output_json),
                    "--sequence_records_json", str(records_json),
                ]
                run_stage(command, pipeline_root / f"eval_{variant_name}_{dataset}.log")
            metrics_by_dataset[dataset] = read_metrics(output_json, dataset)
        return metrics_by_dataset

    def pipeline_worker(name: str, pipeline: dict) -> Tuple[str, bool]:
        pipeline_root = root / name
        pipeline_root.mkdir(parents=True, exist_ok=True)
        variant_cfg = pipeline_root / "config.yaml"
        config = copy.deepcopy(shared_cfg)
        config.update(
            {
                "save_dir": str(pipeline_root),
                "velocity_contact_arch": pipeline["vc_arch"],
                "object_trans_arch": "v2",
                "object_trans_state_feedback": False,
                "joint_train": False,
            }
        )
        write_yaml(variant_cfg, config)
        suffix = command_plan[name]["suffix"]
        update_pipeline(
            name,
            status="running",
            phase="object_trans_training",
            velocity_contact_arch=pipeline["vc_arch"],
            velocity_contact_checkpoint=str(pipeline["vc_ckpt"]),
            config=str(variant_cfg),
        )
        try:
            new_ot_ckpt = find_ot_checkpoint(pipeline_root, suffix)
            if new_ot_ckpt is None:
                run_stage(command_plan[name]["train"], pipeline_root / "object_trans.log")
                new_ot_ckpt = find_ot_checkpoint(pipeline_root, suffix)
            if new_ot_ckpt is None:
                raise FileNotFoundError(f"{name} OT training produced no best.pt")
            update_pipeline(
                name,
                phase="evaluation",
                new_object_trans_checkpoint=str(new_ot_ckpt),
                old_object_trans_checkpoint=str(pipeline["old_ot_only_ckpt"]),
            )
            old_metrics = evaluate(
                name,
                pipeline_root,
                variant_cfg,
                pipeline["vc_arch"],
                pipeline["vc_ckpt"],
                pipeline["old_ot_only_ckpt"],
                "fused",
                "old_feedback_no_joint",
            )
            new_metrics = evaluate(
                name,
                pipeline_root,
                variant_cfg,
                pipeline["vc_arch"],
                pipeline["vc_ckpt"],
                new_ot_ckpt,
                "none",
                "new_no_feedback_no_joint",
            )
            update_pipeline(
                name,
                status="complete",
                phase="complete",
                error=None,
                old_feedback_no_joint_metrics=old_metrics,
                new_no_feedback_no_joint_metrics=new_metrics,
                finished_at=now(),
            )
            return name, True
        except Exception as exc:
            update_pipeline(
                name,
                status="failed",
                phase="failed",
                error=str(exc),
                finished_at=now(),
            )
            log(f"PIPELINE FAILED {name}: {exc}")
            return name, False

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(pipeline_worker, name, pipeline): name
            for name, pipeline in PIPELINES.items()
        }
        results = dict(future.result() for future in as_completed(futures))

    manifest["pipeline_results"] = results
    manifest["status"] = "complete" if all(results.values()) else "failed"
    manifest["phase"] = "complete" if manifest["status"] == "complete" else "pipeline_failed"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
