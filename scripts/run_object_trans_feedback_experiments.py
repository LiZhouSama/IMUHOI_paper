#!/usr/bin/env python3
"""Run the controlled shared-head/ObjectTrans feedback comparison.

This launcher intentionally trains only two models:

1. shared prediction head without OT-state feedback;
2. the same architecture with explicit OT-state feedback.

The feedback checkpoint is evaluated twice (windowed and stateful online), so
the online-policy comparison is not confounded by a second stochastic training
run.  The source config is never modified: all three effective YAML snapshots,
commands, result JSON paths, and GPU memory samples are stored under one
experiment directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _raise_nofile_limit(minimum: int = 131072) -> Dict[str, int]:
    """Raise the inherited descriptor limit for dataset shared-memory preload.

    IMUDataset shares every tensor from thousands of sequence files.  The
    default soft limit (10,240 in this container) is lower than the roughly
    82k storage descriptors required by the configured OMOMO train+test split.
    The raised limit is inherited by the training/evaluation subprocesses.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(int(hard), max(int(soft), int(minimum)))
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    effective_soft, effective_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if effective_soft < minimum:
        raise RuntimeError(
            f"RLIMIT_NOFILE soft limit is {effective_soft}, below the required {minimum}; "
            "increase the container's open-file limit before training."
        )
    return {"soft": int(effective_soft), "hard": int(effective_hard)}


def _parse_gpu_status() -> List[Dict[str, int]]:
    """Read physical-GPU memory state without depending on PyTorch."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: List[Dict[str, int]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "free_mib": int(parts[1]),
                "used_mib": int(parts[2]),
                "total_mib": int(parts[3]),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU memory rows.")
    return rows


def _print_gpu_status(prefix: str) -> List[Dict[str, int]]:
    rows = _parse_gpu_status()
    state = ", ".join(
        f"GPU {row['index']}: free={row['free_mib']}MiB "
        f"used={row['used_mib']}MiB/{row['total_mib']}MiB"
        for row in rows
    )
    print(f"[{prefix}] {state}", flush=True)
    return rows


class _GpuMonitor:
    """Append physical-GPU memory samples while one sequential run is active."""

    def __init__(self, gpu: int, csv_path: Path, interval_sec: float):
        self.gpu = int(gpu)
        self.csv_path = csv_path
        self.interval_sec = float(interval_sec)
        self.phase = "setup"
        self.records: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("timestamp", "phase", "gpu", "free_mib", "used_mib", "total_mib"),
            )
            writer.writeheader()
        self.sample()
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def sample(self) -> None:
        try:
            row = next(item for item in _parse_gpu_status() if item["index"] == self.gpu)
        except (RuntimeError, StopIteration, subprocess.SubprocessError) as exc:
            print(f"[GPU monitor] sample failed: {exc}", flush=True)
            return
        with self._lock:
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "phase": self.phase,
                "gpu": self.gpu,
                "free_mib": row["free_mib"],
                "used_mib": row["used_mib"],
                "total_mib": row["total_mib"],
            }
            self.records.append(record)
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=record.keys()).writerow(record)
        print(
            f"[GPU monitor:{record['phase']}] GPU {self.gpu}: "
            f"free={record['free_mib']}MiB used={record['used_mib']}MiB/"
            f"{record['total_mib']}MiB",
            flush=True,
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.sample()

    def close(self) -> Dict[str, Optional[int]]:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_sec + 1.0))
        self.sample()
        if not self.records:
            return {"max_used_mib": None, "min_free_mib": None, "total_mib": None}
        return {
            "max_used_mib": max(int(row["used_mib"]) for row in self.records),
            "min_free_mib": min(int(row["free_mib"]) for row in self.records),
            "total_mib": int(self.records[-1]["total_mib"]),
        }


@dataclass(frozen=True)
class _Variant:
    name: str
    feedback: bool
    online_mode: str
    run_suffix: str


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    if not tag:
        raise ValueError("--tag must contain at least one alphanumeric character.")
    return tag


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _make_effective_config(
    base_config: Dict[str, Any],
    variant: _Variant,
    tag: str,
    source_config_path: Path,
    path: Path,
    num_workers: Optional[int],
) -> None:
    payload = copy.deepcopy(base_config)
    payload["object_trans_state_feedback"] = bool(variant.feedback)
    payload["object_trans_online_mode"] = variant.online_mode
    payload["object_trans_experiment"] = {
        "tag": tag,
        "variant": variant.name,
        "shared_feedback_checkpoint": variant.name.startswith("feedback"),
        "source_config": str(source_config_path),
    }
    if num_workers is not None:
        payload["num_workers"] = int(num_workers)
    _write_yaml(path, payload)


def _resolve_pretrained_paths(base_config: Dict[str, Any]) -> None:
    modules = base_config.get("pretrained_modules")
    if not isinstance(modules, dict):
        raise ValueError("Base config must define pretrained_modules for HP and VC.")
    for name in ("human_pose", "velocity_contact"):
        value = modules.get(name)
        if not value:
            raise ValueError(f"Base config pretrained_modules.{name} is required for this comparison.")
        path = _resolve_project_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Configured pretrained_modules.{name} does not exist: {path}")


def _find_run_dir(save_root: Path, suffix: str) -> Path:
    matches = sorted(
        path
        for path in save_root.glob(f"object_trans_*_{suffix}")
        if path.is_dir()
    )
    if len(matches) != 1:
        rendered = ", ".join(str(path) for path in matches) or "none"
        raise RuntimeError(
            f"Expected exactly one ObjectTrans run directory for suffix '{suffix}', found: {rendered}."
        )
    return matches[0]


def _assert_no_existing_run(save_root: Path, suffix: str) -> None:
    matches = list(save_root.glob(f"object_trans_*_{suffix}"))
    if matches:
        rendered = ", ".join(str(path) for path in matches)
        raise FileExistsError(
            f"Refusing to reuse run suffix '{suffix}'. Choose a new --tag; existing: {rendered}"
        )


def _run_command(
    command: Iterable[str],
    *,
    label: str,
    environment: Dict[str, str],
    monitor: _GpuMonitor,
    dry_run: bool,
) -> None:
    command_list = [str(item) for item in command]
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join(command_list), flush=True)
    _print_gpu_status(f"pre-{label}")
    if dry_run:
        return
    monitor.set_phase(label)
    monitor.sample()
    subprocess.run(command_list, cwd=PROJECT_ROOT, env=environment, check=True)
    monitor.sample()
    _print_gpu_status(f"post-{label}")


def _write_comparison_summary(
    result_paths: Dict[str, Path],
    output_dir: Path,
) -> Dict[str, str]:
    """Collect all scalar metrics into one JSON and one comparison CSV."""
    collected: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for variant, path in result_paths.items():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results")
        if not isinstance(results, dict):
            raise ValueError(f"Evaluation result has no mapping-valued 'results': {path}")
        collected[variant] = {
            str(dataset): metrics
            for dataset, metrics in results.items()
            if isinstance(metrics, dict)
        }

    json_path = output_dir / "comparison_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(collected, handle, indent=2, ensure_ascii=False)

    datasets = sorted({dataset for value in collected.values() for dataset in value})
    csv_path = output_dir / "comparison_summary.csv"
    fieldnames = ["dataset", "metric", *result_paths.keys()]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in datasets:
            metric_names = sorted(
                {
                    metric
                    for value in collected.values()
                    for metric in value.get(dataset, {})
                    if isinstance(metric, str)
                }
            )
            for metric in metric_names:
                row: Dict[str, Any] = {"dataset": dataset, "metric": metric}
                for variant, values in collected.items():
                    row[variant] = values.get(dataset, {}).get(metric, "")
                writer.writerow(row)
    return {"json": str(json_path), "csv": str(csv_path)}


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run two joint-training OT variants and three controlled online evaluations."
    )
    parser.add_argument("--cfg", default="configs/IMUHOI_train_rnn.yaml", help="Untouched source config.")
    parser.add_argument(
        "--tag",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Unique comparison tag; it is embedded in both ObjectTrans output directories.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Physical GPU index. Default: select the GPU with the largest free memory before launch.",
    )
    parser.add_argument(
        "--online-batch-size",
        type=int,
        default=64,
        help="Batch size for eval_batch_online.py; independent from the training batch size in --cfg.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override DataLoader workers in every effective training snapshot.",
    )
    parser.add_argument(
        "--online-window",
        type=int,
        default=None,
        help="Optional eval override. Default: preserve test.window from --cfg.",
    )
    parser.add_argument(
        "--gpu-monitor-interval-sec",
        type=float,
        default=30.0,
        help="GPU memory sample interval while a child process runs.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable. Invoke this launcher after activating the intended conda environment.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write snapshots and print commands without running them.")
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    if args.online_batch_size < 1:
        raise ValueError("--online-batch-size must be >= 1")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.gpu_monitor_interval_sec <= 0:
        raise ValueError("--gpu-monitor-interval-sec must be > 0")
    nofile_limits = _raise_nofile_limit()
    print(
        f"[preflight] RLIMIT_NOFILE soft={nofile_limits['soft']} hard={nofile_limits['hard']}",
        flush=True,
    )

    source_config_path = _resolve_project_path(args.cfg)
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Source config does not exist: {source_config_path}")
    with source_config_path.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    if not isinstance(base_config, dict):
        raise ValueError(f"Expected a mapping in {source_config_path}")
    _resolve_pretrained_paths(base_config)

    tag = _safe_tag(args.tag)
    seed = int(base_config.get("seed", 42))
    save_root = _resolve_project_path(base_config.get("save_dir", "outputs"))
    experiment_root = save_root / "object_trans_feedback_experiments" / tag
    if experiment_root.exists():
        raise FileExistsError(f"Experiment directory already exists: {experiment_root}. Choose a new --tag.")
    experiment_root.mkdir(parents=True, exist_ok=False)
    configs_dir = experiment_root / "configs"
    results_dir = experiment_root / "results"
    configs_dir.mkdir()
    results_dir.mkdir()
    shutil.copy2(source_config_path, experiment_root / "source_config.yaml")

    initial_gpu_status = _print_gpu_status("preflight")
    gpu_ids = {row["index"] for row in initial_gpu_status}
    gpu = int(args.gpu) if args.gpu is not None else max(initial_gpu_status, key=lambda row: row["free_mib"])["index"]
    if gpu not in gpu_ids:
        raise ValueError(f"--gpu {gpu} is not available. Detected physical GPUs: {sorted(gpu_ids)}")
    selected = next(row for row in initial_gpu_status if row["index"] == gpu)
    print(
        f"Selected physical GPU {gpu}: {selected['free_mib']}MiB free / {selected['total_mib']}MiB total. "
        "CUDA_VISIBLE_DEVICES remaps it to cuda:0 for the existing config.",
        flush=True,
    )

    baseline = _Variant(
        name="shared_head_no_feedback_window",
        feedback=False,
        online_mode="window",
        run_suffix=f"otfb-{tag}-baseline",
    )
    feedback_window = _Variant(
        name="feedback_window",
        feedback=True,
        online_mode="window",
        run_suffix=f"otfb-{tag}-feedback",
    )
    feedback_stateful = _Variant(
        name="feedback_stateful",
        feedback=True,
        online_mode="stateful",
        run_suffix=feedback_window.run_suffix,
    )

    variants = (baseline, feedback_window, feedback_stateful)
    config_paths: Dict[str, Path] = {}
    for variant in variants:
        config_path = configs_dir / f"{variant.name}.yaml"
        _make_effective_config(
            base_config,
            variant,
            tag,
            source_config_path,
            config_path,
            args.num_workers,
        )
        config_paths[variant.name] = config_path

    _assert_no_existing_run(save_root, baseline.run_suffix)
    _assert_no_existing_run(save_root, feedback_window.run_suffix)

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    monitor = _GpuMonitor(gpu, experiment_root / "gpu_memory.csv", args.gpu_monitor_interval_sec)
    monitor.start()

    manifest: Dict[str, Any] = {
        "tag": tag,
        "source_config": str(source_config_path),
        "seed": seed,
        "physical_gpu": gpu,
        "rlimit_nofile": nofile_limits,
        "training_batch_size_from_source_config": base_config.get("batch_size"),
        "num_workers_source_config": base_config.get("num_workers"),
        "num_workers_effective": args.num_workers if args.num_workers is not None else base_config.get("num_workers"),
        "online_batch_size": args.online_batch_size,
        "online_window_override": args.online_window,
        "variants": {
            variant.name: {
                "feedback": variant.feedback,
                "online_mode": variant.online_mode,
                "config": str(config_paths[variant.name]),
            }
            for variant in variants
        },
        "commands": [],
    }

    try:
        train_variants = (baseline, feedback_window)
        run_dirs: Dict[str, Path] = {}
        for variant in train_variants:
            train_command = [
                args.python,
                "train/rnn/train_object_trans.py",
                "--cfg",
                str(config_paths[variant.name]),
                "--joint_train",
                "--seed",
                str(seed),
                "--run_suffix",
                variant.run_suffix,
            ]
            manifest["commands"].append({"label": f"train_{variant.name}", "command": train_command})
            _run_command(
                train_command,
                label=f"train_{variant.name}",
                environment=environment,
                monitor=monitor,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                run_dir = _find_run_dir(save_root, variant.run_suffix)
                for filename in ("best_object_trans.pt", "best_velocity_contact.pt"):
                    checkpoint_path = run_dir / filename
                    if not checkpoint_path.is_file():
                        raise FileNotFoundError(
                            f"Joint training did not produce required checkpoint: {checkpoint_path}"
                        )
                run_dirs[variant.name] = run_dir

        if args.dry_run:
            print("Dry run complete; training output directories/checkpoints were not resolved.", flush=True)
            return 0

        for variant, source_variant in (
            (baseline, baseline),
            (feedback_window, feedback_window),
            (feedback_stateful, feedback_window),
        ):
            run_dir = run_dirs[source_variant.name]
            result_path = results_dir / f"{variant.name}.json"
            eval_command = [
                args.python,
                "eval_batch_online.py",
                "--config",
                str(config_paths[variant.name]),
                "--velocity_contact_ckpt",
                str(run_dir / "best_velocity_contact.pt"),
                "--object_trans_ckpt",
                str(run_dir / "best_object_trans.pt"),
                "--online_batch_size",
                str(args.online_batch_size),
                "--num_workers",
                str(args.num_workers if args.num_workers is not None else base_config.get("num_workers", 0)),
                "--seed",
                str(seed),
                "--compare_3",
                "--output_json",
                str(result_path),
            ]
            if args.online_window is not None:
                eval_command.extend(["--online_window", str(args.online_window)])
            manifest["commands"].append({"label": f"eval_{variant.name}", "command": eval_command})
            _run_command(
                eval_command,
                label=f"eval_{variant.name}",
                environment=environment,
                monitor=monitor,
                dry_run=False,
            )
            manifest["variants"][variant.name]["checkpoint_source_variant"] = source_variant.name
            manifest["variants"][variant.name]["object_trans_checkpoint"] = str(run_dir / "best_object_trans.pt")
            manifest["variants"][variant.name]["velocity_contact_checkpoint"] = str(run_dir / "best_velocity_contact.pt")
            manifest["variants"][variant.name]["result_json"] = str(result_path)
        manifest["comparison_summary"] = _write_comparison_summary(
            {
                baseline.name: results_dir / f"{baseline.name}.json",
                feedback_window.name: results_dir / f"{feedback_window.name}.json",
                feedback_stateful.name: results_dir / f"{feedback_stateful.name}.json",
            },
            results_dir,
        )
    finally:
        manifest["gpu_memory_summary"] = monitor.close()
        with (experiment_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"\nComparison complete: {experiment_root}", flush=True)
    print(f"GPU samples: {experiment_root / 'gpu_memory.csv'}", flush=True)
    print(f"Manifest: {experiment_root / 'manifest.json'}", flush=True)
    print(f"GPU memory summary: {manifest['gpu_memory_summary']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
