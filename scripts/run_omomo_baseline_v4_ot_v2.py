#!/usr/bin/env python3
"""Train selected OMOMO VC pipelines with a shared HP and ObjectTrans v2."""
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

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGE_PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
DEFAULT_CFG = PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml"
DEFAULT_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN2/omomo_baseline_v4_ot_v2_shared_hp_20260826"
)
PIPELINE_ARCHES = {"baseline": "imuhoi", "v3": "v3", "v4": "v4"}
DEFAULT_PIPELINES = ("baseline", "v4")
OBJECT_TRANS_ARCH = "v2"
METRICS = (
    ("mpjpe", "MPJPE (cm)"),
    ("mpjre_angle", "MPJRE (deg)"),
    ("root_trans_err", "Root (cm)"),
    ("jitter", "Jitter (mm/frame^2)"),
    ("obj_trans_err_fusion_raw", "Object raw (cm)"),
    ("obj_trans_err_fusion_refine", "Object published (cm)"),
    ("hoi_err_fusion", "HOI published (cm)"),
    ("contact_f1_lhand", "Left F1"),
    ("contact_f1_rhand", "Right F1"),
    ("contact_f1_obj", "Object F1"),
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def find_best(root: Path, prefix: str, suffix: str) -> Path | None:
    if not root.is_dir():
        return None
    matches = [
        path / "best.pt"
        for path in root.iterdir()
        if path.is_dir()
        and path.name.lower().startswith(prefix.lower())
        and suffix in path.name
        and (path / "best.pt").is_file()
    ]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def find_joint_checkpoints(root: Path, suffix: str) -> tuple[Path, Path] | None:
    if not root.is_dir():
        return None
    matches = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("object_trans_v2")
        and suffix in path.name
        and (path / "best_velocity_contact.pt").is_file()
        and (path / "best_object_trans.pt").is_file()
    ]
    if not matches:
        return None
    stage = max(matches, key=lambda path: path.stat().st_mtime)
    return stage / "best_velocity_contact.pt", stage / "best_object_trans.pt"


def read_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("results", {}).get("omomo")
    if not isinstance(metrics, dict):
        metrics = payload.get("omomo", payload)
    if not isinstance(metrics, dict):
        raise ValueError(f"Evaluation output has no OMOMO metrics: {path}")
    return metrics


def format_metric(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NaN"
    return f"{numeric:.4f}" if math.isfinite(numeric) else "NaN"


def metric_delta(v4_value, baseline_value):
    try:
        delta = float(v4_value) - float(baseline_value)
    except (TypeError, ValueError):
        return None
    return delta if math.isfinite(delta) else None


def write_summary(root: Path, manifest: dict) -> None:
    rows = []
    for name, arch in PIPELINES:
        item = manifest["pipelines"][name]
        if item.get("status") != "complete":
            continue
        rows.append(
            {
                "pipeline": name,
                "velocity_contact_arch": arch,
                "object_trans_arch": OBJECT_TRANS_ARCH,
                "metrics": item["metrics"],
                "joint_vc_ckpt": item["joint_vc_ckpt"],
                "joint_ot_ckpt": item["joint_ot_ckpt"],
                "eval_json": item["eval_json"],
            }
        )
    by_name = {row["pipeline"]: row for row in rows}
    deltas = {}
    if {"baseline", "v4"}.issubset(by_name):
        for key, _ in METRICS:
            deltas[key] = metric_delta(
                by_name["v4"]["metrics"].get(key),
                by_name["baseline"]["metrics"].get(key),
            )
    payload = {
        "generated_at": now(),
        "dataset": "omomo",
        "full_sequence": True,
        "shared_hp_ckpt": manifest["hp_ckpt"],
        "object_trans_arch": OBJECT_TRANS_ARCH,
        "seed": manifest["seed"],
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": manifest["simulate_imu_noise_seed"],
        "rows": rows,
        "delta_v4_minus_baseline": deltas,
    }
    write_json(root / "omomo_baseline_v4_ot_v2_eval.json", payload)

    lines = [
        "# OMOMO baseline vs v4 with shared fresh HP and ObjectTrans v2",
        "",
        f"Shared HP: `{manifest['hp_ckpt']}`",
        "",
        (
            "Both pipelines retrain VC and ObjectTrans v2, run joint training, "
            "and use full-sequence batch-online evaluation with simulated IMU noise."
        ),
        "",
        "| Pipeline | VC arch | "
        + " | ".join(label for _, label in METRICS)
        + " |",
        "|---|---|" + "---:|" * len(METRICS),
    ]
    for row in rows:
        values = " | ".join(
            format_metric(row["metrics"].get(key)) for key, _ in METRICS
        )
        lines.append(
            f"| {row['pipeline']} | `{row['velocity_contact_arch']}` | {values} |"
        )
    if deltas:
        values = " | ".join(format_metric(deltas.get(key)) for key, _ in METRICS)
        lines.append(f"| v4 - baseline | `delta` | {values} |")
    lines.append("")
    (root / "omomo_baseline_v4_ot_v2_eval.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=tuple(PIPELINE_ARCHES),
        default=list(DEFAULT_PIPELINES),
    )
    parser.add_argument("--hp_ckpt", type=Path, default=None)
    parser.add_argument("--run_tag", type=str, default="20260826")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hp_epochs", type=int, default=300)
    parser.add_argument("--vc_epochs", type=int, default=200)
    parser.add_argument("--ot_epochs", type=int, default=300)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_pipelines = tuple(
        (name, PIPELINE_ARCHES[name]) for name in dict.fromkeys(args.pipelines)
    )
    cfg_path = (
        args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg
    ).resolve()
    root = (
        args.experiment_root
        if args.experiment_root.is_absolute()
        else PROJECT_ROOT / args.experiment_root
    ).resolve()
    for path in (SAGE_PYTHON, cfg_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    explicit_hp_ckpt = None
    if args.hp_ckpt is not None:
        explicit_hp_ckpt = (
            args.hp_ckpt
            if args.hp_ckpt.is_absolute()
            else PROJECT_ROOT / args.hp_ckpt
        ).resolve()
        if not explicit_hp_ckpt.is_file():
            raise FileNotFoundError(explicit_hp_ckpt)
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
            "object_trans_arch": OBJECT_TRANS_ARCH,
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

    if args.dry_run:
        print(
            json.dumps(
                {
                    "python": str(SAGE_PYTHON),
                    "source_config": str(cfg_path),
                    "experiment_root": str(root),
                    "pipelines": dict(selected_pipelines),
                    "hp_ckpt": str(explicit_hp_ckpt) if explicit_hp_ckpt else None,
                    "run_tag": args.run_tag,
                    "object_trans_arch": OBJECT_TRANS_ARCH,
                    "epochs": {
                        "human_pose": args.hp_epochs,
                        "velocity_contact": args.vc_epochs,
                        "object_trans": args.ot_epochs,
                        "joint_train": int(base_cfg.get("joint_train_epochs", 0)),
                    },
                    "max_parallel": args.max_parallel,
                    "pretrained_modules": shared_cfg["pretrained_modules"],
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
    shared_cfg_path = root / "shared_config.yaml"
    write_yaml(shared_cfg_path, shared_cfg)
    manifest_path = root / "experiment_manifest.json"
    root_log = root / "orchestrator.log"
    lock = threading.Lock()
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.resume and manifest_path.is_file()
        else {
            "status": "running",
            "phase": "human_pose",
            "started_at": now(),
            "source_config": str(cfg_path),
            "shared_config": str(shared_cfg_path),
            "dataset": "omomo",
            "seed": args.seed,
            "simulate_imu_noise": True,
            "simulate_imu_noise_seed": args.seed,
            "object_trans_arch": OBJECT_TRANS_ARCH,
            "gpu": args.gpu,
            "max_parallel": args.max_parallel,
            "requested_pipelines": [name for name, _ in selected_pipelines],
            "python": str(SAGE_PYTHON),
            "pipelines": {},
        }
    )
    write_json(manifest_path, manifest)

    env = os.environ.copy()
    env.update(
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

    def gpu_snapshot(label: str) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        log(f"GPU {label}: {result.stdout.strip()}")

    def run_stage(command: list[str], log_path: Path) -> None:
        gpu_snapshot("before " + Path(command[1]).stem)
        message = "RUN " + " ".join(command)
        log(message)
        print(f"[{now()}] {message}", flush=True)
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        log(f"END returncode={result.returncode} log={log_path}")
        print(f"[{now()}] END returncode={result.returncode} log={log_path}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); see {log_path}")

    hp_suffix = f"omomo_rnn2_shared_hp_{args.run_tag}"
    hp_ckpt = explicit_hp_ckpt
    manifest_hp = manifest.get("hp_ckpt")
    if isinstance(manifest_hp, str) and Path(manifest_hp).is_file():
        hp_ckpt = Path(manifest_hp)
    if hp_ckpt is None:
        hp_ckpt = find_best(root / "hp", "human_pose", hp_suffix)
    if hp_ckpt is None:
        hp_command = [
            str(SAGE_PYTHON),
            str(PROJECT_ROOT / "train/rnn/train_human_pose.py"),
            "--cfg",
            str(shared_cfg_path),
            "--save_dir",
            str(root / "hp"),
            "--run_suffix",
            hp_suffix,
            "--seed",
            str(args.seed),
            "--train_datasets",
            "omomo",
            "--num_workers",
            str(args.num_workers),
            "--simulate_imu_noise",
            "--epochs",
            str(args.hp_epochs),
        ]
        manifest["hp_stage"] = {
            "status": "running",
            "command": hp_command,
            "log": str(root / "human_pose.log"),
        }
        write_json(manifest_path, manifest)
        try:
            run_stage(hp_command, root / "human_pose.log")
            hp_ckpt = find_best(root / "hp", "human_pose", hp_suffix)
            if hp_ckpt is None:
                raise FileNotFoundError("HumanPose training produced no best.pt")
            manifest["hp_stage"].update(
                {"status": "complete", "best_ckpt": str(hp_ckpt), "finished_at": now()}
            )
        except Exception as exc:
            manifest["hp_stage"].update(
                {"status": "failed", "error": str(exc), "finished_at": now()}
            )
            manifest.update(
                {"status": "failed", "phase": "human_pose_failed", "error": str(exc)}
            )
            write_json(manifest_path, manifest)
            return 1
    manifest["hp_ckpt"] = str(hp_ckpt)
    manifest["phase"] = "pipeline_training"
    write_json(manifest_path, manifest)

    def pipeline(name: str, vc_arch: str) -> bool:
        pipeline_root = root / name
        pipeline_root.mkdir(parents=True, exist_ok=True)
        suffix = f"omomo_rnn2_{name}_ot_v2_{args.run_tag}"
        config = copy.deepcopy(shared_cfg)
        config.update(
            {
                "save_dir": str(pipeline_root),
                "velocity_contact_arch": vc_arch,
                "object_trans_arch": OBJECT_TRANS_ARCH,
            }
        )
        variant_cfg = pipeline_root / "config.yaml"
        write_yaml(variant_cfg, config)
        update_pipeline(
            name,
            pipeline=name,
            velocity_contact_arch=vc_arch,
            object_trans_arch=OBJECT_TRANS_ARCH,
            hp_ckpt=str(hp_ckpt),
            config=str(variant_cfg),
            status="running",
            phase="velocity_contact",
        )
        try:
            vc_ckpt = find_best(
                pipeline_root,
                f"velocity_contact_{vc_arch}",
                suffix,
            )
            if vc_ckpt is None:
                vc_command = [
                    str(SAGE_PYTHON),
                    str(PROJECT_ROOT / "train/rnn/train_velocity_contact.py"),
                    "--cfg",
                    str(variant_cfg),
                    "--save_dir",
                    str(pipeline_root),
                    "--run_suffix",
                    suffix,
                    "--seed",
                    str(args.seed),
                    "--train_datasets",
                    "omomo",
                    "--num_workers",
                    str(args.num_workers),
                    "--simulate_imu_noise",
                    "--epochs",
                    str(args.vc_epochs),
                    "--velocity_contact_arch",
                    vc_arch,
                    "--hp_ckpt",
                    str(hp_ckpt),
                ]
                run_stage(vc_command, pipeline_root / "velocity_contact.log")
                vc_ckpt = find_best(
                    pipeline_root,
                    f"velocity_contact_{vc_arch}",
                    suffix,
                )
                if vc_ckpt is None:
                    raise FileNotFoundError(f"{name} VC training produced no best.pt")
            update_pipeline(
                name,
                phase="object_trans_joint_train",
                vc_ckpt=str(vc_ckpt),
            )

            joint = find_joint_checkpoints(pipeline_root, suffix)
            if joint is None:
                ot_command = [
                    str(SAGE_PYTHON),
                    str(PROJECT_ROOT / "train/rnn/train_object_trans.py"),
                    "--cfg",
                    str(variant_cfg),
                    "--save_dir",
                    str(pipeline_root),
                    "--run_suffix",
                    suffix,
                    "--seed",
                    str(args.seed),
                    "--train_datasets",
                    "omomo",
                    "--num_workers",
                    str(args.num_workers),
                    "--simulate_imu_noise",
                    "--epochs",
                    str(args.ot_epochs),
                    "--joint_train",
                    "--velocity_contact_arch",
                    vc_arch,
                    "--object_trans_arch",
                    OBJECT_TRANS_ARCH,
                    "--hp_ckpt",
                    str(hp_ckpt),
                    "--vc_ckpt",
                    str(vc_ckpt),
                ]
                run_stage(ot_command, pipeline_root / "object_trans_joint_train.log")
                joint = find_joint_checkpoints(pipeline_root, suffix)
                if joint is None:
                    raise FileNotFoundError(
                        f"{name} OT/joint training produced no joint checkpoints"
                    )
            joint_vc, joint_ot = joint
            update_pipeline(
                name,
                phase="eval",
                joint_vc_ckpt=str(joint_vc),
                joint_ot_ckpt=str(joint_ot),
            )

            eval_json = pipeline_root / "eval_batch_online.json"
            if not eval_json.is_file():
                eval_command = [
                    str(SAGE_PYTHON),
                    str(PROJECT_ROOT / "eval_batch_online.py"),
                    "--config",
                    str(variant_cfg),
                    "--dataset",
                    "omomo",
                    "--hp_ckpt",
                    str(hp_ckpt),
                    "--vc_ckpt",
                    str(joint_vc),
                    "--ot_ckpt",
                    str(joint_ot),
                    "--velocity_contact_arch",
                    vc_arch,
                    "--object_trans_arch",
                    OBJECT_TRANS_ARCH,
                    "--simulate_imu_noise",
                    "--simulate_imu_noise_seed",
                    str(args.seed),
                    "--seed",
                    str(args.seed),
                    "--num_workers",
                    str(args.num_workers),
                    "--online_batch_size",
                    str(args.eval_batch_size),
                    "--output_json",
                    str(eval_json),
                ]
                run_stage(eval_command, pipeline_root / "eval_batch_online.log")
            metrics = read_metrics(eval_json)
            update_pipeline(
                name,
                status="complete",
                phase="complete",
                eval_json=str(eval_json),
                metrics=metrics,
                finished_at=now(),
            )
            return True
        except Exception as exc:
            update_pipeline(
                name,
                status="failed",
                phase="failed",
                error=str(exc),
                finished_at=now(),
            )
            log(f"PIPELINE FAILED {name}: {exc}")
            return False

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(pipeline, name, arch): name
            for name, arch in selected_pipelines
        }
        results = {
            futures[future]: bool(future.result()) for future in as_completed(futures)
        }
    manifest["pipeline_results"] = results
    manifest["finished_at"] = now()
    manifest["status"] = "complete" if all(results.values()) else "failed"
    manifest["phase"] = (
        "complete" if manifest["status"] == "complete" else "pipeline_failed"
    )
    write_json(manifest_path, manifest)
    if manifest["status"] == "complete" and {
        name for name, _ in selected_pipelines
    } == set(DEFAULT_PIPELINES):
        write_summary(root, manifest)
    log(f"FINISHED status={manifest['status']}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())