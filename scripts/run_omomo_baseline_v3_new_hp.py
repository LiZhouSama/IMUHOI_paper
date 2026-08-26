#!/usr/bin/env python3
"""Train one fresh OMOMO HP and compare baseline/v3 VC-OT joint pipelines."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGE_PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
PIPELINES = (("baseline", "imuhoi"), ("v3", "v3"))
METRICS = (
    ("mpjpe", "MPJPE (cm)"),
    ("mpjre_angle", "MPJRE (deg)"),
    ("root_trans_err", "Root (cm)"),
    ("jitter", "Jitter (mm/frame^2)"),
    ("obj_trans_err_fusion", "Object Fusion (cm)"),
    ("hoi_err_fusion", "HOI Fusion (cm)"),
    ("contact_f1_lhand", "Left F1"),
    ("contact_f1_rhand", "Right F1"),
    ("contact_f1_obj", "Object F1"),
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def append_log(path: Path, message: str, lock: Lock) -> None:
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def find_best(root: Path, prefix: str, suffix: str, filename: str = "best.pt") -> Path:
    prefix = prefix.lower()
    matches = [
        item for item in root.iterdir()
        if item.is_dir() and item.name.lower().startswith(prefix) and suffix in item.name
        and (item / filename).is_file()
    ]
    if not matches:
        raise FileNotFoundError(f"No {prefix} stage with {filename} under {root}")
    return max(matches, key=lambda item: item.stat().st_mtime) / filename


def eval_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("results", {}).get("omomo")
    if result is None:
        result = payload.get("omomo", payload)
    return {key: result.get(key) for key, _ in METRICS}


def finite(value) -> str:
    if value is None:
        return "NaN"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NaN"
    return f"{value:.4f}" if math.isfinite(value) else "NaN"


def write_summary(root: Path, manifest: dict) -> None:
    rows = []
    for name, arch in PIPELINES:
        item = manifest["pipelines"].get(name, {})
        if item.get("status") == "complete":
            rows.append({"pipeline": name, "velocity_contact_arch": arch,
                         "metrics": item["metrics"], "eval_json": item["eval_json"],
                         "joint_vc_ckpt": item["joint_vc_ckpt"],
                         "joint_ot_ckpt": item["joint_ot_ckpt"]})
    write_json(root / "omomo_baseline_v3_eval.json", {
        "generated_at": now(), "dataset": "omomo", "full_sequence": True,
        "hp_ckpt": manifest.get("hp_ckpt"), "seed": manifest["seed"],
        "simulate_imu_noise": True, "simulate_imu_noise_seed": manifest["seed"],
        "object_trans_arch": "current", "rows": rows,
    })
    lines = [
        "# OMOMO baseline/v3 comparison with a fresh shared HP", "",
        "Both rows retrain VC and OT, run OT joint training, and evaluate full OMOMO sequences with simulated IMU noise.", "",
        "| Pipeline | Architecture | " + " | ".join(label for _, label in METRICS) + " |",
        "|---|---|" + "---:|" * len(METRICS),
    ]
    for row in rows:
        values = " | ".join(finite(row["metrics"].get(key)) for key, _ in METRICS)
        lines.append(f"| {row['pipeline']} | `{row['velocity_contact_arch']}` | {values} |")
    (root / "omomo_baseline_v3_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml")
    parser.add_argument("--experiment_root", type=Path, default=PROJECT_ROOT / "outputs/IMUHOI++/RNN/omomo_baseline_v3_new_hp_20260823")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hp_epochs", type=int, default=300)
    parser.add_argument("--vc_epochs", type=int, default=200)
    parser.add_argument("--ot_epochs", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg_path = (args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg).resolve()
    root = (args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root).resolve()
    if not SAGE_PYTHON.is_file():
        raise FileNotFoundError(f"Missing WSL SAGE interpreter: {SAGE_PYTHON}")
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if args.num_workers != 4:
        raise ValueError("Use num_workers=4 for this controlled experiment.")
    if not 1 <= args.max_parallel <= 2:
        raise ValueError("Use max_parallel in [1, 2] on the 16 GB GPU.")

    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if list(base_cfg.get("train_datasets", [])) != ["omomo"]:
        raise ValueError("The config must select only train_datasets: [omomo].")
    if root.exists() and not args.resume:
        raise FileExistsError(f"Experiment root already exists; choose another root or use --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path, root_log = root / "experiment_manifest.json", root / "orchestrator.log"
    lock = Lock()
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONUNBUFFERED": "1",
        "TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp",
        "IMUHOI_PERSISTENT_WORKERS": "0", "IMUHOI_SKIP_DATALOADER_PRIME": "1",
    })
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if args.resume and manifest_path.is_file() else {
        "status": "running", "phase": "human_pose", "started_at": now(),
        "config": str(cfg_path), "dataset": "omomo", "seed": args.seed,
        "simulate_imu_noise": True, "object_trans_arch": "current",
        "num_workers": args.num_workers, "max_parallel": args.max_parallel,
        "gpu": args.gpu, "python": str(SAGE_PYTHON), "pipelines": {},
    }
    write_json(manifest_path, manifest)

    def gpu_snapshot(label: str) -> None:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], cwd=PROJECT_ROOT, env=env, check=True, capture_output=True, text=True)
        append_log(root_log, f"GPU {label}: {result.stdout.strip()}", lock)

    def run_stage(command: list[str], log_path: Path) -> None:
        gpu_snapshot("before " + Path(command[1]).stem)
        append_log(root_log, "RUN " + " ".join(map(str, command)), lock)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        append_log(root_log, f"END returncode={result.returncode} log={log_path}", lock)
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); see {log_path}")

    hp_ckpt = Path(manifest["hp_ckpt"]) if manifest.get("hp_ckpt") else None
    hp_suffix = "omomo_newhp_20260823"
    if hp_ckpt is None or not hp_ckpt.is_file():
        hp_cmd = [str(SAGE_PYTHON), str(PROJECT_ROOT / "train/rnn/train_human_pose.py"),
                  "--cfg", str(cfg_path), "--save_dir", str(root / "hp"), "--run_suffix", hp_suffix,
                  "--seed", str(args.seed), "--train_datasets", "omomo", "--num_workers", str(args.num_workers),
                  "--simulate_imu_noise", "--epochs", str(args.hp_epochs)]
        manifest["hp_stage"] = {"status": "running", "command": hp_cmd, "log": str(root / "human_pose.log")}
        write_json(manifest_path, manifest)
        try:
            run_stage(hp_cmd, root / "human_pose.log")
            hp_ckpt = find_best(root / "hp", "human_pose", hp_suffix)
            manifest["hp_ckpt"] = str(hp_ckpt)
            manifest["hp_stage"].update({"status": "complete", "best_ckpt": str(hp_ckpt), "finished_at": now()})
            write_json(manifest_path, manifest)
        except Exception as exc:
            manifest.update({"status": "failed", "phase": "human_pose_failed", "error": str(exc)})
            manifest["hp_stage"].update({"status": "failed", "error": str(exc), "finished_at": now()})
            write_json(manifest_path, manifest)
            return 1
    else:
        manifest.setdefault("hp_stage", {}).update({"status": "complete", "best_ckpt": str(hp_ckpt)})
        manifest["hp_ckpt"] = str(hp_ckpt)
        write_json(manifest_path, manifest)

    def pipeline(name: str, arch: str) -> bool:
        pipeline_root = root / name
        pipeline_root.mkdir(parents=True, exist_ok=True)
        suffix = f"omomo_{name}_newhp_20260823"
        with lock:
            item = manifest["pipelines"].setdefault(name, {"pipeline": name, "arch": arch, "status": "running"})
            item.update({"status": "running", "phase": "velocity_contact", "hp_ckpt": str(hp_ckpt)})
            write_json(manifest_path, manifest)
        cfg = copy.deepcopy(base_cfg)
        cfg.update({"save_dir": str(pipeline_root), "train_datasets": ["omomo"],
                    "velocity_contact_arch": arch, "object_trans_arch": "current",
                    "simulate_imu_noise": True, "simulate_imu_noise_seed": args.seed})
        cfg_path_variant = pipeline_root / "config.yaml"
        write_yaml(cfg_path_variant, cfg)
        try:
            vc_cmd = [str(SAGE_PYTHON), str(PROJECT_ROOT / "train/rnn/train_velocity_contact.py"),
                      "--cfg", str(cfg_path_variant), "--save_dir", str(pipeline_root), "--run_suffix", suffix,
                      "--seed", str(args.seed), "--train_datasets", "omomo", "--num_workers", str(args.num_workers),
                      "--simulate_imu_noise", "--epochs", str(args.vc_epochs), "--velocity_contact_arch", arch,
                      "--hp_ckpt", str(hp_ckpt)]
            run_stage(vc_cmd, pipeline_root / "velocity_contact.log")
            vc_ckpt = find_best(pipeline_root, f"velocity_contact_{arch}", suffix)
            with lock:
                item.update({"phase": "object_trans_joint_train", "vc_ckpt": str(vc_ckpt)})
                write_json(manifest_path, manifest)
            ot_cmd = [str(SAGE_PYTHON), str(PROJECT_ROOT / "train/rnn/train_object_trans.py"),
                      "--cfg", str(cfg_path_variant), "--save_dir", str(pipeline_root), "--run_suffix", suffix,
                      "--seed", str(args.seed), "--train_datasets", "omomo", "--num_workers", str(args.num_workers),
                      "--simulate_imu_noise", "--epochs", str(args.ot_epochs), "--joint_train",
                      "--velocity_contact_arch", arch, "--object_trans_arch", "current",
                      "--hp_ckpt", str(hp_ckpt), "--vc_ckpt", str(vc_ckpt)]
            run_stage(ot_cmd, pipeline_root / "object_trans_joint_train.log")
            ot_dir = max((p for p in pipeline_root.iterdir() if p.is_dir() and p.name.startswith("object_trans")
                          and suffix in p.name and (p / "best_velocity_contact.pt").is_file()
                          and (p / "best_object_trans.pt").is_file()), key=lambda p: p.stat().st_mtime)
            joint_vc, joint_ot = ot_dir / "best_velocity_contact.pt", ot_dir / "best_object_trans.pt"
            with lock:
                item.update({"phase": "eval", "joint_vc_ckpt": str(joint_vc), "joint_ot_ckpt": str(joint_ot)})
                write_json(manifest_path, manifest)
            eval_json = pipeline_root / "eval_batch_online.json"
            eval_cmd = [str(SAGE_PYTHON), str(PROJECT_ROOT / "eval_batch_online.py"), "--config", str(cfg_path_variant),
                        "--dataset", "omomo", "--hp_ckpt", str(hp_ckpt), "--vc_ckpt", str(joint_vc),
                        "--ot_ckpt", str(joint_ot), "--velocity_contact_arch", arch, "--object_trans_arch", "current",
                        "--simulate_imu_noise", "--simulate_imu_noise_seed", str(args.seed), "--seed", str(args.seed),
                        "--num_workers", str(args.num_workers), "--output_json", str(eval_json)]
            run_stage(eval_cmd, pipeline_root / "eval_batch_online.log")
            with lock:
                item.update({"status": "complete", "phase": "complete", "eval_json": str(eval_json),
                             "metrics": eval_metrics(eval_json), "finished_at": now()})
                write_json(manifest_path, manifest)
            return True
        except Exception as exc:
            with lock:
                item.update({"status": "failed", "phase": "failed", "error": str(exc), "finished_at": now()})
                write_json(manifest_path, manifest)
            append_log(root_log, f"FAIL {name}: {exc}", lock)
            return False

    manifest["phase"] = "parallel_baseline_v3"
    write_json(manifest_path, manifest)
    gpu_snapshot("before parallel baseline/v3")
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {executor.submit(pipeline, name, arch): name for name, arch in PIPELINES}
        results = {futures[future]: bool(future.result()) for future in as_completed(futures)}
    manifest["pipeline_results"] = results
    manifest["status"] = "complete" if all(results.values()) else "failed"
    manifest["phase"] = "complete" if manifest["status"] == "complete" else "pipeline_failed"
    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)
    if manifest["status"] == "complete":
        write_summary(root, manifest)
    append_log(root_log, f"FINISHED status={manifest['status']}", lock)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
