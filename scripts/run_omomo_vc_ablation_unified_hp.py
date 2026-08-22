#!/usr/bin/env python3
"""Run four controlled OMOMO VC ablations with one fixed HP checkpoint."""
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
PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
DEFAULT_CFG = PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml"
DEFAULT_HP = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN/per_dataset_baseline_v3_independent_hp_20260819/"
    "hp/omomo/human_pose_08191605_hp_omomo_parallel_independent_20260819/best.pt"
)
DEFAULT_ROOT = (
    PROJECT_ROOT / "outputs/IMUHOI++/RNN/omomo_vc_ablation_unified_hp_20260821"
)
EXPERIMENTS = (
    ("baseline_imuhoi_direct", "baseline", "imuhoi"),
    ("01_cond_causal", "01", "imuhoi_cond"),
    ("04_cond_boundary", "04", "imuhoi_cond_boundary"),
    ("07_v3_boundary_objmove_bi", "07", "v3"),
)
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


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_yaml(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def find_stage(root: Path, prefix: str, suffix: str, required: str) -> Path:
    matches = [
        p for p in root.iterdir()
        if p.is_dir() and p.name.lower().startswith(prefix.lower()) and suffix in p.name
        and (p / required).is_file()
    ]
    if not matches:
        raise FileNotFoundError(f"No completed {prefix} stage under {root}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def read_metrics(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value.get("results"), dict) and "omomo" in value["results"]:
        return value["results"]["omomo"]
    if isinstance(value.get("omomo"), dict):
        return value["omomo"]
    return value


def format_metric(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NaN"
    return f"{float(value):.4f}"


def write_summary(root: Path, manifest: dict) -> None:
    rows = []
    for directory, experiment_id, arch in EXPERIMENTS:
        item = manifest["experiments"][directory]
        if item.get("status") == "complete":
            rows.append({
                "id": experiment_id,
                "name": directory,
                "velocity_contact_arch": arch,
                "metrics": item["metrics"],
                "joint_vc_ckpt": item["joint_vc_ckpt"],
                "joint_ot_ckpt": item["joint_ot_ckpt"],
                "eval_json": item["eval_json"],
            })
    output = {
        "generated_at": now(),
        "dataset": "omomo",
        "full_sequence": True,
        "fixed_hp_ckpt": manifest["hp_ckpt"],
        "config_source": manifest["config_source"],
        "seed": manifest["seed"],
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": manifest["simulate_imu_noise_seed"],
        "object_trans_arch": "current",
        "num_workers": manifest["num_workers_per_pipeline"],
        "note": "Every VC/OT was retrained. ID 07 uses current v3 with the marginal-output fix.",
        "rows": rows,
    }
    write_json(root / "omomo_vc_ablation_eval.json", output)
    lines = [
        "# OMOMO VC ablation with unified HumanPose checkpoint", "",
        "All rows retrain VC and OT, run joint training, and evaluate joint checkpoints on OMOMO full sequences with simulated IMU noise.", "",
        "ID 07 uses current v3 with the hand-contact marginal supervision fix.", "",
        "| ID | Architecture | " + " | ".join(label for _, label in METRICS) + " |",
        "|---|---|" + "---:|" * len(METRICS),
    ]
    for row in rows:
        values = " | ".join(format_metric(row["metrics"].get(key)) for key, _ in METRICS)
        lines.append(f"| {row['id']} | `{row['velocity_contact_arch']}` | {values} |")
    lines.append("")
    (root / "omomo_vc_ablation_eval.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--hp_ckpt", type=Path, default=DEFAULT_HP)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_parallel", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = (args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg).resolve()
    hp_ckpt = (args.hp_ckpt if args.hp_ckpt.is_absolute() else PROJECT_ROOT / args.hp_ckpt).resolve()
    root = (args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root).resolve()
    for path in (PYTHON, cfg_path, hp_ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.num_workers != 4:
        raise ValueError("This controlled run requires num_workers=4.")
    if not 1 <= args.max_parallel <= 2:
        raise ValueError("Use at most two parallel pipelines on the 16 GB GPU.")

    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if list(base_cfg.get("train_datasets", [])) != ["omomo"]:
        raise ValueError("Current YAML must select only train_datasets: [omomo].")
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "experiment_manifest.json"
    root_log = root / "orchestrator.log"
    state_lock, log_lock = threading.Lock(), threading.Lock()
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONUNBUFFERED": "1",
        "TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp",
    })
    manifest = {
        "status": "running", "phase": "parallel_training", "started_at": now(),
        "config_source": str(cfg_path), "hp_ckpt": str(hp_ckpt), "dataset": "omomo",
        "full_sequence_eval": True, "seed": int(base_cfg.get("seed", 42)),
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": int(base_cfg.get("simulate_imu_noise_seed", 42)),
        "object_trans_arch": "current", "num_workers_per_pipeline": args.num_workers,
        "max_parallel": args.max_parallel, "gpu": args.gpu, "python": str(PYTHON),
        "experiments": {},
    }

    def log(message: str) -> None:
        with log_lock, root_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")

    def update(name: str, **fields) -> None:
        with state_lock:
            manifest["experiments"].setdefault(name, {}).update(fields)
            write_json(manifest_path, manifest)

    def run(command: list[str], log_path: Path) -> None:
        log("RUN " + " ".join(command))
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env.copy(), stdout=output, stderr=subprocess.STDOUT)
        log(f"END returncode={result.returncode} log={log_path}")
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); see {log_path}")

    def pipeline(directory: str, experiment_id: str, arch: str) -> bool:
        experiment_dir = root / directory
        experiment_dir.mkdir(parents=True, exist_ok=False)
        cfg = copy.deepcopy(base_cfg)
        cfg.update({
            "save_dir": str(experiment_dir), "train_datasets": ["omomo"],
            "velocity_contact_arch": arch, "object_trans_arch": "current",
            "simulate_imu_noise": True, "num_workers": args.num_workers,
        })
        variant_cfg = experiment_dir / "config.yaml"
        write_yaml(variant_cfg, cfg)
        suffix = f"omomo_ablation_{experiment_id}_20260821"
        update(directory, id=experiment_id, velocity_contact_arch=arch, status="running",
               phase="velocity_contact", config=str(variant_cfg), started_at=now())
        try:
            vc_cmd = [str(PYTHON), str(PROJECT_ROOT / "train/rnn/train_velocity_contact.py"),
                      "--cfg", str(variant_cfg), "--hp_ckpt", str(hp_ckpt),
                      "--velocity_contact_arch", arch, "--simulate_imu_noise",
                      "--num_workers", str(args.num_workers), "--run_suffix", suffix]
            run(vc_cmd, experiment_dir / "train_velocity_contact.log")
            vc_dir = find_stage(experiment_dir, f"velocity_contact_{arch}_", suffix, "best.pt")
            vc_ckpt = vc_dir / "best.pt"
            update(directory, phase="object_trans_joint_train", vc_ckpt=str(vc_ckpt), vc_command=vc_cmd)

            ot_cmd = [str(PYTHON), str(PROJECT_ROOT / "train/rnn/train_object_trans.py"),
                      "--cfg", str(variant_cfg), "--hp_ckpt", str(hp_ckpt),
                      "--vc_ckpt", str(vc_ckpt), "--velocity_contact_arch", arch,
                      "--object_trans_arch", "current", "--joint_train", "--simulate_imu_noise",
                      "--num_workers", str(args.num_workers), "--run_suffix", suffix]
            run(ot_cmd, experiment_dir / "train_object_trans_joint.log")
            ot_dir = find_stage(experiment_dir, f"object_trans_vc_{arch}_", suffix, "best_velocity_contact.pt")
            joint_vc, joint_ot = ot_dir / "best_velocity_contact.pt", ot_dir / "best_object_trans.pt"
            if not joint_ot.is_file():
                raise FileNotFoundError(joint_ot)
            update(directory, phase="eval", ot_command=ot_cmd,
                   joint_vc_ckpt=str(joint_vc), joint_ot_ckpt=str(joint_ot))

            eval_json = experiment_dir / "eval_batch_online.json"
            eval_cmd = [str(PYTHON), str(PROJECT_ROOT / "eval_batch_online.py"),
                        "--config", str(variant_cfg), "--dataset", "omomo",
                        "--hp_ckpt", str(hp_ckpt), "--vc_ckpt", str(joint_vc),
                        "--ot_ckpt", str(joint_ot), "--velocity_contact_arch", arch,
                        "--object_trans_arch", "current", "--simulate_imu_noise",
                        "--simulate_imu_noise_seed", str(manifest["simulate_imu_noise_seed"]),
                        "--seed", str(manifest["seed"]), "--num_workers", str(args.num_workers),
                        "--output_json", str(eval_json)]
            run(eval_cmd, experiment_dir / "eval_batch_online.log")
            update(directory, status="complete", phase="complete", eval_command=eval_cmd,
                   eval_json=str(eval_json), metrics=read_metrics(eval_json), finished_at=now())
            log(f"PIPELINE COMPLETE {directory}")
            return True
        except Exception as exc:
            update(directory, status="failed", phase="failed", error=str(exc), finished_at=now())
            log(f"PIPELINE FAILED {directory}: {exc}")
            return False

    write_json(manifest_path, manifest)
    log(f"START root={root} hp={hp_ckpt} max_parallel={args.max_parallel}")
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {executor.submit(pipeline, *item): item[0] for item in EXPERIMENTS}
        results = {futures[f]: bool(f.result()) for f in as_completed(futures)}
    with state_lock:
        manifest["pipeline_results"], manifest["finished_at"] = results, now()
        if all(results.get(name, False) for name, _, _ in EXPERIMENTS):
            manifest["status"], manifest["phase"] = "complete", "complete"
            write_summary(root, manifest)
        else:
            manifest["status"], manifest["phase"] = "failed", "one_or_more_pipelines_failed"
        write_json(manifest_path, manifest)
    log(f"FINISHED status={manifest['status']}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
