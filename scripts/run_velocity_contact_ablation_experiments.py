"""Train and evaluate the controlled IMUHOI VelocityContact ablations.

The fixed HumanPose checkpoint is reused for every run.  Each ablation trains
VelocityContact from scratch, trains the current ObjectTrans architecture from
scratch, runs the configured automatic joint-training phase, and evaluates the
joint VC/OT checkpoints with sequence-level simulated IMU noise.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
DEFAULT_CONFIG = PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml"
DEFAULT_HP_CKPT = PROJECT_ROOT / (
    "outputs/IMUHOI_RNN_2/rnn_sequential_seq_20260728_214819/"
    "human_pose_07282148_seq_20260728_214819/best.pt"
)

EXPERIMENTS = (
    ("01_cond_causal", "imuhoi_cond"),
    ("02_cond_objmove_bi", "imuhoi_cond_bi"),
    ("03_cond_geom", "imuhoi_cond_geom"),
    ("04_cond_boundary", "imuhoi_cond_boundary"),
    ("05_cond_boundary_endpoint_align", "imuhoi_cond_boundary_endpoint"),
    ("06_v2_velocity_cascade", "v2"),
    ("07_v3_boundary_objmove_bi", "v3"),
    ("08_v4_supervised_pose_prior", "v4"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hp_ckpt", type=Path, default=DEFAULT_HP_CKPT)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Experiment root; defaults to outputs/IMUHOI_RNN_2/vc_ablation_<time>.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[name for name, _ in EXPERIMENTS],
        default=None,
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--eval_online_lookahead",
        type=int,
        default=None,
        help="Explicit online lookahead for the post-training evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Persistent DataLoader workers per training process.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)


def write_yaml(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)


def run_command(command, log_path, env, dry_run=False):
    printable = " ".join(str(part) for part in command)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    gpu_status = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"GPU before Python: {gpu_status.stdout.strip()}", flush=True)
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def find_single(pattern: str) -> Path:
    matches = sorted(PROJECT_ROOT.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No output matched: {pattern}")
    return matches[-1]


def extract_metrics(eval_json: Path):
    with eval_json.open() as handle:
        payload = json.load(handle)
    # A single dataset stores metrics below its dataset key.  Keep this helper
    # tolerant of the flat historical output format as well.
    if "omomo" in payload and isinstance(payload["omomo"], dict):
        return payload["omomo"]
    if "results" in payload and isinstance(payload["results"], dict):
        results = payload["results"]
        if "omomo" in results:
            return results["omomo"]
    return payload


def main():
    args = parse_args()
    config_path = resolve(args.config)
    hp_ckpt = resolve(args.hp_ckpt)
    if not PYTHON.is_file():
        raise FileNotFoundError(f"Required SAGE Python not found: {PYTHON}")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not hp_ckpt.is_file():
        raise FileNotFoundError(hp_ckpt)

    if args.output_root is None:
        output_root = PROJECT_ROOT / "outputs/IMUHOI_RNN_2" / (
            "vc_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    else:
        output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with config_path.open() as handle:
        base_config = yaml.safe_load(handle) or {}
    selected = [
        item for item in EXPERIMENTS
        if args.only is None or item[0] in set(args.only)
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Python multiprocessing resource_sharer requires Unix-domain sockets.
    # A temporary directory on a Windows /mnt/* mount returns ENOTSUP in WSL.
    env["TMPDIR"] = "/tmp"
    env["TMP"] = "/tmp"
    env["TEMP"] = "/tmp"
    manifest_path = output_root / "manifest.json"
    manifest = {
        "base_config": str(config_path),
        "hp_ckpt": str(hp_ckpt),
        "gpu": args.gpu,
        "experiments": {},
    }
    write_json(manifest_path, manifest)

    for experiment_name, arch in selected:
        experiment_dir = output_root / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        derived_config = dict(base_config)
        derived_config["save_dir"] = str(experiment_dir)
        derived_config["velocity_contact_arch"] = arch
        derived_config["object_trans_arch"] = "current"
        derived_config["simulate_imu_noise"] = True
        derived_config["boundary_bidirectional"] = False
        derived_config["contact_from_segments"] = False
        loss_weights = dict(derived_config.get("loss_weights", {}))
        # Experiment 5 adds endpoint heatmap supervision and the requested
        # frame/endpoint alignment only.  Exclude endpoint-order regularization
        # so this row does not silently introduce a third intervention.
        loss_weights["contact_endpoint_order"] = 0.0
        derived_config["loss_weights"] = loss_weights
        variant_config_path = experiment_dir / "config.yaml"
        write_yaml(variant_config_path, derived_config)

        run_suffix = experiment_name
        record = {
            "arch": arch,
            "config": str(variant_config_path),
            "status": "running_vc",
        }
        manifest["experiments"][experiment_name] = record
        write_json(manifest_path, manifest)

        vc_command = [
            PYTHON,
            PROJECT_ROOT / "train/rnn/train_velocity_contact.py",
            "--cfg",
            variant_config_path,
            "--hp_ckpt",
            hp_ckpt,
            "--velocity_contact_arch",
            arch,
            "--simulate_imu_noise",
            "--num_workers",
            str(args.num_workers),
            "--run_suffix",
            run_suffix,
        ]
        run_command(
            vc_command,
            experiment_dir / "train_velocity_contact.log",
            env,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        vc_best = find_single(
            str(
                experiment_dir.relative_to(PROJECT_ROOT)
                / f"velocity_contact_{arch}_*_{run_suffix}/best.pt"
            )
        )
        record.update({"status": "running_ot", "vc_best": str(vc_best)})
        write_json(manifest_path, manifest)

        ot_command = [
            PYTHON,
            PROJECT_ROOT / "train/rnn/train_object_trans.py",
            "--cfg",
            variant_config_path,
            "--hp_ckpt",
            hp_ckpt,
            "--vc_ckpt",
            vc_best,
            "--velocity_contact_arch",
            arch,
            "--object_trans_arch",
            "current",
            "--joint_train",
            "--simulate_imu_noise",
            "--num_workers",
            str(args.num_workers),
            "--run_suffix",
            run_suffix,
        ]
        run_command(
            ot_command,
            experiment_dir / "train_object_trans_joint.log",
            env,
        )
        ot_dir = find_single(
            str(
                experiment_dir.relative_to(PROJECT_ROOT)
                / f"object_trans_vc_{arch}_*_{run_suffix}"
            )
        )
        joint_vc = ot_dir / "best_velocity_contact.pt"
        joint_ot = ot_dir / "best_object_trans.pt"
        for checkpoint in (joint_vc, joint_ot):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
        record.update(
            {
                "status": "running_eval",
                "joint_vc_ckpt": str(joint_vc),
                "joint_ot_ckpt": str(joint_ot),
            }
        )
        write_json(manifest_path, manifest)

        eval_json = experiment_dir / "eval_batch_online.json"
        eval_command = [
            PYTHON,
            PROJECT_ROOT / "eval_batch_online.py",
            "--config",
            variant_config_path,
            "--hp_ckpt",
            hp_ckpt,
            "--vc_ckpt",
            joint_vc,
            "--ot_ckpt",
            joint_ot,
            "--simulate_imu_noise",
            "--object_trans_arch",
            "current",
            "--velocity_contact_arch",
            arch,
            "--num_workers",
            str(args.num_workers),
            "--output_json",
            eval_json,
        ]
        if args.eval_online_lookahead is not None:
            eval_command.extend(
                ["--online_lookahead", str(args.eval_online_lookahead)]
            )
        run_command(
            eval_command,
            experiment_dir / "eval_batch_online.log",
            env,
        )
        record.update(
            {
                "status": "complete",
                "eval_json": str(eval_json),
                "metrics": extract_metrics(eval_json),
            }
        )
        write_json(manifest_path, manifest)

    print(f"\nExperiment manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
