#!/usr/bin/env python3
"""Resume only baseline OT/joint/eval after its completed VC stage."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from run_omomo_vc_ablation_unified_hp import (
    EXPERIMENTS,
    PROJECT_ROOT,
    PYTHON,
    find_stage,
    read_metrics,
    write_json,
    write_summary,
)


DEFAULT_ROOT = (
    PROJECT_ROOT / "outputs/IMUHOI++/RNN/omomo_vc_ablation_unified_hp_20260821"
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.num_workers != 4:
        raise ValueError("This controlled run requires num_workers=4.")

    root = (
        args.experiment_root
        if args.experiment_root.is_absolute()
        else PROJECT_ROOT / args.experiment_root
    ).resolve()
    manifest_path = root / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = root / "baseline_imuhoi_direct"
    config = base_dir / "config.yaml"
    hp_ckpt = Path(manifest["hp_ckpt"])
    suffix = "omomo_ablation_baseline_20260821"
    vc_dir = find_stage(base_dir, "velocity_contact_imuhoi_", suffix, "best.pt")
    vc_ckpt = vc_dir / "best.pt"
    for path in (PYTHON, config, hp_ckpt, vc_ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": "/tmp",
        "TMP": "/tmp",
        "TEMP": "/tmp",
    })
    resume_log = root / "baseline_resume_orchestrator.log"

    def log(message: str) -> None:
        with resume_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")

    def commit(**fields) -> None:
        item = manifest["experiments"]["baseline_imuhoi_direct"]
        item.update(fields)
        item.pop("error", None)
        write_json(manifest_path, manifest)

    def run(command: list[str], log_path: Path) -> None:
        log("RUN " + " ".join(command))
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env.copy(),
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        log(f"END returncode={result.returncode} log={log_path}")
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); see {log_path}")

    manifest["status"] = "running"
    manifest["phase"] = "resuming_baseline"
    commit(
        status="running",
        phase="object_trans_joint_train",
        vc_ckpt=str(vc_ckpt),
        resumed_at=now(),
    )
    try:
        ot_command = [
            str(PYTHON), str(PROJECT_ROOT / "train/rnn/train_object_trans.py"),
            "--cfg", str(config),
            "--hp_ckpt", str(hp_ckpt),
            "--vc_ckpt", str(vc_ckpt),
            "--velocity_contact_arch", "imuhoi",
            "--object_trans_arch", "current",
            "--joint_train",
            "--simulate_imu_noise",
            "--num_workers", str(args.num_workers),
            "--run_suffix", suffix,
        ]
        run(ot_command, base_dir / "train_object_trans_joint.log")
        ot_dir = find_stage(
            base_dir,
            "object_trans_vc_imuhoi_",
            suffix,
            "best_velocity_contact.pt",
        )
        joint_vc = ot_dir / "best_velocity_contact.pt"
        joint_ot = ot_dir / "best_object_trans.pt"
        if not joint_ot.is_file():
            raise FileNotFoundError(joint_ot)
        commit(
            phase="eval",
            ot_command=ot_command,
            joint_vc_ckpt=str(joint_vc),
            joint_ot_ckpt=str(joint_ot),
        )

        eval_json = base_dir / "eval_batch_online.json"
        eval_command = [
            str(PYTHON), str(PROJECT_ROOT / "eval_batch_online.py"),
            "--config", str(config),
            "--dataset", "omomo",
            "--hp_ckpt", str(hp_ckpt),
            "--vc_ckpt", str(joint_vc),
            "--ot_ckpt", str(joint_ot),
            "--velocity_contact_arch", "imuhoi",
            "--object_trans_arch", "current",
            "--simulate_imu_noise",
            "--simulate_imu_noise_seed", str(manifest["simulate_imu_noise_seed"]),
            "--seed", str(manifest["seed"]),
            "--num_workers", str(args.num_workers),
            "--output_json", str(eval_json),
        ]
        run(eval_command, base_dir / "eval_batch_online.log")
        commit(
            status="complete",
            phase="complete",
            eval_command=eval_command,
            eval_json=str(eval_json),
            metrics=read_metrics(eval_json),
            finished_at=now(),
        )
        all_complete = all(
            manifest["experiments"].get(name, {}).get("status") == "complete"
            for name, _, _ in EXPERIMENTS
        )
        manifest["pipeline_results"] = {
            name: manifest["experiments"].get(name, {}).get("status") == "complete"
            for name, _, _ in EXPERIMENTS
        }
        manifest["status"] = "complete" if all_complete else "failed"
        manifest["phase"] = "complete" if all_complete else "one_or_more_pipelines_failed"
        manifest["finished_at"] = now()
        if all_complete:
            write_summary(root, manifest)
        write_json(manifest_path, manifest)
        log(f"FINISHED status={manifest['status']}")
        return 0 if all_complete else 1
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["phase"] = "baseline_resume_failed"
        commit(status="failed", phase="failed", error=str(exc), finished_at=now())
        log(f"FAILED {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
