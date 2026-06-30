"""
Run the RNN training stages in order and wire each best checkpoint forward.

Stages:
1. train_human_pose.py --epochs 300
2. train_velocity_contact.py --epochs 200 --hp_ckpt <stage1 best>
3. train_object_trans.py --epochs 400 --batch_size 40 --hp_ckpt <stage1 best> --vc_ckpt <stage2 best>
4. train_object_trans.py --joint_train --lr 0.0002 --epochs 100 --batch_size 30
   --hp_ckpt <stage1 best> --vc_ckpt <stage2 best> --ot_ckpt <stage3 best>
"""
from __future__ import annotations

import argparse
import errno
import glob
import json
import os
import pty
import re
import select
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = "configs/IMUHOI_train_rnn.yaml"


@dataclass
class StageSpec:
    name: str
    script: str
    module_name: str
    fixed_args: list[str]
    dependency_args: list[str] = field(default_factory=list)


@dataclass
class StageResult:
    name: str
    command: list[str]
    log_path: str
    save_dir: str | None = None
    best_ckpt: str | None = None
    returncode: int | None = None
    started_at: str | None = None
    finished_at: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially train RNN HP, VC, OT, then OT joint fine-tuning."
    )
    parser.add_argument("--cfg", default=DEFAULT_CFG, help="Training config file.")
    parser.add_argument(
        "--pipeline_tag",
        default=None,
        help="Stable suffix appended to all stage output dirs. Defaults to seq_<timestamp>.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Forward a seed to every stage.")
    parser.add_argument("--debug", action="store_true", help="Forward --debug to every stage.")
    parser.add_argument("--no_trans", action="store_true", help="Forward --no_trans to every stage.")
    parser.add_argument(
        "--model_arch",
        choices=["rnn", "dit", "mamba_simple"],
        default=None,
        help="Forward --model_arch to every stage.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print/write the planned commands without launching training.",
    )
    return parser.parse_args()


def safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    tag = tag.strip("._-")
    if not tag:
        raise ValueError("pipeline_tag must contain at least one safe character.")
    return tag


def load_base_save_dir(cfg_path: Path) -> Path:
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    save_dir = Path(cfg.get("save_dir", "outputs"))
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir
    return save_dir


def stage_specs() -> list[StageSpec]:
    return [
        StageSpec(
            name="human_pose",
            script="train/rnn/train_human_pose.py",
            module_name="human_pose",
            fixed_args=["--epochs", "300"],
        ),
        StageSpec(
            name="velocity_contact",
            script="train/rnn/train_velocity_contact.py",
            module_name="velocity_contact",
            fixed_args=["--epochs", "200"],
            dependency_args=["--hp_ckpt", "{hp_ckpt}"],
        ),
        StageSpec(
            name="object_trans",
            script="train/rnn/train_object_trans.py",
            module_name="object_trans",
            fixed_args=["--epochs", "400", "--batch_size", "40"],
            dependency_args=["--hp_ckpt", "{hp_ckpt}", "--vc_ckpt", "{vc_ckpt}"],
        ),
        StageSpec(
            name="joint_train",
            script="train/rnn/train_object_trans.py",
            module_name="joint_train",
            fixed_args=[
                "--joint_train",
                "--lr",
                "0.0002",
                "--epochs",
                "100",
                "--batch_size",
                "30",
            ],
            dependency_args=[
                "--hp_ckpt",
                "{hp_ckpt}",
                "--vc_ckpt",
                "{vc_ckpt}",
                "--ot_ckpt",
                "{ot_ckpt}",
            ],
        ),
    ]


def common_args(args: argparse.Namespace, cfg_path: Path, run_suffix: str) -> list[str]:
    values = ["--cfg", str(cfg_path), "--run_suffix", run_suffix]
    if args.seed is not None:
        values.extend(["--seed", str(args.seed)])
    if args.debug:
        values.append("--debug")
    if args.no_trans:
        values.append("--no_trans")
    if args.model_arch:
        values.extend(["--model_arch", args.model_arch])
    return values


def expand_dependency_args(parts: Sequence[str], ckpts: dict[str, str]) -> list[str]:
    expanded = []
    for part in parts:
        expanded.append(part.format(**ckpts))
    return expanded


def required_ckpt_keys(parts: Sequence[str]) -> list[str]:
    keys: list[str] = []
    for part in parts:
        keys.extend(re.findall(r"{([^}]+)}", part))
    return keys


def validate_dependency_ckpts(spec: StageSpec, ckpts: dict[str, str]) -> None:
    for key in required_ckpt_keys(spec.dependency_args):
        path = Path(ckpts[key])
        if not path.is_file():
            raise FileNotFoundError(f"{spec.name} requires existing checkpoint {key}: {path}")


def find_stage_best(base_save_dir: Path, module_name: str, run_suffix: str) -> tuple[Path, Path]:
    pattern = str(base_save_dir / f"{module_name}*{run_suffix}*" / "best.pt")
    matches = [Path(path) for path in glob.glob(pattern)]
    if not matches:
        raise FileNotFoundError(f"No best.pt found for {module_name} with pattern: {pattern}")
    best_ckpt = max(matches, key=lambda path: path.stat().st_mtime)
    return best_ckpt.parent, best_ckpt


def write_manifest(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def _run_command_pipe(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    """Fallback for non-POSIX systems."""
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def run_command(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if os.name != "posix":
        return _run_command_pipe(command, log_path, env)

    master_fd, slave_fd = pty.openpty()
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            env=env,
        )
        os.close(slave_fd)
        try:
            while True:
                if process.poll() is not None:
                    break
                readable, _, _ = select.select([master_fd], [], [], 0.2)
                if not readable:
                    continue
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                text = chunk.decode(errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                log_file.write(text)
                log_file.flush()

            while True:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                text = chunk.decode(errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                log_file.write(text)
                log_file.flush()
        finally:
            os.close(master_fd)
        return process.wait()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg_path = cfg_path.resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = safe_tag(args.pipeline_tag or f"seq_{timestamp}")
    base_save_dir = load_base_save_dir(cfg_path)
    pipeline_dir = base_save_dir / f"rnn_sequential_{run_suffix}"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pipeline_dir / "manifest.json"

    shared_args = common_args(args, cfg_path, run_suffix)
    ckpts: dict[str, str] = {}
    results: list[StageResult] = []

    manifest = {
        "pipeline_tag": run_suffix,
        "cfg": str(cfg_path),
        "base_save_dir": str(base_save_dir),
        "pipeline_dir": str(pipeline_dir),
        "dry_run": bool(args.dry_run),
        "stages": [],
    }

    for spec in stage_specs():
        command = [
            sys.executable,
            str(PROJECT_ROOT / spec.script),
            *shared_args,
            *spec.fixed_args,
            *expand_dependency_args(spec.dependency_args, ckpts),
        ]
        log_path = pipeline_dir / f"{spec.name}.log"
        result = StageResult(
            name=spec.name,
            command=command,
            log_path=str(log_path),
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        results.append(result)
        manifest["stages"] = [asdict(item) for item in results]
        write_manifest(manifest_path, manifest)

        print("\n" + "=" * 80)
        print(f"[{len(results)}/4] {spec.name}")
        print(" ".join(command))
        print("=" * 80)

        if args.dry_run:
            result.returncode = 0
            result.finished_at = datetime.now().isoformat(timespec="seconds")
            placeholder = f"<{spec.name}_best.pt>"
            result.save_dir = f"<{spec.name}_save_dir>"
            result.best_ckpt = placeholder
            if spec.name == "human_pose":
                ckpts["hp_ckpt"] = placeholder
            elif spec.name == "velocity_contact":
                ckpts["vc_ckpt"] = placeholder
            elif spec.name == "object_trans":
                ckpts["ot_ckpt"] = placeholder
            manifest["stages"] = [asdict(item) for item in results]
            manifest["checkpoints"] = dict(ckpts)
            write_manifest(manifest_path, manifest)
            continue

        validate_dependency_ckpts(spec, ckpts)
        result.returncode = run_command(command, log_path)
        result.finished_at = datetime.now().isoformat(timespec="seconds")
        if result.returncode != 0:
            manifest["stages"] = [asdict(item) for item in results]
            write_manifest(manifest_path, manifest)
            raise SystemExit(f"{spec.name} failed with exit code {result.returncode}. See {log_path}")

        save_dir, best_ckpt = find_stage_best(base_save_dir, spec.module_name, run_suffix)
        result.save_dir = str(save_dir)
        result.best_ckpt = str(best_ckpt)
        if spec.name == "human_pose":
            ckpts["hp_ckpt"] = str(best_ckpt)
        elif spec.name == "velocity_contact":
            ckpts["vc_ckpt"] = str(best_ckpt)
        elif spec.name == "object_trans":
            ckpts["ot_ckpt"] = str(best_ckpt)

        manifest["stages"] = [asdict(item) for item in results]
        manifest["checkpoints"] = dict(ckpts)
        write_manifest(manifest_path, manifest)
        print(f"{spec.name} best checkpoint: {best_ckpt}")

    manifest["stages"] = [asdict(item) for item in results]
    manifest["checkpoints"] = dict(ckpts)
    write_manifest(manifest_path, manifest)
    print("\nSequential RNN training finished.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
