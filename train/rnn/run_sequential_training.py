"""
Run the RNN training stages in order and wire each best checkpoint forward.

Stages:
1. train_human_pose.py --epochs 300
2. train_velocity_contact.py --epochs 200 --hp_ckpt <stage1 best>
3. train_object_trans.py --joint_train --epochs 300 --hp_ckpt <stage1 best>
   --vc_ckpt <stage2 best>

The OT script automatically performs its joint-training phase in the same OT
output directory. All three stage directories are created below one new
pipeline parent directory. Training data uses the dataset-level
simulate_imu_noise path by default.
"""
from __future__ import annotations

import argparse
import errno
import glob
import json
import os
import re
import select
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml

if os.name == "posix":
    import pty
else:  # pragma: no cover - exercised on Windows runners
    pty = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = "configs/IMUHOI_train_rnn.yaml"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TQDM_PROGRESS_RE = re.compile(
    r"^(?P<head>(?:Epoch|Test)\s+\d+:\s+.*?\|\s*\d+/\d+)\s+"
    r"\[(?P<time>[^,\]]+),\s*(?P<rate>[?<\d.]+(?:it/s|s/it))"
    r"(?:,\s*(?P<postfix>.*))?\]$"
)


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
    parser.add_argument("--seed", type=int, default=42, help="Forward a seed to every stage.")
    parser.add_argument("--debug", action="store_true", help="Forward --debug to every stage.")
    parser.add_argument("--no_trans", action="store_true", help="Forward --no_trans to every stage.")
    parser.add_argument(
        "--model_arch",
        choices=["rnn", "dit", "mamba_simple"],
        default=None,
        help="Forward --model_arch to every stage.",
    )
    parser.add_argument(
        "--velocity_contact_arch",
        default=None,
        help=(
            "Override the VC architecture for Stage 2, Stage 3/joint_train, "
            "and the final validation eval (for example: imuhoi or v3)."
        ),
    )
    parser.add_argument(
        "--object_trans_arch",
        default=None,
        help=(
            "Override the ObjectTrans architecture for Stage 3/joint_train "
            "and the final validation eval (for example: current or v2)."
        ),
    )
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        default=None,
        help=(
            "Override cfg train_datasets for every stage, e.g. "
            "--train_datasets hodome or --train_datasets imhd behave."
        ),
    )
    parser.add_argument(
        "--start_stage",
        choices=["human_pose", "velocity_contact", "object_trans"],
        default="human_pose",
        help=(
            "Resume an existing pipeline from this stage. Earlier stages must "
            "already have a best.pt checkpoint."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader workers for every stage (0 avoids WSL worker deadlocks).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print/write the planned commands without launching training.",
    )
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--simulate_imu_noise",
        dest="simulate_imu_noise",
        action="store_true",
        help="Enable simulate_imu_noise for this pipeline (the default).",
    )
    noise_group.add_argument(
        "--no_simulate_imu_noise",
        dest="simulate_imu_noise",
        action="store_false",
        help="Disable simulate_imu_noise for this pipeline.",
    )
    parser.set_defaults(simulate_imu_noise=True)
    return parser.parse_args()


def safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    tag = tag.strip("._-")
    if not tag:
        raise ValueError("pipeline_tag must contain at least one safe character.")
    return tag


def load_base_save_dir(cfg_path: Path) -> Path:
    with cfg_path.open("r", encoding="utf-8") as f:
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
            fixed_args=["--epochs", "300", "--joint_train"],
            dependency_args=["--hp_ckpt", "{hp_ckpt}", "--vc_ckpt", "{vc_ckpt}"],
        ),
    ]


def common_args(
    args: argparse.Namespace,
    cfg_path: Path,
    run_suffix: str,
    pipeline_dir: Path,
) -> list[str]:
    values = [
        "--cfg",
        str(cfg_path),
        "--run_suffix",
        run_suffix,
        "--save_dir",
        str(pipeline_dir),
    ]
    if args.seed is not None:
        values.extend(["--seed", str(args.seed)])
    if args.debug:
        values.append("--debug")
    if args.no_trans:
        values.append("--no_trans")
    if args.model_arch:
        values.extend(["--model_arch", args.model_arch])
    if args.train_datasets:
        values.extend(["--train_datasets", *args.train_datasets])
    if args.num_workers is not None:
        values.extend(["--num_workers", str(max(0, int(args.num_workers)))])
    values.append("--simulate_imu_noise" if args.simulate_imu_noise else "--no_simulate_imu_noise")
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


def build_eval_command(
    args: argparse.Namespace,
    cfg_path: Path,
    pipeline_dir: Path,
    results: Sequence[StageResult],
    ckpts: dict[str, str],
) -> tuple[list[str], Path, Path]:
    """Build the final batch-online eval command from this pipeline's outputs."""
    object_result = next((item for item in results if item.name == "object_trans"), None)
    if object_result is None or object_result.save_dir is None:
        raise RuntimeError("The ObjectTrans stage result is required before evaluation.")

    object_save_dir = Path(object_result.save_dir)
    joint_vc_ckpt = object_save_dir / "best_velocity_contact.pt"
    joint_ot_ckpt = object_save_dir / "best_object_trans.pt"
    if not args.dry_run:
        missing = [path for path in (joint_vc_ckpt, joint_ot_ckpt) if not path.is_file()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "OT --joint_train completed without the expected joint checkpoints: "
                f"{missing_text}"
            )

    output_json = pipeline_dir / "eval_batch_online.json"
    log_path = pipeline_dir / "eval_batch_online.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "eval_batch_online.py"),
        "--config",
        str(cfg_path),
        "--hp_ckpt",
        ckpts["hp_ckpt"],
        "--vc_ckpt",
        str(joint_vc_ckpt),
        "--ot_ckpt",
        str(joint_ot_ckpt),
        "--output_json",
        str(output_json),
    ]
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.no_trans:
        command.append("--no_trans")
    if args.object_trans_arch:
        command.extend(["--object_trans_arch", args.object_trans_arch])
    if args.velocity_contact_arch:
        command.extend(["--velocity_contact_arch", args.velocity_contact_arch])
    command.append("--simulate_imu_noise" if args.simulate_imu_noise else "--no_simulate_imu_noise")
    return command, log_path, output_json


def write_manifest(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def compact_tqdm_progress(line: str) -> str | None:
    """Keep tqdm progress/timing plus total loss, dropping component losses."""
    clean_line = ANSI_ESCAPE_RE.sub("", line).strip()
    match = TQDM_PROGRESS_RE.match(clean_line)
    if match is None:
        return None

    compact = (
        f"{match.group('head')} "
        f"[{match.group('time')}, {match.group('rate')}]"
    )
    postfix = match.group("postfix") or ""
    loss_match = re.search(r"(?:^|,\s*)loss=([^,\]]+)", postfix)
    if loss_match is not None:
        compact += f" loss={loss_match.group(1).strip()}"
    return compact


class CompactConsoleRenderer:
    """Render child tqdm updates in place while preserving ordinary messages."""

    def __init__(self) -> None:
        self.buffer = ""
        self.progress_active = False

    def _finish_progress(self) -> None:
        if self.progress_active:
            sys.stdout.write("\n")
            self.progress_active = False

    def _render_record(self, record: str) -> None:
        compact = compact_tqdm_progress(record)
        if compact is not None:
            sys.stdout.write(f"\r\x1b[2K{compact}")
            sys.stdout.flush()
            self.progress_active = True
            return

        clean_record = ANSI_ESCAPE_RE.sub("", record).strip()
        if not clean_record:
            return

        self._finish_progress()
        sys.stdout.write(clean_record + "\n")
        sys.stdout.flush()

    def feed(self, text: str) -> None:
        self.buffer += text
        while True:
            carriage_index = self.buffer.find("\r")
            newline_index = self.buffer.find("\n")
            delimiter_indices = [
                index
                for index in (carriage_index, newline_index)
                if index >= 0
            ]
            if not delimiter_indices:
                break
            delimiter_index = min(delimiter_indices)
            record = self.buffer[:delimiter_index]
            delimiter = self.buffer[delimiter_index]
            self.buffer = self.buffer[delimiter_index + 1:]
            if delimiter == "\r" and self.buffer.startswith("\n"):
                self.buffer = self.buffer[1:]
            self._render_record(record)

    def close(self) -> None:
        if self.buffer:
            self._render_record(self.buffer)
            self.buffer = ""
        self._finish_progress()
        sys.stdout.flush()


def _run_command_pipe(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    """Fallback for non-POSIX systems."""
    renderer = CompactConsoleRenderer()
    with log_path.open("w", encoding="utf-8") as log_file:
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
            log_file.write(line)
            log_file.flush()
            renderer.feed(line)
        renderer.close()
        return process.wait()


def run_command(command: list[str], log_path: Path) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if os.name != "posix":
        return _run_command_pipe(command, log_path, env)

    master_fd, slave_fd = pty.openpty()
    renderer = CompactConsoleRenderer()
    with log_path.open("w", encoding="utf-8") as log_file:
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
                log_file.write(text)
                log_file.flush()
                renderer.feed(text)

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
                log_file.write(text)
                log_file.flush()
                renderer.feed(text)
        finally:
            os.close(master_fd)
            renderer.close()
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

    shared_args = common_args(args, cfg_path, run_suffix, pipeline_dir)
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

    specs = stage_specs()
    start_index = next(
        index for index, spec in enumerate(specs) if spec.name == args.start_stage
    )

    # Keep completed stages in the manifest and restore their checkpoint
    # dependencies when resuming a long-running WSL pipeline.
    if start_index > 0 and not args.dry_run:
        prior_ckpt_keys = {"human_pose": "hp_ckpt", "velocity_contact": "vc_ckpt"}
        for prior_spec in specs[:start_index]:
            save_dir, best_ckpt = find_stage_best(
                pipeline_dir, prior_spec.module_name, run_suffix
            )
            results.append(
                StageResult(
                    name=prior_spec.name,
                    command=[],
                    log_path=str(pipeline_dir / f"{prior_spec.name}.log"),
                    save_dir=str(save_dir),
                    best_ckpt=str(best_ckpt),
                    returncode=0,
                )
            )
            ckpts[prior_ckpt_keys[prior_spec.name]] = str(best_ckpt)

    for spec in specs[start_index:]:
        planning_ckpts = (
            {"hp_ckpt": "<human_pose_best.pt>", "vc_ckpt": "<velocity_contact_best.pt>"}
            if args.dry_run
            else ckpts
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / spec.script),
            *shared_args,
            *spec.fixed_args,
        ]
        if spec.name in {"velocity_contact", "object_trans"} and args.velocity_contact_arch:
            command.extend(["--velocity_contact_arch", args.velocity_contact_arch])
        if spec.name == "object_trans" and args.object_trans_arch:
            command.extend(["--object_trans_arch", args.object_trans_arch])
        command.extend([
            *expand_dependency_args(spec.dependency_args, planning_ckpts),
        ])
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
        print(f"[{len(results)}/{len(specs)}] {spec.name}")
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

        save_dir, best_ckpt = find_stage_best(pipeline_dir, spec.module_name, run_suffix)
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

    eval_command, eval_log_path, eval_output_path = build_eval_command(
        args,
        cfg_path,
        pipeline_dir,
        results,
        ckpts,
    )
    manifest["evaluation"] = {
        "name": "batch_online",
        "command": eval_command,
        "log_path": str(eval_log_path),
        "output_json": str(eval_output_path),
        "returncode": None,
    }
    write_manifest(manifest_path, manifest)

    print("\n" + "=" * 80)
    print("Final batch-online evaluation")
    print(" ".join(eval_command))
    print("=" * 80)
    if args.dry_run:
        manifest["evaluation"]["returncode"] = 0
    else:
        eval_returncode = run_command(eval_command, eval_log_path)
        manifest["evaluation"]["returncode"] = eval_returncode
        if eval_returncode != 0:
            write_manifest(manifest_path, manifest)
            raise SystemExit(
                f"batch_online evaluation failed with exit code {eval_returncode}. "
                f"See {eval_log_path}"
            )
    write_manifest(manifest_path, manifest)
    print("\nSequential RNN training finished.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
