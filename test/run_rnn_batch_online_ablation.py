#!/usr/bin/env python3
"""Run batched RNN online ablations with case-specific checkpoints."""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


CASES = (
    "baseline",
    "vc_boundary_zero",
)

LOG_PROGRESS_RE = re.compile(
    r"(BatchOnline|online \d+|Evaluating|预加载和收集信息):\s*(\d+)%\|.*?\|\s*([0-9]+)/([0-9]+)"
)

SUMMARY_METRICS = (
    "mpjpe",
    "mpjre_angle",
    "root_trans_err",
    "stage1_root_trans_err",
    "jitter",
    "obj_trans_err_fusion",
    "obj_trans_err_imu",
    "hoi_err_fusion",
    "hoi_err_imu",
    "contact_f1_lhand",
    "contact_f1_rhand",
    "contact_precision_lhand",
    "contact_precision_rhand",
    "contact_recall_lhand",
    "contact_recall_rhand",
    "contact_f05_lhand",
    "contact_f05_rhand",
    "contact_hard_neg_fp_rate_any",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _fmt(value) -> str:
    if value is None or _is_nan(value):
        return "NaN"
    return f"{float(value):.4f}"


def _delta(value, baseline) -> str:
    if value is None or baseline is None or _is_nan(value) or _is_nan(baseline):
        return "NaN"
    return f"{float(value) - float(baseline):+.4f}"


def _load_first_result(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if not results:
        return None, {}
    dataset_name = next(iter(results.keys()))
    return dataset_name, results[dataset_name]


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _read_log_progress(log_path: Path) -> str:
    if not log_path.exists() or log_path.stat().st_size == 0:
        return "starting"
    with log_path.open("rb") as f:
        f.seek(max(0, log_path.stat().st_size - 16384))
        text = f.read().decode("utf-8", errors="ignore").replace("\r", "\n")
    matches = LOG_PROGRESS_RE.findall(text)
    if matches:
        stage, percent, current, total = matches[-1]
        if stage.startswith("online "):
            stage_name = "step"
        elif stage == "BatchOnline":
            stage_name = "batch"
        elif stage == "Evaluating":
            stage_name = "metric"
        else:
            stage_name = "preload"
        return f"{stage_name} {percent}% {current}/{total}"
    if "Batch Online Evaluation Results" in text:
        return "finalizing"
    return "running"


def _run_case(cmd: List[str], root: Path, log_path: Path, case_name: str, progress) -> Tuple[int, float]:
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while proc.poll() is None:
            elapsed = _format_duration(time.time() - start_time)
            log_progress = _read_log_progress(log_path)
            progress.set_postfix_str(f"{case_name} | {log_progress} | {elapsed} | {log_path.name}", refresh=True)
            time.sleep(2.0)
    duration = time.time() - start_time
    progress.set_postfix_str(f"{case_name} | done | {_format_duration(duration)} | {log_path.name}", refresh=True)
    return proc.returncode, duration


def _write_summary(output_dir: Path, case_results: Dict[str, dict]) -> None:
    baseline = case_results.get("baseline", {}).get("metrics", {})
    header = ["metric", "baseline"]
    aligns = ["---", "---:"]
    for case_name in CASES:
        if case_name == "baseline":
            continue
        header.extend([case_name, "delta"])
        aligns.extend(["---:", "---:"])

    lines = [
        "# RNN Batch Online Ablation Summary",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for metric in SUMMARY_METRICS:
        base_val = baseline.get(metric)
        row = [metric, _fmt(base_val)]
        for case_name in CASES:
            if case_name == "baseline":
                continue
            case_val = case_results.get(case_name, {}).get("metrics", {}).get(metric)
            row.extend([_fmt(case_val), _delta(case_val, base_val)])
        lines.append("| " + " | ".join(row) + " |")

    (output_dir / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    compact = {
        case: {
            "dataset": data.get("dataset"),
            "json": str(data.get("json")),
            "log": str(data.get("log")),
            "metrics": data.get("metrics", {}),
            "duration_sec": data.get("duration_sec"),
            "returncode": data.get("returncode"),
        }
        for case, data in case_results.items()
    }
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _case_flags(args: argparse.Namespace, case_name: str) -> List[str]:
    flags: List[str] = []
    if args.hp_ckpt:
        flags.extend(["--hp_ckpt", args.hp_ckpt])

    if case_name == "baseline":
        if args.baseline_vc_ckpt:
            flags.extend(["--velocity_contact_ckpt", args.baseline_vc_ckpt])
        if args.baseline_ot_ckpt:
            flags.extend(["--object_trans_ckpt", args.baseline_ot_ckpt])
        return flags

    if case_name == "vc_boundary_zero":
        flags.append("--ablate_vc_boundary")
        if args.vc_boundary_zero_vc_ckpt:
            flags.extend(["--velocity_contact_ckpt", args.vc_boundary_zero_vc_ckpt])
        if args.baseline_ot_ckpt:
            flags.extend(["--object_trans_ckpt", args.baseline_ot_ckpt])
        return flags

    raise ValueError(f"Unknown case: {case_name}")


def _validate_case_ckpts(args: argparse.Namespace) -> None:
    missing = []
    if not args.vc_boundary_zero_vc_ckpt:
        missing.append("--vc_boundary_zero_vc_ckpt")
    if missing and not args.allow_missing_case_ckpts:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing ablation checkpoint arguments: {joined}. "
            "Use --allow_missing_case_ckpts only if you intentionally want config/default checkpoints."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval_batch_online.py RNN ablations with case-specific checkpoints.")
    parser.add_argument("--config", default="configs/IMUHOI_train_rnn.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--test_data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--online_window", type=int, default=None)
    parser.add_argument("--online_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--compare_3", action="store_true")
    parser.add_argument("--no_eval_objects", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--hp_ckpt", default=None, help="Common HumanPose checkpoint override for all cases.")
    parser.add_argument("--baseline_vc_ckpt", default=None, help="Optional baseline VelocityContact checkpoint.")
    parser.add_argument("--baseline_ot_ckpt", default=None, help="Optional baseline ObjectTrans checkpoint.")
    parser.add_argument("--vc_boundary_zero_vc_ckpt", default="outputs/IMUHOI_RNN_2/velocity_contact_vc_boundary_zero_06211816/best.pt", help="VelocityContact checkpoint for vc_boundary_zero.")
    parser.add_argument("--allow_missing_case_ckpts", action="store_true")
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument("--enable_ot_refine", dest="ot_refine", action="store_true", default=None)
    refine_group.add_argument("--disable_ot_refine", dest="ot_refine", action="store_false")
    args = parser.parse_args()
    _validate_case_ckpts(args)

    root = _repo_root()
    output_dir = Path(args.output_dir) if args.output_dir else root / "outputs" / f"rnn_batch_online_ablation_{time.strftime('%m%d%H%M')}"
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_results: Dict[str, dict] = {}
    case_bar = tqdm(total=len(CASES), desc="Ablations", unit="case", dynamic_ncols=True)
    for case_name in CASES:
        json_path = output_dir / f"{case_name}.json"
        log_path = output_dir / f"{case_name}.log"
        cmd = [
            sys.executable,
            str(root / "eval_batch_online.py"),
            "--config",
            args.config,
            "--num_workers",
            str(args.num_workers),
            "--online_batch_size",
            str(args.online_batch_size),
            "--output_json",
            str(json_path),
            "--seed",
            str(args.seed),
            *_case_flags(args, case_name),
        ]
        if args.dataset:
            cmd.extend(["--dataset", args.dataset])
        if args.online_window is not None:
            cmd.extend(["--online_window", str(args.online_window)])
        if args.test_data_dir:
            cmd.extend(["--test_data_dir", args.test_data_dir])
        if args.compare_3:
            cmd.append("--compare_3")
        if args.no_eval_objects:
            cmd.append("--no_eval_objects")
        if args.ot_refine is True:
            cmd.append("--enable_ot_refine")
        elif args.ot_refine is False:
            cmd.append("--disable_ot_refine")

        returncode, duration = _run_case(cmd, root, log_path, case_name, case_bar)
        if returncode != 0:
            case_results[case_name] = {
                "dataset": None,
                "json": json_path,
                "log": log_path,
                "metrics": {},
                "returncode": returncode,
                "duration_sec": duration,
            }
            if not args.keep_going:
                _write_summary(output_dir, case_results)
                case_bar.close()
                print(f"{case_name} failed with exit code {returncode}. See {log_path}")
                return returncode
            case_bar.update(1)
            continue

        dataset_name, metrics = _load_first_result(json_path)
        case_results[case_name] = {
            "dataset": dataset_name,
            "json": json_path,
            "log": log_path,
            "metrics": metrics,
            "returncode": returncode,
            "duration_sec": duration,
        }
        case_bar.update(1)
        tqdm.write(f"{case_name}: done in {duration:.2f}s, results={json_path}")

    case_bar.close()
    _write_summary(output_dir, case_results)
    print(f"Summary written to {output_dir / 'ablation_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
