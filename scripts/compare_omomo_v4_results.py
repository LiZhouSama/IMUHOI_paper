#!/usr/bin/env python3
"""Compare OMOMO v4 evaluation with the shared-HP baseline and v3 results."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN/omomo_baseline_v3_new_hp_20260823"
)
DEFAULT_V4_ROOT = PROJECT_ROOT / "outputs/IMUHOI++/RNN/omomo_v4_pose_prior_20260823"
METRICS = (
    ("mpjpe", "MPJPE (cm)", "lower"),
    ("mpjre_angle", "MPJRE (deg)", "lower"),
    ("root_trans_err", "Root (cm)", "lower"),
    ("jitter", "Jitter", "lower"),
    ("obj_trans_err_fusion", "Object Fusion (cm)", "lower"),
    ("hoi_err_fusion", "HOI Fusion (cm)", "lower"),
    ("contact_f1_lhand", "Left contact F1", "higher"),
    ("contact_f1_rhand", "Right contact F1", "higher"),
    ("contact_f1_obj", "Object contact F1", "higher"),
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_eval(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", {})
    metrics = results.get("omomo") if isinstance(results, dict) else None
    if not isinstance(metrics, dict):
        metrics = payload.get("omomo", payload)
    if not isinstance(metrics, dict):
        raise ValueError(f"Evaluation file has no OMOMO metrics: {path}")
    return {"path": str(path), "protocol": payload, "metrics": metrics}


def _assert_protocol(rows: Dict[str, Dict[str, object]]) -> None:
    keys = (
        "online_window",
        "resolved_online_window",
        "online_lookahead",
        "resolved_online_lookahead",
        "seed",
        "simulate_imu_noise",
        "simulate_imu_noise_seed",
        "object_trans_arch",
    )
    reference = rows["baseline"]["protocol"]
    for name, row in rows.items():
        for key in keys:
            if row["protocol"].get(key) != reference.get(key):
                raise ValueError(
                    f"Protocol mismatch for {name}: {key}="
                    f"{row['protocol'].get(key)!r}, expected {reference.get(key)!r}"
                )


def _finite(value) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _is_better(candidate, other, direction: str) -> Optional[bool]:
    candidate_value = _finite(candidate)
    other_value = _finite(other)
    if candidate_value is None or other_value is None:
        return None
    if direction == "lower":
        return candidate_value < other_value
    return candidate_value > other_value


def _format(value) -> str:
    numeric = _finite(value)
    return "NaN" if numeric is None else f"{numeric:.4f}"


def _delta(left, right) -> Optional[float]:
    left_value = _finite(left)
    right_value = _finite(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _write_outputs(
    output_root: Path,
    rows: Dict[str, Dict[str, object]],
    output_stem: str,
) -> None:
    metric_rows = []
    for key, label, direction in METRICS:
        values = {
            name: row["metrics"].get(key)
            for name, row in rows.items()
        }
        metric_rows.append(
            {
                "metric": key,
                "label": label,
                "direction": direction,
                "baseline": values["baseline"],
                "v3": values["v3"],
                "v4": values["v4"],
                "delta_v4_vs_baseline": _delta(values["v4"], values["baseline"]),
                "delta_v4_vs_v3": _delta(values["v4"], values["v3"]),
                "v4_better_than_baseline": _is_better(
                    values["v4"], values["baseline"], direction
                ),
                "v4_better_than_v3": _is_better(
                    values["v4"], values["v3"], direction
                ),
            }
        )

    protocol_keys = (
        "online_window",
        "resolved_online_window",
        "online_lookahead",
        "resolved_online_lookahead",
        "seed",
        "simulate_imu_noise",
        "simulate_imu_noise_seed",
        "object_trans_arch",
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            key: rows["v4"]["protocol"].get(key) for key in protocol_keys
        },
        "evaluation_files": {
            name: row["path"] for name, row in rows.items()
        },
        "metrics": metric_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{output_stem}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")

    lines = [
        "# OMOMO shared-HP VC comparison",
        "",
        (
            "Protocol: "
            f"window={rows['v4']['protocol'].get('resolved_online_window')}, "
            f"lookahead={rows['v4']['protocol'].get('resolved_online_lookahead')}, "
            f"seed={rows['v4']['protocol'].get('seed')}, "
            "simulated IMU noise "
            f"with seed={rows['v4']['protocol'].get('simulate_imu_noise_seed')}."
        ),
        "",
        "| Metric | Baseline | v3 | v4 | v4 - baseline | v4 - v3 | v4 vs baseline | v4 vs v3 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in metric_rows:
        baseline_status = (
            "better" if item["v4_better_than_baseline"] is True
            else "worse" if item["v4_better_than_baseline"] is False
            else "n/a"
        )
        v3_status = (
            "better" if item["v4_better_than_v3"] is True
            else "worse" if item["v4_better_than_v3"] is False
            else "n/a"
        )
        lines.append(
            f"| {item['label']} | {_format(item['baseline'])} | "
            f"{_format(item['v3'])} | {_format(item['v4'])} | "
            f"{_format(item['delta_v4_vs_baseline'])} | "
            f"{_format(item['delta_v4_vs_v3'])} | {baseline_status} | {v3_status} |"
        )
    markdown_path = output_root / f"{output_stem}.md"
    markdown_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--v4_root", type=Path, default=DEFAULT_V4_ROOT)
    parser.add_argument("--eval_filename", default="eval_batch_online.json")
    parser.add_argument("--output_stem", default="omomo_v4_comparison")
    args = parser.parse_args()
    reference_root = _resolve(args.reference_root)
    v4_root = _resolve(args.v4_root)
    rows = {
        "baseline": _load_eval(
            reference_root / "baseline" / args.eval_filename
        ),
        "v3": _load_eval(reference_root / "v3" / args.eval_filename),
        "v4": _load_eval(
            v4_root / "08_v4_supervised_pose_prior" / args.eval_filename
        ),
    }
    _assert_protocol(rows)
    _write_outputs(v4_root, rows, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())