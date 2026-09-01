#!/usr/bin/env python3
"""Compare predicted, GT-forced, and perturbed ObjectTrans feedback states."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import eval_batch_online as batch_eval
from dataset.dataset_IMUHOI import IMUDataset
from eval_IMUHOI import _set_eval_seed, _unwrap_model
from model.rnn.imuhoi_model import load_model
from model.rnn.online import resolve_online_lookahead, resolve_online_window
from utils.hoi_metrics import compute_hoi_error_at_gt_contact
from utils.human_pose import select_hand_anchor_positions
from utils.utils import load_config


DEFAULT_RUN_DIR = Path(
    "outputs/IMUHOI++/RNN/omomo_v4_pose_prior_20260823/"
    "08_v4_supervised_pose_prior"
)
DEFAULT_HP_CKPT = Path(
    "outputs/IMUHOI++/RNN/omomo_baseline_v3_new_hp_20260823/hp/"
    "human_pose_08230110_omomo_newhp_20260823/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RUN_DIR / "config.yaml",
    )
    parser.add_argument(
        "--sequence-list",
        type=Path,
        default=(
            DEFAULT_RUN_DIR
            / "fusion_sequence_error_analysis_lookahead10"
            / "top_sequence_source_paths.txt"
        ),
    )
    parser.add_argument("--hp-ckpt", type=Path, default=DEFAULT_HP_CKPT)
    parser.add_argument(
        "--vc-ckpt",
        type=Path,
        default=(
            DEFAULT_RUN_DIR
            / "object_trans_vc_v4_08240031_08_v4_supervised_pose_prior"
            / "best_velocity_contact.pt"
        ),
    )
    parser.add_argument(
        "--ot-ckpt",
        type=Path,
        default=(
            DEFAULT_RUN_DIR
            / "object_trans_vc_v4_08240031_08_v4_supervised_pose_prior"
            / "best_object_trans.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_RUN_DIR
            / "fusion_sequence_error_analysis_lookahead10"
            / "feedback_state_comparison"
        ),
    )
    parser.add_argument("--online-window", type=int, default=60)
    parser.add_argument("--online-lookahead", type=int, default=10)
    parser.add_argument("--online-batch-size", type=int, default=8)
    parser.add_argument(
        "--injection-frame",
        type=int,
        default=80,
        help="Zero-based published frame whose previous state receives the impulse.",
    )
    parser.add_argument("--perturbation-cm", type=float, default=5.0)
    parser.add_argument("--perturbation-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulate-imu-noise-seed", type=int, default=42)
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument(
        "--simulate-imu-noise",
        dest="simulate_imu_noise",
        action="store_true",
    )
    noise.add_argument(
        "--no-simulate-imu-noise",
        dest="simulate_imu_noise",
        action="store_false",
    )
    parser.set_defaults(simulate_imu_noise=True)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_sequence_paths(path: Path) -> List[Path]:
    path = resolve_path(path)
    paths = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        sequence_path = resolve_path(Path(value))
        if not sequence_path.is_file() or sequence_path.suffix.lower() != ".pt":
            raise FileNotFoundError(f"Invalid sequence path: {sequence_path}")
        key = str(sequence_path)
        if key not in seen:
            paths.append(sequence_path)
            seen.add(key)
    if not paths:
        raise ValueError(f"No PT paths found in {path}")
    return paths


def prediction_time_tensor(prediction: Dict[str, Any], key: str):
    value = prediction.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    value = value.detach().cpu()
    if value.dim() >= 2 and value.shape[0] == 1:
        value = value[0]
    return value


def series_summary(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {}
    edge = max(1, int(values.numel()) // 10)
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "p90": torch.quantile(values, 0.90).item(),
        "max": values.max().item(),
        "first_10pct_mean": values[:edge].mean().item(),
        "last_10pct_mean": values[-edge:].mean().item(),
        "late_minus_early": (
            values[-edge:].mean() - values[:edge].mean()
        ).item(),
    }


def collect_mode_records(
    pred_cache: Dict[str, Dict[str, Any]],
    sample_cache: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor]]:
    records: Dict[str, Dict[str, Any]] = {}
    trajectories: Dict[str, torch.Tensor] = {}
    candidate_keys = (
        ("left_fk", "pred_lhand_obj_trans"),
        ("right_fk", "pred_rhand_obj_trans"),
        ("velocity", "pred_imu_obj_trans"),
        ("static", "pred_static_obj_trans"),
    )

    for sample in sample_cache:
        sequence_path = str(Path(str(sample["seq_path"])).resolve())
        prediction = pred_cache.get(sequence_path)
        if prediction is None:
            raise KeyError(f"Missing prediction for {sequence_path}")
        pred_obj = prediction_time_tensor(prediction, "pred_obj_trans")
        if not isinstance(pred_obj, torch.Tensor):
            raise ValueError(f"Missing pred_obj_trans for {sequence_path}")
        gt_obj = sample["obj_trans"].detach().cpu()
        length = min(int(pred_obj.shape[0]), int(gt_obj.shape[0]))
        pred_obj = pred_obj[:length]
        gt_obj = gt_obj[:length]
        trajectories[sequence_path] = pred_obj

        obj_frame_error = torch.linalg.norm(pred_obj - gt_obj, dim=-1) * 100.0
        pred_hand = prediction_time_tensor(prediction, "pred_palm_glb_pos")
        if not isinstance(pred_hand, torch.Tensor):
            pred_hand = prediction_time_tensor(prediction, "pred_hand_glb_pos")
        if not isinstance(pred_hand, torch.Tensor):
            raise ValueError(f"Missing predicted hand positions for {sequence_path}")
        pred_hand = pred_hand[:length]
        gt_hand = select_hand_anchor_positions(
            sample["position_global"].detach().cpu()[:length]
        )
        lhand_contact = sample["lhand_contact"].detach().cpu()[:length].bool()
        rhand_contact = sample["rhand_contact"].detach().cpu()[:length].bool()
        hoi_error = compute_hoi_error_at_gt_contact(
            pred_obj,
            gt_obj,
            pred_hand,
            gt_hand,
            lhand_contact,
            rhand_contact,
        )

        candidate_errors = {}
        candidate_frame_errors = []
        for name, key in candidate_keys:
            candidate = prediction_time_tensor(prediction, key)
            if not isinstance(candidate, torch.Tensor):
                continue
            candidate = candidate[:length]
            if candidate.shape != gt_obj.shape:
                continue
            frame_error = torch.linalg.norm(candidate - gt_obj, dim=-1) * 100.0
            candidate_errors[name] = frame_error.mean().item()
            candidate_frame_errors.append(frame_error)
        best_single_candidate = float("nan")
        if candidate_frame_errors:
            best_single_candidate = torch.stack(
                candidate_frame_errors, dim=-1
            ).min(dim=-1).values.mean().item()

        gating = prediction_time_tensor(prediction, "gating_weights")
        gating_mean = {}
        if isinstance(gating, torch.Tensor) and gating.dim() == 2:
            names = ("left_fk", "right_fk", "velocity", "static")
            gating_mean = dict(zip(names, gating[:length].mean(dim=0).tolist()))

        records[sequence_path] = {
            "seq_name": str(sample.get("seq_name", Path(sequence_path).stem)),
            "seq_path": sequence_path,
            "obj_name": str(sample.get("obj_name", "unknown_object")),
            "num_frames": length,
            "obj_error_cm": obj_frame_error.mean().item(),
            "hoi_error_cm": float(hoi_error),
            "obj_error_profile_cm": series_summary(obj_frame_error),
            "candidate_error_cm": candidate_errors,
            "best_single_candidate_cm": best_single_candidate,
            "gating_mean_weights": gating_mean,
        }
    return records, trajectories


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def aggregate_records(records: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    values = list(records.values())
    return {
        "sequence_count": len(values),
        "obj_error_cm": finite_mean([row["obj_error_cm"] for row in values]),
        "hoi_error_cm": finite_mean([row["hoi_error_cm"] for row in values]),
        "best_single_candidate_cm": finite_mean(
            [row["best_single_candidate_cm"] for row in values]
        ),
        "late_minus_early_cm": finite_mean(
            [
                row["obj_error_profile_cm"].get("late_minus_early", float("nan"))
                for row in values
            ]
        ),
    }


def first_sustained_below(
    values: torch.Tensor,
    start: int,
    threshold: float,
    sustain: int = 10,
):
    for index in range(int(start), int(values.numel())):
        end = min(index + int(sustain), int(values.numel()))
        if end - index >= min(int(sustain), int(values.numel()) - index):
            if bool((values[index:end] <= threshold).all().item()):
                return index - int(start)
    return None


def build_comparisons(
    mode_records: Dict[str, Dict[str, Dict[str, Any]]],
    trajectories: Dict[str, Dict[str, torch.Tensor]],
    injection_frame: int,
    perturbation_cm: float,
) -> List[Dict[str, Any]]:
    rows = []
    for sequence_path in sorted(
        mode_records["normal"],
        key=lambda key: int(mode_records["normal"][key]["seq_name"]),
    ):
        normal = mode_records["normal"][sequence_path]
        gt_prev = mode_records["gt_prev"][sequence_path]
        perturb = mode_records["perturb"][sequence_path]
        normal_traj = trajectories["normal"][sequence_path]
        perturb_traj = trajectories["perturb"][sequence_path]
        length = min(int(normal_traj.shape[0]), int(perturb_traj.shape[0]))
        response = torch.linalg.norm(
            perturb_traj[:length] - normal_traj[:length], dim=-1
        ) * 100.0
        if injection_frame >= length:
            raise ValueError(
                f"Injection frame {injection_frame} exceeds {normal['seq_name']} "
                f"published length {length}"
            )
        offsets = (0, 1, 5, 10, 30, 60)
        response_at = {
            str(offset): (
                response[injection_frame + offset].item()
                if injection_frame + offset < length else None
            )
            for offset in offsets
        }
        pre_injection = response[:injection_frame]
        post_injection = response[injection_frame:]
        rows.append({
            "seq_name": normal["seq_name"],
            "seq_path": sequence_path,
            "obj_name": normal["obj_name"],
            "num_frames": length,
            "normal": normal,
            "gt_prev": gt_prev,
            "perturb": perturb,
            "gt_prev_obj_improvement_cm": (
                normal["obj_error_cm"] - gt_prev["obj_error_cm"]
            ),
            "gt_prev_hoi_improvement_cm": (
                normal["hoi_error_cm"] - gt_prev["hoi_error_cm"]
            ),
            "perturb_obj_error_delta_cm": (
                perturb["obj_error_cm"] - normal["obj_error_cm"]
            ),
            "perturb_hoi_error_delta_cm": (
                perturb["hoi_error_cm"] - normal["hoi_error_cm"]
            ),
            "perturb_response": {
                "pre_injection_max_cm": (
                    pre_injection.max().item() if pre_injection.numel() else 0.0
                ),
                "response_at_offset_cm": response_at,
                "max_after_cm": post_injection.max().item(),
                "mean_after_cm": post_injection.mean().item(),
                "max_gain_vs_injected": (
                    post_injection.max().item() / float(perturbation_cm)
                ),
                "recovery_offset_below_1cm": first_sustained_below(
                    response, injection_frame, 1.0
                ),
                "curve_cm": response[injection_frame:].tolist(),
            },
        })
    return rows


def flatten_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for row in rows:
        output.append({
            "seq_name": row["seq_name"],
            "obj_name": row["obj_name"],
            "num_frames": row["num_frames"],
            "normal_obj_cm": row["normal"]["obj_error_cm"],
            "gt_prev_obj_cm": row["gt_prev"]["obj_error_cm"],
            "gt_prev_obj_improvement_cm": row["gt_prev_obj_improvement_cm"],
            "perturb_obj_cm": row["perturb"]["obj_error_cm"],
            "perturb_obj_delta_cm": row["perturb_obj_error_delta_cm"],
            "normal_hoi_cm": row["normal"]["hoi_error_cm"],
            "gt_prev_hoi_cm": row["gt_prev"]["hoi_error_cm"],
            "gt_prev_hoi_improvement_cm": row["gt_prev_hoi_improvement_cm"],
            "perturb_hoi_cm": row["perturb"]["hoi_error_cm"],
            "perturb_hoi_delta_cm": row["perturb_hoi_error_delta_cm"],
            "normal_best_candidate_cm": row["normal"]["best_single_candidate_cm"],
            "gt_prev_best_candidate_cm": row["gt_prev"]["best_single_candidate_cm"],
            "normal_late_minus_early_cm": row["normal"]["obj_error_profile_cm"].get(
                "late_minus_early"
            ),
            "gt_prev_late_minus_early_cm": row["gt_prev"]["obj_error_profile_cm"].get(
                "late_minus_early"
            ),
            "perturb_response_0_cm": row["perturb_response"]["response_at_offset_cm"]["0"],
            "perturb_response_10_cm": row["perturb_response"]["response_at_offset_cm"]["10"],
            "perturb_response_30_cm": row["perturb_response"]["response_at_offset_cm"]["30"],
            "perturb_response_60_cm": row["perturb_response"]["response_at_offset_cm"]["60"],
            "perturb_max_after_cm": row["perturb_response"]["max_after_cm"],
            "perturb_max_gain": row["perturb_response"]["max_gain_vs_injected"],
            "recovery_offset_below_1cm": row["perturb_response"]["recovery_offset_below_1cm"],
        })
    return output


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, rows: Sequence[Dict[str, Any]], injection_frame: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["seq_name"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for axis, key, title in (
        (axes[0], "obj_error_cm", "Raw ObjectTrans error"),
        (axes[1], "hoi_error_cm", "Stage-1-hand HOI error"),
    ):
        axis.bar(x - width, [row["normal"][key] for row in rows], width, label="normal")
        axis.bar(x, [row["gt_prev"][key] for row in rows], width, label="GT previous state")
        axis.bar(x + width, [row["perturb"][key] for row in rows], width, label="+5 cm impulse")
        axis.set_ylabel("cm")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_xlabel("sequence")
    fig.tight_layout()
    fig.savefig(output_dir / "feedback_mode_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 6))
    for row in rows:
        curve = row["perturb_response"]["curve_cm"]
        offsets = np.arange(len(curve))
        axis.plot(offsets, curve, linewidth=1.8, label=str(row["seq_name"]))
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="1 cm")
    axis.set(
        title=f"Response to +5 cm X state impulse at published frame {injection_frame}",
        xlabel="frames after injection",
        ylabel="distance from normal trajectory (cm)",
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "feedback_perturbation_response.png", dpi=180)
    plt.close(fig)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.3f}" if math.isfinite(number) else "-"


def write_markdown(
    path: Path,
    protocol: Dict[str, Any],
    aggregate: Dict[str, Dict[str, float]],
    comparisons: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        "# ObjectTrans feedback three-way comparison",
        "",
        f"Protocol: `{json.dumps(protocol, ensure_ascii=False)}`",
        "",
        "Human refinement is disabled so the comparison isolates raw ObjectTrans feedback.",
        "",
        "## Aggregate",
        "",
        "| Mode | Object cm | HOI cm | Best single candidate cm | Late - early cm |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("normal", "gt_prev", "perturb"):
        row = aggregate[mode]
        lines.append(
            f"| {mode} | {fmt(row['obj_error_cm'])} | {fmt(row['hoi_error_cm'])} | "
            f"{fmt(row['best_single_candidate_cm'])} | {fmt(row['late_minus_early_cm'])} |"
        )
    lines.extend([
        "",
        "## Per sequence",
        "",
        "| Seq | Object | Normal obj | GT-prev obj | Improve | Perturb obj | "
        "Normal HOI | GT-prev HOI | Improve | Response 0/10/30 | Max gain | Recover <1 cm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparisons:
        response = row["perturb_response"]["response_at_offset_cm"]
        response_text = "/".join(
            fmt(response.get(str(offset))) for offset in (0, 10, 30)
        )
        lines.append(
            f"| {row['seq_name']} | {row['obj_name']} | "
            f"{fmt(row['normal']['obj_error_cm'])} | {fmt(row['gt_prev']['obj_error_cm'])} | "
            f"{fmt(row['gt_prev_obj_improvement_cm'])} | {fmt(row['perturb']['obj_error_cm'])} | "
            f"{fmt(row['normal']['hoi_error_cm'])} | {fmt(row['gt_prev']['hoi_error_cm'])} | "
            f"{fmt(row['gt_prev_hoi_improvement_cm'])} | {response_text} | "
            f"{fmt(row['perturb_response']['max_gain_vs_injected'])} | "
            f"{row['perturb_response']['recovery_offset_below_1cm']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    sequence_paths = load_sequence_paths(args.sequence_list)
    module_paths = {
        "human_pose": str(resolve_path(args.hp_ckpt)),
        "velocity_contact": str(resolve_path(args.vc_ckpt)),
        "object_trans": str(resolve_path(args.ot_ckpt)),
    }
    for name, path in module_paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(str(config_path))
    config.model_arch = "rnn"
    config.velocity_contact_arch = "v4"
    config.object_trans_arch = "current"
    config.object_trans_state_feedback = True
    config.enable_ot_refine = False
    config.debug = False

    _set_eval_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = IMUDataset(
        data_dir=str(sequence_paths[0].parent),
        sequence_paths=[str(path) for path in sequence_paths],
        window_size=int(config.test.get("window", 60)),
        debug=False,
        simulate_imu_noise=bool(args.simulate_imu_noise),
        simulate_imu_noise_seed=int(args.simulate_imu_noise_seed),
        min_obj_contact_frames=0,
        full_sequence=True,
        resolve_bimanual_contact_conflicts=config.get(
            "resolve_bimanual_contact_conflicts", False
        ),
    )
    if len(dataset) != len(sequence_paths):
        raise ValueError(
            f"Expected {len(sequence_paths)} sequences, dataset loaded {len(dataset)}"
        )

    model = _unwrap_model(
        load_model(
            config,
            device,
            no_trans=False,
            module_paths=module_paths,
            object_trans_arch="current",
            velocity_contact_arch="v4",
        )
    ).eval()
    online_window = resolve_online_window(config, args.online_window)
    online_lookahead = resolve_online_lookahead(config, args.online_lookahead)
    axis_index = {"x": 0, "y": 1, "z": 2}[args.perturbation_axis]
    perturbation = [0.0, 0.0, 0.0]
    perturbation[axis_index] = float(args.perturbation_cm) / 100.0

    mode_specs = {
        "normal": {
            "feedback_mode": "normal",
            "feedback_perturbation": None,
            "feedback_injection_frame": None,
        },
        "gt_prev": {
            "feedback_mode": "gt_prev",
            "feedback_perturbation": None,
            "feedback_injection_frame": None,
        },
        "perturb": {
            "feedback_mode": "perturb",
            "feedback_perturbation": perturbation,
            "feedback_injection_frame": int(args.injection_frame),
        },
    }
    mode_records = {}
    mode_trajectories = {}
    with torch.inference_mode():
        for mode, spec in mode_specs.items():
            print(f"\n=== feedback mode: {mode} ===")
            _set_eval_seed(args.seed)
            pred_cache, sample_cache = batch_eval._batched_online_predict_dataset(
                model,
                dataset,
                config,
                device,
                online_batch_size=min(int(args.online_batch_size), len(dataset)),
                online_window=online_window,
                online_lookahead=online_lookahead,
                use_object_data=True,
                compute_fk=False,
                refine_human=False,
                sort_desc=True,
                eval_imu_noise=False,
                eval_imu_noise_seed=None,
                **spec,
            )
            records, trajectories = collect_mode_records(pred_cache, sample_cache)
            mode_records[mode] = records
            mode_trajectories[mode] = trajectories
            aggregate = aggregate_records(records)
            print(
                f"{mode}: object={aggregate['obj_error_cm']:.4f} cm | "
                f"HOI={aggregate['hoi_error_cm']:.4f} cm"
            )

    comparisons = build_comparisons(
        mode_records,
        mode_trajectories,
        injection_frame=int(args.injection_frame),
        perturbation_cm=float(args.perturbation_cm),
    )
    aggregate = {
        mode: aggregate_records(records) for mode, records in mode_records.items()
    }
    aggregate["paired"] = {
        "gt_prev_obj_improvement_cm": finite_mean(
            [row["gt_prev_obj_improvement_cm"] for row in comparisons]
        ),
        "gt_prev_hoi_improvement_cm": finite_mean(
            [row["gt_prev_hoi_improvement_cm"] for row in comparisons]
        ),
        "perturb_obj_error_delta_cm": finite_mean(
            [row["perturb_obj_error_delta_cm"] for row in comparisons]
        ),
        "perturb_hoi_error_delta_cm": finite_mean(
            [row["perturb_hoi_error_delta_cm"] for row in comparisons]
        ),
        "perturb_max_gain": finite_mean(
            [row["perturb_response"]["max_gain_vs_injected"] for row in comparisons]
        ),
    }
    protocol = {
        "sequence_count": len(dataset),
        "sequence_paths": [str(path) for path in sequence_paths],
        "online_window": online_window,
        "online_lookahead": online_lookahead,
        "seed": int(args.seed),
        "simulate_imu_noise": bool(args.simulate_imu_noise),
        "simulate_imu_noise_seed": int(args.simulate_imu_noise_seed),
        "human_refine": False,
        "object_trans_state_feedback": True,
        "injection_frame_zero_based": int(args.injection_frame),
        "perturbation_world_m": perturbation,
        "perturbation_position_only": True,
        "gt_prev_forces_position_and_position_delta": True,
        "gate_feedback_remains_predicted": True,
        "checkpoints": module_paths,
    }
    payload = {
        "protocol": protocol,
        "aggregate": aggregate,
        "per_sequence": comparisons,
    }
    (output_dir / "feedback_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(output_dir / "feedback_comparison.csv", flatten_rows(comparisons))
    write_markdown(
        output_dir / "feedback_comparison.md", protocol, aggregate, comparisons
    )
    write_plots(output_dir, comparisons, int(args.injection_frame))
    print(f"\nResults written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())