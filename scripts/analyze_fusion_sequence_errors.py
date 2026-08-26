#!/usr/bin/env python3
"""Analyze per-sequence published object and HOI fusion errors."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


OBJ_METRIC = "obj_trans_err_fusion_published_cm"
HOI_METRIC = "hoi_err_fusion_cm"
HISTOGRAM_EDGES_CM = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, math.inf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-override-json",
        type=Path,
        default=None,
        help="Replace matching records with targeted rerun diagnostics.",
    )
    parser.add_argument("--dataset", default="omomo")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-copy-pt", action="store_true")
    return parser.parse_args()


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def record_key(record: Dict[str, Any]) -> str:
    seq_path = str(record.get("seq_path") or "")
    if seq_path:
        return str(Path(seq_path).expanduser().resolve())
    return str(record.get("seq_name") or "")


def nested_float(record: Dict[str, Any], *keys: str) -> float:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return float("nan")
        value = value.get(key)
    return finite_float(value)


def finite_array(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray([finite_float(value) for value in values], dtype=np.float64)
    return array[np.isfinite(array)]


def distribution_summary(values: Sequence[Any]) -> Dict[str, Any]:
    array = finite_array(values)
    if array.size == 0:
        return {"count": 0}
    quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    counts, _ = np.histogram(array, bins=np.asarray(HISTOGRAM_EDGES_CM))
    histogram = []
    for index, count in enumerate(counts):
        low = HISTOGRAM_EDGES_CM[index]
        high = HISTOGRAM_EDGES_CM[index + 1]
        label = f"[{low:g}, {high:g})" if math.isfinite(high) else f"[{low:g}, inf)"
        histogram.append({
            "bin_cm": label,
            "count": int(count),
            "percentage": float(count / array.size * 100.0),
        })
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(array.max()),
        "histogram": histogram,
    }


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    if values.size < 2 or np.unique(labels).size < 2:
        return float("nan")
    grand_mean = float(values.mean())
    total = float(np.square(values - grand_mean).sum())
    if total <= 0.0:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        between += float(group.size) * float(group.mean() - grand_mean) ** 2
    return between / total


def permutation_test_eta_squared(
    values: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    observed = eta_squared(values, labels)
    if not math.isfinite(observed) or permutations <= 0:
        return {"eta_squared": observed, "permutation_p": float("nan"), "permutations": 0}
    exceedances = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        exceedances += eta_squared(values, permuted) >= observed
    return {
        "eta_squared": observed,
        "permutation_p": float((exceedances + 1) / (permutations + 1)),
        "permutations": int(permutations),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlations(records: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    factors = {
        "num_frames": lambda row: row.get("num_frames"),
        "object_path_length_m": lambda row: nested_float(row, "gt_object_motion", "path_length_m"),
        "object_displacement_m": lambda row: nested_float(row, "gt_object_motion", "displacement_m"),
        "object_mean_speed_mps": lambda row: nested_float(row, "gt_object_motion", "mean_speed_mps"),
        "any_hand_contact_fraction": lambda row: nested_float(row, "contact_fraction", "any_hand"),
        "root_trans_error_cm": lambda row: nested_float(row, "human_error_cm", "root_trans"),
        "refined_hand_error_cm": lambda row: nested_float(row, "human_error_cm", "refined_hand"),
        "oracle_candidate_error_cm": lambda row: nested_float(row, "candidate_error_cm", "oracle_per_frame"),
        "fusion_minus_oracle_cm": lambda row: row.get("fusion_minus_oracle_candidate_cm"),
        "gate_oracle_match_fraction": lambda row: nested_float(
            row, "gating", "argmax_matches_oracle_fraction"
        ),
        "gating_entropy": lambda row: nested_float(row, "gating", "entropy"),
        "published_minus_raw_cm": lambda row: row.get("published_minus_raw_cm"),
        "late_minus_early_cm": lambda row: (
            nested_float(row, "obj_error_profile_cm", "last_10pct_mean")
            - nested_float(row, "obj_error_profile_cm", "first_10pct_mean")
        ),
    }
    output = {}
    for name, getter in factors.items():
        pairs = [
            (finite_float(row.get(metric)), finite_float(getter(row)))
            for row in records
        ]
        pairs = [(left, right) for left, right in pairs if math.isfinite(left) and math.isfinite(right)]
        if len(pairs) < 3:
            output[name] = {"count": len(pairs), "pearson_r": float("nan"), "spearman_r": float("nan")}
            continue
        left = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        right = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        pearson = float(np.corrcoef(left, right)[0, 1]) if left.std() > 0 and right.std() > 0 else float("nan")
        left_rank = rankdata(left)
        right_rank = rankdata(right)
        spearman = (
            float(np.corrcoef(left_rank, right_rank)[0, 1])
            if left_rank.std() > 0 and right_rank.std() > 0 else float("nan")
        )
        output[name] = {"count": len(pairs), "pearson_r": pearson, "spearman_r": spearman}
    return output


def object_summaries(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("obj_name") or "unknown_object")].append(record)
    rows = []
    total = max(len(records), 1)
    for obj_name, group in grouped.items():
        obj_summary = distribution_summary([row.get(OBJ_METRIC) for row in group])
        hoi_summary = distribution_summary([row.get(HOI_METRIC) for row in group])
        rows.append({
            "obj_name": obj_name,
            "sequence_count": len(group),
            "sequence_percentage": len(group) / total * 100.0,
            "obj_mean_cm": obj_summary.get("mean", float("nan")),
            "obj_median_cm": obj_summary.get("median", float("nan")),
            "obj_p90_cm": obj_summary.get("p90", float("nan")),
            "obj_max_cm": obj_summary.get("max", float("nan")),
            "hoi_mean_cm": hoi_summary.get("mean", float("nan")),
            "hoi_median_cm": hoi_summary.get("median", float("nan")),
            "hoi_p90_cm": hoi_summary.get("p90", float("nan")),
            "hoi_max_cm": hoi_summary.get("max", float("nan")),
            "mean_path_length_m": float(np.nanmean([
                nested_float(row, "gt_object_motion", "path_length_m") for row in group
            ])),
            "mean_contact_fraction": float(np.nanmean([
                nested_float(row, "contact_fraction", "any_hand") for row in group
            ])),
        })
    return sorted(rows, key=lambda row: finite_float(row["obj_mean_cm"]), reverse=True)


def validate_aggregate_means(
    payload: Dict[str, Any], dataset: str, records: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    aggregate = payload.get("aggregate_results", {}).get(dataset, {})
    checks = {}
    mappings = {
        OBJ_METRIC: "obj_trans_err_fusion_refine",
        HOI_METRIC: "hoi_err_fusion",
    }
    for record_key, aggregate_key in mappings.items():
        values = finite_array(row.get(record_key) for row in records)
        record_mean = float(values.mean()) if values.size else float("nan")
        aggregate_mean = finite_float(aggregate.get(aggregate_key))
        difference = abs(record_mean - aggregate_mean)
        checks[record_key] = {
            "record_count": int(values.size),
            "record_mean": record_mean,
            "aggregate_key": aggregate_key,
            "aggregate_mean": aggregate_mean,
            "absolute_difference": difference,
            "matches": bool(math.isfinite(difference) and difference <= 1e-6),
        }
        if not checks[record_key]["matches"]:
            raise ValueError(
                f"Per-sequence mean mismatch for {record_key}: "
                f"{record_mean} vs aggregate {aggregate_mean}"
            )
    return checks


def quantile_of(records: Sequence[Dict[str, Any]], getter, quantile: float) -> float:
    values = finite_array(getter(record) for record in records)
    return float(np.quantile(values, quantile)) if values.size else float("nan")


def diagnostic_factors(
    record: Dict[str, Any], records: Sequence[Dict[str, Any]]
) -> List[str]:
    factors = []
    oracle = nested_float(record, "candidate_error_cm", "oracle_per_frame")
    oracle_q75 = quantile_of(records, lambda row: nested_float(row, "candidate_error_cm", "oracle_per_frame"), 0.75)
    fusion_gap = finite_float(record.get("fusion_minus_oracle_candidate_cm"))
    gap_q75 = quantile_of(records, lambda row: row.get("fusion_minus_oracle_candidate_cm"), 0.75)
    gate_match = nested_float(record, "gating", "argmax_matches_oracle_fraction")
    gate_match_q25 = quantile_of(records, lambda row: nested_float(row, "gating", "argmax_matches_oracle_fraction"), 0.25)
    early = nested_float(record, "obj_error_profile_cm", "first_10pct_mean")
    late = nested_float(record, "obj_error_profile_cm", "last_10pct_mean")
    drift = late - early
    drift_q75 = quantile_of(
        records,
        lambda row: nested_float(row, "obj_error_profile_cm", "last_10pct_mean")
        - nested_float(row, "obj_error_profile_cm", "first_10pct_mean"),
        0.75,
    )
    published_delta = finite_float(record.get("published_minus_raw_cm"))
    delta_q75 = quantile_of(records, lambda row: row.get("published_minus_raw_cm"), 0.75)
    obj_error = finite_float(record.get(OBJ_METRIC))
    hoi_error = finite_float(record.get(HOI_METRIC))
    root_error = nested_float(record, "human_error_cm", "root_trans")
    hand_error = nested_float(record, "human_error_cm", "refined_hand")
    root_q75 = quantile_of(records, lambda row: nested_float(row, "human_error_cm", "root_trans"), 0.75)
    hand_q75 = quantile_of(records, lambda row: nested_float(row, "human_error_cm", "refined_hand"), 0.75)
    contact_f1_values = [
        nested_float(record, "contact_prediction", hand_name, "f1")
        for hand_name in ("left", "right")
    ]
    contact_f1_values = [value for value in contact_f1_values if math.isfinite(value)]
    minimum_contact_f1 = min(contact_f1_values) if contact_f1_values else float("nan")
    on_contact_error = finite_float(record.get("obj_error_on_contact_cm"))
    off_contact_error = finite_float(record.get("obj_error_off_contact_cm"))
    no_hand_static_weight = nested_float(
        record, "gating", "mean_weights_by_gt_contact", "no_hand", "static"
    )

    if math.isfinite(oracle) and oracle >= oracle_q75:
        factors.append("candidate trajectories weak")
    if math.isfinite(fusion_gap) and fusion_gap >= gap_q75 and gate_match <= gate_match_q25:
        factors.append("gating misses per-frame best candidate")
    if math.isfinite(drift) and drift >= max(drift_q75, 5.0):
        factors.append("late online drift")
    if math.isfinite(published_delta) and published_delta >= max(delta_q75, 1.0):
        factors.append("published refinement worsens raw fusion")
    if (
        math.isfinite(minimum_contact_f1)
        and minimum_contact_f1 < 0.80
        and math.isfinite(gate_match)
        and gate_match < 0.40
    ):
        factors.append("VC contact errors coincide with gate misrouting")
    if (
        math.isfinite(on_contact_error)
        and math.isfinite(off_contact_error)
        and on_contact_error - off_contact_error >= 5.0
    ):
        factors.append("contact-phase FK/hand-object geometry error")
    if (
        math.isfinite(on_contact_error)
        and math.isfinite(off_contact_error)
        and off_contact_error - on_contact_error >= 5.0
        and math.isfinite(no_hand_static_weight)
        and no_hand_static_weight < 0.50
    ):
        factors.append("no-contact drift with weak static gating")
    if (
        math.isfinite(obj_error) and math.isfinite(hoi_error)
        and obj_error - hoi_error >= 5.0
        and (root_error >= root_q75 or hand_error >= hand_q75)
    ):
        factors.append("shared human/object global error cancels in HOI")
    if math.isfinite(obj_error) and math.isfinite(hoi_error) and hoi_error >= 0.9 * obj_error:
        factors.append("contact-relative geometry remains wrong")
    if not factors:
        factors.append("mixed error; inspect temporal visualization")
    return factors


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "seq_name": record.get("seq_name"),
        "seq_path": record.get("seq_path"),
        "obj_name": record.get("obj_name"),
        "num_frames": record.get("num_frames"),
        "duration_s": record.get("duration_s"),
        "obj_published_cm": record.get(OBJ_METRIC),
        "obj_raw_cm": record.get("obj_trans_err_fusion_raw_cm"),
        "published_minus_raw_cm": record.get("published_minus_raw_cm"),
        "hoi_fusion_cm": record.get(HOI_METRIC),
        "obj_p90_frame_cm": nested_float(record, "obj_error_profile_cm", "p90"),
        "obj_max_frame_cm": nested_float(record, "obj_error_profile_cm", "max"),
        "obj_max_frame_index": nested_float(record, "obj_error_profile_cm", "max_frame"),
        "obj_late_minus_early_cm": (
            nested_float(record, "obj_error_profile_cm", "last_10pct_mean")
            - nested_float(record, "obj_error_profile_cm", "first_10pct_mean")
        ),
        "obj_slope_cm_per_100_frames": nested_float(record, "obj_error_profile_cm", "slope_per_100_frames"),
        "obj_contact_cm": record.get("obj_error_on_contact_cm"),
        "obj_off_contact_cm": record.get("obj_error_off_contact_cm"),
        "candidate_left_fk_cm": nested_float(record, "candidate_error_cm", "left_fk"),
        "candidate_right_fk_cm": nested_float(record, "candidate_error_cm", "right_fk"),
        "candidate_imu_cm": nested_float(record, "candidate_error_cm", "imu"),
        "candidate_static_cm": nested_float(record, "candidate_error_cm", "static"),
        "candidate_oracle_cm": nested_float(record, "candidate_error_cm", "oracle_per_frame"),
        "fusion_minus_oracle_cm": record.get("fusion_minus_oracle_candidate_cm"),
        "gate_oracle_match_fraction": nested_float(record, "gating", "argmax_matches_oracle_fraction"),
        "gate_entropy": nested_float(record, "gating", "entropy"),
        "gate_left_mean": nested_float(record, "gating", "mean_weights", "left_fk"),
        "gate_right_mean": nested_float(record, "gating", "mean_weights", "right_fk"),
        "gate_imu_mean": nested_float(record, "gating", "mean_weights", "imu"),
        "gate_static_mean": nested_float(record, "gating", "mean_weights", "static"),
        "root_error_cm": nested_float(record, "human_error_cm", "root_trans"),
        "hand_error_cm": nested_float(record, "human_error_cm", "refined_hand"),
        "contact_fraction": nested_float(record, "contact_fraction", "any_hand"),
        "contact_left_f1": nested_float(record, "contact_prediction", "left", "f1"),
        "contact_right_f1": nested_float(record, "contact_prediction", "right", "f1"),
        "contact_left_fp_frames": nested_float(
            record, "contact_prediction", "left", "false_positive_frames"
        ),
        "contact_left_fn_frames": nested_float(
            record, "contact_prediction", "left", "false_negative_frames"
        ),
        "contact_right_fp_frames": nested_float(
            record, "contact_prediction", "right", "false_positive_frames"
        ),
        "contact_right_fn_frames": nested_float(
            record, "contact_prediction", "right", "false_negative_frames"
        ),
        "gate_no_hand_static_mean": nested_float(
            record, "gating", "mean_weights_by_gt_contact", "no_hand", "static"
        ),
        "object_path_length_m": nested_float(record, "gt_object_motion", "path_length_m"),
        "object_mean_speed_mps": nested_float(record, "gt_object_motion", "mean_speed_mps"),
    }
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sequence"


def select_and_copy_top(
    records: Sequence[Dict[str, Any]], output_dir: Path, top_k: int, copy_pt: bool
) -> List[Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for metric, label in ((OBJ_METRIC, "top_obj_published"), (HOI_METRIC, "top_hoi_fusion")):
        ranked = sorted(
            (record for record in records if math.isfinite(finite_float(record.get(metric)))),
            key=lambda record: finite_float(record.get(metric)),
            reverse=True,
        )[:top_k]
        for rank, record in enumerate(ranked, start=1):
            key = str(record.get("seq_path") or record.get("seq_name"))
            item = selected.setdefault(key, {"record": record, "selected_for": []})
            item["selected_for"].append({"metric": label, "rank": rank})

    copy_dir = output_dir / "top_sequences_pt"
    if copy_pt:
        copy_dir.mkdir(parents=True, exist_ok=True)
    output = []
    used_names: Dict[str, str] = {}
    for key, item in selected.items():
        record = item["record"]
        source = Path(str(record.get("seq_path", ""))).expanduser()
        destination = None
        copy_status = "disabled"
        if copy_pt:
            copy_status = "missing_source"
            if source.is_file() and source.suffix.lower() == ".pt":
                destination_name = source.name
                previous_source = used_names.get(destination_name)
                if previous_source is not None and previous_source != str(source.resolve()):
                    destination_name = f"{safe_name(str(record.get('seq_name')))}__{source.name}"
                used_names[destination_name] = str(source.resolve())
                destination = copy_dir / destination_name
                shutil.copy2(source, destination)
                copy_status = "copied"
        output.append({
            "seq_name": record.get("seq_name"),
            "obj_name": record.get("obj_name"),
            "source_path": str(source),
            "copied_path": str(destination) if destination is not None else None,
            "copy_status": copy_status,
            "selected_for": item["selected_for"],
            "obj_published_cm": record.get(OBJ_METRIC),
            "hoi_fusion_cm": record.get(HOI_METRIC),
            "diagnostic_factors": diagnostic_factors(record, records),
            "diagnostics": record,
        })
    return sorted(output, key=lambda row: finite_float(row["obj_published_cm"]), reverse=True)


def write_plots(records: Sequence[Dict[str, Any]], output_dir: Path) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    generated = []
    obj_values = finite_array(record.get(OBJ_METRIC) for record in records)
    hoi_values = finite_array(record.get(HOI_METRIC) for record in records)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for values, title, axis, color in (
        (obj_values, "Published object translation error", axes[0, 0], "#176B87"),
        (hoi_values, "HOI fusion error", axes[0, 1], "#B34A2F"),
    ):
        axis.hist(values, bins=40, color=color, alpha=0.85)
        axis.axvline(np.median(values), color="black", linestyle="--", label="median")
        axis.axvline(np.quantile(values, 0.90), color="#D19A00", linestyle=":", label="p90")
        axis.set(title=title, xlabel="Sequence mean error (cm)", ylabel="Sequence count")
        axis.legend()
        sorted_values = np.sort(values)
        ecdf = np.arange(1, sorted_values.size + 1) / sorted_values.size
        ecdf_axis = axes[1, 0] if axis is axes[0, 0] else axes[1, 1]
        ecdf_axis.plot(sorted_values, ecdf, color=color, linewidth=2)
        ecdf_axis.set(title=f"{title} ECDF", xlabel="Sequence mean error (cm)", ylabel="Fraction of sequences")
        ecdf_axis.grid(alpha=0.25)
    fig.tight_layout()
    distribution_path = output_dir / "fusion_error_distributions.png"
    fig.savefig(distribution_path, dpi=180)
    plt.close(fig)
    generated.append(str(distribution_path))

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("obj_name") or "unknown_object")].append(record)
    object_order = sorted(
        grouped,
        key=lambda name: np.nanmedian([finite_float(row.get(OBJ_METRIC)) for row in grouped[name]]),
    )
    fig_height = max(6.0, 0.42 * len(object_order) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, fig_height), sharey=True)
    for axis, metric, title, color in (
        (axes[0], OBJ_METRIC, "Published object error by object", "#176B87"),
        (axes[1], HOI_METRIC, "HOI fusion error by object", "#B34A2F"),
    ):
        data = [finite_array(row.get(metric) for row in grouped[name]) for name in object_order]
        plot = axis.boxplot(data, vert=False, patch_artist=True, labels=object_order, showfliers=True)
        for patch in plot["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set(title=title, xlabel="Sequence mean error (cm)")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    object_path = output_dir / "fusion_error_by_object.png"
    fig.savefig(object_path, dpi=180)
    plt.close(fig)
    generated.append(str(object_path))
    return generated


def format_number(value: Any, digits: int = 3) -> str:
    number = finite_float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NaN"


def write_markdown(
    path: Path,
    payload: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    validation: Dict[str, Any],
    distributions: Dict[str, Any],
    associations: Dict[str, Any],
    object_rows: Sequence[Dict[str, Any]],
    correlation_rows: Dict[str, Any],
    top_rows: Sequence[Dict[str, Any]],
) -> None:
    lines = [
        "# Fusion sequence error analysis",
        "",
        f"- Sequences: {len(records)}",
        f"- Protocol: `{json.dumps(payload.get('protocol', {}), ensure_ascii=False)}`",
        "- Aggregation: each sequence contributes one equally weighted mean error.",
        "",
        "## Aggregate validation",
        "",
        "| Metric | Records | Record mean | Eval mean | Absolute difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, check in validation.items():
        lines.append(
            f"| {metric} | {check['record_count']} | {format_number(check['record_mean'], 6)} | "
            f"{format_number(check['aggregate_mean'], 6)} | {format_number(check['absolute_difference'], 9)} |"
        )
    lines.extend(["", "## Sequence distributions", ""])
    for metric, summary in distributions.items():
        lines.extend([
            f"### {metric}",
            "",
            f"Count={summary['count']}, mean={format_number(summary.get('mean'))} cm, "
            f"median={format_number(summary.get('median'))} cm, p90={format_number(summary.get('p90'))} cm, "
            f"p95={format_number(summary.get('p95'))} cm, max={format_number(summary.get('max'))} cm.",
            "",
            "| Error bin (cm) | Sequences | Percentage |",
            "|---|---:|---:|",
        ])
        for row in summary.get("histogram", []):
            lines.append(f"| {row['bin_cm']} | {row['count']} | {row['percentage']:.2f}% |")
        lines.append("")

    lines.extend([
        "## Object-class association",
        "",
        "Eta-squared is the fraction of sequence-error variance explained by object labels. "
        "The permutation p-value tests association, not causation; action, subject, motion, and sequence length can be confounded with object class.",
        "",
        "| Metric | Eta-squared | Permutation p | Permutations |",
        "|---|---:|---:|---:|",
    ])
    for metric, result in associations.items():
        lines.append(
            f"| {metric} | {format_number(result['eta_squared'], 4)} | "
            f"{format_number(result['permutation_p'], 6)} | {result['permutations']} |"
        )
    lines.extend([
        "",
        "| Object | N | Obj mean | Obj median | Obj p90 | HOI mean | HOI median | HOI p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in object_rows:
        lines.append(
            f"| {row['obj_name']} | {row['sequence_count']} | {format_number(row['obj_mean_cm'])} | "
            f"{format_number(row['obj_median_cm'])} | {format_number(row['obj_p90_cm'])} | "
            f"{format_number(row['hoi_mean_cm'])} | {format_number(row['hoi_median_cm'])} | "
            f"{format_number(row['hoi_p90_cm'])} |"
        )

    lines.extend([
        "",
        "## Numeric associations",
        "",
        "| Factor | Pearson r with obj error | Spearman r with obj error | Pearson r with HOI | Spearman r with HOI |",
        "|---|---:|---:|---:|---:|",
    ])
    obj_corr = correlation_rows[OBJ_METRIC]
    hoi_corr = correlation_rows[HOI_METRIC]
    for factor in obj_corr:
        lines.append(
            f"| {factor} | {format_number(obj_corr[factor]['pearson_r'])} | "
            f"{format_number(obj_corr[factor]['spearman_r'])} | "
            f"{format_number(hoi_corr[factor]['pearson_r'])} | "
            f"{format_number(hoi_corr[factor]['spearman_r'])} |"
        )

    lines.extend([
        "",
        "## Highest-error sequences",
        "",
        "The factors below are diagnostic signals, not proof of a single causal mechanism.",
        "",
        "| Sequence | Object | Selected for | Obj cm | HOI cm | Oracle cm | Gate match | Late-early cm | Factors |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in top_rows:
        record = row["diagnostics"]
        selection = ", ".join(
            f"{item['metric']} #{item['rank']}" for item in row["selected_for"]
        )
        late_early = (
            nested_float(record, "obj_error_profile_cm", "last_10pct_mean")
            - nested_float(record, "obj_error_profile_cm", "first_10pct_mean")
        )
        lines.append(
            f"| {row['seq_name']} | {row['obj_name']} | {selection} | "
            f"{format_number(row['obj_published_cm'])} | {format_number(row['hoi_fusion_cm'])} | "
            f"{format_number(nested_float(record, 'candidate_error_cm', 'oracle_per_frame'))} | "
            f"{format_number(nested_float(record, 'gating', 'argmax_matches_oracle_fraction'))} | "
            f"{format_number(late_early)} | {'; '.join(row['diagnostic_factors'])} |"
        )
    lines.extend([
        "",
        f"Copied PT directory: `{path.parent / 'top_sequences_pt'}`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    records = payload.get("datasets", {}).get(args.dataset)
    if not isinstance(records, list) or not records:
        raise ValueError(f"No sequence records found for dataset {args.dataset!r}")
    if args.diagnostic_override_json is not None:
        override_payload = json.loads(
            args.diagnostic_override_json.read_text(encoding="utf-8")
        )
        override_records = override_payload.get("datasets", {}).get(args.dataset)
        if not isinstance(override_records, list) or not override_records:
            raise ValueError(
                f"No override records found for dataset {args.dataset!r}"
            )
        overrides_by_path = {record_key(record): record for record in override_records}
        overrides_by_name = {
            str(record.get("seq_name") or ""): record for record in override_records
        }
        merged_records = []
        for record in records:
            override = overrides_by_path.get(
                record_key(record),
                overrides_by_name.get(str(record.get("seq_name") or "")),
            )
            if override is None:
                merged_records.append(record)
                continue
            merged = dict(record)
            merged["contact_prediction"] = override.get("contact_prediction", {})
            merged_gating = dict(record.get("gating") or {})
            override_gating = override.get("gating") or {}
            if "mean_weights_by_gt_contact" in override_gating:
                merged_gating["mean_weights_by_gt_contact"] = override_gating[
                    "mean_weights_by_gt_contact"
                ]
            merged["gating"] = merged_gating
            merged_records.append(merged)
        records = merged_records

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_aggregate_means(payload, args.dataset, records)
    distributions = {
        OBJ_METRIC: distribution_summary(record.get(OBJ_METRIC) for record in records),
        HOI_METRIC: distribution_summary(record.get(HOI_METRIC) for record in records),
    }

    rng = np.random.default_rng(args.seed)
    associations = {}
    labels = np.asarray([str(record.get("obj_name") or "unknown_object") for record in records])
    for metric in (OBJ_METRIC, HOI_METRIC):
        values = np.asarray([finite_float(record.get(metric)) for record in records], dtype=np.float64)
        valid = np.isfinite(values)
        associations[metric] = permutation_test_eta_squared(
            values[valid], labels[valid], args.permutations, rng
        )

    object_rows = object_summaries(records)
    correlation_rows = {
        metric: correlations(records, metric) for metric in (OBJ_METRIC, HOI_METRIC)
    }
    top_rows = select_and_copy_top(
        records, args.output_dir, args.top_k, copy_pt=not args.no_copy_pt
    )
    (args.output_dir / "top_sequence_source_paths.txt").write_text(
        "\n".join(str(row["source_path"]) for row in top_rows) + "\n",
        encoding="utf-8",
    )
    plot_paths = write_plots(records, args.output_dir)

    write_csv(args.output_dir / "sequence_metrics.csv", [flatten_record(record) for record in records])
    write_csv(args.output_dir / "object_summary.csv", object_rows)
    top_csv_rows = []
    for row in top_rows:
        flat = flatten_record(row["diagnostics"])
        flat["selected_for"] = "; ".join(
            f"{item['metric']}#{item['rank']}" for item in row["selected_for"]
        )
        flat["diagnostic_factors"] = "; ".join(row["diagnostic_factors"])
        flat["copied_path"] = row["copied_path"]
        top_csv_rows.append(flat)
    write_csv(args.output_dir / "top_sequences.csv", top_csv_rows)

    analysis = {
        "input_json": str(args.input_json.resolve()),
        "diagnostic_override_json": (
            str(args.diagnostic_override_json.resolve())
            if args.diagnostic_override_json is not None else None
        ),
        "dataset": args.dataset,
        "protocol": payload.get("protocol", {}),
        "aggregate_validation": validation,
        "distributions": distributions,
        "object_class_association": associations,
        "object_summary": object_rows,
        "correlations": correlation_rows,
        "top_sequences": top_rows,
        "plots": plot_paths,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(
        args.output_dir / "analysis.md",
        payload,
        records,
        validation,
        distributions,
        associations,
        object_rows,
        correlation_rows,
        top_rows,
    )
    print(f"Analyzed {len(records)} sequences across {len(object_rows)} object classes.")
    print(f"Selected {len(top_rows)} unique top-error sequences.")
    print(f"Results written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())