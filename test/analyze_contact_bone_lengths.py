#!/usr/bin/env python3
"""Analyze hand-object contact bone-length distributions for processed IMUHOI data.

The "bone length" here follows dataset/dataset_IMUHOI.py: distance from the
object translation to the left/right virtual palm anchor, evaluated only on the
corresponding hand-contact frames.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_IMUHOI import suppress_bimanual_contact_by_bone_variation  # noqa: E402
from utils.human_pose import select_hand_anchor_positions  # noqa: E402


def _as_bool_1d(value, length: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(length, dtype=torch.bool)
    tensor = torch.as_tensor(value)
    if tensor.dim() > 1:
        tensor = tensor.reshape(tensor.shape[0], -1)[:, 0]
    return tensor[:length].bool()


def _as_float_tensor(value, length: int | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu().float()
    if length is not None:
        tensor = tensor[:length]
    return tensor


def _finite_np(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _quantile(values: Sequence[float], q: float, default: float = float("nan")) -> float:
    arr = _finite_np(values)
    if arr.size == 0:
        return default
    return float(np.quantile(arr, q))


def _mean(values: Sequence[float], default: float = float("nan")) -> float:
    arr = _finite_np(values)
    if arr.size == 0:
        return default
    return float(arr.mean())


def _std(values: Sequence[float], default: float = float("nan")) -> float:
    arr = _finite_np(values)
    if arr.size == 0:
        return default
    return float(arr.std(ddof=0))


def _safe_min(values: Sequence[float], default: float = float("nan")) -> float:
    arr = _finite_np(values)
    if arr.size == 0:
        return default
    return float(arr.min())


def _safe_max(values: Sequence[float], default: float = float("nan")) -> float:
    arr = _finite_np(values)
    if arr.size == 0:
        return default
    return float(arr.max())


def _find_segments(mask: torch.Tensor) -> List[Tuple[int, int]]:
    indices = torch.where(mask.bool())[0]
    if indices.numel() == 0:
        return []
    starts_ends: List[Tuple[int, int]] = []
    start = int(indices[0])
    end = start
    for idx_tensor in indices[1:]:
        idx = int(idx_tensor)
        if idx == end + 1:
            end = idx
        else:
            starts_ends.append((start, end))
            start = idx
            end = idx
    starts_ends.append((start, end))
    return starts_ends


def _sequence_id(path: Path, data: Dict) -> str:
    return str(data.get("seq_name") or path.stem)


def _has_object(data: Dict) -> bool:
    if "has_object" in data:
        return bool(data["has_object"])
    return data.get("obj_trans") is not None


def _object_extent(data: Dict, length: int) -> Tuple[float, float, float, float]:
    points = _as_float_tensor(data.get("obj_points_canonical"))
    if points is None or points.dim() != 2 or points.shape[-1] != 3 or points.shape[0] == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    extent = points.max(dim=0).values - points.min(dim=0).values
    scale = _as_float_tensor(data.get("obj_scale"), length)
    if scale is not None and scale.numel() > 0:
        extent = extent * float(torch.median(scale.reshape(-1)))
    extent_np = extent.numpy().astype(np.float64)
    min_extent = float(np.min(extent_np))
    max_extent = float(np.max(extent_np))
    aspect = max_extent / max(min_extent, 1e-8)
    return (float(extent_np[0]), float(extent_np[1]), float(extent_np[2]), float(aspect))


def collect_observations(data_dir: Path, resolve_bimanual_contact_conflicts: bool = True):
    observations: List[Dict] = []
    sequence_rows: List[Dict] = []
    segment_rows: List[Dict] = []
    skipped_rows: List[Dict] = []

    paths = sorted(data_dir.glob("*.pt"))
    for path in paths:
        try:
            data = torch.load(path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - diagnostic path
            skipped_rows.append({"seq_file": path.name, "reason": f"load_error: {exc}"})
            continue

        if not isinstance(data, dict):
            skipped_rows.append({"seq_file": path.name, "reason": "not_dict"})
            continue
        if not _has_object(data):
            skipped_rows.append({"seq_file": path.name, "reason": "no_object"})
            continue

        pos = _as_float_tensor(data.get("position_global_full_gt_world"))
        obj_trans = _as_float_tensor(data.get("obj_trans"))
        if pos is None or obj_trans is None:
            skipped_rows.append({"seq_file": path.name, "reason": "missing_position_or_obj_trans"})
            continue
        if obj_trans.dim() == 3 and obj_trans.shape[-1] == 1:
            obj_trans = obj_trans.squeeze(-1)
        if obj_trans.dim() != 2 or obj_trans.shape[-1] != 3:
            skipped_rows.append({"seq_file": path.name, "reason": f"bad_obj_trans_shape_{tuple(obj_trans.shape)}"})
            continue
        if pos.dim() != 3 or pos.shape[-1] != 3:
            skipped_rows.append({"seq_file": path.name, "reason": f"bad_position_shape_{tuple(pos.shape)}"})
            continue

        length = int(min(pos.shape[0], obj_trans.shape[0]))
        if length <= 0:
            skipped_rows.append({"seq_file": path.name, "reason": "empty_sequence"})
            continue
        pos = pos[:length]
        obj_trans = obj_trans[:length]
        l_contact = _as_bool_1d(data.get("lhand_contact"), length)
        r_contact = _as_bool_1d(data.get("rhand_contact"), length)
        raw_l_contact = l_contact.clone()
        raw_r_contact = r_contact.clone()
        raw_contact_union = raw_l_contact | raw_r_contact
        if not bool(raw_contact_union.any()):
            skipped_rows.append({"seq_file": path.name, "reason": "no_hand_contact"})
            continue

        try:
            hand_anchor = select_hand_anchor_positions(pos)
        except Exception as exc:  # pragma: no cover - diagnostic path
            skipped_rows.append({"seq_file": path.name, "reason": f"anchor_error: {exc}"})
            continue

        l_lengths = torch.linalg.norm(obj_trans - hand_anchor[:, 0, :], dim=-1)
        r_lengths = torch.linalg.norm(obj_trans - hand_anchor[:, 1, :], dim=-1)
        if resolve_bimanual_contact_conflicts:
            l_contact, r_contact = suppress_bimanual_contact_by_bone_variation(
                l_contact,
                r_contact,
                l_lengths,
                r_lengths,
            )
        dropped_left = int((raw_l_contact & ~l_contact).sum())
        dropped_right = int((raw_r_contact & ~r_contact).sum())
        contact_union = l_contact | r_contact
        if not bool(contact_union.any()):
            skipped_rows.append({"seq_file": path.name, "reason": "no_hand_contact_after_postprocess"})
            continue

        seq_name = _sequence_id(path, data)
        obj_name = str(data.get("obj_name") or "unknown_object")

        seq_values: List[float] = []
        for hand, mask, lengths in (("left", l_contact, l_lengths), ("right", r_contact, r_lengths)):
            frame_indices = torch.where(mask)[0]
            for frame_tensor in frame_indices:
                frame = int(frame_tensor)
                value = float(lengths[frame])
                if math.isfinite(value):
                    observations.append(
                        {
                            "seq_file": path.name,
                            "seq_name": seq_name,
                            "obj_name": obj_name,
                            "hand": hand,
                            "frame": frame,
                            "bone_length_m": value,
                        }
                    )
                    seq_values.append(value)

        if not seq_values:
            skipped_rows.append({"seq_file": path.name, "reason": "no_finite_contact_length"})
            continue

        seq_arr = _finite_np(seq_values)
        extent_x, extent_y, extent_z, aspect = _object_extent(data, length)
        segment_ranges = []
        segment_stds = []
        for seg_idx, (start, end) in enumerate(_find_segments(contact_union)):
            seg_values: List[float] = []
            for frame in range(start, end + 1):
                if bool(l_contact[frame]):
                    seg_values.append(float(l_lengths[frame]))
                if bool(r_contact[frame]):
                    seg_values.append(float(r_lengths[frame]))
            seg_arr = _finite_np(seg_values)
            if seg_arr.size == 0:
                continue
            seg_range = float(seg_arr.max() - seg_arr.min()) if seg_arr.size > 1 else 0.0
            seg_std = float(seg_arr.std(ddof=0)) if seg_arr.size > 1 else 0.0
            segment_ranges.append(seg_range)
            segment_stds.append(seg_std)
            segment_rows.append(
                {
                    "seq_file": path.name,
                    "seq_name": seq_name,
                    "obj_name": obj_name,
                    "segment_index": seg_idx,
                    "start_frame": start,
                    "end_frame": end,
                    "contact_frames": int(end - start + 1),
                    "contact_observations": int(seg_arr.size),
                    "mean_m": float(seg_arr.mean()),
                    "std_m": seg_std,
                    "min_m": float(seg_arr.min()),
                    "q25_m": float(np.quantile(seg_arr, 0.25)),
                    "median_m": float(np.quantile(seg_arr, 0.50)),
                    "q75_m": float(np.quantile(seg_arr, 0.75)),
                    "max_m": float(seg_arr.max()),
                    "range_m": seg_range,
                }
            )

        sequence_rows.append(
            {
                "seq_file": path.name,
                "seq_name": seq_name,
                "obj_name": obj_name,
                "seq_len": length,
                "raw_lhand_contact_frames": int(raw_l_contact.sum()),
                "raw_rhand_contact_frames": int(raw_r_contact.sum()),
                "raw_both_hand_contact_frames": int((raw_l_contact & raw_r_contact).sum()),
                "dropped_lhand_contact_frames": dropped_left,
                "dropped_rhand_contact_frames": dropped_right,
                "lhand_contact_frames": int(l_contact.sum()),
                "rhand_contact_frames": int(r_contact.sum()),
                "both_hand_contact_frames": int((l_contact & r_contact).sum()),
                "union_contact_frames": int(contact_union.sum()),
                "contact_observations": int(seq_arr.size),
                "contact_segments": len(segment_ranges),
                "mean_m": float(seq_arr.mean()),
                "std_m": float(seq_arr.std(ddof=0)) if seq_arr.size > 1 else 0.0,
                "min_m": float(seq_arr.min()),
                "q25_m": float(np.quantile(seq_arr, 0.25)),
                "median_m": float(np.quantile(seq_arr, 0.50)),
                "q75_m": float(np.quantile(seq_arr, 0.75)),
                "max_m": float(seq_arr.max()),
                "range_m": float(seq_arr.max() - seq_arr.min()) if seq_arr.size > 1 else 0.0,
                "iqr_m": float(np.quantile(seq_arr, 0.75) - np.quantile(seq_arr, 0.25)),
                "mean_segment_range_m": _mean(segment_ranges, default=0.0),
                "median_segment_range_m": _quantile(segment_ranges, 0.50, default=0.0),
                "max_segment_range_m": _safe_max(segment_ranges, default=0.0),
                "mean_segment_std_m": _mean(segment_stds, default=0.0),
                "obj_extent_x_m": extent_x,
                "obj_extent_y_m": extent_y,
                "obj_extent_z_m": extent_z,
                "obj_aspect_ratio": aspect,
            }
        )

    return paths, observations, sequence_rows, segment_rows, skipped_rows


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_histograms(observations: Sequence[Dict], bin_width: float):
    values = np.asarray([row["bone_length_m"] for row in observations], dtype=np.float64)
    if values.size == 0:
        raise ValueError("No contact bone-length observations found.")
    min_edge = math.floor(float(values.min()) / bin_width) * bin_width
    max_edge = math.ceil(float(values.max()) / bin_width) * bin_width
    bins = np.arange(min_edge, max_edge + bin_width * 1.5, bin_width, dtype=np.float64)
    if bins.size < 2:
        bins = np.asarray([min_edge, min_edge + bin_width], dtype=np.float64)

    by_object: Dict[str, List[float]] = defaultdict(list)
    for row in observations:
        by_object[row["obj_name"]].append(float(row["bone_length_m"]))

    long_rows: List[Dict] = []
    wide_rows: List[Dict] = []
    centers = (bins[:-1] + bins[1:]) / 2.0
    object_names = sorted(by_object.keys())
    counts_by_object: Dict[str, np.ndarray] = {}
    for obj_name in object_names:
        counts, _ = np.histogram(np.asarray(by_object[obj_name], dtype=np.float64), bins=bins)
        counts_by_object[obj_name] = counts
        for left, right, center, count in zip(bins[:-1], bins[1:], centers, counts):
            long_rows.append(
                {
                    "obj_name": obj_name,
                    "bin_left_m": float(left),
                    "bin_right_m": float(right),
                    "bin_center_m": float(center),
                    "contact_observation_count": int(count),
                }
            )

    for bin_idx, (left, right, center) in enumerate(zip(bins[:-1], bins[1:], centers)):
        row = {
            "bin_left_m": float(left),
            "bin_right_m": float(right),
            "bin_center_m": float(center),
        }
        for obj_name in object_names:
            row[obj_name] = int(counts_by_object[obj_name][bin_idx])
        wide_rows.append(row)

    return bins, centers, by_object, counts_by_object, long_rows, wide_rows


def object_summary(sequence_rows: Sequence[Dict], observations: Sequence[Dict]) -> List[Dict]:
    obs_by_obj: Dict[str, List[float]] = defaultdict(list)
    for row in observations:
        obs_by_obj[row["obj_name"]].append(float(row["bone_length_m"]))

    seq_by_obj: Dict[str, List[Dict]] = defaultdict(list)
    for row in sequence_rows:
        seq_by_obj[row["obj_name"]].append(row)

    rows: List[Dict] = []
    for obj_name in sorted(obs_by_obj.keys()):
        values = _finite_np(obs_by_obj[obj_name])
        seqs = seq_by_obj.get(obj_name, [])
        seq_medians = [float(row["median_m"]) for row in seqs]
        seq_means = [float(row["mean_m"]) for row in seqs]
        seq_ranges = [float(row["range_m"]) for row in seqs]
        segment_ranges = [float(row["median_segment_range_m"]) for row in seqs]
        aspects = [float(row["obj_aspect_ratio"]) for row in seqs]
        ex = [float(row["obj_extent_x_m"]) for row in seqs]
        ey = [float(row["obj_extent_y_m"]) for row in seqs]
        ez = [float(row["obj_extent_z_m"]) for row in seqs]
        rows.append(
            {
                "obj_name": obj_name,
                "sequence_count": len(seqs),
                "raw_lhand_contact_frames": int(sum(int(row.get("raw_lhand_contact_frames", 0)) for row in seqs)),
                "raw_rhand_contact_frames": int(sum(int(row.get("raw_rhand_contact_frames", 0)) for row in seqs)),
                "raw_both_hand_contact_frames": int(sum(int(row.get("raw_both_hand_contact_frames", 0)) for row in seqs)),
                "dropped_lhand_contact_frames": int(sum(int(row.get("dropped_lhand_contact_frames", 0)) for row in seqs)),
                "dropped_rhand_contact_frames": int(sum(int(row.get("dropped_rhand_contact_frames", 0)) for row in seqs)),
                "contact_observations": int(values.size),
                "mean_m": float(values.mean()),
                "std_m": float(values.std(ddof=0)),
                "min_m": float(values.min()),
                "q05_m": float(np.quantile(values, 0.05)),
                "q25_m": float(np.quantile(values, 0.25)),
                "median_m": float(np.quantile(values, 0.50)),
                "q75_m": float(np.quantile(values, 0.75)),
                "q95_m": float(np.quantile(values, 0.95)),
                "max_m": float(values.max()),
                "seq_median_mean_m": _mean(seq_medians),
                "seq_median_std_m": _std(seq_medians),
                "seq_median_min_m": _safe_min(seq_medians),
                "seq_median_q25_m": _quantile(seq_medians, 0.25),
                "seq_median_median_m": _quantile(seq_medians, 0.50),
                "seq_median_q75_m": _quantile(seq_medians, 0.75),
                "seq_median_max_m": _safe_max(seq_medians),
                "seq_median_range_m": _safe_max(seq_medians) - _safe_min(seq_medians),
                "seq_mean_std_m": _std(seq_means),
                "within_seq_range_mean_m": _mean(seq_ranges),
                "within_seq_range_median_m": _quantile(seq_ranges, 0.50),
                "within_seq_range_q75_m": _quantile(seq_ranges, 0.75),
                "within_seq_range_q95_m": _quantile(seq_ranges, 0.95),
                "within_seq_range_max_m": _safe_max(seq_ranges),
                "median_segment_range_median_m": _quantile(segment_ranges, 0.50),
                "obj_extent_x_median_m": _quantile(ex, 0.50),
                "obj_extent_y_median_m": _quantile(ey, 0.50),
                "obj_extent_z_median_m": _quantile(ez, 0.50),
                "obj_aspect_ratio_median": _quantile(aspects, 0.50),
            }
        )
    rows.sort(key=lambda row: row["contact_observations"], reverse=True)
    return rows


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: Sequence[Dict], columns: Sequence[Tuple[str, str]], limit: int | None = None) -> str:
    shown = list(rows[:limit] if limit is not None else rows)
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in shown:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(_fmt(value, 3))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = _finite_np(xs)
    y = _finite_np(ys)
    if x.size != y.size or x.size < 2:
        return float("nan")
    if float(x.std(ddof=0)) == 0.0 or float(y.std(ddof=0)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _smooth_count_curve(counts: np.ndarray, sigma_bins: float = 1.5) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return counts
    if sigma_bins <= 0:
        return counts
    radius = max(1, int(math.ceil(4.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / float(sigma_bins)) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(counts, (radius, radius), mode="constant", constant_values=0.0)
    return np.convolve(padded, kernel, mode="valid")


def write_analysis(
    path: Path,
    data_dir: Path,
    output_dir: Path,
    total_files: int,
    observations: Sequence[Dict],
    sequence_rows: Sequence[Dict],
    segment_rows: Sequence[Dict],
    object_rows: Sequence[Dict],
    skipped_rows: Sequence[Dict],
    bin_width: float,
    resolve_bimanual_contact_conflicts: bool,
) -> None:
    values = _finite_np([row["bone_length_m"] for row in observations])
    seq_ranges = _finite_np([row["range_m"] for row in sequence_rows])
    seq_stds = _finite_np([row["std_m"] for row in sequence_rows])
    segment_ranges = _finite_np([row["range_m"] for row in segment_rows])

    high_within = sorted(object_rows, key=lambda row: row["within_seq_range_median_m"], reverse=True)
    high_between = sorted(object_rows, key=lambda row: row["seq_median_range_m"], reverse=True)
    high_q95 = sorted(object_rows, key=lambda row: row["q95_m"], reverse=True)
    low_q95 = sorted(object_rows, key=lambda row: row["q95_m"])

    aspect_values = [row["obj_aspect_ratio_median"] for row in object_rows]
    q95_values = [row["q95_m"] for row in object_rows]
    within_values = [row["within_seq_range_median_m"] for row in object_rows]
    between_values = [row["seq_median_range_m"] for row in object_rows]
    aspect_q95_corr = _pearson(aspect_values, q95_values)
    aspect_within_corr = _pearson(aspect_values, within_values)
    aspect_between_corr = _pearson(aspect_values, between_values)

    le_10 = float(np.mean(seq_ranges <= 0.10)) if seq_ranges.size else float("nan")
    le_20 = float(np.mean(seq_ranges <= 0.20)) if seq_ranges.size else float("nan")
    le_30 = float(np.mean(seq_ranges <= 0.30)) if seq_ranges.size else float("nan")
    raw_bimanual_frames = int(sum(int(row.get("raw_both_hand_contact_frames", 0)) for row in sequence_rows))
    dropped_left_frames = int(sum(int(row.get("dropped_lhand_contact_frames", 0)) for row in sequence_rows))
    dropped_right_frames = int(sum(int(row.get("dropped_rhand_contact_frames", 0)) for row in sequence_rows))
    dropped_total_frames = dropped_left_frames + dropped_right_frames

    lines = [
        "# OMOMO BPS 测试集接触骨长分析",
        "",
        "## 方法",
        "",
        (
            "- 骨长按 `||obj_trans - virtual_palm_anchor||` 计算，单位是米；"
            "这和 `dataset/dataset_IMUHOI.py` 中 `lhand_lb/rhand_lb` 的定义一致。"
        ),
        (
            "- 统计单位是左/右手接触观测；如果同一帧双手都接触，该帧会贡献两个骨长观测。"
        ),
        (
            f"- `resolve_bimanual_contact_conflicts` 后处理：{'开启' if resolve_bimanual_contact_conflicts else '关闭'}。"
            "开启时，对双手同时接触的连续片段，清掉段内骨长变化更大的那只手的接触标签。"
        ),
        f"- 数据目录：`{data_dir}`。",
        f"- 平滑曲线的计数 bin 宽度：`{bin_width:.3f} m`。",
        "",
        "## 输出文件",
        "",
        f"- `histogram_by_object_wide.csv`：x 轴 bin center，每个物体一列 y 计数。",
        f"- `histogram_by_object_long.csv`：适合重新画图的 long-form 表。",
        f"- `contact_bone_length_histogram.png`：按物体上色的平滑计数曲线图。",
        f"- `contact_bone_length_smooth_curves.png`：同一张平滑曲线图的语义化文件名。",
        f"- `object_summary.csv`：物体级分布和跨序列变化。",
        f"- `sequence_summary.csv`：每条序列内的接触骨长变化。",
        f"- `segment_summary.csv`：连续接触段内的变化。",
        f"- `contact_bone_observations.csv`：左/右手接触帧原始观测。",
        "",
        "## 数据概况",
        "",
        f"- 扫描 `.pt` 文件数：{total_files}",
        f"- 有有效手部接触骨长的序列数：{len(sequence_rows)}",
        f"- 有接触观测的物体类别数：{len(object_rows)}",
        f"- 接触骨长观测数：{len(observations)}",
        f"- 跳过文件数：{len(skipped_rows)}",
        (
            f"- 原始双手同时接触帧数：{raw_bimanual_frames}；后处理清掉左手 {dropped_left_frames} 帧，"
            f"清掉右手 {dropped_right_frames} 帧，总计减少 {dropped_total_frames} 个左/右手接触观测。"
        ),
        (
            f"- 全局骨长：均值 {_fmt(float(values.mean()))} m，std {_fmt(float(values.std(ddof=0)))} m，"
            f"中位数 {_fmt(float(np.quantile(values, 0.50)))} m，"
            f"5-95% 区间 [{_fmt(float(np.quantile(values, 0.05)))}, {_fmt(float(np.quantile(values, 0.95)))}] m，"
            f"min-max [{_fmt(float(values.min()))}, {_fmt(float(values.max()))}] m。"
        ),
        "",
        "## 1. 同一条序列中接触帧骨长变化",
        "",
        (
            f"对每条序列，把所有左/右手接触观测合在一起看，序列内 range 的中位数是 "
            f"{_fmt(float(np.quantile(seq_ranges, 0.50)))} m，均值 {_fmt(float(seq_ranges.mean()))} m，"
            f"75% 分位 {_fmt(float(np.quantile(seq_ranges, 0.75)))} m，"
            f"95% 分位 {_fmt(float(np.quantile(seq_ranges, 0.95)))} m，"
            f"最大 {_fmt(float(seq_ranges.max()))} m。"
        ),
        (
            f"序列内 std 的中位数是 {_fmt(float(np.quantile(seq_stds, 0.50)))} m，"
            f"95% 分位是 {_fmt(float(np.quantile(seq_stds, 0.95)))} m。"
        ),
        (
            f"按 range 阈值看，{le_10:.1%} 的序列在 0.10 m 内，"
            f"{le_20:.1%} 在 0.20 m 内，{le_30:.1%} 在 0.30 m 内。"
        ),
        (
            f"如果只看连续接触段，段内 range 中位数是 {_fmt(float(np.quantile(segment_ranges, 0.50)))} m，"
            f"95% 分位是 {_fmt(float(np.quantile(segment_ranges, 0.95)))} m。"
            "因此单个连续接触段里通常比较稳定，整条序列的较大变化更多来自不同接触段、左右手切换或接触位置变化。"
        ),
        "",
        "序列内 range 中位数最大的物体：",
        "",
        _markdown_table(
            high_within,
            [
                ("obj_name", "物体"),
                ("sequence_count", "序列数"),
                ("within_seq_range_median_m", "序列 range 中位数 m"),
                ("within_seq_range_q95_m", "序列 range q95 m"),
                ("q95_m", "物体 q95 m"),
            ],
            limit=6,
        ),
        "",
        "## 2. 同一个物体在不同序列中的骨长变化",
        "",
        (
            "这里先取每条序列的骨长中位数，再在同一物体内部比较这些序列中位数。"
            "这样不会让长序列因为帧数更多而主导结论。"
        ),
        "",
        "跨序列中位数 range 最大的物体：",
        "",
        _markdown_table(
            high_between,
            [
                ("obj_name", "物体"),
                ("sequence_count", "序列数"),
                ("seq_median_range_m", "序列中位数 range m"),
                ("seq_median_std_m", "序列中位数 std m"),
                ("seq_median_q25_m", "序列中位数 q25 m"),
                ("seq_median_q75_m", "序列中位数 q75 m"),
            ],
            limit=8,
        ),
        "",
        "95% 骨长较短/上尾更紧的物体：",
        "",
        _markdown_table(
            low_q95,
            [
                ("obj_name", "物体"),
                ("sequence_count", "序列数"),
                ("median_m", "中位数 m"),
                ("q95_m", "q95 m"),
                ("within_seq_range_median_m", "序列 range 中位数 m"),
            ],
            limit=6,
        ),
        "",
        "95% 骨长最长/上尾更长的物体：",
        "",
        _markdown_table(
            high_q95,
            [
                ("obj_name", "物体"),
                ("sequence_count", "序列数"),
                ("median_m", "中位数 m"),
                ("q95_m", "q95 m"),
                ("within_seq_range_median_m", "序列 range 中位数 m"),
            ],
            limit=6,
        ),
        "",
        "## 3. 骨长分布与物体形状、交互姿态的关系",
        "",
        (
            "关系是存在的，但不是纯粹的几何尺寸决定。这里的骨长是物体中心到虚拟掌根的距离，"
            "因此同时受物体大小/形状、接触的是哪一侧、左右手、人体姿态、以及物体中心是否远离真实接触点影响。"
        ),
        (
            f"用 `obj_points_canonical` 的 bbox 长宽比作为粗略形状指标，物体级 Pearson r 分别为："
            f"与物体 q95 骨长 {_fmt(aspect_q95_corr)}，与序列内 range 中位数 {_fmt(aspect_within_corr)}，"
            f"与跨序列中位数 range {_fmt(aspect_between_corr)}。只有 {len(object_rows)} 类物体，"
            "这些数值只能当描述性信号，不能当因果证明。"
        ),
        (
            "定性看，clothesstand、tripod、mop、vacuum、floorlamp 这类细长或可多位置接触的物体，"
            "更容易出现长尾或更大的跨序列变化；trashcan、smallbox、plasticbox 这类较紧凑容器整体更集中。"
            "桌子、箱子、椅子这类宽面物体不一定长宽比很大，但物体中心可能离抓握/接触面较远，因此也会落在中等到偏大的骨长范围。"
        ),
        (
            "交互姿态也明显重要：跨序列差异最大的物体并不只是尺寸最大的物体，"
            "还包括人可以接触不同端点/高度、或在单手和双手交互之间切换的物体。"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(
    output_dir: Path,
    centers: np.ndarray,
    counts_by_object: Dict[str, np.ndarray],
    object_rows: Sequence[Dict],
    sequence_rows: Sequence[Dict],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        (output_dir / "plot_error.txt").write_text(str(exc), encoding="utf-8")
        return

    sorted_objects = [row["obj_name"] for row in object_rows]
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(14, 8))
    for idx, obj_name in enumerate(sorted_objects):
        counts = counts_by_object[obj_name]
        smooth_counts = _smooth_count_curve(counts, sigma_bins=1.5)
        dense_x = np.linspace(float(centers[0]), float(centers[-1]), max(400, centers.size * 8))
        dense_y = np.interp(dense_x, centers, smooth_counts)
        ax.plot(
            dense_x,
            dense_y,
            linewidth=2.0,
            color=cmap(idx % 20),
            label=f"{obj_name} (n={int(counts.sum())})",
        )
    ax.set_title("Smoothed contact bone-length count curves by object")
    ax.set_xlabel("Bone length: ||object translation - virtual palm|| (m)")
    ax.set_ylabel("Smoothed contact observation count")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "contact_bone_length_histogram.png", dpi=180)
    fig.savefig(output_dir / "contact_bone_length_smooth_curves.png", dpi=180)
    plt.close(fig)

    seq_by_object: Dict[str, List[float]] = defaultdict(list)
    range_by_object: Dict[str, List[float]] = defaultdict(list)
    for row in sequence_rows:
        seq_by_object[row["obj_name"]].append(float(row["median_m"]))
        range_by_object[row["obj_name"]].append(float(row["range_m"]))

    labels = sorted_objects
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot([seq_by_object[obj] for obj in labels], labels=labels, showfliers=False)
    ax.set_title("Sequence median contact bone length by object")
    ax.set_xlabel("Object")
    ax.set_ylabel("Sequence median bone length (m)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "sequence_median_by_object.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot([range_by_object[obj] for obj in labels], labels=labels, showfliers=False)
    ax.set_title("Within-sequence contact bone-length range by object")
    ax.set_xlabel("Object")
    ax.set_ylabel("Per-sequence range (m)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "sequence_range_by_object.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="process/processed_split_data_OMOMO_bps/test",
        help="Directory containing processed test .pt sequence files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/contact_bone_lengths_OMOMO_bps_test",
        help="Directory for CSV, plot, and Markdown outputs.",
    )
    parser.add_argument("--bin-width", type=float, default=0.02, help="Histogram bin width in meters.")
    parser.add_argument(
        "--skip-observations-csv",
        action="store_true",
        help="Skip writing the raw per-hand contact observation CSV.",
    )
    parser.add_argument(
        "--no-resolve-bimanual-contact-conflicts",
        action="store_true",
        help="Disable the same bimanual contact conflict postprocess used by IMUDataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    resolve_bimanual_contact_conflicts = not bool(args.no_resolve_bimanual_contact_conflicts)
    paths, observations, sequence_rows, segment_rows, skipped_rows = collect_observations(
        data_dir,
        resolve_bimanual_contact_conflicts=resolve_bimanual_contact_conflicts,
    )
    bins, centers, _by_object, counts_by_object, hist_long, hist_wide = build_histograms(observations, args.bin_width)
    object_rows = object_summary(sequence_rows, observations)

    if not args.skip_observations_csv:
        write_csv(output_dir / "contact_bone_observations.csv", observations)
    write_csv(output_dir / "sequence_summary.csv", sequence_rows)
    write_csv(output_dir / "segment_summary.csv", segment_rows)
    write_csv(output_dir / "object_summary.csv", object_rows)
    write_csv(output_dir / "histogram_by_object_long.csv", hist_long)
    write_csv(output_dir / "histogram_by_object_wide.csv", hist_wide)
    write_csv(output_dir / "skipped_sequences.csv", skipped_rows)

    write_analysis(
        output_dir / "analysis.md",
        data_dir,
        output_dir,
        total_files=len(paths),
        observations=observations,
        sequence_rows=sequence_rows,
        segment_rows=segment_rows,
        object_rows=object_rows,
        skipped_rows=skipped_rows,
        bin_width=args.bin_width,
        resolve_bimanual_contact_conflicts=resolve_bimanual_contact_conflicts,
    )
    make_plots(output_dir, centers, counts_by_object, object_rows, sequence_rows)

    print(f"Scanned files: {len(paths)}")
    print(f"Contact sequences: {len(sequence_rows)}")
    print(f"Objects: {len(object_rows)}")
    print(f"Contact observations: {len(observations)}")
    print(f"Resolve bimanual contact conflicts: {resolve_bimanual_contact_conflicts}")
    print(f"Histogram bins: {len(bins) - 1}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
