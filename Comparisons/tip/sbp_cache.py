"""Sidecar SBP cache helpers for TIP comparison training.

The cache is intentionally stored outside the processed dataset files so the
main IMUHOI preprocessing output remains unchanged.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from configs import FRAME_RATE
from utils.rotation_conversions import matrix_to_axis_angle


TIP_SBP_LINKS = (
    (7, "foot"),    # left ankle / lower leg contact proxy
    (8, "foot"),    # right ankle / lower leg contact proxy
    (20, "wrist"),  # left wrist
    (21, "wrist"),  # right wrist
    (0, "root"),    # pelvis/root
)
TIP_SBP_DIM = len(TIP_SBP_LINKS) * 4
TIP_SBP_VERSION = "tip_rot_center_v1"
V_THRES = 0.15


def _build_grid(kind: str) -> np.ndarray:
    if kind == "wrist":
        xs = np.arange(-0.02, 0.03, 0.01)
        ys = np.arange(-0.02, 0.03, 0.01)
        zs = np.arange(-0.02, 0.03, 0.01)
    elif kind == "foot":
        xs = np.arange(-0.04, 0.05, 0.01)
        ys = np.arange(-0.04, 0.02, 0.01)
        zs = np.arange(-0.15, 0.18, 0.01)
    elif kind == "root":
        xs = np.arange(-0.15, 0.16, 0.01)
        ys = np.arange(-0.10, 0.15, 0.01)
        zs = np.arange(-0.12, -0.04, 0.01)
    else:
        raise ValueError(f"Unknown SBP grid kind: {kind}")
    xx, yy, zz = np.meshgrid(xs, ys, zs)
    return np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1).astype(np.float32)


GRIDS = {kind: _build_grid(kind) for kind in ("wrist", "foot", "root")}


def _source_relative_parts(source_path: str | os.PathLike) -> tuple[str, ...]:
    path = Path(source_path).resolve()
    parts = path.parts
    if "process" in parts:
        idx = parts.index("process")
        return tuple(parts[idx + 1 :])
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    return ("external", digest, path.name)


def cache_path_for_source(source_path: str | os.PathLike, cache_root: str | os.PathLike) -> Path:
    """Map a processed `.pt` path to its sidecar cache path."""
    return Path(cache_root).joinpath(*_source_relative_parts(source_path))


def _angular_velocity_from_pair(rot_prev: np.ndarray, rot_next: np.ndarray, dt: float) -> np.ndarray:
    d_rot = rot_prev.T @ rot_next
    aa = matrix_to_axis_angle(torch.from_numpy(d_rot).float()).numpy()
    return aa / (2.0 * dt)


def _link_sbp_sequence(
    position: np.ndarray,
    rotation: np.ndarray,
    kind: str,
    dt: float,
    threshold: float = V_THRES,
) -> np.ndarray:
    seq_len = int(position.shape[0])
    out = np.zeros((seq_len, 4), dtype=np.float32)
    if seq_len < 5:
        return out

    grid = GRIDS[kind]
    prev_solution = None
    grid_norm = np.linalg.norm(grid, axis=1)

    for t in range(2, seq_len - 2):
        velocity = (position[t + 1] - position[t - 1]) / (2.0 * dt)
        omega = _angular_velocity_from_pair(rotation[t - 1], rotation[t + 1], dt)
        cur_rot = rotation[t]
        candidates = grid @ cur_rot.T

        w1, w2, w3 = omega
        wx = np.array(
            [[0.0, -w3, w2], [w3, 0.0, -w1], [-w2, w1, 0.0]],
            dtype=np.float32,
        )
        candidate_vel = candidates @ wx.T + velocity[None, :]
        if prev_solution is None:
            smooth_term = 0.0
        else:
            smooth_term = np.linalg.norm(candidates - (prev_solution - velocity * dt)[None, :], axis=1)
        residues = np.linalg.norm(candidate_vel, axis=1) + 0.2 * smooth_term + 0.02 * grid_norm
        best = int(np.argmin(residues))
        if residues[best] < threshold:
            out[t, 0] = 1.0
            out[t, 1:] = candidates[best]
            prev_solution = candidates[best]
        else:
            prev_solution = None

    return out


def compute_tip_sbp_from_sequence(seq_data: dict, fps: float = FRAME_RATE) -> torch.Tensor:
    """Compute TIP-style SBP constraints from one processed IMUHOI sequence."""
    if "position_global_full_gt_world" not in seq_data or "rotation_global" not in seq_data:
        raise KeyError("sequence is missing position_global_full_gt_world or rotation_global")

    position = seq_data["position_global_full_gt_world"]
    rotation = seq_data["rotation_global"]
    if torch.is_tensor(position):
        position_np = position.detach().cpu().numpy().astype(np.float32)
    else:
        position_np = np.asarray(position, dtype=np.float32)
    if torch.is_tensor(rotation):
        rotation_np = rotation.detach().cpu().numpy().astype(np.float32)
    else:
        rotation_np = np.asarray(rotation, dtype=np.float32)

    if position_np.ndim != 3 or rotation_np.ndim != 4:
        raise ValueError(f"Unexpected position/rotation shapes: {position_np.shape}, {rotation_np.shape}")

    seq_len, num_joints = position_np.shape[:2]
    sbp_parts = []
    dt = 1.0 / float(fps)
    for joint_idx, kind in TIP_SBP_LINKS:
        if joint_idx >= num_joints or joint_idx >= rotation_np.shape[1]:
            sbp_parts.append(np.zeros((seq_len, 4), dtype=np.float32))
            continue
        sbp_parts.append(_link_sbp_sequence(position_np[:, joint_idx], rotation_np[:, joint_idx], kind, dt))

    return torch.from_numpy(np.concatenate(sbp_parts, axis=1)).float()


def generate_cache_for_file(
    source_path: str | os.PathLike,
    cache_root: str | os.PathLike,
    overwrite: bool = False,
    fps: float = FRAME_RATE,
) -> tuple[str, int, str]:
    """Generate one sidecar file.

    Returns:
        `(status, num_frames, cache_path)`.
    """
    cache_path = cache_path_for_source(source_path, cache_root)
    if cache_path.exists() and not overwrite:
        try:
            cached = torch.load(cache_path, map_location="cpu")
            frames = int(cached["tip_sbp"].shape[0]) if isinstance(cached, dict) and "tip_sbp" in cached else -1
        except Exception:
            frames = -1
        return "skip", frames, str(cache_path)

    seq_data = torch.load(source_path, map_location="cpu")
    tip_sbp = compute_tip_sbp_from_sequence(seq_data, fps=fps)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tip_sbp": tip_sbp,
            "source_file": str(Path(source_path).resolve()),
            "version": TIP_SBP_VERSION,
            "fps": float(fps),
        },
        cache_path,
    )
    return "write", int(tip_sbp.shape[0]), str(cache_path)


def iter_processed_files(data_roots: Sequence[str | os.PathLike]) -> list[Path]:
    files = []
    for root in data_roots:
        root_path = Path(root)
        if root_path.is_file() and root_path.suffix == ".pt":
            files.append(root_path)
        elif root_path.is_dir():
            files.extend(sorted(root_path.rglob("*.pt")))
    return sorted(files)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _lookup_cache_file(source_path, seq_file: str | None, cache_root: str | os.PathLike) -> Path | None:
    if source_path:
        path = cache_path_for_source(str(source_path), cache_root)
        if path.exists():
            return path
    if seq_file:
        matches = list(Path(cache_root).rglob(seq_file))
        if len(matches) == 1:
            return matches[0]
    return None


def attach_tip_sbp_from_cache(
    batch: dict,
    cache_root: str | os.PathLike | None,
    device: torch.device | None = None,
) -> dict:
    """Attach `tip_sbp` to a DataLoader batch using sidecar cache files.

    Missing or mismatched cache entries are represented as NaNs so the existing
    TIP loss masks the SBP term instead of crashing.
    """
    if not cache_root:
        return batch

    seq_len = int(batch["human_imu"].shape[1])
    dtype = batch["human_imu"].dtype if torch.is_tensor(batch.get("human_imu")) else torch.float32
    device = device or (batch["human_imu"].device if torch.is_tensor(batch.get("human_imu")) else torch.device("cpu"))
    seq_paths = _as_list(batch.get("seq_path"))
    seq_files = _as_list(batch.get("seq_file"))
    starts = batch.get("window_start")
    ends = batch.get("window_end")
    batch_size = int(batch["human_imu"].shape[0])

    sbp_batch = []
    for i in range(batch_size):
        source_path = seq_paths[i] if i < len(seq_paths) else None
        seq_file = seq_files[i] if i < len(seq_files) else None
        cache_file = _lookup_cache_file(source_path, seq_file, cache_root)
        if cache_file is None:
            sbp_batch.append(torch.full((seq_len, TIP_SBP_DIM), float("nan"), dtype=dtype, device=device))
            continue

        start = int(starts[i].item() if torch.is_tensor(starts) else starts[i])
        end = int(ends[i].item() if torch.is_tensor(ends) else ends[i])
        cached = torch.load(cache_file, map_location="cpu")
        sbp = cached["tip_sbp"] if isinstance(cached, dict) else cached
        sbp = sbp[start:end].to(device=device, dtype=dtype)
        if sbp.shape[0] != seq_len:
            fixed = torch.full((seq_len, TIP_SBP_DIM), float("nan"), dtype=dtype, device=device)
            fixed[: min(seq_len, sbp.shape[0])] = sbp[:seq_len]
            sbp = fixed
        sbp_batch.append(sbp)

    batch = dict(batch)
    batch["tip_sbp"] = torch.stack(sbp_batch, dim=0)
    return batch

