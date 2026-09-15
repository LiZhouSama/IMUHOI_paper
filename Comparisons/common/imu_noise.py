"""Comparison-only IMU noise layout adapters."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from process.imu_noise import (
    NOITOM_IMU_NOISE_CFG,
    apply_imu_noise as _apply_imu_noise,
    merge_noise_cfg,
)
from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix


def _tip_columns_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Decode TIP's row-major flattening of the first two columns."""
    axes = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    first = F.normalize(axes[..., :, 0], dim=-1)
    second = axes[..., :, 1]
    second = F.normalize(
        second - (first * second).sum(-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def _matrix_to_tip_columns(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def apply_imu_noise(
    acc: torch.Tensor,
    rot6d: torch.Tensor,
    fps: float,
    noise_cfg: Optional[Dict[str, float]] = None,
    generator: Optional[torch.Generator] = None,
    rotation_layout: str = "rows",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Apply the shared noise model to repository or TIP rotation layouts."""
    if rotation_layout == "rows":
        return _apply_imu_noise(
            acc,
            rot6d,
            fps,
            noise_cfg,
            generator=generator,
        )
    if rotation_layout != "tip_columns":
        raise ValueError(
            "rotation_layout must be 'rows' or 'tip_columns', got "
            f"{rotation_layout!r}"
        )

    rotation = _tip_columns_to_matrix(rot6d)
    row_rot6d = matrix_to_rotation_6d(rotation)
    noisy_acc, noisy_row_rot6d, merged_cfg = _apply_imu_noise(
        acc,
        row_rot6d,
        fps,
        noise_cfg,
        generator=generator,
    )
    noisy_rotation = rotation_6d_to_matrix(noisy_row_rot6d)
    return noisy_acc, _matrix_to_tip_columns(noisy_rotation), merged_cfg
