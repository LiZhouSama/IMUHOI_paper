"""
Small rotation conversion helpers used by the Mamba path.

The 6D convention follows PyTorch3D: the first two rows of the rotation
matrix are flattened to form the 6D representation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrices."""
    if d6.shape[-1] != 6:
        raise ValueError(f"6D rotation input must end with 6 dims, got {d6.shape}")

    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the 6D representation."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix input must end with [3,3], got {matrix.shape}")
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle vectors."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix input must end with [3,3], got {matrix.shape}")

    orig_shape = matrix.shape[:-2]
    mat = matrix.reshape(-1, 3, 3)
    trace = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    sin_angle = torch.sin(angle)

    axis = torch.stack(
        (
            mat[:, 2, 1] - mat[:, 1, 2],
            mat[:, 0, 2] - mat[:, 2, 0],
            mat[:, 1, 0] - mat[:, 0, 1],
        ),
        dim=-1,
    )
    denom = (2.0 * sin_angle).clamp_min(1e-7).unsqueeze(-1)
    axis = axis / denom
    aa = axis * angle.unsqueeze(-1)

    small = angle.abs() < 1e-6
    if bool(small.any()):
        aa[small] = 0.0
    return aa.reshape(*orig_shape, 3)

