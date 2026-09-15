"""Geometry helpers used by the comparison baselines.

The comparison methods use slightly different IMU protocols.  These helpers keep
the conversions explicit instead of hiding protocol changes inside each model.
"""
from __future__ import annotations

from typing import Iterable

import torch

from utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def central_difference(x: torch.Tensor, dt: float, dim: int = 1) -> torch.Tensor:
    """Central difference with one-sided endpoints."""
    if x.shape[dim] <= 1:
        return torch.zeros_like(x)

    x_t = x.transpose(dim, 1)
    out = torch.zeros_like(x_t)
    out[:, 1:-1] = (x_t[:, 2:] - x_t[:, :-2]) / (2.0 * dt)
    out[:, 0] = (x_t[:, 1] - x_t[:, 0]) / dt
    out[:, -1] = (x_t[:, -1] - x_t[:, -2]) / dt
    return out.transpose(dim, 1)


def second_difference(x: torch.Tensor, dt: float, dim: int = 1) -> torch.Tensor:
    """Acceleration from positions via two finite differences."""
    return central_difference(central_difference(x, dt=dt, dim=dim), dt=dt, dim=dim)


def rotation_angular_velocity(rot: torch.Tensor, dt: float) -> torch.Tensor:
    """Approximate angular velocity from rotation matrices.

    Args:
        rot: Tensor shaped [B, T, S, 3, 3].

    Returns:
        Tensor shaped [B, T, S, 3] in each sensor's local frame.
    """
    if rot.shape[1] <= 1:
        return torch.zeros(rot.shape[:-2] + (3,), device=rot.device, dtype=rot.dtype)

    delta = rot[:, :-1].transpose(-1, -2).matmul(rot[:, 1:])
    aa = matrix_to_axis_angle(delta.reshape(-1, 3, 3)).reshape(delta.shape[:-2] + (3,))
    local_w = torch.zeros(rot.shape[:-2] + (3,), device=rot.device, dtype=rot.dtype)
    local_w[:, 1:] = aa / dt
    local_w[:, 0] = local_w[:, 1]
    return local_w


def from_to_rotation_matrix(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return rotation matrices which map ``source`` vectors onto ``target``.

    The implementation handles parallel and anti-parallel vectors explicitly,
    which is important for the gravity corrections used by GlobalPose.
    """
    source = source / source.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cross = torch.linalg.cross(source, target, dim=-1)
    cosine = (source * target).sum(dim=-1).clamp(-1.0, 1.0)

    k = torch.zeros(*cross.shape[:-1], 3, 3, device=cross.device, dtype=cross.dtype)
    k[..., 0, 1] = -cross[..., 2]
    k[..., 0, 2] = cross[..., 1]
    k[..., 1, 0] = cross[..., 2]
    k[..., 1, 2] = -cross[..., 0]
    k[..., 2, 0] = -cross[..., 1]
    k[..., 2, 1] = cross[..., 0]

    eye = torch.eye(3, device=cross.device, dtype=cross.dtype)
    eye = eye.expand(*cross.shape[:-1], 3, 3)
    scale = ((1.0 - cosine) / cross.square().sum(dim=-1).clamp_min(1e-8))[
        ..., None, None
    ]
    rotation = eye + k + k.matmul(k) * scale

    anti_parallel = cosine < -1.0 + 1e-5
    if anti_parallel.any():
        basis_index = source.abs().argmin(dim=-1)
        basis = torch.nn.functional.one_hot(basis_index, num_classes=3).to(source)
        axis = torch.linalg.cross(source, basis, dim=-1)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        rotation_pi = 2.0 * axis[..., :, None] * axis[..., None, :] - eye
        rotation = torch.where(anti_parallel[..., None, None], rotation_pi, rotation)
    return rotation


def sixd_imu_to_acc_rotmat(imu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split [acc3, rot6d] IMU features into acceleration and rotation matrix."""
    acc = imu[..., :3]
    rot6d = imu[..., 3:9]
    rot = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(imu.shape[:-1] + (3, 3))
    return acc, rot


def object_imu_to_12d(obj_imu: torch.Tensor) -> torch.Tensor:
    """Convert object [acc3, rot6d] into [acc3, rotmat9]."""
    if obj_imu.dim() == 4 and obj_imu.shape[-2] == 1:
        obj_imu = obj_imu.squeeze(-2)
    acc, rot = sixd_imu_to_acc_rotmat(obj_imu)
    return torch.cat([acc, rot.flatten(-2)], dim=-1)


def local_pose_axis_angle_to_rotmat(pose: torch.Tensor, num_joints: int = 24) -> torch.Tensor:
    """Convert flattened local axis-angle pose to [B, T, J, 3, 3]."""
    batch, seq = pose.shape[:2]
    rot = axis_angle_to_matrix(pose.reshape(batch, seq, -1, 3))
    if rot.shape[2] < num_joints:
        eye = torch.eye(3, device=pose.device, dtype=pose.dtype).view(1, 1, 1, 3, 3)
        pad = eye.expand(batch, seq, num_joints - rot.shape[2], 3, 3)
        rot = torch.cat([rot, pad], dim=2)
    return rot[:, :, :num_joints]


def select_and_flatten_rot6d(rot: torch.Tensor, joint_indices: Iterable[int]) -> torch.Tensor:
    """Select rotations and flatten with the repository's row-based 6D convention."""
    idx = torch.as_tensor(list(joint_indices), device=rot.device, dtype=torch.long)
    selected = rot.index_select(2, idx)
    return matrix_to_rotation_6d(selected.reshape(-1, 3, 3)).reshape(*selected.shape[:3], 6).flatten(2)


def rotation_matrix_to_r6d_columns(matrix: torch.Tensor) -> torch.Tensor:
    """Encode rotations with the first two columns, as used by TIP/articulate."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix input must end with [3,3], got {matrix.shape}")
    return matrix[..., :, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def r6d_columns_to_rotation_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """Decode TIP/articulate column-based 6D rotations with Gram-Schmidt."""
    if r6d.shape[-1] != 6:
        raise ValueError(f"6D rotation input must end with 6 dims, got {r6d.shape}")
    column0 = torch.nn.functional.normalize(r6d[..., :3], dim=-1)
    column1 = r6d[..., 3:6]
    column1 = column1 - (column0 * column1).sum(dim=-1, keepdim=True) * column0
    column1 = torch.nn.functional.normalize(column1, dim=-1)
    column2 = torch.linalg.cross(column0, column1, dim=-1)
    return torch.stack((column0, column1, column2), dim=-1).nan_to_num(0.0)


def rotation_matrix_to_tip_2axis(matrix: torch.Tensor) -> torch.Tensor:
    """Encode TIP's interleaved 3x2 rotation representation."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix input must end with [3,3], got {matrix.shape}")
    return matrix[..., :, :2].reshape(*matrix.shape[:-2], 6)


def tip_2axis_to_rotation_matrix(two_axis: torch.Tensor) -> torch.Tensor:
    """Decode TIP's row-major flattened 3x2 representation."""
    if two_axis.shape[-1] != 6:
        raise ValueError(f"TIP 2-axis input must end with 6 dims, got {two_axis.shape}")
    axes = two_axis.reshape(*two_axis.shape[:-1], 3, 2)
    column0 = torch.nn.functional.normalize(axes[..., :, 0], dim=-1)
    column1 = axes[..., :, 1]
    column1 = column1 - (column0 * column1).sum(dim=-1, keepdim=True) * column0
    column1 = torch.nn.functional.normalize(column1, dim=-1)
    column2 = torch.linalg.cross(column0, column1, dim=-1)
    return torch.stack((column0, column1, column2), dim=-1).nan_to_num(0.0)


def select_and_flatten_r6d_columns(
    rot: torch.Tensor, joint_indices: Iterable[int]
) -> torch.Tensor:
    """Select rotations and flatten with TIP/articulate's column-based 6D."""
    idx = torch.as_tensor(list(joint_indices), device=rot.device, dtype=torch.long)
    selected = rot.index_select(2, idx)
    return rotation_matrix_to_r6d_columns(selected).flatten(2)


def select_and_flatten_tip_2axis(
    rot: torch.Tensor, joint_indices: Iterable[int]
) -> torch.Tensor:
    """Select rotations and flatten with TIP's interleaved 3x2 convention."""
    idx = torch.as_tensor(list(joint_indices), device=rot.device, dtype=torch.long)
    return rotation_matrix_to_tip_2axis(rot.index_select(2, idx)).flatten(2)


def select_and_flatten_rotmat(rot: torch.Tensor, joint_indices: Iterable[int]) -> torch.Tensor:
    """Select rotations and flatten as 3x3 matrices."""
    idx = torch.as_tensor(list(joint_indices), device=rot.device, dtype=torch.long)
    selected = rot.index_select(2, idx)
    return selected.flatten(2)


def root_relative_global_rot6d(
    global_rot: torch.Tensor,
    joint_indices: Iterable[int],
    *,
    column_based: bool = False,
) -> torch.Tensor:
    """Root-normalize global rotations and return selected joints in 6D."""
    root_inv = global_rot[:, :, :1].transpose(-1, -2)
    rel = root_inv.matmul(global_rot)
    if column_based:
        return select_and_flatten_r6d_columns(rel, joint_indices)
    return select_and_flatten_rot6d(rel, joint_indices)


def root_relative_positions(position: torch.Tensor, joint_indices: Iterable[int], root_index: int = 0) -> torch.Tensor:
    """Root-relative joint positions in the root frame."""
    idx = torch.as_tensor(list(joint_indices), device=position.device, dtype=torch.long)
    pos = position.index_select(2, idx)
    root_pos = position[:, :, root_index : root_index + 1]
    root_rot = None
    return pos - root_pos
