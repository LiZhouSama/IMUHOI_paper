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
    """Select rotations and flatten as 6D features."""
    idx = torch.as_tensor(list(joint_indices), device=rot.device, dtype=torch.long)
    selected = rot.index_select(2, idx)
    return matrix_to_rotation_6d(selected.reshape(-1, 3, 3)).reshape(*selected.shape[:3], 6).flatten(2)


def select_and_flatten_rotmat(rot: torch.Tensor, joint_indices: Iterable[int]) -> torch.Tensor:
    """Select rotations and flatten as 3x3 matrices."""
    idx = torch.as_tensor(list(joint_indices), device=rot.device, dtype=torch.long)
    selected = rot.index_select(2, idx)
    return selected.flatten(2)


def root_relative_global_rot6d(global_rot: torch.Tensor, joint_indices: Iterable[int]) -> torch.Tensor:
    """Root-normalize global rotations and return selected joints in 6D."""
    root_inv = global_rot[:, :, :1].transpose(-1, -2)
    rel = root_inv.matmul(global_rot)
    return select_and_flatten_rot6d(rel, joint_indices)


def root_relative_positions(position: torch.Tensor, joint_indices: Iterable[int], root_index: int = 0) -> torch.Tensor:
    """Root-relative joint positions in the root frame."""
    idx = torch.as_tensor(list(joint_indices), device=position.device, dtype=torch.long)
    pos = position.index_select(2, idx)
    root_pos = position[:, :, root_index : root_index + 1]
    root_rot = None
    return pos - root_pos

