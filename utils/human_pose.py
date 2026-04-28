"""Shared helpers for human pose modules."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

from configs import _IGNORED_INDICES, _REDUCED_INDICES, _SENSOR_ROT_INDICES
from utils.rotation_conversions import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix


SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def load_body_model(body_model_path: str, num_betas: int = 16):
    """Load a frozen BodyModel instance."""
    from human_body_prior.body_model.body_model import BodyModel

    body_model = BodyModel(bm_fname=body_model_path, num_betas=num_betas)
    body_model.eval()
    for param in body_model.parameters():
        param.requires_grad_(False)
    return body_model


def ensure_body_model_device(module, device: torch.device):
    """Move module.body_model to device once and track the current device."""
    body_model = getattr(module, "body_model", None)
    if body_model is not None and getattr(module, "body_model_device", None) != device:
        module.body_model = body_model.to(device)
        module.body_model_device = device


def global_to_local_rotmats(
    global_rotmats: torch.Tensor,
    parents: Sequence[int] = SMPL_PARENTS,
    ignored_indices: Optional[Iterable[int]] = _IGNORED_INDICES,
) -> torch.Tensor:
    """Convert global joint rotations to local rotations."""
    if global_rotmats.shape[-2:] != (3, 3):
        raise ValueError(f"global_rotmats must end with [3,3], got {global_rotmats.shape}")

    batch_size, num_joints = global_rotmats.shape[:2]
    local = torch.zeros_like(global_rotmats)
    local[:, 0] = global_rotmats[:, 0]

    for joint_idx in range(1, num_joints):
        parent_idx = parents[joint_idx]
        local[:, joint_idx] = torch.matmul(global_rotmats[:, parent_idx].transpose(-1, -2), global_rotmats[:, joint_idx])

    if ignored_indices:
        ignored = [idx for idx in ignored_indices if idx < num_joints]
        if ignored:
            eye = torch.eye(3, device=global_rotmats.device, dtype=global_rotmats.dtype).view(1, 1, 3, 3)
            local[:, ignored] = eye.expand(batch_size, len(ignored), -1, -1)
    return local


def reduced_root_pose_to_full_global(
    reduced_root_6d: torch.Tensor,
    imu_orientation_mat: torch.Tensor,
    *,
    num_joints: int = 24,
    reduced_indices: Sequence[int] = _REDUCED_INDICES,
    sensor_rot_indices: Sequence[int] = _SENSOR_ROT_INDICES,
    ignored_indices: Sequence[int] = _IGNORED_INDICES,
    parents: Sequence[int] = SMPL_PARENTS,
) -> torch.Tensor:
    """
    Assemble full global rotations from root-relative reduced pose and IMU rotations.

    Args:
        reduced_root_6d: [N, R, 6], reduced joint rotations in root frame.
        imu_orientation_mat: [N, S, 3, 3], root IMU is global, other IMUs are root-relative.
    """
    if reduced_root_6d.dim() != 3 or reduced_root_6d.shape[-1] != 6:
        raise ValueError(f"reduced_root_6d must be [N,R,6], got {reduced_root_6d.shape}")
    if imu_orientation_mat.dim() != 4 or imu_orientation_mat.shape[-2:] != (3, 3):
        raise ValueError(f"imu_orientation_mat must be [N,S,3,3], got {imu_orientation_mat.shape}")

    n = reduced_root_6d.shape[0]
    device = reduced_root_6d.device
    dtype = reduced_root_6d.dtype
    root_rot = imu_orientation_mat[:, 0]
    reduced_local = rotation_6d_to_matrix(reduced_root_6d.reshape(-1, 6)).reshape(n, len(reduced_indices), 3, 3)
    reduced_global = torch.matmul(root_rot.unsqueeze(1), reduced_local)

    full = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(n, num_joints, 1, 1)
    valid_reduced = [idx for idx in reduced_indices if idx < num_joints]
    full[:, valid_reduced] = reduced_global[:, : len(valid_reduced)]

    sensor_count = min(imu_orientation_mat.shape[1], len(sensor_rot_indices))
    sensor_global = imu_orientation_mat[:, :sensor_count].clone()
    if sensor_count > 1:
        sensor_global[:, 1:] = torch.matmul(root_rot.unsqueeze(1), imu_orientation_mat[:, 1:sensor_count])
    valid_sensor_indices = [idx for idx in sensor_rot_indices[:sensor_count] if idx < num_joints]
    full[:, valid_sensor_indices] = sensor_global[:, : len(valid_sensor_indices)]

    ignored = sorted(idx for idx in ignored_indices if idx < num_joints)
    for idx in ignored:
        parent_idx = int(parents[idx])
        if 0 <= parent_idx < num_joints:
            full[:, idx] = full[:, parent_idx]
    return full


def compute_smpl_joints_from_global(
    full_global_rotmats: torch.Tensor,
    body_model,
    *,
    num_joints: int = 24,
    parents: Sequence[int] = SMPL_PARENTS,
) -> Optional[torch.Tensor]:
    """Run SMPL/SMPLH FK from full global rotations."""
    if body_model is None:
        return None
    if full_global_rotmats.dim() != 5 or full_global_rotmats.shape[-2:] != (3, 3):
        raise ValueError(f"full_global_rotmats must be [B,T,J,3,3], got {full_global_rotmats.shape}")

    batch_size, seq_len, joint_count = full_global_rotmats.shape[:3]
    bt = batch_size * seq_len
    full_global = full_global_rotmats.reshape(bt, joint_count, 3, 3)
    if joint_count < num_joints:
        eye = torch.eye(3, device=full_global.device, dtype=full_global.dtype).view(1, 1, 3, 3)
        pad = eye.expand(bt, num_joints - joint_count, -1, -1)
        full_global = torch.cat((full_global, pad), dim=1)
    elif joint_count > num_joints:
        full_global = full_global[:, :num_joints]

    local_pose = global_to_local_rotmats(full_global, parents=parents)
    pose_aa = matrix_to_axis_angle(local_pose.reshape(-1, 3, 3)).reshape(bt, num_joints, 3)

    try:
        body_out = body_model(
            pose_body=pose_aa[:, 1:22].reshape(bt, 63),
            root_orient=pose_aa[:, 0].reshape(bt, 3),
        )
    except Exception:
        return None

    joints = body_out.Jtr
    if joints.size(1) > num_joints:
        joints = joints[:, :num_joints]
    elif joints.size(1) < num_joints:
        pad = torch.zeros(
            joints.size(0),
            num_joints - joints.size(1),
            3,
            device=joints.device,
            dtype=joints.dtype,
        )
        joints = torch.cat((joints, pad), dim=1)
    return joints.reshape(batch_size, seq_len, num_joints, 3)


def compute_root_velocity_from_trans(trans: torch.Tensor, fps: float) -> torch.Tensor:
    """Compute root velocity by first difference."""
    if trans.dim() == 2:
        trans = trans.unsqueeze(1)
    vel = torch.zeros_like(trans)
    if trans.size(1) > 1:
        vel[:, 1:] = (trans[:, 1:] - trans[:, :-1]) * float(fps)
        vel[:, 0] = vel[:, 1]
    return vel


def integrate_root_velocity(root_velocity: torch.Tensor, fps: float, trans_init: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Integrate velocity to root translation, keeping frame 0 at trans_init."""
    trans = torch.zeros_like(root_velocity)
    if root_velocity.size(1) > 1:
        trans[:, 1:] = torch.cumsum(root_velocity[:, 1:], dim=1) / float(fps)
    if trans_init is not None:
        if trans_init.dim() == 1:
            trans_init = trans_init.unsqueeze(0)
        trans = trans + trans_init.unsqueeze(1).to(device=trans.device, dtype=trans.dtype)
    return trans


def full_global_to_reduced_root_6d(
    full_global_rotmats: torch.Tensor,
    *,
    reduced_indices: Sequence[int] = _REDUCED_INDICES,
) -> torch.Tensor:
    """Convert full global rotations to root-relative reduced 6D rotations."""
    root_rot = full_global_rotmats[:, :, 0]
    reduced_global = full_global_rotmats[:, :, reduced_indices]
    reduced_root = torch.matmul(root_rot.unsqueeze(2).transpose(-1, -2), reduced_global)
    return matrix_to_rotation_6d(reduced_root.reshape(-1, 3, 3)).reshape(
        full_global_rotmats.shape[0],
        full_global_rotmats.shape[1],
        len(reduced_indices),
        6,
    )
