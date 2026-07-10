"""Shared helpers for human pose modules."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

from configs import _IGNORED_INDICES, _REDUCED_INDICES, _SENSOR_ROT_INDICES
from utils.rotation_conversions import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix


SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
WRIST_JOINT_INDICES = (20, 21)
FOREARM_PARENT_JOINT_INDICES = (18, 19)
VIRTUAL_PALM_JOINT_INDICES = (22, 23)
# Estimated from zero-pose SMPL-H middle metacarpal offsets. The virtual palm
# anchor intentionally follows the forearm extension instead of native SMPL-H
# hand-joint ordering, where Jtr[22:24] are not left/right palm roots.
PALM_EXTENSION_RATIOS = (0.43, 0.42)
_NO_SHAPE_ARG = object()


def compute_virtual_palm_positions(
    joints: torch.Tensor,
    *,
    forearm_parent_indices: Sequence[int] = FOREARM_PARENT_JOINT_INDICES,
    wrist_indices: Sequence[int] = WRIST_JOINT_INDICES,
    extension_ratios: Sequence[float] = PALM_EXTENSION_RATIOS,
) -> torch.Tensor:
    """Compute left/right virtual palm anchors from elbow and wrist positions."""
    if not isinstance(joints, torch.Tensor):
        raise TypeError("joints must be a torch.Tensor")
    if joints.shape[-1] != 3:
        raise ValueError(f"joints must end with xyz coordinates, got {joints.shape}")

    joint_count = int(joints.shape[-2])
    max_wrist_idx = max(int(idx) for idx in wrist_indices)
    if joint_count <= max_wrist_idx:
        shape = (*joints.shape[:-2], 2, 3)
        return torch.zeros(shape, device=joints.device, dtype=joints.dtype)

    wrists = torch.stack(
        [joints[..., int(wrist_indices[0]), :], joints[..., int(wrist_indices[1]), :]],
        dim=-2,
    )

    max_parent_idx = max(int(idx) for idx in forearm_parent_indices)
    if joint_count <= max_parent_idx:
        return wrists

    forearms = torch.stack(
        [
            joints[..., int(wrist_indices[0]), :] - joints[..., int(forearm_parent_indices[0]), :],
            joints[..., int(wrist_indices[1]), :] - joints[..., int(forearm_parent_indices[1]), :],
        ],
        dim=-2,
    )
    ratios = torch.as_tensor(extension_ratios, device=joints.device, dtype=joints.dtype)
    ratio_shape = (1,) * (forearms.dim() - 2) + (2, 1)
    return wrists + forearms * ratios.view(ratio_shape)


def append_virtual_palm_joints(
    joints: torch.Tensor,
    *,
    num_joints: int = 24,
    palm_indices: Sequence[int] = VIRTUAL_PALM_JOINT_INDICES,
) -> torch.Tensor:
    """Return a custom 24-joint tensor whose 22/23 entries are virtual palms."""
    if joints.shape[-1] != 3:
        raise ValueError(f"joints must end with xyz coordinates, got {joints.shape}")

    joint_count = int(joints.shape[-2])
    if joint_count >= num_joints:
        out = joints[..., :num_joints, :].clone()
    else:
        pad_shape = (*joints.shape[:-2], num_joints - joint_count, 3)
        pad = torch.zeros(pad_shape, device=joints.device, dtype=joints.dtype)
        out = torch.cat((joints, pad), dim=-2)

    palms = compute_virtual_palm_positions(joints)
    left_idx, right_idx = (int(palm_indices[0]), int(palm_indices[1]))
    if left_idx < num_joints:
        out[..., left_idx, :] = palms[..., 0, :]
    if right_idx < num_joints:
        out[..., right_idx, :] = palms[..., 1, :]
    return out


def append_virtual_palm_rotations(
    rotations: torch.Tensor,
    *,
    num_joints: int = 24,
    wrist_indices: Sequence[int] = WRIST_JOINT_INDICES,
    palm_indices: Sequence[int] = VIRTUAL_PALM_JOINT_INDICES,
) -> torch.Tensor:
    """Append virtual palm global rotations by inheriting wrist rotations."""
    if rotations.shape[-2:] != (3, 3):
        raise ValueError(f"rotations must end with [3,3], got {rotations.shape}")

    joint_count = int(rotations.shape[-3])
    if joint_count >= num_joints:
        out = rotations[..., :num_joints, :, :].clone()
    else:
        eye = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        view_shape = (1,) * (rotations.dim() - 3) + (1, 3, 3)
        pad_shape = (*rotations.shape[:-3], num_joints - joint_count, 3, 3)
        pad = eye.view(view_shape).expand(pad_shape)
        out = torch.cat((rotations, pad), dim=-3)

    left_wrist, right_wrist = (int(wrist_indices[0]), int(wrist_indices[1]))
    left_palm, right_palm = (int(palm_indices[0]), int(palm_indices[1]))
    if joint_count > left_wrist and left_palm < num_joints:
        out[..., left_palm, :, :] = rotations[..., left_wrist, :, :]
    if joint_count > right_wrist and right_palm < num_joints:
        out[..., right_palm, :, :] = rotations[..., right_wrist, :, :]
    return out


def select_hand_anchor_positions(joints: torch.Tensor) -> torch.Tensor:
    """Select the left/right hand anchor used by VC/OT: virtual palm positions."""
    return compute_virtual_palm_positions(joints)


def select_wrist_positions(
    joints: torch.Tensor,
    *,
    wrist_indices: Sequence[int] = WRIST_JOINT_INDICES,
) -> torch.Tensor:
    """Select the left/right wrist positions used for wrist-mounted IMU supervision."""
    if not isinstance(joints, torch.Tensor):
        raise TypeError("joints must be a torch.Tensor")
    if joints.shape[-1] != 3:
        raise ValueError(f"joints must end with xyz coordinates, got {joints.shape}")

    joint_count = int(joints.shape[-2])
    max_wrist_idx = max(int(idx) for idx in wrist_indices)
    shape = (*joints.shape[:-2], 2, 3)
    if joint_count <= max_wrist_idx:
        return torch.zeros(shape, device=joints.device, dtype=joints.dtype)
    return torch.stack(
        [joints[..., int(wrist_indices[0]), :], joints[..., int(wrist_indices[1]), :]],
        dim=-2,
    )


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


def _body_model_batch_size(*args) -> int:
    for arg in args:
        if arg is not None:
            return int(arg.shape[0])
    return 1


def _expand_body_model_buffer(body_model, name: str, batch_size: int) -> torch.Tensor:
    value = getattr(body_model, name)
    return value.expand(batch_size, *value.shape[1:])


def _body_model_shape_components(
    body_model,
    *,
    batch_size: int,
    betas: Optional[torch.Tensor],
    dmpls: Optional[torch.Tensor],
    expression: Optional[torch.Tensor],
):
    model_type = getattr(body_model, "model_type", "")
    use_dmpl = bool(getattr(body_model, "use_dmpl", False))
    use_expression = bool(getattr(body_model, "use_expression", False)) or (
        model_type in {"smplx", "flame"} and hasattr(body_model, "exprdirs")
    )

    if use_dmpl:
        if betas is None:
            betas = _expand_body_model_buffer(body_model, "init_betas", batch_size)
        if dmpls is None:
            dmpls = _expand_body_model_buffer(body_model, "init_dmpls", batch_size)
        return torch.cat([betas, dmpls], dim=-1), torch.cat([body_model.shapedirs, body_model.dmpldirs], dim=-1)

    if use_expression:
        if betas is None:
            betas = _expand_body_model_buffer(body_model, "init_betas", batch_size)
        if expression is None:
            expression = _expand_body_model_buffer(body_model, "init_expression", batch_size)
        return torch.cat([betas, expression], dim=-1), torch.cat([body_model.shapedirs, body_model.exprdirs], dim=-1)

    if betas is None:
        return _NO_SHAPE_ARG, body_model.shapedirs
    return betas, body_model.shapedirs


def _body_model_full_pose(
    body_model,
    *,
    batch_size: int,
    root_orient: Optional[torch.Tensor],
    pose_body: Optional[torch.Tensor],
    pose_hand: Optional[torch.Tensor],
    pose_jaw: Optional[torch.Tensor],
    pose_eye: Optional[torch.Tensor],
) -> torch.Tensor:
    model_type = getattr(body_model, "model_type", "")
    if root_orient is None:
        root_orient = _expand_body_model_buffer(body_model, "init_root_orient", batch_size)

    if model_type in {"smpl", "smplh"}:
        if pose_body is None:
            pose_body = _expand_body_model_buffer(body_model, "init_pose_body", batch_size)
        if pose_hand is None:
            pose_hand = _expand_body_model_buffer(body_model, "init_pose_hand", batch_size)
        return torch.cat([root_orient, pose_body, pose_hand], dim=-1)

    if model_type == "smplx":
        if pose_body is None:
            pose_body = _expand_body_model_buffer(body_model, "init_pose_body", batch_size)
        if pose_hand is None:
            pose_hand = _expand_body_model_buffer(body_model, "init_pose_hand", batch_size)
        if pose_jaw is None:
            pose_jaw = _expand_body_model_buffer(body_model, "init_pose_jaw", batch_size)
        if pose_eye is None:
            pose_eye = _expand_body_model_buffer(body_model, "init_pose_eye", batch_size)
        return torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand], dim=-1)

    if model_type == "flame":
        if pose_body is None:
            pose_body = _expand_body_model_buffer(body_model, "init_pose_body", batch_size)
        if pose_jaw is None:
            pose_jaw = _expand_body_model_buffer(body_model, "init_pose_jaw", batch_size)
        if pose_eye is None:
            pose_eye = _expand_body_model_buffer(body_model, "init_pose_eye", batch_size)
        return torch.cat([root_orient, pose_body, pose_jaw, pose_eye], dim=-1)

    if model_type == "mano":
        if pose_hand is None:
            pose_hand = _expand_body_model_buffer(body_model, "init_pose_hand", batch_size)
        return torch.cat([root_orient, pose_hand], dim=-1)

    if model_type in {"animal_horse", "animal_dog", "animal_rat"}:
        if pose_body is None:
            pose_body = _expand_body_model_buffer(body_model, "init_pose_body", batch_size)
        return torch.cat([root_orient, pose_body], dim=-1)

    raise ValueError(f"Unsupported body model type: {model_type}")


def _body_model_joints_from_shape(
    body_model,
    *,
    batch_size: int,
    shape_components,
    shapedirs: torch.Tensor,
    v_template: Optional[torch.Tensor],
    joints: Optional[torch.Tensor],
    v_shaped: Optional[torch.Tensor],
) -> torch.Tensor:
    from human_body_prior.body_model.lbs import vertices2joints

    if joints is not None:
        return joints
    if v_shaped is not None:
        return vertices2joints(body_model.J_regressor, v_shaped)

    if v_template is None:
        template_joints = vertices2joints(body_model.J_regressor, body_model.init_v_template)
        template_joints = template_joints.expand(batch_size, -1, -1)
    else:
        template_joints = vertices2joints(body_model.J_regressor, v_template)

    if shape_components is _NO_SHAPE_ARG:
        return template_joints

    joint_shapedirs = torch.einsum("ji,ikl->jkl", body_model.J_regressor, shapedirs)
    return template_joints + torch.einsum("bl,jkl->bjk", shape_components, joint_shapedirs)


def _full_body_model_joints_fallback(
    body_model,
    *,
    root_orient: Optional[torch.Tensor],
    pose_body: Optional[torch.Tensor],
    pose_hand: Optional[torch.Tensor],
    pose_jaw: Optional[torch.Tensor],
    pose_eye: Optional[torch.Tensor],
    betas: Optional[torch.Tensor],
    trans: Optional[torch.Tensor],
    dmpls: Optional[torch.Tensor],
    expression: Optional[torch.Tensor],
    v_template: Optional[torch.Tensor],
    joints: Optional[torch.Tensor],
    v_shaped: Optional[torch.Tensor],
) -> torch.Tensor:
    body_out = body_model(
        root_orient=root_orient,
        pose_body=pose_body,
        pose_hand=pose_hand,
        pose_jaw=pose_jaw,
        pose_eye=pose_eye,
        betas=betas,
        trans=trans,
        dmpls=dmpls,
        expression=expression,
        v_template=v_template,
        joints=joints,
        v_shaped=v_shaped,
    )
    if isinstance(body_out, dict):
        return body_out["Jtr"]
    return body_out.Jtr


def forward_body_model_joints(
    body_model,
    *,
    root_orient: Optional[torch.Tensor] = None,
    pose_body: Optional[torch.Tensor] = None,
    pose_hand: Optional[torch.Tensor] = None,
    pose_jaw: Optional[torch.Tensor] = None,
    pose_eye: Optional[torch.Tensor] = None,
    betas: Optional[torch.Tensor] = None,
    trans: Optional[torch.Tensor] = None,
    dmpls: Optional[torch.Tensor] = None,
    expression: Optional[torch.Tensor] = None,
    v_template: Optional[torch.Tensor] = None,
    joints: Optional[torch.Tensor] = None,
    v_shaped: Optional[torch.Tensor] = None,
    fallback_to_full: bool = True,
) -> torch.Tensor:
    """BodyModel forward variant that returns Jtr without computing mesh vertices."""
    try:
        from human_body_prior.body_model.lbs import batch_rigid_transform, batch_rodrigues

        batch_size = _body_model_batch_size(
            root_orient,
            pose_body,
            pose_hand,
            pose_jaw,
            pose_eye,
            betas,
            trans,
            dmpls,
            expression,
            v_template,
            joints,
        )
        full_pose = _body_model_full_pose(
            body_model,
            batch_size=batch_size,
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            pose_jaw=pose_jaw,
            pose_eye=pose_eye,
        )
        shape_components, shapedirs = _body_model_shape_components(
            body_model,
            batch_size=batch_size,
            betas=betas,
            dmpls=dmpls,
            expression=expression,
        )
        joint_rest = _body_model_joints_from_shape(
            body_model,
            batch_size=batch_size,
            shape_components=shape_components,
            shapedirs=shapedirs,
            v_template=v_template,
            joints=joints,
            v_shaped=v_shaped,
        )

        dtype = getattr(body_model, "dtype", full_pose.dtype)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)
        jtr, _ = batch_rigid_transform(rot_mats, joint_rest, body_model.kintree_table[0].long(), dtype=dtype)
        if trans is None:
            trans = _expand_body_model_buffer(body_model, "init_trans", batch_size)
        return jtr + trans.unsqueeze(dim=1)
    except Exception:
        if not fallback_to_full:
            raise
        return _full_body_model_joints_fallback(
            body_model,
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            pose_jaw=pose_jaw,
            pose_eye=pose_eye,
            betas=betas,
            trans=trans,
            dmpls=dmpls,
            expression=expression,
            v_template=v_template,
            joints=joints,
            v_shaped=v_shaped,
        )


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
        joints = forward_body_model_joints(
            body_model,
            pose_body=pose_aa[:, 1:22].reshape(bt, 63),
            root_orient=pose_aa[:, 0].reshape(bt, 3),
        )
    except Exception:
        return None

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
