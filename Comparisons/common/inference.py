"""Checkpoint inference and protocol-aware output decoding for comparisons."""
from __future__ import annotations

from typing import Dict

import torch

from configs import FRAME_RATE, _IGNORED_INDICES
from Comparisons.common.adapters import (
    DIP_REDUCED_15,
    TIP_POSE_18,
    TIP_POSE_18_LEGACY,
    TIP_REFERENCE_POSE_18,
    TRANSPOSE_VEL_SCALE,
    adapt_batch,
)
from Comparisons.common.geometry import (
    local_pose_axis_angle_to_rotmat,
    r6d_columns_to_rotation_matrix,
    sixd_imu_to_acc_rotmat,
    tip_2axis_to_rotation_matrix,
)
from utils.human_pose import (
    SMPL_PARENTS,
    append_virtual_palm_joints,
    forward_body_model_joints,
)
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix


def project_to_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Project arbitrary 3x3 predictions onto SO(3)."""
    u, _, vh = torch.linalg.svd(matrix)
    rotation = u.matmul(vh)
    negative = torch.det(rotation) < 0
    if negative.any():
        u = u.clone()
        u[..., :, -1] = torch.where(
            negative[..., None], -u[..., :, -1], u[..., :, -1]
        )
        rotation = u.matmul(vh)
    return rotation


def local_to_global_rotmats(local: torch.Tensor) -> torch.Tensor:
    """Convert batched SMPL local rotations to global rotations."""
    global_rot = torch.empty_like(local)
    global_rot[:, :, 0] = local[:, :, 0]
    for joint in range(1, local.shape[2]):
        parent = SMPL_PARENTS[joint]
        global_rot[:, :, joint] = global_rot[:, :, parent].matmul(
            local[:, :, joint]
        )
    return global_rot


def global_to_local_rotmats(global_rot: torch.Tensor) -> torch.Tensor:
    """Convert batched SMPL global rotations to local rotations."""
    local = torch.empty_like(global_rot)
    local[:, :, 0] = global_rot[:, :, 0]
    for joint in range(1, global_rot.shape[2]):
        parent = SMPL_PARENTS[joint]
        local[:, :, joint] = global_rot[:, :, parent].transpose(-1, -2).matmul(
            global_rot[:, :, joint]
        )
    return local


def _identity_pose(reference: torch.Tensor, num_joints: int = 24) -> torch.Tensor:
    batch_size, seq_len = reference.shape[:2]
    return torch.eye(3, device=reference.device, dtype=reference.dtype).view(
        1, 1, 1, 3, 3
    ).repeat(batch_size, seq_len, num_joints, 1, 1)


def _root_orientation(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    protocol = batch.get("imu_protocol", "")
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0] if protocol else ""
    protocol = str(protocol).lower()
    if protocol == "transpose_reference":
        human = batch["human_imu"]
        if human.shape[-1] != 72:
            raise ValueError("TransPose reference human stream must have 72 channels")
        # Official layout is [six accelerations (18), six rotation matrices
        # (54)] with the root sensor last.
        return human[:, :, 18 + 5 * 9 : 18 + 6 * 9].reshape(
            human.shape[0], human.shape[1], 3, 3
        )
    if protocol == "globalpose_reference":
        # GlobalPose follows the released root-last sensor order
        # [left forearm, right forearm, left leg, right leg, head, root].
        # The final sensor orientation is the root/body orientation used to
        # place the decoded local pose back in the world frame.
        _, orientation = sixd_imu_to_acc_rotmat(batch["human_imu"])
        return orientation[:, :, 5]
    if (
        protocol == "tip_reference"
        or (
            torch.is_tensor(batch.get("tip_root_rot0"))
            and batch["tip_root_rot0"].ndim >= 3
        )
    ):
        # Official TIP stores root orientation as a 3x2 matrix in the
        # sequence's initial-root frame.  Compose it with the saved world
        # orientation of frame zero before decoding the comparison output.
        root_two_axis = batch["human_imu"][:, :, 0, 3:9]
        normalized = tip_2axis_to_rotation_matrix(root_two_axis.reshape(-1, 6)).reshape(
            root_two_axis.shape[:-1] + (3, 3)
        )
        root0 = batch.get("tip_root_rot0")
        if root0 is None:
            root0 = normalized[:, :1]
        return root0[:, None].matmul(normalized)
    _, orientation = sixd_imu_to_acc_rotmat(batch["human_imu"])
    return orientation[:, :, 0]


def _decode_local_pose(
    method: str,
    output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    protocol_version: int,
    dip_variant: str = "legacy",
    tip_variant: str = "legacy",
    transpose_variant: str = "legacy",
) -> torch.Tensor:
    root_orientation = _root_orientation(batch)
    if method == "dip18":
        if dip_variant == "official":
            pose = output.get("pose", output.get("pose_mu"))
            if pose is None or pose.shape[-1] != 22 * 9:
                raise ValueError(
                    "Official DIP output must contain 22*9 raw rotation-matrix values"
                )
            # The published model predicts all 22 local rotations, including
            # the root. Unlike the legacy reduced head, do not overwrite its
            # root with a differently normalized measurement.
            reduced = project_to_rotation_matrix(
                pose.reshape(*pose.shape[:2], 22, 3, 3)
            ).contiguous()
            local = _identity_pose(pose)
            local[:, :, :22] = reduced
            return local
        if dip_variant != "legacy":
            raise ValueError(f"Unknown DIP variant: {dip_variant!r}")
        reduced = project_to_rotation_matrix(
            output["pose_mu"].reshape(*output["pose_mu"].shape[:2], 15, 3, 3)
        )
        local = _identity_pose(output["pose_mu"])
        local[:, :, DIP_REDUCED_15] = reduced
        local[:, :, 0] = root_orientation
        return local

    if method == "tip":
        pose6d = output["state"][:, :, : 18 * 6]
        # The released OMOMO TIP target is copied from
        # ``rotation_local_full_gt_list`` (PyTorch3D's first-two-*rows* 6D),
        # even though the IMU sensor stream uses TIP's interleaved columns.
        # Decode the state with the row convention so the learned checkpoint
        # is evaluated in the same pose space it was trained on.
        decoder = (
            rotation_6d_to_matrix
            if tip_variant == "official"
            else tip_2axis_to_rotation_matrix
            if protocol_version >= 2
            else rotation_6d_to_matrix
        )
        reduced = decoder(pose6d.reshape(-1, 6)).reshape(
            *pose6d.shape[:2], 18, 3, 3
        )
        local = _identity_pose(pose6d)
        pose_indices = (
            TIP_REFERENCE_POSE_18
            if tip_variant == "official"
            else TIP_POSE_18
            if protocol_version >= 2
            else TIP_POSE_18_LEGACY
        )
        local[:, :, pose_indices] = reduced
        local[:, :, 0] = root_orientation
        return local

    reduced6d = output["pose"] if method == "transpose" else output["ik2"]
    decoder = (
        rotation_6d_to_matrix
        if method == "transpose" and transpose_variant == "official"
        else r6d_columns_to_rotation_matrix
        if protocol_version >= 2
        else rotation_6d_to_matrix
    )
    reduced = decoder(reduced6d.reshape(-1, 6)).reshape(
        *reduced6d.shape[:2], 15, 3, 3
    )
    root_relative_global = _identity_pose(reduced6d)
    root_relative_global[:, :, DIP_REDUCED_15] = reduced
    local = global_to_local_rotmats(root_relative_global)
    ignored = [joint for joint in _IGNORED_INDICES if joint < local.shape[2]]
    local[:, :, ignored] = torch.eye(
        3, device=local.device, dtype=local.dtype
    )
    if method == "globalpose" and "root_correction" in output:
        local[:, :, 0] = root_orientation.matmul(output["root_correction"])
    else:
        local[:, :, 0] = root_orientation
    return local


def body_joints_from_local_pose(
    local_pose: torch.Tensor,
    root_trans: torch.Tensor,
    body_model,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Run body-model FK and append the two IMUHOI virtual palm joints."""
    if body_model is None:
        raise ValueError("body_model is required for comparison evaluation")
    batch_size, seq_len = local_pose.shape[:2]
    pose22 = local_pose[:, :, :22].reshape(-1, 22, 3, 3)
    pose_axis = matrix_to_axis_angle(pose22.reshape(-1, 3, 3)).reshape(-1, 22, 3)
    trans = root_trans.reshape(-1, 3)
    chunks = []
    for start in range(0, pose_axis.shape[0], chunk_size):
        end = min(start + chunk_size, pose_axis.shape[0])
        joints = forward_body_model_joints(
            body_model,
            root_orient=pose_axis[start:end, 0],
            pose_body=pose_axis[start:end, 1:22].reshape(end - start, 63),
            trans=trans[start:end],
        )
        chunks.append(joints[:, :22])
    joints22 = torch.cat(chunks, dim=0).reshape(batch_size, seq_len, 22, 3)
    return append_virtual_palm_joints(joints22, num_joints=24)


def _integrate_velocity(
    velocity: torch.Tensor,
    initial_position: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    position = initial_position[:, None].expand(-1, velocity.shape[1], -1).clone()
    if velocity.shape[1] > 1:
        position[:, 1:] += torch.cumsum(velocity[:, 1:] / float(fps), dim=1)
    return position


def object_velocity_to_world(
    method: str,
    velocity: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    local_pose: torch.Tensor,
    protocol_version: int,
    tip_variant: str = "legacy",
) -> tuple[torch.Tensor, str]:
    """Decode each corrected comparison method's object-velocity frame."""
    if protocol_version < 2:
        return velocity, "legacy_unspecified"
    if method == "tip":
        root0 = _root_orientation(batch)[:, :1]
        root0 = root0.expand(-1, velocity.shape[1], -1, -1)
        if tip_variant != "official":
            return (
                torch.einsum("bti,btji->btj", velocity, root0),
                "initial_root",
            )
        return (
            # Dataset normalization uses ``p_local = p_world @ R0.T``;
            # denormalization is therefore ``v_world = v_local @ R0``.
            torch.einsum("bti,btij->btj", velocity, root0),
            "initial_root",
        )
    if method == "globalpose":
        return (
            torch.einsum(
                "bti,btji->btj", velocity, local_pose[:, :, 0]
            ),
            "current_root",
        )
    return velocity, "world"


def _transpose_translation(
    output: Dict[str, torch.Tensor],
    local_pose: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    body_model,
    fps: float,
    protocol_version: int,
    transpose_variant: str = "legacy",
) -> torch.Tensor:
    initial = batch["trans"][:, 0]
    if protocol_version < 2 and transpose_variant != "official":
        return _integrate_velocity(output["root_vel"], initial, fps)

    root_rot = local_pose[:, :, 0]
    velocity_scale = (
        TRANSPOSE_VEL_SCALE if transpose_variant != "official" else 1.0
    )
    velocity_world = torch.einsum(
        "bti,btji->btj", output["root_vel"] * velocity_scale, root_rot
    )
    if transpose_variant == "official":
        velocity_world = velocity_world * TRANSPOSE_VEL_SCALE
    delta_b2 = velocity_world / float(fps)

    zero_trans = torch.zeros_like(batch["trans"])
    joints = body_joints_from_local_pose(local_pose, zero_trans, body_model)
    joints = joints - joints[:, :, :1]
    left_foot, right_foot = joints[:, :, 10], joints[:, :, 11]
    left_delta = torch.zeros_like(left_foot)
    right_delta = torch.zeros_like(right_foot)
    left_delta[:, 1:] = left_foot[:, :-1] - left_foot[:, 1:]
    right_delta[:, 1:] = right_foot[:, :-1] - right_foot[:, 1:]

    contact = output["contact_logits"].sigmoid()
    use_right = contact.argmax(dim=-1).bool().unsqueeze(-1)
    delta_b1 = torch.where(use_right, right_delta, left_delta)
    delta_b1[..., 1] += -0.018 * 60.0 / float(fps)
    weight = ((contact.max(dim=-1).values - 0.5) / 0.4).clamp(0.0, 1.0)
    delta = torch.lerp(delta_b2, delta_b1, weight.unsqueeze(-1))
    delta[:, 0] = 0.0

    root = initial[:, None].expand_as(delta).clone()
    floor = initial[:, 1] + torch.minimum(left_foot[:, 0, 1], right_foot[:, 0, 1])
    for frame in range(1, delta.shape[1]):
        root[:, frame] = root[:, frame - 1] + delta[:, frame]
        current_foot_y = root[:, frame, 1] + torch.minimum(
            left_foot[:, frame, 1], right_foot[:, frame, 1]
        )
        root[:, frame, 1] += (floor - current_foot_y).clamp_min(0.0)
    return root


def _globalpose_translation(
    output: Dict[str, torch.Tensor],
    local_pose: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    body_model,
    fps: float,
) -> torch.Tensor:
    vr = output["vr"]
    root_rot = local_pose[:, :, 0]
    velocity_world = torch.einsum("bti,btji->btj", vr[:, :, 1:4], root_rot)
    velocity_world[:, :, 1] = vr[:, :, 0]

    zero_trans = torch.zeros_like(batch["trans"])
    joints = body_joints_from_local_pose(local_pose, zero_trans, body_model)
    joints = joints - joints[:, :, :1]
    contact_joints = joints[:, :, [0, 10, 11, 22, 23]]
    stationary_weight = (vr[:, :, 4:9].sigmoid() * 5.0 - 3.0).clamp(0.0, 1.0)

    root = batch["trans"][:, :1].expand(-1, vr.shape[1], -1).clone()
    for frame in range(1, vr.shape[1]):
        correction = (
            contact_joints[:, frame - 1] - contact_joints[:, frame]
        ) * float(fps)
        weights = stationary_weight[:, frame]
        fused_velocity = (
            (weights.unsqueeze(-1) * correction).sum(dim=1)
            + velocity_world[:, frame]
        ) / (1.0 + weights.sum(dim=1, keepdim=True))
        root[:, frame] = root[:, frame - 1] + fused_velocity / float(fps)
    return root


def _decode_root_translation(
    method: str,
    output: Dict[str, torch.Tensor],
    local_pose: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    body_model,
    fps: float,
    protocol_version: int,
    tip_variant: str = "legacy",
    transpose_variant: str = "legacy",
) -> tuple[torch.Tensor, str]:
    if method == "dip18":
        return batch["trans"][:, :1].expand(-1, local_pose.shape[1], -1).clone(), "fixed_initial"
    if method == "tip":
        velocity = output["state"][:, :, 108:111]
        if tip_variant == "official":
            root0 = _root_orientation(batch)[:, :1]
            root0 = root0.expand(-1, velocity.shape[1], -1, -1)
            velocity = torch.einsum("bti,btij->btj", velocity, root0)
        return (
            _integrate_velocity(velocity, batch["trans"][:, 0], fps),
            "initial_root_velocity_no_bullet_sbp_correction"
            if tip_variant == "official"
            else "root_velocity_no_bullet_sbp_correction",
        )
    if method == "transpose":
        return (
            _transpose_translation(
                output,
                local_pose,
                batch,
                body_model,
                fps,
                protocol_version,
                transpose_variant=transpose_variant,
            ),
            "b1_b2_contact_fusion"
            if protocol_version >= 2 or transpose_variant == "official"
            else "legacy_velocity",
        )
    if method == "globalpose" and protocol_version >= 2:
        return (
            _globalpose_translation(output, local_pose, batch, body_model, fps),
            "vr_stationary_fusion_no_physics",
        )
    if method == "globalpose":
        vr = output["vr"]
        root_rot = local_pose[:, :, 0]
        velocity_world = torch.einsum("bti,btji->btj", vr[:, :, 1:4], root_rot)
        velocity_world[:, :, 1] = vr[:, :, 0]
        return (
            _integrate_velocity(velocity_world, batch["trans"][:, 0], fps),
            "legacy_vr_velocity_no_physics",
        )
    raise ValueError(method)


def infer_comparison(
    method: str,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    body_model,
    *,
    n_sbps: int = 5,
    fps: float = FRAME_RATE,
    protocol_version: int = 2,
    object_output_mode: str = "velocity",
    dip_past_frames: int = 20,
    dip_future_frames: int = 5,
    dip_window_batch_size: int = 128,
    tip_window: int = 40,
    transpose_past_frames: int = 20,
    transpose_future_frames: int = 5,
    transpose_window_batch_size: int = 256,
    dip_variant: str = "legacy",
    tip_variant: str = "legacy",
    transpose_variant: str = "legacy",
    globalpose_obj_imu_dim: int | None = None,
    globalpose_init_mode: str = "reference",
) -> Dict[str, torch.Tensor | str | bool]:
    """Run one complete sequence and decode method-specific predictions."""
    method = method.lower()
    adapted = adapt_batch(
        batch,
        method=method,
        n_sbps=n_sbps,
        fps=fps,
        protocol_version=protocol_version,
        dip_variant=dip_variant,
        tip_variant=tip_variant,
        transpose_variant=transpose_variant,
        globalpose_obj_imu_dim=(
            int(globalpose_obj_imu_dim)
            if globalpose_obj_imu_dim is not None
            else int(getattr(model, "obj_imu_dim", 12))
        ),
        globalpose_init_mode=(
            globalpose_init_mode if method == "globalpose" else "deploy"
        ),
    )
    if method == "dip18":
        if dip_variant == "official" and not hasattr(model, "reconstruct_chunks"):
            raise TypeError("Official DIP inference requires DIP18ReferenceHOIModel")
        output = model.reconstruct_chunks(
            adapted["imu"],
            adapted["obj_imu"],
            past_frames=dip_past_frames,
            future_frames=dip_future_frames,
            window_batch_size=dip_window_batch_size,
        )
    elif method == "tip":
        if protocol_version >= 2:
            initial_state = adapted["initial_state"]
            emit_initial_state = True
        else:
            initial_state = torch.zeros_like(adapted["state_target"][:, 0])
            emit_initial_state = False
        if tip_variant == "official":
            output = model.infer_autoregressive(
                adapted["imu"],
                window_size=tip_window,
                initial_state=initial_state,
                emit_initial_state=emit_initial_state,
            )
        else:
            output = model.infer_autoregressive(
                adapted["imu"],
                adapted["obj_imu"],
                window_size=tip_window,
                initial_state=initial_state,
                emit_initial_state=emit_initial_state,
            )
    elif method == "transpose":
        if protocol_version >= 2 or transpose_variant == "official":
            output = model.reconstruct_chunks(
                adapted["imu"],
                adapted["obj_imu"],
                past_frames=transpose_past_frames,
                future_frames=transpose_future_frames,
                window_batch_size=transpose_window_batch_size,
            )
        else:
            output = model(adapted["imu"], adapted["obj_imu"])
    elif method == "globalpose":
        output = model(
            adapted["x"],
            adapted["obj_imu"],
            apply_gravity_correction=protocol_version >= 2,
            column_r6d=protocol_version >= 2,
            pl_init=adapted["pl_init"] if protocol_version >= 2 else None,
            vr_init=adapted["vr_init"] if protocol_version >= 2 else None,
        )
    else:
        raise ValueError(f"Unknown comparison method: {method}")

    local_pose = _decode_local_pose(
        method,
        output,
        batch,
        protocol_version,
        dip_variant=dip_variant,
        tip_variant=tip_variant,
        transpose_variant=transpose_variant,
    )
    root_trans, translation_mode = _decode_root_translation(
        method,
        output,
        local_pose,
        batch,
        body_model,
        fps,
        protocol_version,
        tip_variant=tip_variant,
        transpose_variant=transpose_variant,
    )
    joints = body_joints_from_local_pose(local_pose, root_trans, body_model)
    target_local_pose = local_pose_axis_angle_to_rotmat(
        batch["pose"], num_joints=24
    )
    target_joints = body_joints_from_local_pose(
        target_local_pose, batch["trans"], body_model
    )
    global_pose = local_to_global_rotmats(local_pose)

    raw_obj_trans = output["obj_trans"]
    if object_output_mode == "velocity":
        object_velocity_world, object_velocity_frame = object_velocity_to_world(
            method,
            raw_obj_trans,
            batch,
            local_pose,
            protocol_version,
            tip_variant=tip_variant,
        )
        obj_trans = _integrate_velocity(
            object_velocity_world, adapted["obj_trans_init"][:, 0], fps
        )
    elif object_output_mode == "relative_displacement":
        object_velocity_frame = "not_applicable"
        obj_trans = (
            raw_obj_trans
            - raw_obj_trans[:, :1]
            + adapted["obj_trans_init"]
        )
    elif object_output_mode == "absolute":
        object_velocity_frame = "not_applicable"
        obj_trans = raw_obj_trans
    else:
        raise ValueError(
            "object_output_mode must be 'velocity', 'relative_displacement', or 'absolute'"
        )

    result: Dict[str, torch.Tensor | str | bool] = {
        "local_rot": local_pose,
        "global_rot": global_pose,
        "root_trans": root_trans,
        "joints": joints,
        "target_joints": target_joints,
        "obj_trans": obj_trans,
        "obj_trans_raw": raw_obj_trans,
        # Evaluation overrides this for human-only checkpoints.  Keeping the
        # default true preserves the inference API for trained object heads.
        "object_metrics_available": True,
        "object_velocity_frame": object_velocity_frame,
        "translation_mode": translation_mode,
        "initialization_mode": (
            "identity_tpose_zero_motion"
            if method == "tip" and protocol_version >= 2
            else (
                "official_first_frame_pose_zero_vr"
                if globalpose_init_mode == "reference"
                else "smpl_mean_tpose_joint_proxy_zero_velocity"
            )
            if method == "globalpose" and protocol_version >= 2
            else "legacy_model_default"
            if method in ("tip", "globalpose")
            else "not_applicable"
        ),
    }
    if "contact_logits" in output:
        result["contact_prob"] = output["contact_logits"].sigmoid()
    if method == "globalpose":
        result["stationary_prob"] = output["vr"][:, :, 4:9].sigmoid()
    return result
