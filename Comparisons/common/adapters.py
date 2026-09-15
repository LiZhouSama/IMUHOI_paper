"""Adapters from IMUHOI dataset batches to comparison-method protocols."""
from __future__ import annotations

from typing import Dict

import torch

from configs import FRAME_RATE, _SENSOR_POS_INDICES, _SENSOR_ROT_INDICES
from utils.human_pose import SMPL_PARENTS
from utils.rotation_conversions import matrix_to_rotation_6d
from .geometry import (
    central_difference,
    local_pose_axis_angle_to_rotmat,
    object_imu_to_12d,
    root_relative_global_rot6d,
    rotation_matrix_to_tip_2axis,
    rotation_angular_velocity,
    second_difference,
    select_and_flatten_rot6d,
    select_and_flatten_r6d_columns,
    select_and_flatten_tip_2axis,
    select_and_flatten_rotmat,
    sixd_imu_to_acc_rotmat,
)


# Current IMUHOI order is [root, left leg, right leg, head, left forearm, right forearm].
# TransPose and GlobalPose use root as the last IMU.  Their commonly used order in the
# checked reference code is [left forearm, right forearm, left leg, right leg, head, root].
ROOT_LAST_ORDER_FROM_IMUHOI = [4, 5, 1, 2, 3, 0]
TIP_ORDER_FROM_IMUHOI = [0, 4, 5, 1, 2, 3]
DIP_NON_ROOT_ORDER_FROM_IMUHOI = [4, 5, 1, 2, 3]

DIP_REDUCED_15 = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
# TIP stores the root plus 17 active joints in NimblePhysics state order.
TIP_POSE_18 = [0, 1, 4, 7, 3, 6, 9, 13, 16, 18, 12, 15, 14, 17, 19, 2, 5, 8]
TIP_POSE_18_LEGACY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
# The public OMOMO TIP adapter uses the first 18 non-wrist SMPL-H joints in
# this order.  It is deliberately separate from the Nimble order used by the
# repository's corrected extension.
TIP_REFERENCE_POSE_18 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    12, 13, 14, 15, 16, 17, 18, 19,
]
TRANSPOSE_LEAF = [7, 8, 12, 20, 21]
TRANSPOSE_FULL = list(range(1, 24))
TRANSPOSE_REDUCED = DIP_REDUCED_15
TIP_ACC_SUM_WIN_LEN = 40
TIP_ACC_SUM_DOWN_SCALE = 15.0
TRANSPOSE_ACC_SCALE = 30.0
TRANSPOSE_VEL_SCALE = 3.0
GLOBALPOSE_IGNORED = [0, 7, 8, 10, 11, 20, 21, 22, 23]

# Official GlobalPose uses the five mesh-mounted IMU vertices
# (1961, 5424, 1176, 4662, 411), relative to vertex 3021 (the root IMU).
# These are the exact mean-shape SMPL-male T-pose offsets from the released
# ``models/SMPL_male.pkl``.  IMUHOI stores joints rather than mesh vertices, so
# the values are used as a deployable first-frame proxy for PL initialization.
GLOBALPOSE_TPOSE_P_RB_PROXY = (
    0.68417631,
    0.40220556,
    0.04039501,
    -0.68024387,
    0.40014841,
    0.03916638,
    0.07438047,
    -0.66392216,
    0.10829836,
    -0.07849202,
    -0.69400317,
    0.09693606,
    0.00283760,
    0.72230946,
    0.08952383,
)


def tensor_to_device(batch: Dict, device: torch.device) -> Dict:
    """Move tensor values in a DataLoader batch to device."""
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def _has_object_mask(batch: Dict) -> torch.Tensor:
    mask = batch.get("has_object")
    if torch.is_tensor(mask):
        return mask.bool()
    obj_trans = batch.get("obj_trans")
    if torch.is_tensor(obj_trans):
        return torch.ones(obj_trans.shape[0], device=obj_trans.device, dtype=torch.bool)
    raise KeyError("batch must contain has_object or obj_trans")


def _ensure_24_global(batch: Dict) -> tuple[torch.Tensor, torch.Tensor]:
    pos = batch["position_global"]
    rot = batch["rotation_global"]
    batch_size, seq_len = pos.shape[:2]
    if pos.shape[2] < 24:
        pos_pad = torch.zeros(batch_size, seq_len, 24 - pos.shape[2], 3, device=pos.device, dtype=pos.dtype)
        pos = torch.cat([pos, pos_pad], dim=2)
    if rot.shape[2] < 24:
        eye = torch.eye(3, device=rot.device, dtype=rot.dtype).view(1, 1, 1, 3, 3)
        rot_pad = eye.expand(batch_size, seq_len, 24 - rot.shape[2], 3, 3)
        rot = torch.cat([rot, rot_pad], dim=2)
    return pos[:, :, :24], rot[:, :, :24]


def _human_imu_rotmat_input(batch: Dict, orientation_first: bool = True) -> torch.Tensor:
    human_imu = batch["human_imu"]
    acc, rot = sixd_imu_to_acc_rotmat(human_imu)
    if orientation_first:
        return torch.cat([rot.flatten(2), acc.flatten(2)], dim=-1)
    return torch.cat([acc.flatten(2), rot.flatten(2)], dim=-1)


def _tip_acc_sum_features(imu: torch.Tensor) -> torch.Tensor:
    acc = imu[:, :, 6 * 9 : 6 * 9 + 18]
    cumsum = torch.cumsum(acc, dim=1)
    if cumsum.shape[1] > TIP_ACC_SUM_WIN_LEN:
        recent = cumsum[:, TIP_ACC_SUM_WIN_LEN:] - cumsum[:, :-TIP_ACC_SUM_WIN_LEN]
        acc_sum = torch.cat([cumsum[:, :TIP_ACC_SUM_WIN_LEN], recent], dim=1)
    else:
        acc_sum = cumsum
    return torch.cat([imu, acc_sum / TIP_ACC_SUM_DOWN_SCALE], dim=-1)


def _legacy_tip_input(batch: Dict) -> torch.Tensor:
    return _tip_acc_sum_features(
        _human_imu_rotmat_input(batch, orientation_first=True)
    )


def _tip_input(batch: Dict) -> torch.Tensor:
    acc_normalized, ori_normalized, acc_absolute_root, _ = _human_imu_protocol(batch)
    root_acc_world = torch.einsum(
        "bti,btji->btj", acc_normalized[:, :, 0], ori_normalized[:, :, 0]
    )
    acc = torch.cat(
        [root_acc_world.unsqueeze(2), acc_absolute_root[:, :, 1:]], dim=2
    )
    acc = acc[:, :, TIP_ORDER_FROM_IMUHOI]
    ori = ori_normalized[:, :, TIP_ORDER_FROM_IMUHOI]
    imu = torch.cat([ori.flatten(2), acc.flatten(2)], dim=-1)
    return _tip_acc_sum_features(imu)


def _legacy_dip_input(batch: Dict) -> torch.Tensor:
    human_imu = batch["human_imu"]
    acc, rot = sixd_imu_to_acc_rotmat(human_imu)
    return torch.cat([rot[:, :, 1:].flatten(2), acc[:, :, 1:].flatten(2)], dim=-1)


def _reference_dip_input(batch: Dict) -> torch.Tensor:
    """Official OMOMO DIP input: per sensor ``[orientation9, accel3]``."""
    acc, rot = sixd_imu_to_acc_rotmat(batch["human_imu"])
    per_sensor = torch.cat([rot.flatten(-2), acc], dim=-1)
    return per_sensor.flatten(2)


def _reference_object_imu(batch: Dict) -> torch.Tensor:
    """Official object branch feature order: ``[orientation9, acceleration3]``."""
    acc, rot = sixd_imu_to_acc_rotmat(batch["obj_imu"])
    return torch.cat([rot.flatten(-2), acc], dim=-1)


def _transpose_reference_input(batch: Dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact public TransPose human/object feature layout."""
    human = batch["human_imu"]
    if human.shape[-1] == 72:
        human_input = human
    elif human.shape[-1] == 9 and human.ndim == 4:
        # Defensive path for callers that retain per-sensor [acc3,rot6].
        acc, rot = sixd_imu_to_acc_rotmat(human)
        root_acc = acc[:, :, 5:6]
        acc = torch.cat((acc[:, :, :5] - root_acc, root_acc), dim=2)
        root_rot = rot[:, :, 5:6]
        ori = torch.cat(
            (root_rot.transpose(-1, -2).matmul(rot[:, :, :5]), root_rot), dim=2
        )
        human_input = torch.cat((acc.flatten(2) / TRANSPOSE_ACC_SCALE, ori.flatten(2)), dim=-1)
    else:
        raise ValueError(
            "Official TransPose expects a 72-channel flattened stream, got "
            f"{tuple(human.shape)}"
        )
    obj = batch["obj_imu"]
    if obj.ndim == 4 and obj.shape[2] == 1:
        obj = obj.squeeze(2)
    if obj.shape[-1] == 12:
        object_input = obj
    elif obj.shape[-1] == 9:
        # IMUHOI's ordinary object stream is [acc3, rot6].
        acc, rot = sixd_imu_to_acc_rotmat(obj)
        object_input = torch.cat((rot.flatten(-2), acc), dim=-1)
    else:
        raise ValueError(
            "Official TransPose expects a 12-channel object stream, got "
            f"{tuple(obj.shape)}"
        )
    return human_input.contiguous(), object_input.contiguous()


def _dip_input(batch: Dict) -> torch.Tensor:
    acc, rot = sixd_imu_to_acc_rotmat(batch["human_imu"])
    order = DIP_NON_ROOT_ORDER_FROM_IMUHOI
    return torch.cat(
        [rot[:, :, order].flatten(2), acc[:, :, order].flatten(2)], dim=-1
    )


def _root_last_global_imu(batch: Dict, fps: float) -> tuple[torch.Tensor, torch.Tensor]:
    pos, rot = _ensure_24_global(batch)
    pos_idx = torch.as_tensor(_SENSOR_POS_INDICES, device=pos.device, dtype=torch.long)
    rot_idx = torch.as_tensor(_SENSOR_ROT_INDICES, device=rot.device, dtype=torch.long)
    sensor_pos = pos.index_select(2, pos_idx)
    sensor_rot = rot.index_select(2, rot_idx)
    order = torch.as_tensor(ROOT_LAST_ORDER_FROM_IMUHOI, device=pos.device, dtype=torch.long)
    sensor_pos = sensor_pos.index_select(2, order)
    sensor_rot = sensor_rot.index_select(2, order)
    acc = second_difference(sensor_pos, dt=1.0 / fps, dim=1)
    return acc, sensor_rot


def _human_imu_protocol(
    batch: Dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover protocol signals from the measured ``human_imu`` tensor.

    IMUHOI stores the root orientation in the global frame. Other orientations
    are root-relative. All acceleration features are expressed in the root frame;
    non-root accelerations additionally subtract root acceleration.
    """
    acc_normalized, ori_normalized = sixd_imu_to_acc_rotmat(batch["human_imu"])
    root_acc = acc_normalized[:, :, :1]
    acc_absolute_root = torch.cat(
        [root_acc, acc_normalized[:, :, 1:] + root_acc], dim=2
    )

    root_ori = ori_normalized[:, :, :1]
    ori_global = torch.cat(
        [root_ori, root_ori.matmul(ori_normalized[:, :, 1:])], dim=2
    )
    return acc_normalized, ori_normalized, acc_absolute_root, ori_global


def measured_root_last_global_imu(
    batch: Dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover root-last world acceleration/orientation from measured IMU features."""
    _, _, acc_absolute_root, ori_global = _human_imu_protocol(batch)
    root_ori = ori_global[:, :, 0]
    acc_world = torch.einsum("btsi,btji->btsj", acc_absolute_root, root_ori)
    return (
        acc_world[:, :, ROOT_LAST_ORDER_FROM_IMUHOI],
        ori_global[:, :, ROOT_LAST_ORDER_FROM_IMUHOI],
    )


def _legacy_transpose_input(batch: Dict, fps: float) -> torch.Tensor:
    glb_acc, glb_ori = _root_last_global_imu(batch, fps=fps)
    root_acc = glb_acc[:, :, 5:6]
    root_ori = glb_ori[:, :, 5]
    acc = torch.cat([glb_acc[:, :, :5] - root_acc, root_acc], dim=2)
    acc = torch.einsum("btsi,btij->btsj", acc, root_ori) / TRANSPOSE_ACC_SCALE
    root_inv = root_ori.transpose(-1, -2).unsqueeze(2)
    ori = torch.cat(
        [root_inv.matmul(glb_ori[:, :, :5]), glb_ori[:, :, 5:6]], dim=2
    )
    return torch.cat([acc.flatten(2), ori.flatten(2)], dim=-1)


def _transpose_input(batch: Dict, fps: float) -> torch.Tensor:
    del fps
    acc_normalized, ori_normalized, _, _ = _human_imu_protocol(batch)
    acc = acc_normalized[:, :, ROOT_LAST_ORDER_FROM_IMUHOI] / TRANSPOSE_ACC_SCALE
    ori = ori_normalized[:, :, ROOT_LAST_ORDER_FROM_IMUHOI].clone()
    ori[:, :, -1] = ori_normalized[:, :, 0]
    return torch.cat([acc.flatten(2), ori.flatten(2)], dim=-1)


def _legacy_globalpose_input(batch: Dict, fps: float) -> torch.Tensor:
    glb_acc, glb_ori = _root_last_global_imu(batch, fps=fps)
    glb_w = rotation_angular_velocity(glb_ori, dt=1.0 / fps)
    root_ori = glb_ori[:, :, 5]
    a_rb = torch.einsum("btsi,btij->btsj", glb_acc, root_ori)
    w_rb = torch.einsum("btsi,btij->btsj", glb_w, root_ori)
    r_rb = root_ori.transpose(-1, -2).unsqueeze(2).matmul(glb_ori[:, :, :5])
    g_r0 = -root_ori[:, :, 1]
    return torch.cat([a_rb.flatten(2), w_rb.flatten(2), r_rb.flatten(2), g_r0], dim=-1)


def _globalpose_input(batch: Dict, fps: float) -> torch.Tensor:
    protocol = batch.get("imu_protocol", "")
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0] if protocol else ""
    if str(protocol).lower() == "globalpose_reference":
        # The official GlobalPose synthesis uses [elbow, elbow, knee, knee,
        # head, root] with the root sensor last.  The dataset stores this
        # stream as root-relative [acc3, rot6], so recover the model-frame
        # signals without consulting pose/position ground truth.
        a_rb, relative_rot = sixd_imu_to_acc_rotmat(batch["human_imu"])
        # The dataset builder starts from kinematic world acceleration, which
        # is already equivalent to gp_training/data.py's post-processed aM
        # (free acceleration plus gravity). Do not add gravity a second time.
        root_rot = relative_rot[:, :, 5]
        global_rot = torch.cat(
            [
                root_rot.unsqueeze(2).matmul(relative_rot[:, :, :5]),
                root_rot.unsqueeze(2),
            ],
            dim=2,
        )
        dt = 1.0 / float(fps)
        rdot = torch.zeros_like(global_rot)
        if global_rot.shape[1] > 1:
            if global_rot.shape[1] >= 4:
                rdot[:, 1:-1] = (global_rot[:, 2:] - global_rot[:, :-2]) / (2.0 * dt)
                rdot[:, 0] = (
                    -3.0 * global_rot[:, 0]
                    + 4.0 * global_rot[:, 1]
                    - global_rot[:, 2]
                ) / (2.0 * dt)
                rdot[:, -1] = (
                    3.0 * global_rot[:, -1]
                    - 4.0 * global_rot[:, -2]
                    + global_rot[:, -3]
                ) / (2.0 * dt)
            else:
                rdot[:, 0] = (global_rot[:, 1] - global_rot[:, 0]) / dt
                rdot[:, -1] = rdot[:, 0]
        w_hat = global_rot.transpose(-1, -2).matmul(rdot)
        w_hat = (w_hat - w_hat.transpose(-1, -2)) * 0.5
        local_w = torch.stack(
            (w_hat[..., 2, 1], w_hat[..., 0, 2], w_hat[..., 1, 0]), dim=-1
        )
        world_w = torch.einsum("btsij,btsj->btsi", global_rot, local_w)
        w_rb = torch.einsum(
            "btsi,btij->btsj", world_w, root_rot
        )
        g_r0 = -root_rot[:, :, 1]
        return torch.cat(
            [a_rb.flatten(2), w_rb.flatten(2), relative_rot[:, :, :5].flatten(2), g_r0],
            dim=-1,
        )
    _, ori_normalized, acc_absolute_root, ori_global = _human_imu_protocol(batch)
    a_rb = acc_absolute_root[:, :, ROOT_LAST_ORDER_FROM_IMUHOI]
    glb_ori = ori_global[:, :, ROOT_LAST_ORDER_FROM_IMUHOI]
    root_ori = glb_ori[:, :, -1]
    local_w = rotation_angular_velocity(glb_ori, dt=1.0 / fps)
    world_w = torch.einsum("btsi,btsji->btsj", local_w, glb_ori)
    w_rb = torch.einsum("btsi,btij->btsj", world_w, root_ori)
    r_rb = ori_normalized[:, :, ROOT_LAST_ORDER_FROM_IMUHOI[:5]]
    g_r0 = -root_ori[:, :, 1]
    return torch.cat([a_rb.flatten(2), w_rb.flatten(2), r_rb.flatten(2), g_r0], dim=-1)


def _shift_state_right(state: torch.Tensor) -> torch.Tensor:
    prev = torch.zeros_like(state)
    if state.shape[1] > 1:
        prev[:, 1:] = state[:, :-1]
    return prev


def _tip_tpose_initial_state(
    reference: torch.Tensor, n_sbps: int
) -> torch.Tensor:
    """Build TIP's deployable identity-pose/zero-motion initial state."""
    batch_size = reference.shape[0]
    state = reference.new_zeros(batch_size, 111 + n_sbps * 4)
    identity = torch.eye(3, device=reference.device, dtype=reference.dtype)
    pose = rotation_matrix_to_tip_2axis(identity).repeat(18)
    state[:, :108] = pose
    return state


def _tip_reference_initial_state(reference: torch.Tensor) -> torch.Tensor:
    """Identity raw-row 6D state used before the first official TIP frame."""
    state = reference.new_zeros(reference.shape[0], 114)
    identity6 = reference.new_tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    state[:, :108] = identity6.repeat(18)
    return state


def _globalpose_canonical_positions(
    local_rot: torch.Tensor,
    global_rot: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """FK joints after applying GlobalPose's ignored-joint canonicalization."""
    joint_count = min(local_rot.shape[2], global_rot.shape[2], positions.shape[2], 24)
    bone_local = torch.zeros_like(positions[:, :, :joint_count])
    for joint in range(1, joint_count):
        parent = SMPL_PARENTS[joint]
        delta_world = positions[:, :, joint] - positions[:, :, parent]
        bone_local[:, :, joint] = torch.einsum(
            "bti,btij->btj", delta_world, global_rot[:, :, parent]
        )
    # A sequence has one body shape; averaging suppresses numerical FK noise.
    bone_local = bone_local.mean(dim=1, keepdim=True)

    canonical_local = local_rot[:, :, :joint_count].clone()
    eye = torch.eye(3, device=local_rot.device, dtype=local_rot.dtype)
    ignored = [joint for joint in GLOBALPOSE_IGNORED if joint < joint_count]
    canonical_local[:, :, ignored] = eye

    canonical_global = torch.empty_like(canonical_local)
    canonical_position = torch.zeros_like(positions[:, :, :joint_count])
    canonical_global[:, :, 0] = canonical_local[:, :, 0]
    for joint in range(1, joint_count):
        parent = SMPL_PARENTS[joint]
        canonical_global[:, :, joint] = canonical_global[:, :, parent].matmul(
            canonical_local[:, :, joint]
        )
        bone_world = torch.einsum(
            "bti,btji->btj", bone_local[:, :, joint].expand(-1, local_rot.shape[1], -1), canonical_global[:, :, parent]
        )
        canonical_position[:, :, joint] = canonical_position[:, :, parent] + bone_world
    return canonical_position


def _object_targets(
    batch: Dict, fps: float, object_imu_dim: int = 12
) -> Dict[str, torch.Tensor]:
    obj_trans = batch["obj_trans"]
    obj_trans_init = obj_trans[:, :1]
    obj_velocity = torch.zeros_like(obj_trans)
    if obj_velocity.shape[1] > 1:
        obj_velocity[:, 1:] = (
            obj_trans[:, 1:] - obj_trans[:, :-1]
        ) * float(fps)
    object_imu = batch["obj_imu"]
    if object_imu.ndim == 4 and object_imu.shape[2] == 1:
        object_imu = object_imu.squeeze(2)
    if int(object_imu_dim) == 9:
        # GlobalPose's released object extension consumes the original
        # [acc3, rot6] stream (243 + 9 = 252).
        object_input = object_imu
    elif int(object_imu_dim) == 12:
        object_input = object_imu_to_12d(object_imu)
    else:
        raise ValueError(f"Unsupported GlobalPose object IMU width: {object_imu_dim}")
    return {
        "obj_imu": object_input,
        "obj_trans": obj_trans,
        "obj_trans_init": obj_trans_init,
        "obj_trans_target": obj_trans - obj_trans_init,
        "obj_vel_target": obj_velocity,
        "has_object": _has_object_mask(batch),
    }


def _object_imu_initial_root(
    batch: Dict, root_orientation: torch.Tensor
) -> torch.Tensor:
    """Build TIP's object sensor plus rolling acceleration-sum feature."""
    obj_acc_world, obj_orientation_world = sixd_imu_to_acc_rotmat(batch["obj_imu"])
    root0 = root_orientation[:, 0]
    obj_acc_root0 = torch.einsum("bti,bij->btj", obj_acc_world, root0)
    obj_orientation_root0 = root0.transpose(-1, -2).unsqueeze(1).matmul(
        obj_orientation_world
    )
    cumsum = torch.cumsum(obj_acc_root0, dim=1)
    if cumsum.shape[1] > TIP_ACC_SUM_WIN_LEN:
        recent = (
            cumsum[:, TIP_ACC_SUM_WIN_LEN:]
            - cumsum[:, :-TIP_ACC_SUM_WIN_LEN]
        )
        acceleration_sum = torch.cat(
            [cumsum[:, :TIP_ACC_SUM_WIN_LEN], recent], dim=1
        )
    else:
        acceleration_sum = cumsum
    return torch.cat(
        [
            obj_acc_root0,
            rotation_matrix_to_tip_2axis(obj_orientation_root0),
            acceleration_sum / TIP_ACC_SUM_DOWN_SCALE,
        ],
        dim=-1,
    )


def adapt_batch(
    batch: Dict,
    method: str,
    n_sbps: int = 5,
    fps: float = FRAME_RATE,
    protocol_version: int = 2,
    tip_history_context: int = 0,
    dip_variant: str = "legacy",
    tip_variant: str = "legacy",
    transpose_variant: str = "legacy",
    globalpose_obj_imu_dim: int = 12,
    globalpose_init_mode: str = "deploy",
) -> Dict[str, torch.Tensor]:
    """Convert one IMUHOI batch into the requested comparison protocol."""
    method = method.lower()
    corrected_protocol = int(protocol_version) >= 2
    local_rot = local_pose_axis_angle_to_rotmat(batch["pose"], num_joints=24)
    pos, global_rot = _ensure_24_global(batch)
    obj = _object_targets(
        batch,
        fps=fps,
        object_imu_dim=(globalpose_obj_imu_dim if method == "globalpose" else 12),
    )

    if method == "dip18":
        if dip_variant not in ("legacy", "official"):
            raise ValueError(f"Unknown DIP variant: {dip_variant!r}")
        if dip_variant == "official":
            pose_target = local_rot[:, :, :22].flatten(2)
            obj_velocity = central_difference(
                batch["obj_trans"], dt=1.0 / float(fps), dim=1
            )
            obj = {
                **obj,
                "obj_imu": _reference_object_imu(batch),
                "obj_vel_target": obj_velocity,
                "obj_trans_target": batch["obj_trans"] - batch["obj_trans"][:, :1],
            }
            return {
                **obj,
                "imu": _reference_dip_input(batch),
                "pose_target": pose_target,
            }
        pose_target = select_and_flatten_rotmat(local_rot, DIP_REDUCED_15)
        return {
            **obj,
            "imu": _dip_input(batch) if corrected_protocol else _legacy_dip_input(batch),
            "pose_target": pose_target,
        }

    if method == "tip":
        if tip_variant not in ("legacy", "official"):
            raise ValueError(f"Unknown TIP variant: {tip_variant!r}")
        if tip_variant == "official":
            # RefCodes/transformer-inertial-poser concatenates six human IMUs
            # and one object IMU, each as [acc3, 3x2 orientation].
            human_imu = batch["human_imu"]
            object_imu = batch["obj_imu"]
            if object_imu.ndim == 4 and object_imu.shape[2] == 1:
                object_imu = object_imu.squeeze(2)
            imu = torch.cat([human_imu, object_imu.unsqueeze(2)], dim=2).flatten(2)
            if imu.shape[-1] != 63:
                raise ValueError(
                    "Official TIP requires six human + one object IMU (63 channels), "
                    f"got {imu.shape[-1]}"
                )

            raw_local_6d = batch.get("rotation_local_6d")
            if raw_local_6d is None:
                raw_local_6d = matrix_to_rotation_6d(
                    local_rot[:, :, :22].reshape(-1, 3, 3)
                ).reshape(local_rot.shape[0], local_rot.shape[1], 22, 6)
            else:
                if raw_local_6d.ndim == 3:
                    raw_local_6d = raw_local_6d.reshape(
                        raw_local_6d.shape[0], raw_local_6d.shape[1], -1, 6
                    )
                raw_local_6d = raw_local_6d.to(device=local_rot.device, dtype=local_rot.dtype)
            pose_target = raw_local_6d.index_select(
                2,
                torch.as_tensor(
                    TIP_REFERENCE_POSE_18,
                    device=raw_local_6d.device,
                    dtype=torch.long,
                ),
            ).flatten(2)
            root_vel = batch.get("tip_root_vel")
            if root_vel is None:
                root_vel = torch.zeros_like(batch["trans"])
                if root_vel.shape[1] > 1:
                    root_vel[:, 1:] = (
                        batch["trans"][:, 1:] - batch["trans"][:, :-1]
                    ) * float(fps)
            obj_vel = batch.get("tip_obj_vel")
            if obj_vel is None:
                obj_vel = torch.zeros_like(batch["obj_trans"])
                if obj_vel.shape[1] > 1:
                    obj_vel[:, 1:] = (
                        batch["obj_trans"][:, 1:] - batch["obj_trans"][:, :-1]
                    ) * float(fps)
            state_target = torch.cat([pose_target, root_vel, obj_vel], dim=-1)
            if state_target.shape[-1] != 114:
                raise ValueError(f"Official TIP state must have 114 channels, got {state_target.shape[-1]}")
            return {
                **obj,
                "imu": imu,
                "obj_imu": object_imu,
                "obj_vel_target": obj_vel,
                "obj_trans_init": batch["obj_trans"][:, :1],
                "obj_trans_target": batch["obj_trans"] - batch["obj_trans"][:, :1],
                "state_target": state_target,
                "prev_state": _shift_state_right(state_target.nan_to_num(0.0)),
                "initial_state": _tip_reference_initial_state(state_target),
                "sbp_valid": torch.zeros(
                    *state_target.shape[:2],
                    device=state_target.device,
                    dtype=torch.bool,
                ),
                "n_sbps": torch.tensor(0, device=state_target.device),
                "tip_variant": "official",
            }
        if tip_history_context < 0 or tip_history_context >= local_rot.shape[1]:
            raise ValueError(
                "tip_history_context must be in [0, sequence_length), got "
                f"{tip_history_context} for length {local_rot.shape[1]}"
            )
        pose_18 = (
            select_and_flatten_tip_2axis(local_rot, TIP_POSE_18)
            if corrected_protocol
            else select_and_flatten_rot6d(local_rot, TIP_POSE_18_LEGACY)
        )
        root_vel = torch.zeros_like(batch["trans"])
        if root_vel.shape[1] > 1:
            root_vel[:, 1:] = (
                batch["trans"][:, 1:] - batch["trans"][:, :-1]
            ) * float(fps)
        if corrected_protocol:
            _, measured_orientation = sixd_imu_to_acc_rotmat(batch["human_imu"])
            root0 = measured_orientation[:, :1, 0]
            obj = {
                **obj,
                "obj_imu": _object_imu_initial_root(
                    batch, measured_orientation[:, :, 0]
                ),
                "obj_vel_target": torch.einsum(
                    "bti,btij->btj",
                    obj["obj_vel_target"],
                    root0.expand(-1, obj["obj_vel_target"].shape[1], -1, -1),
                ),
            }
        state_core = torch.cat([pose_18, root_vel], dim=-1)
        sbp = batch.get("tip_sbp")
        sbp_valid = None
        if sbp is None:
            sbp = torch.full(
                (*state_core.shape[:2], n_sbps * 4),
                float("nan"),
                device=state_core.device,
                dtype=state_core.dtype,
            )
            sbp_valid = torch.zeros(*state_core.shape[:2], device=state_core.device, dtype=torch.bool)
        else:
            sbp_valid = ~torch.isnan(sbp).any(dim=-1)
        state_target = torch.cat([state_core, sbp], dim=-1)
        imu = _tip_input(batch) if corrected_protocol else _legacy_tip_input(batch)
        if tip_history_context:
            start = int(tip_history_context)
            imu = imu[:, start:]
            state_target = state_target[:, start:]
            sbp_valid = sbp_valid[:, start:]
            obj = {
                **obj,
                "obj_imu": obj["obj_imu"][:, start:],
                "obj_trans": obj["obj_trans"][:, start:],
                "obj_vel_target": obj["obj_vel_target"][:, start:],
            }
            obj["obj_trans_init"] = obj["obj_trans"][:, :1]
            obj["obj_trans_target"] = obj["obj_trans"] - obj["obj_trans_init"]
        return {
            **obj,
            "imu": imu,
            "prev_state": _shift_state_right(state_target.nan_to_num(0.0)),
            "state_target": state_target,
            "initial_state": _tip_tpose_initial_state(state_target, n_sbps),
            "sbp_valid": sbp_valid,
            "n_sbps": torch.tensor(n_sbps, device=state_core.device),
        }

    if method == "transpose":
        if transpose_variant not in ("legacy", "official"):
            raise ValueError(f"Unknown TransPose variant: {transpose_variant!r}")
        if transpose_variant == "official":
            human_input, object_input = _transpose_reference_input(batch)
            root_rot = global_rot[:, :, 0]
            root_inv = root_rot.transpose(-1, -2)
            rel_pos = torch.einsum(
                "btij,btkj->btki", root_inv, pos - pos[:, :, :1]
            )
            leaf = rel_pos.index_select(
                2, torch.as_tensor(TRANSPOSE_LEAF, device=pos.device)
            ).flatten(2)
            full = rel_pos.index_select(
                2, torch.as_tensor(TRANSPOSE_FULL, device=pos.device)
            ).flatten(2)
            pose_target = root_relative_global_rot6d(
                global_rot, TRANSPOSE_REDUCED, column_based=False
            )
            root_position_world = pos[:, :, 0]
            root_vel_world = central_difference(
                root_position_world, dt=1.0 / float(fps), dim=1
            )
            root_rot = global_rot[:, :, 0]
            root_vel_local = torch.einsum(
                "bti,btij->btj", root_vel_world, root_rot
            )
            obj_velocity = central_difference(
                batch["obj_trans"], dt=1.0 / float(fps), dim=1
            )
            contact = torch.stack(
                [
                    batch.get("lfoot_contact", torch.zeros_like(root_vel_world[..., 0])).float(),
                    batch.get("rfoot_contact", torch.zeros_like(root_vel_world[..., 0])).float(),
                ],
                dim=-1,
            )
            return {
                **obj,
                "imu": human_input,
                "obj_imu": object_input,
                "obj_vel_target": obj_velocity,
                "obj_trans": batch["obj_trans"],
                "obj_trans_init": batch["obj_trans"][:, :1],
                "obj_trans_target": batch["obj_trans"] - batch["obj_trans"][:, :1],
                "leaf_target": leaf,
                "full_target": full,
                "pose_target": pose_target,
                "contact_target": contact,
                "root_vel_target": root_vel_local,
                "root_vel_local_target": root_vel_local,
                "root_vel_world_target": root_vel_world,
                "root_position_target": root_position_world - root_position_world[:, :1],
                "transpose_variant": "official",
            }
        root_pos = pos[:, :, :1]
        root_rot = global_rot[:, :, 0]
        rel_pos_world = pos - root_pos
        rel_pos = torch.einsum("btji,btik->btjk", rel_pos_world, root_rot)
        leaf = rel_pos.index_select(2, torch.as_tensor(TRANSPOSE_LEAF, device=pos.device)).flatten(2)
        full = rel_pos.index_select(2, torch.as_tensor(TRANSPOSE_FULL, device=pos.device)).flatten(2)
        reduced_pose = root_relative_global_rot6d(
            global_rot,
            TRANSPOSE_REDUCED,
            column_based=corrected_protocol,
        )
        root_vel_world = torch.zeros_like(batch["trans"])
        if root_vel_world.shape[1] > 1:
            root_vel_world[:, 1:] = (
                batch["trans"][:, 1:] - batch["trans"][:, :-1]
            ) * float(fps)
        root_vel_local = torch.einsum(
            "bti,btij->btj", root_vel_world, root_rot
        )
        root_vel = root_vel_local / TRANSPOSE_VEL_SCALE
        contact = torch.stack(
            [
                batch.get("lfoot_contact", torch.zeros_like(root_vel[..., 0])).float(),
                batch.get("rfoot_contact", torch.zeros_like(root_vel[..., 0])).float(),
            ],
            dim=-1,
        )
        return {
            **obj,
            "imu": (
                _transpose_input(batch, fps=fps)
                if corrected_protocol
                else _legacy_transpose_input(batch, fps=fps)
            ),
            "leaf_target": leaf,
            "full_target": full,
            "pose_target": reduced_pose,
            "contact_target": contact,
            "root_vel_target": root_vel,
            "root_vel_local_target": root_vel_local,
            "root_vel_world_target": root_vel_world,
            "root_position_target": batch["trans"] - batch["trans"][:, :1],
        }

    if method == "globalpose":
        if globalpose_init_mode not in ("deploy", "reference", "reference_train"):
            raise ValueError(
                "globalpose_init_mode must be 'deploy', 'reference', or "
                "'reference_train', got "
                f"{globalpose_init_mode!r}"
            )
        root_rot = global_rot[:, :, 0]
        root_pos = pos[:, :, :1]
        rel = pos - root_pos
        exact_sensor_pos = batch.get("globalpose_sensor_positions")
        if torch.is_tensor(exact_sensor_pos):
            if exact_sensor_pos.ndim != 4 or exact_sensor_pos.shape[2:] != (6, 3):
                raise ValueError(
                    "globalpose_sensor_positions must have shape [B,T,6,3], "
                    f"got {tuple(exact_sensor_pos.shape)}"
                )
            # The final entry is the root IMU vertex.  Keep the full world
            # positions here and transform only the five relative vectors.
            gp_sensor_pos = exact_sensor_pos[:, :, :5] - exact_sensor_pos[:, :, 5:6]
        else:
            gp_sensor_pos = rel.index_select(
                2,
                torch.as_tensor([20, 21, 7, 8, 15], device=pos.device, dtype=torch.long),
            )
        p_rb = torch.einsum("btsi,btij->btsj", gp_sensor_pos, root_rot).flatten(2)
        g_r = -root_rot[:, :, 1]
        if corrected_protocol:
            exact_canonical = batch.get("globalpose_canonical_joints")
            if torch.is_tensor(exact_canonical):
                canonical_pos = exact_canonical
                if canonical_pos.shape[2] < 24:
                    pad = torch.zeros(
                        *canonical_pos.shape[:2],
                        24 - canonical_pos.shape[2],
                        3,
                        device=canonical_pos.device,
                        dtype=canonical_pos.dtype,
                    )
                    canonical_pos = torch.cat([canonical_pos, pad], dim=2)
            else:
                canonical_pos = _globalpose_canonical_positions(
                    local_rot, global_rot, pos
                )
            p_rj = canonical_pos[:, :, 1:24].flatten(2)
        else:
            p_rj = torch.einsum(
                "btni,btij->btnj", rel[:, :, 1:24], root_rot
            ).flatten(2)
        rrj = root_relative_global_rot6d(
            global_rot,
            DIP_REDUCED_15,
            column_based=corrected_protocol,
        )
        root_vel_world = torch.zeros_like(batch["trans"])
        if root_vel_world.shape[1] > 1:
            root_vel_world[:, 1:] = (
                batch["trans"][:, 1:] - batch["trans"][:, :-1]
            ) * float(fps)
        root_vel_local = torch.einsum("bti,btij->btj", root_vel_world, root_rot)
        obj_vel_root = torch.einsum(
            "bti,btij->btj", obj["obj_vel_target"], root_rot
        )
        velocity_target = torch.cat([root_vel_world[..., 1:2], root_vel_local], dim=-1)
        exact_contact = batch.get("globalpose_contact_joints")
        if torch.is_tensor(exact_contact):
            contact_pos = exact_contact.index_select(
                2,
                torch.as_tensor(
                    [0, 10, 11, 22, 23],
                    device=exact_contact.device,
                    dtype=torch.long,
                ),
            )
        else:
            contact_indices = torch.as_tensor(
                [0, 10, 11, 22, 23], device=pos.device, dtype=torch.long
            )
            contact_pos = pos.index_select(2, contact_indices)
        contact_vel = torch.zeros_like(contact_pos)
        if contact_vel.shape[1] > 1:
            contact_vel[:, 1:] = (
                contact_pos[:, 1:] - contact_pos[:, :-1]
            ) * float(fps)
        contact_speed = torch.linalg.norm(contact_vel, dim=-1)
        ground = contact_pos[:, :, :, 1].min(dim=2, keepdim=True).values
        near_ground = (contact_pos[:, :, :, 1] - ground) < 0.06
        stationary_target = ((contact_speed < 0.35) & near_ground).float()
        gp_target = torch.cat(
            [p_rb, g_r, p_rj, g_r, rrj, velocity_target, stationary_target],
            dim=-1,
        )
        gp_input = (
            _globalpose_input(batch, fps=fps)
            if corrected_protocol
            else _legacy_globalpose_input(batch, fps=fps)
        )
        if globalpose_init_mode in ("reference", "reference_train"):
            # RefCodes/GlobalPose trains each TBPTT segment with the first
            # target frame as the RNNWithInit input.  Keep this explicit and
            # separate from deployable online initialization below.
            pl_init = gp_target[:, 0, :18]
            if globalpose_init_mode == "reference_train":
                # The reference batched forward passes the final nine values
                # of the complete target vector to VR's RNNWithInit (see
                # RefCodes/GlobalPose/net.py::forward), i.e. velocity and
                # stationary-contact channels.  The online test script does
                # not have these labels and initializes VR from zero instead.
                vr_init = gp_target[:, 0, -9:]
            else:
                vr_init = torch.zeros_like(gp_target[:, 0, -9:])
        else:
            tpose_p_rb = gp_target.new_tensor(GLOBALPOSE_TPOSE_P_RB_PROXY)
            tpose_p_rb = tpose_p_rb.unsqueeze(0).expand(gp_target.shape[0], -1)
            pl_init = torch.cat([tpose_p_rb, gp_input[:, 0, 81:84]], dim=-1)
            vr_init = torch.zeros_like(gp_target[:, 0, 180:189])
        return {
            **obj,
            "obj_vel_target": (
                obj_vel_root if corrected_protocol else obj["obj_vel_target"]
            ),
            "x": gp_input,
            "target": gp_target,
            "pl_init": pl_init,
            "vr_init": vr_init,
            "globalpose_init_mode": globalpose_init_mode,
        }

    raise ValueError(f"Unknown comparison method: {method}")
