"""Adapters from IMUHOI dataset batches to comparison-method protocols."""
from __future__ import annotations

from typing import Dict

import torch

from configs import FRAME_RATE, _SENSOR_POS_INDICES, _SENSOR_ROT_INDICES
from .geometry import (
    central_difference,
    local_pose_axis_angle_to_rotmat,
    object_imu_to_12d,
    root_relative_global_rot6d,
    rotation_angular_velocity,
    second_difference,
    select_and_flatten_rot6d,
    select_and_flatten_rotmat,
    sixd_imu_to_acc_rotmat,
)


# Current IMUHOI order is [root, left leg, right leg, head, left forearm, right forearm].
# TransPose and GlobalPose use root as the last IMU.  Their commonly used order in the
# checked reference code is [left forearm, right forearm, left leg, right leg, head, root].
ROOT_LAST_ORDER_FROM_IMUHOI = [4, 5, 1, 2, 3, 0]

DIP_REDUCED_15 = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
TIP_POSE_18 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]
TRANSPOSE_LEAF = [7, 8, 12, 20, 21]
TRANSPOSE_FULL = list(range(1, 24))
TRANSPOSE_REDUCED = DIP_REDUCED_15


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


def _dip_input(batch: Dict) -> torch.Tensor:
    human_imu = batch["human_imu"]
    acc, rot = sixd_imu_to_acc_rotmat(human_imu)
    # DIP's official input is root-relative non-root orientation matrices (5*9)
    # plus non-root accelerations (5*3).
    return torch.cat([rot[:, :, 1:].flatten(2), acc[:, :, 1:].flatten(2)], dim=-1)


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


def _transpose_input(batch: Dict, fps: float) -> torch.Tensor:
    glb_acc, glb_ori = _root_last_global_imu(batch, fps=fps)
    root_acc = glb_acc[:, :, 5:6]
    root_ori = glb_ori[:, :, 5]
    acc = torch.cat([glb_acc[:, :, :5] - root_acc, root_acc], dim=2)
    acc = torch.einsum("btsi,btij->btsj", acc, root_ori) / 30.0
    root_inv = root_ori.transpose(-1, -2).unsqueeze(2)
    ori = torch.cat([root_inv.matmul(glb_ori[:, :, :5]), glb_ori[:, :, 5:6]], dim=2)
    return torch.cat([acc.flatten(2), ori.flatten(2)], dim=-1)


def _globalpose_input(batch: Dict, fps: float) -> torch.Tensor:
    glb_acc, glb_ori = _root_last_global_imu(batch, fps=fps)
    glb_w = rotation_angular_velocity(glb_ori, dt=1.0 / fps)
    root_ori = glb_ori[:, :, 5]
    a_rb = torch.einsum("btsi,btij->btsj", glb_acc, root_ori)
    w_rb = torch.einsum("btsi,btij->btsj", glb_w, root_ori)
    r_rb = root_ori.transpose(-1, -2).unsqueeze(2).matmul(glb_ori[:, :, :5])
    g_r0 = -root_ori[:, :, 1]
    return torch.cat([a_rb.flatten(2), w_rb.flatten(2), r_rb.flatten(2), g_r0], dim=-1)


def _shift_state_right(state: torch.Tensor) -> torch.Tensor:
    prev = torch.zeros_like(state)
    if state.shape[1] > 1:
        prev[:, 1:] = state[:, :-1]
    return prev


def _object_targets(batch: Dict) -> Dict[str, torch.Tensor]:
    return {
        "obj_imu": object_imu_to_12d(batch["obj_imu"]),
        "obj_trans": batch["obj_trans"],
        "has_object": _has_object_mask(batch),
    }


def adapt_batch(batch: Dict, method: str, n_sbps: int = 5, fps: float = FRAME_RATE) -> Dict[str, torch.Tensor]:
    """Convert one IMUHOI batch into the requested comparison protocol."""
    method = method.lower()
    local_rot = local_pose_axis_angle_to_rotmat(batch["pose"], num_joints=24)
    pos, global_rot = _ensure_24_global(batch)
    obj = _object_targets(batch)

    if method == "dip18":
        pose_target = select_and_flatten_rotmat(local_rot, DIP_REDUCED_15)
        return {
            **obj,
            "imu": _dip_input(batch),
            "pose_target": pose_target,
        }

    if method == "tip":
        pose_18 = select_and_flatten_rot6d(local_rot, TIP_POSE_18)
        root_vel = batch.get("root_vel")
        if root_vel is None:
            root_vel = central_difference(batch["trans"], dt=1.0 / fps, dim=1)
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
        return {
            **obj,
            "imu": _human_imu_rotmat_input(batch, orientation_first=True),
            "prev_state": _shift_state_right(state_target.nan_to_num(0.0)),
            "state_target": state_target,
            "sbp_valid": sbp_valid,
            "n_sbps": torch.tensor(n_sbps, device=state_core.device),
        }

    if method == "transpose":
        root_pos = pos[:, :, :1]
        rel_pos = pos - root_pos
        leaf = rel_pos.index_select(2, torch.as_tensor(TRANSPOSE_LEAF, device=pos.device)).flatten(2)
        full = rel_pos.index_select(2, torch.as_tensor(TRANSPOSE_FULL, device=pos.device)).flatten(2)
        reduced_pose = root_relative_global_rot6d(global_rot, TRANSPOSE_REDUCED)
        root_vel = batch.get("root_vel")
        if root_vel is None:
            root_vel = central_difference(batch["trans"], dt=1.0 / fps, dim=1)
        contact = torch.stack(
            [
                batch.get("lfoot_contact", torch.zeros_like(root_vel[..., 0])).float(),
                batch.get("rfoot_contact", torch.zeros_like(root_vel[..., 0])).float(),
            ],
            dim=-1,
        )
        return {
            **obj,
            "imu": _transpose_input(batch, fps=fps),
            "leaf_target": leaf,
            "full_target": full,
            "pose_target": reduced_pose,
            "contact_target": contact,
            "root_vel_target": root_vel,
        }

    if method == "globalpose":
        root_rot = global_rot[:, :, 0]
        root_pos = pos[:, :, :1]
        rel = pos - root_pos
        gp_sensor_pos = rel.index_select(
            2,
            torch.as_tensor([20, 21, 7, 8, 15], device=pos.device, dtype=torch.long),
        )
        p_rb = torch.einsum("btsi,btij->btsj", gp_sensor_pos, root_rot).flatten(2)
        g_r = -root_rot[:, :, 1]
        p_rj = torch.einsum("btji,btni->btnj", root_rot, rel[:, :, 1:24]).flatten(2)
        rrj = root_relative_global_rot6d(global_rot, DIP_REDUCED_15)
        root_vel_world = batch.get("root_vel")
        if root_vel_world is None:
            root_vel_world = central_difference(batch["trans"], dt=1.0 / fps, dim=1)
        root_vel_local = torch.einsum("btji,bti->btj", root_rot, root_vel_world)
        velocity_target = torch.cat([root_vel_world[..., 1:2], root_vel_local], dim=-1)
        contact_target = torch.stack(
            [
                torch.zeros_like(root_vel_world[..., 0]),
                batch.get("lfoot_contact", torch.zeros_like(root_vel_world[..., 0])).float(),
                batch.get("rfoot_contact", torch.zeros_like(root_vel_world[..., 0])).float(),
                batch.get("lhand_contact", torch.zeros_like(root_vel_world[..., 0])).float(),
                batch.get("rhand_contact", torch.zeros_like(root_vel_world[..., 0])).float(),
            ],
            dim=-1,
        )
        gp_target = torch.cat([p_rb, g_r, p_rj, g_r, rrj, velocity_target, contact_target], dim=-1)
        return {
            **obj,
            "x": _globalpose_input(batch, fps=fps),
            "target": gp_target,
        }

    raise ValueError(f"Unknown comparison method: {method}")

