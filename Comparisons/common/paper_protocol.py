"""Paper-faithful comparison protocol (v3) on the shared IMUHOI IMU stream.

Every baseline consumes the identical measured stream used by the main IMUHOI
pipeline (``human_imu``/``obj_imu`` of the processed corpus, noised once per
sequence with the main pipeline's seed scheme).  The stream stores, for the
six body sensors in IMUHOI order [root, left lower leg, right lower leg, head,
left forearm, right forearm]:

* root: acceleration in the root frame and the global root orientation;
* others: ``R_root^T (a_i - a_root)`` and ``R_root^T R_i``.

From it this module rebuilds each method's own input layout and
normalization as defined by its paper and released code, the training
targets, and the decoding of network outputs.  Object IMU inputs and object
velocity outputs are deliberate project extensions.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

from configs import FRAME_RATE
from utils.human_pose import SMPL_PARENTS
from utils.rotation_conversions import matrix_to_axis_angle
from .adapters import (
    DIP_REDUCED_15,
    GLOBALPOSE_IGNORED,
    ROOT_LAST_ORDER_FROM_IMUHOI,
    TIP_POSE_18,
    TRANSPOSE_FULL,
    TRANSPOSE_LEAF,
    TRANSPOSE_REDUCED,
    _ensure_24_global,
    _globalpose_canonical_positions,
    _globalpose_input,
    _has_object_mask,
)
from .geometry import (
    local_pose_axis_angle_to_rotmat,
    r6d_columns_to_rotation_matrix,
    root_relative_global_rot6d,
    rotation_matrix_to_tip_2axis,
    sixd_imu_to_acc_rotmat,
    tip_2axis_to_rotation_matrix,
)

PAPER_PROTOCOL_VERSION = 3

# DIP feeds the five non-root sensors (lower arms, lower legs, head).
DIP_SENSOR_ORDER = [4, 5, 1, 2, 3]
# TIP: root first, then left/right forearm, left/right lower leg, head.
TIP_SENSOR_ORDER = [0, 4, 5, 1, 2, 3]
TIP_ACC_SMOOTH_WINDOW = 11  # constants.ACC_MOVING_AVE_LEN (5 past + 5 future)
TIP_ACC_SUM_WINDOW = 40
TIP_ACC_SUM_SCALE = 15.0
TIP_ACC_BIAS_NOISE = 0.1  # constants.BIAS_NOISE_ACC, training only
TIP_STATE_N_SBPS = 5
# Training windows carry history for the 40-frame acc sum and the smoothing
# filter, plus five future frames for the centred moving average.
TIP_TRAIN_HISTORY = TIP_ACC_SUM_WINDOW - 1 + TIP_ACC_SMOOTH_WINDOW // 2
TIP_TRAIN_FUTURE = TIP_ACC_SMOOTH_WINDOW // 2
# Median height of the foot SBP points on stationary frames of the canonical
# OMOMO training split (floor convention: lowest foot joint at y = 0).
TIP_GROUND_SBP_HEIGHT = 0.028
# The same statistic on each dataset's training split (2026-10-23); the floor
# conventions and skeletons differ, so each test set uses its own value.
TIP_GROUND_SBP_HEIGHTS = {
    "omomo": TIP_GROUND_SBP_HEIGHT,
    "behave": 0.0708,
    "imhd": 0.0681,
    "pahoi": 0.0946,
    "hodome": 0.0849,
}
TRANSPOSE_ORDER = ROOT_LAST_ORDER_FROM_IMUHOI  # [L arm, R arm, L leg, R leg, head, root]
GLOBALPOSE_CONTACT_JOINTS = [0, 10, 11, 22, 23]
GLOBALPOSE_STATIONARY_SPEED = 0.15  # m/s, TIP's stationary-point threshold


def common_stream(batch: Dict) -> Dict[str, torch.Tensor]:
    """Decompose the measured IMUHOI stream into frames shared by all methods."""
    acc_norm, ori_norm = sixd_imu_to_acc_rotmat(batch["human_imu"])
    root = ori_norm[:, :, 0]
    acc_root_abs = torch.cat(
        [acc_norm[:, :, :1], acc_norm[:, :, 1:] + acc_norm[:, :, :1]], dim=2
    )
    ori_global = torch.cat(
        [ori_norm[:, :, :1], root.unsqueeze(2).matmul(ori_norm[:, :, 1:])], dim=2
    )
    acc_world = torch.einsum("btij,btsj->btsi", root, acc_root_abs)
    obj = batch["obj_imu"]
    if obj.ndim == 4 and obj.shape[2] == 1:
        obj = obj.squeeze(2)
    obj_acc_world, obj_rot = sixd_imu_to_acc_rotmat(obj)
    return {
        "acc_norm": acc_norm,
        "ori_norm": ori_norm,
        "root": root,
        "acc_root_abs": acc_root_abs,
        "ori_global": ori_global,
        "acc_world": acc_world,
        "obj_acc_world": obj_acc_world,
        "obj_rot": obj_rot,
    }


def _backward_velocity(position: torch.Tensor, fps: float) -> torch.Tensor:
    velocity = torch.zeros_like(position)
    if position.shape[1] > 1:
        velocity[:, 1:] = (position[:, 1:] - position[:, :-1]) * float(fps)
    return velocity


def _object_common(batch: Dict, fps: float) -> Dict[str, torch.Tensor]:
    obj_trans = batch["obj_trans"]
    return {
        "obj_trans": obj_trans,
        "obj_trans_init": obj_trans[:, :1],
        "obj_trans_target": obj_trans - obj_trans[:, :1],
        "obj_vel_target": _backward_velocity(obj_trans, fps),
        "has_object": _has_object_mask(batch),
    }


def _crop(tensors: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in tensors.items():
        if torch.is_tensor(value) and value.ndim >= 2 and key != "has_object":
            out[key] = value[:, start:end]
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------
# DIP
# --------------------------------------------------------------------------
def adapt_dip(batch: Dict, fps: float) -> Dict[str, torch.Tensor]:
    stream = common_stream(batch)
    imu = torch.cat(
        [
            stream["ori_norm"][:, :, DIP_SENSOR_ORDER].flatten(2),
            stream["acc_norm"][:, :, DIP_SENSOR_ORDER].flatten(2),
        ],
        dim=-1,
    )
    obj_imu = torch.cat(
        [stream["obj_rot"].flatten(-2), stream["obj_acc_world"]], dim=-1
    )
    local_rot = local_pose_axis_angle_to_rotmat(batch["pose"], num_joints=24)
    return {
        **_object_common(batch, fps),
        "imu": imu,
        "obj_imu": obj_imu,
        "pose_target": local_rot[:, :, DIP_REDUCED_15].flatten(2),
    }


# --------------------------------------------------------------------------
# TIP
# --------------------------------------------------------------------------
def _moving_average(x: torch.Tensor, window: int) -> torch.Tensor:
    """Centred moving average along time with 'nearest' padding ([B,T,C])."""
    if window <= 1 or x.shape[1] == 0:
        return x
    half = window // 2
    padded = F.pad(x.transpose(1, 2), (half, half), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=window, stride=1).transpose(1, 2)


def _windowed_sum(x: torch.Tensor, window: int) -> torch.Tensor:
    """Sum of the last ``window`` frames, partial at the start ([B,T,C])."""
    cumsum = torch.cumsum(x, dim=1)
    if cumsum.shape[1] > window:
        cumsum = torch.cat(
            [cumsum[:, :window], cumsum[:, window:] - cumsum[:, :-window]], dim=1
        )
    return cumsum


def adapt_tip(
    batch: Dict,
    fps: float,
    training: bool,
    windowed: bool,
    n_sbps: int = TIP_STATE_N_SBPS,
) -> Dict[str, torch.Tensor]:
    """TIP inputs (preprocess_and_combine_syn_amass.py / real_time_runner).

    ``windowed`` batches carry ``TIP_TRAIN_HISTORY`` + ``TIP_TRAIN_FUTURE``
    context frames that are consumed by the filters and then cropped.
    """
    stream = common_stream(batch)
    acc_world = stream["acc_world"][:, :, TIP_SENSOR_ORDER]
    ori_global = stream["ori_global"][:, :, TIP_SENSOR_ORDER]
    obj_acc_world = stream["obj_acc_world"]
    batch_size = acc_world.shape[0]
    if training:
        # Constant acceleration bias per training sequence (BIAS_NOISE_ACC).
        acc_world = acc_world + (
            torch.rand(batch_size, 1, 6, 3, device=acc_world.device, dtype=acc_world.dtype)
            * 2.0 - 1.0
        ) * TIP_ACC_BIAS_NOISE
        obj_acc_world = obj_acc_world + (
            torch.rand(batch_size, 1, 3, device=acc_world.device, dtype=acc_world.dtype)
            * 2.0 - 1.0
        ) * TIP_ACC_BIAS_NOISE
    acc_world = _moving_average(acc_world.flatten(2), TIP_ACC_SMOOTH_WINDOW).reshape(acc_world.shape)
    obj_acc_world = _moving_average(obj_acc_world, TIP_ACC_SMOOTH_WINDOW)
    # imu_rotate_to_local: root keeps its global orientation/acceleration,
    # the other sensors are expressed in the current root frame.
    root = ori_global[:, :, 0]
    root_t = root.transpose(-1, -2)
    ori_local = torch.cat(
        [ori_global[:, :, :1], root_t.unsqueeze(2).matmul(ori_global[:, :, 1:])], dim=2
    )
    acc_local = torch.cat(
        [
            acc_world[:, :, :1],
            torch.einsum("btij,btsj->btsi", root_t, acc_world[:, :, 1:]),
        ],
        dim=2,
    )
    acc_sum = _windowed_sum(acc_local.flatten(2), TIP_ACC_SUM_WINDOW) / TIP_ACC_SUM_SCALE
    imu = torch.cat([ori_local.flatten(2), acc_local.flatten(2), acc_sum], dim=-1)
    obj_sum = _windowed_sum(obj_acc_world, TIP_ACC_SUM_WINDOW) / TIP_ACC_SUM_SCALE
    obj_imu = torch.cat([stream["obj_rot"].flatten(-2), obj_acc_world, obj_sum], dim=-1)

    local_rot = local_pose_axis_angle_to_rotmat(batch["pose"], num_joints=24)
    pose_18 = rotation_matrix_to_tip_2axis(local_rot[:, :, TIP_POSE_18]).flatten(2)
    root_vel = _backward_velocity(batch["trans"], fps)
    sbp = batch.get("tip_sbp")
    if sbp is None or n_sbps <= 0:
        sbp = pose_18.new_full((*pose_18.shape[:2], max(n_sbps, 0) * 4), float("nan"))
    sbp = sbp[..., : n_sbps * 4]
    state_target = torch.cat([pose_18, root_vel, sbp], dim=-1)
    sbp_valid = ~torch.isnan(sbp).any(dim=-1) if n_sbps > 0 else torch.zeros_like(root_vel[..., 0], dtype=torch.bool)
    adapted = {
        **_object_common(batch, fps),
        "imu": imu,
        "obj_imu": obj_imu,
        "state_target": state_target,
        "sbp_valid": sbp_valid,
        "measured_root_2axis": rotation_matrix_to_tip_2axis(stream["root"]),
        "n_sbps": torch.tensor(n_sbps, device=imu.device),
    }
    if windowed:
        adapted = _crop(adapted, TIP_TRAIN_HISTORY, imu.shape[1] - TIP_TRAIN_FUTURE)
        adapted["obj_trans_init"] = adapted["obj_trans"][:, :1]
        adapted["obj_trans_target"] = adapted["obj_trans"] - adapted["obj_trans_init"]
    initial_state = adapted["state_target"][:, 0].nan_to_num(0.0).clone()
    # Official s_init: ground-truth first-frame pose, zero motion/constraints.
    initial_state[:, 108:] = 0.0
    adapted["initial_state"] = initial_state
    return adapted


def _angular_velocity_world(rot_prev: torch.Tensor, rot_next: torch.Tensor, dt: float) -> torch.Tensor:
    delta = rot_next.matmul(rot_prev.transpose(-1, -2)).double()
    return (matrix_to_axis_angle(delta) / dt).to(rot_next.dtype)


def tip_sbp_translation(
    local_pose: torch.Tensor,
    root_velocity: torch.Tensor,
    sbp: torch.Tensor,
    initial_trans: torch.Tensor,
    joints_zero_trans: torch.Tensor,
    fps: float,
    ground_height: float = TIP_GROUND_SBP_HEIGHT,
) -> torch.Tensor:
    """Root translation of TIP's minimal real-time runner (Y-up, flat ground).

    Integrates the predicted root velocity and removes the velocity of the
    predicted stationary body points on both feet
    (``get_cur_step_root_correction_from_all_constr``), with the vertical
    residual replaced by the height of those points above the ground.
    """
    from .inference import local_to_global_rotmats

    dt = 1.0 / float(fps)
    global_rot = local_to_global_rotmats(local_pose)
    root = initial_trans[:, None].expand(-1, local_pose.shape[1], -1).clone()
    feet = ((0, 7), (1, 8))
    for frame in range(1, local_pose.shape[1]):
        uncorrected = root[:, frame - 1] + root_velocity[:, frame] * dt
        residues, heights, masks = [], [], []
        for slot, joint in feet:
            contact = sbp[:, frame, 4 * slot] > 0.0
            offset = sbp[:, frame, 4 * slot + 1 : 4 * slot + 4] / 5.0
            x_prev = root[:, frame - 1] + joints_zero_trans[:, frame - 1, joint]
            x_cur = uncorrected + joints_zero_trans[:, frame, joint]
            omega = _angular_velocity_world(
                global_rot[:, frame - 1, joint], global_rot[:, frame, joint], dt
            )
            residues.append(torch.linalg.cross(omega, offset, dim=-1) + (x_cur - x_prev) / dt)
            heights.append(x_cur[:, 1] + offset[:, 1] - ground_height)
            masks.append(contact)
        residues = torch.stack(residues, dim=1)
        heights = torch.stack(heights, dim=1)
        masks = torch.stack(masks, dim=1).to(residues.dtype)
        count = masks.sum(dim=1, keepdim=True)
        mean_residue = (residues * masks.unsqueeze(-1)).sum(dim=1) / count.clamp_min(1.0)
        correction = mean_residue.clamp(-0.5, 0.5)
        correction[:, 1] = (heights * masks).sum(dim=1)
        root[:, frame] = uncorrected - correction * dt
    return root


def decode_tip_pose(state: torch.Tensor, measured_root: torch.Tensor) -> torch.Tensor:
    from .inference import _identity_pose

    pose6d = state[:, :, :108]
    reduced = tip_2axis_to_rotation_matrix(pose6d.reshape(-1, 6)).reshape(*pose6d.shape[:2], 18, 3, 3)
    local = _identity_pose(pose6d)
    local[:, :, TIP_POSE_18] = reduced
    local[:, :, 0] = measured_root
    return local


# --------------------------------------------------------------------------
# GlobalPose
# --------------------------------------------------------------------------
def globalpose_bone_offsets(pos: torch.Tensor, global_rot: torch.Tensor) -> torch.Tensor:
    """Per-sequence skeleton offsets in the parent frame ([B,24,3])."""
    offsets = torch.zeros_like(pos[:, 0, :24])
    for joint in range(1, 24):
        parent = SMPL_PARENTS[joint]
        delta = pos[:, :, joint] - pos[:, :, parent]
        offsets[:, joint] = torch.einsum("bti,btij->btj", delta, global_rot[:, :, parent]).mean(dim=1)
    return offsets


def globalpose_fk_positions(rrj_6d: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """FK of root-relative reduced rotations; ignored joints keep the parent's."""
    batch_size, seq_len = rrj_6d.shape[:2]
    reduced = r6d_columns_to_rotation_matrix(rrj_6d.reshape(-1, 6)).reshape(
        batch_size, seq_len, 15, 3, 3
    )
    eye = torch.eye(3, device=rrj_6d.device, dtype=rrj_6d.dtype)
    global_rot = [eye.expand(batch_size, seq_len, 3, 3)] * 24
    for index, joint in enumerate(DIP_REDUCED_15):
        global_rot[joint] = reduced[:, :, index]
    positions = [rrj_6d.new_zeros(batch_size, seq_len, 3)]
    for joint in range(1, 24):
        parent = SMPL_PARENTS[joint]
        if joint not in DIP_REDUCED_15:
            global_rot[joint] = global_rot[parent]
        positions.append(
            positions[parent]
            + torch.einsum("btij,bj->bti", global_rot[parent], offsets[:, joint])
        )
    return torch.stack(positions[1:], dim=2).flatten(2)


def adapt_globalpose(batch: Dict, fps: float, init_mode: str) -> Dict[str, torch.Tensor]:
    if init_mode not in ("reference", "reference_train"):
        raise ValueError(f"Unknown GlobalPose init mode: {init_mode!r}")
    stream = common_stream(batch)
    pos, global_rot = _ensure_24_global(batch)
    local_rot = local_pose_axis_angle_to_rotmat(batch["pose"], num_joints=24)
    root_rot = global_rot[:, :, 0]
    rel = pos - pos[:, :, :1]
    sensor_joints = [20, 21, 7, 8, 15]  # positions of the stream's non-root sensors
    p_rb = torch.einsum("btsi,btij->btsj", rel[:, :, sensor_joints], root_rot).flatten(2)
    g_r = -root_rot[:, :, 1]
    canonical = _globalpose_canonical_positions(local_rot, global_rot, pos)
    p_rj = canonical[:, :, 1:24].flatten(2)
    rrj = root_relative_global_rot6d(global_rot, DIP_REDUCED_15, column_based=True)
    root_vel_world = _backward_velocity(batch["trans"], fps)
    root_vel_local = torch.einsum("bti,btij->btj", root_vel_world, root_rot)
    velocity_target = torch.cat([root_vel_world[..., 1:2], root_vel_local], dim=-1)
    # Stationary joints: speed of the decode-consistent canonical body (ignored
    # joints at identity, as predicted poses are) below TIP's threshold.
    contact_world = torch.einsum(
        "btij,btsj->btsi", root_rot, canonical[:, :, GLOBALPOSE_CONTACT_JOINTS]
    ) + batch["trans"].unsqueeze(2)
    contact_speed = torch.linalg.norm(_backward_velocity(contact_world, fps), dim=-1)
    contact_speed[:, 0] = contact_speed[:, 1] if contact_speed.shape[1] > 1 else 0.0
    stationary = (contact_speed < GLOBALPOSE_STATIONARY_SPEED).to(pos.dtype)
    target = torch.cat([p_rb, g_r, p_rj, g_r, rrj, velocity_target, stationary], dim=-1)
    x = _globalpose_input(batch, fps=fps)
    pl_init = target[:, 0, :18]
    vr_init = target[:, 0, -9:] if init_mode == "reference_train" else torch.zeros_like(target[:, 0, -9:])
    # Object extension in the measured root frame, where every GlobalPose
    # feature lives (the network never observes the heading).
    root_t = stream["root"].transpose(-1, -2)
    obj_imu = torch.cat(
        [
            torch.einsum("btij,btj->bti", root_t, stream["obj_acc_world"]),
            root_t.matmul(stream["obj_rot"]).flatten(-2),
        ],
        dim=-1,
    )
    obj = _object_common(batch, fps)
    return {
        **obj,
        "obj_imu": obj_imu,
        "obj_vel_target": torch.einsum("bti,btij->btj", obj["obj_vel_target"], root_rot),
        "x": x,
        "target": target,
        "teacher": {"p_rb": p_rb, "g_r": g_r, "p_rj": p_rj, "rrj": rrj},
        "bone_offsets": globalpose_bone_offsets(pos, global_rot),
        "pl_init": pl_init,
        "vr_init": vr_init,
    }


def adapt_batch_paper(
    batch: Dict,
    method: str,
    *,
    fps: float = FRAME_RATE,
    training: bool = False,
    windowed: bool = False,
    globalpose_init_mode: str = "reference",
    n_sbps: int = TIP_STATE_N_SBPS,
) -> Dict[str, torch.Tensor]:
    method = method.lower()
    if method == "dip18":
        return adapt_dip(batch, fps)
    if method == "tip":
        return adapt_tip(batch, fps, training=training, windowed=windowed, n_sbps=n_sbps)
    if method == "globalpose":
        return adapt_globalpose(batch, fps, init_mode=globalpose_init_mode)
    if method == "transpose":
        return adapt_transpose(batch, fps)
    raise ValueError(f"Unknown comparison method: {method}")


TRANSPOSE_ACC_SCALE = 30.0  # config.acc_scale
TRANSPOSE_VEL_SCALE = 3.0  # config.vel_scale
# Contact label (Sec. 4.2.2): foot moves less than 0.008 m between two 60-fps
# frames, i.e. 0.48 m/s; joints 10/11 are the SMPL feet used by Trans-B1.
TRANSPOSE_CONTACT_SPEED = 0.008 * 60.0
TRANSPOSE_LEAF_NOISE = 0.04  # Pose-S2 / Trans-B1 input noise
TRANSPOSE_FULL_NOISE = 0.025  # Pose-S3 / Trans-B2 input noise


def adapt_transpose(batch: Dict, fps: float) -> Dict[str, torch.Tensor]:
    """TransPose inputs/targets (net.py, config.py, paper Sec. 3-4)."""
    stream = common_stream(batch)
    acc = stream["acc_norm"][:, :, TRANSPOSE_ORDER] / TRANSPOSE_ACC_SCALE
    ori = stream["ori_norm"][:, :, TRANSPOSE_ORDER]  # root (last) stays global
    imu = torch.cat([acc.flatten(2), ori.flatten(2)], dim=-1)
    obj_imu = torch.cat([stream["obj_acc_world"], stream["obj_rot"].flatten(-2)], dim=-1)
    pos, global_rot = _ensure_24_global(batch)
    root_rot = global_rot[:, :, 0]
    # Root-relative positions in the root frame: R_root^T (p_j - p_root).
    rel = torch.einsum("btji,btik->btjk", pos - pos[:, :, :1], root_rot)
    leaf = rel[:, :, TRANSPOSE_LEAF].flatten(2)
    full = rel[:, :, TRANSPOSE_FULL].flatten(2)
    pose_target = root_relative_global_rot6d(global_rot, TRANSPOSE_REDUCED, column_based=True)
    root_vel_world = _backward_velocity(batch["trans"], fps)
    root_vel_target = torch.einsum("bti,btij->btj", root_vel_world, root_rot) / TRANSPOSE_VEL_SCALE
    foot_speed = torch.linalg.norm(_backward_velocity(pos[:, :, [10, 11]], fps), dim=-1)
    if foot_speed.shape[1] > 1:
        foot_speed[:, 0] = foot_speed[:, 1]
    contact = (foot_speed < TRANSPOSE_CONTACT_SPEED).to(pos.dtype)
    return {
        **_object_common(batch, fps),
        "imu": imu,
        "obj_imu": obj_imu,
        "leaf_target": leaf,
        "full_target": full,
        "pose_target": pose_target,
        "contact_target": contact,
        "root_vel_target": root_vel_target,
    }


__all__ = [
    "PAPER_PROTOCOL_VERSION",
    "TIP_TRAIN_HISTORY",
    "TIP_TRAIN_FUTURE",
    "adapt_batch_paper",
    "common_stream",
    "decode_tip_pose",
    "globalpose_fk_positions",
    "tip_sbp_translation",
]
