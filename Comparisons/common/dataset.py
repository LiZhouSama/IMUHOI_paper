from __future__ import annotations

import glob
import os
import gc
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random # Import random for shuffling sequence info in debug mode
from utils.rotation_conversions import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix

from configs import (
    FRAME_RATE, _SENSOR_POS_INDICES, _SENSOR_ROT_INDICES, _VEL_SELECTION_INDICES,
    _REDUCED_INDICES, _IGNORED_INDICES, _SENSOR_NAMES, _SENSOR_VEL_NAMES, _REDUCED_POSE_NAMES
)
from Comparisons.common.imu_noise import (
    NOITOM_IMU_NOISE_CFG,
    apply_imu_noise,
    merge_noise_cfg,
)
from Comparisons.common.globalpose_smpl import GlobalPoseSMPLModel
from utils.human_pose import (
    append_virtual_palm_joints,
    append_virtual_palm_rotations,
    load_body_model,
    select_hand_anchor_positions,
)
from utils.utils import (
    _central_diff,
    _make_noise_generator,
    _smooth_acceleration,
    _stable_sequence_seed,
)

_VEL_SELECTION_INDEX_LIST = (
    _VEL_SELECTION_INDICES.tolist()
    if isinstance(_VEL_SELECTION_INDICES, torch.Tensor)
    else list(_VEL_SELECTION_INDICES)
)

_GLOBALPOSE_OFFICIAL_MODEL_CANDIDATES = (
    "/workspace/EgoIMU_1020/RefCodes/GlobalPose/models/SMPL_male.pkl",
    "/workspace/EgoIMU_1020/smpl_models/smpl/SMPL_male.pkl",
)
_GLOBALPOSE_VERTEX_INDICES = (1961, 5424, 1176, 4662, 411, 3021)
_GLOBALPOSE_JOINT_IGNORE = (0, 7, 8, 10, 11, 20, 21, 22, 23)

# --- IMU 计算函数（公共） ---

def _build_human_imu_root(position_global: torch.Tensor,
                     rotation_global: torch.Tensor,
                     imu_joints_pos,
                     imu_joints_rot,
                     fps: float) -> torch.Tensor:
    """利用关节位置和姿态构建 DynaIP 风格的 IMU 特征。"""
    dt = 1.0 / fps
    pos = position_global[:, imu_joints_pos, :]
    rot = rotation_global[:, imu_joints_rot, :, :]
    vel = _central_diff(pos, dt)
    # 使用平滑加速度函数代替二次差分
    acc = _smooth_acceleration(pos, fps, smooth_n=4)
    root_imu_ori = rot[:, 0]
    imu_acc_norm = torch.cat((acc[:, :1], acc[:, 1:] - acc[:, :1]), dim=1).bmm(root_imu_ori)
    imu_ori_norm = torch.cat((rot[:, :1], rot[:, :1].transpose(2, 3).matmul(rot[:, 1:])), dim=1)
    imu_ori_norm_6d = matrix_to_rotation_6d(imu_ori_norm)
    imu = torch.cat((imu_acc_norm, imu_ori_norm_6d), dim=-1)
    return imu  # [T, 6, 9]


def _build_human_imu_root_central(
    position_global: torch.Tensor,
    rotation_global: torch.Tensor,
    imu_joints_pos,
    imu_joints_rot,
    fps: float,
) -> torch.Tensor:
    """Build the central-difference IMU used by the public DIP adaptation.

    The regular IMUHOI stream intentionally smooths acceleration.  The
    released DIP/OMOMO loader uses two ordinary central differences, so this
    protocol is selectable without changing the default stream used by the
    main model.
    """
    dt = 1.0 / float(fps)
    pos = position_global[:, imu_joints_pos, :]
    rot = rotation_global[:, imu_joints_rot, :, :]
    velocity = _central_diff(pos, dt)
    acceleration = _central_diff(velocity, dt)
    root_rot = rot[:, 0]
    acc_relative = torch.cat(
        (acceleration[:, :1], acceleration[:, 1:] - acceleration[:, :1]), dim=1
    )
    # Match the row-vector convention in RefCodes/dip18/my_simple/dataset_obj.py.
    acc_root = torch.matmul(acc_relative, root_rot)
    ori_root = torch.cat(
        (rot[:, :1], rot[:, :1].transpose(2, 3).matmul(rot[:, 1:])), dim=1
    )
    return torch.cat((acc_root, matrix_to_rotation_6d(ori_root)), dim=-1)


def _build_globalpose_reference_imu(
    position_global: torch.Tensor,
    rotation_global: torch.Tensor,
    sensor_position_joints,
    sensor_rotation_joints,
    fps: float,
) -> torch.Tensor:
    """Build GlobalPose's root-last ``[aRB, RRB]`` stream.

    Unlike the native IMUHOI builder, GlobalPose orders the root sensor last
    and feeds the linear model-frame acceleration after transforming it into
    the root frame.  ``position_global`` is already a kinematic world-space
    trajectory, so its finite-difference acceleration is the post-processed
    ``aM`` signal consumed by the released network.  The released simulator's
    ``aS`` is a free-acceleration signal and is converted to the same ``aM`` by
    ``gp_training/data.py``; adding gravity again here would double-count it.
    """
    dt = 1.0 / float(fps)
    # The released protocol mounts the five non-root sensors on mesh
    # vertices.  The dataset passes those trajectories as tensors; retain the
    # integer-index path for callers that do not provide the exact mesh data.
    if torch.is_tensor(sensor_position_joints):
        pos = sensor_position_joints
    else:
        pos = position_global[:, sensor_position_joints, :]
    if torch.is_tensor(sensor_rotation_joints):
        rot = sensor_rotation_joints
    else:
        rot = rotation_global[:, sensor_rotation_joints, :, :]
    acceleration = torch.zeros_like(pos)
    if pos.shape[0] > 1:
        if pos.shape[0] >= 4:
            acceleration[1:-1] = (
                pos[:-2] - 2.0 * pos[1:-1] + pos[2:]
            ) / (dt * dt)
            acceleration[0] = (
                2.0 * pos[0] - 5.0 * pos[1] + 4.0 * pos[2] - pos[3]
            ) / (dt * dt)
            acceleration[-1] = (
                2.0 * pos[-1] - 5.0 * pos[-2] + 4.0 * pos[-3] - pos[-4]
            ) / (dt * dt)
        else:
            velocity = _central_diff(pos, dt)
            acceleration = _central_diff(velocity, dt)
    root_rot = rot[:, -1]
    acceleration_root = torch.einsum(
        "tsi,tij->tsj", acceleration, root_rot
    )
    relative_rot = torch.cat(
        [
            root_rot.transpose(-1, -2).unsqueeze(1).matmul(rot[:, :-1]),
            root_rot.unsqueeze(1),
        ],
        dim=1,
    )
    return torch.cat(
        [acceleration_root, matrix_to_rotation_6d(relative_rot)], dim=-1
    )

def _compute_joint_velocity_root(rotation_global: torch.Tensor,
                                 position_global: torch.Tensor,
                                 fps: float):
    """计算根坐标系下的关节速度和位置。"""
    pos = position_global.clone()
    root_pos = position_global[:, :1, :].clone()
    pos[:, :, 0] = pos[:, :, 0] - root_pos[:, :, 0]
    pos[:, :, 2] = pos[:, :, 2] - root_pos[:, :, 2]

    vel_world = _central_diff(pos, 1.0 / fps)
    root_vel = vel_world[:, :1]
    rel_vel = torch.cat((root_vel, vel_world[:, 1:] - root_vel), dim=1)

    root_R = rotation_global[:, 0]
    vel_root = rel_vel.bmm(root_R)
    return vel_root

def _build_object_imu(obj_rot: torch.Tensor, obj_pos: torch.Tensor, fps: float) -> torch.Tensor:
    """计算物体的 IMU 特征（姿态 + 加速度）。"""
    if obj_rot.shape[-1] != 6:
        ori_obj_6d = matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3))
    else:
        ori_obj_6d = obj_rot
    vel_obj = _central_diff(obj_pos, 1.0 / fps)
    # 使用平滑加速度函数代替二次差分
    acc_obj = _smooth_acceleration(obj_pos, fps, smooth_n=4)
    return torch.cat([acc_obj, ori_obj_6d], dim=-1)   # [T, 9]


def _build_object_imu_central(
    obj_rot: torch.Tensor, obj_pos: torch.Tensor, fps: float
) -> torch.Tensor:
    """Build object [acc3, rot6] features with ordinary central differences."""
    if obj_rot.shape[-1] != 6:
        ori_obj_6d = matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3)).reshape(
            obj_rot.shape[0], 6
        )
    else:
        ori_obj_6d = obj_rot
    velocity = _central_diff(obj_pos, 1.0 / float(fps))
    acceleration = _central_diff(velocity, 1.0 / float(fps))
    return torch.cat([acceleration, ori_obj_6d], dim=-1)


# TIP's released OMOMO adaptation uses a different six-sensor placement than
# the native IMUHOI stream: pelvis, wrists, knees, and neck.  Keep this order
# local to the reference protocol so existing model inputs remain unchanged.
_TIP_REFERENCE_JOINTS = [0, 20, 21, 7, 8, 15]
_TIP_REFERENCE_POSE_18_FROM_22 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    12, 13, 14, 15, 16, 17, 18, 19,
]

# TransPose's released loader uses a different sensor order and feature
# layout from the IMUHOI stream.  Keep this protocol isolated so loading a
# TransPose checkpoint cannot silently change the other baselines.
_TRANSPOSE_REFERENCE_POS_JOINTS = [7, 8, 15, 20, 21, 0]
_TRANSPOSE_REFERENCE_ROT_JOINTS = [4, 5, 15, 18, 19, 0]
# GlobalPose's released IMU synthesis uses elbow, knee, head and root joints
# in this exact order (root is last).  Keep it separate from IMUHOI's native
# wrist/ankle stream; mixing the two produces a valid-looking but wrong 84-D
# input for the public checkpoint.
# The released GlobalPose simulator uses mesh vertices for positions and
# the corresponding body-joint orientations.  IMUHOI has no mesh vertices in
# the comparison stream, so use the nearest available joint pair separately:
# forearm, forearm, lower-leg, lower-leg, head, pelvis (root last).
_GLOBALPOSE_REFERENCE_POSITION_JOINTS = [20, 21, 7, 8, 15, 0]
_GLOBALPOSE_REFERENCE_ROTATION_JOINTS = [18, 19, 4, 5, 15, 0]


def _transpose_reference_central_diff(values: torch.Tensor, fps: float) -> torch.Tensor:
    """Finite difference used by RefCodes/TransPose (one-sided endpoints)."""
    if values.shape[0] <= 1:
        return torch.zeros_like(values)
    dt = 1.0 / float(fps)
    out = torch.zeros_like(values)
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    return out


def _transpose_reference_normalize_and_concat(
    global_acc: torch.Tensor, global_rot: torch.Tensor
) -> torch.Tensor:
    """Exact equivalent of the public TransPose ``normalize_and_concat``."""
    root_acc = global_acc[:, 5:6]
    acc = torch.cat((global_acc[:, :5] - root_acc, root_acc), dim=1)
    acc = torch.matmul(acc, global_rot[:, 5]) / 30.0
    root_rot = global_rot[:, 5:6]
    ori = torch.cat(
        (root_rot.transpose(-1, -2).matmul(global_rot[:, :5]), root_rot), dim=1
    )
    return torch.cat((acc.flatten(1), ori.flatten(1)), dim=-1)


def _build_transpose_reference_features(
    position_global: torch.Tensor,
    rotation_global: torch.Tensor,
    obj_trans: Optional[torch.Tensor],
    obj_rot: Optional[torch.Tensor],
    fps: float,
    noise_cfg: Optional[Dict[str, float]] = None,
    noise_generator: Optional[torch.Generator] = None,
    simulate_noise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build full-sequence human/object features for the released checkpoint."""
    pos = position_global[:, _TRANSPOSE_REFERENCE_POS_JOINTS]
    rot = rotation_global[:, _TRANSPOSE_REFERENCE_ROT_JOINTS]
    velocity = _transpose_reference_central_diff(pos, fps)
    acceleration = _transpose_reference_central_diff(velocity, fps)

    obj_pos = obj_trans
    if obj_pos is None:
        obj_pos = torch.zeros(
            position_global.shape[0], 3,
            device=position_global.device,
            dtype=position_global.dtype,
        )
    if obj_pos.ndim == 3 and obj_pos.shape[1] == 1:
        obj_pos = obj_pos[:, 0]
    if obj_rot is None:
        obj_rot_mat = torch.eye(
            3, device=position_global.device, dtype=position_global.dtype
        ).expand(position_global.shape[0], 3, 3).clone()
    else:
        obj_rot_mat = obj_rot
        if obj_rot_mat.ndim == 3 and obj_rot_mat.shape[-1] == 6:
            obj_rot_mat = rotation_6d_to_matrix(obj_rot_mat.reshape(-1, 6)).reshape(-1, 3, 3)
        elif obj_rot_mat.ndim != 3 or obj_rot_mat.shape[-2:] != (3, 3):
            raise ValueError(f"Unexpected TransPose object rotation shape: {obj_rot_mat.shape}")
    obj_velocity = _transpose_reference_central_diff(obj_pos, fps)
    obj_acceleration = _transpose_reference_central_diff(obj_velocity, fps)

    if simulate_noise:
        sensor_rot6d = matrix_to_rotation_6d(rot.reshape(-1, 3, 3)).reshape(
            rot.shape[0], rot.shape[1], 6
        )
        acceleration, sensor_rot6d, _ = apply_imu_noise(
            acceleration, sensor_rot6d, fps, noise_cfg, generator=noise_generator
        )
        rot = rotation_6d_to_matrix(sensor_rot6d.reshape(-1, 6)).reshape(rot.shape)
        obj_rot6d = matrix_to_rotation_6d(obj_rot_mat.reshape(-1, 3, 3)).reshape(-1, 6)
        obj_acceleration, obj_rot6d, _ = apply_imu_noise(
            obj_acceleration, obj_rot6d, fps, noise_cfg, generator=noise_generator
        )
        obj_rot_mat = rotation_6d_to_matrix(obj_rot6d.reshape(-1, 6)).reshape(obj_rot_mat.shape)

    human_imu = _transpose_reference_normalize_and_concat(acceleration, rot)
    object_imu = torch.cat((obj_rot_mat.flatten(1), obj_acceleration), dim=-1)
    return human_imu.contiguous(), object_imu.contiguous()


def _tip_reference_second_diff(position: torch.Tensor, fps: float) -> torch.Tensor:
    """Finite-difference acceleration used by RefCodes/TIP dataset_omomo_tip_v2."""
    length = int(position.shape[0])
    acceleration = torch.zeros_like(position)
    if length > 2:
        acceleration[2:] = (
            position[:-2] + position[2:] - 2.0 * position[1:-1]
        ) * float(fps) ** 2
        acceleration[1] = (position[1] - position[0]) * float(fps) ** 2
    return acceleration


def _build_tip_reference_features(
    position_global: torch.Tensor,
    rotation_global: torch.Tensor,
    obj_trans: torch.Tensor | None,
    obj_rot: torch.Tensor | None,
    fps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build the exact [acc3, first-two-columns6] TIP reference stream.

    The public TIP loader normalizes every sensor by the *sequence's* first
    root rotation, then samples windows.  This helper therefore receives the
    complete sequence and returns the corresponding full-length targets too.
    """
    if position_global.ndim != 3 or rotation_global.ndim != 4:
        raise ValueError("TIP reference features require full global position/rotation tensors")
    root_pos0 = position_global[0, 0]
    root_rot0 = rotation_global[0, 0]
    root_inv = root_rot0.transpose(-1, -2)

    human_features = []
    for joint in _TIP_REFERENCE_JOINTS:
        acceleration = _tip_reference_second_diff(position_global[:, joint], fps)
        acceleration = acceleration.matmul(root_inv)
        orientation = root_inv.matmul(rotation_global[:, joint])
        # TIP's ``_rotmat_to_2axis`` flattens the 3x2 matrix in row-major order.
        two_axis = orientation[..., :, :2].reshape(orientation.shape[0], 6)
        human_features.append(torch.cat((acceleration, two_axis), dim=-1))
    human_imu = torch.stack(human_features, dim=1)

    if obj_trans is None:
        obj_trans_full = torch.zeros(
            position_global.shape[0], 3,
            device=position_global.device,
            dtype=position_global.dtype,
        )
    else:
        obj_trans_full = obj_trans
        if obj_trans_full.ndim == 3 and obj_trans_full.shape[1] == 1:
            obj_trans_full = obj_trans_full[:, 0]
        if obj_trans_full.ndim != 2 or obj_trans_full.shape[-1] != 3:
            raise ValueError(f"obj_trans must have shape [T,3], got {tuple(obj_trans_full.shape)}")
    obj_acceleration = _tip_reference_second_diff(obj_trans_full, fps).matmul(root_inv)
    # Keep the released v2 loader's object branch exactly.  In particular, a
    # processed [T,3,3] rotation takes its zero fallback because the reference
    # code only normalizes matrix rotations in a 4-D branch.  A [T,6] value is
    # already the stored two-axis representation and is not re-normalized.
    if obj_rot is None:
        object_two_axis = torch.zeros(
            position_global.shape[0], 6,
            device=position_global.device,
            dtype=position_global.dtype,
        )
    elif obj_rot.ndim == 3 and obj_rot.shape[-1] == 6:
        object_two_axis = obj_rot.to(
            device=position_global.device, dtype=position_global.dtype
        )
    elif obj_rot.ndim == 4 and obj_rot.shape[-2:] == (3, 3):
        object_orientation = root_inv.unsqueeze(0).matmul(obj_rot)
        object_two_axis = object_orientation[..., :, :2].reshape(
            object_orientation.shape[0], 6
        )
    else:
        object_two_axis = torch.zeros(
            position_global.shape[0], 6,
            device=position_global.device,
            dtype=position_global.dtype,
        )
    object_imu = torch.cat((obj_acceleration, object_two_axis), dim=-1)

    root_position = (position_global[:, 0] - root_pos0).matmul(root_inv)
    root_velocity = torch.zeros_like(root_position)
    if root_position.shape[0] > 1:
        root_velocity[1:] = (root_position[1:] - root_position[:-1]) * float(fps)
    object_position = (obj_trans_full - root_pos0).matmul(root_inv)
    object_velocity = torch.zeros_like(object_position)
    if object_position.shape[0] > 1:
        object_velocity[1:] = (object_position[1:] - object_position[:-1]) * float(fps)
    return (
        human_imu,
        object_imu,
        root_velocity,
        object_velocity,
        root_rot0,
        root_pos0,
    )

# --- 结束 IMU 计算函数 ---

def find_contact_segments(contact_mask):
    """找到连续的接触段"""
    segments = []
    contact_indices = torch.where(contact_mask)[0]

    if len(contact_indices) == 0:
        return segments

    start = contact_indices[0].item()
    end = start

    for i in range(1, len(contact_indices)):
        if contact_indices[i] == contact_indices[i-1] + 1:
            end = contact_indices[i].item()
        else:
            segments.append((start, end))
            start = contact_indices[i].item()
            end = start

    segments.append((start, end))
    return segments

def suppress_bimanual_contact_by_bone_variation(
    lhand_contact: torch.Tensor,
    rhand_contact: torch.Tensor,
    lhand_bone_length: torch.Tensor,
    rhand_bone_length: torch.Tensor,
    eps: float = 1e-6,
):
    """Drop the less stable hand label inside simultaneous hand-contact segments."""
    lhand_contact = lhand_contact.clone().bool()
    rhand_contact = rhand_contact.clone().bool()
    both_contact = lhand_contact & rhand_contact
    if not bool(both_contact.any()):
        return lhand_contact, rhand_contact

    def _variation(values: torch.Tensor) -> float:
        finite_values = values[torch.isfinite(values)]
        if finite_values.numel() < 2:
            return 0.0
        return float(finite_values.max() - finite_values.min())

    for seg_start, seg_end in find_contact_segments(both_contact):
        seg_slice = slice(seg_start, seg_end + 1)
        l_var = _variation(lhand_bone_length[seg_slice])
        r_var = _variation(rhand_bone_length[seg_slice])
        if l_var > r_var + eps:
            lhand_contact[seg_slice] = False
        elif r_var > l_var + eps:
            rhand_contact[seg_slice] = False

    return lhand_contact, rhand_contact

def gaussian_label_from_indices(indices, length, sigma=2.0, device=None, dtype=None):
    """根据接触起止帧生成高斯平滑标签"""
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    if isinstance(indices, torch.Tensor):
        idx_list = indices.tolist()
    else:
        idx_list = list(indices)
    if len(idx_list) == 0:
        return torch.zeros(length, device=device, dtype=dtype)
    t = torch.arange(length, device=device, dtype=dtype)
    labels = torch.zeros(length, device=device, dtype=dtype)
    for idx in idx_list:
        labels = torch.maximum(labels, torch.exp(-0.5 * ((t - float(idx)) / sigma) ** 2))
    return labels


# 接触段端点通道顺序: (lhand_start, lhand_end, rhand_start, rhand_end, obj_start, obj_end)
CONTACT_ENDPOINT_CHANNEL_NAMES = ("lhand", "rhand", "obj")


def build_contact_endpoint_labels(
    lhand_contact,
    rhand_contact,
    obj_contact,
    prev_contact=(False, False, False),
    next_contact=(False, False, False),
    sigma=2.0,
    device=None,
    dtype=None,
):
    """构建接触段端点标签，6通道: (l_start, l_end, r_start, r_end, obj_start, obj_end)。

    窗口切片会截断跨窗口的接触段：若某段起点落在窗口首帧且窗口前一帧仍在接触
    （prev_contact 为 True），说明真实起点在窗口外，此时不打 start 标签；
    段终点落在窗口末帧且窗口后一帧仍在接触（next_contact 为 True）时同理不打
    end 标签，避免网络把窗口边界学成端点。

    Args:
        lhand_contact / rhand_contact / obj_contact: [T] 接触标签
        prev_contact: 三通道在窗口前一帧（原始序列 start_idx-1）的接触状态
        next_contact: 三通道在窗口后一帧（原始序列 end_idx）的接触状态
    Returns:
        hard: [T, 6] 端点脉冲标签
        gauss: [T, 6] 高斯平滑标签
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    channels = (lhand_contact, rhand_contact, obj_contact)
    length = int(channels[0].shape[0])
    hard = torch.zeros(length, 6, device=device, dtype=dtype)
    gauss = torch.zeros(length, 6, device=device, dtype=dtype)
    for channel_index, contact in enumerate(channels):
        contact = contact.bool()
        if int(contact.shape[0]) != length:
            raise ValueError("contact label channels must share the same length")
        start_indices = []
        end_indices = []
        for seg_start, seg_end in find_contact_segments(contact):
            if not (seg_start == 0 and bool(prev_contact[channel_index])):
                start_indices.append(seg_start)
            if not (seg_end == length - 1 and bool(next_contact[channel_index])):
                end_indices.append(seg_end)
        start_channel = 2 * channel_index
        end_channel = start_channel + 1
        for index in start_indices:
            hard[index, start_channel] = 1.0
        for index in end_indices:
            hard[index, end_channel] = 1.0
        gauss[:, start_channel] = gaussian_label_from_indices(
            start_indices, length, sigma=sigma, device=device, dtype=dtype
        )
        gauss[:, end_channel] = gaussian_label_from_indices(
            end_indices, length, sigma=sigma, device=device, dtype=dtype
        )
    return hard, gauss

def compute_obj_direction_supervision(position_global, obj_trans, obj_rot,
                                    lhand_contact, rhand_contact):
    """
    计算物体坐标系下的方向向量监督标签
    Returns:
        dict: 包含物体坐标系下的方向向量
    """
    device = position_global.device
    seq_len = obj_trans.shape[0]

    # 将6D旋转表示转换为旋转矩阵
    obj_rot_mat = rotation_6d_to_matrix(obj_rot)  # [seq, 3, 3]

    # 初始化结果
    result = {
        "lhand_obj_direction": torch.zeros(seq_len, 3, device=device),
        "rhand_obj_direction": torch.zeros(seq_len, 3, device=device),
    }

    # 使用虚拟掌根位置作为手-物几何锚点。
    hand_anchor_pos = select_hand_anchor_positions(position_global)
    hand_pos = {
        'left': hand_anchor_pos[:, 0, :],
        'right': hand_anchor_pos[:, 1, :],
    }

    contacts = {'left': lhand_contact, 'right': rhand_contact}

    for hand_name in ['left', 'right']:
        contact_mask = contacts[hand_name]

        if not contact_mask.any():
            continue

        # 找到接触段
        contact_segments = find_contact_segments(contact_mask)

        for seg_start, seg_end in contact_segments:
            # 获取接触段的索引
            seg_indices = torch.arange(seg_start, seg_end + 1, device=obj_trans.device)

            # 批量获取接触段的数据
            hand_pos_seg = hand_pos[hand_name][seg_indices]  # [seg_len, 3]
            obj_trans_seg = obj_trans[seg_indices]  # [seg_len, 3]
            obj_rot_mat_seg = obj_rot_mat[seg_indices]  # [seg_len, 3, 3]

            # 1. 计算世界坐标系下掌根指向物体的向量
            v_HO_world = obj_trans_seg - hand_pos_seg  # [seg_len, 3]

            # 2. 归一化得到世界坐标系下的单位向量
            v_HO_world_unit = v_HO_world / (torch.norm(v_HO_world, dim=1, keepdim=True) + 1e-8)  # [seg_len, 3]

            # 3. 转换到物体坐标系：^Ov_{HO} = ^WR_O^T * ^Wv_{HO}
            obj_rot_mat_inv_seg = obj_rot_mat_seg.transpose(-1, -2)  # [seg_len, 3, 3]
            v_HO_obj = torch.bmm(obj_rot_mat_inv_seg, v_HO_world_unit.unsqueeze(-1)).squeeze(-1)  # [seg_len, 3]

            # 4. 存储结果
            if hand_name == 'left':
                result["lhand_obj_direction"][seg_indices] = v_HO_obj
            else:
                result["rhand_obj_direction"][seg_indices] = v_HO_obj

    return result

class IMUDataset(Dataset):
    def __init__(
        self,
        data_dir,
        window_size=60,
        debug=False,
        min_obj_contact_frames=0,
        full_sequence=False,
        imu_noise_cfg=None,
        simulate_imu_noise=False,
        simulate_imu_noise_seed=None,
        sequence_paths=None,
        obj_points_sample_count=256,
        resolve_bimanual_contact_conflicts=False,
        random_window=True,
        imu_protocol="imuhoi",
        body_model_path=None,
        globalpose_mesh_model_path=None,
    ):
        """
        IMU数据集 - 每个epoch为每个序列随机采样一个窗口
        Args:
            data_dir: 数据目录，可以是字符串（单个目录）或列表（多个目录）
            window_size: 窗口大小
            debug: 是否在调试模式
            min_obj_contact_frames: 序列中至少需要的物体接触帧数，少于此值的序列将被过滤掉
            full_sequence: 是否使用整段模式
            imu_noise_cfg: 可选，IMU 噪声配置（None 使用 Noitom 风格默认值）
            simulate_imu_noise: 是否对合成 IMU 施加真实感噪声模型
            simulate_imu_noise_seed: 可选基础种子；结合序列路径生成与访问顺序无关的固定噪声
            resolve_bimanual_contact_conflicts: 是否后处理双手同时接触段，清掉骨长变化更大的手部接触标签
            random_window: 窗口模式下是否随机选起点；验证集可关闭以获得可重复曲线
        """
        # 支持单个目录或多个目录
        if isinstance(data_dir, str):
            self.data_dirs = [data_dir]
        elif isinstance(data_dir, list):
            self.data_dirs = data_dir
        else:
            raise ValueError("data_dir must be a string or a list of strings")

        self.window_size = window_size
        self.debug = debug
        self.min_obj_contact_frames = min_obj_contact_frames
        self.full_sequence = full_sequence
        self.simulate_imu_noise = simulate_imu_noise
        self.simulate_imu_noise_seed = simulate_imu_noise_seed
        self.imu_noise_cfg = merge_noise_cfg(imu_noise_cfg)
        self.obj_points_sample_count = int(max(1, obj_points_sample_count))
        self.resolve_bimanual_contact_conflicts = bool(resolve_bimanual_contact_conflicts)
        self.random_window = bool(random_window)
        self.imu_protocol = str(imu_protocol).lower()
        self.body_model_path = body_model_path
        self.globalpose_mesh_model_path = globalpose_mesh_model_path
        self._globalpose_mesh_models = {}
        self._globalpose_sensor_cache = {}
        if self.imu_protocol not in {
            "imuhoi", "dip18_reference", "tip_reference", "transpose_reference",
            "globalpose_reference",
        }:
            raise ValueError(
                "imu_protocol must be 'imuhoi', 'dip18_reference', 'tip_reference', "
                "'transpose_reference', or 'globalpose_reference', got "
                f"{imu_protocol!r}"
            )

        # 查找所有目录中的序列文件，或使用显式传入的序列文件列表
        self.sequence_files = []
        if sequence_paths is not None:
            if isinstance(sequence_paths, str):
                sequence_paths = [sequence_paths]
            for seq_path in sequence_paths:
                seq_path_abs = os.path.abspath(os.path.normpath(seq_path))
                if os.path.isfile(seq_path_abs) and seq_path_abs.lower().endswith(".pt"):
                    self.sequence_files.append(seq_path_abs)
                else:
                    print(f"警告: 无效的序列文件路径，已跳过: {seq_path}")
            print(f"使用显式序列列表，共 {len(self.sequence_files)} 个文件")
        else:
            for data_dir_path in self.data_dirs:
                if os.path.exists(data_dir_path):
                    dir_files = glob.glob(os.path.join(data_dir_path, "*.pt"))
                    self.sequence_files.extend(dir_files)
                    print(f"从目录 {data_dir_path} 找到 {len(dir_files)} 个序列文件")
                else:
                    print(f"警告: 目录 {data_dir_path} 不存在，跳过")

        print(f"总共找到 {len(self.sequence_files)} 个序列文件")

        # 初始化用于存储预加载数据和序列信息的容器
        self.loaded_data = {}
        self.sequence_info = [] # Store sequence metadata: {'file_path': ..., 'seq_name': ..., 'seq_len': ...}
        self.has_human_real_imu = False
        self.has_obj_real_imu = False

        # 执行加载、共享和序列信息收集
        self._load_share_and_collect_info()
        print(f"真实IMU使用情况：人体={'存在' if self.has_human_real_imu else '生成'}，物体={'存在' if self.has_obj_real_imu else '生成'}")
        print(f"预加载并收集信息完成，共找到{len(self.sequence_info)}个有效序列")
        if self.full_sequence:
            print(f"过滤条件：整段模式启用，序列长度 >= 1，手部接触帧数(左右手任一) >= {self.min_obj_contact_frames}")
        else:
            print(f"过滤条件：序列长度 >= {self.window_size + 1}，手部接触帧数(左右手任一) >= {self.min_obj_contact_frames}")

        # 调试模式下只使用一小部分序列
        if debug and len(self.sequence_info) > 100:
            # 只缩减 sequence_info 列表
            random.shuffle(self.sequence_info) # Shuffle before taking a subset
            self.sequence_info = self.sequence_info[:100]
            print(f"调试模式：使用{len(self.sequence_info)}个序列")
        elif len(self.sequence_info) == 0:
             print("警告：没有找到有效的序列，数据集为空。请检查数据和窗口参数。")

    def _globalpose_model_for_sequence(self, seq_data):
        """Return a cached exact-mesh SMPL evaluator for one sequence gender."""
        configured = self.globalpose_mesh_model_path
        if configured is None:
            configured = next(
                (candidate for candidate in _GLOBALPOSE_OFFICIAL_MODEL_CANDIDATES if os.path.exists(candidate)),
                self.body_model_path,
            )
        if not configured:
            return None
        configured = os.path.abspath(os.path.expanduser(str(configured)))
        # The public checkpoint and its OMOMO adapter use the original male
        # SMPL pickle for all sequences.  Do not substitute a gender-specific
        # SMPL-H model: the processed joint trajectories were generated with
        # the male body model even when metadata says ``female``.
        candidate = configured
        if not os.path.exists(candidate):
            return None
        if candidate not in self._globalpose_mesh_models:
            if candidate.lower().endswith(".pkl"):
                self._globalpose_mesh_models[candidate] = GlobalPoseSMPLModel(candidate)
            else:
                # Convert the SMPL-H npz through the project's tested body
                # model implementation.  This has the same 6890-vertex
                # topology, and leaves hand pose at its neutral value.
                self._globalpose_mesh_models[candidate] = load_body_model(candidate)
        return self._globalpose_mesh_models[candidate]

    def _globalpose_mesh_targets(self, file_path, seq_data, local_rotation_full, trans_full):
        """Compute/cache exact GlobalPose mesh, canonical and contact targets."""
        if self.imu_protocol != "globalpose_reference":
            return None
        configured = self.globalpose_mesh_model_path
        if configured is None:
            configured = next(
                (candidate for candidate in _GLOBALPOSE_OFFICIAL_MODEL_CANDIDATES if os.path.exists(candidate)),
                self.body_model_path,
            )
        if not configured:
            return None
        cache_key = (file_path, str(configured))
        cached = self._globalpose_sensor_cache.get(cache_key)
        if cached is not None:
            return cached
        model = self._globalpose_model_for_sequence(seq_data)
        if model is None:
            return None
        local_rot = rotation_6d_to_matrix(
            local_rotation_full.reshape(local_rotation_full.shape[0], -1, 6)
        )
        if isinstance(model, GlobalPoseSMPLModel):
            result = model.forward(local_rot[:, :24], trans=trans_full)
            canonical = model.forward(
                local_rot[:, :24], ignored_indices=_GLOBALPOSE_JOINT_IGNORE
            )
            targets = {
                "sensor_positions": result["vertices"],
                "joints": result["joints"],
                "canonical_joints": canonical["joints"],
            }
        else:
            from utils.rotation_conversions import matrix_to_axis_angle

            pose_axis = matrix_to_axis_angle(
                local_rot[:, :22].reshape(-1, 3, 3)
            ).reshape(local_rot.shape[0], 22, 3)
            body_out = model(
                root_orient=pose_axis[:, 0],
                pose_body=pose_axis[:, 1:22].reshape(local_rot.shape[0], 63),
                trans=trans_full,
            )
            pose_canon = local_rot[:, :22].clone()
            eye = torch.eye(3, device=pose_canon.device, dtype=pose_canon.dtype)
            pose_canon[:, list(_GLOBALPOSE_JOINT_IGNORE)] = eye
            pose_canon_axis = matrix_to_axis_angle(
                pose_canon.reshape(-1, 3, 3)
            ).reshape(local_rot.shape[0], 22, 3)
            canonical_out = model(
                root_orient=pose_canon_axis[:, 0],
                pose_body=pose_canon_axis[:, 1:22].reshape(local_rot.shape[0], 63),
            )
            targets = {
                "sensor_positions": body_out.v[:, list(_GLOBALPOSE_VERTEX_INDICES)],
                "joints": body_out.Jtr[:, :24],
                "canonical_joints": canonical_out.Jtr[:, :24],
            }
        targets = {
            key: value.detach().float().cpu() for key, value in targets.items()
        }
        self._globalpose_sensor_cache[cache_key] = targets
        return targets

    def _load_share_and_collect_info(self):
        """加载序列数据，按需共享 Tensor，并收集序列信息。"""
        print("开始预加载、共享内存处理和序列信息收集...")
        # Preloaded tensors are read-only and workers inherit them through
        # Linux fork.  Calling ``share_memory_`` for every storage consumes a
        # file descriptor per tensor (or requires a Docker-compatible shm
        # filesystem), so it is opt-in for environments that explicitly need
        # it rather than a correctness requirement.
        share_memory_enabled = os.environ.get("IMUHOI_ENABLE_SHARED_MEMORY", "0") == "1"
        print(
            "共享内存策略: "
            + ("enabled (IMUHOI_ENABLE_SHARED_MEMORY=1)" if share_memory_enabled else "disabled（只读预加载，避免fd/shm耗尽）")
        )
        for file_path in tqdm(self.sequence_files, desc="预加载和收集信息"):
            try:
                # 加载序列数据
                seq_data = torch.load(file_path)
                seq_name = os.path.basename(file_path).replace(".pt", "")

                # 判定是否有物体：显式 has_object 优先，其次根据 obj_trans 是否存在
                has_object_flag = seq_data.get("has_object", None)
                if has_object_flag is None:
                    has_object_flag = ("obj_trans" in seq_data and seq_data["obj_trans"] is not None)
                has_object_flag = bool(has_object_flag)

                # 检查基本数据是否存在
                if seq_data is None or "rotation_local_full_gt_list" not in seq_data or seq_data["rotation_local_full_gt_list"] is None:
                    print(f"警告：跳过文件 {file_path}，缺少必要的 'rotation_local_full_gt_list' 数据。")
                    continue

                motion = seq_data["rotation_local_full_gt_list"]
                seq_len = motion.shape[0]

                # 检查序列长度是否足够
                if self.full_sequence:
                    # 整段模式：仅需至少1帧即可
                    if seq_len < 1:
                        continue
                else:
                    # 窗口模式：需要能采样到一个窗口（从索引1开始）
                    if seq_len < self.window_size + 1:
                        # print(f"调试：跳过文件 {file_path}，序列长度 {seq_len} 不足以创建大小为 {self.window_size} 的窗口。")
                        continue

                # 检查手部接触帧数是否足够（仅对含物体序列生效，左右手任一达到阈值即通过）
                if has_object_flag:
                    def _contact_count(contact):
                        if contact is None:
                            return 0
                        if isinstance(contact, torch.Tensor):
                            return contact.sum().item()
                        return np.sum(contact)

                    l_contact_cnt = _contact_count(seq_data.get("lhand_contact"))
                    r_contact_cnt = _contact_count(seq_data.get("rhand_contact"))

                    if l_contact_cnt < self.min_obj_contact_frames and r_contact_cnt < self.min_obj_contact_frames:
                        # print(f"跳过文件 {file_path}，左右手接触帧数分别为 {l_contact_cnt}, {r_contact_cnt}，均少于最小要求 {self.min_obj_contact_frames}")
                        continue

                # 将所有Tensor移动到共享内存（可选优化，默认关闭）
                if share_memory_enabled:
                    for key, value in seq_data.items():
                        if isinstance(value, torch.Tensor):
                            value.share_memory_()

                # 记录真实 IMU 是否存在（一次全局统计，不重复打印）
                if seq_data.get("human_imu_real") is not None:
                    self.has_human_real_imu = True
                if seq_data.get("obj_imu_real") is not None:
                    self.has_obj_real_imu = True

                # 存储预加载的数据 (使用 file_path 作为 key)
                self.loaded_data[file_path] = seq_data

                # -- 收集序列信息 --
                self.sequence_info.append({
                    "file_path": file_path,
                    "seq_name": seq_name,
                    "seq_len": seq_len,
                    "has_object": has_object_flag,
                })

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                import traceback
                traceback.print_exc() # 打印更详细的错误追踪

    def cleanup(self):
        """清理共享内存和缓存数据"""
        print(f"清理IMUDataset，释放{len(self.loaded_data)}个序列的共享内存...")
        self.loaded_data.clear()
        self.sequence_info.clear()

        # 强制垃圾回收
        gc.collect()

        print("IMUDataset清理完成")

    def __del__(self):
        """析构函数，确保在对象被删除时清理共享内存"""
        if getattr(sys, "meta_path", None) is None:
            return
        try:
            if hasattr(self, 'loaded_data') and self.loaded_data:
                self.cleanup()
        except Exception:
            pass

    def __len__(self):
        # 返回独立序列的数量
        return len(self.sequence_info)

    def __getitem__(self, idx):
        # 1. 获取序列信息
        try:
            seq_info = self.sequence_info[idx]
            file_path = seq_info["file_path"]
            seq_name = seq_info["seq_name"]
            seq_len = seq_info["seq_len"]
            seq_has_object = seq_info.get("has_object", None)
        except IndexError:
             print(f"错误：索引 {idx} 超出 sequence_info 范围 (大小: {len(self.sequence_info)})")
             raise IndexError(f"索引 {idx} 超出范围")

        # 2. 选择切片范围
        if self.full_sequence:
            # 整段模式：不裁剪，直接使用全长度
            start_idx = 0
            end_idx = seq_len
        else:
            # 随机生成 start_idx（窗口模式）
            # 有效 start_idx 范围：[1, seq_len - window_size] (包含两端)
            max_start_idx = seq_len - self.window_size
            if self.random_window:
                start_idx = torch.randint(1, max_start_idx + 1, (1,)).item()
            else:
                start_idx = 1 + (max_start_idx - 1) // 2
            # start_idx = 1
            end_idx = start_idx + self.window_size # 切片时使用 end_idx

        # 3. 从预加载数据中获取序列数据
        try:
             seq_data = self.loaded_data[file_path]
        except KeyError:
             print(f"错误：无法在预加载数据中找到文件路径 {file_path} 对应的键。序列索引: {idx}")
             raise KeyError(f"无法找到文件 {file_path}")

        # 4. 切片和处理数据
        try:
            # --- 从预加载的 seq_data 中取出窗口数据 ---
            # 注意：无需再次 torch.load(file_path)

            trans = seq_data["trans"][start_idx:end_idx, :]   # [seq, 3]
            if trans.dtype == np.float32:
                trans = torch.from_numpy(trans).float()

            # 获取实际的序列长度（考虑full_sequence模式）
            actual_seq_len = end_idx - start_idx

            pose = matrix_to_axis_angle(rotation_6d_to_matrix(
                seq_data["rotation_local_full_gt_list"].reshape(seq_len, -1, 6)[start_idx:end_idx])).reshape(actual_seq_len, -1) # [seq, 66]
            local_rotation_6d = seq_data["rotation_local_full_gt_list"].reshape(
                seq_len, -1, 6
            )[start_idx:end_idx].float()
            raw_position_global = seq_data["position_global_full_gt_world"]
            raw_rotation_global = seq_data["rotation_global"]
            if not isinstance(raw_position_global, torch.Tensor):
                raw_position_global = torch.as_tensor(raw_position_global)
            if not isinstance(raw_rotation_global, torch.Tensor):
                raw_rotation_global = torch.as_tensor(raw_rotation_global)
            raw_position_global = raw_position_global.float()
            raw_rotation_global = raw_rotation_global.float()
            trans_sequence = seq_data["trans"]
            if not isinstance(trans_sequence, torch.Tensor):
                trans_sequence = torch.as_tensor(trans_sequence)
            trans_sequence = trans_sequence.float()
            globalpose_mesh_targets = self._globalpose_mesh_targets(
                file_path,
                seq_data,
                seq_data["rotation_local_full_gt_list"].float(),
                trans_sequence,
            )
            globalpose_sensor_positions_sequence = (
                globalpose_mesh_targets["sensor_positions"]
                if globalpose_mesh_targets is not None
                else None
            )
            # Preserve the complete sequence for finite-difference IMU
            # synthesis.  Windowing before differentiation creates artificial
            # endpoint accelerations for every random training crop.
            position_global_sequence = append_virtual_palm_joints(
                raw_position_global, num_joints=24
            )
            rotation_global_sequence = append_virtual_palm_rotations(
                raw_rotation_global, num_joints=24
            )
            position_global_full = position_global_sequence[start_idx:end_idx]
            rotation_global_full = rotation_global_sequence[start_idx:end_idx]
            ori_glb_reduced = rotation_global_full[:, _REDUCED_INDICES] # [seq, n_reduced, 3, 3]
            ori_root_reduced = rotation_global_full[:, :1].transpose(-1, -2).matmul(ori_glb_reduced)

            device = position_global_full.device
            dtype = position_global_full.dtype

            def _slice_real_imu(data):
                if data is None:
                    return None
                if isinstance(data, torch.Tensor):
                    tensor = data[start_idx:end_idx]
                else:
                    tensor = torch.from_numpy(np.asarray(data))[start_idx:end_idx]
                if tensor.shape[0] < actual_seq_len:
                    return None
                return tensor.to(device=device, dtype=dtype)

            human_imu_real = _slice_real_imu(seq_data.get("human_imu_real"))
            obj_imu_real = _slice_real_imu(seq_data.get("obj_imu_real"))
            if human_imu_real is not None and human_imu_real.shape[0] > actual_seq_len:
                human_imu_real = human_imu_real[:actual_seq_len]
            if obj_imu_real is not None and obj_imu_real.shape[0] > actual_seq_len:
                obj_imu_real = obj_imu_real[:actual_seq_len]

            seq_noise_cfg_raw = seq_data.get("imu_noise_params", None)
            seq_noise_cfg = merge_noise_cfg(seq_noise_cfg_raw) if seq_noise_cfg_raw is not None else self.imu_noise_cfg
            imu_noise_applied = False
            noise_seed = _stable_sequence_seed(
                self.simulate_imu_noise_seed,
                os.path.realpath(os.path.abspath(file_path)),
            )
            noise_generator = _make_noise_generator(noise_seed, device)

            # The released TIP loader normalizes against frame zero of the
            # complete sequence before extracting a training window.  Build
            # that stream here, once per sequence, and slice it below.
            tip_reference = None
            transpose_reference = None
            if self.imu_protocol == "tip_reference":
                raw_obj_trans = seq_data.get("obj_trans")
                if raw_obj_trans is not None:
                    raw_obj_trans = torch.as_tensor(raw_obj_trans).float()
                    if raw_obj_trans.ndim == 3 and raw_obj_trans.shape[1] == 1:
                        raw_obj_trans = raw_obj_trans[:, 0]
                raw_obj_rot = seq_data.get("obj_rot")
                if raw_obj_rot is not None:
                    raw_obj_rot = torch.as_tensor(raw_obj_rot).float()
                tip_reference = _build_tip_reference_features(
                    raw_position_global,
                    raw_rotation_global,
                    raw_obj_trans,
                    raw_obj_rot,
                    FRAME_RATE,
                )
            elif self.imu_protocol == "transpose_reference":
                raw_obj_trans = seq_data.get("obj_trans")
                if raw_obj_trans is not None:
                    raw_obj_trans = torch.as_tensor(raw_obj_trans).float()
                    if raw_obj_trans.ndim == 3 and raw_obj_trans.shape[1] == 1:
                        raw_obj_trans = raw_obj_trans[:, 0]
                raw_obj_rot = seq_data.get("obj_rot")
                if raw_obj_rot is not None:
                    raw_obj_rot = torch.as_tensor(raw_obj_rot).float()
                transpose_reference = _build_transpose_reference_features(
                    raw_position_global,
                    raw_rotation_global,
                    raw_obj_trans,
                    raw_obj_rot,
                    FRAME_RATE,
                    noise_cfg=seq_noise_cfg,
                    noise_generator=noise_generator,
                    simulate_noise=self.simulate_imu_noise,
                )

            # --- DynaIP 风格的人体 IMU 与速度处理 ---
            tip_root_vel = None
            tip_obj_vel = None
            tip_root_rot0 = None
            tip_root_pos0 = None
            if human_imu_real is not None:
                human_imu = human_imu_real
                imu_noise_applied = True
            elif tip_reference is not None:
                human_full, _, tip_root_vel_full, tip_obj_vel_full, tip_root_rot0, tip_root_pos0 = tip_reference
                if self.simulate_imu_noise:
                    human_acc, human_rot6d, _ = apply_imu_noise(
                        human_full[:, :, :3],
                        human_full[:, :, 3:],
                        FRAME_RATE,
                        seq_noise_cfg,
                        generator=noise_generator,
                        rotation_layout="tip_columns",
                    )
                    human_full = torch.cat((human_acc, human_rot6d), dim=-1)
                    imu_noise_applied = True
                human_imu = human_full[start_idx:end_idx]
                tip_root_vel = tip_root_vel_full[start_idx:end_idx].to(device=device, dtype=dtype)
                tip_obj_vel = tip_obj_vel_full[start_idx:end_idx].to(device=device, dtype=dtype)
                tip_root_rot0 = tip_root_rot0.to(device=device, dtype=dtype)
                tip_root_pos0 = tip_root_pos0.to(device=device, dtype=dtype)
            elif transpose_reference is not None:
                human_full, object_full = transpose_reference
                # Keep the exact released layout: all six accelerations first,
                # followed by all six 3x3 orientations.
                human_imu = human_full[start_idx:end_idx]
                # The official TransPose network consumes flattened
                # [acc(3), rotation(9)] per sensor after normalization.
                # Keep a matrix-valued object stream for the adapter below.
                obj_imu_protocol = object_full[start_idx:end_idx]
            else:
                if self.imu_protocol == "globalpose_reference":
                    # Build GlobalPose's official root-last absolute stream.
                    human_builder = _build_globalpose_reference_imu
                    human_sensor_pos = _GLOBALPOSE_REFERENCE_POSITION_JOINTS
                    human_sensor_rot = _GLOBALPOSE_REFERENCE_ROTATION_JOINTS
                else:
                    human_builder = (
                    _build_human_imu_root_central
                    if self.imu_protocol == "dip18_reference"
                    else _build_human_imu_root
                    )
                    human_sensor_pos = _SENSOR_POS_INDICES
                    human_sensor_rot = _SENSOR_ROT_INDICES
                if self.imu_protocol == "globalpose_reference":
                    human_imu_full = human_builder(
                        position_global_sequence,
                        rotation_global_sequence,
                        (
                            globalpose_sensor_positions_sequence
                            if globalpose_sensor_positions_sequence is not None
                            else human_sensor_pos
                        ),
                        human_sensor_rot,
                        FRAME_RATE,
                    )
                    human_imu = human_imu_full[start_idx:end_idx]
                else:
                    human_imu = human_builder(
                        position_global_full,
                        rotation_global_full,
                        human_sensor_pos,
                        human_sensor_rot,
                        FRAME_RATE,
                    )  # [seq, num_imus, 9]
                if self.simulate_imu_noise:
                    human_acc = human_imu[:, :, :3]
                    human_rot6d = human_imu[:, :, 3:]
                    human_acc, human_rot6d, _ = apply_imu_noise(
                        human_acc,
                        human_rot6d,
                        FRAME_RATE,
                        seq_noise_cfg,
                        generator=noise_generator,
                    )
                    human_imu = torch.cat((human_acc, human_rot6d), dim=-1)
                    imu_noise_applied = True
            joint_vel_root = _compute_joint_velocity_root(rotation_global_full, position_global_full, FRAME_RATE)
            root_vel = _central_diff(position_global_full[:, 0, :], 1.0 / FRAME_RATE)
            sensor_vel_root = joint_vel_root[:, _VEL_SELECTION_INDEX_LIST, :] # [seq, 6, 3]
            sensor_vel_glb = _central_diff(position_global_full[:, _SENSOR_POS_INDICES, :], 1.0 / FRAME_RATE)

            # --- 物体数据准备 ---
            # 优先使用显式标记，其次根据obj字段推断
            has_object = seq_has_object
            if has_object is None:
                has_object = ("obj_trans" in seq_data and seq_data["obj_trans"] is not None)
            has_object = bool(has_object)
            obj_name = ""
            imu_dim = human_imu.shape[-1]
            obj_feature_dim = (
                12 if self.imu_protocol == "transpose_reference" else imu_dim
            )
            obj_trans = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)
            obj_rot = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(actual_seq_len, 1, 1)
            obj_rot_6d = matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3)).reshape(actual_seq_len, 6)
            obj_scale = torch.ones(actual_seq_len, device=device, dtype=dtype)
            obj_imu = torch.zeros(
                actual_seq_len, obj_feature_dim, device=device, dtype=dtype
            )
            obj_vel = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)
            obj_points_canonical = torch.zeros(self.obj_points_sample_count, 3, device=device, dtype=dtype)

            if has_object:
                obj_name = seq_data.get("obj_name", "unknown_object")
                obj_trans = seq_data["obj_trans"][start_idx:end_idx].squeeze(-1).to(device=device, dtype=dtype)
                obj_rot = seq_data["obj_rot"][start_idx:end_idx].to(device=device, dtype=dtype)
                obj_rot_6d = matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3)).reshape(actual_seq_len, 6)
                obj_scale = seq_data["obj_scale"][start_idx:end_idx].to(device=device, dtype=dtype)
                if obj_imu_real is not None:
                    obj_imu = obj_imu_real
                    if obj_imu.dim() == 3 and obj_imu.shape[1] == 1:
                        obj_imu = obj_imu.squeeze(1)
                    imu_noise_applied = True
                elif tip_reference is not None:
                    _, object_full, _, _, _, _ = tip_reference
                    if self.simulate_imu_noise:
                        obj_acc, obj_rot6d, _ = apply_imu_noise(
                            object_full[:, :3],
                            object_full[:, 3:],
                            FRAME_RATE,
                            seq_noise_cfg,
                            generator=noise_generator,
                        )
                        # The public TIP loader uses all-zero orientation for
                        # the processed OMOMO object matrix fallback. Do not
                        # turn that sentinel into a random orientation.
                        if torch.count_nonzero(object_full[:, 3:]) == 0:
                            obj_rot6d = object_full[:, 3:]
                        object_full = torch.cat((obj_acc, obj_rot6d), dim=-1)
                        imu_noise_applied = True
                    obj_imu = object_full[start_idx:end_idx]
                elif transpose_reference is not None:
                    obj_imu = obj_imu_protocol.to(device=device, dtype=dtype)
                else:
                    object_builder = (
                        _build_object_imu_central
                        if self.imu_protocol == "dip18_reference"
                        else _build_object_imu
                    )
                    obj_imu = object_builder(
                        obj_rot_6d, obj_trans, FRAME_RATE
                    )  # [seq, 9]
                    if self.simulate_imu_noise:
                        obj_acc = obj_imu[:, :3]
                        obj_rot6d = obj_imu[:, 3:]
                        obj_acc, obj_rot6d, _ = apply_imu_noise(
                            obj_acc,
                            obj_rot6d,
                            FRAME_RATE,
                            seq_noise_cfg,
                            generator=noise_generator,
                        )
                        obj_imu = torch.cat((obj_acc, obj_rot6d), dim=-1)
                        imu_noise_applied = True
                obj_vel = _central_diff(obj_trans, 1.0 / FRAME_RATE)
                obj_points_val = seq_data.get("obj_points_canonical")
                if isinstance(obj_points_val, torch.Tensor) and obj_points_val.dim() == 2 and obj_points_val.shape[-1] == 3:
                    points = obj_points_val.to(device=device, dtype=dtype)
                    n_pts = int(points.shape[0])
                    if n_pts > self.obj_points_sample_count:
                        idx = torch.linspace(
                            0,
                            max(n_pts - 1, 0),
                            steps=self.obj_points_sample_count,
                            device=device,
                            dtype=torch.float32,
                        ).long()
                        points = points[idx]
                    elif n_pts < self.obj_points_sample_count and n_pts > 0:
                        pad = points[-1:].expand(self.obj_points_sample_count - n_pts, -1)
                        points = torch.cat([points, pad], dim=0)
                    if points.shape[0] == self.obj_points_sample_count:
                        obj_points_canonical = points
            else:
                # 无物体时保持占位但不使用接触信息
                obj_rot_6d = matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3)).reshape(actual_seq_len, 6)

            # --- 提前初始化所有接触窗口变量 ---
            def _slice_contact_window(key, default_dtype):
                value = seq_data.get(key)
                if value is None:
                    return torch.zeros(actual_seq_len, dtype=default_dtype)
                if isinstance(value, torch.Tensor):
                    return value[start_idx:end_idx].clone()
                return torch.from_numpy(np.asarray(value))[start_idx:end_idx].clone()

            lfoot_contact_window = _slice_contact_window("lfoot_contact", torch.float).float()
            rfoot_contact_window = _slice_contact_window("rfoot_contact", torch.float).float()
            lhand_contact_window = _slice_contact_window("lhand_contact", torch.bool).bool()
            rhand_contact_window = _slice_contact_window("rhand_contact", torch.bool).bool()
            obj_contact_window = _slice_contact_window("obj_contact", torch.bool).bool()
            if not has_object:
                lhand_contact_window = torch.zeros_like(lhand_contact_window)
                rhand_contact_window = torch.zeros_like(rhand_contact_window)
                obj_contact_window = torch.zeros_like(obj_contact_window)
            elif self.resolve_bimanual_contact_conflicts:
                hand_pos_for_contact = select_hand_anchor_positions(position_global_full)
                lhand_raw_lb = torch.norm(obj_trans - hand_pos_for_contact[:, 0, :], dim=-1)
                rhand_raw_lb = torch.norm(obj_trans - hand_pos_for_contact[:, 1, :], dim=-1)
                lhand_contact_window, rhand_contact_window = suppress_bimanual_contact_by_bone_variation(
                    lhand_contact_window,
                    rhand_contact_window,
                    lhand_raw_lb,
                    rhand_raw_lb,
                )

            # --- 虚拟关节监督 ---
            lhand_obj_direction = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)
            rhand_obj_direction = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)
            lhand_lb = torch.zeros(actual_seq_len, device=device, dtype=dtype)
            rhand_lb = torch.zeros(actual_seq_len, device=device, dtype=dtype)

            if has_object:
                try:
                    virtual_joint_data = compute_obj_direction_supervision(
                        position_global_full, obj_trans, obj_rot_6d,
                        lhand_contact_window, rhand_contact_window
                    )

                    lhand_obj_direction = virtual_joint_data["lhand_obj_direction"]
                    rhand_obj_direction = virtual_joint_data["rhand_obj_direction"]

                    # 手-物距离（骨长），在无接触时置零
                    hand_pos_gt = select_hand_anchor_positions(position_global_full)
                    lhand_pos_gt = hand_pos_gt[:, 0, :]
                    rhand_pos_gt = hand_pos_gt[:, 1, :]
                    lhand_lb = torch.norm(obj_trans - lhand_pos_gt, dim=-1)
                    rhand_lb = torch.norm(obj_trans - rhand_pos_gt, dim=-1)
                    lhand_lb = lhand_lb * lhand_contact_window.float()
                    rhand_lb = rhand_lb * rhand_contact_window.float()

                except Exception as e:
                    print(f"计算虚拟关节数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    lhand_obj_direction = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)
                    rhand_obj_direction = torch.zeros(actual_seq_len, 3, device=device, dtype=dtype)
                    lhand_lb = torch.zeros(actual_seq_len, device=device, dtype=dtype)
                    rhand_lb = torch.zeros(actual_seq_len, device=device, dtype=dtype)

            # --- 交互起止帧标签（接触段端点，含窗口截断mask）---
            def _full_contact_state(key, index):
                """原始（未切窗）序列在 index 帧的接触状态，越界视为无接触。"""
                value = seq_data.get(key)
                if value is None or index < 0:
                    return False
                if not isinstance(value, torch.Tensor):
                    value = torch.from_numpy(np.asarray(value))
                if index >= int(value.shape[0]):
                    return False
                return bool(value[index])

            contact_keys = ("lhand_contact", "rhand_contact", "obj_contact")
            if has_object:
                prev_contact = tuple(_full_contact_state(key, start_idx - 1) for key in contact_keys)
                next_contact = tuple(_full_contact_state(key, end_idx) for key in contact_keys)
            else:
                prev_contact = (False, False, False)
                next_contact = (False, False, False)

            contact_endpoint, contact_endpoint_gauss = build_contact_endpoint_labels(
                lhand_contact_window,
                rhand_contact_window,
                obj_contact_window,
                prev_contact=prev_contact,
                next_contact=next_contact,
                sigma=2.0,
                device=device,
                dtype=dtype,
            )
            start_label = contact_endpoint[:, 4]
            end_label = contact_endpoint[:, 5]
            start_gauss = contact_endpoint_gauss[:, 4]
            end_gauss = contact_endpoint_gauss[:, 5]

            result = {
                "human_imu": human_imu.float(),  # [seq, num_imus, feature_dim]
                "obj_imu": obj_imu.float(),  # [seq, 9] or [seq, 12]
                "imu_noise_applied": bool(imu_noise_applied),
                "imu_protocol": self.imu_protocol,
                "seq_name": seq_name,
                "seq_path": file_path,
                "seq_file": os.path.basename(file_path),
                "window_start": int(start_idx),
                "window_end": int(end_idx),

                "trans": trans.float(),
                "ori_root_reduced": ori_root_reduced.float(),
                "has_object": has_object,
                "obj_trans": obj_trans.float(),
                "obj_rot": obj_rot_6d.float(),  # 使用6D表示而不是矩阵
                "obj_scale": obj_scale.float(),
                "obj_name": obj_name,
                "obj_points_canonical": obj_points_canonical.float(),

                "root_vel": root_vel.float(),
                "obj_vel": obj_vel.float(),
                "tip_root_vel": (
                    tip_root_vel.float()
                    if tip_root_vel is not None
                    else root_vel.float()
                ),
                "tip_obj_vel": (
                    tip_obj_vel.float()
                    if tip_obj_vel is not None
                    else obj_vel.float()
                ),
                "tip_root_rot0": (
                    tip_root_rot0.float()
                    if tip_root_rot0 is not None
                    else rotation_global_full[0, 0].float()
                ),
                "tip_root_pos0": (
                    tip_root_pos0.float()
                    if tip_root_pos0 is not None
                    else position_global_full[0, 0].float()
                ),
                "sensor_vel_root": sensor_vel_root.float(),
                "sensor_vel_glb": sensor_vel_glb.float(),

                "lhand_contact": lhand_contact_window.bool(),
                "rhand_contact": rhand_contact_window.bool(),
                "obj_contact": obj_contact_window.bool(),
                "interaction_start": start_label.float(),
                "interaction_end": end_label.float(),
                "interaction_start_gauss": start_gauss.float(),
                "interaction_end_gauss": end_gauss.float(),
                "contact_endpoint": contact_endpoint.float(),
                "contact_endpoint_gauss": contact_endpoint_gauss.float(),
                "lfoot_contact": lfoot_contact_window.float(),
                "rfoot_contact": rfoot_contact_window.float(),
                "lhand_obj_direction": lhand_obj_direction.float(),
                "rhand_obj_direction": rhand_obj_direction.float(),
                "lhand_lb": lhand_lb.float(),
                "rhand_lb": rhand_lb.float(),

                "position_global": position_global_full.float(),
                "rotation_global": rotation_global_full.float(),
                "pose": pose.float(),
                "rotation_local_6d": local_rotation_6d.float(),
            }
            if globalpose_sensor_positions_sequence is not None:
                result["globalpose_sensor_positions"] = (
                    globalpose_sensor_positions_sequence[start_idx:end_idx]
                    .to(device=device, dtype=dtype)
                    .float()
                )
                result["globalpose_canonical_joints"] = (
                    globalpose_mesh_targets["canonical_joints"][start_idx:end_idx]
                    .to(device=device, dtype=dtype)
                    .float()
                )
                result["globalpose_contact_joints"] = (
                    globalpose_mesh_targets["joints"][start_idx:end_idx]
                    .to(device=device, dtype=dtype)
                    .float()
                )

            return result

        except Exception as e:
            print(f"处理预加载数据时出错，文件: {file_path}, 窗口索引: {idx}, Start: {start_idx}, End: {end_idx}: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误
            # 抛出异常让DataLoader跳过该样本
            raise RuntimeError(f"处理样本{idx}时出错: {e}")
