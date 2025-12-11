#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo 数据处理脚本：整合 Noitom（人体）与 STAG（物体） IMU，构建 IMUHOI StageNet 所需 data_dict
"""
import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import pytorch3d.transforms as transforms
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.signal import find_peaks

# 确保可以导入项目模块
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from human_body_prior.body_model.body_model import BodyModel
from configs import _SENSOR_VEL_NAMES, _REDUCED_POSE_NAMES

# ===================== 常量配置 =====================
DEFAULT_SENSOR_ORDER = ["Hips", "LeftLeg", "RightLeg", "Head", "LeftForeArm", "RightForeArm"]
SENSOR_JOINT_MAP = {
    "Hips": "Hips", "LeftLeg": "LeftFoot", "RightLeg": "RightFoot",
    "Head": "Head", "LeftForeArm": "LeftHand", "RightForeArm": "RightHand"
}
TARGET_FPS = 30
NOITOM_SRC_FPS = 60
STAG_SRC_FPS = 100
GRAVITY_ACC = 9.81
GRAVITY_MPS2 = 9.80665
ACC_RAW_TO_G = 0.00048828
GYRO_RAW_TO_RAD = 0.00122173


# ===================== 数据结构 =====================
@dataclass
class SensorMeta:
    start_idx: int
    end_idx: int
    tpose_start: int
    tpose_end: int
    start_jumps: List[int]
    end_jumps: List[int]
    tpose_original: Tuple[int, int]


@dataclass
class SensorSequence:
    name: str
    acc: np.ndarray   # [N, 3]
    quat: np.ndarray  # [N, 4]
    rot: np.ndarray   # [N, 3, 3]


# ===================== EKF 姿态估计 =====================
def _ekf_skew(vec):
    x, y, z = vec
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _ekf_quat_normalize(q):
    norm = np.linalg.norm(q)
    return np.array([1.0, 0.0, 0.0, 0.0]) if norm < 1e-12 else q / norm


def _ekf_quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def _ekf_quat_from_small_angle(delta):
    return _ekf_quat_normalize(np.concatenate([[1.0], 0.5 * delta]))


def _ekf_euler_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll*0.5), np.sin(roll*0.5)
    cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
    cy, sy = np.cos(yaw*0.5), np.sin(yaw*0.5)
    return _ekf_quat_normalize(np.array([
        cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy
    ]))


def _ekf_quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def _ekf_rotate_vector_inv(q, vec):
    return _ekf_quat_to_rotmat(q).T @ vec


def _ekf_integrate_quat(q, omega, dt):
    wx, wy, wz = omega
    omega_mat = np.array([
        [0.0, -wx, -wy, -wz], [wx, 0.0, wz, -wy],
        [wy, -wz, 0.0, wx], [wz, wy, -wx, 0.0]
    ])
    return _ekf_quat_normalize(q + 0.5 * omega_mat @ q * dt)


def _ekf_initial_orientation(acc0, mag0=None):
    acc_vec = np.asarray(acc0, dtype=np.float64)
    if np.linalg.norm(acc_vec) < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0]), None
    acc_unit = acc_vec / (np.linalg.norm(acc_vec) + 1e-12)
    pitch = np.arctan2(acc_unit[0], np.sqrt(acc_unit[1]**2 + acc_unit[2]**2))
    roll = np.arctan2(-acc_unit[1], -acc_unit[2])
    yaw = 0.0
    mag_ref_world = None
    if mag0 is not None and np.linalg.norm(mag0) > 1e-6:
        mag_unit = mag0 / (np.linalg.norm(mag0) + 1e-12)
        mx, my, mz = mag_unit
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        mx2 = mx*cp + mz*sp
        my2 = mx*sr*sp + my*cr - mz*sr*cp
        yaw = np.arctan2(-my2, mx2)
        q_tmp = _ekf_euler_to_quat(roll, pitch, yaw)
        mag_ref_world = _ekf_quat_to_rotmat(q_tmp) @ mag_unit
    return _ekf_euler_to_quat(roll, pitch, yaw), mag_ref_world


def run_error_state_ekf(acc, gyro, mag=None, dts=0.01, g=GRAVITY_MPS2, use_mag=True,
                        mag_ref_world=None, acc_noise=0.35, gyro_noise=0.012,
                        bias_noise=1e-4, mag_noise=0.8, accel_gating=2.5):
    """误差状态扩展卡尔曼滤波器，用于从原始 IMU 数据估计姿态和线性加速度"""
    acc = np.asarray(acc, dtype=np.float64)
    gyro = np.asarray(gyro, dtype=np.float64)
    n = acc.shape[0]
    dt_array = np.full(n, dts) if np.isscalar(dts) else np.asarray(dts, dtype=np.float64)
    mag_data = np.asarray(mag, dtype=np.float64) if mag is not None else None

    q, mag_ref_init = _ekf_initial_orientation(acc[0], mag_data[0] if mag_data is not None else None)
    if mag_ref_world is None:
        mag_ref_world = mag_ref_init

    P = np.eye(6) * 1e-3
    quats = np.zeros((n, 4))
    biases = np.zeros((n, 3))
    lin_acc_world = np.zeros((n, 3))
    lin_acc_body = np.zeros((n, 3))
    gravity_vec_world = np.array([0.0, 0.0, -g])

    for i in range(n):
        dt = float(dt_array[i])
        omega = gyro[i] - biases[i-1] if i > 0 else gyro[i]
        q = _ekf_integrate_quat(q, omega, dt)

        F = np.eye(6)
        F[:3, :3] += -_ekf_skew(omega) * dt
        F[:3, 3:] = -np.eye(3) * dt
        G = np.zeros((6, 6))
        G[:3, :3] = -np.eye(3)
        G[3:, 3:] = np.eye(3)
        q_process = np.diag(np.concatenate([np.full(3, gyro_noise**2), np.full(3, bias_noise**2)])) * dt
        P = F @ P @ F.T + G @ q_process @ G.T

        H_blocks, z_blocks, R_blocks = [], [], []
        acc_body = acc[i]
        acc_norm = np.linalg.norm(acc_body)
        if acc_norm > 1e-6 and abs(acc_norm - g) < accel_gating:
            acc_dir_meas = acc_body / acc_norm
            acc_dir_pred = _ekf_rotate_vector_inv(q, np.array([0.0, 0.0, 1.0]))
            H_acc = np.zeros((3, 6))
            H_acc[:, :3] = -_ekf_skew(acc_dir_pred)
            H_blocks.append(H_acc)
            z_blocks.append(acc_dir_meas - acc_dir_pred)
            R_blocks.append(np.eye(3) * (acc_noise**2))

        if use_mag and mag_data is not None and mag_ref_world is not None:
            mag_body = mag_data[i]
            mag_norm, ref_norm = np.linalg.norm(mag_body), np.linalg.norm(mag_ref_world)
            if mag_norm > 1e-6 and ref_norm > 1e-6:
                mag_dir_meas = mag_body / mag_norm
                mag_dir_pred = _ekf_rotate_vector_inv(q, mag_ref_world / ref_norm)
                H_mag = np.zeros((3, 6))
                H_mag[:, :3] = -_ekf_skew(mag_dir_pred)
                H_blocks.append(H_mag)
                z_blocks.append(mag_dir_meas - mag_dir_pred)
                R_blocks.append(np.eye(3) * (mag_noise**2))

        if H_blocks:
            H = np.vstack(H_blocks)
            R = block_diag(*R_blocks)
            z = np.concatenate(z_blocks)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            dx = K @ z
            P = (np.eye(6) - K @ H) @ P
            q = _ekf_quat_multiply(q, _ekf_quat_from_small_angle(dx[:3]))
            q = _ekf_quat_normalize(q)
            if i < n - 1:
                biases[i] = biases[i-1] + dx[3:] if i > 0 else dx[3:]

        quats[i] = q
        if i < n - 1:
            biases[i + 1] = biases[i]
        R_mat = _ekf_quat_to_rotmat(q)
        lin_acc_body[i] = acc_body - R_mat.T @ gravity_vec_world
        lin_acc_world[i] = R_mat @ lin_acc_body[i]

    return {"quat": quats.astype(np.float32), "lin_acc_body": lin_acc_body.astype(np.float32)}


# ===================== IMU 数据处理通用函数 =====================
def smooth_imu_acceleration(acc_data, smooth_n=4, frame_rate=60):
    """对IMU加速度数据进行平滑"""
    if isinstance(acc_data, np.ndarray):
        acc_tensor = torch.tensor(acc_data, dtype=torch.float32)
        return_numpy = True
    else:
        acc_tensor = acc_data
        return_numpy = False

    orig_shape = acc_tensor.shape
    if len(orig_shape) == 3:
        T, N, D = orig_shape
        acc_flat = acc_tensor.reshape(T, -1)
    else:
        acc_flat = acc_tensor
    T = acc_flat.shape[0]
    smoothed_acc = acc_flat.clone()

    if smooth_n // 2 != 0 and T > smooth_n * 2:
        for i in range(smooth_n, T - smooth_n):
            weights = torch.ones(2 * smooth_n + 1, device=acc_flat.device)
            weights = weights / weights.sum()
            window_data = acc_flat[i - smooth_n:i + smooth_n + 1]
            smoothed_acc[i] = torch.sum(window_data * weights.unsqueeze(-1), dim=0)

    if T > 2:
        smoothed_acc[0] = (2 * acc_flat[0] + acc_flat[1]) / 3
        smoothed_acc[-1] = (2 * acc_flat[-1] + acc_flat[-2]) / 3

    smoothed_acc = smoothed_acc.reshape(orig_shape)
    return smoothed_acc.numpy() if return_numpy else smoothed_acc


def detect_jump_sequences(lin_acc_data, magnitude_threshold=20.0, min_peak_distance=20,
                          min_jump_interval=20, max_jump_interval=80):
    """检测连续的跳跃序列"""
    acc_magnitude = np.sqrt(np.sum(lin_acc_data**2, axis=1))
    peaks, _ = find_peaks(acc_magnitude, height=magnitude_threshold, distance=min_peak_distance)

    if len(peaks) < 6:
        print(f"警告：只检测到 {len(peaks)} 个峰值")
        return np.array([]), np.array([])

    print(f"检测到 {len(peaks)} 个加速度峰值: {peaks[:10]}...")

    jump_sequences = []
    i = 0
    while i < len(peaks) - 2:
        current_sequence = [peaks[i]]
        j = i + 1
        while j < len(peaks) and len(current_sequence) < 3:
            time_gap = peaks[j] - current_sequence[-1]
            if min_jump_interval <= time_gap <= max_jump_interval:
                current_sequence.append(peaks[j])
                j += 1
            elif time_gap < min_jump_interval:
                j += 1
            else:
                break
        if len(current_sequence) == 3:
            jump_sequences.append(current_sequence)
            i = j
        else:
            i += 1

    if len(jump_sequences) < 2:
        print("警告：未检测到足够的跳跃序列")
        return np.array([]), np.array([])

    return np.array(jump_sequences[0]), np.array(jump_sequences[-1])


def apply_coordinate_transform(acc_data, quat_data):
    """应用坐标系变换，从IMU坐标系转换到SMPL坐标系"""
    transform_matrix = torch.tensor([
        [-1, 0, 0], [0, 0, 1], [0, 1, 0]
    ], dtype=torch.float32)

    acc_tensor = torch.tensor(acc_data, dtype=torch.float32)
    transformed_acc = torch.matmul(acc_tensor, transform_matrix.T)

    quat_tensor = torch.tensor(quat_data, dtype=torch.float32)
    R_imu = transforms.quaternion_to_matrix(quat_tensor)
    N = R_imu.shape[0]
    transformed_ori = torch.zeros(N, 3, 3, dtype=torch.float32)
    for i in range(N):
        transformed_ori[i] = transform_matrix @ R_imu[i] @ transform_matrix.T

    return transformed_acc.numpy(), transformed_ori.numpy()


# ===================== 数据加载与重采样 =====================
def _resample_linear(data: np.ndarray, src_fps: int, dst_fps: int) -> np.ndarray:
    if src_fps == dst_fps or data.shape[0] == 0:
        return data.copy()
    src_len, dst_len = data.shape[0], max(1, int(round(data.shape[0] * dst_fps / src_fps)))
    t_old, t_new = np.linspace(0.0, 1.0, src_len), np.linspace(0.0, 1.0, dst_len)
    resampled = np.empty((dst_len, data.shape[1]), dtype=np.float32)
    for d in range(data.shape[1]):
        resampled[:, d] = np.interp(t_new, t_old, data[:, d])
    return resampled


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, frac: float) -> torch.Tensor:
    q0, q1 = q0 / (q0.norm() + 1e-8), q1 / (q1.norm() + 1e-8)
    dot = torch.dot(q0, q1)
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = torch.clamp(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + frac * (q1 - q0)
        return result / (result.norm() + 1e-8)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * frac
    return ((torch.sin(theta_0 - theta) / sin_theta_0) * q0 + 
            (torch.sin(theta) / sin_theta_0) * q1)


def _resample_quat(quat: np.ndarray, src_fps: int, dst_fps: int) -> np.ndarray:
    if src_fps == dst_fps or quat.shape[0] == 0:
        return quat.copy()
    src_len, dst_len = quat.shape[0], max(1, int(round(quat.shape[0] * dst_fps / src_fps)))
    t_new = np.linspace(0.0, 1.0, dst_len)
    q_tensor = torch.from_numpy(quat.astype(np.float32))
    result = torch.zeros(dst_len, 4, dtype=torch.float32)
    for i, tn in enumerate(t_new):
        if tn <= 0:
            result[i] = q_tensor[0]
        elif tn >= 1:
            result[i] = q_tensor[-1]
        else:
            pos = tn * (src_len - 1)
            idx0, idx1 = int(np.floor(pos)), min(int(np.floor(pos)) + 1, src_len - 1)
            frac = float(pos - idx0)
            result[i] = q_tensor[idx0] if idx0 == idx1 else _quat_slerp(q_tensor[idx0], q_tensor[idx1], frac)
    return result.numpy()


def _resample_sensor(acc: np.ndarray, quat: np.ndarray, src_fps: int) -> Tuple[np.ndarray, np.ndarray]:
    return _resample_linear(acc, src_fps, TARGET_FPS), _resample_quat(quat, src_fps, TARGET_FPS)


# ===================== Noitom 人体数据处理 =====================
def _load_raw_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"无法找到输入文件: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = [line.strip() for line in f if line.strip()]
    if not rows:
        raise ValueError(f"{path} 内容为空")
    try:
        int(rows[0])
        header, data_rows = rows[1].split(","), [row.split(",") for row in rows[2:]]
    except ValueError:
        header, data_rows = rows[0].split(","), [row.split(",") for row in rows[1:]]
    return header, data_rows


def _resolve_bone_joint_indices(header: List[str], bone_name: str, joint_name: str) -> Dict[str, int]:
    wanted_bone = {
        "quat_x": f"{bone_name}-Bone-Quat-x", "quat_y": f"{bone_name}-Bone-Quat-y",
        "quat_z": f"{bone_name}-Bone-Quat-z", "quat_w": f"{bone_name}-Bone-Quat-w",
    }
    wanted_joint = {
        "pos_x": f"{joint_name}-Joint-Posi-x", "pos_y": f"{joint_name}-Joint-Posi-y",
        "pos_z": f"{joint_name}-Joint-Posi-z",
    }
    indices = {}
    for idx, name in enumerate(header):
        for key, pattern in {**wanted_bone, **wanted_joint}.items():
            if pattern == name or pattern in name:
                indices[key] = idx
    missing = [k for k in {**wanted_bone, **wanted_joint} if k not in indices]
    if missing:
        raise ValueError(f"传感器 {bone_name}/{joint_name} 缺少字段: {', '.join(missing)}")
    return indices


def _read_bone_joint_series(rows: List[List[str]], indices: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    pos_list, quat_list = [], []
    for line in rows:
        try:
            pos_list.append([float(line[indices[k]]) for k in ["pos_x", "pos_y", "pos_z"]])
            quat_list.append([float(line[indices[k]]) for k in ["quat_w", "quat_x", "quat_y", "quat_z"]])
        except (ValueError, IndexError):
            continue
    if not pos_list:
        raise ValueError("无法从 CSV 中解析有效的人体数据")
    return np.asarray(pos_list, dtype=np.float32), np.asarray(quat_list, dtype=np.float32)


def _compute_acc_from_pos(pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.gradient(pos, axis=0) / dt
    return np.gradient(vel, axis=0) / dt


def _parse_noitom_sensors(csv_path: str, sensor_order: List[str], gravity_scale: float) -> Dict[str, SensorSequence]:
    header, rows = _load_raw_csv(csv_path)
    sensors = {}
    dt = 1.0 / NOITOM_SRC_FPS
    for name in sensor_order:
        joint_name = SENSOR_JOINT_MAP.get(name, name)
        indices = _resolve_bone_joint_indices(header, name, joint_name)
        pos_np, quat_np = _read_bone_joint_series(rows, indices)
        acc_np = smooth_imu_acceleration(_compute_acc_from_pos(pos_np, dt), smooth_n=4, frame_rate=NOITOM_SRC_FPS)
        acc_np, quat_np = _resample_sensor(acc_np, quat_np, NOITOM_SRC_FPS)
        rot_np = transforms.quaternion_to_matrix(torch.from_numpy(quat_np)).numpy()
        sensors[name] = SensorSequence(name=name, acc=acc_np, quat=quat_np, rot=rot_np)
    return sensors


def _detect_range_from_hips(hip_seq: SensorSequence, threshold: float = None) -> SensorMeta:
    acc_for_detection = hip_seq.acc[:, 1:2]
    if threshold is None:
        acc_mag = np.sqrt(np.sum(acc_for_detection**2, axis=1))
        threshold = max(2.0, float(np.percentile(acc_mag, 85)))
    print(f"[Noitom] Hips 检测阈值: {threshold:.2f}")

    start_seq, end_seq = detect_jump_sequences(acc_for_detection, magnitude_threshold=threshold,
                                                min_peak_distance=5, min_jump_interval=10, max_jump_interval=30)
    if len(start_seq) > 0:
        start_idx = int(start_seq[-1]) + 100
    else:
        start_idx = 0
    end_idx = int(end_seq[0]) - 100 if len(end_seq) > 0 else len(hip_seq.acc)
    start_idx = max(0, min(start_idx, len(hip_seq.acc)))
    end_idx = max(start_idx, min(end_idx, len(hip_seq.acc)))
    if end_idx <= start_idx:
        start_idx, end_idx = 0, len(hip_seq.acc)

    return SensorMeta(start_idx=start_idx, end_idx=end_idx, tpose_start=-1, tpose_end=-1,
                      start_jumps=start_seq.tolist(), end_jumps=end_seq.tolist(), tpose_original=(-1, -1))


# ===================== STAG 物体数据处理 =====================
def process_stag_with_ekf(csv_path: str, sampling_rate: float = 100.0, use_mag: bool = False):
    """使用误差状态 EKF 处理 STAG 原始 IMU 数据"""
    import pandas as pd
    df = pd.read_csv(csv_path)
    required_cols = ['raw_acc_x', 'raw_acc_y', 'raw_acc_z', 'raw_gyro_x', 'raw_gyro_y', 'raw_gyro_z']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"STAG CSV 缺少必需列: {col}")

    acc_mps2 = df[['raw_acc_x', 'raw_acc_y', 'raw_acc_z']].to_numpy(dtype=np.float64) * ACC_RAW_TO_G * GRAVITY_MPS2
    gyro_rads = df[['raw_gyro_x', 'raw_gyro_y', 'raw_gyro_z']].to_numpy(dtype=np.float64) * GYRO_RAW_TO_RAD
    mag_data = df[['raw_mag_x', 'raw_mag_y', 'raw_mag_z']].to_numpy(dtype=np.float64) if use_mag and {'raw_mag_x', 'raw_mag_y', 'raw_mag_z'}.issubset(df.columns) else None

    if 'frame_index' in df.columns:
        frames = df['frame_index'].to_numpy(dtype=np.float64)
        dt_array = np.diff(frames, prepend=frames[0]) / float(sampling_rate)
    else:
        dt_array = np.full(len(acc_mps2), 1.0 / float(sampling_rate), dtype=np.float64)

    ekf_result = run_error_state_ekf(acc_mps2, gyro_rads, mag=mag_data, dts=dt_array, use_mag=use_mag)
    print(f"[EKF] 处理完成: {len(ekf_result['lin_acc_body'])} 帧")
    return ekf_result["lin_acc_body"], ekf_result["quat"]


def _load_object_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    print(f"[STAG] 使用 EKF 处理原始 IMU 数据: {csv_path}")
    acc, quat = process_stag_with_ekf(csv_path, sampling_rate=STAG_SRC_FPS, use_mag=False)
    return _resample_sensor(acc, quat, STAG_SRC_FPS)


# ===================== ZUPT 偏置估计 =====================
def estimate_static_bias(acc: np.ndarray, static_start: int, static_end: int, acc_threshold: float = 0.5) -> Tuple[np.ndarray, bool]:
    static_start, static_end = max(0, static_start), min(len(acc), static_end)
    if static_end <= static_start:
        print("[ZUPT] 警告: 静止期范围无效")
        return np.zeros(3, dtype=np.float32), False
    static_acc = acc[static_start:static_end]
    acc_std_mean = np.mean(np.std(static_acc, axis=0))
    if acc_std_mean > acc_threshold:
        print(f"[ZUPT] 警告: 静止期加速度波动较大 (std={acc_std_mean:.3f})")
    bias = np.mean(static_acc, axis=0).astype(np.float32)
    print(f"[ZUPT] 静止期帧范围: [{static_start}, {static_end}], 偏置: {bias}")
    return bias, True


# ===================== 特征构建 =====================
def _slice_with_offset(array: np.ndarray, start: int, end: int) -> np.ndarray:
    start, end = max(0, min(start, array.shape[0])), max(start, min(end, array.shape[0]))
    return array[start:end]


def _process_human_sensors(sensors: Dict[str, SensorSequence], start_idx: int, end_idx: int) -> Dict[str, Dict[str, np.ndarray]]:
    processed = {}
    for name, seq in sensors.items():
        acc_slice = _slice_with_offset(seq.acc, start_idx, end_idx)
        smoothed = smooth_imu_acceleration(acc_slice, smooth_n=4, frame_rate=TARGET_FPS)
        rot_slice = _slice_with_offset(seq.rot, start_idx, end_idx)[:acc_slice.shape[0]]
        processed[name] = {"acc": np.asarray(smoothed[:acc_slice.shape[0]]), "ori": rot_slice}
    return processed


def _build_human_features(sensor_order: List[str], processed: Dict[str, Dict[str, np.ndarray]]) -> torch.Tensor:
    acc_stack = np.stack([processed[name]["acc"] for name in sensor_order], axis=1)
    ori_stack = np.stack([processed[name]["ori"] for name in sensor_order], axis=1)
    acc_tensor = torch.tensor(acc_stack, dtype=torch.float32)
    ori_tensor = torch.tensor(ori_stack, dtype=torch.float32)

    root_R = ori_tensor[:, 0]
    acc_rel = torch.cat((acc_tensor[:, :1], acc_tensor[:, 1:] - acc_tensor[:, :1]), dim=1)
    imu_acc = torch.matmul(acc_rel, root_R)
    root_inv = root_R.transpose(1, 2).unsqueeze(1)
    rel_rot = torch.cat((ori_tensor[:, :1], torch.matmul(root_inv, ori_tensor[:, 1:])), dim=1)
    imu_ori_6d = transforms.matrix_to_rotation_6d(rel_rot.reshape(-1, 3, 3)).reshape(rel_rot.shape[0], rel_rot.shape[1], 6)
    return torch.cat((imu_acc, imu_ori_6d), dim=-1)


def _build_object_features(acc: np.ndarray, quat: np.ndarray, obj_start: int, obj_end: int, seq_len: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    acc_slice = _slice_with_offset(acc, obj_start, obj_end)[:seq_len]
    quat_slice = _slice_with_offset(quat, obj_start, obj_end)[:seq_len]
    acc_smpl, ori_smpl = apply_coordinate_transform(acc_slice, quat_slice)
    acc_smoothed = smooth_imu_acceleration(acc_smpl, smooth_n=4, frame_rate=TARGET_FPS)

    if len(ori_smpl) == 0:
        return torch.empty(0, 9), acc_smoothed, ori_smpl

    rot_tensor = torch.from_numpy(ori_smpl).float()
    R_ref_inv = rot_tensor[0].t()
    rot_norm = torch.matmul(R_ref_inv.unsqueeze(0), rot_tensor)
    acc_tensor = torch.from_numpy(acc_smoothed[:seq_len]).float()
    ori_6d = transforms.matrix_to_rotation_6d(rot_norm).reshape(seq_len, 6)
    return torch.cat((acc_tensor, ori_6d), dim=-1), acc_smoothed[:seq_len], rot_norm.numpy()


def _build_data_dict(human_imu: torch.Tensor, obj_imu: torch.Tensor) -> Dict[str, torch.Tensor]:
    bs, seq_len = 1, human_imu.shape[0]
    identity_6d = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    p_init = identity_6d.view(1, 1, 6).repeat(bs, len(_REDUCED_POSE_NAMES), 1)
    return {
        "human_imu": human_imu.unsqueeze(0),
        "obj_imu": obj_imu.unsqueeze(0),
        "v_init": torch.zeros(bs, len(_SENSOR_VEL_NAMES), 3),
        "p_init": p_init,
        "trans_init": torch.tensor([[0.0, 1.2, 0.0]]).repeat(bs, 1),
        "obj_trans_init": torch.tensor([[0.2, 0.94, 0.35]]).repeat(bs, 1),
        "obj_vel_init": torch.zeros(bs, 3),
        "hand_vel_glb_init": torch.zeros(bs, 2, 3),
        "contact_init": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "has_object": torch.ones(bs, dtype=torch.bool),
        "use_object_data": True,
    }


# ===================== 可视化 =====================
def create_demo_overview_visualization(save_path: str, hip_acc_raw: np.ndarray, obj_acc_trans: np.ndarray,
                                        hip_meta: SensorMeta, obj_meta: SensorMeta) -> None:
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        hip_y = np.abs(hip_acc_raw[:, 1])
        time_h = np.arange(len(hip_y)) / TARGET_FPS
        axes[0].plot(time_h, hip_y, label="Hips Acc Y (abs)", color='blue', alpha=0.7)
        if hip_meta.start_jumps:
            valid = [j for j in hip_meta.start_jumps if j < len(hip_y)]
            axes[0].scatter(np.array(valid)/TARGET_FPS, hip_y[valid], color='green', marker='v', s=60, label='Start Jumps', zorder=5)
        if hip_meta.end_jumps:
            valid = [j for j in hip_meta.end_jumps if j < len(hip_y)]
            axes[0].scatter(np.array(valid)/TARGET_FPS, hip_y[valid], color='red', marker='^', s=60, label='End Jumps', zorder=5)
        axes[0].axvline(hip_meta.start_idx/TARGET_FPS, color='orange', linestyle='--', linewidth=2, label='Start Cut')
        axes[0].axvline(hip_meta.end_idx/TARGET_FPS, color='purple', linestyle='--', linewidth=2, label='End Cut')
        axes[0].set_title("Noitom Hips Detection (Y-axis)")
        axes[0].set_ylabel("Acc (m/s^2)")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        obj_y = np.abs(obj_acc_trans[:, 1])
        time_o = np.arange(len(obj_y)) / TARGET_FPS
        axes[1].plot(time_o, obj_y, label="Object Acc Y (abs)", color='tab:orange', alpha=0.7)
        if obj_meta.start_jumps:
            valid = [j for j in obj_meta.start_jumps if j < len(obj_y)]
            axes[1].scatter(np.array(valid)/TARGET_FPS, obj_y[valid], color='green', marker='v', s=60, label='Start Jumps', zorder=5)
        if obj_meta.end_jumps:
            valid = [j for j in obj_meta.end_jumps if j < len(obj_y)]
            axes[1].scatter(np.array(valid)/TARGET_FPS, obj_y[valid], color='red', marker='^', s=60, label='End Jumps', zorder=5)
        axes[1].axvline(obj_meta.start_idx/TARGET_FPS, color='orange', linestyle='--', linewidth=2, label='Start Cut')
        axes[1].axvline(obj_meta.end_idx/TARGET_FPS, color='purple', linestyle='--', linewidth=2, label='End Cut')
        axes[1].set_title("STAG Object Data (Y-axis)")
        axes[1].set_ylabel("Acc (m/s^2)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Process Demo Detection Overview", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"可视化已保存到: {save_path}")
    except Exception as exc:
        print(f"Demo 可视化失败: {exc}")


# ===================== 主流程 =====================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 IMUHOI StageNet demo data_dict")
    parser.add_argument("--human-csv", default="noitom/6IMU/Output/1204_01_.csv", help="Noitom 人体 IMU CSV")
    parser.add_argument("--obj-csv", default="noitom/OBJ/STAG_VQF_DATA_20251204_182847.csv", help="STAG 物体 IMU CSV")
    parser.add_argument("--output", default="noitom/demo_data_dict.pt", help="data_dict 输出路径")
    parser.add_argument("--device", default="cpu", help="BodyModel FK 所用设备")
    parser.add_argument("--visualize", default=True, help="生成可视化")
    parser.add_argument("--human-threshold", type=float, default=10, help="Noitom 跳跃检测阈值")
    parser.add_argument("--obj-threshold", type=float, default=10, help="STAG 跳跃检测阈值")
    return parser.parse_args()


def main():
    args = parse_args()
    sensor_order = list(DEFAULT_SENSOR_ORDER)

    print("=== 处理 Noitom 人体 IMU 数据 ===")
    sensors = _parse_noitom_sensors(args.human_csv, sensor_order, GRAVITY_ACC)
    hip_seq = sensors["Hips"]
    hip_meta = _detect_range_from_hips(hip_seq, args.human_threshold)

    if not hip_meta.start_jumps:
        raise ValueError("Noitom Hips 未检测到跳跃")
    human_jump_idx = hip_meta.start_jumps[0]
    print(f"[Noitom] 使用第一个跳跃点 {human_jump_idx} 对齐")

    print("=== 处理 STAG 物体 IMU 数据 ===")
    obj_acc, obj_quat = _load_object_data(args.obj_csv)
    obj_acc_mag = np.linalg.norm(obj_acc, axis=1, keepdims=True)
    start_seq_obj, end_seq_obj = detect_jump_sequences(obj_acc_mag, magnitude_threshold=args.obj_threshold,
                                                        min_peak_distance=5, min_jump_interval=10, max_jump_interval=30)
    if len(start_seq_obj) == 0:
        raise ValueError("STAG Object 未检测到跳跃")
    obj_jump_idx = int(start_seq_obj[0])
    print(f"[STAG] 使用第一个跳跃点 {obj_jump_idx} 对齐")

    # ZUPT 偏置补偿
    OFFSET_SEC, STATIC_START_SEC, STATIC_END_SEC = 3.0, 5, 8
    offset_frames = int(OFFSET_SEC * TARGET_FPS)
    static_start = obj_jump_idx + int(STATIC_START_SEC * TARGET_FPS)
    static_end = obj_jump_idx + int(STATIC_END_SEC * TARGET_FPS)
    print("=== ZUPT 偏置补偿 ===")
    obj_bias, bias_valid = estimate_static_bias(obj_acc, static_start, static_end)
    if bias_valid:
        obj_acc = (obj_acc - obj_bias).astype(np.float32)

    # 计算对齐后的序列范围
    human_start, obj_start = human_jump_idx + offset_frames, obj_jump_idx + offset_frames
    human_remain, obj_remain = len(hip_seq.acc) - human_start, len(obj_acc) - obj_start
    if human_remain <= 0 or obj_remain <= 0:
        raise ValueError(f"跳跃后3秒超出数据范围")
    final_len = min(human_remain, obj_remain)
    human_end, obj_end = human_start + final_len, obj_start + final_len

    print(f"=== 对齐结果 ===")
    print(f"Human: Start={human_start}, End={human_end}, Final Len={final_len} ({final_len/TARGET_FPS:.2f}s)")
    print(f"Object: Start={obj_start}, End={obj_end}")

    processed_humans = _process_human_sensors(sensors, human_start, human_end)
    human_imu = _build_human_features(sensor_order, processed_humans)
    obj_imu, obj_acc_smpl, obj_ori_norm = _build_object_features(obj_acc, obj_quat, obj_start, obj_end, final_len)

    data_dict = _build_data_dict(human_imu, obj_imu)
    torch.save(data_dict, args.output)
    print(f"demo data_dict 已保存至: {args.output}")

    if args.visualize:
        viz_dir = "noitom/"
        os.makedirs(viz_dir, exist_ok=True)
        hip_meta.start_idx, hip_meta.end_idx = human_start, human_end
        obj_meta = SensorMeta(start_idx=obj_start, end_idx=obj_end, tpose_start=-1, tpose_end=-1,
                              start_jumps=[int(x) for x in start_seq_obj], end_jumps=[int(x) for x in end_seq_obj],
                              tpose_original=(-1, -1))
        obj_trans_full, _ = apply_coordinate_transform(obj_acc, obj_quat)
        create_demo_overview_visualization(os.path.join(viz_dir, "process_demo_overview.png"),
                                           hip_seq.acc, obj_trans_full, hip_meta, obj_meta)


if __name__ == "__main__":
    main()
