#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo 数据处理脚本：整合 Noitom（人体）与 Noitom（物体，LeftFoot） IMU，构建 IMUHOI StageNet 所需 data_dict
直接使用给定的截取起始时间，不进行跳跃检测对齐。
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

# 确保可以导入项目模块
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from configs import _SENSOR_VEL_NAMES, _REDUCED_POSE_NAMES

# ===================== 常量配置 =====================
DEFAULT_SENSOR_ORDER = ["Hips", "LeftLeg", "RightLeg", "Head", "LeftForeArm", "RightForeArm"]
SENSOR_JOINT_MAP = {
    "Hips": "Hips", "LeftLeg": "LeftFoot", "RightLeg": "RightFoot",
    "Head": "Head", "LeftForeArm": "LeftHand", "RightForeArm": "RightHand",
    "LeftFoot": "LeftFoot"
}
TARGET_FPS = 30
NOITOM_SRC_FPS = 96
GRAVITY_ACC = 9.81


# ===================== 数据结构 =====================
@dataclass
class SensorSequence:
    name: str
    acc: np.ndarray   # [N, 3]
    quat: np.ndarray  # [N, 4]
    rot: np.ndarray   # [N, 3, 3]


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
        acc_flat = acc_tensor.reshape(acc_tensor.shape[0], -1)
    else:
        acc_flat = acc_tensor
    T = acc_flat.shape[0]
    smoothed_acc = acc_flat.clone()

    if smooth_n // 2 != 0 and T > smooth_n * 2:
        for i in range(smooth_n, T - smooth_n):
            weights = torch.ones(2 * smooth_n + 1, device=acc_flat.device) / (2 * smooth_n + 1)
            window_data = acc_flat[i - smooth_n:i + smooth_n + 1]
            smoothed_acc[i] = torch.sum(window_data * weights.unsqueeze(-1), dim=0)

    if T > 2:
        smoothed_acc[0] = (2 * acc_flat[0] + acc_flat[1]) / 3
        smoothed_acc[-1] = (2 * acc_flat[-1] + acc_flat[-2]) / 3

    smoothed_acc = smoothed_acc.reshape(orig_shape)
    return smoothed_acc.numpy() if return_numpy else smoothed_acc


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


# ===================== Noitom CSV 解析 =====================
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
    """解析 Bone-Quat 和 Joint-Posi 字段索引"""
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


def _resolve_sensor_indices(header: List[str], sensor_name: str) -> Dict[str, int]:
    """解析 Sensor-Acce/Quat 字段索引"""
    wanted = {
        "acc_x": f"{sensor_name}-Sensor-Acce-x", "acc_y": f"{sensor_name}-Sensor-Acce-y",
        "acc_z": f"{sensor_name}-Sensor-Acce-z",
        "quat_x": f"{sensor_name}-Sensor-Quat-x", "quat_y": f"{sensor_name}-Sensor-Quat-y",
        "quat_z": f"{sensor_name}-Sensor-Quat-z", "quat_w": f"{sensor_name}-Sensor-Quat-w",
    }
    indices = {}
    for idx, name in enumerate(header):
        for key, pattern in wanted.items():
            if pattern == name or pattern in name:
                indices[key] = idx
    missing = [k for k in wanted if k not in indices]
    if missing:
        raise ValueError(f"传感器 {sensor_name} 缺少 Sensor 字段: {', '.join(missing)}")
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
        raise ValueError("无法从 CSV 中解析有效的 Bone/Joint 数据")
    return np.asarray(pos_list, dtype=np.float32), np.asarray(quat_list, dtype=np.float32)


def _read_sensor_series(rows: List[List[str]], indices: Dict[str, int], gravity_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    acc_list, quat_list = [], []
    for line in rows:
        try:
            acc_list.append([float(line[indices[k]]) * gravity_scale for k in ["acc_x", "acc_y", "acc_z"]])
            quat_list.append([float(line[indices[k]]) for k in ["quat_w", "quat_x", "quat_y", "quat_z"]])
        except (ValueError, IndexError):
            continue
    if not acc_list:
        raise ValueError("无法从 CSV 中解析有效的 Sensor 数据")
    return np.asarray(acc_list, dtype=np.float32), np.asarray(quat_list, dtype=np.float32)


def _compute_acc_from_pos(pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.gradient(pos, axis=0) / dt
    return np.gradient(vel, axis=0) / dt


# ===================== Noitom 人体数据处理 =====================
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


# ===================== Noitom 物体数据处理 =====================
def _apply_local_coordinate_transform(acc: np.ndarray, quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """应用局部坐标系调整，使传感器 Z 轴对齐到 Y 轴"""
    rot = transforms.quaternion_to_matrix(torch.from_numpy(quat)).numpy()
    R_fix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    rot_new = np.einsum("tij,jk->tik", rot, R_fix)
    quat_new = transforms.matrix_to_quaternion(torch.from_numpy(rot_new)).numpy()
    return acc, quat_new


def _load_noitom_object(csv_path: str, sensor_name: str, gravity_scale: float) -> SensorSequence:
    """加载 Noitom 物体 IMU 数据 (Sensor 模式)"""
    header, rows = _load_raw_csv(csv_path)
    indices = _resolve_sensor_indices(header, sensor_name)
    acc_np, quat_np = _read_sensor_series(rows, indices, gravity_scale)
    
    print("[Object] Applying Local Z-to-Y coordinate transform...")
    acc_np, quat_np = _apply_local_coordinate_transform(acc_np, quat_np)
    acc_np = smooth_imu_acceleration(acc_np, smooth_n=4, frame_rate=NOITOM_SRC_FPS)
    
    acc_ds, quat_ds = _resample_sensor(acc_np, quat_np, NOITOM_SRC_FPS)
    rot_ds = transforms.quaternion_to_matrix(torch.from_numpy(quat_ds)).numpy()
    return SensorSequence(name=sensor_name, acc=acc_ds, quat=quat_ds, rot=rot_ds)


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


def _build_object_features(acc: np.ndarray, rot: np.ndarray, obj_start: int, obj_end: int, seq_len: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """构建物体 IMU 特征"""
    acc_slice = _slice_with_offset(acc, obj_start, obj_end)[:seq_len]
    rot_slice = _slice_with_offset(rot, obj_start, obj_end)[:seq_len]
    
    if len(rot_slice) == 0:
        return torch.empty(0, 9), acc_slice, rot_slice

    rot_tensor = torch.from_numpy(rot_slice).float()
    R_ref_inv = rot_tensor[0].t()
    rot_norm = torch.matmul(R_ref_inv.unsqueeze(0), rot_tensor)
    
    acc_tensor = torch.from_numpy(acc_slice).float()
    ori_6d = transforms.matrix_to_rotation_6d(rot_norm).reshape(seq_len, 6)
    return torch.cat((acc_tensor, ori_6d), dim=-1), acc_slice, rot_norm.numpy()


def _build_data_dict(human_imu: torch.Tensor, obj_imu: torch.Tensor) -> Dict[str, torch.Tensor]:
    bs, seq_len = 1, human_imu.shape[0]
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    p_init = identity_6d.view(1, 1, 6).repeat(bs, len(_REDUCED_POSE_NAMES), 1)
    return {
        "human_imu": human_imu.unsqueeze(0),
        "obj_imu": obj_imu.unsqueeze(0),
        "v_init": torch.zeros(bs, len(_SENSOR_VEL_NAMES), 3),
        "p_init": p_init,
        "trans_init": torch.tensor([[0.0, 1.2, 0.0]]).repeat(bs, 1),
        "obj_trans_init": torch.tensor([[0.15, 0.94, 0.25]]).repeat(bs, 1),
        "obj_vel_init": torch.zeros(bs, 3),
        "hand_vel_glb_init": torch.zeros(bs, 2, 3),
        "contact_init": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "has_object": torch.ones(bs, dtype=torch.bool),
        "use_object_data": True,
    }


# ===================== 可视化 =====================
def create_demo_overview_visualization(save_path: str, hip_acc_raw: np.ndarray, obj_acc_raw: np.ndarray,
                                        start_idx: int, end_idx: int) -> None:
    """绘制 Human 和 Object 数据及截取范围"""
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Human (Hips) - Y axis
        hip_y = np.abs(hip_acc_raw[:, 1])
        time_h = np.arange(len(hip_y)) / TARGET_FPS
        axes[0].plot(time_h, hip_y, label="Hips Acc Y (abs)", color='blue', alpha=0.7)
        axes[0].axvline(start_idx/TARGET_FPS, color='orange', linestyle='--', linewidth=2, label='Start')
        axes[0].axvline(end_idx/TARGET_FPS, color='purple', linestyle='--', linewidth=2, label='End')
        axes[0].set_title("Noitom Hips (Y-axis)")
        axes[0].set_ylabel("Acc (m/s^2)")
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Object - Magnitude
        obj_mag = np.linalg.norm(obj_acc_raw, axis=1)
        time_o = np.arange(len(obj_mag)) / TARGET_FPS
        axes[1].plot(time_o, obj_mag, label="Object Acc Mag", color='tab:orange', alpha=0.7)
        axes[1].axvline(start_idx/TARGET_FPS, color='orange', linestyle='--', linewidth=2, label='Start')
        axes[1].axvline(end_idx/TARGET_FPS, color='purple', linestyle='--', linewidth=2, label='End')
        axes[1].set_title("Object (Noitom LeftFoot) Magnitude")
        axes[1].set_ylabel("Acc (m/s^2)")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle("Process Demo Overview (Noitom)", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"可视化已保存到: {save_path}")
    except Exception as exc:
        print(f"Demo 可视化失败: {exc}")


# ===================== 主流程 =====================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 IMUHOI StageNet demo data_dict (Noitom Only)")
    parser.add_argument("--human-csv", default="noitom/6IMU/Output/take005_HUMAN.csv", help="Noitom 人体 IMU CSV")
    parser.add_argument("--obj-csv", default="noitom/6IMU/Output/take005_OBJ.csv", help="Noitom 物体 IMU CSV (LeftFoot)")
    parser.add_argument("--output", default="noitom/demo_data_dict.pt", help="data_dict 输出路径")
    parser.add_argument("--start-sec", type=float, default=8.0, help="截取起始时间 (秒)")
    parser.add_argument("--visualize", default=True, help="生成可视化")
    return parser.parse_args()


def main():
    args = parse_args()
    sensor_order = list(DEFAULT_SENSOR_ORDER)

    print("=== 处理 Noitom 人体 IMU 数据 ===")
    sensors = _parse_noitom_sensors(args.human_csv, sensor_order, GRAVITY_ACC)
    hip_seq = sensors["Hips"]

    print("=== 处理 Noitom 物体 IMU 数据 (LeftFoot) ===")
    obj_sensor = _load_noitom_object(args.obj_csv, "LeftFoot", GRAVITY_ACC)
    obj_acc, obj_rot = obj_sensor.acc, obj_sensor.rot

    # 直接使用固定起始时间截取
    start_frame = int(args.start_sec * TARGET_FPS)
    human_len, obj_len = len(hip_seq.acc), len(obj_acc)
    
    if start_frame >= human_len or start_frame >= obj_len:
        raise ValueError(f"起始时间 {args.start_sec}s 超出数据范围 (Human: {human_len/TARGET_FPS:.1f}s, Obj: {obj_len/TARGET_FPS:.1f}s)")
    
    final_len = min(human_len - start_frame, obj_len - start_frame)
    end_frame = start_frame + final_len

    print(f"=== 截取参数 ===")
    print(f"Start: {args.start_sec}s (frame {start_frame})")
    print(f"Human: {human_len} frames ({human_len/TARGET_FPS:.1f}s)")
    print(f"Object: {obj_len} frames ({obj_len/TARGET_FPS:.1f}s)")
    print(f"Final Length: {final_len} frames ({final_len/TARGET_FPS:.2f}s)")

    processed_humans = _process_human_sensors(sensors, start_frame, end_frame)
    human_imu = _build_human_features(sensor_order, processed_humans)
    obj_imu, obj_acc_trim, obj_ori_norm = _build_object_features(obj_acc, obj_rot, start_frame, end_frame, final_len)

    data_dict = _build_data_dict(human_imu, obj_imu)
    torch.save(data_dict, args.output)
    print(f"demo data_dict 已保存至: {args.output}")

    if args.visualize:
        viz_dir = "noitom/"
        os.makedirs(viz_dir, exist_ok=True)
        create_demo_overview_visualization(os.path.join(viz_dir, "process_demo_overview.png"),
                                           hip_seq.acc, obj_acc, start_frame, end_frame)


if __name__ == "__main__":
    main()
