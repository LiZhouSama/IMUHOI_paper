#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Badminton demo for IMUHOI.

This script converts a badminton Noitom Calc CSV plus a synced racket IMU CSV
into the current IMUHOI model input, runs inference with the same broad calling
style as eval_IMUHOI.py, and visualizes GT human, predicted human, and predicted
racket/object in one aitviewer scene.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from scipy.spatial.transform import Rotation, Slerp


PROJECT_ROOT = Path(__file__).resolve().parent
BADMINTON_ROOT_DEFAULT = Path("/mnt/d/a_WORK/Projects/PhD/tasks/badminton")
DEFAULT_NOITOM_CSV = BADMINTON_ROOT_DEFAULT / "noitom_badmin/output_noitom_csv/take006_chr02.csv"
DEFAULT_SYNC_CSV = BADMINTON_ROOT_DEFAULT / "output/20260428_001800/sync.csv"
DEFAULT_RACKET_OBJ = BADMINTON_ROOT_DEFAULT / "obj/badminton_racket/Racket.obj"
DEFAULT_BODY_MODEL = Path("/mnt/d/a_WORK/Projects/PhD/datasets/smpl_models/smplh/neutral/model.npz")

NOITOM_SRC_FPS = 60.0
GRAVITY_MPS2 = 9.80665
HUMAN_SENSOR_ORDER = ["Hips", "LeftLeg", "RightLeg", "Head", "LeftForeArm", "RightForeArm"]
SENSOR_JOINT_MAP = {
    "Hips": "Hips",
    "LeftLeg": "LeftFoot",
    "RightLeg": "RightFoot",
    "Head": "Head",
    "LeftForeArm": "LeftHand",
    "RightForeArm": "RightHand",
}


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_pytorch3d_fallback() -> None:
    try:
        import pytorch3d.transforms  # noqa: F401
        return
    except Exception:
        pass

    from utils import rotation_conversions as _rot

    pytorch3d_mod = types.ModuleType("pytorch3d")
    transforms_mod = types.ModuleType("pytorch3d.transforms")
    transforms_mod.rotation_6d_to_matrix = _rot.rotation_6d_to_matrix
    transforms_mod.matrix_to_rotation_6d = _rot.matrix_to_rotation_6d
    transforms_mod.matrix_to_axis_angle = _rot.matrix_to_axis_angle
    pytorch3d_mod.transforms = transforms_mod
    sys.modules["pytorch3d"] = pytorch3d_mod
    sys.modules["pytorch3d.transforms"] = transforms_mod


_install_pytorch3d_fallback()

import pytorch3d.transforms as t3d  # noqa: E402

from configs import (  # noqa: E402
    FRAME_RATE,
    _IGNORED_INDICES,
    _REDUCED_INDICES,
    _REDUCED_POSE_NAMES,
    _SENSOR_ROT_INDICES,
)
from model import load_model  # noqa: E402
from utils.utils import build_model_input_dict, load_config  # noqa: E402


@dataclass
class SyncDemoSequence:
    imu_host_ts: np.ndarray
    acc_g: np.ndarray
    rotmats: np.ndarray
    pressure: np.ndarray


@dataclass
class DemoPreparedData:
    batch: Dict[str, torch.Tensor]
    data_dict: Dict[str, torch.Tensor]
    gt_targets: Dict[str, torch.Tensor]
    frame_no: np.ndarray
    gt_vertices: np.ndarray
    gt_faces: np.ndarray
    gt_smpl24: np.ndarray
    obj_rotmats_norm: np.ndarray
    contact_state: np.ndarray
    sync_indices: np.ndarray


@dataclass
class SimpleMesh:
    vertices: np.ndarray
    faces: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IMUHOI on badminton Noitom + sync data.")
    parser.add_argument("--badminton-root", type=Path, default=BADMINTON_ROOT_DEFAULT)
    parser.add_argument("--noitom-csv", type=Path, default=DEFAULT_NOITOM_CSV)
    parser.add_argument("--sync-path", type=Path, default=DEFAULT_SYNC_CSV)
    parser.add_argument("--config", type=Path, default=Path("configs/IMUHOI_train_rnn.yaml"))
    parser.add_argument("--body-model", type=Path, default=DEFAULT_BODY_MODEL)
    parser.add_argument("--betas-npz", type=Path, default=None)
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--smpl-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-arch", choices=["rnn", "dit", "mamba", "mamba_simple"], default=None)
    parser.add_argument("--no-trans", action="store_true")
    parser.add_argument("--noitom-fps", type=float, default=NOITOM_SRC_FPS)
    parser.add_argument("--target-fps", type=float, default=float(FRAME_RATE))
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum target/model frames to run.")
    parser.add_argument(
        "--sync-start-frame",
        type=int,
        default=0,
        help="Noitom Frame-No corresponding to the first sync frame. Formula: noitom_time=(Frame-No-sync_start_frame)/noitom_fps.",
    )

    parser.add_argument("--hp-ckpt", default=None)
    parser.add_argument("--velocity-ckpt", default=None)
    parser.add_argument("--object-ckpt", default=None)
    parser.add_argument("--interaction-ckpt", default=None)

    parser.add_argument("--obj-init-source", choices=["right_wrist", "right_hand", "pelvis", "manual"], default="right_wrist")
    parser.add_argument("--obj-trans-init", type=float, nargs=3, default=None)
    parser.add_argument("--obj-points-count", type=int, default=256)
    parser.add_argument("--remove-object-gravity", dest="remove_object_gravity", action="store_true", default=True)
    parser.add_argument("--keep-object-gravity", dest="remove_object_gravity", action="store_false")
    parser.add_argument(
        "--object-gravity-calib-frames",
        type=int,
        default=60,
        help="Number of initial sync IMU frames used to estimate the object gravity vector in g.",
    )
    parser.add_argument(
        "--object-gravity-vector-g",
        type=float,
        nargs=3,
        default=None,
        help="Manual object gravity vector in sync.csv coordinates, in g. Overrides initial-frame estimation.",
    )

    parser.add_argument("--racket-obj", type=Path, default=DEFAULT_RACKET_OBJ)
    parser.add_argument("--racket-face-budget", type=int, default=8000)
    parser.add_argument("--racket-align-roll", type=float, default=0.0)
    parser.add_argument("--racket-align-pitch", type=float, default=0.0)
    parser.add_argument("--racket-align-yaw", type=float, default=0.0)

    parser.add_argument("--contact-smooth-window", type=int, default=3)
    parser.add_argument("--contact-enter-sum", type=float, default=250.0)
    parser.add_argument("--contact-exit-sum", type=float, default=150.0)
    parser.add_argument("--contact-enter-max", type=float, default=60.0)
    parser.add_argument("--contact-exit-max", type=float, default=25.0)
    parser.add_argument("--contact-enter-frames", type=int, default=2)
    parser.add_argument("--contact-exit-frames", type=int, default=3)
    parser.add_argument(
        "--use-pressure-contact",
        action="store_true",
        help="Feed pressure-derived contact labels to model inputs/gt_targets. Disabled by default so VC infers contact itself.",
    )

    parser.add_argument("--compute-fk", dest="compute_fk", action="store_true", default=True)
    parser.add_argument("--no-compute-fk", dest="compute_fk", action="store_false")
    parser.add_argument("--use-object-data", dest="use_object_data", action="store_true", default=True)
    parser.add_argument("--no-object-data", dest="use_object_data", action="store_false")
    parser.add_argument("--interaction-human-source", choices=["pred", "gt"], default="pred")
    parser.add_argument("--interaction-human-trans-source", choices=["pred", "gt"], default="pred")
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--sampler", default=None)
    parser.add_argument("--eta", type=float, default=None)

    parser.add_argument("--save-pred", type=Path, default=None)
    parser.add_argument("--save-data-dict", type=Path, default=None)
    parser.add_argument("--no-viewer", action="store_true")
    return parser.parse_args()


def resolve_path(path: Optional[Path], base: Path = PROJECT_ROOT) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[demo_badmin] CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def load_fit_module(badminton_root: Path):
    module_path = badminton_root / "fit_noitom_smplh.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Cannot find badminton fit script: {module_path}")
    spec = importlib.util.spec_from_file_location("badminton_fit_noitom_smplh", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_betas(betas_npz: Optional[Path], num_betas: int, fit_module) -> np.ndarray:
    if betas_npz is None:
        return np.zeros((num_betas,), dtype=np.float32)
    return fit_module.load_betas(betas_npz, num_betas)


def smooth_imu_acceleration(acc_data: np.ndarray, smooth_n: int = 4) -> np.ndarray:
    acc = np.asarray(acc_data, dtype=np.float32)
    out = acc.copy()
    if out.shape[0] > smooth_n * 2:
        weights = np.full((2 * smooth_n + 1,), 1.0 / float(2 * smooth_n + 1), dtype=np.float32)
        flat = acc.reshape(acc.shape[0], -1)
        smoothed = flat.copy()
        for idx in range(smooth_n, flat.shape[0] - smooth_n):
            smoothed[idx] = np.sum(flat[idx - smooth_n : idx + smooth_n + 1] * weights[:, None], axis=0)
        out = smoothed.reshape(acc.shape)
    if out.shape[0] > 2:
        out[0] = (2.0 * acc[0] + acc[1]) / 3.0
        out[-1] = (2.0 * acc[-1] + acc[-2]) / 3.0
    return out.astype(np.float32, copy=False)


def compute_acc_from_pos(pos: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / float(fps)
    vel = np.gradient(np.asarray(pos, dtype=np.float32), axis=0) / dt
    acc = np.gradient(vel, axis=0) / dt
    return acc.astype(np.float32, copy=False)


def resample_linear_to_length(data: np.ndarray, target_len: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.shape[0] == target_len:
        return data.copy()
    if data.shape[0] == 0:
        return np.zeros((target_len, *data.shape[1:]), dtype=np.float32)
    if data.shape[0] == 1:
        return np.repeat(data, target_len, axis=0).astype(np.float32, copy=False)

    old_shape = data.shape
    flat = data.reshape(old_shape[0], -1)
    t_old = np.linspace(0.0, 1.0, old_shape[0], dtype=np.float64)
    t_new = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float32)
    for idx in range(flat.shape[1]):
        out[:, idx] = np.interp(t_new, t_old, flat[:, idx]).astype(np.float32)
    return out.reshape((target_len, *old_shape[1:]))


def resample_rotmats_to_length(rotmats: np.ndarray, target_len: int) -> np.ndarray:
    rotmats = np.asarray(rotmats, dtype=np.float32)
    if rotmats.shape[0] == target_len:
        return rotmats.copy()
    if rotmats.shape[0] == 0:
        eye = np.eye(3, dtype=np.float32)
        return np.repeat(eye[None], target_len, axis=0)
    if rotmats.shape[0] == 1:
        return np.repeat(rotmats[:1], target_len, axis=0).astype(np.float32, copy=False)

    t_old = np.linspace(0.0, 1.0, rotmats.shape[0], dtype=np.float64)
    t_new = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    return Slerp(t_old, Rotation.from_matrix(rotmats))(t_new).as_matrix().astype(np.float32)


def resample_global_rotmats_to_length(global_rotmats: np.ndarray, target_len: int) -> np.ndarray:
    if global_rotmats.shape[0] == target_len:
        return global_rotmats.astype(np.float32, copy=True)
    joints = global_rotmats.shape[1]
    out = np.empty((target_len, joints, 3, 3), dtype=np.float32)
    for joint_idx in range(joints):
        out[:, joint_idx] = resample_rotmats_to_length(global_rotmats[:, joint_idx], target_len)
    return out


def build_human_imu_from_noitom(
    positions: Dict[str, np.ndarray],
    bone_quats: Dict[str, np.ndarray],
    target_len: int,
    source_fps: float,
) -> torch.Tensor:
    acc_list = []
    rot_list = []
    for sensor_name in HUMAN_SENSOR_ORDER:
        joint_name = SENSOR_JOINT_MAP.get(sensor_name, sensor_name)
        if joint_name not in positions:
            raise KeyError(f"Noitom CSV missing joint position for {joint_name}")
        if sensor_name not in bone_quats:
            raise KeyError(f"Noitom CSV missing bone quaternion for {sensor_name}")

        acc_src = smooth_imu_acceleration(compute_acc_from_pos(positions[joint_name], source_fps), smooth_n=4)
        rot_src = Rotation.from_quat(bone_quats[sensor_name]).as_matrix().astype(np.float32)
        acc_list.append(resample_linear_to_length(acc_src, target_len))
        rot_list.append(resample_rotmats_to_length(rot_src, target_len))

    acc_tensor = torch.tensor(np.stack(acc_list, axis=1), dtype=torch.float32)
    rot_tensor = torch.tensor(np.stack(rot_list, axis=1), dtype=torch.float32)

    root_rot = rot_tensor[:, 0]
    acc_rel = torch.cat((acc_tensor[:, :1], acc_tensor[:, 1:] - acc_tensor[:, :1]), dim=1)
    imu_acc = torch.matmul(acc_rel, root_rot)

    root_inv = root_rot.transpose(1, 2).unsqueeze(1)
    rel_rot = torch.cat((rot_tensor[:, :1], torch.matmul(root_inv, rot_tensor[:, 1:])), dim=1)
    imu_rot6d = t3d.matrix_to_rotation_6d(rel_rot.reshape(-1, 3, 3)).reshape(
        rel_rot.shape[0], rel_rot.shape[1], 6
    )
    return torch.cat((imu_acc, imu_rot6d), dim=-1)


def read_sync_demo_csv(sync_path: Path, fit_module) -> SyncDemoSequence:
    pressure_columns = list(fit_module.PRESSURE_COLUMNS)
    required = {
        "imu_host_ts",
        "imu_ax_g",
        "imu_ay_g",
        "imu_az_g",
        "imu_qw",
        "imu_qx",
        "imu_qy",
        "imu_qz",
        *pressure_columns,
    }
    df = pd.read_csv(sync_path)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {sync_path}: {', '.join(missing)}")

    host_ts = pd.to_numeric(df["imu_host_ts"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(host_ts)
    if not np.any(valid):
        raise ValueError(f"No valid imu_host_ts found in {sync_path}")
    df = df.loc[valid].reset_index(drop=True)
    host_ts = host_ts[valid]

    order = np.argsort(host_ts, kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    host_ts = host_ts[order]

    acc_g = (
        df[["imu_ax_g", "imu_ay_g", "imu_az_g"]]
        .apply(pd.to_numeric, errors="coerce")
        .ffill()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    quats_wxyz = (
        df[["imu_qw", "imu_qx", "imu_qy", "imu_qz"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )
    invalid_quat = ~np.isfinite(quats_wxyz).all(axis=1)
    quats_wxyz[invalid_quat] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    rotmats = np.empty((quats_wxyz.shape[0], 3, 3), dtype=np.float32)
    for idx, quat in enumerate(quats_wxyz):
        rotmats[idx] = fit_module.quat_wxyz_to_rotmat((float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

    pressure = (
        df[pressure_columns]
        .apply(pd.to_numeric, errors="coerce")
        .ffill()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return SyncDemoSequence(
        imu_host_ts=host_ts.astype(np.float64, copy=False),
        acc_g=acc_g,
        rotmats=rotmats,
        pressure=pressure,
    )


def _interp_columns(values: np.ndarray, src_time: np.ndarray, dst_time: np.ndarray) -> np.ndarray:
    out = np.empty((dst_time.shape[0], values.shape[1]), dtype=np.float32)
    for col_idx in range(values.shape[1]):
        out[:, col_idx] = np.interp(
            dst_time,
            src_time,
            values[:, col_idx],
            left=float(values[0, col_idx]),
            right=float(values[-1, col_idx]),
        )
    return out


def remove_object_gravity_from_sync(sync_seq: SyncDemoSequence, args: argparse.Namespace) -> np.ndarray:
    if args.object_gravity_vector_g is not None:
        gravity_g = np.asarray(args.object_gravity_vector_g, dtype=np.float32)
    else:
        calib_frames = max(1, int(args.object_gravity_calib_frames))
        calib = sync_seq.acc_g[: min(calib_frames, sync_seq.acc_g.shape[0])]
        valid = np.isfinite(calib).all(axis=1)
        if not np.any(valid):
            raise ValueError("Cannot estimate object gravity: no finite sync accelerometer frames.")
        gravity_g = calib[valid].mean(axis=0).astype(np.float32)

    sync_seq.acc_g = (sync_seq.acc_g - gravity_g[None, :]).astype(np.float32)
    return gravity_g


def align_sync_to_noitom_model_frames(
    sync_seq: SyncDemoSequence,
    frame_no_model: np.ndarray,
    noitom_fps: float,
    sync_start_frame: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if noitom_fps <= 0.0:
        raise ValueError(f"noitom_fps must be > 0, got {noitom_fps}")
    if sync_seq.imu_host_ts.size == 0:
        raise ValueError("Sync sequence is empty.")

    sync_time = sync_seq.imu_host_ts - float(sync_seq.imu_host_ts[0])
    noitom_time = (frame_no_model.astype(np.float64) - float(sync_start_frame)) / float(noitom_fps)
    sync_indices = np.searchsorted(sync_time, noitom_time, side="right") - 1
    sync_indices = np.clip(sync_indices, 0, sync_time.shape[0] - 1).astype(np.int32, copy=False)

    unique_time, unique_indices = np.unique(sync_time, return_index=True)
    unique_rotmats = sync_seq.rotmats[unique_indices]
    unique_acc = sync_seq.acc_g[unique_indices] * GRAVITY_MPS2
    unique_pressure = sync_seq.pressure[unique_indices]

    if unique_time.shape[0] == 1:
        rotmats = np.repeat(unique_rotmats[:1], frame_no_model.shape[0], axis=0)
        acc_mps2 = np.repeat(unique_acc[:1], frame_no_model.shape[0], axis=0)
        pressure = np.repeat(unique_pressure[:1], frame_no_model.shape[0], axis=0)
        return rotmats, acc_mps2.astype(np.float32), pressure.astype(np.float32), sync_indices

    clipped_time = np.clip(noitom_time, unique_time[0], unique_time[-1])
    rotmats = Slerp(unique_time, Rotation.from_matrix(unique_rotmats))(clipped_time).as_matrix().astype(np.float32)
    acc_mps2 = _interp_columns(unique_acc.astype(np.float32), unique_time, clipped_time)
    pressure = _interp_columns(unique_pressure.astype(np.float32), unique_time, clipped_time)
    return rotmats, acc_mps2, pressure, sync_indices


def _global_to_local_rotmat(global_rotmats: torch.Tensor, parents) -> torch.Tensor:
    seq_len, num_joints = global_rotmats.shape[:2]
    local_rotmats = torch.zeros_like(global_rotmats)
    local_rotmats[:, 0] = global_rotmats[:, 0]
    for joint_idx in range(1, num_joints):
        parent_idx = int(parents[joint_idx]) if joint_idx < len(parents) else 0
        parent_idx = max(0, min(parent_idx, num_joints - 1))
        parent_rot = global_rotmats[:, parent_idx]
        local_rotmats[:, joint_idx] = torch.matmul(parent_rot.transpose(-1, -2), global_rotmats[:, joint_idx])

    valid_ignored = [idx for idx in _IGNORED_INDICES if idx < num_joints]
    if valid_ignored:
        eye = torch.eye(3, device=global_rotmats.device, dtype=global_rotmats.dtype).view(1, 1, 3, 3)
        local_rotmats[:, valid_ignored] = eye.expand(seq_len, len(valid_ignored), 3, 3)
    return local_rotmats


def build_gt_human_sequence(
    args: argparse.Namespace,
    fit_module,
    body_model,
    betas: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_max = None
    if args.max_frames is not None:
        source_max = max(2, int(math.ceil(float(args.max_frames) * float(args.noitom_fps) / float(args.target_fps))))

    frame_no_src, positions_src, bone_quats_src = fit_module.read_noitom_calc(args.noitom_csv)
    frame_no_src, positions_src, bone_quats_src = fit_module.subsample_data(
        frame_no=frame_no_src,
        positions=positions_src,
        bone_quats=bone_quats_src,
        frame_stride=1,
        max_frames=source_max,
    )
    if frame_no_src.shape[0] < 2:
        raise ValueError("Need at least two Noitom frames to build demo input.")

    target_len = max(1, int(round(frame_no_src.shape[0] * float(args.target_fps) / float(args.noitom_fps))))
    if args.max_frames is not None:
        target_len = min(target_len, int(args.max_frames))

    frame_no_model = resample_linear_to_length(frame_no_src.astype(np.float32)[:, None], target_len).reshape(-1)
    global_rotmats_src = fit_module.build_smpl24_global_rotmats(bone_quats_src)
    global_rotmats = resample_global_rotmats_to_length(global_rotmats_src, target_len)
    root_orient, pose_body, _ = fit_module.smpl_pose_from_global_rotmats(global_rotmats)
    hips_pos = resample_linear_to_length(positions_src["Hips"], target_len)

    with torch.no_grad():
        rest_out = body_model(
            pose_body=torch.zeros(1, 63, dtype=torch.float32, device=device),
            pose_hand=torch.zeros(1, 90, dtype=torch.float32, device=device),
            betas=torch.tensor(betas[None], dtype=torch.float32, device=device),
            root_orient=torch.zeros(1, 3, dtype=torch.float32, device=device),
            trans=torch.zeros(1, 3, dtype=torch.float32, device=device),
        )
        rest = rest_out.Jtr[0].detach().cpu().numpy()

    trans = fit_module.compute_trans_from_hips(hips_pos, root_orient, rest[0])
    gt_vertices, gt_joints = fit_module.forward_body_model(
        bm=body_model,
        pose_body=pose_body,
        root_orient=root_orient,
        trans=trans,
        betas=betas,
        device=device,
        batch_size=args.smpl_batch_size,
    )
    gt_smpl24 = gt_joints[:, : len(fit_module.SMPL24_NAMES)]

    human_imu = build_human_imu_from_noitom(
        positions=positions_src,
        bone_quats=bone_quats_src,
        target_len=target_len,
        source_fps=float(args.noitom_fps),
    )

    global_rot_t = torch.tensor(global_rotmats, dtype=torch.float32)
    ori_glb_reduced = global_rot_t[:, _REDUCED_INDICES]
    ori_root_reduced = torch.matmul(global_rot_t[:, :1].transpose(-1, -2), ori_glb_reduced)
    pose = torch.cat(
        [
            torch.tensor(root_orient, dtype=torch.float32),
            torch.tensor(pose_body, dtype=torch.float32),
        ],
        dim=-1,
    )

    batch = {
        "human_imu": human_imu.unsqueeze(0),
        "trans": torch.tensor(trans, dtype=torch.float32).unsqueeze(0),
        "pose": pose.unsqueeze(0),
        "rotation_global": global_rot_t.unsqueeze(0),
        "position_global": torch.tensor(gt_joints, dtype=torch.float32).unsqueeze(0),
        "ori_root_reduced": ori_root_reduced.unsqueeze(0),
        "has_object": torch.ones(1, dtype=torch.bool),
        "imu_noise_applied": torch.ones(1, dtype=torch.bool),
    }
    return batch, frame_no_model, gt_vertices, gt_smpl24, global_rotmats


def load_racket_mesh(args: argparse.Namespace, fit_module) -> Optional[Any]:
    try:
        mesh = fit_module.load_prepare_racket_mesh(
            obj_path=args.racket_obj,
            face_budget=max(100, args.racket_face_budget),
            align_roll=args.racket_align_roll,
            align_pitch=args.racket_align_pitch,
            align_yaw=args.racket_align_yaw,
        )
        print(
            f"[demo_badmin] Loaded racket mesh: {args.racket_obj} "
            f"(verts={mesh.vertices.shape[0]}, faces={mesh.faces.shape[0]})"
        )
        return mesh
    except Exception as exc:
        print(f"[demo_badmin] Failed to load racket mesh, using fallback box: {exc}")
        half = 0.04
        vertices = np.array(
            [
                [half, half, half],
                [half, half, -half],
                [half, -half, half],
                [half, -half, -half],
                [-half, half, half],
                [-half, half, -half],
                [-half, -half, half],
                [-half, -half, -half],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
            ],
            dtype=np.int32,
        )
        return SimpleMesh(vertices=vertices, faces=faces)


def sample_object_points(mesh: Optional[Any], count: int) -> torch.Tensor:
    count = max(1, int(count))
    if mesh is None or getattr(mesh, "vertices", None) is None:
        return torch.zeros(1, count, 3, dtype=torch.float32)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.shape[0] == 0:
        return torch.zeros(1, count, 3, dtype=torch.float32)
    if vertices.shape[0] >= count:
        idx = np.linspace(0, vertices.shape[0] - 1, count).round().astype(np.int64)
        points = vertices[idx]
    else:
        reps = int(math.ceil(count / vertices.shape[0]))
        points = np.tile(vertices, (reps, 1))[:count]
    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)


def resolve_obj_trans_init(args: argparse.Namespace, gt_smpl24: np.ndarray, fit_module) -> torch.Tensor:
    if args.obj_trans_init is not None:
        return torch.tensor(args.obj_trans_init, dtype=torch.float32)
    if args.obj_init_source == "manual":
        raise ValueError("--obj-init-source manual requires --obj-trans-init x y z")

    source_to_idx = {
        "pelvis": 0,
        "right_wrist": fit_module.SMPL24_NAMES.index("right_wrist"),
        "right_hand": fit_module.SMPL24_NAMES.index("right_hand"),
    }
    idx = source_to_idx[args.obj_init_source]
    return torch.tensor(gt_smpl24[0, idx], dtype=torch.float32)


def build_object_inputs(
    args: argparse.Namespace,
    fit_module,
    frame_no_model: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    sync_seq = read_sync_demo_csv(args.sync_path, fit_module)
    rotmats, acc_mps2, pressure, sync_indices = align_sync_to_noitom_model_frames(
        sync_seq=sync_seq,
        frame_no_model=frame_no_model,
        noitom_fps=float(args.noitom_fps),
        sync_start_frame=int(args.sync_start_frame),
    )
    acc_mps2 = smooth_imu_acceleration(acc_mps2, smooth_n=4)

    # sync.csv is written by run_dual_sensor_sage.py with quat_human_wxyz,
    # which already includes the initial racket-to-human orientation.  Keep it
    # absolute here to match fit_noitom_smplh.py visualization semantics.
    rot_tensor = torch.tensor(rotmats, dtype=torch.float32)
    obj_rot6d = t3d.matrix_to_rotation_6d(rot_tensor.reshape(-1, 3, 3)).reshape(rot_tensor.shape[0], 6)
    obj_imu = torch.cat([torch.tensor(acc_mps2, dtype=torch.float32), obj_rot6d], dim=-1)

    contact_state = fit_module.build_filtered_contact_state(
        pressure_frames=pressure,
        smooth_window=args.contact_smooth_window,
        enter_sum=args.contact_enter_sum,
        exit_sum=args.contact_exit_sum,
        enter_max=args.contact_enter_max,
        exit_max=args.contact_exit_max,
        enter_frames=args.contact_enter_frames,
        exit_frames=args.contact_exit_frames,
    )
    contact_tensor = torch.tensor(contact_state, dtype=torch.bool)
    print(
        "[demo_badmin] Sync aligned "
        f"{sync_seq.imu_host_ts.shape[0]} sync frames -> {frame_no_model.shape[0]} model frames; "
        f"sync frame range=[{int(sync_indices.min())}, {int(sync_indices.max())}], "
        f"contact frames={int(contact_state.sum())}"
    )
    return obj_imu, contact_tensor, rot_tensor.detach().cpu().numpy().astype(np.float32), contact_state, sync_indices


def to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def build_prepared_data(
    args: argparse.Namespace,
    fit_module,
    body_model,
    betas: np.ndarray,
    racket_mesh: Optional[Any],
    device: torch.device,
    cfg: edict,
) -> DemoPreparedData:
    batch, frame_no_model, gt_vertices, gt_smpl24, _ = build_gt_human_sequence(
        args=args,
        fit_module=fit_module,
        body_model=body_model,
        betas=betas,
        device=device,
    )
    obj_imu, contact_tensor, obj_rotmats_norm, contact_state, sync_indices = build_object_inputs(
        args=args,
        fit_module=fit_module,
        frame_no_model=frame_no_model,
    )
    obj_init = resolve_obj_trans_init(args, gt_smpl24, fit_module)
    obj_points = sample_object_points(racket_mesh, args.obj_points_count)

    seq_len = batch["human_imu"].shape[1]
    if obj_imu.shape[0] != seq_len:
        min_len = min(seq_len, obj_imu.shape[0])
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[1] == seq_len:
                batch[key] = value[:, :min_len]
        frame_no_model = frame_no_model[:min_len]
        gt_vertices = gt_vertices[:min_len]
        gt_smpl24 = gt_smpl24[:min_len]
        obj_imu = obj_imu[:min_len]
        contact_tensor = contact_tensor[:min_len]
        obj_rotmats_norm = obj_rotmats_norm[:min_len]
        contact_state = contact_state[:min_len]
        sync_indices = sync_indices[:min_len]
        seq_len = min_len

    batch["obj_imu"] = obj_imu.unsqueeze(0)
    batch["obj_rot"] = obj_imu[:, 3:9].unsqueeze(0)
    batch["obj_scale"] = torch.ones(1, seq_len, dtype=torch.float32)
    batch["obj_points_canonical"] = obj_points
    batch["obj_name"] = ["badminton_racket"]
    if args.use_pressure_contact:
        batch["lhand_contact"] = torch.zeros(1, seq_len, dtype=torch.bool)
        batch["rhand_contact"] = contact_tensor.unsqueeze(0)
        batch["obj_contact"] = contact_tensor.unsqueeze(0)

    batch_device = to_device_batch(batch, device)
    data_dict = build_model_input_dict(batch_device, cfg, device, add_noise=False)
    data_dict.pop("obj_trans_gt", None)
    data_dict["obj_trans_init"] = obj_init.to(device=device, dtype=batch_device["human_imu"].dtype).unsqueeze(0)
    data_dict["obj_points_canonical"] = obj_points.to(device=device, dtype=batch_device["human_imu"].dtype)
    data_dict["obj_name"] = ["badminton_racket"]
    data_dict["seq_file"] = [str(args.noitom_csv)]

    gt_target_keys = {
        "pose",
        "trans",
        "position_global",
        "rotation_global",
        "ori_root_reduced",
        "obj_rot",
        "obj_scale",
        "obj_points_canonical",
        "has_object",
    }
    if args.use_pressure_contact:
        gt_target_keys.update({"lhand_contact", "rhand_contact", "obj_contact"})
    gt_targets = {key: value for key, value in batch_device.items() if key in gt_target_keys}
    gt_targets["obj_name"] = ["badminton_racket"]
    gt_targets["seq_file"] = [str(args.noitom_csv)]

    faces = body_model.f.detach().cpu().numpy().astype(np.int32) if torch.is_tensor(body_model.f) else np.asarray(body_model.f, dtype=np.int32)
    print(
        f"[demo_badmin] Prepared model input: human_imu={tuple(data_dict['human_imu'].shape)}, "
        f"obj_imu={tuple(data_dict['obj_imu'].shape)}, obj_init={data_dict['obj_trans_init'][0].detach().cpu().numpy()}"
    )
    return DemoPreparedData(
        batch=batch_device,
        data_dict=data_dict,
        gt_targets=gt_targets,
        frame_no=frame_no_model,
        gt_vertices=gt_vertices.astype(np.float32, copy=False),
        gt_faces=faces,
        gt_smpl24=gt_smpl24.astype(np.float32, copy=False),
        obj_rotmats_norm=obj_rotmats_norm,
        contact_state=contact_state,
        sync_indices=sync_indices,
    )


def configure_model(args: argparse.Namespace, cfg: edict, device: torch.device):
    cfg.device = str(device)
    if device.type == "cuda":
        cfg.gpus = [device.index if device.index is not None else 0]
    else:
        cfg.gpus = []
        cfg.use_multi_gpu = False
    if args.model_arch is not None:
        cfg.model_arch = args.model_arch
    if args.body_model is not None:
        cfg.body_model_path = str(args.body_model)

    module_paths = {}
    if getattr(cfg, "pretrained_modules", None):
        module_paths.update({key: value for key, value in dict(cfg.pretrained_modules).items() if value})

    arch = str(getattr(cfg, "model_arch", "rnn")).lower()
    if args.hp_ckpt:
        module_paths["human_pose"] = args.hp_ckpt
    if args.velocity_ckpt:
        module_paths["velocity_contact"] = args.velocity_ckpt
    if args.object_ckpt:
        module_paths["object_trans"] = args.object_ckpt
    if args.interaction_ckpt:
        if arch in {"dit", "mamba"}:
            module_paths["interaction"] = args.interaction_ckpt
        else:
            module_paths["object_trans"] = args.interaction_ckpt

    module_paths = {key: str(resolve_path(Path(value))) for key, value in module_paths.items() if value}

    cfg_for_model = copy.deepcopy(cfg)
    cfg_for_model.pretrained_modules = {}
    staged_cfg = getattr(cfg_for_model, "staged_training", None)
    if isinstance(staged_cfg, dict):
        staged_cfg.get("modular_training", {}).pop("pretrained_modules", None)

    model = load_model(cfg_for_model, device, no_trans=args.no_trans, module_paths={})
    if module_paths:
        safe_load_pretrained_modules(model, module_paths, device, arch)
    model.eval()
    return model


def safe_load_pretrained_modules(model, module_paths: Dict[str, str], device: torch.device, arch: str) -> None:
    aliases = {
        "human_pose": "human_pose_module",
        "velocity_contact": "velocity_contact_module",
        "object_trans": "object_trans_module",
        "interaction": "interaction_module",
    }
    if arch in {"dit", "mamba"}:
        aliases["velocity_contact"] = "interaction_module"
        aliases["object_trans"] = "interaction_module"
    elif arch in {"rnn", "mamba_simple"}:
        aliases["interaction"] = "object_trans_module"

    print("Loading pretrained modules (shape-filtered for demo compatibility):")
    core = _unwrap_model(model)
    loaded_targets = set()
    for name, path in module_paths.items():
        target = aliases.get(name)
        if target is None:
            print(f"  - skip {name}: unknown module key")
            continue
        if target in loaded_targets and target == "interaction_module":
            print(f"  - skip {name}: {target} already loaded")
            continue
        module = getattr(core, target, None)
        if module is None:
            print(f"  - skip {name}: model has no {target}")
            continue
        if not os.path.exists(path):
            print(f"  - skip {name}: checkpoint not found at {path}")
            continue
        load_checkpoint_shape_filtered(module, path, device, label=name)
        loaded_targets.add(target)


def load_checkpoint_shape_filtered(module, checkpoint_path: str, device: torch.device, label: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if checkpoint.get("ema_state_dict") is not None:
            state_dict = checkpoint["ema_state_dict"]
        else:
            state_dict = checkpoint.get("module_state_dict", checkpoint.get("model_state_dict", checkpoint))

    model_state = module.state_dict()
    filtered = {}
    skipped_shape = 0
    skipped_missing = 0
    for key, value in state_dict.items():
        mapped_key = key
        if mapped_key not in model_state and mapped_key.startswith("module.") and mapped_key[7:] in model_state:
            mapped_key = mapped_key[7:]
        elif mapped_key not in model_state and f"module.{mapped_key}" in model_state:
            mapped_key = f"module.{mapped_key}"
        if mapped_key not in model_state:
            skipped_missing += 1
            continue
        if model_state[mapped_key].shape != value.shape:
            skipped_shape += 1
            continue
        filtered[mapped_key] = value

    missing, unexpected = module.load_state_dict(filtered, strict=False)
    print(
        f"  - loaded {label}: {checkpoint_path} "
        f"(matched={len(filtered)}, missing_after_load={len(missing)}, "
        f"unexpected_after_load={len(unexpected)}, skipped_missing={skipped_missing}, skipped_shape={skipped_shape})"
    )


def run_inference(args: argparse.Namespace, model, data_dict: Dict[str, torch.Tensor], gt_targets: Dict[str, Any]) -> Dict[str, Any]:
    with torch.no_grad():
        if hasattr(model, "inference"):
            try:
                outputs = model.inference(
                    data_dict,
                    gt_targets=gt_targets,
                    use_object_data=args.use_object_data,
                    compute_fk=args.compute_fk,
                    sample_steps=args.sample_steps,
                    sampler=args.sampler,
                    eta=args.eta,
                    interaction_human_source=args.interaction_human_source,
                    interaction_human_trans_source=args.interaction_human_trans_source,
                    interaction_use_human_pred=args.interaction_human_source == "pred",
                )
            except TypeError:
                outputs = model.inference(
                    data_dict,
                    gt_targets=gt_targets,
                    use_object_data=args.use_object_data,
                    compute_fk=args.compute_fk,
                )
        else:
            outputs = model(data_dict, use_object_data=args.use_object_data, compute_fk=args.compute_fk)
    print(f"[demo_badmin] Model inference complete. Output keys: {sorted(outputs.keys())}")
    return outputs


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _get_human_pose_module(model):
    core = _unwrap_model(model)
    module = getattr(core, "human_pose_module", None)
    if module is not None:
        return module
    if core.__class__.__name__.lower() == "humanposemodule":
        return core
    return None


def _select_output_tensor(outputs: Dict[str, Any], names: Tuple[str, ...]) -> Optional[torch.Tensor]:
    for name in names:
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def reconstruct_pred_human(
    model,
    body_model,
    data_dict: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    device: torch.device,
) -> Optional[np.ndarray]:
    root_trans = _select_output_tensor(outputs, ("root_trans_pred", "refined_root_trans", "refined_trans"))
    if root_trans is None:
        root_trans = data_dict.get("trans_gt")
    if root_trans is None:
        print("[demo_badmin] No predicted root translation available; skip Pred-Human mesh.")
        return None
    if root_trans.dim() == 2:
        root_trans = root_trans.unsqueeze(0)

    full_glb = _select_output_tensor(outputs, ("pred_full_pose_rotmat", "R_pred_rotmat", "refined_full_pose_rotmat"))
    human_module = _get_human_pose_module(model)
    if full_glb is not None:
        full_glb = full_glb[0].to(device=device, dtype=torch.float32)
    else:
        p_pred = _select_output_tensor(outputs, ("p_pred", "refined_pose"))
        if p_pred is None or human_module is None or not hasattr(human_module, "_reduced_glb_6d_to_full_glb_mat"):
            print("[demo_badmin] No full pose prediction available; skip Pred-Human mesh.")
            return None
        human_imu = data_dict["human_imu"][0]
        seq_len = min(p_pred.shape[1], human_imu.shape[0]) if p_pred.dim() == 3 else human_imu.shape[0]
        reduced_pose = p_pred[0, :seq_len].reshape(seq_len, len(_REDUCED_POSE_NAMES), 6)
        imu_rot_6d = human_imu[:seq_len, :, -6:]
        imu_rot_mat = t3d.rotation_6d_to_matrix(imu_rot_6d.reshape(-1, 6)).reshape(seq_len, human_imu.shape[1], 3, 3)
        orientation_subset = imu_rot_mat[:, : len(_SENSOR_ROT_INDICES)]
        full_glb = human_module._reduced_glb_6d_to_full_glb_mat(reduced_pose, orientation_subset).to(device=device)

    seq_len = min(full_glb.shape[0], root_trans.shape[1])
    full_glb = full_glb[:seq_len]
    trans_seq = root_trans[0, :seq_len].to(device=device, dtype=torch.float32)

    if human_module is not None and hasattr(human_module, "_global2local") and hasattr(human_module, "smpl_parents"):
        local_rot = human_module._global2local(full_glb, human_module.smpl_parents.tolist())
    else:
        default_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        local_rot = _global_to_local_rotmat(full_glb, default_parents)

    pose_axis = t3d.matrix_to_axis_angle(local_rot.reshape(-1, 3, 3)).reshape(seq_len, full_glb.shape[1], 3)
    with torch.no_grad():
        smpl_out = body_model(
            root_orient=pose_axis[:, 0],
            pose_body=pose_axis[:, 1:22].reshape(seq_len, -1),
            trans=trans_seq,
        )
    return smpl_out.v.detach().cpu().numpy().astype(np.float32)


def build_pred_object_vertices(
    outputs: Dict[str, Any],
    racket_mesh: Optional[Any],
    obj_rotmats_norm: np.ndarray,
) -> Optional[np.ndarray]:
    if racket_mesh is None:
        return None
    pred_obj = _select_output_tensor(outputs, ("pred_obj_trans", "p_obj_pred", "interaction_pred_obj_trans"))
    if pred_obj is None:
        print("[demo_badmin] No object translation prediction available; skip Pred-Object mesh.")
        return None
    if pred_obj.dim() == 2:
        pred_obj = pred_obj.unsqueeze(0)
    pred_trans = pred_obj[0].detach().cpu().numpy().astype(np.float32)
    seq_len = min(pred_trans.shape[0], obj_rotmats_norm.shape[0])
    vertices = np.asarray(racket_mesh.vertices, dtype=np.float32)
    out = np.empty((seq_len, vertices.shape[0], 3), dtype=np.float32)
    for idx in range(seq_len):
        out[idx] = (obj_rotmats_norm[idx] @ vertices.T).T + pred_trans[idx][None, :]
    return out


def detach_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_to_cpu(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_to_cpu(val) for val in value)
    return value


def maybe_save_outputs(args: argparse.Namespace, outputs: Dict[str, Any], prepared: DemoPreparedData) -> None:
    if args.save_pred is not None:
        args.save_pred.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "outputs": detach_to_cpu(outputs),
            "frame_no": prepared.frame_no,
            "contact_state": prepared.contact_state,
            "sync_indices": prepared.sync_indices,
        }
        torch.save(payload, args.save_pred)
        print(f"[demo_badmin] Saved predictions to {args.save_pred}")
    if args.save_data_dict is not None:
        args.save_data_dict.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "data_dict": detach_to_cpu(prepared.data_dict),
                "gt_targets": detach_to_cpu(prepared.gt_targets),
                "frame_no": prepared.frame_no,
            },
            args.save_data_dict,
        )
        print(f"[demo_badmin] Saved data_dict to {args.save_data_dict}")


def visualize_scene(
    gt_vertices: np.ndarray,
    faces: np.ndarray,
    pred_vertices: Optional[np.ndarray],
    pred_object_vertices: Optional[np.ndarray],
    object_faces: Optional[np.ndarray],
) -> None:
    from aitviewer.renderables.meshes import Meshes
    from aitviewer.viewer import Viewer

    viewer = Viewer(title="IMUHOI Badminton Demo")
    viewer.scene.floor.material.diffuse = 0.1
    viewer.scene.floor.tiling = False
    viewer.scene.floor.material.color = (132 / 255, 150 / 255, 183 / 255, 1.0)
    viewer.scene.floor.material.ambient = 0.30
    viewer.scene.background_color = (0.5, 0.5, 0.5, 1.0)

    gt_mesh = Meshes(
        gt_vertices,
        faces,
        name="GT-Human",
        color=(0.72, 0.72, 0.72, 1.0),
        gui_affine=False,
        is_selectable=False,
    )
    gt_mesh.material.ambient = 0.26
    gt_mesh.material.diffuse = 0.41
    viewer.scene.add(gt_mesh)

    if pred_vertices is not None:
        pred_mesh = Meshes(
            pred_vertices,
            faces,
            name="Pred-Human",
            color=(225 / 255, 166 / 255, 100 / 255, 0.86),
            gui_affine=False,
            is_selectable=False,
        )
        pred_mesh.material.ambient = 0.32
        viewer.scene.add(pred_mesh)

    if pred_object_vertices is not None and object_faces is not None:
        obj_mesh = Meshes(
            pred_object_vertices,
            object_faces,
            name="Pred-Object",
            color=(0.12, 0.46, 0.96, 1.0),
            gui_affine=False,
            is_selectable=False,
        )
        obj_mesh.material.ambient = 0.36
        viewer.scene.add(obj_mesh)

    viewer.run()


def main() -> None:
    args = parse_args()
    args.badminton_root = resolve_path(args.badminton_root, Path.cwd())
    args.noitom_csv = resolve_path(args.noitom_csv, PROJECT_ROOT)
    args.sync_path = resolve_path(args.sync_path, PROJECT_ROOT)
    args.config = resolve_path(args.config, PROJECT_ROOT)
    args.body_model = resolve_path(args.body_model, PROJECT_ROOT)
    args.betas_npz = resolve_path(args.betas_npz, PROJECT_ROOT)
    args.racket_obj = resolve_path(args.racket_obj, args.badminton_root)
    args.save_pred = resolve_path(args.save_pred, PROJECT_ROOT)
    args.save_data_dict = resolve_path(args.save_data_dict, PROJECT_ROOT)

    device = resolve_device(args.device)
    cfg = load_config(str(args.config))
    if args.model_arch is not None:
        cfg.model_arch = args.model_arch
    cfg.body_model_path = str(args.body_model)

    fit_module = load_fit_module(args.badminton_root)
    betas = load_betas(args.betas_npz, args.num_betas, fit_module)
    body_model = fit_module.create_body_model(args.body_model, args.num_betas, device)
    racket_mesh = load_racket_mesh(args, fit_module)

    prepared = build_prepared_data(
        args=args,
        fit_module=fit_module,
        body_model=body_model,
        betas=betas,
        racket_mesh=racket_mesh,
        device=device,
        cfg=cfg,
    )
    model = configure_model(args, cfg, device)
    outputs = run_inference(args, model, prepared.data_dict, prepared.gt_targets)
    maybe_save_outputs(args, outputs, prepared)

    pred_vertices = reconstruct_pred_human(model, body_model, prepared.data_dict, outputs, device)
    pred_object_vertices = build_pred_object_vertices(outputs, racket_mesh, prepared.obj_rotmats_norm)

    if args.no_viewer:
        print("[demo_badmin] Visualization skipped (--no-viewer).")
        return

    object_faces = None if racket_mesh is None else np.asarray(racket_mesh.faces, dtype=np.int32)
    visualize_scene(
        gt_vertices=prepared.gt_vertices,
        faces=prepared.gt_faces,
        pred_vertices=pred_vertices,
        pred_object_vertices=pred_object_vertices,
        object_faces=object_faces,
    )


if __name__ == "__main__":
    main()
