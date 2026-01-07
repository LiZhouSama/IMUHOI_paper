"""
Estimate world-frame linear acceleration from MoVi IMU data using an EKF
on raw signals (no bias correction). Data source:
    D:/a_WORK/Projects/PhD/datasets/MoVi/IMUmatlab/imu_Subject_1.mat

Inputs:
    - Acceleration channels A-x/y/z (units: g, includes gravity)
    - Gyro channels W-x/y/z (units: rad/s)
Pipeline:
    - EKF (ahrs.filters.EKF, frame ENU) on raw accel/gyro; initial quat = identity.
    - Rotate body accel to world frame and subtract gravity to get linear accel.

Run with repo venv (needs numpy, scipy, ahrs, matplotlib):
    .\\.venv\\Scripts\\python test/movi_imu_world_estimate.py
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ahrs.filters import EKF
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R


MAT_PATH = Path(r"D:\a_WORK\Projects\PhD\datasets\MoVi\IMUmatlab\imu_Subject_1.mat")
GRAVITY = 9.80665  # m/s^2 per g
SAMPLE_RATE = 120.0  # Hz (MoVi/Xsens IMU export)

# Joint channel layout per joint (stride = 16):
_DATA_TYPES = [("X", 3), ("V", 3), ("Q", 4), ("A", 3), ("W", 3)]
_OFFSETS: Dict[str, int] = {}
_stride = 0
for name, width in _DATA_TYPES:
    _OFFSETS[name] = _stride
    _stride += width
JOINT_STRIDE = _stride  # 16


def _load_s1_synched(mat_path: Path):
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    return mat["IMU"].S1_Synched


def _joint_base(joint_numbers: np.ndarray, joint_id: str) -> int:
    joint_id = str(joint_id).zfill(2)
    try:
        idx = list(joint_numbers.astype(str)).index(joint_id)
    except ValueError as exc:
        raise KeyError(f"Joint {joint_id} not found; available: {joint_numbers}") from exc
    return idx * JOINT_STRIDE


def _slice(data: np.ndarray, base: int, key: str, width: int) -> np.ndarray:
    start = base + _OFFSETS[key]
    return data[:, start : start + width]


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert quaternions from [w, x, y, z] to [x, y, z, w] for scipy Rotation."""
    return np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)


def estimate_world_motion_ekf(joint_id: str = "01") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s1 = _load_s1_synched(MAT_PATH)
    base = _joint_base(s1.jointNumbers, joint_id)
    data = s1.data.astype(np.float64)

    acc_body_g = _slice(data, base, "A", 3)  # includes gravity, units g
    gyro = _slice(data, base, "W", 3)  # rad/s

    acc_body = acc_body_g * GRAVITY
    dt = 1.0 / SAMPLE_RATE

    ekf = EKF(frequency=SAMPLE_RATE, frame="ENU")
    quats = np.zeros((acc_body.shape[0], 4))
    q = np.array([1.0, 0.0, 0.0, 0.0])  # identity init; no bias correction
    for i in range(acc_body.shape[0]):
        q = ekf.update(q, gyr=gyro[i], acc=acc_body[i])
        quats[i] = q

    rot = R.from_quat(_quat_wxyz_to_xyzw(quats))
    acc_world = rot.apply(acc_body)

    gravity_world = np.array([0.0, 0.0, GRAVITY])
    lin_acc_world = acc_world - gravity_world

    euler_ypr = rot.as_euler("zyx", degrees=False)

    return lin_acc_world, acc_world, quats, euler_ypr


def main():
    lin_acc, acc_world, quats_wxyz, euler_ypr = estimate_world_motion_ekf("01")
    t = np.arange(lin_acc.shape[0]) / SAMPLE_RATE
    print(f"Samples: {lin_acc.shape[0]}, dt={1.0/SAMPLE_RATE:.4f}s")
    print(f"Linear acc stats (m/s^2): mean={lin_acc.mean(0)}, std={lin_acc.std(0)}")
    print("First 3 Euler yaw/pitch/roll (rad):")
    print(euler_ypr[:3])

    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    for i, ax_key in enumerate(("x", "y", "z")):
        ax.plot(t, lin_acc[:, i], color=colors[ax_key], label=f"lin acc {ax_key}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear acceleration (m/s^2)")
    ax.set_title("World-frame linear acceleration (EKF, joint 01)")
    ax.set_ylim(-10, 10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    out_path = Path(__file__).with_name("imu_s1_joint01_linacc.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Linear acceleration plot saved to: {out_path}")


if __name__ == "__main__":
    main()
