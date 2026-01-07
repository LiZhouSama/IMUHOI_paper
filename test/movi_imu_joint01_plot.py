"""
Load MoVi IMU data (Subject 1, S1_Synched) and visualize joint 01 acceleration
and velocity. Acceleration (A-x/y/z) is in g with -g on the x-axis; we remove
that bias, convert to m/s^2, integrate to velocity, and compare against the
provided X (displacement) and V (velocity) channels.

Run with the repo-local virtualenv (uses numpy/scipy/matplotlib):
    .\\.venv\\Scripts\\python test/movi_imu_joint01_plot.py
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


MAT_PATH = Path(r"D:\a_WORK\Projects\PhD\datasets\MoVi\IMUmatlab\imu_Subject_1.mat")
GRAVITY = 9.80665  # m/s^2 per g
SAMPLE_RATE = 120.0  # Hz, MoVi IMU export (Xsens) uses 240 Hz

# Data layout per joint (total 16 channels)
_DATA_TYPES = [("X", 3), ("V", 3), ("Q", 4), ("A", 3), ("W", 3)]
_DATA_OFFSETS = {}
_offset = 0
for name, width in _DATA_TYPES:
    _DATA_OFFSETS[name] = _offset
    _offset += width
_JOINT_STRIDE = _offset


def _load_s1_synched(mat_path: Path):
    mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    return mat["IMU"].S1_Synched


def _joint_base_index(joint_numbers: np.ndarray, joint_id: str) -> int:
    try:
        joint_idx = list(joint_numbers).index(joint_id)
    except ValueError as exc:
        raise KeyError(f"Joint {joint_id} not found; available: {joint_numbers}") from exc
    return joint_idx * _JOINT_STRIDE


def _slice(data: np.ndarray, base: int, key: str, width: int) -> np.ndarray:
    start = base + _DATA_OFFSETS[key]
    return data[:, start : start + width]


def _integrate_trap(series: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(series)
    if series.shape[0] <= 1:
        return out
    inc = 0.5 * (series[1:] + series[:-1]) * dt
    out[1:] = np.cumsum(inc, axis=0)
    return out


def main() -> Path:
    s1 = _load_s1_synched(MAT_PATH)
    joint_numbers: np.ndarray = s1.jointNumbers.astype(str)
    base = _joint_base_index(joint_numbers, "01")

    data = s1.data.astype(np.float64)
    pos = _slice(data, base, "X", 3)  # meters
    vel_raw = _slice(data, base, "V", 3)  # m/s

    acc_g = _slice(data, base, "A", 3)  # g
    acc_g[:, 0] += 1.0  # remove -g on x-axis
    acc_ms2 = acc_g * GRAVITY

    dt = 1.0 / SAMPLE_RATE
    vel_from_acc = _integrate_trap(acc_ms2, dt)
    vel_from_pos = np.gradient(pos, dt, axis=0)
    disp_from_vel = _integrate_trap(vel_raw, dt)

    t = np.arange(data.shape[0]) * dt

    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Displacement: integrate V vs provided X.
    for i, ax_key in enumerate(("x", "y", "z")):
        axes[0].plot(t, disp_from_vel[:, i], color=colors[ax_key], label=f"disp-{ax_key} (∫V)")
        axes[0].plot(t, pos[:, i], color=colors[ax_key], linestyle="--", label=f"X-{ax_key} (provided)")
    axes[0].set_ylabel("Displacement (m)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(ncol=3, fontsize=8)

    # Velocity: integrated A vs provided V vs dX/dt.
    for i, ax_key in enumerate(("x", "y", "z")):
        axes[1].plot(t, vel_from_acc[:, i], color=colors[ax_key], label=f"V-{ax_key} (∫A)")
        axes[1].plot(t, vel_raw[:, i], color=colors[ax_key], linestyle="--", alpha=0.6, label=f"V-{ax_key} (raw)")
        axes[1].plot(t, vel_from_pos[:, i], color=colors[ax_key], linestyle=":", alpha=0.6, label=f"dX-{ax_key}/dt")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(ncol=3, fontsize=8)

    fig.suptitle("MoVi Subject 1 – Joint 01: Displacement (∫V vs X) and Velocity")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(__file__).with_name("imu_s1_joint01_disp_vel.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Figure saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
