"""
Quick inspection script to compare object IMU data with ground-truth translation.

Assumptions:
- Sampling rate is unknown in the dataset; default to 60 Hz (Movella DOT typical).
- Acceleration is assumed gravity-compensated (mean near zero).
- Initial velocity is set to zero and initial position is aligned to GT frame 0.

This is a sanity check only; full-scale inertial navigation would require
proper calibration, gravity handling, bias removal, and filtering.
"""

import pickle

import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore
    _mpl_import_error = None
except Exception as exc:  # noqa: BLE001
    plt = None  # type: ignore
    _mpl_import_error = exc


DATA_PARTS = [
    "20231015",
    "20231015_dujsh_chair",
    "freestyle1"]

def _candidate_roots() -> List[Path]:
    """Return possible dataset root locations (handles Windows symlink)."""
    here = Path(__file__).resolve()
    candidates = []

    # 1) Repo root / datasets / IMHD / IMHD Dataset
    repo_root = here.parents[1]
    candidates.append(repo_root / "datasets" / "IMHD" / "IMHD Dataset")

    # 2) Walk up ancestors, append datasets/IMHD/IMHD Dataset under each.
    for parent in here.parents:
        cand = parent / "datasets" / "IMHD" / "IMHD Dataset"
        if cand not in candidates:
            candidates.append(cand)

    # 3) Absolute known host path (symlink target).
    candidates.append(Path(r"D:\a_WORK\Projects\PhD\datasets\IMHD\IMHD Dataset"))

    return candidates


def _resolve_file(prefix: str, rel_parts: List[str], filename: str) -> Path:
    for root in _candidate_roots():
        path = root / prefix / rel_parts[0] / rel_parts[1] / rel_parts[2] / filename
        try:
            if path.exists():
                return path
        except OSError:
            continue
    raise FileNotFoundError(f"Cannot locate {filename} under known dataset roots.")


def resolve_paths():
    imu = _resolve_file("imu_preprocessed", DATA_PARTS, "imu_0_305_-1.pkl")
    gt = _resolve_file("ground_truth", DATA_PARTS, "gt_0_305_-1.pkl")
    return imu, gt


def integrate_positions(acc: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Double integrate acceleration with simple bias removal."""
    acc_detrend = acc - acc.mean(axis=0, keepdims=True)
    vel = np.cumsum(acc_detrend * dt, axis=0)
    pos = np.cumsum(vel * dt, axis=0)
    return vel, pos


def finite_difference(series: np.ndarray, dt: float, n: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocity by first difference; acceleration by smoothed second-order difference:
        a_t = (T_{t-n} + T_{t+n} - 2*T_t) / (n*dt)^2
    Boundary frames (<n or >len-n-1) are zeroed.
    """
    vel = np.vstack([np.zeros((1, series.shape[1])), np.diff(series, axis=0) / dt])
    acc = np.zeros_like(series)
    denom = (n * dt) ** 2
    for t in range(n, series.shape[0] - n):
        acc[t] = (series[t - n] + series[t + n] - 2 * series[t]) / denom
    return vel, acc


def main(dt: float = 1.0 / 60.0):
    imu_path, gt_path = resolve_paths()
    print(f"IMU file: {imu_path}")
    print(f"GT file:  {gt_path}")

    imu_data = pickle.load(open(imu_path, "rb"))
    gt_data = pickle.load(open(gt_path, "rb"))

    acc = imu_data["objectImuAcc"]  # (T, 3)
    ori = imu_data["objectImuOri"]  # (T, 3)
    gt_trans = gt_data["objectTrans"]  # (T, 3)
    gt_rot = gt_data["objectRot"]  # (T, 3)

    assert acc.shape[0] == gt_trans.shape[0], "IMU and GT lengths differ"

    print("\nShapes:")
    print(f"  acc: {acc.shape}, dtype={acc.dtype}")
    print(f"  ori: {ori.shape}, dtype={ori.dtype}")
    print(f"  gt_trans: {gt_trans.shape}, dtype={gt_trans.dtype}")
    print(f"  gt_rot: {gt_rot.shape}, dtype={gt_rot.dtype}")

    print("\nBasic stats:")
    print(f"  acc range per axis: min {acc.min(axis=0)}, max {acc.max(axis=0)}")
    print(f"  acc mean: {acc.mean(axis=0)}")
    print(f"  acc std : {acc.std(axis=0)}")
    print(f"  gt_trans range per axis: min {gt_trans.min(axis=0)}, max {gt_trans.max(axis=0)}")
    print(f"  gt_rot mean: {gt_rot.mean(axis=0)}, std: {gt_rot.std(axis=0)}")

    # Naive integration (bias-removed) and alignment to GT initial position
    vel_rel, pos_rel = integrate_positions(acc, dt)
    pos_est = pos_rel + gt_trans[0:1]

    diff = pos_est - gt_trans
    rmse = np.sqrt((diff ** 2).mean(axis=0))
    print("\nNaive acc->pos comparison (dt = {:.6f}s):".format(dt))
    print(f"  RMSE per axis: {rmse}")
    print(f"  Final drift (est - gt) at last frame: {diff[-1]}")
    print(f"  Mean abs error per axis: {np.abs(diff).mean(axis=0)}")

    # Direct orientation (component-wise) comparison
    rot_diff = ori - gt_rot
    rot_rmse = np.sqrt((rot_diff ** 2).mean(axis=0))
    print("\nOrientation component comparison (IMU ori vs GT rot):")
    print(f"  RMSE per axis: {rot_rmse}")
    print(f"  Mean abs diff per axis: {np.abs(rot_diff).mean(axis=0)}")
    print(f"  Max abs diff per axis: {np.abs(rot_diff).max(axis=0)}")

    # GT finite differences
    gt_vel, gt_acc_est = finite_difference(gt_trans, dt)

    # Visualization
    if plt is None:
        print("\nmatplotlib unavailable; skipping plot.")
        if _mpl_import_error:
            print(f"Import error: {_mpl_import_error}")
        return

    t = np.arange(acc.shape[0]) * dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.set_title("IMU: acc vs integrated vel/pos")
    # ax1.plot(t, acc[:, 0], label="acc x", color="tab:red", alpha=0.7)
    # ax1.plot(t, acc[:, 1], label="acc y", color="tab:green", alpha=0.7)
    # ax1.plot(t, acc[:, 2], label="acc z", color="tab:blue", alpha=0.7)
    ax1.plot(t, vel_rel[:, 0], "--", label="vel x (int)", color="maroon")
    ax1.plot(t, vel_rel[:, 1], "--", label="vel y (int)", color="darkgreen")
    ax1.plot(t, vel_rel[:, 2], "--", label="vel z (int)", color="navy")
    ax1.plot(t, pos_est[:, 0], ":", label="pos x (int+align)", color="orange")
    ax1.plot(t, pos_est[:, 1], ":", label="pos y (int+align)", color="olive")
    ax1.plot(t, pos_est[:, 2], ":", label="pos z (int+align)", color="teal")
    ax1.legend(loc="upper right", ncol=2, fontsize=8)
    ax1.set_ylabel("Value")
    # ax1.set_xlim(0, 10)
    ax1.set_ylim(-5, 5)

    ax2.set_title("GT: trans, diff vel, diff acc")
    ax2.plot(t, gt_trans[:, 0], label="gt trans x", color="orange")
    ax2.plot(t, gt_trans[:, 1], label="gt trans y", color="olive")
    ax2.plot(t, gt_trans[:, 2], label="gt trans z", color="teal")
    ax2.plot(t, gt_vel[:, 0], "--", label="gt vel x (diff)", color="maroon")
    ax2.plot(t, gt_vel[:, 1], "--", label="gt vel y (diff)", color="darkgreen")
    ax2.plot(t, gt_vel[:, 2], "--", label="gt vel z (diff)", color="navy")
    # ax2.plot(t, gt_acc_est[:, 0], label="gt acc x (diff2)", color="tab:red")
    # ax2.plot(t, gt_acc_est[:, 1], label="gt acc y (diff2)", color="tab:green")
    # ax2.plot(t, gt_acc_est[:, 2], label="gt acc z (diff2)", color="tab:blue")
    ax2.legend(loc="upper right", ncol=2, fontsize=8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Value")
    # ax2.set_xlim(0, 10)
    ax2.set_ylim(-5, 5)

    fig.tight_layout()
    out_path = Path(__file__).resolve().with_name("compare_imu_to_gt.svg")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
