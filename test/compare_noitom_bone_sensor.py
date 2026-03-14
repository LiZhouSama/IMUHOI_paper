#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare Noitom Bone-Quat/Joint-Posi vs Sensor-Quat/Acce for DEFAULT_SENSOR_ORDER.

Goal: check whether the Bone-Quat and Sensor-Quat differ by a fixed rotation.
"""
import argparse
import os
from itertools import permutations, product
from typing import Dict, List, Tuple

import numpy as np


DEFAULT_SENSOR_ORDER = ["Hips", "LeftLeg", "RightLeg", "Head", "LeftForeArm", "RightForeArm"]
SENSOR_JOINT_MAP = {
    "Hips": "Hips", "LeftLeg": "LeftFoot", "RightLeg": "RightFoot",
    "Head": "Head", "LeftForeArm": "LeftHand", "RightForeArm": "RightHand",
}
NOITOM_SRC_FPS = 60


def _load_raw_csv(path: str) -> Tuple[List[str], List[List[str]]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing csv: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = [line.strip() for line in f if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
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
        raise ValueError(f"{bone_name}/{joint_name} missing fields: {', '.join(missing)}")
    return indices


def _resolve_sensor_indices(header: List[str], sensor_name: str) -> Dict[str, int]:
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
        raise ValueError(f"{sensor_name} missing fields: {', '.join(missing)}")
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
        raise ValueError("No valid Bone/Joint rows")
    return np.asarray(pos_list, dtype=np.float64), np.asarray(quat_list, dtype=np.float64)


def _read_sensor_series(rows: List[List[str]], indices: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    acc_list, quat_list = [], []
    for line in rows:
        try:
            acc_list.append([float(line[indices[k]]) for k in ["acc_x", "acc_y", "acc_z"]])
            quat_list.append([float(line[indices[k]]) for k in ["quat_w", "quat_x", "quat_y", "quat_z"]])
        except (ValueError, IndexError):
            continue
    if not acc_list:
        raise ValueError("No valid Sensor rows")
    return np.asarray(acc_list, dtype=np.float64), np.asarray(quat_list, dtype=np.float64)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return q / norm


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def _rotation_angle_deg(R: np.ndarray) -> np.ndarray:
    trace = np.einsum("nii->n", R)
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def _mean_rotation(R_seq: np.ndarray) -> np.ndarray:
    M = np.mean(R_seq, axis=0)
    return _project_to_so3(M)


def _axis_permutation_candidates() -> List[np.ndarray]:
    axes = np.eye(3)
    candidates = []
    for perm in permutations([0, 1, 2]):
        for signs in product([1.0, -1.0], repeat=3):
            M = axes[:, perm] * np.asarray(signs)[None, :]
            if np.linalg.det(M) > 0.0:
                candidates.append(M)
    return candidates


def _best_axis_permutation(R: np.ndarray, candidates: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    best_R = None
    best_angle = 1e9
    for cand in candidates:
        err = R.T @ cand
        ang = float(_rotation_angle_deg(err.reshape(1, 3, 3))[0])
        if ang < best_angle:
            best_angle = ang
            best_R = cand
    return best_R, best_angle


def _stats_for_delta(delta: np.ndarray) -> Dict[str, np.ndarray]:
    R_mean = _mean_rotation(delta)
    err = np.einsum("ij,njk->nik", R_mean.T, delta)
    angles = _rotation_angle_deg(err)
    return {
        "R_mean": R_mean,
        "angles": angles,
        "mean": float(np.mean(angles)),
        "std": float(np.std(angles)),
        "max": float(np.max(angles)),
    }


def _select_tpose_indices(R_bone: np.ndarray, tpose_frames: int, angle_thresh_deg: float, fallback: int) -> np.ndarray:
    total = R_bone.shape[0]
    if tpose_frames > 0:
        return np.arange(min(tpose_frames, total))
    angles = _rotation_angle_deg(R_bone)
    idx = np.where(angles < angle_thresh_deg)[0]
    if idx.size == 0:
        return np.arange(min(fallback, total))
    return idx


def _format_matrix(R: np.ndarray) -> str:
    return np.array2string(R, precision=4, suppress_small=True)


def _compute_r_fix_from_tpose(R_sensor: np.ndarray, R_bone: np.ndarray, tpose_idx: np.ndarray) -> np.ndarray:
    # Assumption B: R_bone ~= R_sensor * R_fix  => R_fix ~= R_sensor^T * R_bone
    delta = np.einsum("nij,njk->nik", np.transpose(R_sensor, (0, 2, 1)), R_bone)
    return _mean_rotation(delta[tpose_idx])


def _compute_error_angles(R_sensor: np.ndarray, R_bone: np.ndarray, R_fix: np.ndarray) -> np.ndarray:
    # err = (R_sensor * R_fix)^T * R_bone
    R_pred = np.einsum("nij,jk->nik", R_sensor, R_fix)
    err = np.einsum("nij,njk->nik", np.transpose(R_pred, (0, 2, 1)), R_bone)
    return _rotation_angle_deg(err)


def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = int(np.ceil(len(x) / max_points))
    return x[::step], y[::step]


def _write_svg_plot(plot_path: str,
                    error_curves: Dict[str, np.ndarray],
                    time_axis: np.ndarray,
                    tpose_ranges: Dict[str, np.ndarray],
                    start_frame: int,
                    tpose_frames: int) -> None:
    names = list(error_curves.keys())
    if not names:
        return
    cols = 2
    rows = (len(names) + cols - 1) // cols
    subplot_w, subplot_h = 600, 220
    margin_l, margin_r, margin_t, margin_b = 50, 20, 30, 35
    width = cols * subplot_w + margin_l + margin_r
    height = rows * subplot_h + margin_t + margin_b

    t_min, t_max = float(time_axis[0]), float(time_axis[-1]) if len(time_axis) > 1 else float(time_axis[0] + 1.0)
    global_max = max(float(np.max(v)) for v in error_curves.values())
    y_max = max(5.0, global_max * 1.1)

    def x_map(t, col):
        x0 = margin_l + col * subplot_w
        return x0 + (t - t_min) / (t_max - t_min) * (subplot_w - 10)

    def y_map(a, row):
        y0 = margin_t + row * subplot_h
        return y0 + (subplot_h - 20) - (a / y_max) * (subplot_h - 30)

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append(f'<text x="{margin_l}" y="{margin_t - 10}" font-size="14">R_bone vs R_sensor * R_fix error curves</text>')

    for idx, name in enumerate(names):
        row, col = divmod(idx, cols)
        x0 = margin_l + col * subplot_w
        y0 = margin_t + row * subplot_h
        svg.append(f'<rect x="{x0}" y="{y0}" width="{subplot_w - 10}" height="{subplot_h - 20}" fill="none" stroke="#888"/>')
        svg.append(f'<text x="{x0 + 5}" y="{y0 + 14}" font-size="12">{name} error (deg)</text>')

        # T-pose span
        if tpose_frames > 0:
            t0 = t_min
            t1 = (tpose_frames - 1 + start_frame) / float(NOITOM_SRC_FPS)
        else:
            idxs = tpose_ranges.get(name, np.array([]))
            if idxs.size > 0:
                t0 = (idxs.min() + start_frame) / float(NOITOM_SRC_FPS)
                t1 = (idxs.max() + start_frame) / float(NOITOM_SRC_FPS)
            else:
                t0 = t1 = None
        if t0 is not None and t1 is not None and t1 > t0:
            x_left = x_map(t0, col)
            x_right = x_map(t1, col)
            y_top = y0 + 5
            svg.append(f'<rect x="{x_left:.1f}" y="{y_top:.1f}" width="{(x_right - x_left):.1f}" '
                       f'height="{subplot_h - 30:.1f}" fill="orange" fill-opacity="0.15" stroke="none"/>')

        # Curve
        angles = error_curves[name]
        tx, ty = _downsample_xy(time_axis[:len(angles)], angles)
        pts = " ".join(f"{x_map(t, col):.2f},{y_map(a, row):.2f}" for t, a in zip(tx, ty))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="1"/>')

        # Y-axis labels
        svg.append(f'<text x="{x0 + 2}" y="{y0 + subplot_h - 5}" font-size="10">0</text>')
        svg.append(f'<text x="{x0 + 2}" y="{y0 + 18}" font-size="10">{y_max:.1f}</text>')

        # X-axis labels
        svg.append(f'<text x="{x0 + 2}" y="{y0 + subplot_h - 8}" font-size="10">{t_min:.1f}s</text>')
        svg.append(f'<text x="{x0 + subplot_w - 45}" y="{y0 + subplot_h - 8}" font-size="10">{t_max:.1f}s</text>')

    svg.append("</svg>")
    with open(plot_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def _fit_world_mount_right(R_sensor: np.ndarray,
                           R_bone: np.ndarray,
                           tpose_idx: np.ndarray,
                           iters: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    # Solve R_bone ≈ R_world * R_sensor * R_mount
    # R_world shared, R_mount per sensor.
    R_world = np.eye(3, dtype=np.float64)
    R_mount = np.eye(3, dtype=np.float64)
    idx = tpose_idx
    Rs = R_sensor[idx]
    Rb = R_bone[idx]
    for _ in range(iters):
        # R_mount <- mean( (R_world * R_sensor)^T * R_bone )
        delta_m = np.einsum("nij,njk->nik",
                            np.transpose(np.einsum("ij,njk->nik", R_world, Rs), (0, 2, 1)),
                            Rb)
        R_mount = _mean_rotation(delta_m)
        # R_world <- mean( R_bone * (R_sensor * R_mount)^T )
        delta_w = np.einsum("nij,njk->nik",
                            Rb,
                            np.transpose(np.einsum("nij,jk->nik", Rs, R_mount), (0, 2, 1)))
        R_world = _mean_rotation(delta_w)
    return R_world, R_mount


def _compute_error_angles_world_mount(R_sensor: np.ndarray,
                                      R_bone: np.ndarray,
                                      R_world: np.ndarray,
                                      R_mount: np.ndarray) -> np.ndarray:
    R_pred = np.einsum("ij,njk->nik", R_world, np.einsum("nij,jk->nik", R_sensor, R_mount))
    err = np.einsum("nij,njk->nik", np.transpose(R_pred, (0, 2, 1)), R_bone)
    return _rotation_angle_deg(err)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Bone-Quat vs Sensor-Quat alignment (Noitom CSV)")
    parser.add_argument("--csv", default="noitom/7IMU/output/raw_imu002_human.csv",
                        help="Noitom human CSV containing Bone-Quat/Joint-Posi and Sensor-Quat/Acce")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time (sec)")
    parser.add_argument("--duration-sec", type=float, default=-1.0, help="Duration (sec), -1 for full")
    parser.add_argument("--tpose-frames", type=int, default=60, help="Use first N frames as T-pose, 0 to auto-detect")
    parser.add_argument("--tpose-angle-deg", type=float, default=5.0, help="Auto-detect T-pose threshold (deg)")
    parser.add_argument("--fallback-frames", type=int, default=30, help="Fallback frames if auto-detect finds none")
    parser.add_argument("--max-frames", type=int, default=-1, help="Optional cap on total frames")
    parser.add_argument("--plot", action="store_true", help="Save error curve plot (R_bone vs R_sensor*R_fix)")
    parser.add_argument("--plot-out", default="test/compare_noitom_bone_sensor_error.svg",
                        help="Output path for error curve plot (SVG)")
    parser.add_argument("--fit-world-mount", action="store_true",
                        help="Fit R_world (shared) and R_mount (per sensor) for R_bone ≈ R_world * R_sensor * R_mount")
    parser.add_argument("--wm-iters", type=int, default=8, help="Iterations for world/mount fitting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header, rows = _load_raw_csv(args.csv)
    candidates = _axis_permutation_candidates()

    print(f"CSV: {args.csv}")
    print(f"Sensors: {DEFAULT_SENSOR_ORDER}")
    print(f"Start sec: {args.start_sec}, duration sec: {args.duration_sec}")
    print(f"T-pose frames: {args.tpose_frames} (auto if 0), angle thresh: {args.tpose_angle_deg}")

    error_curves = {}
    tpose_ranges = {}
    time_axis = None
    mw_world = None

    for sensor_name in DEFAULT_SENSOR_ORDER:
        joint_name = SENSOR_JOINT_MAP.get(sensor_name, sensor_name)
        bone_idx = _resolve_bone_joint_indices(header, sensor_name, joint_name)
        sensor_idx = _resolve_sensor_indices(header, sensor_name)

        _, bone_quat = _read_bone_joint_series(rows, bone_idx)
        _, sensor_quat = _read_sensor_series(rows, sensor_idx)

        total = min(len(bone_quat), len(sensor_quat))
        if total == 0:
            print(f"[{sensor_name}] empty data")
            continue

        start_frame = int(max(0.0, args.start_sec) * NOITOM_SRC_FPS)
        if args.duration_sec > 0:
            end_frame = min(total, start_frame + int(args.duration_sec * NOITOM_SRC_FPS))
        else:
            end_frame = total
        if start_frame >= end_frame:
            print(f"[{sensor_name}] invalid slice {start_frame}:{end_frame}")
            continue

        bone_quat = bone_quat[start_frame:end_frame]
        sensor_quat = sensor_quat[start_frame:end_frame]
        if args.max_frames > 0:
            bone_quat = bone_quat[:args.max_frames]
            sensor_quat = sensor_quat[:args.max_frames]

        R_bone = _quat_to_rot(bone_quat)
        R_sensor = _quat_to_rot(sensor_quat)
        n = min(R_bone.shape[0], R_sensor.shape[0])
        R_bone = R_bone[:n]
        R_sensor = R_sensor[:n]

        tpose_idx = _select_tpose_indices(R_bone, args.tpose_frames, args.tpose_angle_deg, args.fallback_frames)
        if tpose_idx.size == 0:
            print(f"[{sensor_name}] no T-pose frames found")
            continue

        delta_a = np.einsum("nij,njk->nik", R_bone, np.transpose(R_sensor, (0, 2, 1)))
        delta_b = np.einsum("nij,njk->nik", np.transpose(R_bone, (0, 2, 1)), R_sensor)

        stats_a_all = _stats_for_delta(delta_a)
        stats_b_all = _stats_for_delta(delta_b)
        stats_a_tp = _stats_for_delta(delta_a[tpose_idx])
        stats_b_tp = _stats_for_delta(delta_b[tpose_idx])

        best_perm_a, best_perm_ang_a = _best_axis_permutation(stats_a_tp["R_mean"], candidates)
        best_perm_b, best_perm_ang_b = _best_axis_permutation(stats_b_tp["R_mean"], candidates)

        print("")
        print(f"=== {sensor_name} ===")
        print(f"Frames used: {n}, T-pose frames: {len(tpose_idx)}")
        print("Assumption A: R_bone ~= R_fix * R_sensor")
        print(f"All frames mean/std/max (deg): {stats_a_all['mean']:.3f}/{stats_a_all['std']:.3f}/{stats_a_all['max']:.3f}")
        print(f"T-pose mean/std/max (deg): {stats_a_tp['mean']:.3f}/{stats_a_tp['std']:.3f}/{stats_a_tp['max']:.3f}")
        print(f"R_fix (T-pose mean): { _format_matrix(stats_a_tp['R_mean']) }")
        print(f"Best axis-perm angle (deg): {best_perm_ang_a:.3f}")
        if best_perm_a is not None:
            print(f"Best axis-perm matrix: { _format_matrix(best_perm_a) }")

        print("Assumption B: R_bone ~= R_sensor * R_fix")
        print(f"All frames mean/std/max (deg): {stats_b_all['mean']:.3f}/{stats_b_all['std']:.3f}/{stats_b_all['max']:.3f}")
        print(f"T-pose mean/std/max (deg): {stats_b_tp['mean']:.3f}/{stats_b_tp['std']:.3f}/{stats_b_tp['max']:.3f}")
        print(f"R_fix (T-pose mean): { _format_matrix(stats_b_tp['R_mean']) }")
        print(f"Best axis-perm angle (deg): {best_perm_ang_b:.3f}")
        if best_perm_b is not None:
            print(f"Best axis-perm matrix: { _format_matrix(best_perm_b) }")

        R_fix = _compute_r_fix_from_tpose(R_sensor, R_bone, tpose_idx)
        err_angles = _compute_error_angles(R_sensor, R_bone, R_fix)
        error_curves[sensor_name] = err_angles
        tpose_ranges[sensor_name] = tpose_idx

        if time_axis is None or len(time_axis) != n:
            time_axis = (np.arange(n) + start_frame) / float(NOITOM_SRC_FPS)

        print(f"R_fix (T-pose, assumption B): { _format_matrix(R_fix) }")

        if args.fit_world_mount:
            R_world, R_mount = _fit_world_mount_right(R_sensor, R_bone, tpose_idx, iters=args.wm_iters)
            if mw_world is None:
                mw_world = R_world
            err_wm = _compute_error_angles_world_mount(R_sensor, R_bone, R_world, R_mount)
            print("Fit: R_bone ~= R_world * R_sensor * R_mount")
            print(f"R_world: { _format_matrix(R_world) }")
            print(f"R_mount: { _format_matrix(R_mount) }")
            print(f"World/Mount err mean/std/max (deg): {err_wm.mean():.3f}/{err_wm.std():.3f}/{err_wm.max():.3f}")

    if args.plot and error_curves:
        out_path = args.plot_out
        if not out_path.lower().endswith(".svg"):
            base, _ = os.path.splitext(out_path)
            out_path = f"{base}.svg"
        _write_svg_plot(out_path, error_curves, time_axis, tpose_ranges,
                        int(max(0.0, args.start_sec) * NOITOM_SRC_FPS), args.tpose_frames)
        print(f"Saved SVG plot: {out_path}")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
