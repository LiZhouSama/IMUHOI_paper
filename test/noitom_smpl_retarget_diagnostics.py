#!/usr/bin/env python3
"""
Diagnose Noitom Calc Bone-Quat -> SMPLH retargeting twist issues.

This is a visualization/diagnostic script, not a replacement retargeter.  It
compares the current direct Bone-Quat retargeting with an arm-only swing variant
that uses Noitom Joint-Posi bone directions and removes forearm/wrist twist.
The swing variant does not require a T-pose frame; it is intended to answer
whether the visible artifact is mainly a rest-frame/twist problem.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BADMINTON_ROOT_DEFAULT = Path("/mnt/d/a_WORK/Projects/PhD/tasks/badminton")
DEFAULT_FIT_SCRIPT = BADMINTON_ROOT_DEFAULT / "fit_noitom_smplh.py"
DEFAULT_NOITOM_CSV = BADMINTON_ROOT_DEFAULT / "noitom_badmin/output_noitom_csv/take006_chr02.csv"
DEFAULT_BODY_MODEL = Path("/mnt/d/a_WORK/Projects/PhD/datasets/smpl_models/smplh/neutral/model.npz")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "test/noitom_retarget_diagnostics"

ARM_SEGMENTS = [
    ("left_upper_arm", "left_shoulder", "left_elbow", "LeftArm", "LeftForeArm"),
    ("left_forearm", "left_elbow", "left_wrist", "LeftForeArm", "LeftHand"),
    ("right_upper_arm", "right_shoulder", "right_elbow", "RightArm", "RightForeArm"),
    ("right_forearm", "right_elbow", "right_wrist", "RightForeArm", "RightHand"),
]

FOREARM_JOINTS = [
    ("left_forearm", "left_elbow", "left_wrist", "LeftForeArm", "LeftHand"),
    ("right_forearm", "right_elbow", "right_wrist", "RightForeArm", "RightHand"),
]

ARM_BONES_SMPL = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noitom-csv", type=Path, default=DEFAULT_NOITOM_CSV)
    parser.add_argument("--fit-script", type=Path, default=DEFAULT_FIT_SCRIPT)
    parser.add_argument("--body-model", type=Path, default=DEFAULT_BODY_MODEL)
    parser.add_argument("--betas-npz", type=Path, default=None)
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Source indices after stride/max filtering.")
    parser.add_argument("--num-auto-frames", type=int, default=4)
    parser.add_argument("--face-stride", type=int, default=3, help="Render every Nth mesh face for faster static images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--views", nargs="*", default=["front", "side"], choices=["front", "side", "top"])
    parser.add_argument("--closeup-sides", nargs="*", default=["left", "right"], choices=["left", "right"])
    return parser.parse_args()


def import_fit_module(path: Path):
    spec = importlib.util.spec_from_file_location("fit_noitom_smplh_diag", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def shortest_arc_rotmat(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_n = normalize(np.asarray(src, dtype=np.float64))
    dst_n = normalize(np.asarray(dst, dtype=np.float64))
    if src_n.ndim == 1:
        src_n = src_n[None]
    if dst_n.ndim == 1:
        dst_n = dst_n[None]

    out = np.empty((src_n.shape[0], 3, 3), dtype=np.float32)
    for idx, (a, b) in enumerate(zip(src_n, dst_n)):
        cross = np.cross(a, b)
        sin_angle = float(np.linalg.norm(cross))
        cos_angle = float(np.clip(np.dot(a, b), -1.0, 1.0))

        if sin_angle < 1e-8 and cos_angle > 0.0:
            out[idx] = np.eye(3, dtype=np.float32)
            continue

        if sin_angle < 1e-8:
            helper = np.zeros(3, dtype=np.float64)
            helper[int(np.argmin(np.abs(a)))] = 1.0
            axis = normalize(np.cross(a, helper))
            out[idx] = Rotation.from_rotvec(axis * math.pi).as_matrix().astype(np.float32)
            continue

        axis = cross / sin_angle
        angle = math.atan2(sin_angle, cos_angle)
        out[idx] = Rotation.from_rotvec(axis * angle).as_matrix().astype(np.float32)
    return out


def signed_twist_angle_deg(rel_rot: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis_n = normalize(np.asarray(axis, dtype=np.float64))
    quats_xyzw = Rotation.from_matrix(rel_rot).as_quat()
    vec = quats_xyzw[:, :3]
    w = quats_xyzw[:, 3]
    proj = np.sum(vec * axis_n, axis=1)
    angle = 2.0 * np.arctan2(np.abs(proj), np.maximum(np.abs(w), 1e-12))
    sign = np.sign(proj * w)
    signed = np.rad2deg(angle * np.where(sign == 0.0, 1.0, sign))
    return ((signed + 180.0) % 360.0) - 180.0


def build_arm_swing_global(
    direct_global: np.ndarray,
    noitom_positions: Dict[str, np.ndarray],
    rest_joints: np.ndarray,
    fit_module,
) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(fit_module.SMPL24_NAMES)}
    swing_global = direct_global.copy()

    for _, smpl_parent, smpl_child, noitom_parent, noitom_child in ARM_SEGMENTS:
        smpl_parent_idx = name_to_idx[smpl_parent]
        smpl_child_idx = name_to_idx[smpl_child]
        rest_dir = rest_joints[smpl_child_idx] - rest_joints[smpl_parent_idx]
        target_dir = noitom_positions[noitom_child] - noitom_positions[noitom_parent]
        swing_global[:, smpl_parent_idx] = shortest_arc_rotmat(
            np.repeat(rest_dir[None], target_dir.shape[0], axis=0),
            target_dir,
        )

    for side in ("left", "right"):
        elbow_idx = name_to_idx[f"{side}_elbow"]
        wrist_idx = name_to_idx[f"{side}_wrist"]
        hand_idx = name_to_idx[f"{side}_hand"]
        swing_global[:, wrist_idx] = swing_global[:, elbow_idx]
        swing_global[:, hand_idx] = swing_global[:, wrist_idx]

    return swing_global


def segment_direction_dots(
    smpl_joints: np.ndarray,
    noitom_positions: Dict[str, np.ndarray],
    fit_module,
) -> Dict[str, Dict[str, float]]:
    name_to_idx = {name: idx for idx, name in enumerate(fit_module.SMPL24_NAMES)}
    metrics: Dict[str, Dict[str, float]] = {}
    for label, smpl_parent, smpl_child, noitom_parent, noitom_child in ARM_SEGMENTS:
        smpl_dir = normalize(smpl_joints[:, name_to_idx[smpl_child]] - smpl_joints[:, name_to_idx[smpl_parent]])
        src_dir = normalize(noitom_positions[noitom_child] - noitom_positions[noitom_parent])
        dots = np.sum(smpl_dir * src_dir, axis=1)
        metrics[label] = {
            "mean": float(np.mean(dots)),
            "min": float(np.min(dots)),
            "max": float(np.max(dots)),
        }
    return metrics


def forearm_twist_metrics(
    direct_global: np.ndarray,
    noitom_positions: Dict[str, np.ndarray],
    rest_joints: np.ndarray,
    fit_module,
) -> Dict[str, np.ndarray]:
    name_to_idx = {name: idx for idx, name in enumerate(fit_module.SMPL24_NAMES)}
    out: Dict[str, np.ndarray] = {}
    for label, smpl_parent, smpl_child, noitom_parent, noitom_child in FOREARM_JOINTS:
        parent_idx = name_to_idx[smpl_parent]
        child_idx = name_to_idx[smpl_child]
        rest_dir = rest_joints[child_idx] - rest_joints[parent_idx]
        target_dir = noitom_positions[noitom_child] - noitom_positions[noitom_parent]
        swing = shortest_arc_rotmat(np.repeat(rest_dir[None], target_dir.shape[0], axis=0), target_dir)
        rel = np.matmul(np.swapaxes(swing, -1, -2), direct_global[:, parent_idx])
        out[label] = signed_twist_angle_deg(rel, np.repeat(normalize(rest_dir)[None], rel.shape[0], axis=0))
    return out


def load_rest_joints(fit_module, body_model_path: Path, betas: np.ndarray, device: torch.device, num_betas: int):
    body_model = fit_module.create_body_model(body_model_path, num_betas, device)
    with torch.no_grad():
        rest_out = body_model(
            pose_body=torch.zeros(1, 63, dtype=torch.float32, device=device),
            pose_hand=torch.zeros(1, 90, dtype=torch.float32, device=device),
            betas=torch.tensor(betas[None], dtype=torch.float32, device=device),
            root_orient=torch.zeros(1, 3, dtype=torch.float32, device=device),
            trans=torch.zeros(1, 3, dtype=torch.float32, device=device),
        )
    rest_joints = rest_out.Jtr[0].detach().cpu().numpy().astype(np.float32)
    return body_model, rest_joints


def forward_variant(
    fit_module,
    body_model,
    global_rotmats: np.ndarray,
    hips_pos: np.ndarray,
    rest_root: np.ndarray,
    betas: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root_orient, pose_body, _ = fit_module.smpl_pose_from_global_rotmats(global_rotmats)
    trans = fit_module.compute_trans_from_hips(hips_pos, root_orient, rest_root)
    vertices, joints = fit_module.forward_body_model(
        bm=body_model,
        pose_body=pose_body,
        root_orient=root_orient,
        trans=trans,
        betas=betas,
        device=device,
        batch_size=batch_size,
    )
    return vertices, joints, root_orient, pose_body


def select_frames(args: argparse.Namespace, frame_no: np.ndarray, twist: Dict[str, np.ndarray]) -> List[int]:
    if args.frames:
        frames = [idx for idx in args.frames if 0 <= idx < frame_no.shape[0]]
        if not frames:
            raise ValueError("--frames did not contain any valid source indices.")
        return frames

    score = np.maximum(np.abs(twist["left_forearm"]), np.abs(twist["right_forearm"]))
    candidates = [0, int(np.argmax(np.abs(twist["left_forearm"]))), int(np.argmax(np.abs(twist["right_forearm"]))), len(score) // 2]
    if len(score) > 4:
        candidates.append(int(np.argmax(score)))

    selected: List[int] = []
    for idx in candidates:
        if idx not in selected and 0 <= idx < len(score):
            selected.append(idx)
        if len(selected) >= args.num_auto_frames:
            break
    return selected


def to_plot_coords(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points)
    return arr[..., [0, 2, 1]]


def set_equal_axes(ax, points: np.ndarray) -> None:
    pts = to_plot_coords(points.reshape(-1, 3))
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins)) * 0.58, 0.25)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def set_axes_around_points(ax, points: np.ndarray, min_radius: float = 0.28) -> None:
    pts = to_plot_coords(points.reshape(-1, 3))
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins)) * 0.78, min_radius)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def draw_mesh(ax, vertices: np.ndarray, faces: np.ndarray, color: Tuple[float, float, float, float]) -> None:
    verts_plot = to_plot_coords(vertices)
    ax.plot_trisurf(
        verts_plot[:, 0],
        verts_plot[:, 1],
        verts_plot[:, 2],
        triangles=faces,
        color=color[:3],
        alpha=color[3],
        linewidth=0.0,
        shade=True,
    )


def draw_bones(
    ax,
    joints: np.ndarray,
    bones: Iterable[Tuple[str, str]],
    name_to_idx: Dict[str, int],
    color: str,
    label: str,
    linestyle: str = "-",
) -> None:
    first = True
    for parent, child in bones:
        if parent not in name_to_idx or child not in name_to_idx:
            continue
        p = to_plot_coords(joints[name_to_idx[parent]])
        c = to_plot_coords(joints[name_to_idx[child]])
        ax.plot(
            [p[0], c[0]],
            [p[1], c[1]],
            [p[2], c[2]],
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
            label=label if first else None,
        )
        first = False


def source_direction_segments_anchored(
    smpl_joints: np.ndarray,
    source_joints: Dict[str, np.ndarray],
    segment_map: Iterable[Tuple[str, str, str, str, str]],
    fit_module,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    smpl_name_to_idx = {name: idx for idx, name in enumerate(fit_module.SMPL24_NAMES)}
    segments = []
    for _, smpl_parent, smpl_child, noitom_parent, noitom_child in segment_map:
        if noitom_parent not in source_joints or noitom_child not in source_joints:
            continue
        parent_idx = smpl_name_to_idx[smpl_parent]
        child_idx = smpl_name_to_idx[smpl_child]
        anchor = smpl_joints[parent_idx]
        smpl_len = float(np.linalg.norm(smpl_joints[child_idx] - smpl_joints[parent_idx]))
        src_dir = normalize(source_joints[noitom_child] - source_joints[noitom_parent]).reshape(3)
        segments.append((anchor, anchor + src_dir * smpl_len))
    return segments


def draw_direction_segments(ax, segments: Sequence[Tuple[np.ndarray, np.ndarray]], color: str, label: str) -> None:
    for idx, (start, end) in enumerate(segments):
        p = to_plot_coords(start)
        c = to_plot_coords(end)
        ax.plot(
            [p[0], c[0]],
            [p[1], c[1]],
            [p[2], c[2]],
            color=color,
            linewidth=2.2,
            linestyle="--",
            label=label if idx == 0 else None,
        )


def render_comparison_frame(
    out_path: Path,
    frame_label: str,
    view: str,
    source_joints: Dict[str, np.ndarray],
    direct_vertices: np.ndarray,
    swing_vertices: np.ndarray,
    direct_joints: np.ndarray,
    swing_joints: np.ndarray,
    faces: np.ndarray,
    fit_module,
    left_twist: float,
    right_twist: float,
) -> None:
    views = {
        "front": (12.0, -90.0),
        "side": (12.0, 0.0),
        "top": (86.0, -90.0),
    }
    fig = plt.figure(figsize=(13.5, 6.2), dpi=140)
    smpl_name_to_idx = {name: idx for idx, name in enumerate(fit_module.SMPL24_NAMES)}
    all_points = np.concatenate(
        [
            direct_vertices,
            swing_vertices,
        ],
        axis=0,
    )

    panels = [
        ("Current direct Bone-Quat retarget", direct_vertices, direct_joints, (0.72, 0.72, 0.72, 0.78)),
        ("Arm swing/no-twist diagnostic", swing_vertices, swing_joints, (0.63, 0.72, 0.86, 0.78)),
    ]
    for panel_idx, (title, vertices, joints, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, panel_idx, projection="3d")
        draw_mesh(ax, vertices, faces, color)
        draw_bones(ax, joints, ARM_BONES_SMPL, smpl_name_to_idx, "#d62728", "SMPL arm")
        source_segments = source_direction_segments_anchored(joints, source_joints, ARM_SEGMENTS, fit_module)
        draw_direction_segments(ax, source_segments, "#1f77b4", "Noitom dir")
        set_equal_axes(ax, all_points)
        ax.view_init(elev=views[view][0], azim=views[view][1])
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        if panel_idx == 1:
            ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        f"{frame_label} | residual direct-vs-swing forearm twist: "
        f"L={left_twist:+.1f} deg, R={right_twist:+.1f} deg",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_arm_closeup(
    out_path: Path,
    frame_label: str,
    side: str,
    view: str,
    source_joints: Dict[str, np.ndarray],
    direct_vertices: np.ndarray,
    swing_vertices: np.ndarray,
    direct_joints: np.ndarray,
    swing_joints: np.ndarray,
    faces: np.ndarray,
    fit_module,
    twist_deg: float,
) -> None:
    views = {
        "front": (9.0, -90.0),
        "side": (9.0, 0.0),
        "top": (86.0, -90.0),
    }
    smpl_name_to_idx = {name: idx for idx, name in enumerate(fit_module.SMPL24_NAMES)}
    smpl_bones = [(f"{side}_shoulder", f"{side}_elbow"), (f"{side}_elbow", f"{side}_wrist")]
    side_segment_map = [seg for seg in ARM_SEGMENTS if seg[0].startswith(side)]

    focus_names = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
    focus_idx = [smpl_name_to_idx[name] for name in focus_names]

    fig = plt.figure(figsize=(13.5, 6.2), dpi=150)
    panels = [
        ("Current direct Bone-Quat retarget", direct_vertices, direct_joints, (0.72, 0.72, 0.72, 0.88)),
        ("Arm swing/no-twist diagnostic", swing_vertices, swing_joints, (0.63, 0.72, 0.86, 0.88)),
    ]
    for panel_idx, (title, vertices, joints, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, panel_idx, projection="3d")
        draw_mesh(ax, vertices, faces, color)
        draw_bones(ax, joints, smpl_bones, smpl_name_to_idx, "#d62728", "SMPL arm")
        source_segments = source_direction_segments_anchored(joints, source_joints, side_segment_map, fit_module)
        draw_direction_segments(ax, source_segments, "#1f77b4", "Noitom dir")
        source_points = np.concatenate([np.stack([start, end], axis=0) for start, end in source_segments], axis=0)
        focus_points = np.concatenate([joints[focus_idx], source_points], axis=0)
        set_axes_around_points(ax, focus_points)
        ax.view_init(elev=views[view][0], azim=views[view][1])
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        if panel_idx == 1:
            ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        f"{frame_label} | {side} arm close-up | direct-vs-swing forearm twist: {twist_deg:+.1f} deg",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_html_report(
    out_path: Path,
    image_paths: Sequence[Path],
    summary: Dict[str, Any],
) -> None:
    rel_images = [path.name for path in image_paths]
    rows = []
    for item in summary["selected_frames"]:
        rows.append(
            "<tr>"
            f"<td>{item['source_index']}</td>"
            f"<td>{item['frame_no']:.1f}</td>"
            f"<td>{item['left_forearm_twist_deg']:+.2f}</td>"
            f"<td>{item['right_forearm_twist_deg']:+.2f}</td>"
            "</tr>"
        )
    image_tags = "\n".join(
        f'<figure><img src="{img}" style="max-width: 100%; border: 1px solid #ccc;">'
        f"<figcaption>{img}</figcaption></figure>"
        for img in rel_images
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Noitom SMPL Retarget Diagnostics</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
    figure {{ margin: 18px 0 28px; }}
    figcaption {{ color: #666; font-size: 13px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Noitom SMPL Retarget Diagnostics</h1>
  <p>
    Left panel: current direct <code>Bone-Quat</code> retarget.
    Right panel: arm-only swing/no-twist diagnostic using <code>Joint-Posi</code> bone directions.
    The right panel is not a final retarget result; it is a visual test for rest-frame/twist mismatch.
  </p>
  <h2>Selected Frames</h2>
  <table>
    <tr><th>source index</th><th>Frame-No</th><th>left forearm twist deg</th><th>right forearm twist deg</th></tr>
    {''.join(rows)}
  </table>
  <h2>Direction Dot Summary</h2>
  <pre>{json.dumps(summary["direction_dot_current"], indent=2)}</pre>
  <h2>Images</h2>
  {image_tags}
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fit_module = import_fit_module(args.fit_script)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    frame_no, positions, bone_quats = fit_module.read_noitom_calc(args.noitom_csv)
    frame_no, positions, bone_quats = fit_module.subsample_data(
        frame_no=frame_no,
        positions=positions,
        bone_quats=bone_quats,
        frame_stride=max(1, args.frame_stride),
        max_frames=args.max_frames,
    )

    betas = fit_module.load_betas(args.betas_npz, args.num_betas)
    body_model, rest_joints_full = load_rest_joints(fit_module, args.body_model, betas, device, args.num_betas)
    rest_joints = rest_joints_full[: len(fit_module.SMPL24_NAMES)]

    direct_global = fit_module.build_smpl24_global_rotmats(bone_quats)
    swing_global = build_arm_swing_global(direct_global, positions, rest_joints, fit_module)

    twist = forearm_twist_metrics(direct_global, positions, rest_joints, fit_module)
    selected = select_frames(args, frame_no, twist)

    direct_vertices, direct_joints, _, _ = forward_variant(
        fit_module,
        body_model,
        direct_global[selected],
        positions["Hips"][selected],
        rest_joints_full[0],
        betas,
        device,
    )
    swing_vertices, swing_joints, _, _ = forward_variant(
        fit_module,
        body_model,
        swing_global[selected],
        positions["Hips"][selected],
        rest_joints_full[0],
        betas,
        device,
    )

    all_direct_vertices, all_direct_joints, _, _ = forward_variant(
        fit_module,
        body_model,
        direct_global,
        positions["Hips"],
        rest_joints_full[0],
        betas,
        device,
    )
    direction_metrics = segment_direction_dots(
        all_direct_joints[:, : len(fit_module.SMPL24_NAMES)],
        positions,
        fit_module,
    )
    del all_direct_vertices

    faces = body_model.f.detach().cpu().numpy().astype(np.int32) if torch.is_tensor(body_model.f) else np.asarray(body_model.f, dtype=np.int32)
    faces = faces[:: max(1, args.face_stride)]

    image_paths: List[Path] = []
    selected_summary: List[Dict[str, Any]] = []
    noitom_joint_names = sorted({name for _, _, _, parent, child in ARM_SEGMENTS for name in (parent, child)})
    for out_idx, src_idx in enumerate(selected):
        selected_summary.append(
            {
                "source_index": int(src_idx),
                "frame_no": float(frame_no[src_idx]),
                "left_forearm_twist_deg": float(twist["left_forearm"][src_idx]),
                "right_forearm_twist_deg": float(twist["right_forearm"][src_idx]),
            }
        )
        source_joints = {name: positions[name][src_idx] for name in noitom_joint_names if name in positions}
        for view in args.views:
            image_path = args.output_dir / f"frame_{src_idx:05d}_{view}.png"
            render_comparison_frame(
                out_path=image_path,
                frame_label=f"source index {src_idx}, Frame-No {float(frame_no[src_idx]):.1f}",
                view=view,
                source_joints=source_joints,
                direct_vertices=direct_vertices[out_idx],
                swing_vertices=swing_vertices[out_idx],
                direct_joints=direct_joints[out_idx, : len(fit_module.SMPL24_NAMES)],
                swing_joints=swing_joints[out_idx, : len(fit_module.SMPL24_NAMES)],
                faces=faces,
                fit_module=fit_module,
                left_twist=float(twist["left_forearm"][src_idx]),
                right_twist=float(twist["right_forearm"][src_idx]),
            )
            image_paths.append(image_path)
            for side in args.closeup_sides:
                closeup_path = args.output_dir / f"frame_{src_idx:05d}_{side}_arm_{view}.png"
                twist_key = f"{side}_forearm"
                render_arm_closeup(
                    out_path=closeup_path,
                    frame_label=f"source index {src_idx}, Frame-No {float(frame_no[src_idx]):.1f}",
                    side=side,
                    view=view,
                    source_joints=source_joints,
                    direct_vertices=direct_vertices[out_idx],
                    swing_vertices=swing_vertices[out_idx],
                    direct_joints=direct_joints[out_idx, : len(fit_module.SMPL24_NAMES)],
                    swing_joints=swing_joints[out_idx, : len(fit_module.SMPL24_NAMES)],
                    faces=faces,
                    fit_module=fit_module,
                    twist_deg=float(twist[twist_key][src_idx]),
                )
                image_paths.append(closeup_path)

    summary = {
        "noitom_csv": str(args.noitom_csv),
        "fit_script": str(args.fit_script),
        "body_model": str(args.body_model),
        "frame_stride": int(args.frame_stride),
        "max_frames": None if args.max_frames is None else int(args.max_frames),
        "selected_frames": selected_summary,
        "direction_dot_current": direction_metrics,
        "twist_summary_deg": {
            key: {
                "mean_abs": float(np.mean(np.abs(value))),
                "max_abs": float(np.max(np.abs(value))),
                "median": float(np.median(value)),
            }
            for key, value in twist.items()
        },
        "output_images": [str(path) for path in image_paths],
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = args.output_dir / "retarget_diagnostics.html"
    write_html_report(report_path, image_paths, summary)

    print(f"[diag] Loaded frames: {len(frame_no)}")
    print(f"[diag] Selected source indices: {selected}")
    print(f"[diag] Direction dot current: {json.dumps(direction_metrics, indent=2)}")
    print(f"[diag] Twist summary deg: {json.dumps(summary['twist_summary_deg'], indent=2)}")
    print(f"[diag] Wrote summary: {summary_path}")
    print(f"[diag] Wrote report: {report_path}")


if __name__ == "__main__":
    main()
