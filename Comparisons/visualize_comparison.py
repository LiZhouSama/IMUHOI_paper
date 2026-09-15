"""Render a compact GT/predicted skeleton diagnostic from eval predictions.

The evaluator stores body-model joints in metres.  This script deliberately
uses the same 24-joint parent table as the metric code, so a malformed output
is visible immediately instead of being hidden by a scalar error report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils.human_pose import SMPL_PARENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, help="Serialized sequence .pt")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument(
        "--frames",
        default="0,33,66,99",
        help="Comma-separated frame indices (up to 6)",
    )
    parser.add_argument("--max-joints", type=int, default=22)
    return parser.parse_args()


def _load(path: str) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = ("joints", "target_joints")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"{path} does not contain {missing}; rerun eval with --save_predictions"
        )
    return payload


def _frame_indices(spec: str, length: int) -> list[int]:
    values = []
    for token in spec.split(","):
        if not token.strip():
            continue
        index = int(token)
        if index < 0 or index >= length:
            raise ValueError(f"frame {index} is outside [0, {length})")
        if index not in values:
            values.append(index)
    if not values:
        raise ValueError("--frames must contain at least one frame")
    return values[:6]


def _plot_skeleton(ax, joints: torch.Tensor, color: str, label: str) -> None:
    xyz = joints.numpy()
    ax.scatter(xyz[:, 0], xyz[:, 2], xyz[:, 1], s=8, c=color, alpha=0.9, label=label)
    for child, parent in enumerate(SMPL_PARENTS[: xyz.shape[0]]):
        if parent < 0 or parent >= xyz.shape[0]:
            continue
        segment = xyz[[parent, child]]
        ax.plot(segment[:, 0], segment[:, 2], segment[:, 1], color=color, linewidth=1.2)


def main() -> None:
    args = parse_args()
    payload = _load(args.prediction)
    pred = payload["joints"].float()
    target = payload["target_joints"].float()
    length = min(pred.shape[0], target.shape[0])
    pred, target = pred[:length], target[:length]
    frames = _frame_indices(args.frames, length)
    joint_count = min(int(args.max_joints), pred.shape[1], target.shape[1], 24)
    pred, target = pred[:, :joint_count], target[:, :joint_count]

    # Put all panels in a common root-relative, equal-aspect coordinate frame.
    all_points = torch.cat((pred[frames] - pred[frames, :1], target[frames] - target[frames, :1]))
    limits = all_points.reshape(-1, 3)
    center = (limits.min(dim=0).values + limits.max(dim=0).values) * 0.5
    radius = (limits.max(dim=0).values - limits.min(dim=0).values).max().item() * 0.58
    radius = max(radius, 0.25)

    fig = plt.figure(figsize=(4.0 * len(frames), 4.2), constrained_layout=True)
    for panel, frame in enumerate(frames, start=1):
        ax = fig.add_subplot(1, len(frames), panel, projection="3d")
        pred_frame = pred[frame] - pred[frame, :1]
        target_frame = target[frame] - target[frame, :1]
        _plot_skeleton(ax, target_frame, "#2563eb", "GT")
        _plot_skeleton(ax, pred_frame, "#dc2626", "pred")
        ax.set_title(f"frame {frame}")
        ax.set_xlim(center[0].item() - radius, center[0].item() + radius)
        ax.set_ylim(center[2].item() - radius, center[2].item() + radius)
        ax.set_zlim(center[1].item() - radius, center[1].item() + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.view_init(elev=16, azim=-70)
        if panel == 1:
            ax.legend(loc="upper left", fontsize=8)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved skeleton diagnostic to {output}")


if __name__ == "__main__":
    main()
