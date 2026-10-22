"""Plot per-sequence obj_move timelines to audit the obj_move labels.

For each processed .pt sequence the figure shows, against the frame index:
  1. windowed net object translation (cm) with the obj_move threshold;
  2. windowed net object rotation (deg) with its threshold;
  3. label strips: final obj_move (obj_contact), the pre-cleanup obj_move
     (obj_contact_raw), and left / right hand contact.
Frames where the final obj_move is on are shaded in panels 1-2.  The motion
statistics are recomputed with process.preprocess.compute_object_motion_stats,
i.e. exactly what the label was derived from.

A summary CSV (``obj_move_summary.csv``) lists per sequence the moving fraction,
frames turned on by hole filling (label on, motion below both thresholds) and
frames turned off as short episodes (motion above a threshold, label off), which
is a quick way to pick sequences worth opening.

Usage:
    python scripts/diag/plot_obj_move_timeline.py \
        process/IMUHOI++cano/processed_split_data_BEHAVE/test \
        --out-dir outputs/analysis/obj_move_timeline/BEHAVE_test --limit 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from configs import FRAME_RATE  # noqa: E402
from process.preprocess import (  # noqa: E402
    OBJ_MOTION_ROT_THRESHOLD_DEG,
    OBJ_MOTION_TRANS_THRESHOLD_M,
    compute_object_motion_stats,
)

SERIES_COLOR = "#2a78d6"
MOVING_COLOR = "#eb6834"
HAND_COLOR = "#52514e"
THRESHOLD_COLOR = "#8a8984"
TEXT_SECONDARY = "#52514e"


def _bool(data, key, length):
    value = data.get(key)
    if value is None:
        return None
    return torch.as_tensor(value).reshape(-1)[:length].bool().numpy()


def _runs(mask):
    """(start, length) runs of True in a bool array, for broken_barh."""
    runs, start = [], None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


def plot_sequence(path: Path, out_dir: Path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not bool(data.get("has_object", True)):
        return None
    net_trans, net_rot = compute_object_motion_stats(data["obj_trans"], data["obj_rot"])
    num_frames = net_trans.shape[0]
    obj_move = _bool(data, "obj_contact", num_frames)
    obj_move_raw = _bool(data, "obj_contact_raw", num_frames)
    lhand = _bool(data, "lhand_contact", num_frames)
    rhand = _bool(data, "rhand_contact", num_frames)
    if obj_move is None:
        return None

    above = (net_trans > OBJ_MOTION_TRANS_THRESHOLD_M) | (net_rot > OBJ_MOTION_ROT_THRESHOLD_DEG)
    frames = np.arange(num_frames)
    seq_name = str(data.get("seq_name", path.stem))

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.2, 1.6], "hspace": 0.12},
    )
    panels = (
        (axes[0], net_trans * 100.0, OBJ_MOTION_TRANS_THRESHOLD_M * 100.0, "net translation (cm)"),
        (axes[1], net_rot, OBJ_MOTION_ROT_THRESHOLD_DEG, "net rotation (deg)"),
    )
    for axis, values, threshold, label in panels:
        for start, length in _runs(obj_move):
            axis.axvspan(start - 0.5, start + length - 0.5, color=MOVING_COLOR, alpha=0.13, lw=0)
        axis.plot(frames, values, color=SERIES_COLOR, lw=1.2)
        axis.axhline(threshold, color=THRESHOLD_COLOR, lw=1.0, ls="--")
        axis.text(
            num_frames - 1, threshold, f" threshold {threshold:g}", color=TEXT_SECONDARY,
            fontsize=8, va="bottom", ha="right",
        )
        # Clip the y range so the threshold region stays readable during large motions.
        upper = max(threshold * 6.0, float(np.percentile(values, 90)) * 1.1)
        axis.set_ylim(0.0, upper)
        axis.set_ylabel(label, color=TEXT_SECONDARY, fontsize=9)
        axis.grid(axis="y", color="#e6e5e0", lw=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    strip_axis = axes[2]
    rows = [
        ("obj_move", obj_move, MOVING_COLOR),
        ("obj_move raw", obj_move_raw, MOVING_COLOR),
        ("L hand", lhand, HAND_COLOR),
        ("R hand", rhand, HAND_COLOR),
    ]
    rows = [row for row in rows if row[1] is not None]
    for row_index, (name, mask, color) in enumerate(rows):
        y = len(rows) - 1 - row_index
        strip_axis.broken_barh(
            [(start - 0.5, length) for start, length in _runs(mask)], (y - 0.35, 0.7),
            facecolors=color, alpha=0.55 if name == "obj_move raw" else 0.9, lw=0,
        )
    strip_axis.set_yticks(range(len(rows)))
    strip_axis.set_yticklabels([row[0] for row in reversed(rows)], fontsize=9, color=TEXT_SECONDARY)
    strip_axis.set_ylim(-0.6, len(rows) - 0.4)
    strip_axis.set_xlim(-0.5, num_frames - 0.5)
    strip_axis.set_xlabel(f"frame ({FRAME_RATE} fps)", color=TEXT_SECONDARY, fontsize=9)
    for side in ("top", "right", "left"):
        strip_axis.spines[side].set_visible(False)

    hole_filled = int((obj_move & ~above).sum())
    dropped = int((~obj_move & above).sum())
    axes[0].set_title(
        f"{seq_name}   obj_move {obj_move.mean():.1%} of {num_frames} frames   "
        f"hole-filled {hole_filled}   dropped-short {dropped}",
        loc="left", fontsize=10,
    )
    out_path = out_dir / f"{seq_name}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {
        "seq_name": seq_name,
        "frames": num_frames,
        "obj_move_frac": round(float(obj_move.mean()), 4),
        "hole_filled_frames": hole_filled,
        "dropped_short_frames": dropped,
        "hand_contact_frac": round(float((lhand | rhand).mean()), 4) if lhand is not None and rhand is not None else "",
        "png": out_path.name,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help=".pt files or directories of .pt files")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Plot at most this many sequences")
    args = parser.parse_args()

    paths = []
    for item in args.inputs:
        item_path = Path(item)
        paths.extend(sorted(item_path.glob("*.pt")) if item_path.is_dir() else [item_path])
    if args.limit is not None:
        paths = paths[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, path in enumerate(paths, 1):
        row = plot_sequence(path, out_dir)
        if row is not None:
            rows.append(row)
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
    if rows:
        with open(out_dir / "obj_move_summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(rows)} figures and obj_move_summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
