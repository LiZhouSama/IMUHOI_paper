#!/usr/bin/env python3
"""Oracle upper bound of the FK object-translation estimator.

The FK hypothesis: at the instant a hand starts touching the object, the vector
from the palm anchor to the object centre is frozen in the *object* frame
(direction + length).  Inside the contact segment the object centre is then
recovered as a virtual bone attached to the palm:

    p_obj(t) = p_hand(t) + R_obj(t) @ dir_obj_frame(t0) * length(t0)

This script feeds that estimator ground-truth palm positions, ground-truth
object rotations and ground-truth hand-contact labels, so the only remaining
error source is the frozen-bone assumption itself (plus the static hold used
when neither hand touches the object).  The resulting obj_err / hoi_err are the
performance ceiling of the FK branch in model/rnn.

Metric definitions match eval_IMUHOI.py:
  obj_err : mean_t ||p_pred(t) - p_gt(t)|| over the whole sequence, in cm
  hoi_err : compute_hoi_error_at_gt_contact(), i.e. the error of the
            hand-relative object offset on GT contact frames, in cm
Both are averaged per sequence first, then across sequences (np.mean over the
non-NaN values), exactly like evaluate_model().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_IMUHOI import find_contact_segments  # noqa: E402
from utils.hoi_metrics import compute_hoi_error_at_gt_contact  # noqa: E402
from utils.human_pose import select_hand_anchor_positions  # noqa: E402

DEFAULT_DATA_DIR = "process/IMUHOI++/processed_split_data_OMOMO_smplx_full/test"


def _load_sequence(path: Path) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    joints = data["position_global_full_gt_world"][..., :24, :].float()
    return {
        "seq_name": str(data.get("seq_name", path.stem)),
        "obj_name": str(data.get("obj_name", "")),
        "has_object": bool(data.get("has_object", True)),
        "hand_pos": select_hand_anchor_positions(joints),      # [T, 2, 3]
        "obj_trans": data["obj_trans"].float(),                # [T, 3]
        "obj_rot": data["obj_rot"].float(),                    # [T, 3, 3]
        "lhand_contact": data["lhand_contact"].bool(),         # [T]
        "rhand_contact": data["rhand_contact"].bool(),         # [T]
    }


def fk_per_hand(
    hand_pos: torch.Tensor,
    obj_rot: torch.Tensor,
    obj_trans: torch.Tensor,
    contact: torch.Tensor,
    anchor: str = "segment_start",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FK object position for one hand, frozen at each contact segment's first frame.

    Returns (positions [T, 3], valid mask [T]); positions are undefined where
    the mask is False.
    """
    seq_len = obj_trans.shape[0]
    positions = torch.zeros(seq_len, 3, dtype=obj_trans.dtype)
    valid = torch.zeros(seq_len, dtype=torch.bool)

    for seg_start, seg_end in find_contact_segments(contact):
        idx = torch.arange(seg_start, seg_end + 1)
        if anchor == "per_frame":
            # Oracle re-anchoring every frame: exactly reproduces the GT.
            positions[idx] = obj_trans[idx]
            valid[idx] = True
            continue
        offset = obj_trans[seg_start] - hand_pos[seg_start]          # world vector
        length = torch.linalg.vector_norm(offset)
        direction_world = offset / (length + 1e-8)
        # Freeze the bone in the object frame: ^O v = ^W R_O^T ^W v
        direction_obj = obj_rot[seg_start].transpose(-1, -2) @ direction_world
        direction_seg = (obj_rot[idx] @ direction_obj.unsqueeze(-1)).squeeze(-1)
        positions[idx] = hand_pos[idx] + direction_seg * length
        valid[idx] = True
    return positions, valid


def fk_trajectory(
    hand_pos: torch.Tensor,
    obj_rot: torch.Tensor,
    obj_trans: torch.Tensor,
    lhand_contact: torch.Tensor,
    rhand_contact: torch.Tensor,
    anchor: str = "segment_start",
) -> torch.Tensor:
    """Fuse both hands (average when both touch) and hold static when neither does."""
    seq_len = obj_trans.shape[0]
    l_pos, l_valid = fk_per_hand(hand_pos[:, 0], obj_rot, obj_trans, lhand_contact, anchor)
    r_pos, r_valid = fk_per_hand(hand_pos[:, 1], obj_rot, obj_trans, rhand_contact, anchor)

    pred = torch.zeros(seq_len, 3, dtype=obj_trans.dtype)
    # Same initial state as the RNN predictor: obj_trans_init = GT frame 0.
    previous = obj_trans[0].clone()
    for t in range(seq_len):
        if bool(l_valid[t]) and bool(r_valid[t]):
            current = 0.5 * (l_pos[t] + r_pos[t])
        elif bool(l_valid[t]):
            current = l_pos[t]
        elif bool(r_valid[t]):
            current = r_pos[t]
        else:
            current = previous
        pred[t] = current
        previous = current
    return pred


def _mean_cm(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    if mask is not None:
        if not bool(mask.any()):
            return float("nan")
        values = values[mask]
    if values.numel() == 0:
        return float("nan")
    return float(values.mean().item() * 100.0)


def evaluate_sequence(sample: Dict[str, torch.Tensor], anchor: str) -> Dict[str, float]:
    obj_trans = sample["obj_trans"]
    hand_pos = sample["hand_pos"]
    lhand_contact = sample["lhand_contact"]
    rhand_contact = sample["rhand_contact"]

    pred = fk_trajectory(
        hand_pos, sample["obj_rot"], obj_trans, lhand_contact, rhand_contact, anchor
    )
    frame_error = torch.linalg.norm(pred - obj_trans, dim=-1)
    any_contact = lhand_contact | rhand_contact
    both_contact = lhand_contact & rhand_contact

    return {
        "seq_name": sample["seq_name"],
        "obj_name": sample["obj_name"],
        "frames": int(obj_trans.shape[0]),
        "contact_frames": int(any_contact.sum().item()),
        "both_hand_frames": int(both_contact.sum().item()),
        "n_segments": len(find_contact_segments(lhand_contact))
        + len(find_contact_segments(rhand_contact)),
        "obj_err": _mean_cm(frame_error),
        "obj_err_on_contact": _mean_cm(frame_error, any_contact),
        "obj_err_off_contact": _mean_cm(frame_error, ~any_contact),
        "hoi_err": compute_hoi_error_at_gt_contact(
            pred, obj_trans, hand_pos, hand_pos, lhand_contact, rhand_contact
        ),
        "_frame_error": frame_error,
        "_any_contact": any_contact,
    }


def _summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "count": 0}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "count": int(array.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N sequences.")
    parser.add_argument(
        "--anchor",
        default="segment_start",
        choices=["segment_start", "per_frame"],
        help="segment_start = FK upper bound; per_frame = static-hold-only reference.",
    )
    parser.add_argument("--json-out", default="", help="Optional per-sequence dump.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    files = sorted(data_dir.glob("*.pt"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .pt sequences found under {data_dir}")

    records: List[Dict[str, float]] = []
    all_frame_errors: List[torch.Tensor] = []
    all_contact_errors: List[torch.Tensor] = []
    for path in files:
        sample = _load_sequence(path)
        if not sample["has_object"]:
            continue
        record = evaluate_sequence(sample, args.anchor)
        all_frame_errors.append(record.pop("_frame_error"))
        contact_mask = record.pop("_any_contact")
        all_contact_errors.append(all_frame_errors[-1][contact_mask])
        records.append(record)

    obj_summary = _summarize([r["obj_err"] for r in records])
    hoi_summary = _summarize([r["hoi_err"] for r in records])
    on_summary = _summarize([r["obj_err_on_contact"] for r in records])
    off_summary = _summarize([r["obj_err_off_contact"] for r in records])

    frame_errors = torch.cat(all_frame_errors) if all_frame_errors else torch.zeros(0)
    contact_errors = torch.cat(all_contact_errors) if all_contact_errors else torch.zeros(0)
    total_frames = sum(r["frames"] for r in records)
    total_contact = sum(r["contact_frames"] for r in records)

    print(f"\nData dir : {data_dir}")
    print(f"Anchor   : {args.anchor}")
    print(f"Sequences: {len(records)} | frames: {total_frames} | "
          f"contact frames: {total_contact} ({100.0 * total_contact / max(total_frames, 1):.1f}%)")

    print("\n--- FK oracle upper bound (per-sequence mean, then averaged; cm) ---")
    print(f"obj_err                : {obj_summary['mean']:.4f}  "
          f"(median {obj_summary['median']:.4f}, p90 {obj_summary['p90']:.4f}, n={obj_summary['count']})")
    print(f"hoi_err                : {hoi_summary['mean']:.4f}  "
          f"(median {hoi_summary['median']:.4f}, p90 {hoi_summary['p90']:.4f}, n={hoi_summary['count']})")
    print(f"obj_err  (contact only): {on_summary['mean']:.4f}  "
          f"(median {on_summary['median']:.4f}, n={on_summary['count']})")
    print(f"obj_err  (no contact)  : {off_summary['mean']:.4f}  "
          f"(median {off_summary['median']:.4f}, n={off_summary['count']})")

    print("\n--- Frame-weighted over the whole test split (cm) ---")
    print(f"obj_err                : {frame_errors.mean().item() * 100.0:.4f}")
    if contact_errors.numel():
        print(f"obj_err  (contact only): {contact_errors.mean().item() * 100.0:.4f}")

    worst = sorted(records, key=lambda r: (-r["obj_err"] if not np.isnan(r["obj_err"]) else 0.0))[:10]
    print("\n--- Worst 10 sequences by obj_err (cm) ---")
    for record in worst:
        print(f"  {record['seq_name']:<28} obj {record['obj_err']:8.2f}  hoi {record['hoi_err']:8.2f}  "
              f"frames {record['frames']:4d}  contact {record['contact_frames']:4d}  segs {record['n_segments']:2d}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "data_dir": str(data_dir),
            "anchor": args.anchor,
            "summary": {
                "obj_err": obj_summary,
                "hoi_err": hoi_summary,
                "obj_err_on_contact": on_summary,
                "obj_err_off_contact": off_summary,
                "obj_err_frame_weighted": float(frame_errors.mean().item() * 100.0),
            },
            "sequences": records,
        }, indent=2))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
