"""Measure whether the contact labels support the conditional factorization.

VelocityContact predicts ``P(hand contact) = P(object moving) *
P(hand contact | object moving)``.  That product is only well posed if hand
contact implies object motion in the labels.  ``postprocess_contact_labels``
runs its hole-filling and minimum-segment cleanup on the three channels
independently, so the cleanup can produce frames where a hand is labeled as
contacting while the object channel is off.  On those frames the conditional
BCE (pushing the product toward 1) and the object-move BCE (pushing
``P(object moving)`` toward 0) pull against each other.

Run this before trusting a VC ablation that depends on the factorization::

    python scripts/check_contact_label_consistency.py \
        process/IMUHOI++/processed_split_data_OMOMO/train

Reports the violating-frame rate overall and the worst sequences.  A rate that
is not negligible means the cleanup should re-impose ``hand <= obj`` after
postprocessing, or the product should be replaced by a fusion with a residual.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


CONTACT_KEYS = ("lhand_contact", "rhand_contact", "obj_contact")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_dirs",
        nargs="+",
        type=Path,
        help="Processed sequence directories (each holding *.pt sequences).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many worst sequences to list.",
    )
    return parser.parse_args()


def load_contacts(path: Path):
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        return None
    has_object = data.get("has_object")
    if has_object is not None and not bool(has_object):
        return None
    channels = []
    for key in CONTACT_KEYS:
        value = data.get(key)
        if not isinstance(value, torch.Tensor):
            return None
        channels.append(value.reshape(-1).bool())
    if len({int(channel.shape[0]) for channel in channels}) != 1:
        return None
    return channels


def main():
    args = parse_args()
    sequences = sorted(
        path
        for data_dir in args.data_dirs
        for path in Path(data_dir).rglob("*.pt")
    )
    if not sequences:
        raise SystemExit(f"No .pt sequences found under {args.data_dirs}")

    total_frames = 0
    total_violations = 0
    total_hand_frames = 0
    total_obj_frames = 0
    skipped = 0
    per_sequence = []

    for path in sequences:
        channels = load_contacts(path)
        if channels is None:
            skipped += 1
            continue
        lhand, rhand, obj = channels
        hand = lhand | rhand
        violations = int((hand & ~obj).sum())
        frames = int(hand.shape[0])
        total_frames += frames
        total_violations += violations
        total_hand_frames += int(hand.sum())
        total_obj_frames += int(obj.sum())
        if violations:
            per_sequence.append((violations / max(frames, 1), violations, frames, path))

    print(f"sequences scanned : {len(sequences) - skipped} (skipped {skipped})")
    print(f"frames            : {total_frames}")
    print(f"object-move frames: {total_obj_frames} "
          f"({total_obj_frames / max(total_frames, 1):.2%})")
    print(f"hand-contact frames: {total_hand_frames} "
          f"({total_hand_frames / max(total_frames, 1):.2%})")
    print(
        "hand & ~obj frames : "
        f"{total_violations} ({total_violations / max(total_frames, 1):.4%} of all "
        f"frames, {total_violations / max(total_hand_frames, 1):.4%} of hand frames)"
    )
    if not per_sequence:
        print("\nNo violations: the conditional factorization is consistent.")
        return

    per_sequence.sort(reverse=True)
    print(f"\nworst {min(args.top, len(per_sequence))} sequences:")
    for rate, violations, frames, path in per_sequence[: args.top]:
        print(f"  {rate:7.2%}  {violations:6d}/{frames:6d}  {path}")


if __name__ == "__main__":
    main()
