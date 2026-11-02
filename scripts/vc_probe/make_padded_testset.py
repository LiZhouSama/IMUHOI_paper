#!/usr/bin/env python3
"""Copy test sequences with a static prefix: PAD frames that repeat the first frame (body pose, root translation,
object pose), no hand / object contact, then the real sequence.

Only sequences without hand contact at frame 1 are used (frame 0 is never labelled as contact). Every derived
field (synthetic IMU, velocities, hand-object directions, endpoint labels) is rebuilt from the padded raw fields
with ``process.preprocess.precompute_sequence_features``, which reproduces the stored fields of the original
sequences to 2e-4. The original data are not modified.

    python scripts/vc_probe/make_padded_testset.py --src process/IMUHOI++cano/processed_split_data_OMOMO_full/test \
        --dst outputs/IMUHOI++/RNN_cano/vc_probe_pilot_20261027/omomo_pad90_data/test --pad 90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from process.preprocess import precompute_sequence_features  # noqa: E402

REPEAT_FIRST = ("position_global_full_gt_world", "rotation_global", "trans", "pose", "obj_trans", "obj_rot",
                "obj_scale", "lfoot_contact", "rfoot_contact")
NON_TEMPORAL = ("obj_points_canonical",)  # 256 object points, not frames
NO_CONTACT = ("lhand_contact", "rhand_contact", "obj_contact", "lhand_contact_raw", "rhand_contact_raw", "obj_contact_raw")


def pad_sequence(d, pad):
    T = int(d["trans"].shape[0])
    out = dict(d)
    for key in REPEAT_FIRST + NO_CONTACT:
        value = torch.as_tensor(d[key])
        if value.shape[0] != T:
            raise ValueError(f"{key}: {tuple(value.shape)} is not per-frame")
        prefix = value[:1].repeat(pad, *([1] * (value.dim() - 1)))
        if key in NO_CONTACT:
            prefix = torch.zeros_like(prefix)
        out[key] = torch.cat([prefix, value], dim=0)
    feats = precompute_sequence_features(
        out["position_global_full_gt_world"], out["rotation_global"], out["trans"], out["obj_trans"], out["obj_rot"],
        lhand_contact=out["lhand_contact"], rhand_contact=out["rhand_contact"], obj_contact=out["obj_contact"],
        has_object=d["has_object"])
    out.update(feats)
    leftover = [k for k, v in out.items() if hasattr(v, "shape") and len(v.shape) and v.shape[0] == T
                and k not in NON_TEMPORAL]
    if leftover:
        raise ValueError(f"per-frame fields not padded: {leftover}")
    out["pad_frames"] = int(pad)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--pad", type=int, default=90)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    for path in sorted(args.src.glob("*.pt")):
        d = torch.load(path, map_location="cpu", weights_only=False)
        if bool(d["lhand_contact"][1]) or bool(d["rhand_contact"][1]):
            skipped += 1
            continue
        torch.save(pad_sequence(d, args.pad), args.dst / path.name)
        kept += 1
    print(f"padded {kept} sequences by {args.pad} frames into {args.dst}; skipped {skipped} in contact at frame 1")


if __name__ == "__main__":
    main()
