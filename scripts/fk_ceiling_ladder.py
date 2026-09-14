#!/usr/bin/env python3
"""Oracle ceiling of the rigid-attachment (FK) assumption, and the anchor-point comparison.

Everything here is measured with ground-truth joints, ground-truth object pose and
ground-truth contact labels, so the only error left is the *modelling assumption*.
Errors are per-hand, in-contact, frame-weighted, in cm.

Anchor points compared (the "contact point" the virtual bone is attached to):
  wrist       SMPL joint 20/21
  palm        SMPL-24 slot 22/23 = SMPL-X middle1 (middle-finger MCP), the anchor the pipeline uses
  palm_rigid  the palm anchor recomputed with the wrist rotation locked to its segment-start value
              (what a 6-IMU system can actually observe: the rotation sensors sit on the elbows)
  best_pivot  the per-segment least-squares optimal rigid attachment point

Assumption ladder (all with the palm anchor unless stated):
  L0 frozen first-frame bone                      what the v1/v2 FK branch assumes
  L1 best constant bone in the segment            oracle anchor, still no residual
  L2 frozen anchor + per-segment pivot residual    pivot expressed in the observable elbow frame
  L3 best (anchor, pivot), elbow frame             6 params, fully observable rotation source
  L4 best (anchor, pivot), wrist frame             6 params, needs unobservable wrist rotation
  L5 L3 re-estimated every K frames                how much of the residual is slow drift
  L6 per-frame re-anchoring                        exactly 0 by construction

Usage:  python scripts/fk_ceiling_ladder.py [--data DIR] [--json OUT.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_IMUHOI import find_contact_segments  # noqa: E402

DEFAULT_DATA = "process/IMUHOI++/processed_split_data_OMOMO_full/test"
WRIST = (20, 21)
PALM = (22, 23)
ELBOW = (18, 19)
MIN_SEG = 10


def bone_in_object_frame(obj_rot, obj_trans, anchor):
    """c(t) = R_obj(t)^T (p_obj(t) - p_anchor(t)); the "virtual bone" the FK assumption freezes."""
    return torch.einsum("tji,tj->ti", obj_rot, obj_trans - anchor)


def lstsq3(A, b):
    return torch.linalg.lstsq(A.reshape(-1, 3).double(), b.reshape(-1, 1).double()).solution.squeeze(-1).float()


def fit_anchor_pivot(c, M):
    """Solve min_{a,v} || c(t) - a + (M(t)-I) v ||^2 -> 6 parameters."""
    n = c.shape[0]
    eye = torch.eye(3).expand(n, 3, 3)
    design = torch.cat((eye, -(M - eye)), dim=2).reshape(3 * n, 6).double()
    target = c.reshape(3 * n, 1).double()
    sol = torch.linalg.lstsq(design, target).solution.squeeze(-1).float()
    resid = (design.float() @ sol.unsqueeze(-1) - target.float()).reshape(n, 3)
    return sol[:3], sol[3:], resid.norm(dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--refit_every", type=int, default=30)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    acc = defaultdict(list)
    weights = []
    per_object = defaultdict(lambda: defaultdict(list))
    seg_meta = []

    files = sorted(glob.glob(str(PROJECT_ROOT / args.data / "*.pt")))
    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not data.get("has_object", True):
            continue
        joints = data["position_global_full_gt_world"][..., :24, :].float()
        rot_glb = data["rotation_global"].float()
        obj_rot = data["obj_rot"].float()
        obj_trans = data["obj_trans"].float()
        obj_name = str(data.get("obj_name", ""))

        for hand, key in ((0, "lhand_contact"), (1, "rhand_contact")):
            contact = data[key].bool()
            for start, end in find_contact_segments(contact):
                n = end - start + 1
                if n < MIN_SEG:
                    continue
                idx = torch.arange(start, end + 1)
                R = obj_rot[idx]
                p = obj_trans[idx]
                palm = joints[idx, PALM[hand]]
                wrist = joints[idx, WRIST[hand]]
                R_wrist = rot_glb[idx, WRIST[hand]]
                R_elbow = rot_glb[idx, ELBOW[hand]]

                # the palm anchor a rigid wrist would produce: the palm offset is
                # frozen in the *elbow* frame at the segment start, because the
                # forearm IMU is the only rotation the system observes there.
                offset_elbow = torch.einsum("ji,j->i", R_elbow[0], palm[0] - wrist[0])
                palm_rigid = wrist + torch.einsum("tij,j->ti", R_elbow, offset_elbow)

                out = {}
                for name, anchor in (("wrist", wrist), ("palm", palm), ("palm_rigid", palm_rigid)):
                    c = bone_in_object_frame(R, p, anchor)
                    # L0: freeze the first-frame bone
                    out[f"L0_{name}"] = (c - c[0]).norm(dim=-1).mean()
                    # L1: best constant bone
                    out[f"L1_{name}"] = (c - c.mean(0)).norm(dim=-1).mean()

                c = bone_in_object_frame(R, p, palm)
                for frame_name, R_hand in (("elbow", R_elbow), ("wrist", R_wrist)):
                    rel = torch.einsum("tji,tjk->tik", R, R_hand)          # R_obj^T R_hand
                    M = torch.einsum("tij,kj->tik", rel, rel[0])           # M(t) = Rrel(t) Rrel(0)^T
                    eye = torch.eye(3).expand(n, 3, 3)
                    if frame_name == "elbow":
                        # L2: anchor frozen at the first frame, pivot free (3 params)
                        v = lstsq3(M - eye, -(c - c[0]))
                        out["L2_pivot_elbow"] = (c - c[0] + torch.einsum("tij,j->ti", M - eye, v)).norm(dim=-1).mean()
                        # L5: refit (anchor, pivot) every K frames
                        pieces = []
                        for lo in range(0, n, args.refit_every):
                            hi = min(lo + args.refit_every, n)
                            if hi - lo < 4:
                                if pieces:
                                    pieces.append(pieces[-1].new_zeros(hi - lo))
                                continue
                            Mi = torch.einsum("tij,kj->tik", rel[lo:hi], rel[lo])
                            _, _, res = fit_anchor_pivot(c[lo:hi], Mi)
                            pieces.append(res)
                        out["L5_refit_elbow"] = torch.cat(pieces).mean() if pieces else torch.tensor(float("nan"))
                    # L3 / L4: best (anchor, pivot)
                    _, pivot, res = fit_anchor_pivot(c, M)
                    out[f"L34_{frame_name}"] = res.mean()
                    out[f"pivot_{frame_name}"] = pivot.norm()

                # relative hand-object rotation inside the segment (drives the residual)
                rel_w = torch.einsum("tji,tjk->tik", R, R_wrist)
                Mw = torch.einsum("tij,kj->tik", rel_w, rel_w[0])
                trace = Mw[:, 0, 0] + Mw[:, 1, 1] + Mw[:, 2, 2]
                angle = torch.rad2deg(torch.arccos(((trace - 1) / 2).clamp(-1, 1)))

                weights.append(n)
                for k, v in out.items():
                    value = float(v) * 100.0   # metres -> cm
                    acc[k].append(value)
                    per_object[obj_name][k].append((value, n))
                acc["angle"].append(float(angle.mean()))
                acc["bone_len"].append(float(bone_in_object_frame(R, p, palm).norm(dim=-1).mean()) * 100.0)
                seg_meta.append({"obj": obj_name, "n": n, "angle": float(angle.mean())})

    w = np.array(weights, dtype=np.float64)
    summary = {k: float(np.average(np.array(v), weights=w)) for k, v in acc.items()}
    summary["segments"] = int(len(w))
    summary["frames"] = int(w.sum())

    def show(label, key):
        print(f"  {label:<52} {summary[key]:6.2f}")

    print(f"OMOMO test: {summary['segments']} contact segments, {summary['frames']} hand-frames")
    print(f"mean bone length {summary['bone_len']:.1f} cm, mean in-segment hand-object rotation {summary['angle']:.1f} deg\n")
    print("Anchor point comparison (L0 = frozen first-frame bone, L1 = best constant bone):")
    print(f"  {'anchor':<20} {'L0 frozen':>10} {'L1 best const':>14}")
    for name, label in (("wrist", "wrist joint (20/21)"), ("palm", "palm anchor = middle1 MCP"), ("palm_rigid", "palm anchor, wrist locked")):
        print(f"  {label:<28} {summary['L0_'+name]:8.2f} {summary['L1_'+name]:12.2f}")
    print("\nAssumption ladder (palm anchor):")
    show("L0 frozen first-frame bone  [what v1/v2 assume]", "L0_palm")
    show("L1 best constant bone in segment  [oracle]", "L1_palm")
    show("L2 frozen anchor + pivot residual, elbow frame", "L2_pivot_elbow")
    show("L3 best (anchor,pivot), elbow frame  [observable]", "L34_elbow")
    show("L4 best (anchor,pivot), wrist frame  [unobservable]", "L34_wrist")
    show(f"L5 L3 refit every {args.refit_every} frames", "L5_refit_elbow")
    print(f"  {'L6 per-frame re-anchoring':<52} {0.0:6.2f}")
    print(f"\n  |pivot| (offset from the palm anchor): wrist frame {summary['pivot_wrist']:.1f} cm, elbow frame {summary['pivot_elbow']:.1f} cm")

    print("\nPer-object L0 with the palm anchor (cm):")
    rows = []
    for obj, d in per_object.items():
        vals = np.array([x for x, _ in d["L0_palm"]])
        ws = np.array([n for _, n in d["L0_palm"]], dtype=np.float64)
        rows.append((obj, len(vals), float(np.average(vals, weights=ws)),
                     float(np.average([x for x, _ in d["L34_elbow"]], weights=ws))))
    for obj, cnt, l0, l3 in sorted(rows, key=lambda r: -r[2]):
        print(f"  {obj:<14} n={cnt:4d}   L0 {l0:6.2f}   L3 {l3:6.2f}")

    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "per_object": rows}, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
