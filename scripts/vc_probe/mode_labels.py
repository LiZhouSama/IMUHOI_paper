#!/usr/bin/env python3
"""Physically defined interaction mode of every GT hand-contact segment (labels for VC probing).

Segment = run of GT left|right hand contact (gaps <= 5 frames merged, >= 15 frames). Object height uses
the lowest of the canonical object points; rules are applied in this order:
  ball       - yogaball / basketball (rolling makes tilt meaningless)
  lift       - the lowest point rises > LIFT_M above its height at segment start (p90 over the segment)
  static     - horizontal path < STATIC_M, rotation < STATIC_DEG and no rise (touch / lean)
  tilt       - the object's up axis tilts > 15 deg without leaving the ground (tip, pivot, slanted drag)
  ground     - lowest point within GROUND_M on > 90 % of frames and path > MOVE_M (push / drag / slide)
  ground_rot - stays on the ground and only turns about the vertical axis (> 30 deg)
  other      - everything else
Hands: L, R or LR (both hands in contact on >= 30 % of the segment frames).

    python scripts/vc_probe/mode_labels.py <dataset_dir> [--splits train test] [--out labels.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import torch

LIFT_M, GROUND_M, MOVE_M, STATIC_M, STATIC_DEG = 0.08, 0.04, 0.15, 0.05, 10.0
MIN_SEG = 15


def runs(mask: np.ndarray, max_gap: int = 5):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            out.append((start, prev + 1))
            start = i
        prev = i
    out.append((start, prev + 1))
    return [(a, b) for a, b in out if b - a >= MIN_SEG]


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def analyse(path: Path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if not d.get("has_object", True):
        return None, []
    pos = d["position_global_full_gt_world"].numpy()
    up = int(np.argmax((pos[:, 15] - pos[:, [10, 11]].mean(1)).mean(0)))  # head minus toes
    horiz = [a for a in range(3) if a != up]
    R = d["obj_rot"].numpy().astype(np.float64)
    t = d["obj_trans"].numpy().astype(np.float64)
    s = d["obj_scale"].numpy().astype(np.float64)
    p = d["obj_points_canonical"].float().numpy().astype(np.float64)
    world = np.einsum("tij,nj->tni", R, p) * s[:, None, None] + t[:, None, :]
    low = world[..., up].min(1)
    floor = np.percentile(pos[:, [10, 11], up].min(1), 2)
    lh = d["lhand_contact"].numpy().astype(bool)
    rh = d["rhand_contact"].numpy().astype(bool)
    segs = []
    for a, b in runs(lh | rh):
        h0 = low[a]
        rise = np.percentile(low[a:b] - h0, 90)
        dxy = np.diff(t[a:b][:, horiz], axis=0)
        path_len = float(np.linalg.norm(dxy, axis=1).sum())
        net = float(np.linalg.norm(t[b - 1, horiz] - t[a, horiz]))
        ang = max(rot_angle_deg(R[a], R[i]) for i in range(a, b, 5))
        # tilt = change of the object's world-up axis; yaw = heading change of a world-horizontal axis
        e_up = np.eye(3)[up]
        u0 = R[a].T @ e_up
        tilt = float(np.degrees(np.arccos(np.clip(np.einsum("tij,j->ti", R[a:b], u0) @ e_up, -1, 1))).max())
        x0 = R[a].T @ np.eye(3)[horiz[0]]
        xt = np.einsum("tij,j->ti", R[a:b], x0)[:, horiz]
        yaw = float(np.degrees(np.abs(np.arctan2(xt[:, 1], xt[:, 0]))).max())
        ground_frac = float(np.mean(np.abs(low[a:b] - h0) < GROUND_M))
        if d["obj_name"] in ("yogaball", "basketball"):
            mode = "ball"
        elif rise > LIFT_M:
            mode = "lift"
        elif path_len < STATIC_M and ang < STATIC_DEG and rise < 0.03:
            mode = "static"
        elif tilt > 15.0:
            mode = "tilt"
        elif ground_frac > 0.9 and path_len > MOVE_M:
            mode = "ground"
        elif ground_frac > 0.9 and yaw > 30.0:
            mode = "ground_rot"
        else:
            mode = "other"
        both = float(np.mean(lh[a:b] & rh[a:b]))
        hands = "LR" if both >= 0.3 else ("L" if lh[a:b].sum() >= rh[a:b].sum() else "R")
        root_travel = float(np.linalg.norm(np.diff(pos[a:b, 0][:, horiz], axis=0), axis=1).sum())
        segs.append(dict(seq=d["seq_name"], file=path.stem, T=int(len(t)), obj=d["obj_name"], a=int(a), b=int(b),
                         mode=mode, hands=hands,
                         rise=float(rise), path=path_len, net=net, ang=ang, tilt=tilt, yaw=yaw, start_clear=float(h0 - floor),
                         root_travel=root_travel))
    meta = dict(seq=d["seq_name"], obj=d["obj_name"], T=int(len(t)), up=up)
    return meta, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    all_segs, metas = [], []
    for split in args.splits:
        files = sorted((args.root / split).glob("*.pt"))
        if args.limit:
            files = files[: args.limit]
        for f in files:
            meta, segs = analyse(f)
            if meta is None:
                continue
            meta["split"] = split
            for sg in segs:
                sg["split"] = split
            metas.append(meta)
            all_segs.extend(segs)
    print(f"sequences {len(metas)}, segments {len(all_segs)}, up axes {collections.Counter(m['up'] for m in metas)}")
    by = collections.Counter((s["split"], s["mode"]) for s in all_segs)
    for split in args.splits:
        row = {m: by[(split, m)] for m in ("lift", "ground", "ground_rot", "tilt", "static", "ball", "other")}
        seqs_ground = len({s["seq"] for s in all_segs if s["split"] == split and s["mode"] == "ground"})
        print(f"  {split}: {row}  (sequences with a ground segment: {seqs_ground})")
    hm = collections.Counter((s["mode"], s["hands"]) for s in all_segs)
    print("  mode x hands:", {f"{m}/{h}": n for (m, h), n in sorted(hm.items())})
    frames = collections.defaultdict(int)
    for s in all_segs:
        frames[s["mode"]] += s["b"] - s["a"]
    print("  contact frames per mode:", dict(frames))
    # BEHAVE: cross-tab with the action word in the sequence name
    tags = collections.defaultdict(collections.Counter)
    for s in all_segs:
        m = re.match(r"Date\d+_Sub\d+_([a-z]+)(?:_([a-z]+))?", s["seq"])
        if m:
            tags[f"{m.group(1)}_{m.group(2) or '-'}"][s["mode"]] += 1
    if tags:
        print("  name tag -> segment modes:")
        for k in sorted(tags):
            print(f"    {k:24s} {dict(tags[k])}")
    obj = collections.defaultdict(collections.Counter)
    for s in all_segs:
        obj[s["obj"]][s["mode"]] += 1
    print("  object -> segment modes:")
    for k in sorted(obj):
        print(f"    {k:18s} {dict(obj[k])}")
    sc = np.array([s["start_clear"] for s in all_segs])
    print(f"  start clearance of lowest point above floor estimate: median {np.median(sc):.3f} m, "
          f"p10 {np.percentile(sc, 10):.3f}, p90 {np.percentile(sc, 90):.3f}")
    if args.out:
        args.out.write_text(json.dumps(all_segs, indent=1))


if __name__ == "__main__":
    main()
