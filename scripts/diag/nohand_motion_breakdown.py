#!/usr/bin/env python3
"""Where does the error of "object moving without hand contact" frames come from?

GT regimes per frame: static (object not moving), hand (moving, any GT hand
contact), nohand (moving, no GT hand contact).  For every maximal nohand run
starting at frame s the published object error e(t) = pub(t) - gt(t) splits into
the offset inherited at the run start, e(s), and the part accrued inside the
run, e(t) - e(s).  Also reported per run: fusion weights and VC probabilities,
what precedes the run, and the GT physics of the motion:

  * rigid carry: object centre expressed in each body joint's frame,
    q_j(t) = R_j(t)^T (o(t) - p_j(t)); the best joint's mean drift of q_j over
    the run vs. the object's own world displacement;
  * contact: closest body joint to the object's surface points.

    python scripts/diag/nohand_motion_breakdown.py --dump dump.pt [--dump_b other.pt] \
        --data process/IMUHOI++cano/processed_split_data_BEHAVE/test
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

JOINT_NAMES = ("pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3",
               "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow",
               "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand")
GROUPS = {"hand/arm": (18, 19, 20, 21, 22, 23), "torso/back": (3, 6, 9, 12, 13, 14, 15, 16, 17),
          "pelvis/hip": (0, 1, 2), "leg/foot": (4, 5, 7, 8, 10, 11)}


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def group_of(joint):
    for name, joints in GROUPS.items():
        if joint in joints:
            return name
    return "?"


def load_records(path):
    return {r["seq_name"]: r for r in torch.load(path, map_location="cpu")["records"]}


def analyse(records, records_b, data_dir):
    frames = defaultdict(list)
    runs = []
    for name, rec in records.items():
        gt = rec["gt_obj_trans"].numpy().astype(np.float64)
        T = len(gt)
        pub = rec["pred_obj_trans_published"].numpy().astype(np.float64)
        hands = (rec["gt_lhand_contact"].numpy() > 0.5) | (rec["gt_rhand_contact"].numpy() > 0.5)
        moving = rec["gt_obj_contact"].numpy() > 0.5
        regime = np.where(~moving, 0, np.where(hands, 1, 2))  # 0 static, 1 hand, 2 nohand
        err = (pub - gt) * 100.0
        err_b = None
        if records_b is not None and name in records_b:
            err_b = (records_b[name]["pred_obj_trans_published"].numpy().astype(np.float64) - gt) * 100.0
        gates = rec["gating_weights"].numpy()
        probs = rec["pred_hand_contact_prob"].numpy()
        seq = torch.load(Path(data_dir) / rec["seq_file"], map_location="cpu", weights_only=False)
        joints = np.asarray(seq["position_global_full_gt_world"], dtype=np.float64)[:T]
        rots = np.asarray(seq["rotation_global"], dtype=np.float64)[:T]
        points = np.asarray(seq["obj_points_canonical"], dtype=np.float64)
        obj_rot = np.asarray(seq["obj_rot"], dtype=np.float64)[:T]
        scale = np.asarray(seq["obj_scale"], dtype=np.float64).reshape(-1)[:T]
        for key, value in (("all", np.ones(T, bool)), ("static", regime == 0), ("hand", regime == 1), ("nohand", regime == 2)):
            frames[f"err_{key}"].append(np.linalg.norm(err[value], axis=-1))
            if err_b is not None:
                frames[f"errb_{key}"].append(np.linalg.norm(err_b[value], axis=-1))
        for s, e in runs_of(regime == 2):
            idx = np.arange(s, e)
            inherited = np.linalg.norm(err[s])
            accrued = np.linalg.norm(err[idx] - err[s], axis=-1)
            total = np.linalg.norm(err[idx], axis=-1)
            rel = np.einsum("tjba,tjb->tja", rots[idx], gt[idx, None, :] - joints[idx])  # R_j^T (o - p_j)
            drift_local = np.linalg.norm(rel - rel[:1], axis=-1).mean(0)  # per joint, metres
            disp_world = np.linalg.norm(gt[idx] - gt[s], axis=-1).mean()
            best = int(np.argmin(drift_local))
            mid = idx[len(idx) // 2]
            surface = (points * scale[mid]) @ obj_rot[mid].T + gt[mid]
            dist = np.linalg.norm(surface[None, :, :] - joints[mid][:, None, :], axis=-1).min(1)
            near = int(np.argmin(dist))
            runs.append({
                "seq": name, "obj": rec["obj_name"], "start": s, "len": e - s,
                "prev": "start" if s == 0 else ("static", "hand")[int(regime[s - 1])],
                "inherited": inherited, "accrued": accrued.mean(), "total": total.mean(),
                "total_b": np.linalg.norm(err_b[idx], axis=-1).mean() if err_b is not None else np.nan,
                "inherited_b": np.linalg.norm(err_b[s]) if err_b is not None else np.nan,
                "w_fk": gates[idx, :2].sum(-1).mean(), "w_imu": gates[idx, 2].mean(), "w_static": gates[idx, 3].mean(),
                "p_hand": probs[idx, :2].max(-1).mean(), "p_move": probs[idx, 2].mean(),
                "disp_world": disp_world, "carry_joint": best, "carry_drift": drift_local[best],
                "near_joint": near, "near_dist": dist[near],
            })
            frames["nohand_inherited"].append(np.full(len(idx), inherited))
            frames["nohand_accrued"].append(accrued)
    return {k: np.concatenate(v) for k, v in frames.items()}, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--dump_b", default=None, help="Second dump of the same sequences (e.g. GT-velocity oracle).")
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    records = load_records(args.dump)
    records_b = load_records(args.dump_b) if args.dump_b else None
    frames, runs = analyse(records, records_b, args.data)
    total_frames = len(frames["err_all"])
    print(f"frames {total_frames}; error sum share and mean (cm) per GT regime" + (" | dump_b mean" if records_b else ""))
    for key in ("static", "hand", "nohand"):
        e = frames[f"err_{key}"]
        line = f"  {key:7s} frames {len(e)/total_frames:6.1%}  share of error sum {e.sum()/frames['err_all'].sum():6.1%}  mean {e.mean():6.2f}"
        if records_b:
            line += f" | {frames[f'errb_{key}'].mean():6.2f}"
        print(line)
    print(f"  all     mean {frames['err_all'].mean():6.2f}" + (f" | {frames['errb_all'].mean():6.2f}" if records_b else ""))
    print(f"nohand frames: |e| {frames['err_nohand'].mean():.2f} = inherited offset |e(s)| {frames['nohand_inherited'].mean():.2f}"
          f" vs accrued |e(t)-e(s)| {frames['nohand_accrued'].mean():.2f} (frame-weighted)")
    n = np.array([r["len"] for r in runs])
    print(f"nohand runs {len(runs)}: length median {np.median(n):.0f}, mean {n.mean():.0f}, frames in runs >= 60: {n[n >= 60].sum()/n.sum():.0%}")

    def wmean(key, sel=None):
        rr = [r for r in runs if sel is None or sel(r)]
        w = np.array([r["len"] for r in rr], float)
        return (np.array([r[key] for r in rr]) * w).sum() / max(w.sum(), 1), w.sum()

    print("frame-weighted by preceding regime:")
    for prev in ("hand", "static", "start"):
        tot, w = wmean("total", lambda r, p=prev: r["prev"] == p)
        if w == 0:
            continue
        inh, _ = wmean("inherited", lambda r, p=prev: r["prev"] == p)
        acc, _ = wmean("accrued", lambda r, p=prev: r["prev"] == p)
        print(f"  after {prev:6s} frames {w/n.sum():5.1%}  |e| {tot:6.2f}  inherited {inh:6.2f}  accrued {acc:6.2f}")
    for key in ("w_fk", "w_imu", "w_static", "p_hand", "p_move"):
        print(f"  mean {key:8s} {wmean(key)[0]:.3f}")
    carried = [r for r in runs if r["carry_drift"] < 0.3 * max(r["disp_world"], 1e-6) and r["disp_world"] > 0.05]
    print(f"rigidly carried by a joint (local drift < 0.3 x world displacement, displacement > 5 cm): "
          f"{sum(r['len'] for r in carried)/n.sum():.0%} of nohand frames")
    by_group = defaultdict(lambda: [0, 0.0])
    for r in runs:
        g = group_of(r["near_joint"]) if r["near_dist"] < 0.15 else "no body part < 15 cm"
        by_group[g][0] += r["len"]
        by_group[g][1] += r["total"] * r["len"]
    print("closest body part to the object surface (mid-run):")
    for g, (cnt, s) in sorted(by_group.items(), key=lambda kv: -kv[1][0]):
        print(f"  {g:22s} frames {cnt/n.sum():5.1%}  |e| {s/max(cnt,1):6.2f}")
    print("per object (nohand frames, |e|, inherited, accrued, world displacement, best carry joint drift, nearest part):")
    by_obj = defaultdict(list)
    for r in runs:
        by_obj[r["obj"]].append(r)
    rows = []
    for obj, rr in by_obj.items():
        w = np.array([r["len"] for r in rr], float)
        avg = lambda k: (np.array([r[k] for r in rr]) * w).sum() / w.sum()
        parts = defaultdict(float)
        for r in rr:
            parts[group_of(r["near_joint"]) if r["near_dist"] < 0.15 else "none"] += r["len"]
        top = max(parts, key=parts.get)
        rows.append((w.sum(), obj, avg("total"), avg("inherited"), avg("accrued"), avg("disp_world"), avg("carry_drift"),
                     f"{top} {parts[top]/w.sum():.0%}", avg("total_b")))
    for cnt, obj, tot, inh, acc, disp, drift, part, tot_b in sorted(rows, reverse=True):
        extra = f"  dump_b {tot_b:6.2f}" if records_b else ""
        print(f"  {obj:14s} frames {cnt/n.sum():5.1%}  |e| {tot:6.2f}  inh {inh:6.2f}  acc {acc:6.2f}  disp {disp*100:5.1f}cm"
              f"  carry-drift {drift*100:5.1f}cm  near {part}{extra}")


if __name__ == "__main__":
    main()
