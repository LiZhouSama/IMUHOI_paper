#!/usr/bin/env python3
"""Open-loop upper bounds for an absolute anchor in GT moving-without-hands frames.

Replays the v2 fixed fusion (dump weights/candidates):
  cur_t = wL*FK_L + wR*FK_R + wI*(prev + d_t) + wS*prev
and, on a mask, replaces cur_t by the GT position (a perfect anchor), so the
following frames inherit it.  Masks: every no-hand frame, no-hand frames with a
non-hand body part within 15 cm of the object surface (also split by body part
and by run length), only the first frame of each no-hand run, GT height only,
an anchor that carries the HP root-translation error (GT + root error: the best
any body-attached anchor can do), and (for scale) every hand frame.  A last
variant removes the HP root-translation error from the FK candidates instead
(GT root translation).  The replay of the dump weights/candidates reproduces the
published error exactly.

    python scripts/diag/nohand_anchor_oracle.py dump.pt process/IMUHOI++cano/processed_split_data_BEHAVE/test
"""
import sys
from pathlib import Path

import numpy as np
import torch

PARTS = (("leg/foot", (4, 5, 7, 8, 10, 11)), ("pelvis/hip", (0, 1, 2)),
         ("torso/back", (3, 6, 9, 12, 13, 14, 15, 16, 17)))  # SMPL joints, arms/hands excluded


def replay(w, fk_l, fk_r, disp, start, gt, reset=None, height_only=False, target=None):
    out = np.zeros_like(gt)
    prev = start
    for t in range(len(gt)):
        cur = w[t, 0] * fk_l[t] + w[t, 1] * fk_r[t] + w[t, 2] * (prev + disp[t]) + w[t, 3] * prev
        if reset is not None and reset[t]:
            if height_only:
                cur = cur.copy(); cur[1] = gt[t, 1]
            else:
                cur = gt[t] if target is None else target[t]
        out[t] = cur
        prev = cur
    return out


def main():
    recs = torch.load(sys.argv[1], map_location="cpu")["records"]
    data = Path(sys.argv[2])
    res = {}
    frames = {}
    for r in recs:
        gt = r["gt_obj_trans"].numpy().astype(np.float64); T = len(gt)
        w = r["gating_weights"].numpy().astype(np.float64)
        stage1 = r["stage1_pred_palm_glb_pos"].numpy()
        refined = r["pred_palm_glb_pos"].numpy()
        fk_l = refined[:, 0] + (r["pred_lhand_obj_trans"].numpy() - stage1[:, 0])
        fk_r = refined[:, 1] + (r["pred_rhand_obj_trans"].numpy() - stage1[:, 1])
        disp = (r["pred_imu_obj_trans"].numpy() - r["pred_static_obj_trans"].numpy()).astype(np.float64)
        start = r["pred_static_obj_trans"].numpy()[0].astype(np.float64)
        hands = (r["gt_lhand_contact"].numpy() > .5) | (r["gt_rhand_contact"].numpy() > .5)
        moving = r["gt_obj_contact"].numpy() > .5
        nohand = moving & ~hands
        s = torch.load(data / r["seq_file"], map_location="cpu", weights_only=False)
        J = np.asarray(s["position_global_full_gt_world"], np.float64)[:T]
        P = np.asarray(s["obj_points_canonical"], np.float64)
        R = np.asarray(s["obj_rot"], np.float64)[:T]; sc = np.asarray(s["obj_scale"], np.float64).reshape(-1)[:T]
        surf = np.einsum("pc,tdc->tpd", P, R) * sc[:, None, None] + gt[:, None, :]
        part_dist = {}
        for part, joints in PARTS:
            d = np.full(T, np.inf)
            for j in joints:
                d = np.minimum(d, np.linalg.norm(surf - J[:, j][:, None, :], axis=-1).min(1))
            part_dist[part] = d
        dist = np.minimum.reduce(list(part_dist.values()))
        body = dist < 0.15
        nearest = np.array(list(part_dist))[np.argmin(np.stack(list(part_dist.values())), 0)]
        onset = nohand & ~np.concatenate(([False], nohand[:-1]))
        run_len = np.zeros(T, int)
        edges = np.diff(np.concatenate(([0], nohand.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            run_len[a:b] = b - a
        variants = {
            "published (dump)": None,
            "replay base": np.zeros(T, bool),
            "GT anchor: every no-hand frame": nohand,
            "GT anchor: no-hand frames with body contact": nohand & body,
            "GT anchor: no-hand + body contact, leg/foot": nohand & body & (nearest == "leg/foot"),
            "GT anchor: no-hand + body contact, pelvis/hip": nohand & body & (nearest == "pelvis/hip"),
            "GT anchor: no-hand + body contact, torso/back": nohand & body & (nearest == "torso/back"),
            "GT anchor: first frame of each no-hand run": onset,
            "GT height only: every no-hand frame": ("height", nohand),
            "GT anchor: no-hand runs < 15 frames": nohand & (run_len < 15),
            "GT anchor: no-hand runs 15-59 frames": nohand & (run_len >= 15) & (run_len < 60),
            "GT anchor: no-hand runs >= 60 frames": nohand & (run_len >= 60),
            "GT anchor: every hand frame (scale)": moving & hands,
            "GT+root-error anchor: no-hand frames with body contact": ("root", nohand & body),
            "GT+root-error anchor: every no-hand frame": ("root", nohand),
            "FK candidates with GT root translation": ("fk_root", None),
        }
        root_err = r["root_trans_pred"].numpy().astype(np.float64)[:T] - r["gt_trans"].numpy().astype(np.float64)[:T]
        for name, mask in variants.items():
            if mask is None:
                pub = r["pred_obj_trans_published"].numpy().astype(np.float64)
            elif isinstance(mask, tuple) and mask[0] == "root":
                pub = replay(w, fk_l, fk_r, disp, start, gt, mask[1], target=gt + root_err)
            elif isinstance(mask, tuple) and mask[0] == "fk_root":
                pub = replay(w, fk_l - root_err, fk_r - root_err, disp, start, gt)
            elif isinstance(mask, tuple):
                pub = replay(w, fk_l, fk_r, disp, start, gt, mask[1], height_only=True)
            else:
                pub = replay(w, fk_l, fk_r, disp, start, gt, mask)
            e = np.linalg.norm(pub - gt, axis=-1) * 100
            res.setdefault(name, []).append(e.mean())
            frames.setdefault(name, []).append(e)
    for name in res:
        e = np.concatenate(frames[name])
        print(f"  {name:54s} seq-mean obj {np.mean(res[name]):6.2f}   frame-mean {e.mean():6.2f}")


if __name__ == "__main__":
    main()
