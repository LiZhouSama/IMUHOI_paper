#!/usr/bin/env python3
"""Separate 'bone quality' from 'fusion policy': publish each model FK-only and compare.

HOI is exactly the world-frame hand->object vector (bone) error: q = (p - h) - (p_gt - h_gt).  A fusion that mixes
the IMU/static branches moves the object away from hand + bone, which can lower the ABSOLUTE object error while
raising HOI.  This replays the evaluator's causal publication twice per model:

  as-is    : the model's own 4-way gate
  fk_only  : the two FK weights renormalised to 1 (object := weighted hand + bone, no IMU/static pull)

and reports HOI and object error for both, on GT contact hand-frames / all published frames.

    python scripts/diag/fk_only_replay.py ours=a.pt imuhoi=b.pt
"""
import sys

import numpy as np
import torch


def _np(x):
    return np.asarray(x, dtype=np.float64)


def publish(fk_l, fk_r, imu, static, w):
    prev = static[0].copy()
    out = np.zeros_like(fk_l)
    for t in range(len(w)):
        cur = w[t, 0] * fk_l[t] + w[t, 1] * fk_r[t] + w[t, 2] * (prev + imu[t] - static[t]) + w[t, 3] * prev
        out[t] = cur
        prev = cur
    return out


def main():
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        records = torch.load(path, map_location="cpu")["records"]
        acc = {k: [] for k in ("hoi_asis", "hoi_fk", "obj_asis", "obj_fk", "w", "fk_share", "hoi_seq_asis", "hoi_seq_fk")}
        for r in records:
            n = int(r["publish_len"])
            if not r.get("has_object", True) or n == 0:
                continue
            hs = _np(r.get("stage1_pred_palm_glb_pos", r["pred_palm_glb_pos"])[:n])
            hr = _np(r["pred_palm_glb_pos"][:n])
            fk = [hr[:, h] + (_np(r[f"pred_{s}hand_obj_trans"][:n]) - hs[:, h]) for h, s in ((0, "l"), (1, "r"))]
            imu, static = _np(r["pred_imu_obj_trans"][:n]), _np(r["pred_static_obj_trans"][:n])
            w = _np(r["gating_weights"][:n])
            p_gt = _np(r["gt_obj_trans"][:n])
            g = _np(r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n])
            cl = np.asarray(r["gt_lhand_contact"][:n]).astype(bool)
            cr = np.asarray(r["gt_rhand_contact"][:n]).astype(bool)
            w_fk = np.zeros_like(w)
            total = (w[:, 0] + w[:, 1]).clip(1e-8)
            w_fk[:, 0], w_fk[:, 1] = w[:, 0] / total, w[:, 1] / total
            tracks = {"asis": publish(fk[0], fk[1], imu, static, w), "fk": publish(fk[0], fk[1], imu, static, w_fk)}
            acc["w"].append(w)
            acc["fk_share"].append(total)
            for key, p in tracks.items():
                acc[f"obj_{key}"].append(np.linalg.norm(p - p_gt, axis=-1) * 100.0)
                errs = [np.linalg.norm((p[m] - hr[m, h]) - (p_gt[m] - g[m, h]), axis=-1) * 100.0
                        for h, m in ((0, cl), (1, cr)) if m.any()]
                if errs:
                    acc[f"hoi_{key}"].append(np.concatenate(errs))
                    acc[f"hoi_seq_{key}"].append(np.concatenate(errs).mean())
        cat = lambda k: np.concatenate(acc[k]) if acc[k] else np.zeros(1)  # noqa: E731
        print(f"\n=== {name}")
        print(f"  gate weights [FK_L, FK_R, IMU, static]: {np.round(np.concatenate(acc['w'], axis=0).mean(0), 3)}"
              f"   FK share mean {cat('fk_share').mean():.3f}")
        for key, label in (("asis", "as-is (model gate)"), ("fk", "FK-only publication")):
            print(f"  {label:22s} HOI seq-equal {np.mean(acc[f'hoi_seq_{key}']):7.3f} | HOI pooled {cat(f'hoi_{key}').mean():7.3f}"
                  f" | object error {cat(f'obj_{key}').mean():7.3f}")


if __name__ == "__main__":
    main()
