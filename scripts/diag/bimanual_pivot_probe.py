#!/usr/bin/env python3
"""Can the inter-hand span identify the per-grasp pivots that drive in-segment bone changes?

Rigid-grip kinematics: a grip point fixed in both the object frame (q_h) and the forearm frame (v_h, offset from the
palm anchor) gives   palm_h = p + R q_h - R_f,h v_h,   hence with B_h = R^T R_f,h (observable: object + forearm IMUs)

    bone_h(t)  = R^T (p - palm_h) = -q_h + B_h(t) v_h          ->  in-segment change  dB_h(t) v_h
    span(t)    = R^T (palm_R - palm_L) = s0 - B_R(t) v_R + B_L(t) v_L   (s0 = q_R - q_L)

A single hand has no measurement of v_h (the object position is what we are estimating); with two hands the SPAN is
one.  Per GT bimanual segment this probe fits (v_L, v_R, s0) by ridge least squares on the span and checks whether
dB_h v_h explains the in-segment change of the GT bone.  Variants: span from predicted palms (realisable), span from GT
palms (identifiability without HP noise), and the oracle pivot fitted on the GT bones themselves (ceiling).

    python scripts/diag/bimanual_pivot_probe.py dump.pt
"""
import sys

import numpy as np
import torch


def rot6d_to_matrix(x):
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True).clip(1e-8)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def fit_span(span, B_l, B_r, ridge):
    """Ridge LS for x = [v_L, v_R, s0] in span = s0 - B_R v_R + B_L v_L.  Returns v_L, v_R."""
    n = span.shape[0]
    A = np.zeros((n, 3, 9))
    A[:, :, 0:3] = B_l
    A[:, :, 3:6] = -B_r
    A[:, :, 6:9] = np.eye(3)[None]
    A2 = A.reshape(-1, 9)
    reg = np.diag([ridge] * 6 + [0.0] * 3) * n
    x = np.linalg.solve(A2.T @ A2 + reg, A2.T @ span.reshape(-1))
    return x[0:3], x[3:6]


def fit_bone(bone, B, ridge):
    """Oracle: bone(t) = -q + B(t) v  -> ridge LS for v (q free)."""
    n = bone.shape[0]
    A = np.zeros((n, 3, 6))
    A[:, :, 0:3] = B
    A[:, :, 3:6] = -np.eye(3)[None]
    A2 = A.reshape(-1, 6)
    reg = np.diag([ridge] * 3 + [0.0] * 3) * n
    x = np.linalg.solve(A2.T @ A2 + reg, A2.T @ bone.reshape(-1))
    return x[0:3]


def main():
    records = torch.load(sys.argv[1], map_location="cpu")["records"]
    min_len = 30
    for ridge in (0.05, 0.3, 2.0):
        acc = {k: [] for k in ("frozen", "pred_span", "gt_span", "oracle", "rot_deg", "v_pred", "v_oracle")}
        for r in records:
            n = int(r["publish_len"])
            if not r.get("has_object", True) or n == 0:
                continue
            both = np.asarray(r["gt_lhand_contact"][:n]).astype(bool) & np.asarray(r["gt_rhand_contact"][:n]).astype(bool)
            if not both.any():
                continue
            R = rot6d_to_matrix(np.asarray(r["gt_obj_rot"][:n], dtype=np.float64))
            imu = np.asarray(r["gt_human_imu"][:n], dtype=np.float64)
            R_root = rot6d_to_matrix(imu[:, 0, 3:9])
            B = [np.einsum("tji,tjk->tik", R, R_root @ rot6d_to_matrix(imu[:, 4 + h, 3:9])) for h in (0, 1)]  # R^T R_forearm
            hs = np.asarray(r.get("stage1_pred_palm_glb_pos", r["pred_palm_glb_pos"])[:n], dtype=np.float64)
            g = np.asarray(r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n], dtype=np.float64)
            p_gt = np.asarray(r["gt_obj_trans"][:n], dtype=np.float64)
            to_obj = lambda v: np.einsum("tji,tj->ti", R, v)  # noqa: E731
            span_pred, span_gt = to_obj(hs[:, 1] - hs[:, 0]), to_obj(g[:, 1] - g[:, 0])
            bone = [to_obj(p_gt - g[:, h]) for h in (0, 1)]
            t = 0
            while t < n:
                if not both[t]:
                    t += 1
                    continue
                s = t
                while t < n and both[t]:
                    t += 1
                if t - s < min_len:
                    continue
                sl = slice(s, t)
                vp = fit_span(span_pred[sl], B[0][sl], B[1][sl], ridge)
                vg = fit_span(span_gt[sl], B[0][sl], B[1][sl], ridge)
                for h in (0, 1):
                    Bh, bh = B[h][sl], bone[h][sl]
                    dB = Bh - Bh[:1]
                    db = bh - bh[:1]
                    vo = fit_bone(bh, Bh, ridge)
                    acc["frozen"].append(np.linalg.norm(db, axis=-1))
                    acc["pred_span"].append(np.linalg.norm(db - dB @ vp[h], axis=-1))
                    acc["gt_span"].append(np.linalg.norm(db - dB @ vg[h], axis=-1))
                    acc["oracle"].append(np.linalg.norm(db - dB @ vo, axis=-1))
                    cos = ((np.einsum("tii->t", np.einsum("tji,jk->tik", Bh, Bh[0])) - 1.0) * 0.5).clip(-1, 1)
                    acc["rot_deg"].append(np.degrees(np.arccos(cos)))
                    acc["v_pred"].append([np.linalg.norm(vp[h])])
                    acc["v_oracle"].append([np.linalg.norm(vo)])
        m = {k: float(np.concatenate(v).mean()) for k, v in acc.items()}
        print(f"ridge {ridge:4.2f} | in-segment GT bone change (cm): frozen-onset {m['frozen'] * 100:.2f} | pivots from PREDICTED span {m['pred_span'] * 100:.2f}"
              f" | pivots from GT span {m['gt_span'] * 100:.2f} | oracle pivots (fit on GT bones) {m['oracle'] * 100:.2f}"
              f" || mean forearm-vs-object rotation since onset {m['rot_deg']:.1f} deg, |v| pred {m['v_pred'] * 100:.1f} cm, oracle {m['v_oracle'] * 100:.1f} cm")


if __name__ == "__main__":
    main()
