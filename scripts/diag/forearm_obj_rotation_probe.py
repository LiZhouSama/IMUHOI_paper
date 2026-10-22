#!/usr/bin/env python3
"""How much of the in-segment bone change is predictable from the forearm-vs-object rotation?

Rigid grip: bone_h(t) = R_obj^T (p - palm_h) = -q_h + B_h(t) v_h  with  B_h(t) = R_obj(t)^T R_forearm,h(t)
observable from the object IMU + (root, forearm) IMUs.  The main predictor currently cannot form B, because its input
carries ROOT-RELATIVE forearm orientation and no root orientation.

This probe predicts the GT bone change since the window start (60 frames, the online publication reference) on GT
contact hand-frames, 5-fold CV by sequence, small MLP:
  base   : what the predictor sees today (object 6D + rot delta, root-relative forearm 6D + acc, palm positions in the
           object frame, contact probs, bone at the window start)
  +B     : adds B(t) 6D, the per-frame delta of B, and B(t0)^T B(t) 6D + its angle
Reported in cm of bone error (equivalently the FK-branch HOI on those frames).

    python scripts/diag/forearm_obj_rotation_probe.py dump.pt
"""
import sys

import numpy as np
import torch
import torch.nn as nn

WIN = 60


def rot6d_to_matrix(x):
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True).clip(1e-8)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def matrix_to_rot6d(R):
    return np.concatenate((R[..., 0, :], R[..., 1, :]), axis=-1)


def build(records):
    X, Y, S, base_dim = [], [], [], None
    for si, r in enumerate(records):
        n = int(r["publish_len"])
        if not r.get("has_object", True) or n <= WIN:
            continue
        R = rot6d_to_matrix(np.asarray(r["gt_obj_rot"][:n], dtype=np.float64))
        imu = np.asarray(r["gt_human_imu"][:n], dtype=np.float64)
        R_root = rot6d_to_matrix(imu[:, 0, 3:9])
        g = np.asarray(r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n], dtype=np.float64)
        hs = np.asarray(r.get("stage1_pred_palm_glb_pos", r["pred_palm_glb_pos"])[:n], dtype=np.float64)
        p_gt = np.asarray(r["gt_obj_trans"][:n], dtype=np.float64)
        prob = np.asarray(r["pred_hand_contact_prob"][:n], dtype=np.float64)[:, :3]
        obj6d = np.asarray(r["gt_obj_rot"][:n], dtype=np.float64)
        rot_delta = np.zeros((n, 6))
        rot_delta[1:] = matrix_to_rot6d(np.einsum("tji,tjk->tik", R[:-1], R[1:]))
        cont = [np.asarray(r["gt_lhand_contact"][:n]).astype(bool), np.asarray(r["gt_rhand_contact"][:n]).astype(bool)]
        for h in (0, 1):
            B = np.einsum("tji,tjk->tik", R, R_root @ rot6d_to_matrix(imu[:, 4 + h, 3:9]))
            B6 = matrix_to_rot6d(B)
            dB = np.zeros((n, 6))
            dB[1:] = matrix_to_rot6d(np.einsum("tji,tjk->tik", B[:-1], B[1:]))
            bone = np.einsum("tji,tj->ti", R, p_gt - g[:, h])                    # GT bone, object frame
            palm_obj = np.einsum("tji,tj->ti", R, hs[:, h] - hs[:, 1 - h])       # predicted inter-hand vector
            arm_acc = imu[:, 4 + h, :3]
            for t0 in range(0, n - WIN, 10):                                     # windows every 10 frames
                sl = slice(t0, t0 + WIN)
                m = cont[h][sl]
                if not m.any():
                    continue
                B0 = B[t0]
                rel = matrix_to_rot6d(np.einsum("ji,tjk->tik", B0, B[sl]))        # B(t0)^T B(t)
                ang = np.arccos(((np.einsum("tii->t", np.einsum("ji,tjk->tik", B0, B[sl])) - 1.0) * 0.5).clip(-1, 1))[:, None]
                base = np.concatenate((obj6d[sl], rot_delta[sl], imu[sl, 4 + h, 3:9], arm_acc[sl], palm_obj[sl],
                                       prob[sl], np.repeat(bone[t0][None], WIN, 0), np.full((WIN, 1), 0.0 if h == 0 else 1.0),
                                       np.arange(WIN)[:, None] / WIN), axis=-1)
                extra = np.concatenate((B6[sl], dB[sl], rel, ang), axis=-1)
                base_dim = base.shape[-1]
                X.append(np.concatenate((base, extra), axis=-1)[m])
                Y.append((bone[sl] - bone[t0])[m])                               # change since the window start
                S.append(np.full(int(m.sum()), si))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S), base_dim


def fit(x_tr, y_tr, x_te, epochs=30, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    xe = torch.tensor((x_te - mu) / sd, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(xt.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 3)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        perm = torch.randperm(xt.shape[0], device=dev)
        for i in range(0, xt.shape[0], 8192):
            idx = perm[i:i + 8192]
            loss = (net(xt[idx]) - yt[idx]).norm(dim=-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    with torch.no_grad():
        return net(xe).cpu().numpy().astype(np.float64)


def main():
    x, y, s, base_dim = build(torch.load(sys.argv[1], map_location="cpu")["records"])
    print(f"samples {len(y)}  sequences {len(np.unique(s))}  base features {base_dim}  +B features {x.shape[1] - base_dim}")
    print(f"frozen-at-window-start bone change |dbone|: {100 * np.linalg.norm(y, axis=-1).mean():.2f} cm")
    folds = np.array_split(np.random.RandomState(0).permutation(np.unique(s)), 5)
    for name, cols in (("today's inputs", slice(0, base_dim)), ("+ forearm-vs-object rotation B", slice(0, x.shape[1]))):
        pred = np.zeros_like(y)
        for k, te_seqs in enumerate(folds):
            te = np.isin(s, te_seqs)
            pred[te] = fit(x[~te][:, cols], y[~te], x[te][:, cols], seed=k)
        print(f"[{name}] residual bone error: {100 * np.linalg.norm(y - pred, axis=-1).mean():.2f} cm")


if __name__ == "__main__":
    main()
