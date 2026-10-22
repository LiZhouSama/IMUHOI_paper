#!/usr/bin/env python3
"""Is the bimanual differential hand error (span error) learnable?  Training-free probe on an OT dump.

On GT bimanual frames the HOI error splits into a common part c (object vs. the hand pair) and a
differential part d = R (v_pred - v_gt) / 2 with v = R^T (h_R - h_L) the inter-hand vector in the
object frame.  d depends on the hands only, so no object-side module can reduce it.  This probe asks
how much of the span error delta_v = v_pred - v_gt is

  * a per-segment constant (bias) vs. an in-segment fluctuation (what object-frame rigidity can see), and
  * predictable from state-free, test-time features (5-fold CV by sequence, small MLP), with and
    without causal object-frame EMA features of the predicted span (the rigidity signal).

    python scripts/diag/bimanual_span_probe.py dump.pt
"""
import sys

import numpy as np
import torch
import torch.nn as nn


def rot6d_to_matrix(x):
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True).clip(1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2)  # rows = basis vectors (pytorch3d convention)


def causal_ema(x, alpha):
    out = np.zeros_like(x)
    acc = x[0].copy()
    for t in range(x.shape[0]):
        acc = (1 - alpha) * acc + alpha * x[t]
        out[t] = acc
    return out


def build(records, min_seg=15):
    feats, targets, seq_ids, seg_ids, c_vec, q_pack = [], [], [], [], [], []
    seg_counter = 0
    for si, r in enumerate(records):
        n = int(r["publish_len"])
        if not r.get("has_object", True) or n == 0:
            continue
        cl = np.asarray(r["gt_lhand_contact"][:n]).astype(bool)
        cr = np.asarray(r["gt_rhand_contact"][:n]).astype(bool)
        both = cl & cr
        if not both.any():
            continue
        hs = np.asarray(r.get("stage1_pred_palm_glb_pos", r["pred_palm_glb_pos"])[:n], dtype=np.float64)
        hr = np.asarray(r["pred_palm_glb_pos"][:n], dtype=np.float64)
        g = np.asarray(r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n], dtype=np.float64)
        R = rot6d_to_matrix(np.asarray(r["gt_obj_rot"][:n], dtype=np.float64))
        to_obj = lambda v: np.einsum("tij,tj->ti", np.transpose(R, (0, 2, 1)), v)  # noqa: E731  R^T v
        v_pred = to_obj(hs[:, 1] - hs[:, 0])
        v_gt = to_obj(g[:, 1] - g[:, 0])
        dv = v_pred - v_gt
        imu = np.asarray(r["gt_human_imu"][:n], dtype=np.float64)  # [n,6,9]: root, legs, head, L/R forearm
        R_root = rot6d_to_matrix(imu[:, 0, 3:9])
        arm6d = imu[:, 4:6, 3:9].reshape(n, 12)       # root-relative forearm orientation
        arm_acc = imu[:, 4:6, :3].reshape(n, 6)
        root = np.asarray(r.get("stage1_root_trans_pred", r["root_trans_pred"])[:n], dtype=np.float64)
        rel = np.einsum("tij,thj->thi", np.transpose(R_root, (0, 2, 1)), hs - root[:, None]).reshape(n, 6)  # palms in the root frame
        prob = np.asarray(r["pred_hand_contact_prob"][:n], dtype=np.float64)[:, :3]
        obj6d = np.asarray(r["gt_obj_rot"][:n], dtype=np.float64)
        ema_f, ema_s = causal_ema(v_pred, 0.2), causal_ema(v_pred, 0.03)
        base = np.concatenate((v_pred, np.linalg.norm(v_pred, axis=-1, keepdims=True), rel, arm6d, arm_acc, obj6d, prob), axis=-1)
        temporal = np.concatenate((v_pred - ema_f, v_pred - ema_s, ema_s), axis=-1)
        p_gt = np.asarray(r["gt_obj_trans"][:n], dtype=np.float64)
        pub = np.asarray(r["pred_obj_trans_published"][:n], dtype=np.float64)
        q_l = (pub - hr[:, 0]) - (p_gt - g[:, 0])
        q_r = (pub - hr[:, 1]) - (p_gt - g[:, 1])
        seg = np.zeros(n, dtype=np.int64) - 1
        t = 0
        while t < n:
            if both[t]:
                s = t
                while t < n and both[t]:
                    t += 1
                if t - s >= min_seg:
                    seg[s:t] = seg_counter
                    seg_counter += 1
            else:
                t += 1
        m = seg >= 0
        if not m.any():
            continue
        feats.append(np.concatenate((base, temporal), axis=-1)[m])
        targets.append(dv[m])
        seq_ids.append(np.full(int(m.sum()), si))
        seg_ids.append(seg[m])
        c_vec.append(0.5 * (q_l + q_r)[m])
        q_pack.append(np.einsum("tij,tj->ti", R, dv)[m] * 0.0)  # placeholder (keeps shapes aligned)
    cat = lambda xs: np.concatenate(xs, axis=0)  # noqa: E731
    return cat(feats), cat(targets), cat(seq_ids), cat(seg_ids), cat(c_vec), base.shape[-1]


def fit_predict(x_tr, y_tr, x_te, epochs=40, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)
    xe = torch.tensor((x_te - mu) / sd, dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Linear(xt.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 3)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        perm = torch.randperm(xt.shape[0], device=dev)
        for i in range(0, xt.shape[0], 4096):
            idx = perm[i:i + 4096]
            loss = (net(xt[idx]) - yt[idx]).norm(dim=-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        return net(xe).cpu().numpy().astype(np.float64)


def pair_hoi(c, d):
    return 50.0 * (np.linalg.norm(c + d, axis=-1) + np.linalg.norm(c - d, axis=-1))


def main():
    records = torch.load(sys.argv[1], map_location="cpu")["records"]
    x, dv, seq, seg, c, n_base = build(records)
    print(f"bimanual frames (segments >= 15 f): {len(dv)}  segments: {len(np.unique(seg))}  sequences: {len(np.unique(seq))}")
    d_norm = 50.0 * np.linalg.norm(dv, axis=-1)
    print(f"|d| = |delta_v|/2: mean {d_norm.mean():.2f} cm  median {np.median(d_norm):.2f}")
    # bias vs fluctuation inside each bimanual segment
    seg_mean = np.zeros_like(dv)
    for s in np.unique(seg):
        m = seg == s
        seg_mean[m] = dv[m].mean(0)
    print(f"  per-segment constant part |mean_seg(delta_v)|/2: {50.0 * np.linalg.norm(seg_mean, axis=-1).mean():.2f} cm"
          f"   in-segment fluctuation |delta_v - mean|/2: {50.0 * np.linalg.norm(dv - seg_mean, axis=-1).mean():.2f} cm")
    # d is expressed in the object frame here; rotate c consistently is unnecessary for norms of c +/- d only if both
    # live in one frame, so bring d to the world frame through the stored rotation-free identity: use norms via Gram terms.
    # (c is world-frame; |c +/- d| needs d in world frame -> recompute below with R applied in build() if required.)
    folds = np.array_split(np.random.RandomState(0).permutation(np.unique(seq)), 5)
    for name, cols in (("state-free features", slice(0, n_base)), ("+ causal object-frame EMA of the span", slice(0, x.shape[1]))):
        pred = np.zeros_like(dv)
        for k, test_seqs in enumerate(folds):
            te = np.isin(seq, test_seqs)
            pred[te] = fit_predict(x[~te][:, cols], dv[~te], x[te][:, cols], seed=k)
        res = dv - pred
        print(f"[{name}] 5-fold CV: |d| {d_norm.mean():.2f} -> {50.0 * np.linalg.norm(res, axis=-1).mean():.2f} cm"
              f"   (constant-prediction baseline: {50.0 * np.linalg.norm(dv - dv.mean(0), axis=-1).mean():.2f})")
        np.save(f"/tmp/claude-1000/-mnt-d-a-WORK-Projects-PhD-tasks-IMUHOI/aae0cd67-2bbe-4cdc-b16a-fa22cbf81d94/scratchpad/span_probe_pred_{'base' if cols.stop == n_base else 'ema'}.npy", pred)


if __name__ == "__main__":
    main()
