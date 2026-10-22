#!/usr/bin/env python3
"""Would down-weighting the sliding hand in bimanual fusion recover HOI?  Publisher-level oracle replays.

On GT bimanual frames the gate's total FK weight (w_L + w_R) is kept and only its split between the two hands is
changed; the evaluator's causal re-fusion is replayed so the IMU/static recursion follows the new track.  The FK bones
themselves stay as the model produced them (no closed-loop effect on the bones -> first-order estimate).

Splits:
  model      : the learned gate
  equal      : 0.5 / 0.5
  slide_hard : follow the hand whose GT bone (object-frame palm->object vector) moves slower (the non-sliding hand)
  slide_soft : share_L = sigmoid((speed_R - speed_L) / tau)
  err_frame  : follow the hand whose own FK bone error is smaller, per frame      (best possible reliability oracle)
  err_segment: same, one choice per bimanual segment (L, R or equal)

    python scripts/diag/sliding_hand_gate_probe.py dump.pt
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


def smooth(x, k, causal):
    out = np.zeros_like(x)
    for t in range(len(x)):
        lo = max(0, t - k)
        hi = t + 1 if causal else min(len(x), t + k + 1)
        out[t] = x[lo:hi].mean()
    return out


def publish(fk_l, fk_r, imu, static, w):
    n = len(w)
    prev = static[0].copy()
    pub = np.zeros((n, 3))
    for t in range(n):
        cur = w[t, 0] * fk_l[t] + w[t, 1] * fk_r[t] + w[t, 2] * (prev + imu[t] - static[t]) + w[t, 3] * prev
        pub[t] = cur
        prev = cur
    return pub


def main():
    records = torch.load(sys.argv[1], map_location="cpu")["records"]
    names = ("model", "equal", "slide_hard", "slide_soft", "err_frame", "err_segment")
    seq_hoi = {k: [] for k in names}
    bi = {k: [] for k in names}
    stats = {k: [] for k in ("speed_l", "speed_r", "err_l", "err_r", "asym_slide", "one_slides")}
    for r in records:
        n = int(r["publish_len"])
        if not r.get("has_object", True) or n == 0:
            continue
        hs = np.asarray(r.get("stage1_pred_palm_glb_pos", r["pred_palm_glb_pos"])[:n], dtype=np.float64)
        hr = np.asarray(r["pred_palm_glb_pos"][:n], dtype=np.float64)
        g = np.asarray(r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n], dtype=np.float64)
        p_gt = np.asarray(r["gt_obj_trans"][:n], dtype=np.float64)
        R = rot6d_to_matrix(np.asarray(r["gt_obj_rot"][:n], dtype=np.float64))
        fk = [hr[:, h] + (np.asarray(r[f"pred_{s}hand_obj_trans"][:n], dtype=np.float64) - hs[:, h]) for h, s in ((0, "l"), (1, "r"))]
        imu, static = np.asarray(r["pred_imu_obj_trans"][:n], dtype=np.float64), np.asarray(r["pred_static_obj_trans"][:n], dtype=np.float64)
        w = np.asarray(r["gating_weights"][:n], dtype=np.float64)
        cl, cr = np.asarray(r["gt_lhand_contact"][:n]).astype(bool), np.asarray(r["gt_rhand_contact"][:n]).astype(bool)
        both = cl & cr
        rel_gt = p_gt[:, None] - g
        # GT bone speed per hand (object frame, m/s), smoothed over +-7 frames
        bone = [np.einsum("tji,tj->ti", R, rel_gt[:, h]) for h in (0, 1)]
        speed = [smooth(np.concatenate(([0.0], np.linalg.norm(np.diff(b, axis=0), axis=-1) * 30.0)), 7, causal=False) for b in bone]
        # own-hand FK bone error (what HOI_h would be if the object followed hand h exactly)
        err = [np.linalg.norm((fk[h] - hr[:, h]) - rel_gt[:, h], axis=-1) for h in (0, 1)]
        shares = {"model": None, "equal": np.full(n, 0.5)}
        shares["slide_hard"] = (speed[0] < speed[1]).astype(np.float64)
        shares["slide_soft"] = 1.0 / (1.0 + np.exp(-(speed[1] - speed[0]) / 0.03))
        shares["err_frame"] = (err[0] < err[1]).astype(np.float64)
        seg_share = np.full(n, 0.5)
        t = 0
        while t < n:
            if not both[t]:
                t += 1
                continue
            s = t
            while t < n and both[t]:
                t += 1
            el, er = err[0][s:t].mean(), err[1][s:t].mean()
            ea = np.linalg.norm(0.5 * ((fk[0] - hr[:, 0]) - rel_gt[:, 0] + (fk[1] - hr[:, 1]) - rel_gt[:, 1]), axis=-1)[s:t].mean()
            seg_share[s:t] = {0: 1.0, 1: 0.0, 2: 0.5}[int(np.argmin((el, er, ea)))]
        shares["err_segment"] = seg_share
        if both.any():
            stats["speed_l"].append(speed[0][both]); stats["speed_r"].append(speed[1][both])
            stats["err_l"].append(err[0][both]); stats["err_r"].append(err[1][both])
            hi, lo = np.maximum(speed[0], speed[1])[both], np.minimum(speed[0], speed[1])[both]
            stats["asym_slide"].append(((hi > 0.05) & (hi > 2.0 * lo)).astype(np.float64))
            stats["one_slides"].append(((hi > 0.05) & (lo <= 0.05)).astype(np.float64))
        for name in names:
            w_new = w.copy()
            if shares[name] is not None:
                total = w[:, 0] + w[:, 1]
                w_new[both, 0] = shares[name][both] * total[both]
                w_new[both, 1] = (1.0 - shares[name][both]) * total[both]
            pub = publish(fk[0], fk[1], imu, static, w_new)
            e = [np.linalg.norm((pub - hr[:, h]) - rel_gt[:, h], axis=-1) * 100.0 for h in (0, 1)]
            errs = np.concatenate((e[0][cl], e[1][cr]))
            if errs.size:
                seq_hoi[name].append(errs.mean())
            if both.any():
                bi[name].append(0.5 * (e[0] + e[1])[both])
    cat = np.concatenate
    sl, sr = cat(stats["speed_l"]), cat(stats["speed_r"])
    el, er = cat(stats["err_l"]) * 100, cat(stats["err_r"]) * 100
    print(f"bimanual frames: GT bone speed L/R mean {sl.mean() * 100:.1f}/{sr.mean() * 100:.1f} cm/s, corr(speed_L, speed_R) {np.corrcoef(sl, sr)[0, 1]:.2f};"
          f" one hand slides (>5 cm/s) while the other holds: {cat(stats['one_slides']).mean() * 100:.1f}% of frames; asymmetric (>2x): {cat(stats['asym_slide']).mean() * 100:.1f}%")
    print(f"own-hand FK bone error L/R mean {el.mean():.2f}/{er.mean():.2f} cm, corr {np.corrcoef(el, er)[0, 1]:.2f}; |err_L - err_R| mean {np.abs(el - er).mean():.2f} cm;"
          f" corr(speed_h, err_h): L {np.corrcoef(sl, el)[0, 1]:.2f}  R {np.corrcoef(sr, er)[0, 1]:.2f};"
          f" corr(speed_L - speed_R, err_L - err_R) {np.corrcoef(sl - sr, el - er)[0, 1]:.2f}")
    print("| split | official-style HOI | bimanual-frame pair-HOI |\n|---|---|---|")
    for name in names:
        print(f"| {name} | {np.mean(seq_hoi[name]):.3f} | {cat(bi[name]).mean():.3f} |")


if __name__ == "__main__":
    main()
