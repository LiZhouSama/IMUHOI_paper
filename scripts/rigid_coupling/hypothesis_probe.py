#!/usr/bin/env python3
"""Can the two-IMU rigid-coupling residual pick the right grasp hypothesis online?

Grasp offsets are multi-modal (seat vs backrest of a chair, edge vs centre of a
table): modes differ by 10-30 cm, which is exactly the scale at which the
lever-arm physics of ``lever_arm_probe.py`` is informative even though its
absolute accuracy is limited by non-rigid contact.  This probe builds a set of
candidate palm-anchored offsets b_k (k-means over TRAIN contact frames, per
object+hand when the identity is known, or one object-agnostic set) and, on
TEST contact segments, scores every candidate causally with the accumulated
residual  sum_s || y'_s - S[R]_s b_k ||^2,  y' = a_obj - a_wrist - S[R_fore] d.

Reported per frames-since-onset (cm, == HOI error with GT hands):
  prior mode   : most populated candidate (what a regressor without evidence would pick)
  oracle       : candidate nearest to the true offset at that frame (quantisation floor)
  phys argmin  : causal minimum-residual candidate
  phys post    : posterior-weighted mean with prior weights pi_k and residual likelihood
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from lever_arm_probe import BINS, ELBOW, SENSOR, WRIST, add_noise, segments, smooth, world_from_features  # noqa: E402
from utils.human_pose import PALM_EXTENSION_RATIOS  # noqa: E402


def kmeans(X, K, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) <= K:
        return X.copy(), np.ones(len(X)) / len(X)
    C = X[rng.choice(len(X), K, replace=False)]
    for _ in range(iters):
        a = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        for k in range(K):
            if (a == k).any():
                C[k] = X[a == k].mean(0)
    a = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
    pi = np.bincount(a, minlength=K).astype(float); pi /= pi.sum()
    return C, pi


def load_offsets(data_dir, stride=3, limit=0):
    """Palm-anchored offsets in the object frame on GT contact frames, keyed by (obj, hand)."""
    out = defaultdict(list)
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    if limit:
        files = files[:limit]
    for f in files:
        x = torch.load(f, map_location="cpu", weights_only=False)
        if not x.get("has_object", True):
            continue
        p = x["obj_trans"].double().numpy(); R = x["obj_rot"].double().numpy()
        J = x["position_global_full_gt_world"].double().numpy()
        w = J[:, list(WRIST)]; e = J[:, list(ELBOW)]
        palm = w + (w - e) * np.asarray(PALM_EXTENSION_RATIOS)[None, :, None]
        for k, key in enumerate(("lhand_contact", "rhand_contact")):
            m = x[key].numpy().astype(bool)
            if not m.any():
                continue
            b = np.einsum("tji,tj->ti", R[m], p[m] - palm[m, k])[::stride]
            out[(x.get("obj_name", "?"), k)].append(b)
    return {key: np.concatenate(v, 0) for key, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="process/IMUHOI++cano/processed_split_data_OMOMO_full/train")
    ap.add_argument("--test", default="process/IMUHOI++cano/processed_split_data_OMOMO_full/test")
    ap.add_argument("--label", default="")
    ap.add_argument("--K", type=int, default=6, help="candidates per (object, hand)")
    ap.add_argument("--K_global", type=int, default=24, help="candidates per hand for the object-agnostic set")
    ap.add_argument("--identity", choices=["known", "agnostic"], default="known")
    ap.add_argument("--sigma", type=float, default=0.8, help="residual scale (m/s^2) for the posterior")
    ap.add_argument("--skip", type=int, default=4)
    ap.add_argument("--acc_std", type=float, default=0.2)
    ap.add_argument("--rot_std", type=float, default=0.005)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dump", default=None, help="online dump: adds the model's proposal as an extra candidate")
    args = ap.parse_args()

    offs = load_offsets(args.train, limit=args.limit_train)
    cands = {}
    if args.identity == "known":
        for key, X in offs.items():
            cands[key] = kmeans(X, args.K)
    for k in (0, 1):
        X = np.concatenate([v for (o, h), v in offs.items() if h == k], 0)
        cands[("*", k)] = kmeans(X, args.K_global)
    seps = [np.linalg.norm(C[:, None] - C[None], axis=-1)[np.triu_indices(len(C), 1)].mean() for C, _ in cands.values() if len(C) > 1]
    print(f"== {args.label or args.test}: identity={args.identity} K={args.K}/{args.K_global}; "
          f"{len(offs)} (object,hand) groups in train; mean pairwise candidate distance {np.mean(seps)*100:.1f} cm")

    dump = None
    if args.dump:
        dump = {r["seq_file"]: r for r in torch.load(args.dump, map_location="cpu", weights_only=False)["records"]}
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(args.test, "*.pt")))
    if args.limit:
        files = files[: args.limit]
    names = ["prior mode", "oracle nearest", "phys argmin (causal)", "phys posterior mean", "phys argmin + prior"]
    if dump is not None:
        names += ["model proposal", "model + phys posterior", "model+cands phys argmin"]
    errs = {n: [] for n in names}; ts = []; hit = defaultdict(list); n_seg = 0; n_seq = 0
    for f in files:
        x = torch.load(f, map_location="cpu", weights_only=False)
        if not x.get("has_object", True):
            continue
        p = x["obj_trans"].double().numpy(); R = x["obj_rot"].double().numpy()
        J = x["position_global_full_gt_world"].double().numpy(); Rg = x["rotation_global"].double().numpy()
        T = p.shape[0]
        if T < 12:
            continue
        himu = x["human_imu"].double().numpy(); oimu = x["obj_imu"].double().numpy()
        w = J[:, list(WRIST)]; e = J[:, list(ELBOW)]
        palm = w + (w - e) * np.asarray(PALM_EXTENSION_RATIOS)[None, :, None]
        a_w, R_fore, a_o, R_o = world_from_features(add_noise(himu, args.acc_std, args.rot_std, rng),
                                                    add_noise(oimu, args.acc_std, args.rot_std, rng))
        SR = smooth(R_o); SRf = smooth(R_fore)
        rec = dump.get(os.path.basename(f)) if dump else None
        obj = x.get("obj_name", "?")
        n_seq += 1
        for k, key in enumerate(("lhand_contact", "rhand_contact")):
            mask = x[key].numpy().astype(bool)
            C, pi = cands.get((obj, k), cands[("*", k)]) if args.identity == "known" else cands[("*", k)]
            d_k = np.einsum("tji,tj->ti", Rg[:, ELBOW[k]], palm[:, k] - w[:, k]).mean(0)
            for a, b in segments(mask):
                n = b - a + 1
                if n < args.min_len:
                    continue
                sl = slice(a, b + 1)
                b_true = np.einsum("tji,tj->ti", R[sl], p[sl] - palm[sl, k])
                y = a_o[sl] - a_w[sl, k] - np.einsum("nij,j->ni", SRf[sl, k], d_k)
                A = SR[sl]
                # cumulative residual for every candidate: sum ||y - A b||^2 = b^T H b - 2 g^T b + yy
                H = np.zeros((n, 3, 3)); g = np.zeros((n, 3)); yy = np.zeros(n)
                if n > args.skip:
                    H[args.skip:] = np.cumsum(np.einsum("nij,nik->njk", A[args.skip:], A[args.skip:]), 0)
                    g[args.skip:] = np.cumsum(np.einsum("nij,ni->nj", A[args.skip:], y[args.skip:]), 0)
                    yy[args.skip:] = np.cumsum((y[args.skip:] ** 2).sum(-1))
                Cc = C
                if rec is not None and b < rec["publish_len"]:
                    fk = rec["pred_lhand_obj_trans" if k == 0 else "pred_rhand_obj_trans"].double().numpy()[sl]
                    hands = rec["ot_hand_positions"].double().numpy()[sl, k]
                    b_model = np.einsum("tji,tj->ti", R_o[sl], fk - hands)        # (n,3) per-frame proposal
                else:
                    b_model = None
                res = np.einsum("nij,ki,kj->nk", H, Cc, Cc) - 2 * np.einsum("nj,kj->nk", g, Cc) + yy[:, None]  # (n,K)
                t = np.arange(n)
                e_prior = np.linalg.norm(Cc[np.argmax(pi)][None] - b_true, axis=1)
                dist = np.linalg.norm(Cc[None] - b_true[:, None], axis=-1)          # (n,K)
                e_oracle = dist.min(1)
                sel = np.argmin(res, 1)
                e_arg = dist[t, sel]
                logw = -res / (2 * args.sigma ** 2)
                wpost = np.exp(logw - logw.max(1, keepdims=True)); wpost /= wpost.sum(1, keepdims=True)
                e_post = np.linalg.norm(wpost @ Cc - b_true, axis=1)
                logwp = logw + np.log(pi + 1e-9)[None]
                e_argp = dist[t, np.argmax(logwp, 1)]
                for nme, val in zip(names[:5], (e_prior, e_oracle, e_arg, e_post, e_argp)):
                    errs[nme].append(val)
                if b_model is not None:
                    e_model = np.linalg.norm(b_model - b_true, axis=1)
                    errs["model proposal"].append(e_model)
                    # model proposal as an extra candidate with its own running residual
                    res_m = np.einsum("nij,ni,nj->n", H, b_model, b_model) - 2 * np.einsum("nj,nj->n", g, b_model) + yy
                    allres = np.concatenate((res, res_m[:, None]), 1)
                    alld = np.concatenate((dist, e_model[:, None]), 1)
                    errs["model+cands phys argmin"].append(alld[t, np.argmin(allres, 1)])
                    # posterior over candidates, prior centred on the model proposal (Gaussian, 8 cm)
                    prior_m = -np.linalg.norm(Cc[None] - b_model[:, None], axis=-1) ** 2 / (2 * 0.10 ** 2)
                    lw = -res / (2 * args.sigma ** 2) + prior_m
                    wm = np.exp(lw - lw.max(1, keepdims=True)); wm /= wm.sum(1, keepdims=True)
                    errs["model + phys posterior"].append(np.linalg.norm(wm @ Cc - b_true, axis=1))
                elif dump is not None:
                    for nme in names[5:]:
                        errs[nme].append(np.full(n, np.nan))
                ts.append(t)
                orc = np.argmin(dist, 1)
                for tt in (15, 30, 60):
                    if n > tt:
                        hit[tt].append(sel[tt] == orc[tt])
                hit["end"].append(sel[-1] == orc[-1])
                n_seg += 1
    t = np.concatenate(ts)
    print(f"   {n_seq} sequences, {n_seg} segments; frames {len(t)}")
    print("   error (cm) by frames since onset:")
    for nme in names:
        e = np.concatenate(errs[nme])
        parts = [f"   {nme:<26s} all {np.nanmean(e)*100:6.2f}"]
        for lo, hi in BINS:
            m = (t >= lo) & (t < hi)
            parts.append(f"| {lo:>3d}-{min(hi,999):<3d} {np.nanmean(e[m])*100 if m.any() else float('nan'):5.2f}")
        print(" ".join(parts))
    print("   phys argmin == oracle candidate: " + ", ".join(f"t={k}: {np.mean(v)*100:.0f}%" for k, v in hit.items()))


if __name__ == "__main__":
    main()
