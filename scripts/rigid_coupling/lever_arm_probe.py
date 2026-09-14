#!/usr/bin/env python3
"""Feasibility probe: is the wrist->object lever arm observable from the two IMUs?

Physics.  During a rigid grasp the wrist joint w (an IMU acceleration site,
``_SENSOR_POS_INDICES`` 20/21) moves rigidly with the object, so

    p(t) = w(t) + R(t) c          with c constant over the contact segment,

where p is the object centre and R the object orientation (object IMU).  Both
IMU accelerations are linear filters S of world trajectories (``smooth_acceleration``),
hence

    a_obj(t) - a_wrist(t) = S[p](t) - S[w](t) = S[R](t) c

is a linear measurement of c whose coefficient matrix S[R] is computable from
the object IMU orientation alone.  The wrist acceleration in world frame is
recovered exactly from the body IMU features: a_k = R_root (imu_k + imu_root).

The palm used by the HOI metric is a forearm-fixed point (wrist + ratio * forearm),
so p - palm = R c - R_forearm d with d a canonical-skeleton constant, i.e. the
HOI error of a published object w + R c_hat equals ||c_hat - c_true||.

Reports, per dataset:
  * rigidity of the wrist anchor vs the palm anchor inside GT contact segments,
  * exact-model residual check (clean data), and
  * causal recursive-LS error curves of c (prior only / physics+prior, clean and
    with the eval IMU noise) vs time since contact onset and vs excitation.
Optionally uses an online dump to (i) take the model's FK proposal as the prior
and (ii) compare against the model's own FK relative-vector error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix  # noqa: E402
from utils.utils import smooth_acceleration  # noqa: E402
from utils.human_pose import PALM_EXTENSION_RATIOS  # noqa: E402

FPS = 30.0
WRIST = (20, 21)
ELBOW = (18, 19)
SENSOR = (4, 5)  # human_imu slots of the L/R forearm IMUs
BINS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 120), (120, 10 ** 9)]


def segments(mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def rot6d_to_mat(x6):
    return rotation_6d_to_matrix(torch.as_tensor(x6, dtype=torch.float64)).numpy()


def add_noise(imu, acc_std, rot_std, rng):
    out = imu.copy()
    out[..., :3] += rng.standard_normal(out[..., :3].shape) * acc_std
    out[..., 3:] += rng.standard_normal(out[..., 3:].shape) * rot_std
    return out


def world_from_features(human_imu, obj_imu):
    """Undo the DynaIP-style root normalisation: world wrist acc + world forearm rot."""
    R_root = rot6d_to_mat(human_imu[:, 0, 3:])                       # (T,3,3)
    a_root = np.einsum("tij,tj->ti", R_root, human_imu[:, 0, :3])
    a_w = np.stack([np.einsum("tij,tj->ti", R_root, human_imu[:, k, :3]) + a_root for k in SENSOR], 1)
    R_fore = np.stack([R_root @ rot6d_to_mat(human_imu[:, k, 3:]) for k in SENSOR], 1)
    a_o = obj_imu[:, :3]
    R_o = rot6d_to_mat(obj_imu[:, 3:])
    return a_w, R_fore, a_o, R_o


def smooth(x):
    return smooth_acceleration(torch.as_tensor(x, dtype=torch.float64), FPS, smooth_n=4).numpy()


def rls_curve(A, y, c0, lam, skip, rho=1.0):
    """Causal ridge LS over frames of one segment with forgetting factor rho.

    A: (n,3,m) design matrices, y: (n,3), c0: (m,) or (n,m) prior mean.
    Returns the per-frame estimate (n,m) and the excitation sqrt(lambda_min(H)).
    """
    n, _, m = A.shape
    c0 = np.broadcast_to(np.asarray(c0, dtype=np.float64), (n, m))
    H = np.zeros((n, m, m)); g = np.zeros((n, m))
    if n > skip:
        AtA = np.einsum("nij,nik->njk", A[skip:], A[skip:])
        Aty = np.einsum("nij,ni->nj", A[skip:], y[skip:])
        if rho >= 1.0:
            H[skip:] = np.cumsum(AtA, 0); g[skip:] = np.cumsum(Aty, 0)
        else:
            h = np.zeros((m, m)); gg = np.zeros(m)
            for i in range(skip, n):
                h = rho * h + AtA[i - skip]; gg = rho * gg + Aty[i - skip]
                H[i] = h; g[i] = gg
    c = np.linalg.solve(H + lam * np.eye(m), (g + lam * c0)[..., None])[..., 0]
    exc = np.sqrt(np.maximum(np.linalg.eigvalsh(H)[:, 0], 0.0))
    return c, exc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="full-sequence split dir (canonical .pt files)")
    ap.add_argument("--label", default="")
    ap.add_argument("--dump", default=None, help="online dump .pt to take the model prior / FK error from")
    ap.add_argument("--lams", type=float, nargs="+", default=[3.0, 10.0, 30.0])
    ap.add_argument("--skip", type=int, default=4, help="frames after onset excluded from the LS (filter leak)")
    ap.add_argument("--acc_std", type=float, default=0.2)
    ap.add_argument("--rot_std", type=float, default=0.005)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--rho", type=float, default=1.0, help="RLS forgetting factor (1 = constant lever arm)")
    ap.add_argument("--anchor", choices=["wrist", "palm", "joint"], default="wrist",
                    help="wrist: p=w+Rc; palm: p=palm+Rb (needs S[R_fore]d); joint: p=w+R_fore e+Rc (6 params)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
    if args.limit:
        files = files[: args.limit]
    dump = None
    if args.dump:
        dump = {r["seq_file"]: r for r in torch.load(args.dump, map_location="cpu", weights_only=False)["records"]}
    rng = np.random.default_rng(0)

    checks = {"acc_wrist": 0.0, "acc_obj": 0.0, "R_fore": 0.0, "palm": 0.0}
    rig = {"wrist": [], "palm": [], "resid_clean": [], "seg_len": [], "seg_rot_deg": []}
    # per-frame rows: [hand, t_since_onset, excitation, e_prior_pop, e_phys_clean(lams), e_phys_noisy(lams), e_model, e_phys_modelprior_noisy(lams)]
    rows = []
    d_vecs = []
    n_seq = 0
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
        a_w, R_fore, a_o, R_o = world_from_features(himu, oimu)
        # sanity: the feature inversion reproduces the world quantities
        checks["acc_wrist"] = max(checks["acc_wrist"], np.abs(a_w - smooth(w)).max())
        checks["acc_obj"] = max(checks["acc_obj"], np.abs(a_o - smooth(p)).max())
        checks["R_fore"] = max(checks["R_fore"], np.abs(R_fore - Rg[:, list(ELBOW)]).max())
        SR = smooth(R)  # (T,3,3): filter applied entry-wise
        SRf = smooth(R_fore)  # (T,2,3,3)
        # noisy copies (eval-style split Gaussian on the features)
        a_w_n, R_fore_n, a_o_n, R_o_n = world_from_features(
            add_noise(himu, args.acc_std, args.rot_std, rng), add_noise(oimu, args.acc_std, args.rot_std, rng))
        SR_n = smooth(R_o_n); SRf_n = smooth(R_fore_n)
        rec = dump.get(os.path.basename(f)) if dump else None
        if rec is not None:
            checks["palm"] = max(checks["palm"], np.abs(rec["gt_palm_pos"].double().numpy() - palm[: rec["publish_len"]]).max())
        for k, key in enumerate(("lhand_contact", "rhand_contact")):
            mask = x[key].numpy().astype(bool)
            d_vecs.append(np.einsum("tji,tj->ti", R_fore[:, k], palm[:, k] - w[:, k]))
            for a, b in segments(mask):
                if b - a + 1 < args.min_len:
                    continue
                sl = slice(a, b + 1)
                c_true = np.einsum("tji,tj->ti", R[sl], p[sl] - w[sl, k])      # R^T (p - w)
                b_true = np.einsum("tji,tj->ti", R[sl], p[sl] - palm[sl, k])   # R^T (p - palm)
                c_bar = c_true.mean(0)
                rig["wrist"].append(np.linalg.norm(c_true - c_bar, axis=1))
                rig["palm"].append(np.linalg.norm(b_true - b_true.mean(0), axis=1))
                rig["seg_len"].append(b - a + 1)
                rel = R[a].T @ R[sl]
                rig["seg_rot_deg"].append(np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))).max())
                y = a_o[sl] - a_w[sl, k]; A = SR[sl]
                inner = slice(args.skip, max(args.skip, (b - a + 1) - args.skip))
                if (b - a + 1) > 2 * args.skip:
                    rig["resid_clean"].append(np.linalg.norm(y[inner] - np.einsum("nij,j->ni", A[inner], c_bar), axis=1))
                y_n = a_o_n[sl] - a_w_n[sl, k]; A_n = SR_n[sl]
                d_k = np.einsum("tji,tj->ti", R_fore[:, k], palm[:, k] - w[:, k]).mean(0)  # canonical constant
                if args.anchor == "palm":
                    # p = palm + R b  with palm = w + R_fore d  ->  y - S[R_fore] d = S[R] b
                    y = y - np.einsum("nij,j->ni", SRf[sl, k], d_k); y_n = y_n - np.einsum("nij,j->ni", SRf_n[sl, k], d_k)
                    c_true = b_true
                elif args.anchor == "joint":
                    # p = w + R_fore e + R c  (6 unknowns): forearm-fixed + object-fixed parts
                    A = np.concatenate((SRf[sl, k], A), -1); A_n = np.concatenate((SRf_n[sl, k], A_n), -1)
                    c_true = np.concatenate((np.zeros_like(c_true), c_true), -1)
                c_bar = c_true.mean(0)
                n = b - a + 1
                t_since = np.arange(n)
                # priors: population (filled later, placeholder = zero, corrected below), model
                e_model = np.full(n, np.nan); c_model = None
                if rec is not None and b < rec["publish_len"]:
                    fk = rec["pred_lhand_obj_trans" if k == 0 else "pred_rhand_obj_trans"].double().numpy()[sl]
                    hands = rec["ot_hand_positions"].double().numpy()[sl, k]
                    r_model = fk - hands                                        # model FK relative vector (world)
                    e_model = np.linalg.norm(r_model - (p[sl] - palm[sl, k]), axis=1)
                    # convert the model's palm-anchored proposal to a wrist-anchored lever arm
                    if args.anchor == "wrist":
                        c_model = np.einsum("tji,tj->ti", R_o_n[sl], r_model + (palm[sl, k] - w[sl, k]))
                    elif args.anchor == "palm":
                        c_model = np.einsum("tji,tj->ti", R_o_n[sl], r_model)
                    else:
                        c_model = np.concatenate((np.zeros((n, 3)), np.einsum("tji,tj->ti", R_o_n[sl], r_model + (palm[sl, k] - w[sl, k]))), -1)
                block = [np.full(n, k), t_since, np.full(n, np.nan), np.full(n, np.nan)]
                _, exc = rls_curve(A_n, y_n, np.zeros(A_n.shape[-1]), args.lams[0], args.skip)
                block[2] = exc
                # store what is needed to evaluate priors later: c_true, A, y (clean/noisy), c_model
                rows.append({
                    "hand": k, "t": t_since, "exc": exc, "c_true": c_true, "c_bar": c_bar,
                    "A": A, "y": y, "A_n": A_n, "y_n": y_n, "e_model": e_model, "c_model": c_model,
                    "seq": os.path.basename(f), "obj": x.get("obj_name", ""),
                    "R": R[sl], "Rf": R_fore[sl, k], "d": d_k, "rel_true": p[sl] - palm[sl, k],
                })
        n_seq += 1

    print(f"== {args.label or args.data}: {n_seq} sequences, {len(rows)} contact segments | anchor={args.anchor} rho={args.rho} skip={args.skip}")
    print("   inversion checks (max abs): " + ", ".join(f"{k}={v:.2e}" for k, v in checks.items()))
    for k in (0, 1):
        d = np.concatenate(d_vecs[k::2], 0)
        print(f"   palm offset d[{'LR'[k]}] in forearm frame: mean {np.round(d.mean(0), 4)} m, std {np.round(d.std(0), 5)} (canonical constant)")
    wr = np.concatenate(rig["wrist"]); pa = np.concatenate(rig["palm"]); rc = np.concatenate(rig["resid_clean"]) if rig["resid_clean"] else np.zeros(1)
    sl_ = np.asarray(rig["seg_len"]); sr = np.asarray(rig["seg_rot_deg"])
    print(f"   rigidity |c_t - c_bar| (cm): wrist anchor mean {wr.mean()*100:.2f} / median {np.median(wr)*100:.2f} / p90 {np.percentile(wr,90)*100:.2f}"
          f" | palm anchor mean {pa.mean()*100:.2f} / median {np.median(pa)*100:.2f} / p90 {np.percentile(pa,90)*100:.2f}")
    print(f"   clean model residual |y - S[R] c_bar| (m/s^2): mean {rc.mean():.3f} / median {np.median(rc):.3f} / p90 {np.percentile(rc,90):.3f}")
    print(f"   segments: len median {np.median(sl_):.0f} frames (p10 {np.percentile(sl_,10):.0f}, p90 {np.percentile(sl_,90):.0f});"
          f" max in-segment rotation median {np.median(sr):.1f} deg (p25 {np.percentile(sr,25):.1f}, p75 {np.percentile(sr,75):.1f})")

    # population prior per hand (frame-weighted mean of c_bar) -- a deliberately weak prior
    pop = [np.average(np.stack([r["c_bar"] for r in rows if r["hand"] == k]), 0,
                      weights=[len(r["t"]) for r in rows if r["hand"] == k]) for k in (0, 1)]
    print(f"   population prior c0 (m): L {np.round(pop[0],3)}  R {np.round(pop[1],3)}")

    def hoi_err(r, c):
        """HOI error (m) of publishing the object from the estimated parameters with GT hands."""
        if c.shape[-1] == 6:
            # p - palm = R_fore (e - d) + R c ; truth uses the GT relative vector
            rel = np.einsum("tij,tj->ti", r["Rf"], c[:, :3] - r["d"]) + np.einsum("tij,tj->ti", r["R"], c[:, 3:])
            return np.linalg.norm(rel - r["rel_true"], axis=1)
        return np.linalg.norm(c - r["c_true"], axis=1)

    def evaluate(prior_fn, use_noisy, lam):
        errs, ts, ex, hands = [], [], [], []
        for r in rows:
            c0 = prior_fn(r)
            if c0 is None:
                continue
            A, y = (r["A_n"], r["y_n"]) if use_noisy else (r["A"], r["y"])
            c, exc = rls_curve(A, y, c0, lam, args.skip, args.rho)
            errs.append(hoi_err(r, c)); ts.append(r["t"]); ex.append(exc); hands.append(np.full(len(r["t"]), r["hand"]))
        return np.concatenate(errs), np.concatenate(ts), np.concatenate(ex), np.concatenate(hands)

    def prior_only(prior_fn):
        errs, ts = [], []
        for r in rows:
            c0 = prior_fn(r)
            if c0 is None:
                continue
            c0 = np.broadcast_to(np.asarray(c0), r["c_true"].shape)
            errs.append(hoi_err(r, c0)); ts.append(r["t"])
        return np.concatenate(errs), np.concatenate(ts)

    def bins_line(err, t, name):
        parts = [f"{name:<34s} all {err.mean()*100:6.2f}"]
        for lo, hi in BINS:
            m = (t >= lo) & (t < hi)
            parts.append(f"| {lo:>3d}-{min(hi, 999):<3d} {err[m].mean()*100 if m.any() else float('nan'):5.2f}")
        print("   " + " ".join(parts))

    print("   --- lever-arm error (cm) == HOI error with GT hands; columns = frames since onset ---")
    e, t = prior_only(lambda r: pop[r["hand"]]); bins_line(e, t, "prior only (population)")
    for lam in args.lams:
        e, t, _, _ = evaluate(lambda r: pop[r["hand"]], False, lam); bins_line(e, t, f"physics+pop prior, clean, lam={lam:g}")
    for lam in args.lams:
        e, t, ex, _ = evaluate(lambda r: pop[r["hand"]], True, lam); bins_line(e, t, f"physics+pop prior, NOISY, lam={lam:g}")
    # error vs excitation (noisy, middle lambda)
    lam = args.lams[len(args.lams) // 2]
    e, t, ex, _ = evaluate(lambda r: pop[r["hand"]], True, lam); e0, _ = prior_only(lambda r: pop[r["hand"]])
    qs = np.quantile(ex[ex > 0], [0.25, 0.5, 0.75]) if (ex > 0).any() else np.array([0, 0, 0])
    edges = [-1, 0, qs[0], qs[1], qs[2], np.inf]
    print(f"   --- noisy physics+pop prior (lam={lam:g}) by excitation sqrt(lambda_min(H)) [quartiles {np.round(qs,2)}] ---")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ex > lo) & (ex <= hi)
        if m.any():
            print(f"      exc in ({lo:.2f},{hi:.2f}]: frames {m.sum():7d} ({m.mean()*100:4.1f}%)  prior-only {e0[m].mean()*100:5.2f}  physics {e[m].mean()*100:5.2f}")
    if dump is not None:
        have = [r for r in rows if r["c_model"] is not None]
        print(f"   --- with model dump ({len(have)} segments) ---")
        em = np.concatenate([r["e_model"] for r in have]); tm = np.concatenate([r["t"] for r in have])
        bins_line(em, tm, "model FK relative-vector error")
        e, t = prior_only(lambda r: r["c_model"]); bins_line(e, t, "model proposal as c (GT hands)")
        for lam in args.lams:
            e, t, _, _ = evaluate(lambda r: r["c_model"], True, lam); bins_line(e, t, f"physics + model prior(t), NOISY, lam={lam:g}")
        for lam in args.lams:
            e, t, _, _ = evaluate(lambda r: None if r["c_model"] is None else r["c_model"][min(args.skip, len(r["t"]) - 1)], True, lam)
            bins_line(e, t, f"physics + model onset prior, NOISY, lam={lam:g}")
    if args.out:
        np.savez_compressed(args.out, rows=np.array(rows, dtype=object), pop=np.stack(pop), allow_pickle=True)


if __name__ == "__main__":
    main()
