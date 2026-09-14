#!/usr/bin/env python3
"""Replay mesh-based corrections on the real online pipeline predictions.

Uses the frozen v3_gate online test cache (1015 OMOMO sequences, pre-canonical
schema) from hoi_contact_consistency_20260905/test_online.pt.

  1. per-axis error share of the published object (is vertical error large?)
  2. how far off the mesh surface are the model's bone tips vs GT tips
  3. replay the fusion with the bone tip snapped to the surface (hard / soft)
  4. floor-support correction on static frames (rule detector / oracle)
  5. mesh as a hand landmark at contact onset: snap the predicted palm to the
     surface of the object at its (predicted) pre-onset pose
All errors in cm, seq-mean unless stated.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_nn import SurfaceNN, load_mesh  # noqa: E402
from scripts.analyze_ot_dumps import hysteresis, rot6d, segments_of  # noqa: E402

CACHE = os.path.join(ROOT, "outputs/IMUHOI++/RNN2/hoi_contact_consistency_20260905/test_online.pt")
OLD_TEST = os.path.join(ROOT, "process/IMUHOI++/processed_split_data_OMOMO/test")
OUT = os.path.join(ROOT, "outputs/analysis/mesh_probe_20260907", "mesh_replay_pred.json")
FPS = 30.0
ON, OFF = 0.15, 0.075
DEV = "cuda"


def norm(x):
    return np.linalg.norm(x, axis=-1)


def seq_mean(per_seq):
    return float(np.mean([v for v in per_seq if np.isfinite(v)]))


def hoi_err(p_obj, h_pred, p_gt, h_gt, c_gt):
    errs = []
    for k in range(2):
        m = c_gt[:, k]
        if m.any():
            errs.append(norm((p_obj - h_pred[:, k]) - (p_gt - h_gt[:, k]))[m])
    if not errs:
        return float("nan")
    return float(np.concatenate(errs).mean() * 100)


def replay(rec, R, h, bones, prev0, v, w):
    """Fusion replay with fixed gating weights and given per-frame bones [T,2,3]."""
    T = R.shape[0]
    out = np.zeros((T, 3))
    prev = prev0.astype(np.float64).copy()
    for t in range(T):
        pL = h[t, 0] + R[t] @ bones[t, 0]
        pR = h[t, 1] + R[t] @ bones[t, 1]
        p = w[t, 0] * pL + w[t, 1] * pR + w[t, 2] * (prev + v[t] / FPS) + w[t, 3] * prev
        out[t] = p
        prev = p
    return out


def imu_static(R, acc, k=5, ang_deg=0.6, acc_thr=0.35):
    T = R.shape[0]
    st = np.zeros(T, dtype=bool)
    for t in range(T):
        t0 = max(t - k, 0)
        rel = R[t].T @ R[t0]
        ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
        st[t] = (ang < ang_deg) and (norm(acc[t0 : t + 1]).max() < acc_thr)
    return st


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    recs = cache["records"]
    nn = {}
    res = defaultdict(list)
    per_obj = defaultdict(lambda: defaultdict(list))
    onset_rows = []
    tipdist_model, tipdist_gt = [], []
    wrist_offsets = {0: [], 1: []}
    forearm_offsets = {0: [], 1: []}
    for ri, rec in enumerate(recs):
        name = rec["obj_name"]
        if name not in nn:
            nn[name] = SurfaceNN(load_mesh(name), device=DEV)
        q = nn[name]
        old = torch.load(os.path.join(OLD_TEST, rec["seq_file"]), map_location="cpu", weights_only=False)
        T = int(rec["publish_len"])
        s = old["obj_scale"][:T].double().numpy()
        p_gt = rec["gt_obj_trans"].numpy().astype(np.float64)
        R_gt = rot6d(rec["gt_obj_rot"]).astype(np.float64)
        R = rot6d(rec["gt_obj_imu"][:, 3:9]).astype(np.float64)
        acc = rec["gt_obj_imu"][:, :3].numpy()
        h_gt = rec["gt_palm_pos"].numpy().astype(np.float64)
        h = rec["ot_hand_positions"].numpy().astype(np.float64)          # hands OT consumed
        h_pub = rec["pred_palm_glb_pos"].numpy().astype(np.float64)      # refined hands used by eval
        c_gt = np.stack((rec["gt_lhand_contact"].numpy(), rec["gt_rhand_contact"].numpy()), 1).astype(bool)
        moving_gt = rec["gt_obj_contact"].numpy().astype(bool)
        prob = rec["pred_hand_contact_prob"].numpy().astype(np.float64)
        w = rec["gating_weights"].numpy().astype(np.float64)
        v = rec["pred_obj_vel"].numpy().astype(np.float64)
        p_raw = rec["pred_obj_trans"].numpy().astype(np.float64)
        p_pub = rec["pred_obj_trans_published"].numpy().astype(np.float64)
        bones = np.stack((rec["pred_lhand_obj_vector"].numpy(), rec["pred_rhand_obj_vector"].numpy()), 1).astype(np.float64)
        c_h = hysteresis(prob[:, :2], ON, OFF)
        Rw_gt = old["rotation_global"][:T, [20, 21]].double().numpy()      # GT wrist rotations
        Rf_gt = old["rotation_global"][:T, [18, 19]].double().numpy()      # GT forearm (elbow joint) rotations

        # ---- 0. sanity: replay reproduces raw prediction ----
        p_re = replay(rec, R, h, bones, p_raw[0], v, w)
        res["replay_max_dev_cm"].append(float(np.abs(p_re - p_raw).max() * 100))

        base_obj = float(norm(p_pub - p_gt).mean() * 100)
        base_hoi = hoi_err(p_pub, h_pub, p_gt, h_gt, c_gt)
        res["base_obj"].append(base_obj)
        res["base_hoi"].append(base_hoi)
        res["base_obj_raw"].append(float(norm(p_raw - p_gt).mean() * 100))
        # 1. per-axis error of the published object
        e = np.abs(p_pub - p_gt) * 100
        for i, ax in enumerate("xyz"):
            res[f"abs_err_{ax}"].append(float(e[:, i].mean()))
        res["abs_err_y_static"].append(float(e[~moving_gt, 1].mean()) if (~moving_gt).any() else np.nan)
        res["abs_err_y_moving"].append(float(e[moving_gt, 1].mean()) if moving_gt.any() else np.nan)

        # ---- 2. tip-to-surface distance: model bones vs GT bones on GT contact frames ----
        for k in range(2):
            m = c_gt[:, k]
            if m.sum() < 5:
                continue
            tip_m = (-bones[m, k]) / s[m, None]
            tip_g = np.einsum("tji,tj->ti", R_gt[m], h_gt[m, k] - p_gt[m]) / s[m, None]
            _, dm = q.query(torch.from_numpy(tip_m))
            _, dg = q.query(torch.from_numpy(tip_g))
            tipdist_model.extend((dm.cpu().numpy() * s[m]).tolist())
            tipdist_gt.extend((dg.cpu().numpy() * s[m]).tolist())

        # ---- 2b. anatomical offset in the wrist frame: how tight can the surface constraint get ----
        for k in range(2):
            m = c_gt[:, k]
            if m.sum() < 5:
                continue
            tip_g = np.einsum("tji,tj->ti", R_gt[m], h_gt[m, k] - p_gt[m]) / s[m, None]
            cl, _ = q.query(torch.from_numpy(tip_g))
            q_world = p_gt[m] + np.einsum("tij,tj->ti", R_gt[m], cl.cpu().numpy().astype(np.float64) * s[m, None])
            v_wrist = np.einsum("tji,tj->ti", Rw_gt[m, k], q_world - h_gt[m, k])   # offset palm->contact in wrist frame
            wrist_offsets[k].append(v_wrist)
            v_fore = np.einsum("tji,tj->ti", Rf_gt[m, k], q_world - h_gt[m, k])
            forearm_offsets[k].append(v_fore)

        # ---- 3. snap bone tips to the surface (all frames, matters only where w>0) ----
        snapped = bones.copy()
        soft = bones.copy()
        for k in range(2):
            tip = (-bones[:, k]) / s[:, None]
            cl, d = q.query(torch.from_numpy(tip))
            cl = cl.cpu().numpy().astype(np.float64)
            d = d.cpu().numpy().astype(np.float64) * s
            snapped[:, k] = -(cl * s[:, None])
            # soft: pull the tip to within margin of the surface
            margin = 0.03
            dirn = (tip * s[:, None]) - cl * s[:, None]
            dn = norm(dirn)[:, None] + 1e-9
            target = cl * s[:, None] + dirn / dn * np.minimum(d, margin)[:, None]
            soft[:, k] = -target
        p_snap = replay(rec, R, h, snapped, p_raw[0], v, w)
        p_soft = replay(rec, R, h, soft, p_raw[0], v, w)
        res["snap_obj_raw"].append(float(norm(p_snap - p_gt).mean() * 100))
        res["soft_obj_raw"].append(float(norm(p_soft - p_gt).mean() * 100))
        res["snap_hoi_raw"].append(hoi_err(p_snap, h, p_gt, h_gt, c_gt))
        res["soft_hoi_raw"].append(hoi_err(p_soft, h, p_gt, h_gt, c_gt))
        res["model_hoi_raw"].append(hoi_err(p_raw, h, p_gt, h_gt, c_gt))

        # ---- 4. floor support ----
        min_y_gt = q.min_y_world(torch.from_numpy(R_gt), torch.from_numpy(s), torch.from_numpy(p_gt)).cpu().numpy()
        static_gt = ~moving_gt
        floor = float(np.quantile(min_y_gt[static_gt], 0.05)) if static_gt.sum() > 5 else float(np.quantile(min_y_gt, 0.05))
        resting_gt = np.abs(min_y_gt - floor) < 0.02
        res["frac_rest_gt_static"].append(float(resting_gt[static_gt].mean()) if static_gt.any() else np.nan)
        res["frac_rest_gt_moving"].append(float(resting_gt[moving_gt].mean()) if moving_gt.any() else np.nan)
        min_y_pred = q.min_y_world(torch.from_numpy(R), torch.from_numpy(s), torch.from_numpy(p_pub)).cpu().numpy()
        rule_static = imu_static(R, acc)
        pmove_static = prob[:, 2] < 0.5

        def apply_floor(mask, propagate=False):
            out = p_pub.copy()
            out[mask, 1] += floor - min_y_pred[mask]
            return out

        for label, mask in (("floor_rule", rule_static), ("floor_pmove", pmove_static), ("floor_gt_rest", resting_gt)):
            pf = apply_floor(mask, propagate=False)
            res[f"{label}_obj"].append(float(norm(pf - p_gt).mean() * 100))
            res[f"{label}_hoi"].append(hoi_err(pf, h_pub, p_gt, h_gt, c_gt))
        # one-sided non-penetration, always on
        pen = np.maximum(0.0, floor - min_y_pred)
        pf = p_pub.copy()
        pf[:, 1] += pen
        res["floor_nonpen_obj"].append(float(norm(pf - p_gt).mean() * 100))
        res["floor_nonpen_hoi"].append(hoi_err(pf, h_pub, p_gt, h_gt, c_gt))
        res["frac_pred_penetrating_2cm"].append(float((pen > 0.02).mean()))

        # ---- 5b. full replay: hand-side correction at each predicted onset ----
        h_corr = h.copy()
        b_corr = bones.copy()
        for k in range(2):
            for a, b_end in segments_of(c_h[:, k]):
                if a == 0:
                    continue
                p_prev = p_pub[a - 1]
                tip = (R[a].T @ (h[a, k] - p_prev)) / s[a]
                cl, d = q.query(torch.from_numpy(tip[None]))
                cl = cl.cpu().numpy()[0].astype(np.float64) * s[a]
                d = float(d.cpu().numpy()[0] * s[a])
                q_world = p_prev + R[a] @ cl
                dirn = h[a, k] - q_world
                dn = norm(dirn) + 1e-9
                delta = (q_world + dirn / dn * min(d, 0.03)) - h[a, k]
                h_corr[a : b_end + 1, k] += delta
                b_corr[a : b_end + 1, k] -= R[a].T @ delta
        p_lm = replay(rec, R, h_corr, b_corr, p_raw[0], v, w)
        res["landmark_obj_raw"].append(float(norm(p_lm - p_gt).mean() * 100))
        res["landmark_hoi_raw"].append(hoi_err(p_lm, h_corr, p_gt, h_gt, c_gt))
        # hand error itself on GT contact frames (OT hands) before / after
        for lab, hh in (("palm_err_contact_pred", h), ("palm_err_contact_landmark", h_corr)):
            errs = [norm(hh[c_gt[:, k], k] - h_gt[c_gt[:, k], k]) for k in range(2) if c_gt[:, k].any()]
            res[lab].append(float(np.concatenate(errs).mean() * 100) if errs else np.nan)

        # ---- 5. mesh as hand landmark at predicted onsets ----
        first = True
        for k in range(2):
            for a, b in segments_of(c_h[:, k]):
                if a == 0:
                    continue
                gt_on = c_gt[max(a - 5, 0) : min(a + 6, T), k].any()
                p_prev = p_pub[a - 1]
                tip = (R[a].T @ (h[a, k] - p_prev)) / s[a]
                cl, d = q.query(torch.from_numpy(tip[None]))
                cl = cl.cpu().numpy()[0].astype(np.float64) * s[a]
                d = float(d.cpu().numpy()[0] * s[a])
                q_world = p_prev + R[a] @ cl
                dirn = h[a, k] - q_world
                dn = norm(dirn) + 1e-9
                h_snap = q_world + dirn / dn * min(d, 0.03)
                e_pred = norm(h[a, k] - h_gt[a, k]) * 100
                e_snap = norm(h_snap - h_gt[a, k]) * 100
                # how good was the pre-onset object position
                e_obj_prev = norm(p_prev - p_gt[a - 1]) * 100
                onset_rows.append((name, first, bool(gt_on), e_pred, e_snap, d * 100, e_obj_prev))
                first = False

        for key in ("base_obj", "base_hoi", "snap_obj_raw", "soft_obj_raw", "base_obj_raw", "floor_rule_obj", "floor_pmove_obj", "floor_gt_rest_obj", "floor_nonpen_obj", "landmark_obj_raw", "landmark_hoi_raw", "model_hoi_raw"):
            per_obj[name][key].append(res[key][-1])
        if ri % 100 == 0:
            print(ri, "/", len(recs), flush=True)

    summary = {k: seq_mean(v) for k, v in res.items()}
    tm, tg = np.asarray(tipdist_model) * 100, np.asarray(tipdist_gt) * 100
    summary["tip_surface_dist_model_cm_q50_q90_mean"] = [float(np.median(tm)), float(np.quantile(tm, 0.9)), float(tm.mean())]
    summary["tip_surface_dist_gt_cm_q50_q90_mean"] = [float(np.median(tg)), float(np.quantile(tg, 0.9)), float(tg.mean())]
    rows = np.array([(r[3], r[4], r[5], r[6]) for r in onset_rows], dtype=np.float64)
    flags = np.array([(r[1], r[2]) for r in onset_rows], dtype=bool)
    onset = {}
    for label, m in (("all", np.ones(len(rows), bool)), ("first_onset", flags[:, 0]), ("later_onset", ~flags[:, 0]), ("gt_matched", flags[:, 1]), ("first_gt_matched", flags[:, 0] & flags[:, 1])):
        if m.sum() == 0:
            continue
        onset[label] = {
            "n": int(m.sum()),
            "palm_err_pred_cm": float(rows[m, 0].mean()),
            "palm_err_snapped_cm": float(rows[m, 1].mean()),
            "frac_improved": float((rows[m, 1] < rows[m, 0]).mean()),
            "palm_surface_dist_pred_cm_median": float(np.median(rows[m, 2])),
            "obj_err_before_onset_cm": float(rows[m, 3].mean()),
        }
    summary["onset_landmark"] = onset
    wo = {}
    for k in range(2):
        V = np.concatenate(wrist_offsets[k], axis=0)
        vbar = V.mean(axis=0)
        resid = norm(V - vbar)
        wo[f"hand{k}"] = {"mean_offset_wrist_frame_cm": (vbar * 100).tolist(), "offset_norm_cm_median": float(np.median(norm(V)) * 100),
                          "resid_after_mean_offset_cm_median": float(np.median(resid) * 100), "resid_q90": float(np.quantile(resid, 0.9) * 100)}
    summary["wrist_frame_contact_offset"] = wo
    fo = {}
    for k in range(2):
        V = np.concatenate(forearm_offsets[k], axis=0)
        vbar = V.mean(axis=0)
        resid = norm(V - vbar)
        fo[f"hand{k}"] = {"mean_offset_forearm_frame_cm": (vbar * 100).tolist(),
                          "resid_after_mean_offset_cm_median": float(np.median(resid) * 100), "resid_q90": float(np.quantile(resid, 0.9) * 100)}
    summary["forearm_frame_contact_offset"] = fo
    summary["per_object"] = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in per_obj.items()}
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_object"}, indent=2))
    for n, d in sorted(summary["per_object"].items()):
        print(f"{n:13s} base={d['base_obj']:.2f} raw={d['base_obj_raw']:.2f} snap={d['snap_obj_raw']:.2f} soft={d['soft_obj_raw']:.2f} lm={d['landmark_obj_raw']:.2f} hoi raw={d['model_hoi_raw']:.2f} lm={d['landmark_hoi_raw']:.2f} | floor_rule={d['floor_rule_obj']:.2f} floor_gt={d['floor_gt_rest_obj']:.2f} nonpen={d['floor_nonpen_obj']:.2f}")


if __name__ == "__main__":
    main()
