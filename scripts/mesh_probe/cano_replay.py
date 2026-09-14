#!/usr/bin/env python3
"""Replays on the canonical OMOMO online dumps (ot_current / ot_v3 on VC-v4).

All variants re-run the evaluator's publisher recurrence

    p(t) = w0 (h_ref0 + R b0) + w1 (h_ref1 + R b1) + w2 (p(t-1) + d_imu(t)) + w3 p(t-1)

with refined hands h_ref, so obj = published object error and hoi = the official
HOI (published object vs refined palm on GT contact frames).  Sanity: the
'model' variant must reproduce the official eval JSON.

Q1 (mesh at test time): bone surface snap (hard/soft), hand snap at onset and
continuous, floor non-penetration (joint hand+object shift, train-calibrated).
Q3 (why v3 == current): cross replays bones x gating weights, gating statistics,
bone error at onset / inside segments, per-object and per-sequence deltas.
"""
from __future__ import annotations

import argparse
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

RUN = os.path.join(ROOT, "outputs/IMUHOI++/RNN_cano/omomo_cano_20260907")
TEST = os.path.join(ROOT, "process/IMUHOI++cano/processed_split_data_OMOMO_full/test")
CALIB = os.path.join(ROOT, "outputs/analysis/mesh_probe_20260907/floor_calibration_train.json")
OUT = os.path.join(RUN, "dumps/cano_replay.json")
ON, OFF = 0.15, 0.075


def norm(x):
    return np.linalg.norm(x, axis=-1)


def f64(t):
    return t.numpy().astype(np.float64)


class Rec:
    def __init__(self, rec, aux):
        self.name = rec["seq_file"]
        self.obj = rec["obj_name"]
        self.T = int(rec["publish_len"])
        self.p_gt = f64(rec["gt_obj_trans"])
        self.R_gt = rot6d(rec["gt_obj_rot"]).astype(np.float64)
        self.R = rot6d(rec["gt_obj_imu"][:, 3:9]).astype(np.float64)
        self.acc = f64(rec["gt_obj_imu"][:, :3])
        self.h_gt = f64(rec["gt_palm_pos"])
        self.h_s = f64(rec.get("ot_hand_positions", rec["stage1_pred_palm_glb_pos"]))
        self.h_r = f64(rec["pred_palm_glb_pos"])
        self.c_gt = np.stack((rec["gt_lhand_contact"].numpy(), rec["gt_rhand_contact"].numpy()), 1).astype(bool)
        self.moving_gt = rec["gt_obj_contact"].numpy().astype(bool)
        self.prob = f64(rec["pred_hand_contact_prob"])
        self.w = f64(rec["gating_weights"])
        self.p_raw = f64(rec["pred_obj_trans"])
        self.p_pub = f64(rec["pred_obj_trans_published"])
        fk = np.stack((f64(rec["pred_lhand_obj_trans"]), f64(rec["pred_rhand_obj_trans"])), 1)
        self.Rb = fk - self.h_s                                   # world-frame bones as used by the model
        self.d_imu = f64(rec["pred_imu_obj_trans"]) - f64(rec["pred_static_obj_trans"])
        self.p0 = f64(rec["pred_static_obj_trans"])[0]
        self.c_h = hysteresis(self.prob[:, :2], ON, OFF)
        # dataset obj_scale covers the full sequence; the dump publishes T frames
        self.s = np.asarray(aux["obj_scale"], dtype=np.float64)[: self.T]
        if self.s.shape[0] < self.T:
            self.s = np.concatenate((self.s, np.full(self.T - self.s.shape[0], self.s[-1] if self.s.size else 1.0)))
        self.ok = self.s > 1e-6

    def bones_obj(self):
        """Object-frame bones [T,2,3] (R^T (fk - h_s))."""
        return np.einsum("tji,thj->thi", self.R, self.Rb)


def publish(r: Rec, Rb=None, hands=None, w=None, d_imu=None, p0=None):
    Rb = r.Rb if Rb is None else Rb
    h = r.h_r if hands is None else hands
    w = r.w if w is None else w
    d = r.d_imu if d_imu is None else d_imu
    prev = (r.p0 if p0 is None else p0).copy()
    out = np.zeros((r.T, 3))
    for t in range(r.T):
        p = w[t, 0] * (h[t, 0] + Rb[t, 0]) + w[t, 1] * (h[t, 1] + Rb[t, 1]) + w[t, 2] * (prev + d[t]) + w[t, 3] * prev
        out[t] = p
        prev = p
    return out


def obj_err(p, r: Rec):
    return float(norm(p - r.p_gt).mean() * 100)


def hoi_err(p, h, r: Rec):
    errs = []
    for k in range(2):
        m = r.c_gt[:, k]
        if m.any():
            errs.append(norm((p - h[:, k]) - (r.p_gt - r.h_gt[:, k]))[m])
    return float(np.concatenate(errs).mean() * 100) if errs else float("nan")


def jitter(p):
    if p.shape[0] < 3:
        return float("nan")
    a = p[2:] - 2 * p[1:-1] + p[:-2]
    return float(norm(a).mean() * 1000)


def load_dump(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    return cache["records"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default=os.path.join(RUN, "dumps/ot_current_online.pt"))
    ap.add_argument("--v3", default=os.path.join(RUN, "dumps/ot_v3_online.pt"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    calib = json.load(open(CALIB))
    floor_global = calib["global_median_cm"] / 100.0
    floor_obj = {k: v / 100.0 for k, v in calib["per_object_median_cm"].items()}

    recs = {"current": load_dump(args.current), "v3": load_dump(args.v3)}
    by_name = {lab: {r["seq_file"]: r for r in rs} for lab, rs in recs.items()}
    names = sorted(set(by_name["current"]) & set(by_name["v3"]))
    if args.limit:
        names = names[: args.limit]
    print("matched sequences", len(names), flush=True)

    nn = {}
    res = {lab: defaultdict(list) for lab in ("current", "v3")}
    per_obj = {lab: defaultdict(lambda: defaultdict(list)) for lab in ("current", "v3")}
    cross = defaultdict(list)
    gate_stats = {lab: defaultdict(list) for lab in ("current", "v3")}
    bone_stats = {lab: defaultdict(list) for lab in ("current", "v3")}
    onset_rows = {lab: [] for lab in ("current", "v3")}
    seq_rows = []

    for i, name in enumerate(names):
        aux_d = torch.load(os.path.join(TEST, name), map_location="cpu", weights_only=False)
        aux = {"obj_scale": aux_d["obj_scale"].double().numpy()}
        R_gt_rot = aux_d["rotation_global"].double().numpy()
        rr = {lab: Rec(by_name[lab][name], aux) for lab in ("current", "v3")}
        obj = rr["v3"].obj
        if obj not in nn:
            nn[obj] = SurfaceNN(load_mesh(obj))
        q = nn[obj]
        row = {"seq": name, "obj": obj}
        for lab, r in rr.items():
            R_ = res[lab]
            # ---- sanity + baseline ----
            p_model = publish(r)
            R_["repro_max_dev_cm"].append(float(np.abs(p_model - r.p_pub).max() * 100))
            base_obj, base_hoi = obj_err(r.p_pub, r), hoi_err(r.p_pub, r.h_r, r)
            R_["base_obj"].append(base_obj)
            R_["base_hoi"].append(base_hoi)
            R_["base_raw_obj"].append(obj_err(r.p_raw, r))
            R_["base_jitter"].append(jitter(r.p_pub))
            row[f"{lab}_obj"] = base_obj
            row[f"{lab}_hoi"] = base_hoi
            e = np.abs(r.p_pub - r.p_gt) * 100
            for ax_i, ax in enumerate("xyz"):
                R_[f"abs_err_{ax}"].append(float(e[:, ax_i].mean()))

            # ---- gating statistics ----
            static_gt = ~r.moving_gt
            any_c = r.c_gt.any(1)
            gs = gate_stats[lab]
            gs["w_static_on_gt_static"].append(float(r.w[static_gt, 3].mean()) if static_gt.any() else np.nan)
            gs["w_imu_on_gt_static"].append(float(r.w[static_gt, 2].mean()) if static_gt.any() else np.nan)
            gs["w_fk_on_gt_contact"].append(float(r.w[any_c, :2].sum(1).mean()) if any_c.any() else np.nan)
            gs["w_static_on_gt_contact_moving"].append(float(r.w[any_c & r.moving_gt, 3].mean()) if (any_c & r.moving_gt).any() else np.nan)
            gs["w_imu_on_gt_moving_nocontact"].append(float(r.w[r.moving_gt & ~any_c, 2].mean()) if (r.moving_gt & ~any_c).any() else np.nan)
            step = norm(np.diff(r.p_pub, axis=0)) * 100
            gs["pred_step_on_gt_static_cm"].append(float(step[static_gt[1:]].mean()) if static_gt[1:].any() else np.nan)
            gs["err_on_gt_static"].append(float(norm(r.p_pub - r.p_gt)[static_gt].mean() * 100) if static_gt.any() else np.nan)
            gs["err_on_gt_moving"].append(float(norm(r.p_pub - r.p_gt)[r.moving_gt].mean() * 100) if r.moving_gt.any() else np.nan)

            # ---- bone error on GT contact frames (published hands) ----
            bs = bone_stats[lab]
            for k in range(2):
                for a, b in segments_of(r.c_gt[:, k]):
                    idx = np.arange(a, b + 1)
                    bp = np.einsum("tji,tj->ti", r.R[idx], r.p_pub[idx] - r.h_r[idx, k])
                    bg = np.einsum("tji,tj->ti", r.R_gt[idx], r.p_gt[idx] - r.h_gt[idx, k])
                    e_b = norm(bp - bg) * 100
                    bs["bone_err"].extend(e_b.tolist())
                    bs["bone_err_onset"].append(float(e_b[0]))
                    since = np.arange(len(idx))
                    for lo, hi, key in ((1, 15, "1-15"), (16, 30, "16-30"), (31, 60, "31-60"), (61, 10**9, "61+")):
                        m = (since >= lo) & (since <= hi)
                        if m.any():
                            bs[f"bone_err_{key}"].extend(e_b[m].tolist())
                            bs[f"bone_growth_{key}"].extend((e_b[m] - e_b[0]).tolist())
                    # palm error inside the segment
                    bs["palm_err_contact"].extend((norm(r.h_r[idx, k] - r.h_gt[idx, k]) * 100).tolist())

            # ---- compensation: does the bone error cancel the hand error? ----
            for k in range(2):
                m = r.c_gt[:, k]
                if m.sum() < 3:
                    continue
                e_b = (r.p_pub[m] - r.h_r[m, k]) - (r.p_gt[m] - r.h_gt[m, k])      # world bone error (= HOI vector)
                e_h = r.h_r[m, k] - r.h_gt[m, k]                                    # hand error
                hh = (e_h * e_h).sum(1) + 1e-9
                alpha = (e_b * e_h).sum(1) / hh                                     # -1 = full compensation
                e_par = alpha[:, None] * e_h
                e_orth = e_b - e_par
                bs["comp_alpha"].extend(alpha.tolist())
                bs["comp_par_cm"].extend((norm(e_par) * 100).tolist())
                bs["comp_orth_cm"].extend((norm(e_orth) * 100).tolist())
                bs["hoi_vec_cm"].extend((norm(e_b) * 100).tolist())
                bs["hand_err_cm"].extend((norm(e_h) * 100).tolist())
                bs["obj_err_contact_cm"].extend((norm(r.p_pub[m] - r.p_gt[m]) * 100).tolist())
                # obj error if the bone did NOT compensate (bone error orthogonal part only)
                bs["obj_err_if_uncomp_cm"].extend((norm(e_h + e_orth) * 100).tolist())

            # ---- oracles: GT bone frozen at each GT onset (uncompensated), predicted hands ----
            for key, per_frame in (("gtbone_L0", False), ("gtbone_perframe", True)):
                Rb_o = r.Rb.copy()
                for k in range(2):
                    for a, b in segments_of(r.c_gt[:, k]):
                        idx = np.arange(a, b + 1)
                        bg = np.einsum("tji,tj->ti", r.R_gt[idx], r.p_gt[idx] - r.h_gt[idx, k])
                        if not per_frame:
                            bg = np.repeat(bg[:1], len(idx), axis=0)
                        Rb_o[idx, k] = np.einsum("tij,tj->ti", r.R[idx], bg)
                p = publish(r, Rb=Rb_o)
                R_[f"{key}_obj"].append(obj_err(p, r))
                R_[f"{key}_hoi"].append(hoi_err(p, r.h_r, r))

            # ---- Q1: bone surface snap (hard / soft) ----
            bones = r.bones_obj()
            snapped, soft = bones.copy(), bones.copy()
            for k in range(2):
                tip = (-bones[:, k]) / np.where(r.ok, r.s, 1.0)[:, None]
                cl, d = q.query(torch.from_numpy(tip))
                cl = cl.cpu().numpy().astype(np.float64) * r.s[:, None]
                d = d.cpu().numpy().astype(np.float64) * r.s
                snapped[:, k] = -cl
                dirn = (-bones[:, k]) - cl
                dn = norm(dirn)[:, None] + 1e-9
                soft[:, k] = -(cl + dirn / dn * np.minimum(d, 0.03)[:, None])
            for key, bb in (("bone_snap_hard", snapped), ("bone_snap_soft", soft)):
                Rb = np.einsum("tij,thj->thi", r.R, bb)
                p = publish(r, Rb=Rb)
                R_[f"{key}_obj"].append(obj_err(p, r))
                R_[f"{key}_hoi"].append(hoi_err(p, r.h_r, r))

            # ---- Q1: hand snap at predicted onsets (landmark) and continuous ----
            h_on = r.h_r.copy()
            Rb_on = r.Rb.copy()
            for k in range(2):
                for a, b in segments_of(r.c_h[:, k]):
                    if a == 0 or not r.ok[a]:
                        continue
                    p_prev = r.p_pub[a - 1]
                    tip = (r.R[a].T @ (r.h_r[a, k] - p_prev)) / r.s[a]
                    cl, d = q.query(torch.from_numpy(tip[None]))
                    cl = cl.cpu().numpy()[0].astype(np.float64) * r.s[a]
                    d = float(d.cpu().numpy()[0] * r.s[a])
                    q_world = p_prev + r.R[a] @ cl
                    dirn = r.h_r[a, k] - q_world
                    delta = (q_world + dirn / (norm(dirn) + 1e-9) * min(d, 0.03)) - r.h_r[a, k]
                    h_on[a : b + 1, k] += delta
                    Rb_on[a : b + 1, k] -= delta          # keep the FK output unchanged at the onset
                    gt_on = r.c_gt[max(a - 5, 0) : min(a + 6, r.T), k].any()
                    onset_rows[lab].append((norm(r.h_r[a, k] - r.h_gt[a, k]) * 100, norm(h_on[a, k] - r.h_gt[a, k]) * 100, bool(gt_on), a))
            p = publish(r, Rb=Rb_on, hands=h_on)
            R_["hand_snap_onset_obj"].append(obj_err(p, r))
            R_["hand_snap_onset_hoi"].append(hoi_err(p, h_on, r))
            # continuous: every predicted-contact frame, pull the refined hand to 3 cm of the published object surface
            h_c = r.h_r.copy()
            for k in range(2):
                m = r.c_h[:, k] & r.ok
                if not m.any():
                    continue
                tip = np.einsum("tji,tj->ti", r.R[m], r.h_r[m, k] - r.p_pub[m]) / r.s[m, None]
                cl, d = q.query(torch.from_numpy(tip))
                cl = cl.cpu().numpy().astype(np.float64) * r.s[m, None]
                d = d.cpu().numpy().astype(np.float64) * r.s[m]
                q_world = r.p_pub[m] + np.einsum("tij,tj->ti", r.R[m], cl)
                dirn = r.h_r[m, k] - q_world
                h_c[m, k] = q_world + dirn / (norm(dirn)[:, None] + 1e-9) * np.minimum(d, 0.03)[:, None]
            R_["hand_snap_cont_obj"].append(obj_err(r.p_pub, r))       # object unchanged
            R_["hand_snap_cont_hoi"].append(hoi_err(r.p_pub, h_c, r))
            pe_b = [norm(r.h_r[r.c_gt[:, k], k] - r.h_gt[r.c_gt[:, k], k]) for k in range(2) if r.c_gt[:, k].any()]
            pe_a = [norm(h_c[r.c_gt[:, k], k] - r.h_gt[r.c_gt[:, k], k]) for k in range(2) if r.c_gt[:, k].any()]
            if pe_b:
                R_["hand_snap_cont_palm_before"].append(float(np.concatenate(pe_b).mean() * 100))
                R_["hand_snap_cont_palm_after"].append(float(np.concatenate(pe_a).mean() * 100))

            # ---- Q1: floor non-penetration, joint hand+object shift ----
            min_y_pred = q.min_y_world(torch.from_numpy(r.R), torch.from_numpy(r.s), torch.from_numpy(r.p_pub)).cpu().numpy()
            min_y_gt = q.min_y_world(torch.from_numpy(r.R_gt), torch.from_numpy(r.s), torch.from_numpy(r.p_gt)).cpu().numpy()
            floor_seq = float(np.quantile(min_y_gt[static_gt & r.ok], 0.05)) if (static_gt & r.ok).sum() > 5 else float(np.quantile(min_y_gt[r.ok], 0.05))
            for lab_f, floor in (("global", floor_global), ("perobj", floor_obj.get(obj, floor_global)), ("seqGT", floor_seq)):
                for margin in (0.0, 0.02):
                    pen = np.maximum(0.0, (floor - margin) - min_y_pred)
                    pen[~r.ok] = 0.0
                    po = r.p_pub.copy()
                    po[:, 1] += pen
                    ho = r.h_r.copy()
                    ho[:, :, 1] += pen[:, None]
                    key = f"floor_{lab_f}_m{int(margin*100)}"
                    R_[f"{key}_obj"].append(obj_err(po, r))
                    R_[f"{key}_hoi_joint"].append(hoi_err(po, ho, r))
                    R_[f"{key}_hoi_objonly"].append(hoi_err(po, r.h_r, r))
                    R_[f"{key}_fire"].append(float((pen > 0).mean()))
                    R_[f"{key}_jitter"].append(jitter(po))
                    if pe_b:
                        pe_f = [norm(ho[r.c_gt[:, k], k] - r.h_gt[r.c_gt[:, k], k]) for k in range(2) if r.c_gt[:, k].any()]
                        R_[f"{key}_palm_after"].append(float(np.concatenate(pe_f).mean() * 100))
            R_["frac_pred_penetrating_2cm_global"].append(float(((floor_global - 0.02) - min_y_pred > 0)[r.ok].mean()))
            R_["frac_gt_penetrating_2cm_global"].append(float(((floor_global - 0.02) - min_y_gt > 0)[r.ok].mean()))

            for key in ("base_obj", "base_hoi", "base_raw_obj", "bone_snap_soft_obj", "bone_snap_soft_hoi", "hand_snap_cont_hoi", "floor_global_m0_obj", "floor_global_m2_obj", "floor_perobj_m0_obj", "gtbone_L0_obj", "gtbone_L0_hoi"):
                per_obj[lab][obj][key].append(R_[key][-1])

        # ---- Q3: cross replays (bones x weights), same refined hands (v3's) ----
        rc, rv = rr["current"], rr["v3"]
        if rc.T == rv.T:
            combos = {
                "cur_bones_cur_w": (rc.Rb, rc.w, rc.d_imu),
                "v3_bones_v3_w": (rv.Rb, rv.w, rv.d_imu),
                "v3_bones_cur_w": (rv.Rb, rc.w, rc.d_imu),
                "cur_bones_v3_w": (rc.Rb, rv.w, rv.d_imu),
            }
            for key, (Rb, w, d) in combos.items():
                p = publish(rv, Rb=Rb, hands=rv.h_r, w=w, d_imu=d)
                cross[f"{key}_obj"].append(obj_err(p, rv))
                cross[f"{key}_hoi"].append(hoi_err(p, rv.h_r, rv))
            # hands: current's refined hands with v3 bones/weights
            p = publish(rv, hands=rc.h_r)
            cross["v3_all_cur_hands_obj"].append(obj_err(p, rv))
            cross["v3_all_cur_hands_hoi"].append(hoi_err(p, rc.h_r, rv))
            # palm error of each refine head on GT contact frames
            for lab, r in rr.items():
                pe = [norm(r.h_r[r.c_gt[:, k], k] - r.h_gt[r.c_gt[:, k], k]) for k in range(2) if r.c_gt[:, k].any()]
                ps = [norm(r.h_s[r.c_gt[:, k], k] - r.h_gt[r.c_gt[:, k], k]) for k in range(2) if r.c_gt[:, k].any()]
                if pe:
                    cross[f"{lab}_palm_refined_contact"].append(float(np.concatenate(pe).mean() * 100))
                    cross[f"{lab}_palm_stage1_contact"].append(float(np.concatenate(ps).mean() * 100))
            # contact agreement between the two joint-trained VCs
            cross["contact_prob_abs_diff"].append(float(np.abs(rc.prob[:, :2] - rv.prob[:, :2]).mean()))
        seq_rows.append(row)
        if i % 100 == 0:
            print(i, "/", len(names), flush=True)

    def sm(v):
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    summary = {"n_sequences": len(names), "floor_global_cm": floor_global * 100}
    for lab in ("current", "v3"):
        summary[lab] = {k: sm(v) for k, v in res[lab].items()}
        summary[lab]["gate"] = {k: sm(v) for k, v in gate_stats[lab].items()}
        bs = bone_stats[lab]
        summary[lab]["bone"] = {k: sm(v) for k, v in bs.items()}
        rows = np.array([(a, b, c, d) for a, b, c, d in onset_rows[lab]], dtype=np.float64)
        if rows.size:
            gtm = rows[:, 2] > 0.5
            summary[lab]["onset_landmark"] = {
                "n": int(len(rows)), "palm_err_pred": float(rows[:, 0].mean()), "palm_err_snapped": float(rows[:, 1].mean()),
                "gt_matched_frac": float(gtm.mean()),
                "gt_matched_palm_err_pred": float(rows[gtm, 0].mean()), "gt_matched_palm_err_snapped": float(rows[gtm, 1].mean()),
            }
        summary[lab]["per_object"] = {o: {k: sm(v) for k, v in d.items()} for o, d in per_obj[lab].items()}
    summary["cross"] = {k: sm(v) for k, v in cross.items()}
    # per-sequence deltas v3 - current
    d_obj = np.array([r["v3_obj"] - r["current_obj"] for r in seq_rows])
    d_hoi = np.array([r["v3_hoi"] - r["current_hoi"] for r in seq_rows if np.isfinite(r["v3_hoi"]) and np.isfinite(r["current_hoi"])])
    summary["per_sequence"] = {
        "obj_delta_mean": float(d_obj.mean()), "obj_v3_better_frac": float((d_obj < 0).mean()),
        "obj_delta_gt1cm_frac": float((np.abs(d_obj) > 1).mean()), "obj_v3_worse_gt1cm_frac": float((d_obj > 1).mean()), "obj_v3_better_gt1cm_frac": float((d_obj < -1).mean()),
        "hoi_delta_mean": float(d_hoi.mean()), "hoi_v3_better_frac": float((d_hoi < 0).mean()),
        "oracle_per_seq_best_obj": float(np.mean([min(r["v3_obj"], r["current_obj"]) for r in seq_rows])),
        "avg_of_two_obj": None,
    }
    summary["seq_rows"] = seq_rows
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=1)
    show = {k: v for k, v in summary.items() if k != "seq_rows"}
    for lab in ("current", "v3"):
        show[lab] = {k: v for k, v in show[lab].items() if k != "per_object"}
    print(json.dumps(show, indent=1))
    print("\nper-object (published obj | hoi): current vs v3")
    for o in sorted(summary["v3"]["per_object"]):
        c, v = summary["current"]["per_object"][o], summary["v3"]["per_object"][o]
        print(f"{o:13s} obj {c['base_obj']:.2f} vs {v['base_obj']:.2f} | hoi {c['base_hoi']:.2f} vs {v['base_hoi']:.2f} | soft-snap hoi {c['bone_snap_soft_hoi']:.2f} vs {v['bone_snap_soft_hoi']:.2f} | floor obj {c['floor_global_m0_obj']:.2f} vs {v['floor_global_m0_obj']:.2f}")


if __name__ == "__main__":
    main()
