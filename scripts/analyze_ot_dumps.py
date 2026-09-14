#!/usr/bin/env python3
"""Error decomposition of ObjectTrans predictions from ``dump_ot_predictions.py``.

Given one or more dumps (``label=path``), the script answers, per dump:

  A. Does the dump reproduce the published obj_err / hoi_err?
  B. How does the object error evolve with frames since a GT contact onset?
  C. Which phase (before first contact / in contact / gaps / after last
     release) carries the error, and how much error is *added* inside
     contact segments versus inside no-contact gaps?
  D. What happens on frames where predicted contact disagrees with GT?
  E. How wrong is the hand->object bone on GT contact frames, and does it
     inherit the error present at onset or correct it?
  F. A ladder of hybrid FK estimators built from the same dumped inputs
     (GT everything -> predicted hands -> predicted contact -> dead-reckoned
     anchor), to attribute the model's error to its inputs.
  G. Per-sequence comparison between two labelled dumps.
  H. v3-only: how stable the latched anchor is inside a segment.

All errors in cm.  "seq-mean" means per-sequence mean then mean over
sequences (the eval convention); "frame" means pooled over frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.rotation_conversions import rotation_6d_to_matrix  # noqa: E402

HAND_THRESHOLD = 0.15
STATIC_SPEED_CM = 0.2   # GT object speed below this (cm/frame) counts as static
FPS = 30.0
ONSET_BINS = [(0, 0), (1, 15), (16, 30), (31, 60), (61, 120), (121, 10**9)]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def segments_of(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) runs of True."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[cuts + 1]))
    ends = np.concatenate((idx[cuts], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def hysteresis(prob: np.ndarray, on: float, off: float) -> np.ndarray:
    """Per-hand on/off contact with the v3 hysteresis rule."""
    out = np.zeros_like(prob, dtype=bool)
    active = np.zeros(prob.shape[1], dtype=bool)
    for t in range(prob.shape[0]):
        active = np.where(active, prob[t] >= off, prob[t] >= on)
        out[t] = active
    return out


def imu_integrate(acc: np.ndarray, start: int, end: int) -> np.ndarray:
    """Displacement from ``start`` by double Euler integration of world acc, v(start)=0."""
    seg = acc[start : end + 1]
    vel = np.cumsum(seg, axis=0) / FPS
    disp = np.cumsum(vel, axis=0) / FPS
    return disp - disp[0]


def ema(x: np.ndarray, tau_frames: float) -> np.ndarray:
    """Causal exponential moving average along axis 0."""
    alpha = 1.0 / max(tau_frames, 1.0)
    out = np.zeros_like(x)
    acc = x[0].copy()
    for t in range(x.shape[0]):
        acc = acc + alpha * (x[t] - acc)
        out[t] = acc
    return out


def imu_static_detector(R: np.ndarray, acc: np.ndarray, k: int = 5, ang_deg: float = 0.6, acc_thr: float = 0.35) -> np.ndarray:
    """Causal object-static flag from the (noisy) object IMU alone."""
    T = R.shape[0]
    static = np.zeros(T, dtype=bool)
    for t in range(T):
        t0 = max(t - k, 0)
        rel = R[t].T @ R[t0]
        ang = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
        a = norm(acc[t0 : t + 1]).max()
        static[t] = (ang < ang_deg) and (a < acc_thr)
    return static


def rot6d(x: torch.Tensor) -> np.ndarray:
    return rotation_6d_to_matrix(x.reshape(-1, 6)).reshape(*x.shape[:-1], 3, 3).numpy()


def norm(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


class Seq:
    """One sequence of a dump, unpacked to numpy with derived masks."""

    def __init__(self, rec: Dict):
        self.name = rec["seq_name"]
        self.obj = rec["obj_name"]
        self.T = int(rec["publish_len"])
        self.p_gt = rec["gt_obj_trans"].numpy()
        self.p_pred = rec["pred_obj_trans"].numpy()
        self.h_gt = rec["gt_palm_pos"].numpy()
        hands = rec.get("ot_hand_positions", rec.get("stage1_pred_palm_glb_pos", rec.get("pred_palm_glb_pos")))
        self.h_pred = hands.numpy()
        self.R_gt = rot6d(rec["gt_obj_rot"])
        self.R_imu = rot6d(rec["gt_obj_imu"][:, 3:9])
        self.c_gt = np.stack((rec["gt_lhand_contact"].numpy(), rec["gt_rhand_contact"].numpy()), axis=1).astype(bool)
        self.prob = rec["pred_hand_contact_prob"].numpy()
        self.c_pred = self.prob[:, :2] >= HAND_THRESHOLD
        self.w = rec["gating_weights"].numpy()
        self.any_gt = self.c_gt.any(axis=1)
        self.any_pred = self.c_pred.any(axis=1)
        self.err = norm(self.p_pred - self.p_gt) * 100.0
        self.seg_gt = segments_of(self.any_gt)
        self.speed_gt = np.r_[0.0, norm(np.diff(self.p_gt, axis=0))] * 100.0      # cm/frame
        self.moving_gt = self.speed_gt >= STATIC_SPEED_CM
        self.acc_imu = rec["gt_obj_imu"][:, :3].numpy()                            # world-frame m/s^2 (noisy)
        self.c_hyst = hysteresis(self.prob[:, :2], HAND_THRESHOLD, HAND_THRESHOLD * 0.5)
        root_pred = rec.get("stage1_root_trans_pred", rec.get("root_trans_pred"))
        self.d_root = root_pred.numpy() - rec["gt_trans"].numpy()          # [T,3] root error (m)
        self.d_hand = self.h_pred - self.h_gt                                 # [T,2,3] palm error (m)
        self.rec = rec

    # frames-since-onset for GT (any hand) segments; -1 outside contact
    def since_onset(self) -> np.ndarray:
        out = np.full(self.T, -1, dtype=np.int64)
        for s, e in self.seg_gt:
            out[s : e + 1] = np.arange(e - s + 1)
        return out

    def phase(self) -> np.ndarray:
        """0 pre-first-contact, 1 in contact, 2 gap, 3 post-last-release, 4 never contact."""
        out = np.full(self.T, 4, dtype=np.int64)
        if not self.seg_gt:
            return out
        first = self.seg_gt[0][0]
        last = self.seg_gt[-1][1]
        out[:first] = 0
        out[last + 1 :] = 3
        out[first : last + 1] = 2
        for s, e in self.seg_gt:
            out[s : e + 1] = 1
        return out


def fk_estimator(
    hand: np.ndarray,
    R: np.ndarray,
    contact: np.ndarray,
    p_gt: np.ndarray,
    anchor: str,
    p_init: Optional[np.ndarray] = None,
    moving: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Frozen-bone FK with per-hand segments, averaged when both, static hold otherwise.

    anchor='gt'   bone from the GT object position at each onset (oracle reset)
    anchor='self' bone from the estimator's own previous output (dead reckoning)
    """
    T = hand.shape[0]
    out = np.zeros((T, 3), dtype=np.float64)
    prev = (p_gt[0] if p_init is None else p_init).astype(np.float64)
    bone = np.zeros((2, 3))
    active = np.zeros(2, dtype=bool)
    if moving is not None:
        contact = contact & moving[:, None]
    for t in range(T):
        est = []
        for h in range(2):
            if contact[t, h]:
                if not active[h]:
                    ref = p_gt[t] if anchor == "gt" else prev
                    bone[h] = R[t].T @ (ref - hand[t, h])
                    active[h] = True
                est.append(hand[t, h] + R[t] @ bone[h])
            else:
                active[h] = False
        out[t] = np.mean(est, axis=0) if est else prev
        prev = out[t]
    return out


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------
def analyze(label: str, seqs: List[Seq]) -> Dict:
    R: Dict = {"label": label, "n_seq": len(seqs)}

    # A. metrics ---------------------------------------------------------
    obj_err, hoi_err, jitter = [], [], []
    for s in seqs:
        obj_err.append(s.err.mean())
        rel = []
        for h in range(2):
            m = s.c_gt[:, h]
            if m.any():
                rel.append(norm((s.p_pred - s.h_pred[:, h]) - (s.p_gt - s.h_gt[:, h]))[m] * 100.0)
        hoi_err.append(np.concatenate(rel).mean() if rel else np.nan)
        if s.T >= 3:
            acc = s.p_pred[2:] - 2 * s.p_pred[1:-1] + s.p_pred[:-2]
            jitter.append(norm(acc).mean() * 1000.0)
    R["metrics"] = {
        "obj_err_seqmean": nanmean(obj_err),
        "obj_err_median": float(np.nanmedian(obj_err)),
        "hoi_err_seqmean": nanmean(hoi_err),
        "obj_jitter": nanmean(jitter),
        "obj_err_frame": float(np.concatenate([s.err for s in seqs]).mean()),
    }

    # B. error vs frames since onset --------------------------------------
    since = np.concatenate([s.since_onset() for s in seqs])
    err = np.concatenate([s.err for s in seqs])
    # growth relative to the error at that segment's onset
    growth = np.concatenate([
        np.concatenate([s.err[st : en + 1] - s.err[st] for st, en in s.seg_gt]) if s.seg_gt else np.zeros(0)
        for s in seqs
    ])
    since_in = since[since >= 0]
    R["since_onset"] = {}
    for lo, hi in ONSET_BINS:
        m = (since >= lo) & (since <= hi)
        m_in = (since_in >= lo) & (since_in <= hi)
        R["since_onset"][f"{lo}-{hi if hi < 10**8 else 'inf'}"] = {
            "frames": int(m.sum()),
            "err": float(err[m].mean()) if m.any() else np.nan,
            "growth": float(growth[m_in].mean()) if m_in.any() else np.nan,
        }
    R["in_contact_frames_beyond_60"] = float((since_in > 60).mean())

    # C. phases and error added per phase ---------------------------------
    phase = np.concatenate([s.phase() for s in seqs])
    names = ["pre_first_contact", "in_contact", "gap", "post_last_release", "never"]
    R["phase"] = {}
    for k, nm in enumerate(names):
        m = phase == k
        R["phase"][nm] = {
            "frac_frames": float(m.mean()),
            "err": float(err[m].mean()) if m.any() else np.nan,
            "contrib": float(err[m].sum() / err.size),
        }
    d_seg, d_gap, e_first, seg_len, gap_len = [], [], [], [], []
    for s in seqs:
        if not s.seg_gt:
            continue
        e_first.append(s.err[s.seg_gt[0][0]])
        for i, (st, en) in enumerate(s.seg_gt):
            d_seg.append(s.err[en] - s.err[st])
            seg_len.append(en - st + 1)
            if i + 1 < len(s.seg_gt):
                nst = s.seg_gt[i + 1][0]
                d_gap.append(s.err[nst] - s.err[en])
                gap_len.append(nst - en - 1)
    R["error_delta"] = {
        "err_at_first_onset": nanmean(e_first),
        "delta_over_segment": nanmean(d_seg),
        "delta_over_segment_per100f": float(np.sum(d_seg) / np.sum(seg_len) * 100.0),
        "delta_over_gap": nanmean(d_gap),
        "delta_over_gap_per100f": float(np.sum(d_gap) / max(np.sum(gap_len), 1) * 100.0),
        "n_segments": len(d_seg),
        "n_gaps": len(d_gap),
        "mean_seg_len": float(np.mean(seg_len)),
        "mean_gap_len": float(np.mean(gap_len)) if gap_len else np.nan,
    }

    # D. contact disagreement --------------------------------------------
    any_gt = np.concatenate([s.any_gt for s in seqs])
    any_pred = np.concatenate([s.any_pred for s in seqs])
    dp_pred = np.concatenate([np.r_[0.0, norm(np.diff(s.p_pred, axis=0))] for s in seqs]) * 100.0
    dp_gt = np.concatenate([np.r_[0.0, norm(np.diff(s.p_gt, axis=0))] for s in seqs]) * 100.0
    derr = np.concatenate([np.r_[0.0, np.diff(s.err)] for s in seqs])
    w_static = np.concatenate([s.w[:, 3] for s in seqs])
    cats = {
        "TP": any_gt & any_pred,
        "FN": any_gt & ~any_pred,
        "FP": ~any_gt & any_pred,
        "TN": ~any_gt & ~any_pred,
    }
    R["contact_cats"] = {}
    for nm, m in cats.items():
        R["contact_cats"][nm] = {
            "frac_frames": float(m.mean()),
            "err": float(err[m].mean()) if m.any() else np.nan,
            "derr_per_frame": float(derr[m].mean()) if m.any() else np.nan,
            "pred_motion_cm_per_frame": float(dp_pred[m].mean()) if m.any() else np.nan,
            "gt_motion_cm_per_frame": float(dp_gt[m].mean()) if m.any() else np.nan,
            "w_static": float(w_static[m].mean()) if m.any() else np.nan,
        }
    R["gt_object_motion_when_untouched_cm_per_frame"] = float(dp_gt[~any_gt].mean())
    R["gt_object_motion_when_touched_cm_per_frame"] = float(dp_gt[any_gt].mean())

    # E. bone error on GT contact frames -----------------------------------
    bone_err, bone_len_err, bone_ang, bone_since, bone_onset_err = [], [], [], [], []
    for s in seqs:
        for h in range(2):
            for st, en in segments_of(s.c_gt[:, h]):
                idx = np.arange(st, en + 1)
                bp = np.einsum("tji,tj->ti", s.R_imu[idx], s.p_pred[idx] - s.h_pred[idx, h])
                bg = np.einsum("tji,tj->ti", s.R_gt[idx], s.p_gt[idx] - s.h_gt[idx, h])
                e = norm(bp - bg) * 100.0
                bone_err.append(e)
                bone_len_err.append(np.abs(norm(bp) - norm(bg)) * 100.0)
                cos = np.sum(bp * bg, -1) / (norm(bp) * norm(bg) + 1e-9)
                bone_ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                bone_since.append(np.arange(en - st + 1))
                bone_onset_err.append(np.full(en - st + 1, e[0]))
    bone_err = np.concatenate(bone_err)
    bone_len_err = np.concatenate(bone_len_err)
    bone_ang = np.concatenate(bone_ang)
    bone_since = np.concatenate(bone_since)
    bone_onset_err = np.concatenate(bone_onset_err)
    R["bone"] = {
        "err_frame": float(bone_err.mean()),
        "len_err_frame": float(bone_len_err.mean()),
        "angle_deg_frame": float(bone_ang.mean()),
        "err_at_onset": float(bone_err[bone_since == 0].mean()),
        "corr_err_t_vs_err_onset": float(np.corrcoef(bone_err, bone_onset_err)[0, 1]),
        "by_since": {},
    }
    for lo, hi in ONSET_BINS:
        m = (bone_since >= lo) & (bone_since <= hi)
        if m.any():
            R["bone"]["by_since"][f"{lo}-{hi if hi < 10**8 else 'inf'}"] = {
                "err": float(bone_err[m].mean()),
                "err_minus_onset": float((bone_err[m] - bone_onset_err[m]).mean()),
                "len_err": float(bone_len_err[m].mean()),
                "angle": float(bone_ang[m].mean()),
            }

    # E2. hand / root error and their drift inside contact segments ----------
    abs_hand, abs_root, drift_hand, drift_root, drift_local, drift_since = [], [], [], [], [], []
    for s in seqs:
        abs_hand.append(norm(s.d_hand).reshape(-1) * 100.0)
        abs_root.append(norm(s.d_root) * 100.0)
        for h in range(2):
            for st, en in segments_of(s.c_gt[:, h]):
                idx = np.arange(st, en + 1)
                dh = s.d_hand[idx, h] - s.d_hand[st, h]
                dr = s.d_root[idx] - s.d_root[st]
                drift_hand.append(norm(dh) * 100.0)
                drift_root.append(norm(dr) * 100.0)
                drift_local.append(norm(dh - dr) * 100.0)
                drift_since.append(np.arange(en - st + 1))
    drift_hand = np.concatenate(drift_hand); drift_root = np.concatenate(drift_root)
    drift_local = np.concatenate(drift_local); drift_since = np.concatenate(drift_since)
    R["hand"] = {
        "abs_palm_err": float(np.concatenate(abs_hand).mean()),
        "abs_root_err": float(np.concatenate(abs_root).mean()),
        "in_seg_drift_palm": float(drift_hand.mean()),
        "in_seg_drift_root_part": float(drift_root.mean()),
        "in_seg_drift_local_part": float(drift_local.mean()),
        "by_since": {},
    }
    for lo, hi in ONSET_BINS:
        m = (drift_since >= lo) & (drift_since <= hi)
        if m.any():
            R["hand"]["by_since"][f"{lo}-{hi if hi < 10**8 else 'inf'}"] = {
                "palm": float(drift_hand[m].mean()), "root": float(drift_root[m].mean()), "local": float(drift_local[m].mean())}

    # F. hybrid FK ladder ----------------------------------------------------
    ladder = defaultdict(list)
    for s in seqs:
        h_root_fixed = s.h_pred - s.d_root[:, None, :]      # remove root drift, keep local pose error
        h_root_only = s.h_gt + s.d_root[:, None, :]         # only root drift
        cases = {
            "O0_gt_all": (s.h_gt, s.R_gt, s.c_gt, "gt"),
            "O1_pred_hands": (s.h_pred, s.R_gt, s.c_gt, "gt"),
            "O1r_pred_hands_root_fixed": (h_root_fixed, s.R_gt, s.c_gt, "gt"),
            "O1l_gt_hands_plus_root_drift": (h_root_only, s.R_gt, s.c_gt, "gt"),
            "O2_pred_hands_pred_contact": (s.h_pred, s.R_imu, s.c_pred, "gt"),
            "O3_pred_hands_pred_contact_deadreckon": (s.h_pred, s.R_imu, s.c_pred, "self"),
            "O4_gt_hands_gt_contact_deadreckon": (s.h_gt, s.R_gt, s.c_gt, "self"),
            "O5_gt_hands_pred_contact_deadreckon": (s.h_gt, s.R_imu, s.c_pred, "self"),
            "O6_pred_hands_gt_contact_deadreckon": (s.h_pred, s.R_imu, s.c_gt, "self"),
        }
        # latch only once the object actually moves (oracle static detector)
        cases["O1m_pred_hands_latch_at_motion_onset"] = (s.h_pred, s.R_gt, s.c_gt, "gt", s.moving_gt)
        cases["O3m_pred_hands_pred_contact_deadreckon_static_aware"] = (s.h_pred, s.R_imu, s.c_pred, "self", s.moving_gt)
        cases["O6m_pred_hands_gt_contact_deadreckon_static_aware"] = (s.h_pred, s.R_imu, s.c_gt, "self", s.moving_gt)
        for nm, tup in cases.items():
            hand, Rm, c, anc = tup[:4]
            mv = tup[4] if len(tup) > 4 else None
            est = fk_estimator(hand, Rm, c, s.p_gt, anc, moving=mv)
            if nm == "O1_pred_hands":
                # object-IMU double integration inside GT segments, reset at onset (v0 = 0)
                est_imu = est.copy()
                est_avg = est.copy()
                for st, en in s.seg_gt:
                    disp = imu_integrate(s.acc_imu, st, en)
                    est_imu[st : en + 1] = s.p_gt[st] + disp
                    est_avg[st : en + 1] = 0.5 * (est[st : en + 1] + s.p_gt[st] + disp)
                for extra_nm, extra in (("O7_objimu_double_integration", est_imu), ("O7f_avg_fk_imu", est_avg)):
                    e2 = norm(extra - s.p_gt) * 100.0
                    ladder[extra_nm].append((e2.mean(), np.nan))
            e = norm(est - s.p_gt) * 100.0
            rel = []
            for h in range(2):
                m = s.c_gt[:, h]
                if m.any():
                    rel.append(norm((est - hand[:, h]) - (s.p_gt - s.h_gt[:, h]))[m] * 100.0)
            ladder[nm].append((e.mean(), np.concatenate(rel).mean() if rel else np.nan))
        ladder["model"].append((s.err.mean(), hoi_err[seqs.index(s)]))
    R["ladder"] = {nm: {"obj_err": nanmean([v[0] for v in vals]), "hoi_err": nanmean([v[1] for v in vals])} for nm, vals in ladder.items()}

    # F2. static veto, complementary filter, velocity integration ---------------
    hyb = defaultdict(list)
    det_tp = det_fp = det_fn = det_tn = 0
    hf_rms, lf_rms, tot_rms = [], [], []
    for s in seqs:
        inc = np.diff(s.p_pred, axis=0)
        # H1: model increments vetoed on GT-static frames
        keep = s.moving_gt[1:]
        p = np.vstack((s.p_pred[:1], s.p_pred[0] + np.cumsum(inc * keep[:, None], axis=0)))
        hyb["H1_model_static_veto_gt"].append(norm(p - s.p_gt).mean() * 100.0)
        # H2: same with a causal object-IMU static detector
        st_imu = imu_static_detector(s.R_imu, s.acc_imu)
        det_tp += int((st_imu & ~s.moving_gt).sum()); det_fp += int((st_imu & s.moving_gt).sum())
        det_fn += int((~st_imu & ~s.moving_gt).sum()); det_tn += int((~st_imu & s.moving_gt).sum())
        keep2 = ~st_imu[1:]
        p2 = np.vstack((s.p_pred[:1], s.p_pred[0] + np.cumsum(inc * keep2[:, None], axis=0)))
        hyb["H2_model_static_veto_objimu"].append(norm(p2 - s.p_gt).mean() * 100.0)
        # H3: complementary filter on O1 (GT anchor, pred hands): low freq from FK, high freq from object acc
        fk = fk_estimator(s.h_pred, s.R_gt, s.c_gt, s.p_gt, "gt")
        for tau in (8, 15, 30):
            cf = fk.copy()
            for st, en in s.seg_gt:
                p_imu = s.p_gt[st] + imu_integrate(s.acc_imu, st, en)
                d = p_imu - fk[st : en + 1]
                cf[st : en + 1] = fk[st : en + 1] + (d - ema(d, tau))
            hyb[f"H3_O1_plus_objacc_highpass_tau{tau}"].append(norm(cf - s.p_gt).mean() * 100.0)
        # H3b: same on the dead-reckoned realistic base O3
        base = fk_estimator(s.h_pred, s.R_imu, s.c_pred, s.p_gt, "self")
        cf = base.copy()
        for st, en in segments_of(s.any_pred):
            p_imu = base[st] + imu_integrate(s.acc_imu, st, en)
            d = p_imu - base[st : en + 1]
            cf[st : en + 1] = base[st : en + 1] + (d - ema(d, 15))
        hyb["H3b_O3_plus_objacc_highpass_tau15"].append(norm(cf - s.p_gt).mean() * 100.0)
        # H4: VC velocity integrated inside GT segments (reset at onset with GT)
        if "pred_obj_vel" in s.rec:
            v = s.rec["pred_obj_vel"].numpy()
            vi = fk.copy()
            for st, en in s.seg_gt:
                disp = np.cumsum(v[st : en + 1], axis=0) / FPS
                vi[st : en + 1] = s.p_gt[st] + disp - disp[0]
            hyb["H4_vc_velocity_integration_in_segment"].append(norm(vi - s.p_gt).mean() * 100.0)
        # H5: spectral split of the palm error inside GT segments (per hand)
        for h in range(2):
            for st, en in segments_of(s.c_gt[:, h]):
                e = s.d_hand[st : en + 1, h] - s.d_hand[st, h]
                lp = ema(e, 30)
                tot_rms.append(np.sqrt((norm(e) ** 2).mean()))
                lf_rms.append(np.sqrt((norm(lp) ** 2).mean()))
                hf_rms.append(np.sqrt((norm(e - lp) ** 2).mean()))
    R["hybrids"] = {k: nanmean(v) for k, v in hyb.items()}
    R["imu_static_detector"] = {
        "precision": det_tp / max(det_tp + det_fp, 1), "recall": det_tp / max(det_tp + det_fn, 1),
        "frac_frames_static_gt": (det_tp + det_fn) / max(det_tp + det_fp + det_fn + det_tn, 1),
    }
    R["palm_drift_spectrum"] = {"total_rms": float(np.mean(tot_rms)) * 100, "lowpass_1s_rms": float(np.mean(lf_rms)) * 100, "highpass_1s_rms": float(np.mean(hf_rms)) * 100}

    # G2. predicted-onset flicker and hand spikes ------------------------------
    n_gt_on, n_pred_on, multi = 0, 0, 0
    contact_to_motion = []
    for s in seqs:
        for h in range(2):
            gt_segs = segments_of(s.c_gt[:, h])
            pred_segs = segments_of(s.c_hyst[:, h])
            n_gt_on += len(gt_segs)
            n_pred_on += len(pred_segs)
            for st, en in gt_segs:
                inside = sum(1 for ps, pe in pred_segs if st <= ps <= en)
                multi += int(inside >= 2)
                mv = np.flatnonzero(s.moving_gt[st : en + 1])
                contact_to_motion.append(int(mv[0]) if mv.size else en - st + 1)
    palm_all = np.concatenate([norm(s.d_hand).reshape(-1) for s in seqs]) * 100.0
    R["flicker"] = {
        "gt_onsets": n_gt_on,
        "pred_onsets_hysteresis": n_pred_on,
        "frac_gt_segments_with_2plus_pred_onsets": multi / max(n_gt_on, 1),
        "mean_frames_contact_to_motion_onset": float(np.mean(contact_to_motion)),
        "median_frames_contact_to_motion_onset": float(np.median(contact_to_motion)),
        "gt_contact_frames_static": float((~np.concatenate([s.moving_gt[s.any_gt] for s in seqs])).mean()),
    }
    R["hand_spikes"] = {
        "palm_err_p95": float(np.percentile(palm_all, 95)),
        "frac_palm_err_gt_15cm": float((palm_all > 15).mean()),
    }

    # H. v3 anchor stability ---------------------------------------------------
    if "pred_lhand_segment_anchor" in seqs[0].rec:
        jumps, n = [], 0
        for s in seqs:
            for h, key in enumerate(("pred_lhand_segment_anchor", "pred_rhand_segment_anchor")):
                a = s.rec[key].numpy()
                act = s.rec["segment_contact_active"].numpy()[:, h] > 0.5
                for st, en in segments_of(act):
                    if en > st:
                        jumps.append(norm(np.diff(a[st : en + 1], axis=0)).mean() * 100.0)
                        n += 1
        R["v3_anchor_mean_step_cm_inside_pred_segment"] = nanmean(jumps)
    return R


def per_sequence_table(seqs_a: List[Seq], seqs_b: List[Seq], label_a: str, label_b: str, top: int = 12) -> Dict:
    ea = {s.name: s.err.mean() for s in seqs_a}
    eb = {s.name: s for s in seqs_b}
    rows = []
    for name, a in ea.items():
        s = eb.get(name)
        if s is None:
            continue
        seg_len = np.mean([en - st + 1 for st, en in s.seg_gt]) if s.seg_gt else 0
        rows.append({
            "seq": name,
            "obj": s.obj,
            "T": s.T,
            "n_seg": len(s.seg_gt),
            "mean_seg_len": float(seg_len),
            "bimanual_frac": float((s.c_gt.all(axis=1)).mean()),
            label_a: float(a),
            label_b: float(s.err.mean()),
            "delta": float(s.err.mean() - a),
        })
    deltas = np.array([r["delta"] for r in rows])
    out = {
        "n": len(rows),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "frac_b_worse": float((deltas > 0).mean()),
        "frac_b_worse_by_1cm": float((deltas > 1).mean()),
        "frac_a_worse_by_1cm": float((deltas < -1).mean()),
        "worst_for_b": sorted(rows, key=lambda r: -r["delta"])[:top],
        "best_for_b": sorted(rows, key=lambda r: r["delta"])[:top],
    }
    # correlations of delta with sequence factors
    for key in ("T", "n_seg", "mean_seg_len", "bimanual_frac"):
        x = np.array([r[key] for r in rows], dtype=np.float64)
        out[f"corr_delta_{key}"] = float(np.corrcoef(x, deltas)[0, 1]) if x.std() > 0 else np.nan
    return out


def load(path: str) -> List[Seq]:
    payload = torch.load(path, weights_only=False)
    return [Seq(r) for r in payload["records"] if r.get("has_object", True)]


def fmt(v) -> str:
    return "  nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:6.2f}"


def print_report(R: Dict) -> None:
    print(f"\n===== {R['label']}  ({R['n_seq']} seq) =====")
    m = R["metrics"]
    print(f"obj_err(seq-mean) {fmt(m['obj_err_seqmean'])}  median {fmt(m['obj_err_median'])}  frame {fmt(m['obj_err_frame'])}  hoi_err {fmt(m['hoi_err_seqmean'])}  jitter {fmt(m['obj_jitter'])}")
    print("-- error vs frames since GT onset (err | growth since onset | frames)")
    for k, v in R["since_onset"].items():
        print(f"   {k:>8}: {fmt(v['err'])} | {fmt(v['growth'])} | {v['frames']}")
    print(f"   in-contact frames beyond 60 from onset: {R['in_contact_frames_beyond_60']*100:.1f}%")
    print("-- phases (frac | err | contribution to frame-mean)")
    for k, v in R["phase"].items():
        print(f"   {k:>18}: {v['frac_frames']*100:5.1f}% | {fmt(v['err'])} | {fmt(v['contrib'])}")
    d = R["error_delta"]
    print(f"-- err at first onset {fmt(d['err_at_first_onset'])} | Δ over segment {fmt(d['delta_over_segment'])} ({fmt(d['delta_over_segment_per100f'])}/100f, mean len {d['mean_seg_len']:.0f}) | Δ over gap {fmt(d['delta_over_gap'])} ({fmt(d['delta_over_gap_per100f'])}/100f, mean len {d['mean_gap_len']:.0f})")
    print("-- contact categories (frac | err | Δerr/frame | pred motion | gt motion | w_static)")
    for k, v in R["contact_cats"].items():
        print(f"   {k}: {v['frac_frames']*100:5.1f}% | {fmt(v['err'])} | {v['derr_per_frame']:+.3f} | {v['pred_motion_cm_per_frame']:.3f} | {v['gt_motion_cm_per_frame']:.3f} | {v['w_static']:.2f}")
    print(f"   GT object motion/frame: untouched {R['gt_object_motion_when_untouched_cm_per_frame']:.3f} cm, touched {R['gt_object_motion_when_touched_cm_per_frame']:.3f} cm")
    b = R["bone"]
    print(f"-- bone on GT contact frames: err {fmt(b['err_frame'])} (len {fmt(b['len_err_frame'])}, angle {b['angle_deg_frame']:.1f}°) | at onset {fmt(b['err_at_onset'])} | corr(err_t, err_onset) {b['corr_err_t_vs_err_onset']:.2f}")
    for k, v in b["by_since"].items():
        print(f"   {k:>8}: err {fmt(v['err'])} | minus onset {v['err_minus_onset']:+.2f} | len {fmt(v['len_err'])} | angle {v['angle']:.1f}°")
    hd = R["hand"]
    print(f"-- hands: abs palm err {fmt(hd['abs_palm_err'])} | abs root err {fmt(hd['abs_root_err'])} | in-segment drift palm {fmt(hd['in_seg_drift_palm'])} = root part {fmt(hd['in_seg_drift_root_part'])} + local part {fmt(hd['in_seg_drift_local_part'])}")
    for k, v in hd["by_since"].items():
        print(f"   {k:>8}: palm {fmt(v['palm'])} | root {fmt(v['root'])} | local {fmt(v['local'])}")
    fl = R["flicker"]
    print(f"-- onsets: GT {fl['gt_onsets']} | predicted (hysteresis) {fl['pred_onsets_hysteresis']} | GT segments with >=2 predicted onsets {fl['frac_gt_segments_with_2plus_pred_onsets']*100:.1f}% | contact->motion delay mean {fl['mean_frames_contact_to_motion_onset']:.1f} f, median {fl['median_frames_contact_to_motion_onset']:.0f} f | GT-contact frames with static object {fl['gt_contact_frames_static']*100:.1f}%")
    hs = R["hand_spikes"]
    print(f"-- palm error p95 {hs['palm_err_p95']:.1f} cm | frames with palm err > 15 cm {hs['frac_palm_err_gt_15cm']*100:.2f}%")
    print("-- hybrid FK ladder (obj_err | hoi_err)")
    for k, v in R["ladder"].items():
        print(f"   {k:>40}: {fmt(v['obj_err'])} | {fmt(v['hoi_err'])}")
    print("-- hybrids on top of the model / ladder (obj_err)")
    for k, v in R["hybrids"].items():
        print(f"   {k:>44}: {fmt(v)}")
    dd = R["imu_static_detector"]
    print(f"   obj-IMU static detector: precision {dd['precision']:.3f} recall {dd['recall']:.3f} (GT static frames {dd['frac_frames_static_gt']*100:.1f}%)")
    ps = R["palm_drift_spectrum"]
    print(f"   palm drift inside segments RMS {ps['total_rms']:.2f} = lowpass(1s) {ps['lowpass_1s_rms']:.2f} + highpass {ps['highpass_1s_rms']:.2f}")
    if "v3_anchor_mean_step_cm_inside_pred_segment" in R:
        print(f"-- v3 latched anchor mean per-frame step inside predicted segments: {R['v3_anchor_mean_step_cm_inside_pred_segment']:.3f} cm")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+", help="label=path.pt")
    parser.add_argument("--compare", nargs=2, default=None, metavar=("A", "B"), help="labels for per-sequence comparison (B minus A)")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    loaded = {}
    reports = {}
    for item in args.dumps:
        label, path = item.split("=", 1)
        loaded[label] = load(path)
        reports[label] = analyze(label, loaded[label])
        print_report(reports[label])
    if args.compare:
        a, b = args.compare
        cmp = per_sequence_table(loaded[a], loaded[b], a, b)
        reports[f"compare_{a}_vs_{b}"] = cmp
        print(f"\n===== per-sequence: {b} minus {a} =====")
        print(f"n={cmp['n']} mean Δ {cmp['mean_delta']:+.2f} median {cmp['median_delta']:+.2f} | {b} worse on {cmp['frac_b_worse']*100:.0f}% (by >1cm: {cmp['frac_b_worse_by_1cm']*100:.0f}%), {a} worse by >1cm: {cmp['frac_a_worse_by_1cm']*100:.0f}%")
        for key in ("T", "n_seg", "mean_seg_len", "bimanual_frac"):
            print(f"   corr(Δ, {key}) = {cmp[f'corr_delta_{key}']:+.2f}")
        print(f"   worst for {b}:")
        for r in cmp["worst_for_b"]:
            print(f"     {r['seq']:>6} {r['obj']:<12} T={r['T']:4d} seg={r['n_seg']:2d} len={r['mean_seg_len']:5.0f} bi={r['bimanual_frac']:.2f}  {a}={r[a]:6.2f} {b}={r[b]:6.2f} Δ={r['delta']:+6.2f}")
        print(f"   best for {b}:")
        for r in cmp["best_for_b"]:
            print(f"     {r['seq']:>6} {r['obj']:<12} T={r['T']:4d} seg={r['n_seg']:2d} len={r['mean_seg_len']:5.0f} bi={r['bimanual_frac']:.2f}  {a}={r[a]:6.2f} {b}={r[b]:6.2f} Δ={r['delta']:+6.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(reports, f, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
