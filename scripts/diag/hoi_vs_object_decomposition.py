#!/usr/bin/env python3
"""Why can a model have a much better HOI error and a much worse object error?

HOI is object-RELATIVE-to-hand: q_h = e_obj - e_hand with e_obj = p_pred - p_gt and e_hand = h_pred - h_gt.
So a model whose object follows its own (drifting) hands keeps |q| small however far both are from the truth,
while a model that pulls the object toward its absolute position necessarily breaks the relative geometry.

Per model, on GT contact hand-frames (the official HOI support) this reports
  * |e_obj|, |e_hand|, |q| and the common-mode fraction: how much of e_obj is explained by e_hand
    (alpha = <e_obj, e_hand> / |e_hand|^2, and the residual |e_obj - e_hand|),
  * the same split for the sequence-mean offset (global drift) vs. the in-sequence fluctuation,
  * object error on contact vs. non-contact frames, gate weights, and root/palm errors,
  * counterfactuals: HOI if the object were the GT object (pure hand error), and HOI after removing each
    sequence's mean common offset (a drift-insensitive HOI).

    python scripts/diag/hoi_vs_object_decomposition.py ours=a.pt imuhoi=b.pt
"""
import sys

import numpy as np
import torch


def _np(x):
    return np.asarray(x, dtype=np.float64)


def analyse(records):
    acc = {k: [] for k in ("e_obj", "e_hand", "q", "resid", "alpha_num", "alpha_den", "q_centered", "q_gt_obj",
                           "obj_all", "obj_contact", "obj_free", "root", "palm", "w", "drift_obj", "drift_hand",
                           "fluct_obj", "fluct_hand", "cos", "obj_minus_root", "cos_obj_root", "hand_minus_root",
                           "obj_err_no_root")}
    seq_hoi, seq_hoi_centered = [], []
    for r in records:
        n = int(r["publish_len"])
        if not r.get("has_object", True) or n == 0:
            continue
        p_pred = _np(r["pred_obj_trans_published"][:n])
        p_gt = _np(r["gt_obj_trans"][:n])
        h_pred = _np(r["pred_palm_glb_pos"][:n])
        h_gt = _np(r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n])
        cl = np.asarray(r["gt_lhand_contact"][:n]).astype(bool)
        cr = np.asarray(r["gt_rhand_contact"][:n]).astype(bool)
        e_obj = p_pred - p_gt
        e_hand = h_pred - h_gt
        acc["obj_all"].append(np.linalg.norm(e_obj, axis=-1) * 100.0)
        any_contact = cl | cr
        if any_contact.any():
            acc["obj_contact"].append(np.linalg.norm(e_obj[any_contact], axis=-1) * 100.0)
        if (~any_contact).any():
            acc["obj_free"].append(np.linalg.norm(e_obj[~any_contact], axis=-1) * 100.0)
        e_root = _np(r["root_trans_pred"][:n]) - _np(r["gt_trans"][:n])
        acc["root"].append(np.linalg.norm(e_root, axis=-1) * 100.0)
        # How much of the object error is just the human's global drift?
        acc["obj_minus_root"].append(np.linalg.norm(e_obj - e_root, axis=-1) * 100.0)
        acc["hand_minus_root"].append(np.linalg.norm(e_hand - e_root[:, None], axis=-1).reshape(-1) * 100.0)
        denom_r = (np.linalg.norm(e_obj, axis=-1) * np.linalg.norm(e_root, axis=-1)).clip(1e-9)
        acc["cos_obj_root"].append((e_obj * e_root).sum(-1) / denom_r)
        acc["w"].append(_np(r["gating_weights"][:n]))
        # sequence-level common offset (drift) of the object and of the hands
        mean_obj = e_obj.mean(0)
        mean_hand = e_hand.reshape(-1, 3).mean(0)
        acc["drift_obj"].append([np.linalg.norm(mean_obj) * 100.0])
        acc["drift_hand"].append([np.linalg.norm(mean_hand) * 100.0])
        acc["fluct_obj"].append(np.linalg.norm(e_obj - mean_obj, axis=-1) * 100.0)
        acc["fluct_hand"].append(np.linalg.norm(e_hand - mean_hand, axis=-1).reshape(-1) * 100.0)
        q_list, qc_list = [], []
        for hand, mask in ((0, cl), (1, cr)):
            if not mask.any():
                continue
            eo, eh = e_obj[mask], e_hand[mask, hand]
            q = eo - eh
            acc["e_obj"].append(np.linalg.norm(eo, axis=-1) * 100.0)
            acc["e_hand"].append(np.linalg.norm(eh, axis=-1) * 100.0)
            acc["q"].append(np.linalg.norm(q, axis=-1) * 100.0)
            acc["palm"].append(np.linalg.norm(eh, axis=-1) * 100.0)
            acc["alpha_num"].append((eo * eh).sum(-1))
            acc["alpha_den"].append((eh * eh).sum(-1))
            acc["resid"].append(np.linalg.norm(eo - eh, axis=-1) * 100.0)
            denom = (np.linalg.norm(eo, axis=-1) * np.linalg.norm(eh, axis=-1)).clip(1e-9)
            acc["cos"].append((eo * eh).sum(-1) / denom)
            acc["q_gt_obj"].append(np.linalg.norm(-eh, axis=-1) * 100.0)  # object = GT -> HOI is the hand error
            q_list.append(np.linalg.norm(q, axis=-1) * 100.0)
            qc_list.append(q)
        if q_list:
            seq_hoi.append(np.concatenate(q_list).mean())
            qc = np.concatenate(qc_list, axis=0)
            centered = np.linalg.norm(qc - qc.mean(0), axis=-1) * 100.0   # remove this sequence's mean HOI offset
            acc["q_centered"].append(centered)
            seq_hoi_centered.append(centered.mean())
    cat = lambda k: np.concatenate(acc[k]) if acc[k] else np.zeros(1)  # noqa: E731
    w = np.concatenate(acc["w"], axis=0)
    return {
        "HOI (sequence-equal)": float(np.mean(seq_hoi)),
        "HOI drift-removed (sequence-equal)": float(np.mean(seq_hoi_centered)),
        "|e_obj| on contact frames": cat("e_obj").mean(),
        "|e_hand| (palm) on contact frames": cat("e_hand").mean(),
        "|q| = HOI (pooled)": cat("q").mean(),
        "|e_obj - e_hand| (= |q|, check)": cat("resid").mean(),
        "alpha = <e_obj,e_hand>/|e_hand|^2": float(cat("alpha_num").sum() / max(cat("alpha_den").sum(), 1e-12)),
        "cos(e_obj, e_hand)": float(cat("cos").mean()),
        "HOI if object were GT (= palm error)": cat("q_gt_obj").mean(),
        "object error: all / contact / contact-free": (cat("obj_all").mean(), cat("obj_contact").mean(), cat("obj_free").mean()),
        "object drift (per-seq mean offset) / fluctuation": (cat("drift_obj").mean(), cat("fluct_obj").mean()),
        "hand drift (per-seq mean offset) / fluctuation": (cat("drift_hand").mean(), cat("fluct_hand").mean()),
        "root error": cat("root").mean(),
        "object error after removing the root drift": cat("obj_minus_root").mean(),
        "palm error after removing the root drift": cat("hand_minus_root").mean(),
        "cos(e_obj, e_root)": float(cat("cos_obj_root").mean()),
        "gate weights [FK_L, FK_R, IMU, static]": tuple(np.round(w.mean(0), 3)),
        "frames": len(cat("obj_all")),
    }


def main():
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        print(f"\n=== {name}")
        for k, v in analyse(torch.load(path, map_location="cpu")["records"]).items():
            if isinstance(v, tuple):
                print(f"  {k:44s} " + " / ".join(f"{x:.2f}" for x in v))
            elif isinstance(v, int):
                print(f"  {k:44s} {v}")
            else:
                print(f"  {k:44s} {v:.3f}")


if __name__ == "__main__":
    main()
