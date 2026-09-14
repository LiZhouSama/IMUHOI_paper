#!/usr/bin/env python3
"""Floor non-penetration replay with test-time-legal floor levels, plus the
tip-to-surface statistics (NaN-safe).  Same cache as mesh_replay_pred.py."""
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
from scripts.analyze_ot_dumps import rot6d  # noqa: E402

CACHE = os.path.join(ROOT, "outputs/IMUHOI++/RNN2/hoi_contact_consistency_20260905/test_online.pt")
OLD_TEST = os.path.join(ROOT, "process/IMUHOI++/processed_split_data_OMOMO/test")
OUT = os.path.join(ROOT, "outputs/analysis/mesh_probe_20260907", "mesh_replay_floor.json")


def norm(x):
    return np.linalg.norm(x, axis=-1)


def hoi_err(p_obj, h_pred, p_gt, h_gt, c_gt):
    errs = []
    for k in range(2):
        m = c_gt[:, k]
        if m.any():
            errs.append(norm((p_obj - h_pred[:, k]) - (p_gt - h_gt[:, k]))[m])
    return float(np.concatenate(errs).mean() * 100) if errs else float("nan")


def main():
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    recs = cache["records"]
    nn = {}
    seqs = []
    tip_m_all, tip_g_all = [], []
    print("pass 1: geometry", flush=True)
    for ri, rec in enumerate(recs):
        name = rec["obj_name"]
        if name not in nn:
            nn[name] = SurfaceNN(load_mesh(name))
        q = nn[name]
        old = torch.load(os.path.join(OLD_TEST, rec["seq_file"]), map_location="cpu", weights_only=False)
        T = int(rec["publish_len"])
        s = old["obj_scale"][:T].double().numpy()
        ok = s > 1e-6
        p_gt = rec["gt_obj_trans"].numpy().astype(np.float64)
        R_gt = rot6d(rec["gt_obj_rot"]).astype(np.float64)
        R = rot6d(rec["gt_obj_imu"][:, 3:9]).astype(np.float64)
        h_gt = rec["gt_palm_pos"].numpy().astype(np.float64)
        h_pub = rec["pred_palm_glb_pos"].numpy().astype(np.float64)
        c_gt = np.stack((rec["gt_lhand_contact"].numpy(), rec["gt_rhand_contact"].numpy()), 1).astype(bool)
        moving = rec["gt_obj_contact"].numpy().astype(bool)
        p_pub = rec["pred_obj_trans_published"].numpy().astype(np.float64)
        bones = np.stack((rec["pred_lhand_obj_vector"].numpy(), rec["pred_rhand_obj_vector"].numpy()), 1).astype(np.float64)
        min_y_gt = q.min_y_world(torch.from_numpy(R_gt), torch.from_numpy(s), torch.from_numpy(p_gt)).cpu().numpy()
        min_y_pred = q.min_y_world(torch.from_numpy(R), torch.from_numpy(s), torch.from_numpy(p_pub)).cpu().numpy()
        static = ~moving
        floor_seq = float(np.quantile(min_y_gt[static & ok], 0.05)) if (static & ok).sum() > 5 else float(np.quantile(min_y_gt[ok], 0.05))
        # tip statistics (NaN safe)
        for k in range(2):
            m = c_gt[:, k] & ok
            if m.sum() < 5:
                continue
            tm = (-bones[m, k]) / s[m, None]
            tg = np.einsum("tji,tj->ti", R_gt[m], h_gt[m, k] - p_gt[m]) / s[m, None]
            _, dm = q.query(torch.from_numpy(tm))
            _, dg = q.query(torch.from_numpy(tg))
            tip_m_all.extend((dm.cpu().numpy() * s[m]).tolist())
            tip_g_all.extend((dg.cpu().numpy() * s[m]).tolist())
        seqs.append(dict(name=name, p_gt=p_gt, p_pub=p_pub, h_gt=h_gt, h_pub=h_pub, c_gt=c_gt, min_y_pred=min_y_pred, floor_seq=floor_seq, ok=ok, static=static))
        if ri % 200 == 0:
            print(ri, flush=True)

    floors = np.array([s_["floor_seq"] for s_ in seqs])
    floor_global = float(np.median(floors))
    floor_obj = defaultdict(list)
    for s_ in seqs:
        floor_obj[s_["name"]].append(s_["floor_seq"])
    floor_obj = {k: float(np.median(v)) for k, v in floor_obj.items()}
    print("per-seq GT floor level cm: median %.2f q10 %.2f q90 %.2f" % (floor_global * 100, np.quantile(floors, 0.1) * 100, np.quantile(floors, 0.9) * 100))
    print("per-object floor (cm):", {k: round(v * 100, 1) for k, v in floor_obj.items()})

    res = defaultdict(list)
    per_obj = defaultdict(lambda: defaultdict(list))
    for s_ in seqs:
        p_gt, p_pub, h_gt, h_pub, c_gt = s_["p_gt"], s_["p_pub"], s_["h_gt"], s_["h_pub"], s_["c_gt"]
        my = s_["min_y_pred"]
        base = float(norm(p_pub - p_gt).mean() * 100)
        res["base_obj"].append(base)
        res["base_hoi"].append(hoi_err(p_pub, h_pub, p_gt, h_gt, c_gt))
        per_obj[s_["name"]]["base_obj"].append(base)
        for label, floor in (("seqGT", s_["floor_seq"]), ("global", floor_global), ("perobj", floor_obj[s_["name"]])):
            for margin in (0.0, 0.02, 0.05):
                pen = np.maximum(0.0, (floor - margin) - my)
                pen[~s_["ok"]] = 0.0
                po = p_pub.copy()
                po[:, 1] += pen
                ho = h_pub.copy()
                ho[:, :, 1] += pen[:, None]
                key = f"nonpen_{label}_m{int(margin*100)}"
                res[f"{key}_obj"].append(float(norm(po - p_gt).mean() * 100))
                res[f"{key}_hoi_objonly"].append(hoi_err(po, h_pub, p_gt, h_gt, c_gt))
                res[f"{key}_hoi_joint"].append(hoi_err(po, ho, p_gt, h_gt, c_gt))
                res[f"{key}_fire_frac"].append(float((pen > 0).mean()))
                # does the hand shift actually reduce palm error on contact frames?
                errs_b, errs_a = [], []
                for k in range(2):
                    m = c_gt[:, k]
                    if m.any():
                        errs_b.append(norm(h_pub[m, k] - h_gt[m, k]))
                        errs_a.append(norm(ho[m, k] - h_gt[m, k]))
                if errs_b:
                    res[f"{key}_palm_before"].append(float(np.concatenate(errs_b).mean() * 100))
                    res[f"{key}_palm_after"].append(float(np.concatenate(errs_a).mean() * 100))
                per_obj[s_["name"]][f"{key}_obj"].append(res[f"{key}_obj"][-1])
    summary = {k: float(np.mean(v)) for k, v in res.items()}
    tm, tg = np.asarray(tip_m_all) * 100, np.asarray(tip_g_all) * 100
    summary["tip_surface_dist_model_cm_q50_q90_mean"] = [float(np.median(tm)), float(np.quantile(tm, 0.9)), float(tm.mean())]
    summary["tip_surface_dist_gt_cm_q50_q90_mean"] = [float(np.median(tg)), float(np.quantile(tg, 0.9)), float(tg.mean())]
    summary["floor_global_cm"] = floor_global * 100
    summary["floor_per_object_cm"] = {k: v * 100 for k, v in floor_obj.items()}
    summary["per_object"] = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in per_obj.items()}
    with open(OUT, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_object"}, indent=2))
    for n, d in sorted(summary["per_object"].items()):
        print(f"{n:13s} base={d['base_obj']:.2f} seqGT0={d['nonpen_seqGT_m0_obj']:.2f} glob0={d['nonpen_global_m0_obj']:.2f} glob2={d['nonpen_global_m2_obj']:.2f} obj0={d['nonpen_perobj_m0_obj']:.2f} obj2={d['nonpen_perobj_m2_obj']:.2f}")


if __name__ == "__main__":
    main()
