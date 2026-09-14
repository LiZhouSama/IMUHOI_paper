#!/usr/bin/env python3
"""Compact OT diagnostics for several online dumps: label=path ...

Per dump (published object + refined palms, GT contact frames):
  official-style obj / HOI / jitter, bone error at onset and growth, the
  compensation split of the HOI vector along / orthogonal to the hand error,
  fusion weight statistics on GT static / contact frames, phase errors,
  and per-object HOI.  No mesh needed.
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
from scripts.analyze_ot_dumps import hysteresis, rot6d, segments_of  # noqa: E402


def norm(x):
    return np.linalg.norm(x, axis=-1)


def f64(t):
    return t.numpy().astype(np.float64)


def stats(path):
    recs = torch.load(path, map_location="cpu", weights_only=False)["records"]
    S = defaultdict(list)
    per_obj = defaultdict(lambda: defaultdict(list))
    for rec in recs:
        T = int(rec["publish_len"])
        p_gt = f64(rec["gt_obj_trans"]); R_gt = rot6d(rec["gt_obj_rot"]).astype(np.float64)
        R = rot6d(rec["gt_obj_imu"][:, 3:9]).astype(np.float64)
        h_gt = f64(rec["gt_palm_pos"]); h_r = f64(rec["pred_palm_glb_pos"])
        c_gt = np.stack((rec["gt_lhand_contact"].numpy(), rec["gt_rhand_contact"].numpy()), 1).astype(bool)
        moving = rec["gt_obj_contact"].numpy().astype(bool)
        w = f64(rec["gating_weights"]); prob = f64(rec["pred_hand_contact_prob"])
        p_pub = f64(rec["pred_obj_trans_published"]); p_raw = f64(rec["pred_obj_trans"])
        obj = rec["obj_name"]
        e = norm(p_pub - p_gt) * 100
        S["obj"].append(e.mean()); S["obj_raw"].append(norm(p_raw - p_gt).mean() * 100)
        hoi = []
        for k in range(2):
            m = c_gt[:, k]
            if m.any():
                hoi.append(norm((p_pub - h_r[:, k]) - (p_gt - h_gt[:, k]))[m] * 100)
        hv = float(np.concatenate(hoi).mean()) if hoi else np.nan
        S["hoi"].append(hv); per_obj[obj]["hoi"].append(hv); per_obj[obj]["obj"].append(e.mean())
        if T >= 3:
            S["jitter"].append(norm(p_pub[2:] - 2 * p_pub[1:-1] + p_pub[:-2]).mean() * 1000)
        static = ~moving; anyc = c_gt.any(1)
        if static.any():
            S["err_static"].append(e[static].mean()); S["w_static_on_static"].append(w[static, 3].mean())
            S["step_on_static_cm"].append((norm(np.diff(p_pub, axis=0)) * 100)[static[1:]].mean() if static[1:].any() else np.nan)
        if anyc.any():
            S["w_fk_on_contact"].append(w[anyc, :2].sum(1).mean()); S["err_contact"].append(e[anyc].mean())
        c_h = hysteresis(prob[:, :2], 0.15, 0.075)
        fp = c_h.any(1) & ~anyc
        if fp.any():
            S["err_fp"].append(e[fp].mean()); S["frac_fp"].append(fp.mean())
        segs = segments_of(anyc)
        if segs:
            S["err_first_onset"].append(e[segs[0][0]])
            for a, b in segs:
                S["seg_growth"].append(e[b] - e[a])
        for k in range(2):
            for a, b in segments_of(c_gt[:, k]):
                idx = np.arange(a, b + 1)
                e_b = (p_pub[idx] - h_r[idx, k]) - (p_gt[idx] - h_gt[idx, k])
                e_h = h_r[idx, k] - h_gt[idx, k]
                eb = norm(e_b) * 100
                S["bone_onset"].append(eb[0]); S["bone_all"].extend(eb.tolist())
                if len(idx) > 60:
                    S["bone_growth_60"].append(eb[60:].mean() - eb[0])
                alpha = (e_b * e_h).sum(1) / ((e_h * e_h).sum(1) + 1e-9)
                S["alpha"].extend(alpha.tolist())
                S["par_cm"].extend((norm(alpha[:, None] * e_h) * 100).tolist())
                S["orth_cm"].extend((norm(e_b - alpha[:, None] * e_h) * 100).tolist())
                S["hand_cm"].extend((norm(e_h) * 100).tolist())
    out = {k: float(np.nanmean(v)) for k, v in S.items()}
    out["n"] = len(recs)
    out["per_object_hoi"] = {o: float(np.nanmean(d["hoi"])) for o, d in per_obj.items()}
    out["per_object_obj"] = {o: float(np.nanmean(d["obj"])) for o, d in per_obj.items()}
    return out


def main():
    dumps = [a.split("=", 1) for a in sys.argv[1:]]
    res = {lab: stats(path) for lab, path in dumps}
    keys = ["obj", "obj_raw", "hoi", "jitter", "err_static", "step_on_static_cm", "w_static_on_static", "w_fk_on_contact",
            "err_contact", "err_fp", "frac_fp", "err_first_onset", "seg_growth", "bone_onset", "bone_all", "bone_growth_60",
            "alpha", "par_cm", "orth_cm", "hand_cm"]
    print(f"{'stat':20s}" + "".join(f"{lab:>14s}" for lab, _ in dumps))
    for k in keys:
        print(f"{k:20s}" + "".join(f"{res[lab].get(k, float('nan')):14.3f}" for lab, _ in dumps))
    objs = sorted(res[dumps[0][0]]["per_object_hoi"])
    print("\nper-object HOI")
    for o in objs:
        print(f"{o:14s}" + "".join(f"{res[lab]['per_object_hoi'][o]:14.2f}" for lab, _ in dumps))
    print("\nper-object obj")
    for o in objs:
        print(f"{o:14s}" + "".join(f"{res[lab]['per_object_obj'][o]:14.2f}" for lab, _ in dumps))
    out = os.path.join(os.path.dirname(dumps[0][1]), "variant_stats.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
