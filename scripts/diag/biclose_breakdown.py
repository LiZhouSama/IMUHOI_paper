#!/usr/bin/env python3
"""Where does bimanual modelling help (or not)?  Breakdown of OT dumps on GT bimanual frames.

Reads online dumps (scripts/dump_ot_predictions.py) and reports, per model:

* the official-style HOI (sequence-equal-weighted) under two publication replays:
  the evaluator's 4-way re-fusion (FK_L, FK_R, IMU, static with the 4-way weights) and,
  for bimanual-closure models, the true 5-way fusion including the hand-pair candidate;
* pooled HOI on GT bimanual vs single-hand hand-frames;
* the common / differential decomposition on GT bimanual frames
  (q_h = e_obj - e_hand, c = (q_L+q_R)/2, d = (q_L-q_R)/2; d depends on the hands only);
* pair-HOI of every fusion candidate and the per-frame selection oracle;
* bimanual gate weight statistics and whether the gate knows when the pair candidate is better.

    python scripts/diag/biclose_breakdown.py baseline=a.pt biclose=b.pt
"""
import sys

import numpy as np
import torch

AGE_BINS = [(1, 15), (16, 30), (31, 60), (61, 120), (121, 10 ** 9)]


def _np(x):
    return np.asarray(x, dtype=np.float64)


def replay(rec, five_way):
    """Causal publication replay; returns published object [n,3] and the re-anchored candidates."""
    n = int(rec["publish_len"])
    hs = _np(rec.get("stage1_pred_palm_glb_pos", rec["pred_palm_glb_pos"])[:n])
    hr = _np(rec["pred_palm_glb_pos"][:n])
    fk_l = hr[:, 0] + (_np(rec["pred_lhand_obj_trans"][:n]) - hs[:, 0])
    fk_r = hr[:, 1] + (_np(rec["pred_rhand_obj_trans"][:n]) - hs[:, 1])
    imu, static = _np(rec["pred_imu_obj_trans"][:n]), _np(rec["pred_static_obj_trans"][:n])
    w4 = _np(rec["gating_weights"][:n])
    pair = None
    if "pred_bimanual_obj_trans" in rec:
        pair = _np(rec["pred_bimanual_obj_trans"][:n]) + (hr.mean(1) - hs.mean(1))  # moves with the refined midpoint
    w5 = _np(rec["gating_weights_5"][:n]) if "gating_weights_5" in rec else None
    use5 = five_way and pair is not None and w5 is not None
    prev = static[0].copy()
    pub, imu_c, static_c = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    for t in range(n):
        ic = prev + (imu[t] - static[t])
        imu_c[t], static_c[t] = ic, prev
        if use5:
            cur = w5[t, 0] * fk_l[t] + w5[t, 1] * fk_r[t] + w5[t, 2] * ic + w5[t, 3] * prev + w5[t, 4] * pair[t]
        else:
            cur = w4[t, 0] * fk_l[t] + w4[t, 1] * fk_r[t] + w4[t, 2] * ic + w4[t, 3] * prev
        pub[t] = cur
        prev = cur
    return pub, {"fk_l": fk_l, "fk_r": fk_r, "imu": imu_c, "static": static_c, "pair": pair}, w5


def pair_hoi(p, hr, rel_gt):
    """Per-frame mean of the two hands' HOI error for object track p (cm)."""
    q_l = (p - hr[:, 0]) - rel_gt[:, 0]
    q_r = (p - hr[:, 1]) - rel_gt[:, 1]
    return 50.0 * (np.linalg.norm(q_l, axis=-1) + np.linalg.norm(q_r, axis=-1)), q_l, q_r


def analyse(records, five_way):
    seq_hoi, seq_obj = [], []
    pool = {k: [] for k in ("bi", "single", "c", "d", "d_stage1", "pairhoi", "wB", "gain_pair_vs_fkavg", "sep", "age")}
    cand_err = {}
    max_replay_diff = 0.0
    for rec in records:
        if not rec.get("has_object", True) or int(rec["publish_len"]) == 0:
            continue
        n = int(rec["publish_len"])
        pub, cands, w5 = replay(rec, five_way)
        if not five_way:
            ref = _np(rec["pred_obj_trans_published"][:n])
            max_replay_diff = max(max_replay_diff, float(np.abs(ref - pub).max()))
        p_gt = _np(rec["gt_obj_trans"][:n])
        g = _np(rec.get("gt_palm_pos_eval", rec["gt_palm_pos"])[:n])
        hr = _np(rec["pred_palm_glb_pos"][:n])
        hs = _np(rec.get("stage1_pred_palm_glb_pos", rec["pred_palm_glb_pos"])[:n])
        cl = np.asarray(rec["gt_lhand_contact"][:n]).astype(bool)
        cr = np.asarray(rec["gt_rhand_contact"][:n]).astype(bool)
        rel_gt = p_gt[:, None, :] - g
        seq_obj.append(np.linalg.norm(pub - p_gt, axis=-1).mean() * 100.0)
        _, q_l, q_r = pair_hoi(pub, hr, rel_gt)
        e_l, e_r = np.linalg.norm(q_l, axis=-1) * 100.0, np.linalg.norm(q_r, axis=-1) * 100.0
        errs = np.concatenate((e_l[cl], e_r[cr]))
        if errs.size:
            seq_hoi.append(errs.mean())
        both = cl & cr
        pool["bi"].append(np.concatenate((e_l[both], e_r[both])))
        pool["single"].append(np.concatenate((e_l[cl & ~cr], e_r[cr & ~cl])))
        if not both.any():
            continue
        c = 0.5 * (q_l + q_r)[both]
        d = 0.5 * (q_l - q_r)[both]
        pool["c"].append(np.linalg.norm(c, axis=-1) * 100.0)
        pool["d"].append(np.linalg.norm(d, axis=-1) * 100.0)
        d1 = 0.5 * ((hs[:, 1] - g[:, 1]) - (hs[:, 0] - g[:, 0]))[both]  # hands only (stage-1 palms)
        pool["d_stage1"].append(np.linalg.norm(d1, axis=-1) * 100.0)
        pool["pairhoi"].append(0.5 * (e_l + e_r)[both])
        # segment age since the GT bimanual onset
        age = np.zeros(n)
        run = 0
        for t in range(n):
            run = run + 1 if both[t] else 0
            age[t] = run
        pool["age"].append(age[both])
        tracks = {"fused": pub, "fk_l": cands["fk_l"], "fk_r": cands["fk_r"], "fk_avg": 0.5 * (cands["fk_l"] + cands["fk_r"]),
                  "imu": cands["imu"], "static": cands["static"]}
        if cands["pair"] is not None:
            tracks["pair"] = cands["pair"]
        per = {}
        for name, track in tracks.items():
            per[name] = pair_hoi(track, hr, rel_gt)[0][both]
            cand_err.setdefault(name, []).append(per[name])
        base4 = np.stack([per[k] for k in ("fk_l", "fk_r", "fk_avg", "imu", "static")], 0)
        cand_err.setdefault("oracle_4", []).append(base4.min(0))
        if "pair" in per:
            cand_err.setdefault("oracle_5", []).append(np.minimum(base4.min(0), per["pair"]))
            pool["gain_pair_vs_fkavg"].append(per["fk_avg"] - per["pair"])
            pool["sep"].append(np.linalg.norm(tracks["pair"] - tracks["fk_avg"], axis=-1)[both] * 100.0)
            if w5 is not None:
                pool["wB"].append(w5[both, 4])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
    out = {"hoi_seq": float(np.mean(seq_hoi)), "obj_seq": float(np.mean(seq_obj)), "replay_diff_m": max_replay_diff,
           "hoi_bi": cat(pool["bi"]).mean(), "hoi_single": cat(pool["single"]).mean(),
           "n_bi": cat(pool["bi"]).size, "n_single": cat(pool["single"]).size,
           "c": cat(pool["c"]).mean(), "d": cat(pool["d"]).mean(), "d_stage1": cat(pool["d_stage1"]).mean(),
           "pairhoi": cat(pool["pairhoi"]).mean(), "cand": {k: cat(v).mean() for k, v in cand_err.items()}}
    age, pairhoi = cat(pool["age"]), cat(pool["pairhoi"])
    out["by_age"] = {}
    for lo, hi in AGE_BINS:
        m = (age >= lo) & (age <= hi)
        row = {"n": int(m.sum()), "fused": pairhoi[m].mean() if m.any() else np.nan}
        for k in ("fk_avg", "pair"):
            if k in cand_err:
                row[k] = cat(cand_err[k])[m].mean() if m.any() else np.nan
        out["by_age"][f"{lo}-{hi if hi < 10 ** 8 else ''}"] = row
    if pool["wB"]:
        wb, gain, sep = cat(pool["wB"]), cat(pool["gain_pair_vs_fkavg"]), cat(pool["sep"])
        out["wB"] = {"mean": wb.mean(), "median": float(np.median(wb)), "frac>0.1": (wb > 0.1).mean(), "frac>0.3": (wb > 0.3).mean(),
                     "pair_better_frac": (gain > 0).mean(), "corr(wB, fk_avg-pair)": float(np.corrcoef(wb, gain)[0, 1]),
                     "wB|pair_better": wb[gain > 0].mean(), "wB|pair_worse": wb[gain <= 0].mean(), "|pair-fk_avg|": sep.mean()}
    return out


def main():
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        records = torch.load(path, map_location="cpu")["records"]
        has_pair = any("pred_bimanual_obj_trans" in r for r in records[:5])
        for five_way in ([False, True] if has_pair else [False]):
            r = analyse(records, five_way)
            tag = f"{name} [{'5-way publication (pair candidate kept)' if five_way else '4-way publication (evaluator)'}]"
            print(f"\n=== {tag}")
            print(f"official-style: HOI {r['hoi_seq']:.3f}  obj {r['obj_seq']:.3f}" + ("" if five_way else f"   (replay vs dump max |diff| {r['replay_diff_m']:.2e} m)"))
            print(f"pooled HOI: bimanual hand-frames {r['hoi_bi']:.2f} (n={r['n_bi']})   single-hand {r['hoi_single']:.2f} (n={r['n_single']})")
            print(f"bimanual frames: pair-HOI {r['pairhoi']:.2f} | common |c| {r['c']:.2f} | differential |d| {r['d']:.2f} (stage-1 palms {r['d_stage1']:.2f})")
            print("candidates (pair-HOI on bimanual frames): " + "  ".join(f"{k} {v:.2f}" for k, v in r["cand"].items()))
            print("by bimanual-segment age: " + "  ".join(f"[{k}] n={v['n']} " + " ".join(f"{kk} {vv:.2f}" for kk, vv in v.items() if kk != 'n') for k, v in r["by_age"].items()))
            if "wB" in r:
                print("pair gate: " + "  ".join(f"{k} {v:.3f}" for k, v in r["wB"].items()))


if __name__ == "__main__":
    main()
