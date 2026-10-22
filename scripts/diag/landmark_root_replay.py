#!/usr/bin/env python3
"""Open-loop feasibility of 'static object as landmark' root correction (BEHAVE dump).

At a grasp onset t0 (hand contact starts after the object was static), the stored
object estimate pub[t0-1] and the hand-based FK candidate disagree by
d = mean_{t0..t0+K-1} FK(t) - pub[t0-1] = e_root + e_fk - e_static.
Shifting the human root by -k*d (from t0+K on, cumulatively) trusts the static
object.  Replays the v2 fusion with FK candidates shifted by the cumulative
correction D(t) and reports obj / HOI / root error.
Onsets: GT (hand contact rise after >= 15 frames without contact, object static
for the 10 frames before) or VC (same rule on predicted probabilities).

    python scripts/diag/landmark_root_replay.py dump.pt
"""
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from nohand_anchor_oracle import replay  # noqa: E402


def onsets(contact_any, moving, min_gap=15, static_before=10):
    out = []
    T = len(contact_any)
    last_contact = -10**9
    for t in range(1, T):
        if contact_any[t - 1]:
            last_contact = t - 1
        if contact_any[t] and not contact_any[t - 1] and t - last_contact > min_gap \
                and t >= static_before and not moving[t - static_before:t].any():
            out.append(t)
    return out


def hoi(pub, gt, palms_pred, palms_gt, gl, gr):
    errs = []
    for h, m in ((0, gl), (1, gr)):
        if m.any():
            rel_p = pub - palms_pred[:, h]
            rel_g = gt - palms_gt[:, h]
            errs.append(np.linalg.norm(rel_p - rel_g, axis=-1)[m])
    return np.concatenate(errs) if errs else np.zeros(0)


def main():
    recs = torch.load(sys.argv[1], map_location="cpu")["records"]
    K = 5
    configs = [("base", None, 0.0)] + [(f"{src} onsets, k={k}", src, k) for src in ("GT", "VC") for k in (0.5, 1.0)]
    res = {name: {"obj": [], "hoi": [], "root": [], "n": 0} for name, _, _ in configs}
    diag = {"e_root": [], "after_k1": [], "e_fk": [], "e_static": []}
    for r in recs:
        gt = r["gt_obj_trans"].numpy().astype(np.float64); T = len(gt)
        w = r["gating_weights"].numpy().astype(np.float64)
        stage1 = r["stage1_pred_palm_glb_pos"].numpy().astype(np.float64)
        palms = r["pred_palm_glb_pos"].numpy().astype(np.float64)
        fk_l = palms[:, 0] + (r["pred_lhand_obj_trans"].numpy() - stage1[:, 0])
        fk_r = palms[:, 1] + (r["pred_rhand_obj_trans"].numpy() - stage1[:, 1])
        disp = (r["pred_imu_obj_trans"].numpy() - r["pred_static_obj_trans"].numpy()).astype(np.float64)
        start = r["pred_static_obj_trans"].numpy()[0].astype(np.float64)
        pub0 = r["pred_obj_trans_published"].numpy().astype(np.float64)
        e_root = r["root_trans_pred"].numpy().astype(np.float64)[:T] - r["gt_trans"].numpy().astype(np.float64)[:T]
        gl = r["gt_lhand_contact"].numpy() > .5; gr = r["gt_rhand_contact"].numpy() > .5
        gm = r["gt_obj_contact"].numpy() > .5
        p = r["pred_hand_contact_prob"].numpy().astype(np.float64)
        palms_gt = r["gt_palm_pos_eval"].numpy().astype(np.float64)[:T]
        for name, src, k in configs:
            D = np.zeros((T, 3))
            if src is not None:
                if src == "GT":
                    ons = onsets(gl | gr, gm)
                    hand_of = lambda t: 0 if gl[t] and (not gr[t]) else (1 if gr[t] and not gl[t] else -1)
                else:
                    pc = p[:, :2] > 0.5
                    ons = onsets(pc.any(-1), p[:, 2] > 0.5)
                    hand_of = lambda t: int(np.argmax(p[t, :2])) if abs(p[t, 0] - p[t, 1]) > 0.2 else -1
                total = np.zeros(3)
                pub = pub0
                for t0 in ons:
                    if t0 + K >= T:
                        continue
                    # FK of the grasping hand(s), with the corrections applied so far
                    h = hand_of(t0)
                    seg = slice(t0, t0 + K)
                    fk = (fk_l[seg] if h == 0 else fk_r[seg] if h == 1 else 0.5 * (fk_l[seg] + fk_r[seg])) - total
                    d = fk.mean(0) - pub[t0 - 1]
                    total = total + k * d
                    D[t0 + K:] = total
                    if src == "GT" and k == 1.0:
                        diag["e_root"].append(np.linalg.norm(e_root[t0] - (D[t0 - 1] if t0 > 0 else 0)))
                        diag["after_k1"].append(np.linalg.norm(e_root[t0] - total))
                    # replay so far to get the stored object for the next onset
                    pub = replay(w, fk_l - D, fk_r - D, disp, start, gt)
            out = replay(w, fk_l - D, fk_r - D, disp, start, gt)
            res[name]["obj"].append(np.linalg.norm(out - gt, axis=-1).mean() * 100)
            hv = hoi(out, gt, palms - D[:, None, :], palms_gt, gl, gr)
            if len(hv):
                res[name]["hoi"].append(hv.mean() * 100)
            res[name]["root"].append(np.linalg.norm(e_root - D, axis=-1).mean() * 100)
    for name, v in res.items():
        print(f"  {name:22s} obj {np.mean(v['obj']):6.2f}  HOI {np.mean(v['hoi']):6.2f}  root {np.mean(v['root']):6.2f}")
    if diag["e_root"]:
        print(f"  GT onsets: {len(diag['e_root'])}, |root error| at onset {np.mean(diag['e_root'])*100:.2f} -> after k=1 correction {np.mean(diag['after_k1'])*100:.2f} cm")


if __name__ == "__main__":
    main()
