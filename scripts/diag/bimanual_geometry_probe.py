#!/usr/bin/env python3
"""Does the object's geometry make bimanual closure informative?  Training-free publisher-level replay.

On frames where both hands touch a rigid object of known shape, both palms must lie near the surface.
Given the predicted palms and the IMU-measured object rotation, the object position is re-solved per frame

    p* = argmin_p  sum_h (dist_surface(R^T (h_h - p) / s) * s - delta0)^2 + lam * |p - p_published|^2

(Gauss-Newton with nearest-surface-point normals).  Reported on GT bimanual frames: pair-HOI before/after,
with predicted palms (realisable) and with GT palms (information content of the constraint = object error).
The same solve with one hand is reported on GT single-hand frames for comparison.

    python scripts/diag/bimanual_geometry_probe.py dump.pt [--limit N]
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "mesh_probe"))
from mesh_nn import SurfaceNN, load_mesh  # noqa: E402

DATA = os.path.join(ROOT, "process/IMUHOI++cano/processed_split_data_OMOMO_full/test")


def rot6d_to_matrix(x):
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True).clip(1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True).clip(1e-8)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def surface(nn, hands, R, s, p):
    """hands [n,H,3] world -> unsigned distance (m) [n,H] and world unit normal surface->hand [n,H,3]."""
    x = np.einsum("nji,nhj->nhi", R, hands - p[:, None]) / s[:, None, None]
    y, dist = nn.query(torch.from_numpy(x.reshape(-1, 3)))
    y = y.cpu().numpy().astype(np.float64).reshape(x.shape)
    diff = x - y
    norm = np.linalg.norm(diff, axis=-1, keepdims=True).clip(1e-6)
    normal_w = np.einsum("nij,nhj->nhi", R, diff / norm)
    return dist.cpu().numpy().astype(np.float64).reshape(x.shape[:2]) * s[:, None], normal_w


def solve(nn, hands, R, s, p0, delta0, lam, iters=5):
    p = p0.copy()
    eye = np.eye(3)[None]
    for _ in range(iters):
        r, nw = surface(nn, hands, R, s, p)
        A = np.einsum("nhi,nhj->nij", nw, nw) + lam * eye
        # moving the object by dp changes the hand's distance by -n.dp ; target r - n.dp = delta0.
        # Sign: the normal points surface->hand, so moving the object along +n brings the surface closer.
        b = np.einsum("nhi,nh->ni", nw, r - delta0) + lam * (p0 - p)
        p = p + np.linalg.solve(A, b[..., None])[..., 0]
    return p


def pair_hoi(p, h, rel_gt):
    return 50.0 * (np.linalg.norm((p - h[:, 0]) - rel_gt[:, 0], axis=-1) + np.linalg.norm((p - h[:, 1]) - rel_gt[:, 1], axis=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delta0", type=float, default=None, help="target palm-surface distance (m); default: GT median on this split")
    args = ap.parse_args()
    records = torch.load(args.dump, map_location="cpu")["records"]
    if args.limit:
        records = records[: args.limit]
    nns, gt_dist = {}, []
    prepared = []
    for r in records:
        n = int(r["publish_len"])
        if not r.get("has_object", True) or n == 0:
            continue
        name = r["obj_name"]
        if name not in nns:
            try:
                nns[name] = SurfaceNN(load_mesh(name), n_samples=40000)
            except Exception as exc:  # noqa: BLE001
                print(f"no mesh for {name}: {exc}")
                nns[name] = None
        if nns[name] is None:
            continue
        seq = torch.load(os.path.join(DATA, r["seq_file"]), map_location="cpu")
        s = np.asarray(seq["obj_scale"][:n], dtype=np.float64)
        s = np.where(s > 1e-3, s, np.median(s[s > 1e-3]) if (s > 1e-3).any() else 1.0)  # a few frames store scale 0
        R = rot6d_to_matrix(np.asarray(r["gt_obj_rot"][:n], dtype=np.float64))
        item = {"nn": nns[name], "s": s, "R": R, "p_gt": np.asarray(r["gt_obj_trans"][:n], dtype=np.float64),
                "g": np.asarray(r.get("gt_palm_pos_eval", r["gt_palm_pos"])[:n], dtype=np.float64),
                "h": np.asarray(r["pred_palm_glb_pos"][:n], dtype=np.float64),
                "pub": np.asarray(r["pred_obj_trans_published"][:n], dtype=np.float64),
                "cl": np.asarray(r["gt_lhand_contact"][:n]).astype(bool), "cr": np.asarray(r["gt_rhand_contact"][:n]).astype(bool)}
        d_gt, _ = surface(item["nn"], item["g"], R, s, item["p_gt"])
        gt_dist.append(np.concatenate((d_gt[item["cl"], 0], d_gt[item["cr"], 1])))
        prepared.append(item)
    gd = np.concatenate(gt_dist) * 100.0
    print(f"sequences {len(prepared)} | GT palm-to-surface distance on GT contact hand-frames: median {np.median(gd):.2f} cm  mean {gd.mean():.2f}  p75 {np.percentile(gd, 75):.2f}  p90 {np.percentile(gd, 90):.2f}")
    delta0 = args.delta0 if args.delta0 is not None else float(np.median(gd)) / 100.0
    print(f"delta0 = {delta0 * 100:.2f} cm")
    for lam in (0.25, 1.0, 4.0):
        acc = {k: [] for k in ("bi_before", "bi_after", "bi_obj_before", "bi_obj_after", "bi_gt_hands_obj", "si_before", "si_after")}
        for it in prepared:
            both = it["cl"] & it["cr"]
            rel_gt = it["p_gt"][:, None] - it["g"]
            if both.any():
                idx = np.where(both)[0]
                p_new = solve(it["nn"], it["h"][idx], it["R"][idx], it["s"][idx], it["pub"][idx], delta0, lam)
                acc["bi_before"].append(pair_hoi(it["pub"][idx], it["h"][idx], rel_gt[idx]))
                acc["bi_after"].append(pair_hoi(p_new, it["h"][idx], rel_gt[idx]))
                acc["bi_obj_before"].append(np.linalg.norm(it["pub"][idx] - it["p_gt"][idx], axis=-1) * 100.0)
                acc["bi_obj_after"].append(np.linalg.norm(p_new - it["p_gt"][idx], axis=-1) * 100.0)
                p_or = solve(it["nn"], it["g"][idx], it["R"][idx], it["s"][idx], it["pub"][idx], delta0, lam)
                acc["bi_gt_hands_obj"].append(np.linalg.norm(p_or - it["p_gt"][idx], axis=-1) * 100.0)
            for hand, mask in ((0, it["cl"] & ~it["cr"]), (1, it["cr"] & ~it["cl"])):
                if mask.any():
                    idx = np.where(mask)[0]
                    hh = it["h"][idx][:, hand:hand + 1]
                    p_new = solve(it["nn"], hh, it["R"][idx], it["s"][idx], it["pub"][idx], delta0, lam)
                    q0 = (it["pub"][idx] - hh[:, 0]) - rel_gt[idx, hand]
                    q1 = (p_new - hh[:, 0]) - rel_gt[idx, hand]
                    acc["si_before"].append(np.linalg.norm(q0, axis=-1) * 100.0)
                    acc["si_after"].append(np.linalg.norm(q1, axis=-1) * 100.0)
        m = {k: float(np.concatenate(v).mean()) for k, v in acc.items() if v}
        print(f"lam {lam:4.2f} | bimanual frames: pair-HOI {m['bi_before']:.2f} -> {m['bi_after']:.2f}   obj err {m['bi_obj_before']:.2f} -> {m['bi_obj_after']:.2f}"
              f"   [GT palms: obj err -> {m['bi_gt_hands_obj']:.2f}] | single-hand frames: HOI {m['si_before']:.2f} -> {m['si_after']:.2f}")


if __name__ == "__main__":
    main()
