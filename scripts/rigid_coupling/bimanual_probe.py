#!/usr/bin/env python3
"""Bimanual structure of the HOI error (evidence for a hand-pair / closure formulation).

P1  bone / HOI error of the current model on bimanual vs single-hand GT-contact frames.
P2  pair-frame oracle: on bimanual frames publish p = midpoint(h_L,h_R) + R c_obj with c_obj the
    per-object TRAIN mean of R^T (p - midpoint) -> how much of the per-hand grasp ambiguity is
    common-mode (resolved by the pair frame) vs differential (needs the prior)?
"""
import argparse, glob, os, sys
from collections import defaultdict
import numpy as np, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.analyze_ot_dumps import rot6d  # noqa: E402
from utils.human_pose import PALM_EXTENSION_RATIOS  # noqa: E402
W, E = (20, 21), (18, 19)

def f64(t): return t.numpy().astype(np.float64)
def norm(x): return np.linalg.norm(x, axis=-1)

def palms_of(x):
    J = x["position_global_full_gt_world"].double().numpy(); w, e = J[:, list(W)], J[:, list(E)]
    return w + (w - e) * np.asarray(PALM_EXTENSION_RATIOS)[None, :, None]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train", default="process/IMUHOI++cano/processed_split_data_OMOMO_full/train")
    ap.add_argument("--dump", default="outputs/IMUHOI++/RNN_cano/omomo_cano_ot_variants_20260909_c/dumps/cur_hoi_online.pt")
    ap.add_argument("--test", default="process/IMUHOI++cano/processed_split_data_OMOMO_full/test"); a = ap.parse_args()
    # per-object train statistics: pair-frame common mode c and per-hand offsets b
    c_obj, b_obj = defaultdict(list), defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.train, "*.pt")))[::2]:
        x = torch.load(f, map_location="cpu", weights_only=False); p = x["obj_trans"].double().numpy(); R = x["obj_rot"].double().numpy()
        palm = palms_of(x); L = x["lhand_contact"].numpy().astype(bool); Rr = x["rhand_contact"].numpy().astype(bool); bi = L & Rr
        if bi.any():
            mid = palm[bi].mean(1); c_obj[x["obj_name"]].append(np.einsum("tji,tj->ti", R[bi], p[bi] - mid)[::3])
            for k, m in enumerate((L, Rr)):
                b_obj[(x["obj_name"], k)].append(np.einsum("tji,tj->ti", R[bi], p[bi] - palm[bi, k])[::3])
    c_mean = {o: np.concatenate(v).mean(0) for o, v in c_obj.items()}
    c_std = np.mean([norm(np.concatenate(v) - np.concatenate(v).mean(0)).mean() for v in c_obj.values()])
    b_std = np.mean([norm(np.concatenate(v) - np.concatenate(v).mean(0)).mean() for v in b_obj.values()])
    print(f"train (bimanual frames): mean spread of pair-frame common mode c: {c_std*100:.1f} cm | per-hand offset b: {b_std*100:.1f} cm")
    recs = {r["seq_file"]: r for r in torch.load(a.dump, map_location="cpu", weights_only=False)["records"]}
    stats = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.test, "*.pt"))):
        r = recs.get(os.path.basename(f))
        if r is None: continue
        x = torch.load(f, map_location="cpu", weights_only=False); T = r["publish_len"]
        p_gt = f64(r["gt_obj_trans"]); palm_gt = f64(r["gt_palm_pos"]); h = f64(r["ot_hand_positions"]); pub = f64(r["pred_obj_trans_published"]); palm_pred = f64(r["pred_palm_glb_pos"])
        R = x["obj_rot"].double().numpy()[:T]; fk = np.stack((f64(r["pred_lhand_obj_trans"]), f64(r["pred_rhand_obj_trans"])), 1)
        c = np.stack((r["gt_lhand_contact"].numpy(), r["gt_rhand_contact"].numpy()), 1).astype(bool); bi = c[:, 0] & c[:, 1]
        cm = c_mean.get(x["obj_name"])
        for k in range(2):
            m = c[:, k]
            if not m.any(): continue
            bone = norm((fk[:, k] - h[:, k]) - (p_gt - palm_gt[:, k])); hoi = norm((pub - palm_pred[:, k]) - (p_gt - palm_gt[:, k]))
            stats["bone_bi"].append(bone[m & bi]); stats["bone_uni"].append(bone[m & ~bi]); stats["hoi_bi"].append(hoi[m & bi]); stats["hoi_uni"].append(hoi[m & ~bi])
            if cm is not None and bi.any():
                mid = h[bi].mean(1); p_pair = mid + np.einsum("tij,j->ti", R[bi], cm)
                stats["hoi_pair_predhands"].append(norm((p_pair - palm_pred[bi, k]) - (p_gt[bi] - palm_gt[bi, k])))
                mid_gt = palm_gt[bi].mean(1); p_pair_gt = mid_gt + np.einsum("tij,j->ti", R[bi], cm)
                stats["hoi_pair_gthands"].append(norm((p_pair_gt - palm_gt[bi, k]) - (p_gt[bi] - palm_gt[bi, k])))
                stats["hoi_model_on_same"].append(hoi[bi])
    S = {k: np.concatenate(v) for k, v in stats.items()}
    print(f"P1 model bone err: bimanual {S['bone_bi'].mean()*100:.2f} (n={len(S['bone_bi'])}) | single-hand {S['bone_uni'].mean()*100:.2f} (n={len(S['bone_uni'])})")
    print(f"P1 model HOI    : bimanual {S['hoi_bi'].mean()*100:.2f} | single-hand {S['hoi_uni'].mean()*100:.2f}")
    print(f"P2 bimanual frames, HOI: model {S['hoi_model_on_same'].mean()*100:.2f} | pair-frame + per-object mean c, pred hands {S['hoi_pair_predhands'].mean()*100:.2f} | same with GT hands {S['hoi_pair_gthands'].mean()*100:.2f}")

if __name__ == "__main__": main()
