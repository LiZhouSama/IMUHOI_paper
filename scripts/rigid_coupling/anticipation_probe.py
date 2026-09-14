#!/usr/bin/env python3
"""Can the human's approach motion predict the grasp offset at contact onset?

For every GT contact onset (hand k, first frame t0 of a segment), features computed from the
human/IMU stream only (no object position): approach velocity of the palm over the last 10
frames expressed in the object frame, forearm-vs-object relative rotation at t0 (6D), object
orientation (6D, gravity-aligned canonical world), the other hand's state, palm height.
Target: palm-anchored grasp offset b(t0) = R^T (p - palm) (metres, object frame).
A small MLP is trained on OMOMO-train onsets (GT hands) and tested on OMOMO-test onsets with
GT hands and with the pipeline's predicted hands (from the online dump).  Compared against the
population prior and the current ObjectTrans bone error at the same onset frames.
"""
import argparse, glob, os, sys
import numpy as np, torch, torch.nn as nn
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from scripts.rigid_coupling.lever_arm_probe import world_from_features, segments  # noqa: E402
from utils.human_pose import PALM_EXTENSION_RATIOS  # noqa: E402
from pytorch3d.transforms import matrix_to_rotation_6d  # noqa: E402
W, E = (20, 21), (18, 19); K = 10

def f64(t): return t.numpy().astype(np.float64)
def norm(x): return np.linalg.norm(x, axis=-1)
def r6(Rm): return matrix_to_rotation_6d(torch.as_tensor(Rm)).numpy()

def onset_samples(x, palm, R_fore, p, R, hand_src="gt"):
    """features / targets / onset frames for both hands of one sequence."""
    T = p.shape[0]; out = []
    L = x["lhand_contact"].numpy().astype(bool)[:T]; Rr = x["rhand_contact"].numpy().astype(bool)[:T]
    cs = (L, Rr)
    for k in range(2):
        for a, b in segments(cs[k]):
            if a < K or b - a + 1 < 5: continue
            t0 = a; Rt = R[t0]
            v = (palm[t0, k] - palm[t0 - K, k]) / (K / 30.0)                # m/s world
            v_o = Rt.T @ v; sp = np.linalg.norm(v)
            d_o = v_o / (sp + 1e-6)
            rel = r6(Rt.T @ R_fore[t0, k])                                   # forearm in object frame
            obj6 = r6(Rt)
            other = cs[1 - k][t0]
            oth_o = Rt.T @ (palm[t0, 1 - k] - palm[t0, k])
            # approach curvature / deceleration: velocity 10 frames earlier vs now
            v_prev = (palm[t0 - K // 2, k] - palm[t0 - K, k]) / (K / 2 / 30.0)
            feat = np.concatenate([v_o, d_o, [sp], rel, obj6, [float(other)], oth_o, [palm[t0, k, 1]], Rt.T @ v_prev, [k]])
            tgt = Rt.T @ (p[t0] - palm[t0, k])
            tgt30 = np.mean([R[t].T @ (p[t] - palm[t, k]) for t in range(t0, min(b + 1, t0 + 30))], 0)
            out.append((feat, tgt, tgt30, t0, k))
    return out

def load_split(files, dump=None, hand_src="gt"):
    F, Y, Y30, meta = [], [], [], []
    for f in files:
        x = torch.load(f, map_location="cpu", weights_only=False)
        if not x.get("has_object", True): continue
        p = x["obj_trans"].double().numpy(); R = x["obj_rot"].double().numpy(); J = x["position_global_full_gt_world"].double().numpy()
        w, e = J[:, list(W)], J[:, list(E)]; palm = w + (w - e) * np.asarray(PALM_EXTENSION_RATIOS)[None, :, None]
        himu = x["human_imu"].double().numpy(); oimu = x["obj_imu"].double().numpy()
        _, R_fore, _, _ = world_from_features(himu, oimu)                   # forearm orientation from the IMU stream
        if hand_src == "pred":
            rec = dump.get(os.path.basename(f))
            if rec is None: continue
            Tp = rec["publish_len"]; palm = f64(rec["ot_hand_positions"]); p, R, R_fore = p[:Tp], R[:Tp], R_fore[:Tp]
        for feat, tgt, tgt30, t0, k in onset_samples(x, palm, R_fore, p, R):
            F.append(feat); Y.append(tgt); Y30.append(tgt30); meta.append((os.path.basename(f), t0, k))
    return np.array(F), np.array(Y), np.array(Y30), meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="process/IMUHOI++cano/processed_split_data_OMOMO_full/train")
    ap.add_argument("--test", default="process/IMUHOI++cano/processed_split_data_OMOMO_full/test")
    ap.add_argument("--dump", default="outputs/IMUHOI++/RNN_cano/omomo_cano_ot_variants_20260909_c/dumps/cur_hoi_online.pt")
    a = ap.parse_args()
    Ftr, Ytr, Ytr30, _ = load_split(sorted(glob.glob(os.path.join(a.train, "*.pt"))))
    Fte, Yte, Yte30, meta = load_split(sorted(glob.glob(os.path.join(a.test, "*.pt"))))
    dump = {r["seq_file"]: r for r in torch.load(a.dump, map_location="cpu", weights_only=False)["records"]}
    Fpr, Ypr, _, meta_pr = load_split(sorted(glob.glob(os.path.join(a.test, "*.pt"))), dump, "pred")
    print(f"onsets: train {len(Ftr)}  test {len(Fte)}  test(pred hands) {len(Fpr)} | feature dim {Ftr.shape[1]}")
    def fit(X, Y, seed=0):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        torch.manual_seed(seed); Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
        net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 3))
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        for ep in range(400):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 256):
                idx = perm[i:i + 256]; loss = (net(Xt[idx]) - Yt[idx]).norm(dim=-1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        return lambda X_: net(torch.tensor((X_ - mu) / sd, dtype=torch.float32)).detach().numpy()
    pred = fit(Ftr, Ytr)
    pop = Ytr.mean(0)
    print(f"population prior            : test err {norm(Yte - pop).mean()*100:.2f} cm")
    print(f"MLP(approach+rel-rot), GT hands  : test err {norm(Yte - pred(Fte)).mean()*100:.2f} cm (vs 30-frame mean target {norm(Yte30 - pred(Fte)).mean()*100:.2f})")
    print(f"MLP(approach+rel-rot), PRED hands: test err {norm(Ypr - pred(Fpr)).mean()*100:.2f} cm")
    # ablations: drop relative rotation / drop approach velocity
    keep_rel = np.ones(Ftr.shape[1], bool); keep_rel[7:13] = False
    pr2 = fit(Ftr[:, keep_rel], Ytr); print(f"  ablation without forearm-object relative rotation: {norm(Yte - pr2(Fte[:, keep_rel])).mean()*100:.2f}")
    keep_v = np.ones(Ftr.shape[1], bool); keep_v[0:7] = False; keep_v[-4:-1] = False
    pr3 = fit(Ftr[:, keep_v], Ytr); print(f"  ablation without approach velocity            : {norm(Yte - pr3(Fte[:, keep_v])).mean()*100:.2f}")
    # current ObjectTrans bone error at the same onset frames (pred hands)
    errs = []
    for (sf, t0, k), y in zip(meta_pr, Ypr):
        r = dump[sf]; fk = f64(r["pred_lhand_obj_trans" if k == 0 else "pred_rhand_obj_trans"])[t0]; h = f64(r["ot_hand_positions"])[t0, k]
        p_gt = f64(r["gt_obj_trans"])[t0]; palm_gt = f64(r["gt_palm_pos"])[t0, k]
        errs.append(np.linalg.norm((fk - h) - (p_gt - palm_gt)))
    print(f"current ObjectTrans bone at the same onsets (pred hands): {np.mean(errs)*100:.2f} cm")
    # bimanual disagreement of the two FK proposals
    dis = []
    for r in dump.values():
        c = np.stack((r["gt_lhand_contact"].numpy(), r["gt_rhand_contact"].numpy()), 1).astype(bool); bi = c[:, 0] & c[:, 1]
        if bi.any(): dis.append(norm(f64(r["pred_lhand_obj_trans"])[bi] - f64(r["pred_rhand_obj_trans"])[bi]))
    print(f"bimanual frames: |P_L - P_R| of the two FK proposals: mean {np.concatenate(dis).mean()*100:.2f} cm, median {np.median(np.concatenate(dis))*100:.2f}")

if __name__ == "__main__": main()
