#!/usr/bin/env python3
"""Replay the OT fusion from a dump with alternative hand->object bones.

The fusion in v2/v3 is

    p(t) = wL (hL + R bL) + wR (hR + R bR) + wI (p(t-1) + v/FPS) + wS p(t-1)

with the gating weights ``w`` and the VC velocity ``v`` fixed from the dump, so
only the bones ``bL, bR`` change between variants.  Variants:

  model        the bones the model actually used (sanity check: must reproduce
               ``pred_obj_trans`` up to numerical noise)
  inherit      bone frozen at each (hysteresis) predicted onset from the
               published position: b = R^T (p(t0-1) - h(t0))   (dead reckoning)
  tether{r}    inherited bone plus the model's per-frame deviation from it,
               clipped to a radius of r cm ("anchor-tethered regression")
  nopivot      v3 only: the latched anchor without the pivot residual
  inherit+piv  v3 only: inherited anchor at onset, model pivot residual kept

Reports obj_err (seq-mean, cm) for each variant.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_ot_dumps import HAND_THRESHOLD, Seq, hysteresis, norm  # noqa: E402

FPS = 30.0


def model_bones(s: Seq) -> np.ndarray:
    """[T,2,3] object-frame bones the model used."""
    if "pred_lhand_obj_vector" in s.rec:
        return np.stack((s.rec["pred_lhand_obj_vector"].numpy(), s.rec["pred_rhand_obj_vector"].numpy()), axis=1)
    dl = s.rec["pred_lhand_obj_direction"].numpy() * s.rec["pred_lhand_lb"].numpy()[:, None]
    dr = s.rec["pred_rhand_obj_direction"].numpy() * s.rec["pred_rhand_lb"].numpy()[:, None]
    return np.stack((dl, dr), axis=1)


def replay(s: Seq, bone_fn) -> np.ndarray:
    """bone_fn(t, h, prev, active_flags, onset_flags) -> [2,3] bones for frame t."""
    T = s.T
    v = s.rec["pred_obj_vel"].numpy() if "pred_obj_vel" in s.rec else np.zeros((T, 3))
    w = s.w
    out = np.zeros((T, 3))
    prev = s.p_pred[0].astype(np.float64)
    c = s.c_hyst
    active = np.zeros(2, dtype=bool)
    for t in range(T):
        onset = c[t] & ~active
        active = c[t].copy()
        b = bone_fn(t, prev, onset)
        pL = s.h_pred[t, 0] + s.R_imu[t] @ b[0]
        pR = s.h_pred[t, 1] + s.R_imu[t] @ b[1]
        p = w[t, 0] * pL + w[t, 1] * pR + w[t, 2] * (prev + v[t] / FPS) + w[t, 3] * prev
        out[t] = p
        prev = p
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump")
    parser.add_argument("--radii", type=float, nargs="*", default=[3.0, 5.0, 10.0, 15.0])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    payload = torch.load(args.dump, weights_only=False)
    recs = payload["records"][: args.limit] if args.limit else payload["records"]
    seqs = [Seq(r) for r in recs if r.get("has_object", True)]
    is_v3 = "pred_lhand_segment_anchor" in seqs[0].rec

    results = {"published": [], "model": [], "inherit": []}
    for r in args.radii:
        results[f"tether{r:g}"] = []
    if is_v3:
        results["nopivot"] = []
        results["inherit+piv"] = []
        results["inherit+piv_tether5"] = []

    for s in seqs:
        bm = model_bones(s)
        results["published"].append(s.err.mean())

        def fn_model(t, prev, onset):
            return bm[t]

        # inherited bone latched at onset
        state = {"b": np.zeros((2, 3))}

        def fn_inherit(t, prev, onset, state=state):
            for h in range(2):
                if onset[h]:
                    state["b"][h] = s.R_imu[t].T @ (prev - s.h_pred[t, h])
            return state["b"].copy()

        def make_tether(radius):
            st = {"b": np.zeros((2, 3))}

            def fn(t, prev, onset):
                for h in range(2):
                    if onset[h]:
                        st["b"][h] = s.R_imu[t].T @ (prev - s.h_pred[t, h])
                d = bm[t] - st["b"]
                n = norm(d)[:, None]
                d = d * np.minimum(1.0, radius / 100.0 / np.maximum(n, 1e-9))
                return st["b"] + d
            return fn

        results["model"].append(norm(replay(s, fn_model) - s.p_gt).mean() * 100)
        results["inherit"].append(norm(replay(s, fn_inherit) - s.p_gt).mean() * 100)
        for r in args.radii:
            results[f"tether{r:g}"].append(norm(replay(s, make_tether(r)) - s.p_gt).mean() * 100)
        if is_v3:
            anchor = np.stack((s.rec["pred_lhand_segment_anchor"].numpy(), s.rec["pred_rhand_segment_anchor"].numpy()), axis=1)
            resid = bm - anchor  # = -(M-I) pivot

            def fn_nopivot(t, prev, onset):
                return anchor[t]

            st2 = {"a": np.zeros((2, 3))}

            def fn_inh_piv(t, prev, onset):
                for h in range(2):
                    if onset[h]:
                        st2["a"][h] = s.R_imu[t].T @ (prev - s.h_pred[t, h])
                return st2["a"] + resid[t]

            st3 = {"a": np.zeros((2, 3))}

            def fn_inh_piv_t(t, prev, onset):
                for h in range(2):
                    if onset[h]:
                        st3["a"][h] = s.R_imu[t].T @ (prev - s.h_pred[t, h])
                d = resid[t]
                n = norm(d)[:, None]
                d = d * np.minimum(1.0, 0.05 / np.maximum(n, 1e-9))
                return st3["a"] + d

            results["nopivot"].append(norm(replay(s, fn_nopivot) - s.p_gt).mean() * 100)
            results["inherit+piv"].append(norm(replay(s, fn_inh_piv) - s.p_gt).mean() * 100)
            results["inherit+piv_tether5"].append(norm(replay(s, fn_inh_piv_t) - s.p_gt).mean() * 100)

    print(f"{payload['variant']}/{payload['mode']} n={len(seqs)}")
    for k, v in results.items():
        v = np.array(v)
        print(f"  {k:>22}: obj_err {v.mean():6.2f}  median {np.median(v):6.2f}  >25cm {np.mean(v > 25) * 100:4.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
