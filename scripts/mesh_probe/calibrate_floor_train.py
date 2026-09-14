#!/usr/bin/env python3
"""Calibrate the mesh-bottom floor offset on the canonical TRAIN split (GT only).

For each training sequence: floor_seq = 5% quantile over GT static frames of the
lowest transformed mesh vertex height (canonical data: min foot joint y == 0).
Reports the global median and per-object medians, which are the only constants a
test-time non-penetration correction is allowed to use.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_nn import SurfaceNN, load_mesh  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
TRAIN = os.path.join(ROOT, "process/IMUHOI++cano/processed_split_data_OMOMO_full/train")
OUT = os.path.join(ROOT, "outputs/analysis/mesh_probe_20260907/floor_calibration_train.json")


def main():
    nn = {}
    per_obj = defaultdict(list)
    files = sorted(glob.glob(os.path.join(TRAIN, "*.pt")))
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        if not d.get("has_object", False):
            continue
        name = d["obj_name"]
        if name not in nn:
            nn[name] = SurfaceNN(load_mesh(name), n_samples=1000)
        R = d["obj_rot"].double()
        p = d["obj_trans"].double()
        s = d["obj_scale"].double()
        moving = d["obj_contact"].bool()
        min_y = nn[name].min_y_world(R, s, p).cpu().numpy()
        static = (~moving).numpy()
        ref = min_y[static] if static.sum() > 5 else min_y
        per_obj[name].append(float(np.quantile(ref, 0.05)))
        if i % 500 == 0:
            print(i, "/", len(files), flush=True)
    allv = np.concatenate([np.asarray(v) for v in per_obj.values()])
    rep = {
        "n_sequences": int(allv.size),
        "global_median_cm": float(np.median(allv) * 100),
        "global_q10_q90_cm": [float(np.quantile(allv, 0.1) * 100), float(np.quantile(allv, 0.9) * 100)],
        "per_object_median_cm": {k: float(np.median(v) * 100) for k, v in sorted(per_obj.items())},
        "per_object_n": {k: len(v) for k, v in sorted(per_obj.items())},
    }
    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
