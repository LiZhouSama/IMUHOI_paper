"""Summarize the per-joint rotation errors of scripts/run_per_joint_mpjre_20261025.py.

Writes outputs/Comparisons/per_joint_mpjre_20261025/summary.md with
  1. MPJRE per dataset (canonical axis-angle; the old pytorch3d value of the
     main rows for reference) and the SO(3) geodesic error,
  2. per-joint errors averaged over the five datasets,
  3. joint groups and their share of the MPJRE gap to the best baseline.
It also checks that the re-run reproduces the published MPJPE / object / HOI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "outputs/Comparisons/per_joint_mpjre_20261025"
MAIN = ROOT / "outputs/IMUHOI++/RNN_cano/all5_v5ar_vs_imuhoi3b_20261012/metrics.json"
DATASETS = ("omomo", "behave", "imhd", "pahoi", "hodome")
NAMES = {"omomo": "OMOMO", "behave": "BEHAVE", "imhd": "IMHD", "pahoi": "PAHOI", "hodome": "HODome"}
METHODS = {
    "imuhoi++": "IMUHOI++ (ours)", "imuhoi3b": "IMUHOI (3-branch)", "dip18": "DIP",
    "tip": "TIP", "transpose": "TransPose", "globalpose": "GlobalPose",
}
BASELINES = ("dip18", "tip", "transpose", "globalpose")
JOINTS = (
    "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2", "L_ankle", "R_ankle", "spine3",
    "L_foot", "R_foot", "neck", "L_collar", "R_collar", "head", "L_shoulder", "R_shoulder",
    "L_elbow", "R_elbow", "L_wrist", "R_wrist",
)
# Local joint index (1..21) groups.
GROUPS = {
    "hips+knees": (1, 2, 4, 5),
    "spine+neck+head": (3, 6, 9, 12, 15),
    "collar+shoulder": (13, 14, 16, 17),
    "elbows": (18, 19),
    "ankles": (7, 8),
    "feet+wrists": (10, 11, 20, 21),
}


def load(dataset: str, method: str) -> dict | None:
    path = DIR / dataset / f"{method}.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if method in ("imuhoi++", "imuhoi3b"):
        result = report["results"][dataset]
        per_joint = result["per_joint_rot"]
        return {
            "axis": np.array(per_joint["mpjre_axis"]), "geo": np.array(per_joint["geodesic_deg"]),
            "axis_old": np.array(per_joint["mpjre_axis_pytorch3d"]), "mpjre": result["mpjre_angle"],
            "check": {"mpjpe": result["mpjpe"], "obj": result.get("obj_trans_err_fusion_refine"),
                      "hoi": result.get("hoi_err_fusion")},
        }
    per_joint = report["per_joint_rot"]
    return {
        "axis": np.array(per_joint["human_ang_err_deg"]), "geo": np.array(per_joint["human_geodesic_err_deg"]),
        "mpjre": report["metrics"]["human_ang_err_deg"],
        "check": {"mpjpe": report["metrics"]["human_pos_err_cm"]},
    }


def published() -> dict:
    out = {}
    for entry in json.loads(MAIN.read_text()):
        if entry["checkpoint"] == "best":
            result = json.loads((ROOT / entry["eval_json"]).read_text())["results"][entry["dataset"]]
            out[(entry["dataset"], entry["method"])] = {
                "mpjpe": entry["mpjpe"], "obj": entry["obj_trans_err_fusion_refine"],
                "hoi": entry["hoi_err_fusion"], "mpjre": result["mpjre_angle"],
            }
    return out


def main() -> None:
    data = {(d, m): load(d, m) for d in DATASETS for m in METHODS}
    missing = [f"{d}/{m}" for (d, m), v in data.items() if v is None]
    pub = published()
    lines = ["# Per-joint MPJRE (online, five datasets)", ""]
    if missing:
        lines += [f"Pending: {', '.join(missing)}", ""]

    # Reproduction check of the main rows.
    lines += ["## Reproduction of the published main rows", "",
              "| Dataset | Method | MPJPE pub/new | Obj pub/new | HOI pub/new | MPJRE pub / new-legacy |",
              "|---|---|---|---|---|---|"]
    for (d, m), v in data.items():
        if v is None or m not in ("imuhoi++", "imuhoi3b"):
            continue
        p, c = pub[(d, m)], v["check"]
        lines.append(f"| {NAMES[d]} | {METHODS[m]} | {p['mpjpe']:.3f}/{c['mpjpe']:.3f} | {p['obj']:.2f}/{c['obj']:.2f} | "
                     f"{p['hoi']:.2f}/{c['hoi']:.2f} | {p['mpjre']:.3f} / {v['axis_old'].mean():.3f} |")
    lines.append("")

    # Dataset-level table.
    lines += ["## MPJRE (axis difference, canonical) / geodesic, deg", "",
              "| Method | " + " | ".join(NAMES[d] for d in DATASETS) + " | Mean |",
              "|---|" + "---|" * (len(DATASETS) + 1)]
    for m, label in METHODS.items():
        cells, axis, geo = [], [], []
        for d in DATASETS:
            v = data[(d, m)]
            if v is None:
                cells.append("–")
                continue
            axis.append(v["axis"].mean())
            geo.append(v["geo"].mean())
            old = f" (old {v['axis_old'].mean():.2f})" if "axis_old" in v else ""
            cells.append(f"{v['axis'].mean():.2f}{old} / {v['geo'].mean():.2f}")
        mean = f"{np.mean(axis):.2f} / {np.mean(geo):.2f}" if len(axis) == len(DATASETS) else "–"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {mean} |")
    lines.append("")

    complete = [m for m in METHODS if all(data[(d, m)] is not None for d in DATASETS)]
    if not complete:
        (DIR / "summary.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    mean_axis = {m: np.mean([data[(d, m)]["axis"] for d in DATASETS], axis=0) for m in complete}
    mean_geo = {m: np.mean([data[(d, m)]["geo"] for d in DATASETS], axis=0) for m in complete}

    lines += ["## Per joint, mean over the five datasets (axis difference / geodesic, deg)", "",
              "| # | Joint | " + " | ".join(METHODS[m] for m in complete) + " |",
              "|---|---|" + "---|" * len(complete)]
    for j, name in enumerate(JOINTS):
        cells = [f"{mean_axis[m][j]:.2f} / {mean_geo[m][j]:.2f}" for m in complete]
        lines.append(f"| {j + 1} | {name} | " + " | ".join(cells) + " |")
    lines.append("")

    # Group contributions to the MPJRE mean (= sum over joints / 21).
    lines += ["## Joint groups: contribution to MPJRE (sum of joint errors / 21, deg)", "",
              "| Group | Joints | " + " | ".join(METHODS[m] for m in complete) + " |",
              "|---|---|" + "---|" * len(complete)]
    for group, joints in GROUPS.items():
        idx = [j - 1 for j in joints]
        cells = [f"{mean_axis[m][idx].sum() / 21:.2f}" for m in complete]
        lines.append(f"| {group} | {', '.join(JOINTS[i] for i in idx)} | " + " | ".join(cells) + " |")
    lines.append("| total | 21 | " + " | ".join(f"{mean_axis[m].mean():.2f}" for m in complete) + " |")
    lines.append("")

    (DIR / "summary.md").write_text("\n".join(lines) + "\n")
    (DIR / "per_joint.json").write_text(json.dumps(
        {f"{d}/{m}": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data[(d, m)].items()}
         for d in DATASETS for m in METHODS if data[(d, m)] is not None}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
