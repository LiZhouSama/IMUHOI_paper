"""Merge the protocol-v3 baselines into the five-dataset IMUHOI++ vs IMUHOI table.

IMUHOI++ / IMUHOI rows come from all5_v5ar_vs_imuhoi3b_20261012 (``best``
checkpoints, metrics re-read from each row's eval JSON).  Baseline rows come
from Comparisons/eval_comparison.py: OMOMO from omomo_cano_20261023, the other
datasets from all4_cano_20261023/<dataset>.  Both evaluators use the same
definitions (macro mean over full test sequences; MPJPE root-relative on 22
joints; jitter on global joints; contact-palm HOI on GT contact frames;
object anchored at GT frame 0), and the same Noitom noise realization (seed 42).

Usage: python scripts/summarize_comparisons_all5_20261023.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "outputs/IMUHOI++/RNN_cano/all5_v5ar_vs_imuhoi3b_20261012/metrics.json"
# Re-evaluation of the same IMUHOI++ / IMUHOI checkpoints after the 2026-10-25
# MPJRE fix (canonical axis-angle; pytorch3d 0.7.3 returned angles > pi).
RECHECK = ROOT / "outputs/Comparisons/per_joint_mpjre_20261025"
# Four-branch IMUHOI (ObjectTrans imuhoi_4heads): same HP / VC checkpoints,
# recipe and schedule as the three-branch rows, only OT + joint phase retrained
# (scripts/run_imuhoi4h_all5_20261026.sh).
IMUHOI4H = ROOT / "outputs/IMUHOI++/RNN_cano/imuhoi4h_all5_20261026"
OUT = ROOT / "outputs/Comparisons/all5_baselines_vs_imuhoi_20261023"
DATASETS = ("omomo", "behave", "imhd", "pahoi", "hodome")
NAMES = {"omomo": "OMOMO", "behave": "BEHAVE", "imhd": "IMHD", "pahoi": "PAHOI", "hodome": "HODome"}
BASELINES = (
    ("dip18", "_gtroot", "DIP (GT root trans.)"),
    ("tip", "", "TIP"),
    ("transpose", "", "TransPose"),
    ("globalpose", "", "GlobalPose"),
)
COLUMNS = (
    ("obj", "Object (cm)", 2), ("hoi", "HOI (cm)", 2), ("mpjpe", "MPJPE (cm)", 2),
    ("mpjre", "MPJRE (deg)", 2), ("geodesic", "Geodesic (deg)", 2), ("jitter", "Jitter (mm/f²)", 2), ("root", "Root (cm)", 2),
    ("obj_jitter", "Obj jitter (mm/f²)", 2), ("f1_l", "L F1", 3), ("f1_r", "R F1", 3),
    ("f1_obj", "Obj F1", 3),
)
PROTOCOL_KEYS = (
    "hoi_metric_protocol", "hoi_hand_anchor", "hoi_aggregation", "inference_mode", "resolved_online_window",
    "resolved_online_lookahead", "online_batch_size", "human_pose_smoothing_half_width",
    "online_obj_vel_init_diff_frames", "contact_init_frame", "velocity_contact_arch", "seed",
    "simulate_imu_noise", "simulate_imu_noise_seed",
)
EXPECTED_SEQUENCES = {"omomo": 1173, "behave": 59, "imhd": 15, "pahoi": 113, "hodome": 36}


def baseline_json(dataset: str, method: str, suffix: str) -> Path:
    root = (
        ROOT / "outputs/Comparisons/omomo_cano_20261023"
        if dataset == "omomo"
        else ROOT / "outputs/Comparisons/all4_cano_20261023" / dataset
    )
    return root / method / f"eval_{dataset}_test_online{suffix}.json"


def main_rows() -> list[dict]:
    rows = []
    for entry in json.loads(MAIN.read_text()):
        if entry["checkpoint"] != "best":
            continue
        result = json.loads((ROOT / entry["eval_json"]).read_text())["results"][entry["dataset"]]
        recheck_path = RECHECK / entry["dataset"] / f"{entry['method']}.json"
        recheck = json.loads(recheck_path.read_text())
        if recheck.get("mpjre_axis_angle_conversion") != "canonical_float64":
            raise ValueError(f"{recheck_path} predates the MPJRE fix")
        recheck = recheck["results"][entry["dataset"]]
        # Same checkpoints and settings: everything but MPJRE must reproduce.
        for key in ("mpjpe", "obj_trans_err_fusion_refine", "hoi_err_fusion", "root_trans_err"):
            if not math.isclose(recheck[key], result[key], rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(f"{recheck_path}: {key} {recheck[key]} != {result[key]}")
        geodesic = recheck["per_joint_rot"]["geodesic_deg"]
        rows.append({
            "dataset": entry["dataset"],
            "method": "IMUHOI++ (ours)" if entry["method"] == "imuhoi++" else "IMUHOI (3-branch)",
            "seq": entry["num_full_sequences"],
            "obj": entry["obj_trans_err_fusion_refine"], "hoi": entry["hoi_err_fusion"],
            "mpjpe": entry["mpjpe"], "mpjre": recheck["mpjre_angle"],
            "geodesic": sum(geodesic) / len(geodesic), "jitter": result["jitter"],
            "root": entry["root_trans_err"], "obj_jitter": entry["obj_jitter"],
            "f1_l": entry["contact_f1_lhand"], "f1_r": entry["contact_f1_rhand"],
            "f1_obj": entry["contact_f1_obj"], "source": entry["eval_json"],
            "mpjre_source": str(recheck_path.relative_to(ROOT)),
        })
    return rows


def imuhoi4h_row(dataset: str) -> dict | None:
    manifest_path = IMUHOI4H / dataset / "experiment_manifest.json"
    if not manifest_path.is_file():
        return None
    pipeline = json.loads(manifest_path.read_text())["pipelines"].get("vc_imuhoi_ot_4heads", {})
    path = IMUHOI4H / dataset / "vc_imuhoi_ot_4heads" / f"eval_{dataset}_best.json"
    if pipeline.get("status") != "complete" or not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get("mpjre_axis_angle_conversion") != "canonical_float64":
        raise ValueError(f"{path} predates the MPJRE fix")
    # Same evaluation protocol as the three-branch re-evaluation of this dataset.
    reference = json.loads((RECHECK / dataset / "imuhoi3b.json").read_text())
    for key in PROTOCOL_KEYS:
        if report.get(key) != reference.get(key):
            raise ValueError(f"{path}: {key} {report.get(key)!r} != {reference.get(key)!r}")
    result = report["results"][dataset]
    geodesic = result["per_joint_rot"]["geodesic_deg"]
    return {
        "dataset": dataset, "method": "IMUHOI (4-branch)", "seq": EXPECTED_SEQUENCES[dataset],
        "obj": result["obj_trans_err_fusion_refine"], "hoi": result["hoi_err_fusion"],
        "mpjpe": result["mpjpe"], "mpjre": result["mpjre_angle"],
        "geodesic": sum(geodesic) / len(geodesic), "jitter": result["jitter"],
        "root": result["root_trans_err"], "obj_jitter": result["obj_jitter"],
        "f1_l": result["contact_f1_lhand"], "f1_r": result["contact_f1_rhand"],
        "f1_obj": result["contact_f1_obj"], "source": str(path.relative_to(ROOT)),
    }


def baseline_row(dataset: str, method: str, suffix: str, label: str) -> dict | None:
    path = baseline_json(dataset, method, suffix)
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report.get("protocol_version") != 3 or report.get("simulate_imu_noise_seed") != 42:
        raise ValueError(f"{path} is not a protocol-v3 / noise-seed-42 evaluation")
    if report["num_sequences"] != EXPECTED_SEQUENCES[dataset]:
        raise ValueError(f"{path}: {report['num_sequences']} sequences")
    metrics = report["metrics"]
    gt_root = bool(report.get("root_translation_oracle"))
    return {
        "dataset": dataset, "method": label, "seq": report["num_sequences"],
        "obj": metrics["obj_err_cm"], "hoi": metrics["hoi_err_cm"],
        "mpjpe": metrics["human_pos_err_cm"], "mpjre": metrics["human_ang_err_deg"],
        "geodesic": metrics["human_geodesic_err_deg"],
        "jitter": metrics["human_jitter_mm_frame2"],
        "root": None if gt_root else metrics["root_trans_err_cm"],
        "obj_jitter": metrics["obj_trans_jitter_mm_frame2"],
        "f1_l": None, "f1_r": None, "f1_obj": None,
        "epoch": report.get("checkpoint_epoch"), "source": str(path.relative_to(ROOT)),
        "gt_root": gt_root,
    }


def fmt(row: dict, key: str, digits: int) -> str:
    value = row.get(key)
    if value is None:
        return "GT" if key == "root" and row.get("gt_root") else "–"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def table(rows: list[dict], bold_best: bool = True) -> list[str]:
    head = "| Dataset | Method | " + " | ".join(name for _, name, _ in COLUMNS) + " |"
    lines = [head, "|" + "---|" * (len(COLUMNS) + 2)]
    by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset, group in by_dataset.items():
        best = {}
        for key, _, _ in COLUMNS:
            values = [r[key] for r in group if isinstance(r.get(key), float)]
            if values:
                best[key] = max(values) if key.startswith("f1") else min(values)
        for row in group:
            cells = []
            for key, _, digits in COLUMNS:
                text = fmt(row, key, digits)
                if bold_best and isinstance(row.get(key), float) and row[key] == best.get(key):
                    text = f"**{text}**"
                cells.append(text)
            name = NAMES.get(dataset, dataset)
            lines.append(f"| {name} | {row['method']} | " + " | ".join(cells) + " |")
    return lines


def mean_rows(rows: list[dict]) -> list[dict]:
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        group = [r for r in rows if r["method"] == method]
        if {r["dataset"] for r in group} != set(DATASETS):
            continue
        mean = {"dataset": "Mean (5)", "method": method, "gt_root": group[0].get("gt_root")}
        for key, _, _ in COLUMNS:
            values = [r[key] for r in group]
            mean[key] = None if any(v is None for v in values) else sum(values) / len(values)
        out.append(mean)
    return out


def main() -> None:
    ours = {(r["dataset"], r["method"]): r for r in main_rows()}
    rows, missing, extra = [], [], []
    for dataset in DATASETS:
        for method in ("IMUHOI++ (ours)", "IMUHOI (3-branch)"):
            rows.append(ours[(dataset, method)])
        four = imuhoi4h_row(dataset)
        if four is None:
            missing.append(f"{dataset}/imuhoi_4branch")
        else:
            rows.append(four)
        for method, suffix, label in BASELINES:
            row = baseline_row(dataset, method, suffix, label)
            if row is None:
                missing.append(f"{dataset}/{method}{suffix}")
            else:
                rows.append(row)
        own_root = baseline_row(dataset, "dip18", "", "DIP (fixed initial root)")
        if own_root is not None:
            extra.append(own_root)

    lines = [
        "# Five datasets: IMUHOI++ vs IMUHOI vs paper baselines (online)",
        "",
        "IMUHOI++ / IMUHOI rows: `outputs/IMUHOI++/RNN_cano/all5_v5ar_vs_imuhoi3b_20261012` (best checkpoints).",
        "IMUHOI (4-branch): ObjectTrans imuhoi_4heads (adds a static-hold branch) retrained on the 3-branch rows'",
        "HP / VC-IMUHOI checkpoints with the same recipe and schedule (`outputs/IMUHOI++/RNN_cano/imuhoi4h_all5_20261026`).",
        "On OMOMO and PAHOI its learned gate puts ~0.6 weight on the static-hold branch during GT contact (carried",
        "objects freeze online); on OMOMO this reproduces over three training seeds (separate table:",
        "`outputs/IMUHOI++/RNN_cano/imuhoi4h_omomo_seeds_20261026/summary.md`). An earlier OMOMO 4-branch run",
        "(relabel_20260920_omomo_imuhoi; bit-identical VC, same HP) weighted IMU there and scores 19.74 / 16.51 obj / HOI:",
        "it was trained without HP output smoothing; retraining today with training-time smoothing off reproduces it",
        "(19.82 / 16.62), so the static-heavy gate comes from human_pose_smoothing_half_width 7 being active in OT training.",
        "Baselines: protocol v3 (paper recipes on the shared stored IMU stream, Noitom noise seed 42), each trained",
        "and tested on one dataset; OMOMO from `outputs/Comparisons/omomo_cano_20261023`, the other four from",
        "`outputs/Comparisons/all4_cano_20261023/<dataset>` (same training segments as the main pipeline, epoch",
        "budgets scaled to the OMOMO run's optimizer steps). Test = the main pipeline's full-sequence test sets.",
        "Same metric definitions in both evaluators (macro mean over sequences; MPJPE root-relative, 22 joints;",
        "jitter on global joints; contact-palm HOI on GT contact frames; object anchored at GT frame 0).",
        "MPJRE = mean absolute axis-angle difference over local joints 1..21 (canonical axis-angle, float64);",
        "Geodesic = mean SO(3) angle between predicted and GT local rotations over the same joints.",
        "IMUHOI++ / IMUHOI MPJRE and geodesic come from a 2026-10-25 re-evaluation of the same checkpoints",
        "(`outputs/Comparisons/per_joint_mpjre_20261025`; all other metrics reproduce exactly): the earlier",
        "values used pytorch3d 0.7.3's matrix_to_axis_angle, which returns angles > pi for rotations above",
        "~94 deg and inflated MPJRE by 0.8-1.4 deg. Baseline values were always canonical and are unchanged.",
        "DIP has no translation estimator: its row uses the GT root translation (root orientation measured).",
        "Baselines have no contact head, so contact F1 is not available. Bold = best per dataset and column.",
        "",
    ]
    if missing:
        lines += [f"Pending: {', '.join(missing)}", ""]
    lines += table(rows) + [""]
    means = mean_rows(rows)
    if means:
        lines += ["## Mean over the five datasets", ""] + table(means) + [""]
    if extra:
        lines += ["## DIP with its own (fixed initial) root, for reference", ""] + table(extra, False) + [""]
    lines += ["## Sources", "", "| Dataset | Method | Seqs | Checkpoint epoch | Eval JSON |", "|---|---|---|---|---|"]
    for row in rows + extra:
        lines.append(
            f"| {NAMES[row['dataset']]} | {row['method']} | {row['seq']} | {row.get('epoch', 'best')} | `{row['source']}` |"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "metrics.json").write_text(json.dumps(rows + extra, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
