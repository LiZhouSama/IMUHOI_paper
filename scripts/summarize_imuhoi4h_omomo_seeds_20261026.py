"""Seed repeats of the four-branch IMUHOI on OMOMO: separate table with gate diagnostics.

Rows: seed 42 (imuhoi4h_all5_20261026/omomo) and the repeats under
imuhoi4h_omomo_seeds_20261026/seed<N>.  Every row is scored on the seed-42 Noitom
test stream.  For each checkpoint an extra evaluation with
--sequence_records_json gives the per-sequence gate weights by GT contact
state; the "velocity gain" w_I / (w_I + w_S) is how much of the VC velocity the
fused object keeps (the IMU and static candidates differ only by w_I * v * dt).
Reference rows (not in the mean): IMUHOI++ and 3-branch IMUHOI from the all5
table, and the 2026-09-20 four-branch OMOMO run re-scored under this protocol.

Output: outputs/IMUHOI++/RNN_cano/imuhoi4h_omomo_seeds_20261026/summary.md (+ metrics.json)
Usage: python scripts/summarize_imuhoi4h_omomo_seeds_20261026.py
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/l/anaconda3/envs/SAGE/bin/python"
R = ROOT / "outputs/IMUHOI++/RNN_cano"
OUT = R / "imuhoi4h_omomo_seeds_20261026"
PIPE = "vc_imuhoi_ot_4heads"
SEED_ROOTS = {42: R / "imuhoi4h_all5_20261026/omomo", **{s: OUT / f"seed{s}" for s in (43, 44)}}
OLD_4H = R / "relabel_20260920_omomo_imuhoi"
ALL5 = ROOT / "outputs/Comparisons/all5_baselines_vs_imuhoi_20261023/metrics.json"
EVAL_FLAGS = ["--dataset", "omomo", "--velocity_contact_arch", "imuhoi", "--object_trans_arch", "imuhoi_4heads",
              "--simulate_imu_noise", "--simulate_imu_noise_seed", "42", "--seed", "42", "--num_workers", "2",
              "--online_batch_size", "48", "--human_pose_smoothing_half_width", "7",
              "--online_obj_vel_init_diff_frames", "3"]
METRICS = (("obj", "obj_trans_err_fusion_refine", "Object (cm)"), ("hoi", "hoi_err_fusion", "HOI (cm)"),
           ("mpjpe", "mpjpe", "MPJPE (cm)"), ("root", "root_trans_err", "Root (cm)"),
           ("obj_jitter", "obj_jitter", "Obj jitter"), ("f1_obj", "contact_f1_obj", "Obj F1"))
NAMES = ("left_fk", "right_fk", "imu", "static")
STATES = ("bimanual", "left_only", "right_only", "no_hand")


def diagnose(root: Path, tag: str) -> tuple[dict, dict]:
    """Evaluate the joint best checkpoints once more with per-sequence records."""
    pipeline = json.loads((root / "experiment_manifest.json").read_text())["pipelines"][PIPE]
    out = OUT / "diag" / f"{tag}.json"
    records = OUT / "diag" / f"{tag}_records.json"
    if not records.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [PY, "eval_batch_online.py", "--config", pipeline["config"], "--hp_ckpt", pipeline["joint_hp_ckpt"],
                   "--vc_ckpt", pipeline["joint_vc_ckpt"], "--ot_ckpt", pipeline["joint_ot_ckpt"], *EVAL_FLAGS,
                   "--output_json", str(out), "--sequence_records_json", str(records)]
        with open(out.with_suffix(".log"), "w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(out.read_text())["results"]["omomo"]
    sequences = json.loads(records.read_text())["datasets"]["omomo"]
    gate = {}
    for state in STATES:
        rows = [[r["gating"]["mean_weights_by_gt_contact"][state][n] for n in NAMES]
                for r in sequences if state in r["gating"].get("mean_weights_by_gt_contact", {})]
        w = np.nanmean(rows, axis=0)  # mean over sequences of per-sequence means
        gate[state] = {**dict(zip(NAMES, w.tolist())), "gain": float(w[2] / (w[2] + w[3]))}
    return result, gate


def stopping(root: Path) -> str:
    logs = sorted((root / PIPE).glob("object_trans_joint_train.log"))
    text = logs[0].read_text(errors="ignore") if logs else ""
    stops = re.findall(r"Early stopping at epoch (\d+): no test-metric improvement since epoch (\d+)", text)
    # The OT stage logs its early stop before the joint phase starts.
    joint_start = text.find("joint训练起点")
    ot = [s for s in re.finditer(r"Early stopping at epoch (\d+)", text) if joint_start < 0 or s.start() < joint_start]
    ot_text = f"stop {ot[0].group(1)}" if ot else "no stop (all epochs)"
    joint_text = f"best {stops[-1][1]}" if stops and (not ot or len(stops) > 1) else "–"
    return f"OT {ot_text}; joint {joint_text}"


def fmt(value, digits=2) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def main() -> None:
    rows, missing = [], []
    for seed, root in SEED_ROOTS.items():
        manifest = root / "experiment_manifest.json"
        if not manifest.is_file() or json.loads(manifest.read_text())["pipelines"].get(PIPE, {}).get("status") != "complete":
            missing.append(f"seed {seed}")
            continue
        published = json.loads((root / PIPE / "eval_omomo_best.json").read_text())
        if published.get("simulate_imu_noise_seed") != 42 or published.get("seed") != 42:
            raise ValueError(f"seed {seed}: evaluation is not on the seed-42 stream")
        result, gate = diagnose(root, f"seed{seed}")
        reference = published["results"]["omomo"]
        for _, key, _ in METRICS:  # the diagnostic re-run must reproduce the published evaluation
            if not math.isclose(result[key], reference[key], rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(f"seed {seed}: {key} {result[key]} != {reference[key]}")
        rows.append({"label": f"4-branch seed {seed}", "seed": seed, **{k: reference[key] for k, key, _ in METRICS},
                     "gate": gate, "training": stopping(root), "source": str((root / PIPE).relative_to(ROOT))})

    old_result, old_gate = diagnose(OLD_4H, "relabel_20260920")
    all5 = {r["method"]: r for r in json.loads(ALL5.read_text()) if r["dataset"] == "omomo"}
    references = [
        {"label": "4-branch, 2026-09-20 run (same VC, no HP smoothing in training)", **{k: old_result[key] for k, key, _ in METRICS},
         "gate": old_gate, "training": stopping(OLD_4H), "source": str((OLD_4H / PIPE).relative_to(ROOT))},
        *({"label": m, **{k: all5[m].get(k) for k, _, _ in METRICS}, "gate": None, "training": "",
           "source": all5[m]["source"]} for m in ("IMUHOI (3-branch)", "IMUHOI++ (ours)")),
    ]

    head = "| Run | " + " | ".join(name for _, _, name in METRICS) + " | w_I / w_S / gain (bimanual) | w_I / w_S / gain (one hand) | Training |"
    sep = "|" + "---|" * (len(METRICS) + 4)

    def line(row: dict) -> str:
        cells = [fmt(row.get(k), 3 if k == "f1_obj" else 2) for k, _, _ in METRICS]
        gate = row.get("gate")
        if gate:
            one = {n: (gate["left_only"][n] + gate["right_only"][n]) / 2 for n in ("imu", "static")}
            cells += [f"{gate['bimanual']['imu']:.2f} / {gate['bimanual']['static']:.2f} / {gate['bimanual']['gain']:.2f}",
                      f"{one['imu']:.2f} / {one['static']:.2f} / {one['imu'] / (one['imu'] + one['static']):.2f}"]
        else:
            cells += ["–", "–"]
        return f"| {row['label']} | " + " | ".join(cells) + f" | {row.get('training', '')} |"

    lines = [
        "# Four-branch IMUHOI on OMOMO: training-seed repeats",
        "",
        "Separate from the method comparison table (all5_baselines_vs_imuhoi_20261023). Same recipe, HP",
        "(omomo_cano_20260907) and VC-IMUHOI (imuhoi3b_20261008_omomo) as the all5 four-branch row; only the",
        "training seed (and with it the training noise realization) differs. All rows are scored on the seed-42",
        "Noitom test stream with the 7/3 online protocol (1173 sequences, joint best checkpoints).",
        "Gate columns: mean gate weight of the IMU (w_I) and static-hold (w_S) candidates over GT-contact frames",
        "(per-sequence means averaged over sequences); gain = w_I / (w_I + w_S) = share of the VC velocity the",
        "fused object keeps, since the two candidates differ only by w_I * v * dt.",
        "",
    ]
    if missing:
        lines += [f"Pending: {', '.join(missing)}", ""]
    lines += [head, sep] + [line(r) for r in rows]
    if len(rows) > 1:
        stats = {k: np.array([r[k] for r in rows]) for k, _, _ in METRICS}
        lines.append("| **Mean ± std (n=%d)** | " % len(rows) + " | ".join(
            f"{stats[k].mean():.{3 if k == 'f1_obj' else 2}f} ± {stats[k].std(ddof=1):.{3 if k == 'f1_obj' else 2}f}"
            for k, _, _ in METRICS) + " | | | |")
    lines += ["", "## References (not in the mean)", "", head, sep] + [line(r) for r in references] + [""]
    lines += ["## Sources", ""] + [f"- {r['label']}: `{r['source']}`" for r in rows + references] + [""]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "metrics.json").write_text(json.dumps({"rows": rows, "references": references}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
