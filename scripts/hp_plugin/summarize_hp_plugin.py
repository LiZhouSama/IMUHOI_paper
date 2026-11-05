#!/usr/bin/env python3
"""Summarize the HP plug-in experiment (scripts/hp_plugin/run_plugin.sh).

For every dataset and comparison method X, three rows on the same full-sequence test set:
  X                     the method alone (its own object head), from the cache build's eval of the
                        full test set (== the published comparison row; DIP with GT root),
  X + IMUHOI++ VC/OT    our VC + OT (+ joint stage) retrained on X's frozen pose,
  IMUHOI++              the published pipeline (our HP).
Columns: object / HOI error (cm), MPJPE (cm), root error (cm), object jitter; for obj / HOI the
paired sequence bootstrap 95% CI of (X + VC/OT) - X.  Missing jobs are shown as "-".

    python scripts/hp_plugin/summarize_hp_plugin.py [--root outputs/IMUHOI++/RNN_cano/hp_plugin_20261105]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
R = PROJECT_ROOT / "outputs/IMUHOI++/RNN_cano"
DATASETS = ("omomo", "behave", "imhd", "pahoi", "hodome")
METHODS = ("dip18", "tip", "transpose", "globalpose")
LABELS = {"dip18": "DIP (GT root)", "tip": "TIP", "transpose": "TransPose", "globalpose": "GlobalPose",
          "omomo": "OMOMO", "behave": "BEHAVE", "imhd": "IMHD", "pahoi": "PAHOI", "hodome": "HODome"}
PUBLISHED = {
    "omomo": R / "vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/eval_omomo_best.json",
    "behave": R / "vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/eval_behave_best.json",
    **{ds: R / f"remaining3_v5ar_vs_imuhoi3b_20261010/{ds}/proposed/v2_armrot_biclose/eval_{ds}_best.json"
       for ds in ("imhd", "pahoi", "hodome")},
}
COLUMNS = ("obj", "hoi", "mpjpe", "root", "obj_jitter")


def ours_metrics(path: Path, dataset: str):
    if not path.is_file():
        return None
    result = json.loads(path.read_text())["results"][dataset]
    return {"obj": result["obj_trans_err_fusion_refine"], "hoi": result["hoi_err_fusion"],
            "mpjpe": result["mpjpe"], "root": result["root_trans_err"], "obj_jitter": result["obj_jitter"]}


def baseline_eval(cache_dir: Path):
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None, None
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["directories"][manifest["full_sequence_test"]]
    report = json.loads((cache_dir / entry["eval_json"]).read_text())
    metrics = report["metrics"]
    gt_root = "ground_truth_oracle" in (report.get("translation_mode") or [])
    row = {"obj": metrics["obj_err_cm"], "hoi": metrics["hoi_err_cm"], "mpjpe": metrics["human_pos_err_cm"],
           "root": None if gt_root else metrics["root_trans_err_cm"],
           "obj_jitter": metrics["obj_trans_jitter_mm_frame2"]}
    per_seq = {r["seq_name"]: {"obj": r["metrics"].get("obj_err_cm"), "hoi": r["metrics"].get("hoi_err_cm")}
               for r in report.get("per_sequence", [])}
    return row, per_seq


def plugin_records(path: Path, dataset: str):
    if not path.is_file():
        return None
    rows = json.loads(path.read_text())["datasets"][dataset]
    return {r["seq_name"]: {"obj": r.get("obj_trans_err_fusion_published_cm"), "hoi": r.get("hoi_err_fusion_cm")}
            for r in rows}


def paired_bootstrap(a: dict, b: dict, key: str, draws: int = 1000, seed: int = 0):
    """Mean of b - a over common sequences with a percentile 95% CI (sequence resampling)."""
    names = sorted(n for n in set(a) & set(b) if a[n].get(key) is not None and b[n].get(key) is not None)
    diffs = [b[n][key] - a[n][key] for n in names
             if a[n][key] == a[n][key] and b[n][key] == b[n][key]]  # drop NaN
    if not diffs:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(draws))
    return {"mean": sum(diffs) / len(diffs), "lo": means[int(0.025 * draws)],
            "hi": means[int(0.975 * draws) - 1], "n": len(diffs)}


def fmt(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=R / "hp_plugin_20261105")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root

    lines = [
        "# HP plug-in: IMUHOI++ VC + OT on other methods' human pose",
        "",
        "Rows per method: X alone (its own object head; DIP uses the GT root as in the comparison table), "
        "X + our VC/OT (retrained on X's frozen online pose), and IMUHOI++ (our HP). Full-sequence test sets, "
        "noise seed 42. Delta = (X + VC/OT) - X, mean over sequences with a paired sequence-bootstrap 95% CI.",
        "",
        "| Dataset | HP | Pipeline | Obj (cm) | HOI (cm) | MPJPE (cm) | Root (cm) | Obj jitter | dObj [95% CI] | dHOI [95% CI] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    summary = {}
    for dataset in DATASETS:
        ours = ours_metrics(PUBLISHED[dataset], dataset)
        for method in METHODS:
            base, base_seq = baseline_eval(root / "cache" / dataset / method)
            pipe = root / dataset / method / "pipeline/v2_armrot_biclose"
            plug = ours_metrics(pipe / f"eval_{dataset}_best.json", dataset)
            plug_seq = plugin_records(pipe / f"eval_{dataset}_best_records.json", dataset)
            deltas = {}
            if base_seq and plug_seq:
                for key in ("obj", "hoi"):
                    deltas[key] = paired_bootstrap(base_seq, plug_seq, key)
            summary.setdefault(dataset, {})[method] = {"baseline": base, "plugin": plug, "deltas": deltas}

            def ci(key):
                d = deltas.get(key)
                return "-" if not d else f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"

            for label, row, extra in (("X", base, ("", "")), ("X + VC/OT", plug, (ci("obj"), ci("hoi")))):
                cells = [fmt(row[c]) if row else "-" for c in COLUMNS]
                lines.append(f"| {LABELS[dataset]} | {LABELS[method]} | {label} | " + " | ".join(cells)
                             + f" | {extra[0]} | {extra[1]} |")
        cells = [fmt(ours[c]) if ours else "-" for c in COLUMNS]
        lines.append(f"| {LABELS[dataset]} | IMUHOI++ HP | IMUHOI++ | " + " | ".join(cells) + " |  |  |")
        summary[dataset]["imuhoi_pp"] = ours
    out_md = root / "summary.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
