#!/usr/bin/env python3
"""Summarize paired offline diagnostics without reading or executing EgoIMU code."""
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_omomo_egoimu_split as restored

FOLDER = ROOT / "outputs/analysis/egoimu_contract_audit_20261017"


def mean(values):
    values = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.mean(values) if values else None


def main():
    experiment = ROOT / "outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016"
    frozen = json.loads((experiment / "provenance.json").read_text())
    if frozen["fingerprint"]["source_sha256"] != restored.sources():
        raise RuntimeError("Frozen training/evaluation source changed during diagnostics")
    probes = json.loads((FOLDER / "probes.json").read_text())
    if not probes.get("registered_baseline_reproduced") or len(probes["probes"]) != 4:
        raise RuntimeError("Wait until all four complete probes and the baseline equality check finish")
    base = json.loads((FOLDER / "registered_baseline_sequences.json").read_text())
    keys = ("obj_trans_err_fusion_published_cm", "hoi_err_fusion_cm")
    summary = {
        "count": len(base),
        "frozen_source_hashes_verified": True,
        "diagnostic_only": True,
        "baseline_first_10pct_obj_cm": mean(x["obj_error_profile_cm"]["first_10pct_mean"] for x in base),
        "baseline_last_10pct_obj_cm": mean(x["obj_error_profile_cm"]["last_10pct_mean"] for x in base),
        "baseline_length_groups": {}, "paired_changes": {},
    }
    for label, low, high in (("frames_le_120", 0, 120), ("frames_121_240", 121, 240), ("frames_gt_240", 241, float("inf"))):
        rows = [x for x in base if low <= x["num_frames"] <= high]
        summary["baseline_length_groups"][label] = {
            "n": len(rows), **{key: mean(x[key] for x in rows) for key in keys}
        }
    for name in probes["probes"]:
        rows = json.loads((FOLDER / f"{name}_sequences.json").read_text())
        if [x["seq_path"] for x in rows] != [x["seq_path"] for x in base]:
            raise RuntimeError("Paired sequence order mismatch")
        changes = {}
        for key in keys:
            diffs = [new[key] - old[key] for old, new in zip(base, rows)]
            changes[key] = {
                "mean_delta_cm": mean(diffs), "median_delta_cm": statistics.median(diffs),
                "improved_sequences": sum(d < -1e-6 for d in diffs),
                "worse_sequences": sum(d > 1e-6 for d in diffs),
                "unchanged_sequences": sum(abs(d) <= 1e-6 for d in diffs),
            }
        summary["paired_changes"][name] = changes
    (FOLDER / "paired_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
