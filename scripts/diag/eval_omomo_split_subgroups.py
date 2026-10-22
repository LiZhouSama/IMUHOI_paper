#!/usr/bin/env python3
"""Re-evaluate only current held-out sequences; report old-split membership.

This does not evaluate the 954 old-test sequences used to train the current
checkpoint. All diagnostics use existing current-repository inference code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import eval_IMUHOI as evaluator

OUT = ROOT / "outputs/analysis/omomo_preprocessing_audit_20261016"
EXPERIMENT = ROOT / "outputs/IMUHOI++/paper_noise_omomo_20261016/baseline/vc_imuhoi_ot_3branch"


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    original = evaluator.evaluate_model

    def collect(*args, **kwargs):
        kwargs["sequence_records"] = records
        return original(*args, **kwargs)

    evaluator.evaluate_model = collect
    command = json.loads((EXPERIMENT / "eval_omomo_offline_final.command.json").read_text())
    arguments = command[2:]
    arguments[arguments.index("--output_json") + 1] = str(OUT / "offline_final_recheck.json")
    sys.argv = ["eval_IMUHOI.py", *arguments]
    evaluator.main()
    old_split = json.loads((ROOT.parent / "EgoIMU/process/processed_split_data_OMOMO/split_info.json").read_text())
    old_test, old_train = set(old_split["test_files"]), set(old_split["train_files"])
    groups = {"all_current_test": [], "shared_old_and_current_test": [], "old_train_now_current_test": []}
    for record in records:
        name = Path(record["seq_path"]).name
        if name not in old_test | old_train:
            raise ValueError(f"Unknown source ID {name}")
        record["old_split"] = "test" if name in old_test else "train"
        groups["all_current_test"].append(record)
        groups["shared_old_and_current_test" if name in old_test else "old_train_now_current_test"].append(record)
    if len(records) != 1173 or len(groups["shared_old_and_current_test"]) != 223:
        raise ValueError("Unexpected test-set membership/count")
    summary = {"scope": "Frozen current final model on current held-out data; descriptive strata, not causal split ablation.",
               "inference_mode": "offline", "groups": {}}
    for group, values in groups.items():
        metrics = {}
        for key in ("obj_trans_err_fusion_published_cm", "hoi_err_fusion_cm", "duration_s", "obj_error_on_contact_cm", "obj_error_off_contact_cm"):
            numbers = [float(v[key]) for v in values if v.get(key) is not None and math.isfinite(float(v[key]))]
            metrics[key] = {"n": len(numbers), "mean": float(np.mean(numbers)), "median": float(np.median(numbers))} if numbers else None
        summary["groups"][group] = {"n": len(values), "metrics": metrics}
    aggregate = json.loads((OUT / "offline_final_recheck.json").read_text())["results"]["omomo"]
    previous = json.loads((EXPERIMENT / "eval_omomo_offline_final.json").read_text())["results"]["omomo"]
    summary["previous_run_max_abs_metric_difference"] = max(abs(aggregate[k] - previous[k]) for k in ("obj_trans_err_fusion_refine", "hoi_err_fusion", "mpjpe", "root_trans_err"))
    (OUT / "offline_sequence_records.json").write_text(json.dumps(clean(records), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (OUT / "offline_split_subgroups.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
