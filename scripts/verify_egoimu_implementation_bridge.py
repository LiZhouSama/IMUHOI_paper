#!/usr/bin/env python3
"""Validate bridge receipts and read optimizer-step counts from checkpoints."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from scripts import run_omomo_egoimu_split as restored
    exp = ROOT / "outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016"
    out = ROOT / "outputs/analysis/egoimu_implementation_bridge_20261017"
    ego = ROOT.parent / "EgoIMU"
    provenance = json.loads((exp / "provenance.json").read_text())
    current_unchanged = restored.sources() == provenance["fingerprint"]["source_sha256"]
    native = json.loads((ROOT / "outputs/analysis/egoimu_native_eval_20261017/modules_best_test_disabled/results.json").read_text())
    original_unchanged = all(hashlib.sha256((ego / name).read_bytes()).hexdigest() == value
                             for name, value in native["source_sha256"].items())
    if not current_unchanged or not original_unchanged:
        raise RuntimeError("A production source changed during diagnostics")
    manifest = json.loads((exp / "baseline/experiment_manifest.json").read_text())
    vc_folder = Path(manifest["stages"]["velocity_contact"]["checkpoint"]).parent
    stage = Path(manifest["pipelines"]["vc_imuhoi_ot_3branch"]["stage_dir"])
    paths = {"original_vc_best": ego / "outputs/IMUHOI/omomo/stage_best_velocity_contact.pt",
             "original_joint_best": ego / "outputs/IMUHOI/omomo/stage_best_joint_training.pt",
             "original_joint_last": ego / "outputs/IMUHOI/omomo/final_stage_joint_training_epoch_1199.pt",
             "current_vc_best": vc_folder / "best.pt", "current_vc_final": vc_folder / "final.pt",
             "current_joint_best": stage / "best_joint.pt", "current_joint_final": stage / "final_joint.pt"}
    records = {}
    for label, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        optimizer = payload.get("optimizer_state_dict", {})
        steps = [float(s["step"]) for s in optimizer.get("state", {}).values() if "step" in s]
        records[label] = {"path": str(path), "epoch": payload.get("epoch"),
                          "optimizer_step_min": min(steps) if steps else None,
                          "optimizer_step_max": max(steps) if steps else None,
                          "optimizer_lrs": [g.get("lr") for g in optimizer.get("param_groups", [])]}
    bridge = json.loads((out / "bridge_results.json").read_text())
    drift = json.loads((out / "velocity_drift_results.json").read_text())
    if bridge["sequence_count"] != 1177 or drift["n"] != 1177:
        raise RuntimeError("Incomplete paired diagnostics")
    if bridge["max_absolute_differences"]["vc_original_weights_pred_obj_vel"] != 0:
        raise RuntimeError("VC forward parity did not hold")
    result = {"original_source_unchanged": original_unchanged, "current_frozen_source_unchanged": current_unchanged,
              "paired_sequences": 1177, "optimizer_checkpoints": records}
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
