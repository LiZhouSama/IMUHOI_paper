#!/usr/bin/env python3
"""Measure baseline OT branch use on the full clean offline test set."""
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import eval_IMUHOI as evaluator
from scripts import run_omomo_paper_noise as queue


def main():
    if Path(sys.executable).resolve() != queue.PYTHON.resolve():
        raise RuntimeError(f"Use {queue.PYTHON}")
    root = ROOT / "outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016"
    output = ROOT / "outputs/analysis/egoimu_contract_audit_20261017"
    protocol = json.loads((root / "protocol.json").read_text())
    command = queue.offline_command(root, "baseline", protocol, "final")
    command[command.index("--output_json") + 1] = str(output / "branch_audit_recheck.json")
    original_evaluate = evaluator.evaluate_model

    def evaluate(model, *args, **kwargs):
        original_inference = model.inference
        records = []

        def inference(_self, data, *infer_args, **infer_kwargs):
            result = original_inference(data, *infer_args, **infer_kwargs)
            gt = infer_kwargs["gt_targets"]
            target = gt["obj_trans"]
            weights = result["gating_weights"]
            for i in range(weights.shape[0]):
                record = {"frames": weights.shape[1]}
                w = weights[i].float()
                record["mean_weights_left_right_imu"] = w.mean(0).cpu().tolist()
                record["dominant_fractions_left_right_imu"] = [
                    (w.argmax(-1) == j).float().mean().item() for j in range(3)
                ]
                record["imu_weight_gt_099_fraction"] = (w[:, 2] > 0.99).float().mean().item()
                any_contact = (gt["lhand_contact"][i].bool() | gt["rhand_contact"][i].bool())
                record["contact_frames"] = any_contact.sum().item()
                record["mean_weights_on_gt_contact"] = (
                    w[any_contact].mean(0).cpu().tolist() if any_contact.any() else None
                )
                errors = {}
                for name, key in (("left", "pred_lhand_obj_trans"), ("right", "pred_rhand_obj_trans"),
                                  ("imu", "pred_imu_obj_trans"), ("fusion", "pred_obj_trans")):
                    error = (result[key][i] - target[i]).float().norm(dim=-1) * 100
                    errors[name] = error
                    record[f"{name}_error_cm"] = error.mean().item()
                for side, joint, index in (("left", 20, 0), ("right", 21, 1)):
                    contact = gt[f"{side[0]}hand_contact"][i].bool()
                    record[f"{side}_gt_contact_frames"] = contact.sum().item()
                    record[f"{side}_error_on_own_contact_cm"] = (
                        errors[side][contact].mean().item() if contact.any() else None
                    )
                    pred_hand = result["pred_hand_glb_pos"][i, :, index]
                    gt_hand = gt["position_global"][i, :, joint]
                    hand_error = (pred_hand - gt_hand).float().norm(dim=-1) * 100
                    candidate = result[f"pred_{side[0]}hand_obj_trans"][i]
                    vector_error = ((candidate - pred_hand) - (target[i] - gt_hand)).float().norm(dim=-1) * 100
                    record[f"{side}_hand_error_on_own_contact_cm"] = hand_error[contact].mean().item() if contact.any() else None
                    record[f"{side}_vector_error_on_own_contact_cm"] = vector_error[contact].mean().item() if contact.any() else None
                record["frame_oracle_candidate_error_cm"] = torch.stack(
                    [errors[key] for key in ("left", "right", "imu")], dim=-1
                ).min(-1).values.mean().item()
                records.append(record)
            return result

        model.inference = types.MethodType(inference, model)
        try:
            metrics = original_evaluate(model, *args, **kwargs)
        finally:
            model.inference = original_inference
        if len(records) != 1177:
            raise RuntimeError(f"Incomplete branch audit: {len(records)}")
        summary = {"kind": "read_only_baseline_offline_branch_audit", "sequence_count": len(records),
                   "imu_candidate_semantics": "previous fused position plus one predicted velocity step; not the independent pure velocity integral",
                   "branch_order": ["left", "right", "imu"], "averaging": "equal_sequence",
                   "mean_weights": [sum(r["mean_weights_left_right_imu"][j] for r in records) / len(records) for j in range(3)],
                   "mean_dominant_fractions": [sum(r["dominant_fractions_left_right_imu"][j] for r in records) / len(records) for j in range(3)],
                   "mean_imu_weight_gt_099_fraction": sum(r["imu_weight_gt_099_fraction"] for r in records) / len(records)}
        for key in ("left_error_cm", "right_error_cm", "imu_error_cm", "fusion_error_cm", "frame_oracle_candidate_error_cm"):
            summary[key] = sum(r[key] for r in records) / len(records)
        for side in ("left", "right"):
            for suffix in ("error_on_own_contact_cm", "hand_error_on_own_contact_cm", "vector_error_on_own_contact_cm"):
                key = f"{side}_{suffix}"
                values = [r[key] for r in records if r[key] is not None]
                summary[key] = sum(values) / len(values)
            summary[f"{side}_gt_contact_sequences"] = sum(r[f"{side}_gt_contact_frames"] > 0 for r in records)
        on_contact = [r["mean_weights_on_gt_contact"] for r in records if r["mean_weights_on_gt_contact"] is not None]
        summary["mean_weights_on_gt_contact"] = [sum(w[j] for w in on_contact) / len(on_contact) for j in range(3)]
        summary["gt_contact_sequences"] = len(on_contact)
        (output / "branch_usage.json").write_text(json.dumps(summary, indent=2))
        (output / "branch_usage_sequences.json").write_text(json.dumps(records))
        print("BRANCH AUDIT " + json.dumps(summary), flush=True)
        return metrics

    evaluator.evaluate_model = evaluate
    sys.argv = command[1:]
    evaluator.main()
    if not (output / "branch_audit_recheck.json").exists():
        raise RuntimeError("Evaluator did not complete")


if __name__ == "__main__":
    main()
