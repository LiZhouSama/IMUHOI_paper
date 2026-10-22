#!/usr/bin/env python3
"""Read-only inference probes; never replace the registered baseline evaluation."""
from __future__ import annotations

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
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "protocol.json").read_text())
    command = queue.offline_command(root, "baseline", protocol, "final")
    command[command.index("--output_json") + 1] = str(output / "baseline_recheck.json")
    (output / "command.json").write_text(json.dumps(command, indent=2))
    original_evaluate = evaluator.evaluate_model

    def probe_evaluate(model, *args, **kwargs):
        original_inference = model.inference
        original_anchors = model._resolve_ot_hand_positions
        summary = {
            "kind": "inference_only_diagnostic_not_retraining",
            "inference_mode": "offline", "seed": 10,
            "hoi_metric_protocol": "egoimu", "test_sequences": 1177,
            "checkpoint_selection": "baseline/final",
            "checkpoint_paths": {
                key: command[command.index(key) + 1]
                for key in ("--hp_ckpt", "--vc_ckpt", "--ot_ckpt")
            },
            "probes": {},
        }
        baseline = None
        try:
            for name, palms, gt_init in (
                ("registered_baseline", False, False),
                ("palm_anchors_match_training", True, False),
                ("original_gate_initial_velocity", False, True),
                ("palms_and_original_gate_init", True, True),
            ):
                evaluator._set_eval_seed(10)
                counts = {"calls": 0, "success": 0, "modified_initializers": 0}

                def inference(_self, data_dict, *infer_args, **infer_kwargs):
                    counts["calls"] += 1
                    if gt_init:
                        data_dict = dict(data_dict)
                        contact = data_dict["contact_init"]
                        if contact.shape[-1] != 3:
                            raise ValueError("Expected the registered three-state contact initializer")
                        data_dict["contact_init"] = torch.cat(
                            (contact, data_dict["obj_vel_init"]), dim=-1
                        )
                        counts["modified_initializers"] += 1
                    result = original_inference(data_dict, *infer_args, **infer_kwargs)
                    counts["success"] += 1
                    return result

                def palm_anchors(_self, hp_out):
                    return hp_out["pred_palm_glb_pos"]

                model.inference = types.MethodType(inference, model)
                model._resolve_ot_hand_positions = (
                    types.MethodType(palm_anchors, model) if palms else original_anchors
                )
                records = []
                probe_kwargs = {**kwargs, "sequence_records": records}
                print(f"\nSTART DIAGNOSTIC {name}", flush=True)
                metrics = original_evaluate(model, *args, **probe_kwargs)
                if len(records) != 1177 or counts["calls"] != counts["success"]:
                    raise RuntimeError(f"Incomplete probe {name}: {len(records)}, {counts}")
                summary["probes"][name] = {
                    "metrics": metrics, "sequence_count": len(records), "inference_counts": counts,
                    "palm_anchor_override": palms, "gate_gt_initial_velocity_override": gt_init,
                }
                (output / "probes.json").write_text(json.dumps(summary, indent=2))
                (output / f"{name}_sequences.json").write_text(json.dumps(records))
                print(f"FINISHED DIAGNOSTIC {name}: {json.dumps(metrics)}", flush=True)
                if baseline is None:
                    baseline = metrics
        finally:
            model.inference = original_inference
            model._resolve_ot_hand_positions = original_anchors
        registered = queue.read_evaluation(root, "baseline", "offline", "final")
        for key in ("mpjpe", "root_trans_err", "obj_trans_err_fusion_refine", "hoi_err_fusion"):
            if abs(baseline[key] - registered[key]) > 1e-4:
                raise RuntimeError(f"Baseline recheck diverged on {key}")
        summary["registered_baseline_reproduced"] = True
        (output / "probes.json").write_text(json.dumps(summary, indent=2))
        return baseline

    evaluator.evaluate_model = probe_evaluate
    sys.argv = command[1:]
    evaluator.main()
    if not (output / "baseline_recheck.json").is_file():
        raise RuntimeError("Evaluator did not save a successful result")


if __name__ == "__main__":
    main()
