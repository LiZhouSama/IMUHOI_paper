#!/usr/bin/env python3
"""Compare effective recipes against original saved configuration before launch."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_omomo_original_recipe_20261018 import OUT, STAGES, original_config, read, write, verify, PYTHON, now
from scripts.validate_egoimu_vc_recipe import check_gpu


def main():
    import yaml
    original = original_config()
    stages = {s["name"]: s for s in original["staged_training"]["stages"]}
    names = {"hp": "human_pose", "vc": "velocity_contact", "ot": "object_trans", "joint": "joint_training"}
    protocol = read(OUT / "protocol.json")
    verify(protocol)
    rows = []
    for method, phase in STAGES:
        cfg = yaml.safe_load(Path(protocol["configs"][f"{method}_{phase}"]).read_text())
        expected = stages[names[phase]]
        for key in ("batch_size", "epochs", "milestones", "gamma"):
            assert cfg[key] == expected[key], (method, phase, key)
        for key in ("seed", "lr", "weight_decay", "num_workers"):
            assert cfg[key] == original[key], (method, phase, key)
        assert cfg["windows_per_sequence"] == 1
        assert cfg["train_data_window"] == cfg["test_data_window"] == original["train"]["window"] == 120
        assert cfg["deterministic_test_windows"] is False and cfg["eval_interval"] == 10
        assert cfg["early_stopping_patience_epochs"] == 0 and cfg["amp_enabled"] is False
        for weight, value in original["loss_weights"].items():
            assert cfg["loss_weights"][weight] == cfg["imuhoi_loss_weights"][weight] == value
        probe = OUT / "preflight" / method / phase
        loader, effective = read(probe / "loader_receipt.json"), read(probe / "effective_recipe.json")
        assert loader["unique_sequence_indices_per_epoch"] == loader["train_sequences"] == 4115
        assert loader["replacement"] is False and loader["windows_per_sequence"] == 1
        assert loader["validation_sequences"] == 1016 and loader["steps_per_epoch"] == 4115 // cfg["batch_size"]
        assert effective["optimizer"] == "AdamW" and effective["amp_enabled"] is False
        assert effective["gradient_clipping"] is False
        for group in effective["optimizer_groups"]:
            assert group["lr"] == original["lr"] and group["weight_decay"] == original["weight_decay"]
            assert group["foreach"] is False
        assert sorted(map(int, effective["milestones"])) == expected["milestones"]
        assert read(probe / "smoke.json")["status"] == "passed"
        rows.append({"stage": f"{method}/{phase}", "batch_size": cfg["batch_size"],
                     "updates": loader["steps_per_epoch"] * cfg["epochs"], "status": "passed"})
    receipt = {"status": "passed", "checked_at": now(), "stages": rows,
               "source": "EgoIMU saved training_config.yaml vs new recipes AND actual runtime optimizer/sampler receipts",
               "remaining_differences": protocol["remaining_differences"]}
    write(OUT / "recipe_audit.json", receipt)
    print(receipt)


if __name__ == "__main__":
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    check_gpu(0)
    main()
