"""Read-only verification of restored data, frozen recipe and active HP config."""
from pathlib import Path
import hashlib
import json
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.run_omomo_egoimu_split import validate_split


def main():
    experiment = ROOT / "outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016"
    previous = yaml.safe_load((ROOT / "outputs/IMUHOI++/paper_noise_omomo_20261016/recipe.yaml").read_text())
    current = yaml.safe_load((experiment / "recipe.yaml").read_text())
    restored = json.loads((experiment / "data/split_info.json").read_text())
    old = json.loads((ROOT.parent / "EgoIMU/process/processed_split_data_OMOMO/split_info.json").read_text())
    assert validate_split(restored) == validate_split(old)
    inventory = json.loads((experiment / "data/inventory.json").read_text())
    lookup = {entry["name"]: entry for entries in inventory.values() for entry in entries}
    verified = 0
    for split in ("train", "test"):
        for path in (ROOT / previous["datasets"]["omomo"][split + "_path"]).glob("*.pt"):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == lookup[path.name]["sha256"], path
            verified += 1
    changes = [key for key in set(previous) | set(current) if previous.get(key) != current.get(key)]
    assert set(changes) == {"seed", "simulate_imu_noise_seed", "datasets", "eval_hoi_protocol"}
    assert current["seed"] == 10 and current["eval_hoi_protocol"] == "egoimu"
    assert current["pretrained_modules"] == {"human_pose": None, "velocity_contact": None, "object_trans": None}
    command = json.loads((experiment / "baseline_command.json").read_text())
    assert command[command.index("--seed") + 1] == "10"
    assert "--skip_hp" not in command and "--skip_vc" not in command
    hp_configs = sorted((experiment / "baseline/hp").glob("*/config.yaml"))
    assert len(hp_configs) == 1
    hp = yaml.safe_load(hp_configs[0].read_text())
    assert hp["seed"] == 10 and hp["eval_hoi_protocol"] == "egoimu"
    assert not hp["simulate_imu_noise"] and hp["validation_add_imu_noise"]
    assert hp["epochs"] == 300 and hp["batch_size"] == 800
    assert hp["windows_per_sequence"] == 4 and hp["num_workers"] == 0
    assert not hp["no_trans"]
    assert hp["pretrained_modules"] == {"human_pose": None, "velocity_contact": None, "object_trans": None}
    result = {"split_counts": {key: len(value) for key, value in inventory.items()},
              "previously_assigned_current_files_byte_verified": verified,
              "recovered_unassigned_ids": sorted(set(lookup) - {p.name for split in ("train", "test")
                                                                 for p in (ROOT / previous["datasets"]["omomo"][split + "_path"]).glob("*.pt")}),
              "recipe_changes": sorted(changes), "training_seed": hp["seed"],
              "hp_config": str(hp_configs[0]), "passed": True}
    path = experiment / "launch_verification.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
