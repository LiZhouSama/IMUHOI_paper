#!/usr/bin/env python3
"""Inspect trusted local EgoIMU artifacts before native evaluation."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EGO = ROOT.parent / "EgoIMU"
OUT = ROOT / "outputs/analysis/egoimu_native_eval_20261017"


def main():
    import torch
    sys.path.insert(0, str(EGO))
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = {"python": sys.executable, "torch": torch.__version__, "imports": {}, "checkpoints": {}}
    for name in ("pytorch3d.transforms", "eval_IMUHOI"):
        module = importlib.import_module(name)
        metadata["imports"][name] = str(Path(module.__file__).resolve())
    paths = list((EGO / "outputs/IMUHOI/omomo/modules").glob("*.pt"))
    paths += [EGO / "outputs/IMUHOI/omomo" / name for name in (
        "stage_best_joint_training.pt", "final.pt", "final_stage_joint_training_epoch_1199.pt")]
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload.get("module_state_dict", payload.get("state_dict", {})))
        metadata["checkpoints"][str(path.relative_to(EGO))] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "keys": list(payload), "epoch": payload.get("epoch"),
            "module_name": payload.get("module_name"), "stage_info": payload.get("stage_info"),
            "state_keys": len(state), "first_state_keys": list(state)[:5],
            "state_prefixes": sorted({key.split(".")[0] for key in state}),
        }
        del payload, state
    metadata["data_counts"] = {split: len(list((EGO / "process/processed_split_data_OMOMO" / split).glob("*.pt")))
                               for split in ("debug", "train", "test")}
    (OUT / "preflight.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
