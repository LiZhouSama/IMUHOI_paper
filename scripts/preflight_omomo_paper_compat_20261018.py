#!/usr/bin/env python3
"""Verify shared inputs and effective loss/initialization contracts before VC++ runs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_omomo_paper_compat_20261018 import OUT, VC_RUN, PYTHON, read, write, verify
from scripts.validate_egoimu_vc_recipe import check_gpu


def main():
    import torch
    import yaml
    from easydict import EasyDict
    from model.rnn.velocity_contact import build_velocity_contact_module
    from train.rnn.loss import IMUHOIVelocityContactLoss, VelocityContactLoss
    from utils.utils import setup_seed

    protocol = read(OUT / "protocol.json")
    verify(protocol)
    configs = {method: EasyDict(yaml.safe_load(Path(path).read_text())) for method, path in protocol["methods"].items()}
    common = ("seed", "train_data_window", "test_data_window", "batch_size", "windows_per_sequence", "epochs", "joint_train_epochs",
              "imu_acc_noise_std", "imu_rot_noise_std", "obj_imu_acc_noise_std", "obj_imu_rot_noise_std", "eval_hoi_protocol")
    for key in common:
        if configs["baseline"][key] != configs["proposed"][key]:
            raise RuntimeError(f"Common condition mismatch: {key}")
    initial = torch.load(VC_RUN / "initial.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    setup_seed(10)
    model = build_velocity_contact_module(configs["proposed"])
    count = 0
    for name in ("hand_vel_net", "obj_vel_net"):
        subset = {key[len(name)+1:]: value for key, value in initial.items() if key.startswith(name + ".")}
        getattr(model, name).load_state_dict(subset, strict=True)
        if not all(torch.equal(value, subset[key]) for key, value in getattr(model, name).state_dict().items()):
            raise RuntimeError("Shared velocity initializer mismatch")
        count += len(subset)
    base_loss = IMUHOIVelocityContactLoss(weights=configs["baseline"].imuhoi_loss_weights)
    proposed_loss = VelocityContactLoss(weights=configs["proposed"].loss_weights)
    for loss in (base_loss, proposed_loss):
        if any(loss.weights[key] != 10 for key in ("obj_vel", "hand_vel")):
            raise RuntimeError("Effective velocity loss weight mismatch")
    checkpoint = torch.load(protocol["baseline_vc"], map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 299 or checkpoint["n_iter"] != 6000:
        raise RuntimeError("Reused baseline VC has wrong budget")
    result = {"status": "passed", "shared_conditions": {key: configs["baseline"][key] for key in common},
              "shared_velocity_initial_tensor_count": count, "strict_velocity_initializer_match": True,
              "effective_velocity_weights_both_methods": [10, 10], "baseline_vc_epoch": 299, "baseline_vc_optimizer_steps": 6000,
              "training_members": len(protocol["data"]["train"]), "test_members": len(protocol["data"]["test"]),
              "proposed_velocity_stage_epochs": configs["proposed"].vc_velocity_stage_epochs,
              "proposed_configured_contact_stage_epochs": configs["proposed"].vc_joint_stage_epochs,
              "proposed_endpoint_only_stage_supported": bool(model.SUPPORTS_ENDPOINT_FINETUNE_STAGE),
              "proposed_effective_contact_stage_epochs": configs["proposed"].epochs - configs["proposed"].vc_velocity_stage_epochs,
              "baseline_wrist_anchor_runtime_check": "Required during first real OT and joint forwards; receipts saved separately"}
    write(OUT / "preflight.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Use WSL native SAGE")
    check_gpu(minimum=0)
    main()
