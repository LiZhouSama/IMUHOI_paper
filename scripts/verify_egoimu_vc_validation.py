#!/usr/bin/env python3
"""Check paired crop contracts and the largest planned training batch."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import OUT, EXP, PYTHON, check_gpu, verify_receipt, write_json


def main():
    import torch
    import yaml
    from easydict import EasyDict
    from torch.utils.data._utils.collate import default_collate
    from dataset.dataset_IMUHOI import IMUDataset
    from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule
    from train.rnn.loss import IMUHOIVelocityContactLoss
    from train.rnn.train_utils import build_model_input_dict, autocast
    from utils.utils import setup_seed

    protocol = json.loads((OUT / "protocol.json").read_text())
    verify_receipt(protocol)
    path = EXP / "data/train" / protocol["training_files"][0]["file"]
    samples = []
    for window in (60, 120):
        dataset = IMUDataset(data_dir=[], sequence_paths=[str(path)], window_size=window, require_canonical_skeleton=True)
        dataset.sequence_info[0]["seq_len"] -= 120 - window
        setup_seed(10)
        samples.append(dataset[0])
    checks = {}
    for key in ("obj_imu", "human_imu", "obj_vel", "sensor_vel_glb", "obj_trans"):
        a, b = samples[0][key], samples[1][key][:60]
        checks[key] = bool(torch.equal(a, b))
    if not all(checks.values()):
        raise RuntimeError(f"Paired crop mismatch: {checks}")
    cfg = EasyDict(yaml.safe_load((OUT / "w120_v1/config.yaml").read_text()))
    setup_seed(10)
    device = torch.device("cuda:0")
    model = IMUHOIVelocityContactModule(cfg).to(device).train()
    initial = torch.load(OUT / "initial.pt", map_location=device, weights_only=False)["model_state_dict"]
    model.load_state_dict(initial, strict=True)
    batch = default_collate([samples[1]] * cfg.batch_size)
    data = build_model_input_dict(batch, cfg, device, add_noise=True)
    torch.cuda.reset_peak_memory_stats()
    with autocast(enabled=cfg.amp_enabled, dtype=torch.bfloat16):
        pred = model(data)
        loss, _, _ = IMUHOIVelocityContactLoss(weights=cfg.imuhoi_loss_weights)(pred, batch, device)
    loss.backward()
    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError("Largest training shape has nonfinite loss/gradients")
    result = {"paired_cached_window_prefix_bitwise_equal": checks, "batch_shape": [cfg.batch_size, 120],
              "loss": loss.item(), "gradients_finite": True,
              "max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "note": "One real training window repeated only for memory/shape validation; not an accuracy result"}
    write_json(OUT / "runtime_verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError("Use WSL SAGE")
    check_gpu(minimum=6000)
    main()
