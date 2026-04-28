"""HumanPoseModule standalone training script for the Mamba path."""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.mamba import HumanPoseModule
from train.mamba.loss.human_pose_loss import HumanPoseLoss
from train.mamba.train_utils import (
    BaseTrainer,
    create_dataloaders,
    create_save_dir,
    get_base_args,
    merge_config,
    setup_device,
    setup_seed,
)


def get_args():
    parser = get_base_args()
    parser.description = "HumanPoseModule Mamba training (Stage 1)"
    return parser.parse_args()


class HumanPoseTrainer(BaseTrainer):
    """Trainer for Mamba Stage-1 HumanPose."""

    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None):
        super().__init__(cfg, model, loss_fn, train_loader, test_loader)
        self.no_trans = cfg.no_trans

    def model_forward(self, data_dict, batch=None):
        return self.model(data_dict)


def main():
    args = get_args()
    cfg = merge_config(args)
    cfg.model_arch = "mamba"

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    save_dir = create_save_dir(cfg, "human_pose_mamba")

    mode_str = "noTrans" if cfg.no_trans else "normal"
    print("=" * 50)
    print(f"Stage 1: Mamba HumanPoseModule training ({mode_str})")
    print(f"device: {cfg.device}")
    print(f"batch size: {cfg.batch_size}")
    print(f"epochs: {cfg.epoch}")
    print(f"noTrans: {cfg.no_trans}")
    print(f"save dir: {save_dir}")
    print("=" * 50)

    train_loader, test_loader = create_dataloaders(cfg)
    if train_loader is None or len(train_loader) == 0:
        print("Error: failed to create training dataloader")
        return

    device = torch.device(cfg.device)
    model = HumanPoseModule(cfg, device, no_trans=cfg.no_trans).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}")

    loss_weights = getattr(cfg, "loss_weights", {})
    loss_fn = HumanPoseLoss(weights=loss_weights, no_trans=cfg.no_trans)

    trainer = HumanPoseTrainer(cfg, model, loss_fn, train_loader, test_loader)
    trainer.train()

    print(f"\nTraining complete. Model saved to: {save_dir}")

    if not cfg.debug:
        import yaml

        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dict(cfg), f)


if __name__ == "__main__":
    main()

