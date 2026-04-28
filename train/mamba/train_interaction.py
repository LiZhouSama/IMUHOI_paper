"""InteractionModule standalone training script for the Mamba path."""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.mamba import HumanPoseModule, InteractionModule
from train.mamba.loss.interaction_loss import InteractionLoss
from train.mamba.train_utils import (
    BaseTrainer,
    create_dataloaders,
    create_save_dir,
    get_base_args,
    load_checkpoint,
    merge_config,
    setup_device,
    setup_seed,
)


def get_args():
    parser = get_base_args()
    parser.description = "InteractionModule Mamba training (Stage 2)"
    parser.add_argument("--hp_ckpt", type=str, default=None, help="Frozen HumanPoseModule checkpoint")
    return parser.parse_args()


class InteractionTrainer(BaseTrainer):
    """Trainer for the Mamba interaction stage with a frozen Stage-1 model."""

    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None, hp_model=None):
        super().__init__(cfg, model, loss_fn, train_loader, test_loader)
        self.hp_model = hp_model
        if self.hp_model is not None:
            self.hp_model.eval()
            for param in self.hp_model.parameters():
                param.requires_grad_(False)

    def _set_interaction_epoch(self, epoch: int):
        module = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(module, "set_epoch"):
            module.set_epoch(epoch)

    def model_forward(self, data_dict, batch=None):
        hp_out = None
        if self.hp_model is not None:
            with torch.no_grad():
                hp_out = self.hp_model(data_dict)
        return self.model(data_dict, hp_out=hp_out, gt_targets=batch)

    def train_epoch(self, epoch):
        self._set_interaction_epoch(epoch)
        return super().train_epoch(epoch)

    def evaluate(self, epoch):
        self._set_interaction_epoch(epoch)
        return super().evaluate(epoch)


def _resolve_hp_checkpoint(cfg, arg_path):
    if arg_path:
        return arg_path
    pretrained = getattr(cfg, "pretrained_modules", None)
    if isinstance(pretrained, dict):
        return pretrained.get("human_pose")
    if pretrained is not None:
        return getattr(pretrained, "human_pose", None)
    return None


def main():
    args = get_args()
    cfg = merge_config(args)
    cfg.model_arch = "mamba"

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    save_dir = create_save_dir(cfg, "interaction_mamba")

    print("=" * 50)
    print("Stage 2: Mamba InteractionModule training")
    print(f"device: {cfg.device}")
    print(f"batch size: {cfg.batch_size}")
    print(f"epochs: {cfg.epoch}")
    print(f"save dir: {save_dir}")
    print("=" * 50)

    train_loader, test_loader = create_dataloaders(cfg)
    if train_loader is None or len(train_loader) == 0:
        print("Error: failed to create training dataloader")
        return

    device = torch.device(cfg.device)
    hp_ckpt = _resolve_hp_checkpoint(cfg, args.hp_ckpt)
    hp_model = None
    if hp_ckpt:
        hp_model = HumanPoseModule(cfg, device, no_trans=cfg.no_trans).to(device)
        if os.path.exists(hp_ckpt):
            load_checkpoint(hp_model, hp_ckpt, device, strict=False)
            print(f"Loaded frozen HumanPose checkpoint: {hp_ckpt}")
        else:
            print(f"Warning: HumanPose checkpoint not found at {hp_ckpt}; Interaction will fall back to GT human context where available.")
            hp_model = None
    else:
        print("Warning: --hp_ckpt/pretrained_modules.human_pose not set; Stage 2 will not use Stage 1 predictions.")

    model = InteractionModule(cfg).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}")

    loss_weights = getattr(cfg, "loss_weights", {})
    loss_fn = InteractionLoss(weights=loss_weights)
    trainer = InteractionTrainer(cfg, model, loss_fn, train_loader, test_loader, hp_model=hp_model)
    trainer.train()

    print(f"\nTraining complete. Model saved to: {save_dir}")

    if not cfg.debug:
        import yaml

        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dict(cfg), f)


if __name__ == "__main__":
    main()
