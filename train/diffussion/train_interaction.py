"""
InteractionModule standalone training script (Stage 2).
"""
from __future__ import annotations

import os
import sys

import torch

# add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import InteractionModule
from train.diffussion.loss.interaction_loss import InteractionLoss
from train.diffussion.train_utils import (
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
    parser.description = "InteractionModule training (Stage 2)"
    return parser.parse_args()


class InteractionTrainer(BaseTrainer):
    """Trainer for Stage-2 Interaction diffusion model."""

    def model_forward(
        self,
        data_dict,
        batch=None,
        use_gt_targets: bool = True,
        force_inference: bool = False,
        sample_steps: int | None = None,
        sampler: str | None = None,
        eta: float | None = None,
    ):
        gt_arg = batch if use_gt_targets else None
        if force_inference:
            return self.model.inference(
                data_dict,
                gt_targets=batch,
                sample_steps=sample_steps,
                sampler=sampler,
                eta=eta,
            )
        return self.model(data_dict, gt_targets=gt_arg)


def main():
    args = get_args()
    cfg = merge_config(args)

    # Interaction training is DiT-only.
    cfg.model_arch = "dit"

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    save_dir = create_save_dir(cfg, "interaction")

    print("=" * 50)
    print("Stage 2: InteractionModule training")
    print(f"device: {cfg.device}")
    print(f"batch size: {cfg.batch_size}")
    print(f"epochs: {cfg.epochs}")
    print(f"save dir: {save_dir}")
    print("=" * 50)

    train_loader, test_loader = create_dataloaders(cfg)
    if train_loader is None or len(train_loader) == 0:
        print("Error: failed to create training dataloader")
        return

    device = torch.device(cfg.device)
    model = InteractionModule(cfg).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}")

    loss_weights = getattr(cfg, "loss_weights", {})
    test_metric_weights = getattr(cfg, "test_metric_weights", {})
    loss_fn = InteractionLoss(weights=loss_weights, test_metric_weights=test_metric_weights)

    trainer = InteractionTrainer(cfg, model, loss_fn, train_loader, test_loader)
    trainer.train()

    print(f"\nTraining complete. Model saved to: {save_dir}")

    if not cfg.debug:
        import yaml

        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dict(cfg), f)


if __name__ == "__main__":
    main()
