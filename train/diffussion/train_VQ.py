"""
CondVQModule standalone training script.
"""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.diffussion.cond_VQ import CondVQModule
from train.diffussion.loss.cond_vq_loss import CondVQLoss
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
    parser.description = "CondVQModule training"
    return parser.parse_args()


class CondVQTrainer(BaseTrainer):
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
        return self.model(data_dict, gt_targets=batch)


def main():
    args = get_args()
    cfg = merge_config(args)
    cfg.model_arch = "dit"

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    save_dir = create_save_dir(cfg, "cond_vq")

    print("=" * 50)
    print("Stage 1: CondVQModule training")
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
    model = CondVQModule(cfg).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())}")

    loss_fn = CondVQLoss(
        weights=getattr(cfg, "loss_weights", {}),
        test_metric_weights=getattr(cfg, "test_metric_weights", {}),
        commit_beta=float(getattr(cfg, "vq_commit_beta", 0.25)),
    )

    trainer = CondVQTrainer(cfg, model, loss_fn, train_loader, test_loader)
    trainer.train()

    if not cfg.debug:
        import yaml

        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dict(cfg), f)


if __name__ == "__main__":
    main()
