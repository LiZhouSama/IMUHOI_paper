"""
InteractionVQ training script for stage-2 obs diffusion, stage-3 main DiT, and joint training.
"""
from __future__ import annotations

import os
import sys

import torch
from torch import optim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.diffussion.interaction_VQ import InteractionVQModule, ObsTokenDiffusionModule
from train.diffussion.loss.interaction_vq_loss import InteractionVQLoss
from train.diffussion.train_utils import (
    BaseTrainer,
    create_dataloaders,
    create_save_dir,
    get_base_args,
    merge_config,
    setup_device,
    setup_seed,
)
from utils.utils import load_checkpoint


def get_args():
    parser = get_base_args()
    parser.description = "InteractionVQ training"
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["obs_diffusion", "main_dit", "joint"],
        help="interaction_vq training stage",
    )
    return parser.parse_args()


def _resolve_stage(cfg, args) -> str:
    if args.stage:
        return str(args.stage).lower()
    section = getattr(cfg, "interaction_vq", {})
    if isinstance(section, dict) and section.get("stage"):
        return str(section["stage"]).lower()
    return "obs_diffusion"


class InteractionVQTrainer(BaseTrainer):
    def __init__(self, *args, stage: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = str(stage).lower()

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
        if self.stage == "obs_diffusion":
            return self.model(data_dict, gt_targets=batch)
        if force_inference:
            return self.model.inference(
                data_dict,
                gt_targets=batch,
                sample_steps=sample_steps,
                sampler=sampler,
                eta=eta,
            )
        joint = self.stage == "joint"
        return self.model(data_dict, gt_targets=batch, joint_training=joint)


def _load_stage_dependencies(cfg, model, stage: str, device: torch.device):
    pretrained = getattr(cfg, "pretrained_modules", {}) or {}
    if stage == "obs_diffusion":
        cond_vq_path = pretrained.get("cond_vq")
        if cond_vq_path:
            load_checkpoint(model.tokenizer, cond_vq_path, device, strict=False)
            print(f"Loaded CondVQ tokenizer from: {cond_vq_path}")
        model.freeze_tokenizer()
        return

    if stage == "joint":
        # Joint training: load all pretrained weights but do NOT freeze
        cond_vq_path = pretrained.get("cond_vq")
        if cond_vq_path:
            load_checkpoint(model.obs_token_diffusion.tokenizer, cond_vq_path, device, strict=False)
            print(f"Loaded CondVQ tokenizer from: {cond_vq_path}")
        obs_path = pretrained.get("interaction_vq_obs")
        if obs_path:
            load_checkpoint(model.obs_token_diffusion, obs_path, device, strict=False)
            print(f"Loaded obs token diffusion from: {obs_path}")
        main_path = pretrained.get("interaction_vq")
        if main_path:
            load_checkpoint(model, main_path, device, strict=False)
            print(f"Loaded InteractionVQ from: {main_path}")
        # All parameters remain trainable
        return

    # stage == "main_dit"
    cond_vq_path = pretrained.get("cond_vq")
    if cond_vq_path:
        load_checkpoint(model.obs_token_diffusion.tokenizer, cond_vq_path, device, strict=False)
        print(f"Loaded CondVQ tokenizer from: {cond_vq_path}")
    obs_path = pretrained.get("interaction_vq_obs")
    if obs_path:
        load_checkpoint(model.obs_token_diffusion, obs_path, device, strict=False)
        print(f"Loaded obs token diffusion from: {obs_path}")

    model.obs_token_diffusion.freeze_all()


def _build_joint_optimizer(model, cfg):
    """Build optimizer with per-module learning rate groups for joint training."""
    joint_cfg = getattr(cfg, "joint_training", {})
    if not isinstance(joint_cfg, dict):
        joint_cfg = {}
    vq_lr_factor = float(joint_cfg.get("vq_lr_factor", 0.1))
    obs_lr_factor = float(joint_cfg.get("obs_lr_factor", 0.5))

    base_lr = cfg.lr
    if cfg.use_multi_gpu:
        base_lr = base_lr * len(cfg.gpus)

    # Collect parameter id sets for grouping
    vq_param_ids = set(id(p) for p in model.obs_token_diffusion.tokenizer.parameters())
    obs_all_param_ids = set(id(p) for p in model.obs_token_diffusion.parameters())
    obs_only_param_ids = obs_all_param_ids - vq_param_ids

    vq_params = [p for p in model.obs_token_diffusion.tokenizer.parameters() if p.requires_grad]
    obs_params = [p for p in model.obs_token_diffusion.parameters()
                  if p.requires_grad and id(p) in obs_only_param_ids]
    dit_params = [p for p in model.parameters()
                  if p.requires_grad and id(p) not in obs_all_param_ids]

    param_groups = [
        {"params": vq_params, "lr": base_lr * vq_lr_factor},
        {"params": obs_params, "lr": base_lr * obs_lr_factor},
        {"params": dit_params, "lr": base_lr},
    ]

    print(f"Joint optimizer param groups:")
    print(f"  VQ encoder: {sum(p.numel() for p in vq_params)} params, lr={base_lr * vq_lr_factor:.6f}")
    print(f"  Obs denoiser: {sum(p.numel() for p in obs_params)} params, lr={base_lr * obs_lr_factor:.6f}")
    print(f"  Main DiT: {sum(p.numel() for p in dit_params)} params, lr={base_lr:.6f}")

    optimizer = optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.milestones,
        gamma=cfg.gamma,
    )
    return optimizer, scheduler


def main():
    args = get_args()
    cfg = merge_config(args)
    cfg.model_arch = "dit"

    stage = _resolve_stage(cfg, args)
    setup_seed(cfg.seed)
    cfg = setup_device(cfg)

    save_names = {
        "obs_diffusion": "interaction_vq_obs",
        "main_dit": "interaction_vq",
        "joint": "interaction_vq_joint",
    }
    save_dir = create_save_dir(cfg, save_names.get(stage, "interaction_vq"))

    print("=" * 50)
    print(f"InteractionVQ training: stage={stage}")
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
    if stage == "obs_diffusion":
        model = ObsTokenDiffusionModule(cfg).to(device)
    else:
        model = InteractionVQModule(cfg).to(device)
    _load_stage_dependencies(cfg, model, stage, device)

    print(f"model params: {sum(p.numel() for p in model.parameters())}")
    print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    loss_fn = InteractionVQLoss(
        stage=stage,
        weights=getattr(cfg, "loss_weights", {}),
        test_metric_weights=getattr(cfg, "test_metric_weights", {}),
    )
    trainer = InteractionVQTrainer(cfg, model, loss_fn, train_loader, test_loader, stage=stage)

    # Override optimizer for joint training with per-module LR groups
    if stage == "joint":
        raw_model = trainer._unwrap_model(trainer.model)
        trainer.optimizer, trainer.scheduler = _build_joint_optimizer(raw_model, cfg)

    trainer.train()

    if not cfg.debug:
        import yaml

        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dict(cfg), f)


if __name__ == "__main__":
    main()
