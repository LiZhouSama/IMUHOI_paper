"""
InteractionVQ standalone training script for stage-2 obs diffusion and stage-3 main DiT.
"""
from __future__ import annotations

import os
import sys

import torch

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
        choices=["obs_diffusion", "main_dit"],
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
        return self.model(data_dict, gt_targets=batch)


def _load_stage_dependencies(cfg, model, stage: str, device: torch.device):
    pretrained = getattr(cfg, "pretrained_modules", {}) or {}
    if stage == "obs_diffusion":
        cond_vq_path = pretrained.get("cond_vq")
        if cond_vq_path:
            load_checkpoint(model.tokenizer, cond_vq_path, device, strict=False)
            print(f"Loaded CondVQ tokenizer from: {cond_vq_path}")
        model.freeze_tokenizer()
        return

    cond_vq_path = pretrained.get("cond_vq")
    if cond_vq_path:
        load_checkpoint(model.obs_token_diffusion.tokenizer, cond_vq_path, device, strict=False)
        print(f"Loaded CondVQ tokenizer from: {cond_vq_path}")
    obs_path = pretrained.get("interaction_vq_obs")
    if obs_path:
        load_checkpoint(model.obs_token_diffusion, obs_path, device, strict=False)
        print(f"Loaded obs token diffusion from: {obs_path}")

    model.obs_token_diffusion.freeze_all()


def main():
    args = get_args()
    cfg = merge_config(args)
    cfg.model_arch = "dit"

    stage = _resolve_stage(cfg, args)
    setup_seed(cfg.seed)
    cfg = setup_device(cfg)

    save_name = "interaction_vq_obs" if stage == "obs_diffusion" else "interaction_vq"
    save_dir = create_save_dir(cfg, save_name)

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
    loss_fn = InteractionVQLoss(
        stage=stage,
        weights=getattr(cfg, "loss_weights", {}),
        test_metric_weights=getattr(cfg, "test_metric_weights", {}),
    )
    trainer = InteractionVQTrainer(cfg, model, loss_fn, train_loader, test_loader, stage=stage)
    trainer.train()

    if not cfg.debug:
        import yaml

        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dict(cfg), f)


if __name__ == "__main__":
    main()
