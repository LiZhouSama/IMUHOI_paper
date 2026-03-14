"""
Unified interaction training script (DiT path).

- Stage A: freeze HP, train interaction module with HP predictions (detached)
- Stage B: unfreeze HP, joint fine-tuning HP + interaction
"""
from __future__ import annotations

import os
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import IMUHOIModel
from train.diffussion.loss.human_pose_loss import HumanPoseLoss
from train.diffussion.loss.interaction_loss import InteractionLoss
from train.diffussion.train_utils import (
    BaseTrainer,
    build_model_input_dict,
    create_dataloaders,
    create_save_dir,
    get_base_args,
    merge_config,
    setup_device,
    setup_seed,
)


def get_args():
    parser = get_base_args()
    parser.description = "InteractionModule training (merged VC+OT, DiT)"
    parser.add_argument("--hp_ckpt", type=str, default=None, help="Path to pretrained HumanPose module checkpoint")
    parser.add_argument("--interaction_ckpt", type=str, default=None, help="Path to pretrained Interaction module checkpoint")
    parser.add_argument("--vc_ckpt", type=str, default=None, help="Legacy VC checkpoint path (mapped to interaction)")
    parser.add_argument("--ot_ckpt", type=str, default=None, help="Legacy OT checkpoint path (mapped to interaction)")
    parser.add_argument("--freeze_ratio", type=float, default=None, help="Freeze HP phase ratio in [0,1]")
    parser.add_argument("--joint_phase_lr_factor", type=float, default=None, help="LR multiplier in joint fine-tuning phase")
    parser.add_argument("--hp_aux_weight", type=float, default=None, help="Optional HP auxiliary loss weight in joint phase")
    return parser.parse_args()


class InteractionTrainer(BaseTrainer):
    """Trainer with 80/20 HP-freeze scheduling."""

    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None):
        super().__init__(cfg, model, loss_fn, train_loader, test_loader)

        inter_cfg = getattr(cfg, "interaction_training", {})

        def _inter_cfg(name, default):
            if isinstance(inter_cfg, dict) and name in inter_cfg:
                return inter_cfg[name]
            return getattr(cfg, name, default)

        self.freeze_ratio = float(_inter_cfg("freeze_ratio", 0.8))
        self.freeze_ratio = min(max(self.freeze_ratio, 0.0), 1.0)
        self.freeze_epochs = int(round(self.cfg.epoch * self.freeze_ratio))
        self.freeze_epochs = min(max(self.freeze_epochs, 0), self.cfg.epoch)

        self.joint_phase_lr_factor = float(_inter_cfg("joint_phase_lr_factor", 0.2))
        self.hp_aux_weight = float(_inter_cfg("hp_aux_weight", 0.0))
        self.hp_loss_fn = HumanPoseLoss(weights=getattr(cfg, "loss_weights", {}), no_trans=cfg.no_trans)

        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.phase = None
        self._set_phase(0)

    @property
    def core_model(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def _set_phase(self, epoch: int):
        new_phase = "freeze" if epoch < self.freeze_epochs else "joint"
        if new_phase == self.phase:
            return

        self.phase = new_phase
        hp_module = self.core_model.human_pose_module

        if new_phase == "freeze":
            for p in hp_module.parameters():
                p.requires_grad_(False)
            hp_module.eval()
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr
            print(
                f"[InteractionTrainer] Phase=freeze (epoch<{self.freeze_epochs}), "
                f"HP frozen, lr={[g['lr'] for g in self.optimizer.param_groups]}"
            )
        else:
            for p in hp_module.parameters():
                p.requires_grad_(True)
            hp_module.train()
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * self.joint_phase_lr_factor
            print(
                f"[InteractionTrainer] Phase=joint (epoch>={self.freeze_epochs}), "
                f"HP unfrozen, lr={[g['lr'] for g in self.optimizer.param_groups]}"
            )

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
                use_object_data=True,
                compute_fk=False,
                sample_steps=sample_steps,
                sampler=sampler,
                eta=eta,
            )

        return self.model(
            data_dict,
            use_object_data=True,
            compute_fk=False,
            gt_targets=gt_arg,
            detach_hp=(self.phase == "freeze"),
            sample_steps=sample_steps,
            sampler=sampler,
            eta=eta,
        )

    def train_epoch(self, epoch):
        self._set_phase(epoch)

        self.model.train()
        if self.phase == "freeze":
            # Ensure HP stays in eval mode during freeze while whole model is train().
            self.core_model.human_pose_module.eval()

        total_loss_acc = 0.0
        loss_components = {}
        train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch} [{self.phase}]", leave=False)

        for batch in train_iter:
            data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)

            self.optimizer.zero_grad()
            pred_dict = self.model_forward(data_dict, batch=batch)

            inter_total, inter_losses, inter_weighted = self.loss_fn(pred_dict, batch, self.device)

            total_loss = inter_total
            hp_losses = {}
            if self.phase == "joint" and self.hp_aux_weight > 0.0:
                hp_total, hp_losses, _ = self.hp_loss_fn(pred_dict, batch, self.device)
                total_loss = total_loss + hp_total * self.hp_aux_weight

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_ema()

            total_loss_acc += total_loss.item()

            for key, value in inter_weighted.items():
                if isinstance(value, torch.Tensor):
                    loss_components[key] = loss_components.get(key, 0.0) + value.item()

            if hp_losses:
                for key, value in hp_losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[f"hp_{key}"] = loss_components.get(f"hp_{key}", 0.0) + value.item()

            postfix = {
                "loss": total_loss.item(),
                "obj": inter_losses.get("obj_trans", torch.tensor(0.0, device=self.device)).item(),
                "contact": inter_losses.get("contact_logits", torch.tensor(0.0, device=self.device)).item(),
            }
            train_iter.set_postfix(postfix)

            if self.writer is not None:
                self.writer.add_scalar("train/total_loss", total_loss.item(), self.n_iter)
                for key, value in inter_weighted.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f"train/{key}", value.item(), self.n_iter)
                if hp_losses:
                    for key, value in hp_losses.items():
                        if isinstance(value, torch.Tensor):
                            self.writer.add_scalar(f"train/hp_{key}", value.item(), self.n_iter)

            self.n_iter += 1

        total_loss_acc /= max(len(self.train_loader), 1)
        for key in loss_components:
            loss_components[key] /= max(len(self.train_loader), 1)

        return total_loss_acc, loss_components


def main():
    args = get_args()
    cfg = merge_config(args)

    # Interaction training is DiT-only.
    cfg.model_arch = "dit"

    if args.freeze_ratio is not None:
        cfg.freeze_ratio = float(args.freeze_ratio)
    if args.joint_phase_lr_factor is not None:
        cfg.joint_phase_lr_factor = float(args.joint_phase_lr_factor)
    if args.hp_aux_weight is not None:
        cfg.hp_aux_weight = float(args.hp_aux_weight)

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    save_dir = create_save_dir(cfg, "interaction")

    print("=" * 50)
    print("Interaction training (merged VC+OT)")
    print(f"Device: {cfg.device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epoch}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Save dir: {save_dir}")
    print("=" * 50)

    train_loader, test_loader = create_dataloaders(cfg)
    if train_loader is None or len(train_loader) == 0:
        print("Error: failed to create train dataloader")
        return

    device = torch.device(cfg.device)
    model = IMUHOIModel(cfg, device, no_trans=cfg.no_trans).to(device)

    module_paths = {}
    if args.hp_ckpt:
        module_paths["human_pose"] = args.hp_ckpt
    if args.interaction_ckpt:
        module_paths["interaction"] = args.interaction_ckpt
    if args.vc_ckpt:
        module_paths["velocity_contact"] = args.vc_ckpt
    if args.ot_ckpt:
        module_paths["object_trans"] = args.ot_ckpt

    if module_paths:
        model.load_pretrained_modules(module_paths, strict=False)

    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    loss_weights = getattr(cfg, "loss_weights", {})
    loss_fn = InteractionLoss(weights=loss_weights)

    trainer = InteractionTrainer(cfg, model, loss_fn, train_loader, test_loader)
    trainer.train()

    print(f"\nTraining complete! Models saved to: {save_dir}")

    if not cfg.debug:
        config_path = os.path.join(save_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml_ready = dict(cfg)
            yaml_ready["hp_ckpt"] = args.hp_ckpt
            yaml_ready["interaction_ckpt"] = args.interaction_ckpt
            yaml_ready["vc_ckpt"] = args.vc_ckpt
            yaml_ready["ot_ckpt"] = args.ot_ckpt
            import yaml

            yaml.dump(yaml_ready, f)


if __name__ == "__main__":
    main()
