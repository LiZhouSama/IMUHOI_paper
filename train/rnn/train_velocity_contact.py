"""
VelocityContactModule standalone training (Stage 2).
Freeze a pretrained HumanPoseModule (Stage 1) to supply hp_out for VC training.
"""
from __future__ import annotations
import os
import sys
import torch
from tqdm import tqdm

# add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import VelocityContactModule, HumanPoseModule
from train.rnn.loss.velocity_contact_loss import VelocityContactLoss
from torch.nn.utils import clip_grad_norm_
from train.rnn.train_utils import (
    get_base_args,
    merge_config,
    setup_seed,
    setup_device,
    create_save_dir,
    create_dataloaders,
    build_model_input_dict,
    save_config_snapshot,
    BaseTrainer,
    load_checkpoint,
    call_model_inference,
)
from train.rnn.scheduled_inputs import (
    build_gt_human_pose_outputs,
    prediction_mix_probability,
    sample_mix_dict,
)


def _cfg_get(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _resolve_hp_ckpt(cfg, cli_hp_ckpt=None):
    """Prefer explicit CLI, then cfg.hp_ckpt, then pretrained_modules.human_pose."""
    if cli_hp_ckpt:
        return cli_hp_ckpt
    cfg_hp_ckpt = getattr(cfg, "hp_ckpt", None)
    if cfg_hp_ckpt:
        return cfg_hp_ckpt
    pretrained_modules = getattr(cfg, "pretrained_modules", None)
    return _cfg_get(pretrained_modules, "human_pose")


def get_args():
    parser = get_base_args()
    parser.description = 'VelocityContactModule training (Stage 2)'
    parser.add_argument(
        '--hp_ckpt',
        type=str,
        default=None,
        help='Path to pretrained HumanPoseModule (frozen to provide hp_out).',
    )
    parser.add_argument(
        '--ablate_vc_boundary',
        action='store_true',
        default=None,
        help='Train VelocityContact with boundary logits/prob/hidden states forced to zero.',
    )
    return parser.parse_args()


class VelocityContactTrainer(BaseTrainer):
    """Trainer for VelocityContactModule."""

    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None, hp_model=None):
        super().__init__(cfg, model, loss_fn, train_loader, test_loader)
        self.hp_model = hp_model
        self.current_epoch = 0
        if self.hp_model is not None:
            self.hp_model.eval()
            for p in self.hp_model.parameters():
                p.requires_grad_(False)

    def model_forward(
        self,
        data_dict,
        batch=None,
        epoch=None,
        training=None,
    ):
        """Use frozen HP/GT scheduled hp_out, then forward VC."""
        training = self.model.training if training is None else bool(training)
        epoch = self.current_epoch if epoch is None else int(epoch)
        pred_hp_out = None
        if self.hp_model is not None:
            with torch.no_grad():
                pred_hp_out = call_model_inference(self.hp_model, data_dict, inference_mode="offline")

        hp_out = pred_hp_out
        if batch is not None:
            gt_hp_out = build_gt_human_pose_outputs(
                batch,
                self.device,
                dtype=data_dict['human_imu'].dtype,
            )
            if pred_hp_out is None:
                hp_out = gt_hp_out
            elif training:
                hp_out = sample_mix_dict(
                    gt_hp_out,
                    pred_hp_out,
                    (
                        "p_pred",
                        "pred_full_pose_6d",
                        "pred_joints_local",
                        "pred_joints_global",
                        "pred_hand_glb_pos",
                        "root_vel_pred",
                        "root_trans_pred",
                    ),
                    prediction_mix_probability(epoch, self.cfg),
                )
        return call_model_inference(self.model, data_dict, hp_out=hp_out, inference_mode="offline")

    def train_epoch(self, epoch):
        """Override to add gradient clipping for stability."""
        self.current_epoch = epoch
        self.model.train()
        total_loss = 0
        loss_components = {}

        train_iter = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)

        for batch in train_iter:
            data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)

            self.optimizer.zero_grad()
            pred_dict = self.model_forward(data_dict, batch=batch, epoch=epoch, training=True)

            total_loss_tensor, losses, weighted_losses = self.loss_fn(pred_dict, batch, self.device)

            self.scaler.scale(total_loss_tensor).backward()
            # gradient clipping
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_val = total_loss_tensor.item()
            total_loss += total_loss_val
            for key, value in weighted_losses.items():
                if isinstance(value, torch.Tensor):
                    loss_components[key] = loss_components.get(key, 0) + value.item()

            postfix = {'loss': total_loss_val}
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.item() != 0.0:
                    postfix[key] = value.item()
            train_iter.set_postfix(postfix)

            if self.writer is not None:
                self.writer.add_scalar('train/total_loss', total_loss_val, self.n_iter)
                for key, value in weighted_losses.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'train/{key}', value.item(), self.n_iter)

            self.n_iter += 1

        total_loss /= len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)

        return total_loss, loss_components


def main():
    args = get_args()
    cfg = merge_config(args)
    cfg.hp_ckpt = _resolve_hp_ckpt(cfg, getattr(args, "hp_ckpt", None))
    cfg.ablate_vc_boundary = (
        bool(args.ablate_vc_boundary)
        if args.ablate_vc_boundary is not None
        else bool(getattr(cfg, "ablate_vc_boundary", False))
    )

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    module_name = 'velocity_contact_vc_boundary_zero' if cfg.ablate_vc_boundary else 'velocity_contact'
    save_dir = create_save_dir(cfg, module_name)
    save_config_snapshot(cfg)

    print("=" * 50)
    print("Stage 2: VelocityContactModule training")
    print(f"Device: {cfg.device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Learning rate: {cfg.lr}")
    if getattr(cfg, "pretrained_ckpt", None):
        print(f"Pretrained: {cfg.pretrained_ckpt}")
    if getattr(cfg, "hp_ckpt", None):
        print(f"HumanPose checkpoint: {cfg.hp_ckpt}")
    print(f"VC boundary ablation: {'enabled' if cfg.ablate_vc_boundary else 'disabled'}")
    print(f"Save dir: {save_dir}")
    print("=" * 50)

    # dataloaders
    train_loader, test_loader = create_dataloaders(cfg)
    if train_loader is None or len(train_loader) == 0:
        print("Error: failed to create train dataloader")
        return

    # VC model
    model = VelocityContactModule(cfg)
    model = model.to(cfg.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # frozen HP model (Stage 1) for hp_out
    hp_model = None
    if cfg.hp_ckpt:
        device = torch.device(cfg.device)
        try:
            hp_model = HumanPoseModule(cfg, device, no_trans=cfg.no_trans)
            if os.path.exists(cfg.hp_ckpt):
                load_checkpoint(hp_model, cfg.hp_ckpt, device, strict=False)
                print(f"Loaded HumanPose checkpoint: {cfg.hp_ckpt}")
            else:
                print(f"Warning: HumanPose checkpoint not found at {cfg.hp_ckpt}, fallback to GT hp_out.")
            hp_model = hp_model.to(device)
        except Exception as exc:
            print(f"Warning: failed to init/load HumanPose; fallback to GT hp_out. {exc}")
            hp_model = None

    # loss
    loss_weights = getattr(cfg, 'loss_weights', {})
    loss_fn = VelocityContactLoss(weights=loss_weights)

    # trainer
    trainer = VelocityContactTrainer(cfg, model, loss_fn, train_loader, test_loader, hp_model=hp_model)

    # train
    model = trainer.train()

    print(f"\nTraining complete! Models saved to: {save_dir}")


if __name__ == "__main__":
    main()
