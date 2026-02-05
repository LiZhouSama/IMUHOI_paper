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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    BaseTrainer,
    load_checkpoint,
)


def get_args():
    parser = get_base_args()
    parser.description = 'VelocityContactModule training (Stage 2)'
    parser.add_argument(
        '--hp_ckpt',
        type=str,
        default=None,
        help='Path to pretrained HumanPoseModule (frozen to provide hp_out).',
    )
    return parser.parse_args()


class VelocityContactTrainer(BaseTrainer):
    """Trainer for VelocityContactModule."""

    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None, hp_model=None):
        super().__init__(cfg, model, loss_fn, train_loader, test_loader)
        self.hp_model = hp_model
        if self.hp_model is not None:
            self.hp_model.eval()
            for p in self.hp_model.parameters():
                p.requires_grad_(False)

    def model_forward(
        self,
        data_dict,
        batch=None,
    ):
        """Use frozen HP to produce hp_out, then forward VC."""
        hp_out = None
        if self.hp_model is not None:
            with torch.no_grad():
                hp_out = self.hp_model(data_dict)
        return self.model(data_dict, hp_out=hp_out)

    def train_epoch(self, epoch):
        """Override to add gradient clipping for stability."""
        self.model.train()
        total_loss = 0
        loss_components = {}

        train_iter = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)

        for batch in train_iter:
            data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)

            self.optimizer.zero_grad()
            pred_dict = self.model_forward(data_dict, batch=batch)

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
    cfg.hp_ckpt = getattr(args, "hp_ckpt", None)

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    save_dir = create_save_dir(cfg, 'velocity_contact')

    print("=" * 50)
    print("Stage 2: VelocityContactModule training")
    print(f"Device: {cfg.device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epoch}")
    print(f"Learning rate: {cfg.lr}")
    if getattr(cfg, "pretrained_ckpt", None):
        print(f"Pretrained: {cfg.pretrained_ckpt}")
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
                print(f"Warning: HumanPose checkpoint not found at {cfg.hp_ckpt}, hp_out will be zeros.")
            hp_model = hp_model.to(device)
        except Exception as exc:
            print(f"Warning: failed to init/load HumanPose; hp_out will be zeros. {exc}")
            hp_model = None

    # loss
    loss_weights = getattr(cfg, 'loss_weights', {})
    loss_fn = VelocityContactLoss(weights=loss_weights)

    # trainer
    trainer = VelocityContactTrainer(cfg, model, loss_fn, train_loader, test_loader, hp_model=hp_model)

    # train
    model = trainer.train()

    print(f"\nTraining complete! Models saved to: {save_dir}")

    # save config
    if not cfg.debug:
        import yaml

        config_path = os.path.join(save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(dict(cfg), f)


if __name__ == "__main__":
    main()
