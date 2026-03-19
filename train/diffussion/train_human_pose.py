"""
HumanPoseModule独立训练脚本 (Stage 1)
支持--no_trans参数禁用根节点位移预测
"""
from __future__ import annotations

import os
import sys
import torch
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import HumanPoseModule
from train.diffussion.loss.human_pose_loss import HumanPoseLoss
from train.diffussion.train_utils import (
    get_base_args,
    merge_config,
    setup_seed,
    setup_device,
    create_save_dir,
    create_dataloaders,
    build_model_input_dict,
    BaseTrainer,
)


def get_args():
    """获取命令行参数"""
    parser = get_base_args()
    parser.description = 'HumanPoseModule训练 (Stage 1)'
    return parser.parse_args()


class HumanPoseTrainer(BaseTrainer):
    """HumanPoseModule专用训练器"""

    CURRICULUM_K = (3, 5, 10, 30)
    REF_K = 3

    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None):
        super().__init__(cfg, model, loss_fn, train_loader, test_loader)
        self.no_trans = cfg.no_trans
        self.current_rollout_k = None

    @classmethod
    def rollout_k_for_epoch(cls, epoch: int, max_epoch: int) -> int:
        if max_epoch <= 0:
            return int(cls.CURRICULUM_K[-1])
        phase = min((int(epoch) * len(cls.CURRICULUM_K)) // int(max_epoch), len(cls.CURRICULUM_K) - 1)
        return int(cls.CURRICULUM_K[phase])

    def _set_rollout_k(self, epoch: int):
        rollout_k = self.rollout_k_for_epoch(epoch, int(self.cfg.epoch))
        core_model = self._unwrap_model(self.model)
        if hasattr(core_model, "set_train_rollout_k"):
            core_model.set_train_rollout_k(rollout_k)
        if self.current_rollout_k != rollout_k:
            print(f"[HumanPoseTrainer] Epoch {epoch}: rollout_k={rollout_k}")
            self.current_rollout_k = rollout_k

    def train_epoch(self, epoch):
        self._set_rollout_k(epoch)
        self.model.train()
        train_loss = 0.0
        loss_components = {}

        train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in train_iter:
            batch_size = int(batch["human_imu"].shape[0])
            rollout_k = int(self.current_rollout_k if self.current_rollout_k is not None else self.CURRICULUM_K[-1])
            micro_batch_size = self.micro_batch_size_for_rollout(batch_size, rollout_k)

            self.optimizer.zero_grad()

            batch_loss = 0.0
            batch_weighted_losses = {}
            batch_raw_losses = {}

            for start in range(0, batch_size, micro_batch_size):
                end = min(start + micro_batch_size, batch_size)
                chunk_ratio = float(end - start) / float(batch_size)

                chunk_batch = self._slice_batch(batch, start, end, batch_size)
                data_dict = build_model_input_dict(chunk_batch, self.cfg, self.device, add_noise=True)

                pred_dict = self.model_forward(data_dict, batch=chunk_batch)
                total_loss, losses, weighted_losses = self.loss_fn(pred_dict, chunk_batch, self.device)

                self.scaler.scale(total_loss * chunk_ratio).backward()
                batch_loss += total_loss.item() * chunk_ratio

                for key, value in weighted_losses.items():
                    if isinstance(value, torch.Tensor):
                        batch_weighted_losses[key] = batch_weighted_losses.get(key, 0.0) + value.item() * chunk_ratio

                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        batch_raw_losses[key] = batch_raw_losses.get(key, 0.0) + value.item() * chunk_ratio

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_ema()

            train_loss += batch_loss
            for key, value in batch_weighted_losses.items():
                loss_components[key] = loss_components.get(key, 0.0) + value

            postfix = {"loss": batch_loss, "k": rollout_k, "mb": micro_batch_size}
            for key, value in batch_raw_losses.items():
                if value != 0.0:
                    postfix[key] = value
            train_iter.set_postfix(postfix)

            if self.writer is not None:
                self.writer.add_scalar("train/total_loss", batch_loss, self.n_iter)
                self.writer.add_scalar("train/rollout_k", rollout_k, self.n_iter)
                self.writer.add_scalar("train/micro_batch_size", micro_batch_size, self.n_iter)
                for key, value in batch_weighted_losses.items():
                    self.writer.add_scalar(f"train/{key}", value, self.n_iter)

            self.n_iter += 1

        train_loss /= max(len(self.train_loader), 1)
        for key in loss_components:
            loss_components[key] /= max(len(self.train_loader), 1)

        return train_loss, loss_components

    @classmethod
    def micro_batch_size_for_rollout(cls, full_batch_size: int, rollout_k: int) -> int:
        if full_batch_size <= 1:
            return max(int(full_batch_size), 1)
        k = max(int(rollout_k), 1)
        ref = max(int(cls.REF_K), 1)
        # Keep memory roughly stable as K grows: micro-batch is scaled by ref/k.
        micro = (int(full_batch_size) * ref + k - 1) // k
        return max(1, min(int(full_batch_size), int(micro)))

    @classmethod
    def _slice_batch(cls, batch, start: int, end: int, full_batch_size: int):
        def _slice_value(value):
            if isinstance(value, torch.Tensor):
                if value.dim() > 0 and value.shape[0] == full_batch_size:
                    return value[start:end]
                return value
            if isinstance(value, dict):
                return {k: _slice_value(v) for k, v in value.items()}
            if isinstance(value, list) and len(value) == full_batch_size:
                return value[start:end]
            if isinstance(value, tuple) and len(value) == full_batch_size:
                return value[start:end]
            return value

        return {key: _slice_value(val) for key, val in batch.items()}

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
        """模型前向传播（统一由HumanPoseModule内部根据no_trans处理）"""
        gt_arg = batch if use_gt_targets else None
        if force_inference:
            return self.model.inference(
                data_dict,
                gt_targets=batch,
                sample_steps=sample_steps,
                sampler=sampler,
                eta=eta,
            )
        try:
            return self.model(data_dict, gt_targets=gt_arg)
        except TypeError:
            return self.model(data_dict)


def main():
    """主函数"""
    args = get_args()
    cfg = merge_config(args)
    
    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    save_dir = create_save_dir(cfg, 'human_pose')
    
    mode_str = "noTrans" if cfg.no_trans else "普通"
    
    print("=" * 50)
    print(f"Stage 1: HumanPoseModule训练 ({mode_str}模式)")
    print(f"设备: {cfg.device}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"训练轮数: {cfg.epoch}")
    print(f"noTrans模式: {cfg.no_trans}")
    print(f"保存目录: {save_dir}")
    print("=" * 50)
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(cfg)
    
    if train_loader is None or len(train_loader) == 0:
        print("错误: 无法创建训练数据加载器")
        return
    
    # 创建模型
    device = torch.device(cfg.device)
    model = HumanPoseModule(cfg, device, no_trans=cfg.no_trans)
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建损失函数
    loss_weights = getattr(cfg, 'loss_weights', {})
    loss_fn = HumanPoseLoss(weights=loss_weights, no_trans=cfg.no_trans)
    
    # 创建训练器
    trainer = HumanPoseTrainer(cfg, model, loss_fn, train_loader, test_loader)
    
    # 开始训练
    model = trainer.train()
    
    print(f"\n训练完成！模型保存到: {save_dir}")
    
    # 保存配置
    if not cfg.debug:
        import yaml
        config_path = os.path.join(save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(dict(cfg), f)


if __name__ == "__main__":
    main()
