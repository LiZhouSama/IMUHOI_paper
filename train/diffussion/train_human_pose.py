"""
HumanPoseModule独立训练脚本 (Stage 1)
可与VelocityContactModule同时训练
支持--no_trans参数禁用根节点位移预测
"""
from __future__ import annotations

import os
import sys
import torch

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
    BaseTrainer,
)


def get_args():
    """获取命令行参数"""
    parser = get_base_args()
    parser.description = 'HumanPoseModule训练 (Stage 1)'
    return parser.parse_args()


class HumanPoseTrainer(BaseTrainer):
    """HumanPoseModule专用训练器"""
    
    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None):
        super().__init__(cfg, model, loss_fn, train_loader, test_loader)
        self.no_trans = cfg.no_trans
    
    def model_forward(
        self,
        data_dict,
        batch=None,
        use_gt_targets: bool = True,
        force_inference: bool = False,
        sample_steps: int | None = None,
    ):
        """模型前向传播（统一由HumanPoseModule内部根据no_trans处理）"""
        gt_arg = batch if use_gt_targets else None
        if force_inference:
            return self.model.inference(data_dict, sample_steps=sample_steps)
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
