"""
VelocityContactModule独立训练脚本 (Stage 1)
可与HumanPoseModule同时训练
"""
import os
import sys
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import VelocityContactModule
from train.loss.velocity_contact_loss import VelocityContactLoss
from train.train_utils import (
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
    parser.description = 'VelocityContactModule训练 (Stage 1)'
    parser.add_argument('--hp_ckpt', type=str, default=None,
                        help='HumanPoseModule权重路径（便于分阶段训练时先加载HP模型）')
    return parser.parse_args()


class VelocityContactTrainer(BaseTrainer):
    """VelocityContactModule专用训练器"""
    
    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None):
        super().__init__(cfg, model, loss_fn, train_loader, test_loader)
    
    def model_forward(self, data_dict):
        """模型前向传播"""
        return self.model(data_dict)


def main():
    """主函数"""
    args = get_args()
    cfg = merge_config(args)
    cfg.hp_ckpt = getattr(args, "hp_ckpt", None)
    
    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    save_dir = create_save_dir(cfg, 'velocity_contact')
    
    print("=" * 50)
    print("Stage 1: VelocityContactModule训练")
    print(f"设备: {cfg.device}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"训练轮数: {cfg.epoch}")
    print(f"学习率: {cfg.lr}")
    if getattr(cfg, "pretrained_ckpt", None):
        print(f"预训练权重: {cfg.pretrained_ckpt}")
    print(f"保存目录: {save_dir}")
    print("=" * 50)
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(cfg)
    
    if train_loader is None or len(train_loader) == 0:
        print("错误: 无法创建训练数据加载器")
        return
    
    # 创建模型
    model = VelocityContactModule(cfg)
    model = model.to(cfg.device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建损失函数
    loss_weights = getattr(cfg, 'loss_weights', {})
    loss_fn = VelocityContactLoss(weights=loss_weights)
    
    # 创建训练器
    trainer = VelocityContactTrainer(cfg, model, loss_fn, train_loader, test_loader)
    
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
