"""
ObjectTransModule训练脚本 (Stage 3) + 联合训练
依赖Stage1和Stage2的预训练权重

用法:
    # 仅训练Stage3 (Object Trans)
    python train_object_trans.py --cfg config.yaml --vc_ckpt path/to/vc.pt --hp_ckpt path/to/hp.pt
    
    # 联合训练所有模块
    python train_object_trans.py --cfg config.yaml --vc_ckpt path/to/vc.pt --hp_ckpt path/to/hp.pt --joint_train
"""
import os
import sys
import glob
import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import VelocityContactModule, HumanPoseModule, ObjectTransModule
from train.loss import VelocityContactLoss, HumanPoseLoss, ObjectTransLoss
from train.train_utils import (
    get_base_args,
    merge_config,
    setup_seed,
    setup_device,
    create_save_dir,
    create_dataloaders,
    build_model_input_dict,
    flatten_lstm_parameters,
    save_checkpoint,
    load_checkpoint,
)
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter


def get_args():
    """获取命令行参数"""
    parser = get_base_args()
    parser.description = 'ObjectTransModule训练 (Stage 3) + 联合训练'
    
    # Stage3特有参数
    parser.add_argument('--vc_ckpt', type=str, default=None, 
                        help='VelocityContactModule权重路径')
    parser.add_argument('--hp_ckpt', type=str, default=None, 
                        help='HumanPoseModule权重路径')
    parser.add_argument('--ot_ckpt', type=str, default=None,
                        help='ObjectTransModule预训练权重路径（联合微调起点）')
    parser.add_argument('--joint_train', action='store_true', 
                        help='是否进行联合训练（微调所有模块）')
    
    return parser.parse_args()


def find_latest_checkpoint(base_dir, module_name):
    """自动搜索最新的检查点文件"""
    patterns = [
        os.path.join(base_dir, f'{module_name}*', 'best.pt'),
        # os.path.join(base_dir, f'{module_name}*', 'final.pt'),
        # os.path.join(base_dir, 'modules', f'{module_name}_best.pt'),
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        return None
    
    # 按修改时间排序，返回最新的
    all_files.sort(key=os.path.getmtime, reverse=True)
    return all_files[0]


class Stage3JointTrainer:
    """Stage3和联合训练器"""
    
    def __init__(self, cfg, vc_model, hp_model, ot_model, train_loader, test_loader=None, joint_train=False):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.joint_train = joint_train
        
        self.vc_model = vc_model.to(self.device)
        self.hp_model = hp_model.to(self.device)
        self.ot_model = ot_model.to(self.device)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # 设置训练/冻结状态
        if joint_train:
            # 联合微调：VC 仍然冻结，仅微调 HP + OT
            print("联合训练模式：冻结VelocityContact，微调HumanPose + ObjectTrans")
            for param in self.vc_model.parameters():
                param.requires_grad = False
        else:
            # Stage3：冻结VC和HP，只训练OT
            print("Stage3模式：冻结VelocityContact和HumanPose，只训练ObjectTrans")
            for param in self.vc_model.parameters():
                param.requires_grad = False
            for param in self.hp_model.parameters():
                param.requires_grad = False
        
        # 多GPU包装
        if cfg.use_multi_gpu:
            print(f'多GPU训练: {cfg.gpus}')
            self.vc_model = torch.nn.DataParallel(self.vc_model, device_ids=cfg.gpus)
            self.hp_model = torch.nn.DataParallel(self.hp_model, device_ids=cfg.gpus)
            self.ot_model = torch.nn.DataParallel(self.ot_model, device_ids=cfg.gpus)
        
        flatten_lstm_parameters(self.vc_model)
        flatten_lstm_parameters(self.hp_model)
        flatten_lstm_parameters(self.ot_model)
        
        # 收集可训练参数
        trainable_params = []
        if joint_train:
            trainable_params.extend(self.hp_model.parameters())
        trainable_params.extend(self.ot_model.parameters())
        trainable_params = [p for p in trainable_params if p.requires_grad]
        
        # 优化器
        lr = cfg.lr
        if joint_train:
            lr_factor = getattr(cfg, "joint_lr_factor", 0.1)
            lr = lr * lr_factor
        if cfg.use_multi_gpu:
            lr = lr * len(cfg.gpus)
        
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=cfg.milestones, gamma=cfg.gamma
        )
        self.scaler = GradScaler()
        
        print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
        
        # 损失函数
        loss_weights = getattr(cfg, 'loss_weights', {})
        self.vc_loss_fn = None  # VC冻结，不参与训练
        self.hp_loss_fn = HumanPoseLoss(weights=loss_weights, no_trans=cfg.no_trans) if joint_train else None
        self.ot_loss_fn = ObjectTransLoss(weights=loss_weights)
        
        # TensorBoard
        self.writer = None
        if cfg.use_tensorboard and not cfg.debug:
            log_dir = os.path.join(cfg.save_dir, 'tensorboard_logs')
            self.writer = SummaryWriter(log_dir=log_dir)
        
        self.best_loss = float('inf')
        self.n_iter = 0
    
    def forward_all(self, data_dict):
        """完整的前向传播"""
        # Stage 1: VelocityContact
        vc_out = self.vc_model(data_dict)
        
        # Stage 2: HumanPose
        hp_out = self.hp_model(data_dict)
        
        # Stage 3: ObjectTrans
        ot_out = self.ot_model(
            hp_out['pred_hand_glb_pos'],
            vc_out['pred_hand_contact_prob'],
            data_dict['obj_trans_init'],
            obj_imu=data_dict['obj_imu'],
            human_imu=data_dict['human_imu'],
            obj_vel_input=vc_out['pred_obj_vel'],
            contact_init=data_dict['contact_init'],
            has_object_mask=data_dict['has_object'],
        )
        
        # 合并输出
        results = {}
        results.update(vc_out)
        results.update(hp_out)
        results.update(ot_out)
        
        return results, vc_out, hp_out, ot_out
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.vc_model.eval()
        self.hp_model.train() if self.joint_train else self.hp_model.eval()
        self.ot_model.train()
        
        total_loss = 0
        loss_components = {}
        
        train_iter = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch in train_iter:
            data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)
            
            self.optimizer.zero_grad()
            
            results, vc_out, hp_out, ot_out = self.forward_all(data_dict)
            
            # 计算损失
            losses = {}
            
            if self.joint_train and self.hp_loss_fn:
                hp_loss, hp_losses, _ = self.hp_loss_fn(hp_out, batch, self.device)
                losses.update({f'hp_{k}': v for k, v in hp_losses.items()})
            else:
                hp_loss = torch.tensor(0.0, device=self.device)
            
            ot_loss, ot_losses, _ = self.ot_loss_fn(ot_out, batch, self.device)
            losses.update({f'ot_{k}': v for k, v in ot_losses.items()})
            
            # 总损失
            batch_loss = hp_loss + ot_loss
            
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += batch_loss.item()
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    loss_components[key] = loss_components.get(key, 0) + value.item()
            
            # 更新进度条
            postfix = {'loss': batch_loss.item(), 'ot': ot_loss.item()}
            if self.joint_train:
                postfix['hp'] = hp_loss.item()
            train_iter.set_postfix(postfix)
            
            if self.writer is not None:
                self.writer.add_scalar('train/total_loss', batch_loss.item(), self.n_iter)
                self.writer.add_scalar('train/ot_loss', ot_loss.item(), self.n_iter)
                if self.joint_train:
                    self.writer.add_scalar('train/hp_loss', hp_loss.item(), self.n_iter)
            
            self.n_iter += 1
        
        total_loss /= len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return total_loss, loss_components
    
    def evaluate(self, epoch):
        """评估"""
        if self.test_loader is None:
            return None, {}
        
        self.vc_model.eval()
        self.hp_model.eval()
        self.ot_model.eval()
        
        total_loss = 0
        loss_components = {}
        
        with torch.no_grad():
            test_iter = tqdm(self.test_loader, desc=f'Test {epoch}', leave=False)
            
            for batch in test_iter:
                data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=False)
                
                results, vc_out, hp_out, ot_out = self.forward_all(data_dict)
                
                # 计算测试损失
                hp_loss = torch.tensor(0.0, device=self.device)
                hp_losses = {}
                if self.joint_train and self.hp_loss_fn:
                    if hasattr(self.hp_loss_fn, "compute_test_loss"):
                        hp_loss, hp_losses = self.hp_loss_fn.compute_test_loss(hp_out, batch, self.device)
                    else:
                        hp_loss, hp_losses, _ = self.hp_loss_fn(hp_out, batch, self.device)

                ot_loss, ot_losses = self.ot_loss_fn.compute_test_loss(ot_out, batch, self.device)

                combined_loss = ot_loss + hp_loss
                total_loss += combined_loss.item()
                for key, value in ot_losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[key] = loss_components.get(key, 0) + value.item()
                for key, value in hp_losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[f"hp_{key}"] = loss_components.get(f"hp_{key}", 0) + value.item()
        
        total_loss /= len(self.test_loader)
        for key in loss_components:
            loss_components[key] /= len(self.test_loader)
        
        return total_loss, loss_components
    
    def train(self):
        """完整训练循环"""
        max_epoch = self.cfg.epoch
        
        for epoch in range(max_epoch):
            train_loss, train_components = self.train_epoch(epoch)
            
            print(f'\rEpoch {epoch}, Train Loss: {train_loss:.4f}', end='')
            
            if epoch % 10 == 0 and self.test_loader is not None:
                test_loss, test_components = self.evaluate(epoch)
                
                if test_loss is not None:
                    print(f', Test Loss: {test_loss:.4f}')
                    
                    if test_loss < self.best_loss:
                        self.best_loss = test_loss
                        self._save_models(epoch, test_loss, 'best')
                        print(f'新的最佳测试损失: {self.best_loss:.4f}')
                    
                    if self.writer is not None:
                        self.writer.add_scalar('test/total_loss', test_loss, self.n_iter)
            else:
                print()
            
            self.scheduler.step()
        
        # 保存最终模型
        self._save_models(max_epoch - 1, train_loss, 'final')
        
        if self.writer is not None:
            self.writer.close()
    
    def _save_models(self, epoch, loss, prefix):
        """保存模型"""
        # 保存ObjectTrans模块
        ot_path = os.path.join(self.cfg.save_dir, f'{prefix}_object_trans.pt')
        ot_state = self.ot_model.module.state_dict() if isinstance(self.ot_model, torch.nn.DataParallel) else self.ot_model.state_dict()
        torch.save({
            'epoch': epoch,
            'module_state_dict': ot_state,
            'loss': loss,
        }, ot_path)
        
        if self.joint_train:
            # 联合训练时保存所有模块
            vc_path = os.path.join(self.cfg.save_dir, f'{prefix}_velocity_contact.pt')
            vc_state = self.vc_model.module.state_dict() if isinstance(self.vc_model, torch.nn.DataParallel) else self.vc_model.state_dict()
            torch.save({
                'epoch': epoch,
                'module_state_dict': vc_state,
                'loss': loss,
            }, vc_path)
            
            hp_path = os.path.join(self.cfg.save_dir, f'{prefix}_human_pose.pt')
            hp_state = self.hp_model.module.state_dict() if isinstance(self.hp_model, torch.nn.DataParallel) else self.hp_model.state_dict()
            torch.save({
                'epoch': epoch,
                'module_state_dict': hp_state,
                'loss': loss,
                'no_trans': self.cfg.no_trans,
            }, hp_path)
        
        print(f'保存模型: {prefix}')


def main():
    """主函数"""
    args = get_args()
    cfg = merge_config(args)
    
    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    
    module_name = 'joint_train' if args.joint_train else 'object_trans'
    save_dir = create_save_dir(cfg, module_name)
    
    mode_str = "联合训练" if args.joint_train else "Stage 3"
    trans_str = "noTrans" if cfg.no_trans else "普通"
    
    print("=" * 50)
    print(f"{mode_str}: ObjectTransModule训练 ({trans_str}模式)")
    print(f"设备: {cfg.device}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"训练轮数: {cfg.epoch}")
    print(f"noTrans模式: {cfg.no_trans}")
    print(f"保存目录: {save_dir}")
    print("=" * 50)
    
    # 查找/加载预训练权重
    vc_ckpt = args.vc_ckpt
    hp_ckpt = args.hp_ckpt
    ot_ckpt = args.ot_ckpt

    base_save_dir = os.path.dirname(save_dir)
    if vc_ckpt is None:
        vc_ckpt = find_latest_checkpoint(base_save_dir, 'velocity_contact')
        print(f"自动找到VelocityContact权重: {vc_ckpt}")
    if hp_ckpt is None:
        hp_ckpt = find_latest_checkpoint(base_save_dir, 'human_pose')
        print(f"自动找到HumanPose权重: {hp_ckpt}")
    if ot_ckpt is None:
        ot_ckpt = find_latest_checkpoint(base_save_dir, 'object_trans')
        if args.joint_train:
            print(f"自动找到ObjectTrans预训练权重(用于联合微调): {ot_ckpt}")
    
    if vc_ckpt is None or hp_ckpt is None:
        print("警告: 未找到预训练权重，将使用随机初始化")
        print("建议先训练Stage1和Stage2，或使用--auto_find_ckpt自动搜索")
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(cfg)
    
    if train_loader is None or len(train_loader) == 0:
        print("错误: 无法创建训练数据加载器")
        return
    
    # 创建模型
    device = torch.device(cfg.device)
    
    vc_model = VelocityContactModule(cfg)
    if vc_ckpt and os.path.exists(vc_ckpt):
        load_checkpoint(vc_model, vc_ckpt, device)
    
    hp_model = HumanPoseModule(cfg, device, no_trans=cfg.no_trans)
    if hp_ckpt and os.path.exists(hp_ckpt):
        load_checkpoint(hp_model, hp_ckpt, device)
    
    ot_model = ObjectTransModule(cfg)
    if args.joint_train:
        if ot_ckpt and os.path.exists(ot_ckpt):
            load_checkpoint(ot_model, ot_ckpt, device)
            print(f"联合微调起点加载ObjectTrans权重: {ot_ckpt}")
        else:
            raise ValueError("联合微调需要已训练的ObjectTrans权重，请先进行单独Stage3训练或指定 --ot_ckpt")
    
    print(f"VelocityContact参数量: {sum(p.numel() for p in vc_model.parameters())}")
    print(f"HumanPose参数量: {sum(p.numel() for p in hp_model.parameters())}")
    print(f"ObjectTrans参数量: {sum(p.numel() for p in ot_model.parameters())}")
    
    # 创建训练器
    trainer = Stage3JointTrainer(
        cfg, vc_model, hp_model, ot_model,
        train_loader, test_loader,
        joint_train=args.joint_train
    )
    
    # 开始训练
    trainer.train()
    
    print(f"\n训练完成！模型保存到: {save_dir}")
    
    # 保存配置
    if not cfg.debug:
        import yaml
        config_path = os.path.join(save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            cfg_dict = dict(cfg)
            cfg_dict['vc_ckpt'] = vc_ckpt
            cfg_dict['hp_ckpt'] = hp_ckpt
            cfg_dict['joint_train'] = args.joint_train
            yaml.dump(cfg_dict, f)


if __name__ == "__main__":
    main()
