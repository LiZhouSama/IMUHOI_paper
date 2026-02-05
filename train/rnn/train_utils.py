"""
训练专用工具函数
"""
from __future__ import annotations
import os
import argparse
import yaml
from easydict import EasyDict as edict
from datetime import datetime
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

# 从 utils 导入共用函数
from utils.utils import (
    setup_seed,
    setup_device,
    load_checkpoint,
    save_checkpoint,
    flatten_lstm_parameters,
    build_model_input_dict,
)
from dataset.dataset_IMUHOI import IMUDataset

# 重新导出以保持兼容性
__all__ = [
    'get_base_args',
    'merge_config',
    'setup_seed',
    'setup_device',
    'create_save_dir',
    'create_dataloaders',
    'flatten_lstm_parameters',
    'build_optimizer_and_scheduler',
    'save_checkpoint',
    'load_checkpoint',
    'build_model_input_dict',
    'BaseTrainer',
]


def get_base_args():
    """获取基础命令行参数"""
    parser = argparse.ArgumentParser(description='IMUHOI模块化训练')
    parser.add_argument('--cfg', type=str, default='configs/IMUHOI_train.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=10, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=None, help='批量大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='学习率（优先于配置文件）')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--no_trans', action='store_true', help='禁用根节点位移预测')
    parser.add_argument('--model_arch', type=str, choices=['rnn', 'dit'], default=None, help='选择模型架构(rnn/dit)')
    return parser


def merge_config(args):
    """合并配置文件和命令行参数"""
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg = edict(cfg)
    
    # 命令行参数覆盖
    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epoch = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    cfg.pretrained_ckpt = args.pretrained_ckpt or getattr(cfg, 'pretrained_ckpt', None)
    
    cfg.debug = args.debug
    cfg.no_trans = args.no_trans
    cfg.cfg_file = args.cfg
    cfg.model_arch = args.model_arch or getattr(cfg, 'model_arch', 'rnn')
    
    if args.debug:
        cfg.batch_size = min(cfg.batch_size, 4)
    
    return cfg


def create_save_dir(cfg, module_name):
    """创建保存目录"""
    time_stamp = datetime.now().strftime("%m%d%H%M")
    suffix = "_noTrans" if cfg.no_trans else ""
    run_name = f"{module_name}{suffix}_{time_stamp}"
    if cfg.debug:
        run_name = f"{run_name}_debug"
    save_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    cfg.save_dir = save_dir
    return save_dir


def create_dataloaders(cfg, project_root=None):
    """
    创建数据加载器
    
    Args:
        cfg: 配置对象
        project_root: 项目根目录，默认自动检测
    
    Returns:
        train_loader, test_loader
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 获取要使用的数据集列表
    train_datasets = getattr(cfg, 'train_datasets', None)
    if train_datasets is None:
        train_datasets = list(cfg.datasets.keys())
    
    train_paths = []
    test_paths = []
    
    for ds_name in train_datasets:
        if ds_name not in cfg.datasets:
            print(f"警告: 数据集 {ds_name} 未在配置中找到")
            continue
        
        ds_cfg = cfg.datasets[ds_name]
        if cfg.debug and hasattr(ds_cfg, 'debug_path'):
            train_paths.append(os.path.join(project_root, ds_cfg.debug_path))
        else:
            train_paths.append(os.path.join(project_root, ds_cfg.train_path))
            if hasattr(ds_cfg, 'test_path'):
                test_paths.append(os.path.join(project_root, ds_cfg.test_path))
    
    if not train_paths:
        raise ValueError("未找到可用的数据集配置")
    
    # 过滤存在的路径
    train_paths = [p for p in train_paths if os.path.exists(p)]
    test_paths = [p for p in test_paths if os.path.exists(p)]
    
    if not train_paths:
        raise ValueError("训练数据路径不存在")
    
    print(f"训练数据路径: {train_paths}")
    
    train_dataset = IMUDataset(
        data_dir=train_paths,
        window_size=cfg.train.window,
        debug=cfg.debug
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"训练数据集大小: {len(train_dataset)}")
    
    test_loader = None
    if test_paths:
        test_dataset = IMUDataset(
            data_dir=test_paths,
            window_size=cfg.test.window,
            debug=cfg.debug
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False
            )
            print(f"测试数据集大小: {len(test_dataset)}")
    
    return train_loader, test_loader


def build_optimizer_and_scheduler(model, cfg):
    """创建优化器和调度器"""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = cfg.lr
    
    if cfg.use_multi_gpu:
        lr = lr * len(cfg.gpus)
        print(f"多GPU训练，学习率调整为: {lr}")
    
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=cfg.milestones, 
        gamma=cfg.gamma
    )
    
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    print(f"优化器参数: lr={lr}, weight_decay={cfg.weight_decay}")
    
    return optimizer, scheduler


class BaseTrainer:
    """基础训练器"""
    
    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(cfg.device)

        # 预训练权重加载（可选）
        self.pretrained_epoch = None
        pretrained_ckpt = getattr(cfg, "pretrained_ckpt", None)
        if pretrained_ckpt:
            if os.path.exists(pretrained_ckpt):
                try:
                    self.pretrained_epoch = load_checkpoint(self.model, pretrained_ckpt, self.device, strict=False)
                    print(f"加载预训练权重: {pretrained_ckpt} (epoch {self.pretrained_epoch})")
                except Exception as exc:
                    print(f"警告：加载预训练权重失败 {pretrained_ckpt}: {exc}")
            else:
                print(f"警告：预训练权重文件不存在: {pretrained_ckpt}")
        
        # 多GPU包装
        if cfg.use_multi_gpu:
            print(f'Wrapping model with DataParallel for GPUs: {cfg.gpus}')
            self.model = torch.nn.DataParallel(self.model, device_ids=cfg.gpus)
        
        flatten_lstm_parameters(self.model)
        
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.model, cfg)
        self.scaler = GradScaler()
        
        # TensorBoard
        self.writer = None
        if cfg.use_tensorboard and not cfg.debug:
            log_dir = os.path.join(cfg.save_dir, 'tensorboard_logs')
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f'TensorBoard logs: {log_dir}')
        
        self.best_loss = float('inf')
        self.n_iter = 0

    def model_forward(
        self,
        data_dict,
        batch=None,
    ):
        """默认的前向封装，可被子类重写"""
        return self.model(data_dict)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0
        loss_components = {}
        
        train_iter = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch in train_iter:
            data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)
            
            self.optimizer.zero_grad()
            pred_dict = self.model_forward(data_dict, batch=batch)
            
            total_loss, losses, weighted_losses = self.loss_fn(pred_dict, batch, self.device)
            
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss += total_loss.item()
            for key, value in weighted_losses.items():
                if isinstance(value, torch.Tensor):
                    loss_components[key] = loss_components.get(key, 0) + value.item()
            
            # 更新进度条
            postfix = {'loss': total_loss.item()}
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.item() != 0.0:
                    postfix[key] = value.item()
            train_iter.set_postfix(postfix)
            
            # TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('train/total_loss', total_loss.item(), self.n_iter)
                for key, value in weighted_losses.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'train/{key}', value.item(), self.n_iter)
            
            self.n_iter += 1
        
        train_loss /= len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return train_loss, loss_components
    
    def evaluate(self, epoch):
        """评估"""
        if self.test_loader is None:
            return None, {}
        
        self.model.eval()
        test_loss = 0
        loss_components = {}
        
        with torch.no_grad():
            test_iter = tqdm(self.test_loader, desc=f'Test {epoch}', leave=False)
            
            for batch in test_iter:
                data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=False)
                pred_dict = self.model_forward(data_dict, batch=batch)
                
                if hasattr(self.loss_fn, 'compute_test_loss'):
                    total_loss, losses = self.loss_fn.compute_test_loss(pred_dict, batch, self.device)
                else:
                    total_loss, losses, _ = self.loss_fn(pred_dict, batch, self.device)
                
                test_loss += total_loss.item()
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[key] = loss_components.get(key, 0) + value.item()
        
        test_loss /= len(self.test_loader)
        for key in loss_components:
            loss_components[key] /= len(self.test_loader)
        
        return test_loss, loss_components
    
    def train(self):
        """完整训练循环"""
        max_epoch = self.cfg.epoch
        
        for epoch in range(max_epoch):
            train_loss, train_components = self.train_epoch(epoch)
            
            print(f'\rEpoch {epoch}, Train Loss: {train_loss:.4f}', end='')
            
            # 每10个epoch评估一次
            if epoch % 10 == 0 and self.test_loader is not None:
                test_loss, test_components = self.evaluate(epoch)
                
                if test_loss is not None:
                    print(f', Test Loss: {test_loss:.4f}')
                    
                    if test_loss < self.best_loss:
                        self.best_loss = test_loss
                        save_path = os.path.join(self.cfg.save_dir, 'best.pt')
                        save_checkpoint(
                            self.model, self.optimizer, epoch, save_path, test_loss,
                            {'test_components': test_components}
                        )
                        print(f'新的最佳测试损失: {self.best_loss:.4f}')
                    
                    if self.writer is not None:
                        self.writer.add_scalar('test/total_loss', test_loss, self.n_iter)
                        for key, value in test_components.items():
                            self.writer.add_scalar(f'test/{key}', value, self.n_iter)
            else:
                print()
            
            self.scheduler.step()
        
        # 保存最终模型
        final_path = os.path.join(self.cfg.save_dir, 'final.pt')
        save_checkpoint(self.model, self.optimizer, max_epoch - 1, final_path, train_loss)
        
        if self.writer is not None:
            self.writer.close()
        
        return self.model
