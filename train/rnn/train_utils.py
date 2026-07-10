"""
训练专用工具函数
"""
from __future__ import annotations
import os
import argparse
import random
import sys
import yaml
import numpy as np
from easydict import EasyDict as edict
from datetime import datetime
import torch
from torch import optim
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:
    SummaryWriter = None
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

# 从 utils 导入共用函数
from utils.utils import (
    setup_seed,
    setup_device,
    load_checkpoint,
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
    'save_config_snapshot',
    'resolve_resume_checkpoint',
    'get_model_state_dict',
    'load_state_dict_flexible',
    'load_model_state_from_checkpoint',
    'load_checkpoint',
    'build_model_input_dict',
    'BaseTrainer',
]


def get_base_args():
    """获取基础命令行参数"""
    parser = argparse.ArgumentParser(description='IMUHOI模块化训练')
    parser.add_argument('--cfg', type=str, default='configs/IMUHOI_train_rnn.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=10, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=None, help='批量大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='学习率（优先于配置文件）')
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--resume_dir', type=str, default=None, help='从已有训练输出目录恢复训练')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--no_trans', action='store_true', help='禁用根节点位移预测')
    parser.add_argument('--model_arch', type=str, choices=['rnn', 'dit', 'mamba_simple'], default=None, help='选择模型架构(rnn/dit/mamba_simple)')
    parser.add_argument('--run_suffix', type=str, default=None, help='追加到本次训练输出目录名的后缀')
    parser.add_argument(
        '--train_datasets',
        nargs='+',
        default=None,
        help='覆盖配置文件中的 train_datasets，例如: --train_datasets hodome 或 --train_datasets imhd behave',
    )
    return parser


def _cli_has_option(name):
    flags = {f"--{name}", f"--{name.replace('_', '-')}"}
    prefixes = {f"{flag}=" for flag in flags}
    for arg in sys.argv[1:]:
        if arg in flags or any(arg.startswith(prefix) for prefix in prefixes):
            return True
    return False


def _normalize_train_datasets(value):
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = []
        for item in value:
            parts.extend(str(item).split(","))
    datasets = [item.strip() for item in parts if item.strip()]
    return datasets or None


def merge_config(args):
    """合并配置文件和命令行参数"""
    resume_dir = getattr(args, "resume_dir", None)
    config_path = args.cfg
    resume_config_missing = False
    if resume_dir:
        resume_config_path = os.path.join(os.path.normpath(resume_dir), "config.yaml")
        if os.path.isfile(resume_config_path):
            config_path = resume_config_path
            print(f"从 resume_dir 读取配置: {config_path}")
        else:
            resume_config_missing = True
            print(f"警告: resume_dir 内未找到 config.yaml，将回退读取 --cfg: {args.cfg}")

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg = edict(cfg)
    is_resume = bool(resume_dir)
    
    # 命令行参数覆盖
    if args.seed is not None and (not is_resume or _cli_has_option("seed")):
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.pretrained_ckpt is not None or not hasattr(cfg, 'pretrained_ckpt'):
        cfg.pretrained_ckpt = args.pretrained_ckpt or getattr(cfg, 'pretrained_ckpt', None)

    cfg.resume_dir = os.path.normpath(resume_dir) if resume_dir else None
    cfg.resume_config_missing = resume_config_missing
    cfg.debug = bool(args.debug) if (not is_resume or _cli_has_option("debug")) else bool(getattr(cfg, 'debug', False))
    cfg.no_trans = bool(args.no_trans) if (not is_resume or _cli_has_option("no_trans")) else bool(getattr(cfg, 'no_trans', False))
    cfg.run_suffix = getattr(args, "run_suffix", None) or getattr(cfg, "run_suffix", None)
    cfg.cfg_file = config_path
    cfg.model_arch = args.model_arch or getattr(cfg, 'model_arch', 'rnn')
    train_datasets = _normalize_train_datasets(getattr(args, "train_datasets", None))
    if train_datasets is not None:
        cfg.train_datasets = train_datasets
    
    if cfg.debug:
        cfg.batch_size = min(cfg.batch_size, 4)
    
    return cfg


def create_save_dir(cfg, module_name):
    """创建保存目录"""
    resume_dir = getattr(cfg, "resume_dir", None)
    if resume_dir:
        resume_dir = os.path.normpath(resume_dir)
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"resume_dir does not exist or is not a directory: {resume_dir}")
        cfg.save_dir = resume_dir
        return resume_dir

    time_stamp = datetime.now().strftime("%m%d%H%M")
    suffix = "_noTrans" if cfg.no_trans else ""
    run_name = f"{module_name}{suffix}_{time_stamp}"
    run_suffix = getattr(cfg, "run_suffix", None)
    if run_suffix:
        run_name = f"{run_name}_{run_suffix}"
    if cfg.debug:
        run_name = f"{run_name}_debug"
    save_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    cfg.save_dir = save_dir
    return save_dir


def _to_plain_data(value):
    if isinstance(value, dict):
        return {key: _to_plain_data(val) for key, val in dict.items(value)}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value


def save_config_snapshot(cfg, extra=None, filename="config.yaml"):
    """保存当前训练配置快照。"""
    save_dir = getattr(cfg, "save_dir", None)
    if not save_dir:
        raise ValueError("cfg.save_dir must be set before saving config snapshot.")
    os.makedirs(save_dir, exist_ok=True)
    cfg_dict = _to_plain_data(cfg)
    if extra:
        cfg_dict.update(_to_plain_data(extra))
    config_path = os.path.join(save_dir, filename)
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg_dict, f, allow_unicode=True, sort_keys=False)
    print(f"保存配置: {config_path}")
    return config_path


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
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
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
    resolve_bimanual_contact_conflicts = bool(getattr(cfg, "resolve_bimanual_contact_conflicts", True))
    
    train_dataset = IMUDataset(
        data_dir=train_paths,
        window_size=cfg.train.window,
        debug=cfg.debug,
        obj_points_sample_count=int(getattr(cfg, "mesh_downsample_points", 256)),
        simulate_imu_noise=False,
        resolve_bimanual_contact_conflicts=resolve_bimanual_contact_conflicts,
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
            debug=cfg.debug,
            obj_points_sample_count=int(getattr(cfg, "mesh_downsample_points", 256)),
            simulate_imu_noise=False,
            resolve_bimanual_contact_conflicts=resolve_bimanual_contact_conflicts,
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


def call_model_inference(model, *args, inference_mode: str = "offline", **kwargs):
    """Call the standardized inference API without changing offline training semantics."""
    mode = str(inference_mode or "offline").lower()
    if mode == "offline" and isinstance(model, torch.nn.DataParallel):
        return model(*args, **kwargs)

    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    if hasattr(target, "inference"):
        return target.inference(*args, inference_mode=mode, **kwargs)
    if mode != "offline":
        raise RuntimeError(f"Model {type(target).__name__} does not support online inference.")
    return model(*args, **kwargs)


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not isinstance(state, dict):
        return
    try:
        if state.get("python") is not None:
            random.setstate(state["python"])
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:
        print(f"警告：恢复随机数状态失败: {exc}")


def _torch_load_checkpoint(checkpoint_path, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def get_model_state_dict(model):
    return model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()


def _select_state_dict(checkpoint, key=None):
    if key and isinstance(checkpoint, dict) and checkpoint.get(key) is not None:
        return checkpoint[key]
    if not isinstance(checkpoint, dict):
        return checkpoint
    return checkpoint.get("module_state_dict", checkpoint.get("model_state_dict", checkpoint))


def load_state_dict_flexible(model, state_dict, strict=False, name="model"):
    if strict:
        model.load_state_dict(state_dict, strict=True)
        return

    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = []
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        mapped_key = key
        if mapped_key not in model_state and mapped_key.startswith("module.") and mapped_key[7:] in model_state:
            mapped_key = mapped_key[7:]
        elif mapped_key not in model_state and f"module.{mapped_key}" in model_state:
            mapped_key = f"module.{mapped_key}"

        if mapped_key not in model_state:
            continue
        if model_state[mapped_key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[mapped_key].shape)))
            continue
        filtered_state[mapped_key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if skipped_shape:
        print(f"{name} 加载检查点时跳过 {len(skipped_shape)} 个shape不匹配参数")
    if missing:
        print(f"{name} 加载检查点缺失参数数量: {len(missing)}")
    if unexpected:
        print(f"{name} 加载检查点多余参数数量: {len(unexpected)}")


def load_model_state_from_checkpoint(model, checkpoint, strict=False, state_key=None, name="model"):
    state_dict = _select_state_dict(checkpoint, key=state_key)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a valid state_dict for {name}.")
    load_state_dict_flexible(model, state_dict, strict=strict, name=name)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    save_path,
    loss,
    additional_info=None,
    scheduler=None,
    scaler=None,
):
    """保存可完整恢复训练的检查点，并保留旧 module_state_dict 兼容格式。"""
    model_state_dict = get_model_state_dict(model)
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "module_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "rng_state": _capture_rng_state(),
        "loss": loss,
    }
    if additional_info:
        checkpoint_data.update(additional_info)
    tmp_path = f"{save_path}.tmp"
    torch.save(checkpoint_data, tmp_path)
    os.replace(tmp_path, save_path)
    print(f"保存检查点: {save_path}")


def resolve_resume_checkpoint(resume_dir):
    if not resume_dir:
        return None
    candidates = [
        os.path.join(resume_dir, "last.pt"),
        os.path.join(resume_dir, "final.pt"),
        os.path.join(resume_dir, "best.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"resume_dir contains no last.pt, final.pt, or best.pt: {resume_dir}")


class BaseTrainer:
    """基础训练器"""
    
    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(cfg.device)
        self.resume_path = resolve_resume_checkpoint(getattr(cfg, "resume_dir", None))
        self.resume_checkpoint = None
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.n_iter = 0

        # 预训练权重加载（可选）
        self.pretrained_epoch = None
        pretrained_ckpt = getattr(cfg, "pretrained_ckpt", None)
        if self.resume_path:
            print(f"Resume checkpoint: {self.resume_path}")
            self.resume_checkpoint = _torch_load_checkpoint(self.resume_path, map_location=self.device)
            load_model_state_from_checkpoint(
                self.model,
                self.resume_checkpoint,
                strict=False,
                name=self.model.__class__.__name__,
            )
            print(f"恢复模型权重: {self.resume_path}")
        elif pretrained_ckpt:
            if os.path.exists(pretrained_ckpt):
                try:
                    self.pretrained_epoch = load_checkpoint(self.model, pretrained_ckpt, self.device, strict=False)
                    print(f"加载预训练权重: {pretrained_ckpt} (epoch {self.pretrained_epoch})")
                except Exception as exc:
                    print(f"警告：加载预训练权重失败 {pretrained_ckpt}: {exc}")
            else:
                print(f"警告：预训练权重文件不存在: {pretrained_ckpt}")
        if self.resume_path and pretrained_ckpt:
            print("检测到 --resume_dir，忽略 --pretrained_ckpt；resume 会恢复断点权重。")
        
        # 多GPU包装
        if cfg.use_multi_gpu:
            print(f'Wrapping model with DataParallel for GPUs: {cfg.gpus}')
            self.model = torch.nn.DataParallel(self.model, device_ids=cfg.gpus)
        
        flatten_lstm_parameters(self.model)
        
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.model, cfg)
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        if self.resume_checkpoint is not None:
            self._restore_training_state(self.resume_checkpoint)
        
        # TensorBoard
        self.writer = None
        if cfg.use_tensorboard and not cfg.debug and SummaryWriter is not None:
            log_dir = os.path.join(cfg.save_dir, 'tensorboard_logs')
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f'TensorBoard logs: {log_dir}')
        elif cfg.use_tensorboard and not cfg.debug:
            print('TensorBoard is unavailable; continue without SummaryWriter.')
        
    def _restore_training_state(self, checkpoint):
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Resume checkpoint must be a dict: {self.resume_path}")

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except Exception as exc:
                print(f"警告：恢复 optimizer_state_dict 失败，将只恢复权重: {exc}")
        else:
            print("警告：resume checkpoint 缺少 optimizer_state_dict，将只恢复权重。")

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except Exception as exc:
                print(f"警告：恢复 scheduler_state_dict 失败，学习率调度器将从当前配置重新开始: {exc}")
        else:
            print("警告：resume checkpoint 缺少 scheduler_state_dict，学习率调度器将从当前配置重新开始。")

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            try:
                self.scaler.load_state_dict(scaler_state)
            except Exception as exc:
                print(f"警告：恢复 scaler_state_dict 失败: {exc}")

        completed_epoch = int(checkpoint.get("epoch", -1))
        self.start_epoch = completed_epoch + 1
        self.n_iter = int(checkpoint.get("n_iter", self.start_epoch * max(len(self.train_loader), 1)))
        self.best_loss = self._resolve_resume_best_loss(checkpoint)
        _restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"恢复训练状态: completed_epoch={completed_epoch}, "
            f"next_epoch={self.start_epoch}, n_iter={self.n_iter}, best_loss={self.best_loss:.6f}"
        )

    def _resolve_resume_best_loss(self, checkpoint):
        if isinstance(checkpoint.get("best_loss"), (int, float)):
            return float(checkpoint["best_loss"])
        if os.path.basename(self.resume_path or "") == "best.pt" and isinstance(checkpoint.get("loss"), (int, float)):
            return float(checkpoint["loss"])

        best_path = os.path.join(self.cfg.save_dir, "best.pt")
        if os.path.isfile(best_path):
            try:
                best_checkpoint = _torch_load_checkpoint(best_path, map_location="cpu")
                if isinstance(best_checkpoint, dict) and isinstance(best_checkpoint.get("loss"), (int, float)):
                    return float(best_checkpoint["loss"])
            except Exception as exc:
                print(f"警告：读取 best.pt 的 best_loss 失败: {exc}")
        return float("inf")

    def model_forward(
        self,
        data_dict,
        batch=None,
    ):
        """默认的前向封装，可被子类重写"""
        return call_model_inference(self.model, data_dict, inference_mode="offline")
    
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
                    component_losses = losses
                else:
                    total_loss, losses, weighted_losses = self.loss_fn(pred_dict, batch, self.device)
                    component_losses = weighted_losses
                
                test_loss += total_loss.item()
                for key, value in component_losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[key] = loss_components.get(key, 0) + value.item()
        
        test_loss /= len(self.test_loader)
        for key in loss_components:
            loss_components[key] /= len(self.test_loader)
        
        return test_loss, loss_components
    
    def train(self):
        """完整训练循环"""
        max_epoch = self.cfg.epochs
        train_loss = 0.0
        if self.start_epoch >= max_epoch:
            print(f"Resume checkpoint already reached epoch {self.start_epoch - 1}; target epochs={max_epoch}.")
            if self.writer is not None:
                self.writer.close()
            return self.model
        
        for epoch in range(self.start_epoch, max_epoch):
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
                            {
                                'test_components': test_components,
                                'best_loss': self.best_loss,
                                'n_iter': self.n_iter,
                            },
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                        )
                        print(f'新的最佳测试损失: {self.best_loss:.4f}')
                    
                    if self.writer is not None:
                        self.writer.add_scalar('test/total_loss', test_loss, self.n_iter)
                        for key, value in test_components.items():
                            self.writer.add_scalar(f'test/{key}', value, self.n_iter)
            else:
                print()
            
            self.scheduler.step()
            last_path = os.path.join(self.cfg.save_dir, 'last.pt')
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                last_path,
                train_loss,
                {
                    'train_components': train_components,
                    'best_loss': self.best_loss,
                    'n_iter': self.n_iter,
                },
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
        
        # 保存最终模型
        final_path = os.path.join(self.cfg.save_dir, 'final.pt')
        save_checkpoint(
            self.model,
            self.optimizer,
            max_epoch - 1,
            final_path,
            train_loss,
            {
                'best_loss': self.best_loss,
                'n_iter': self.n_iter,
            },
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        
        if self.writer is not None:
            self.writer.close()
        
        return self.model
