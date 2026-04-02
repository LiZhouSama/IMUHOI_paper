"""
训练专用工具函数
"""
from __future__ import annotations

import argparse
import copy
import os
from contextlib import contextmanager
from datetime import datetime

import torch
import yaml
from easydict import EasyDict as edict
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:
    SummaryWriter = None

from dataset.dataset_IMUHOI import IMUDataset
from utils.utils import (
    build_model_input_dict,
    flatten_lstm_parameters,
    load_checkpoint,
    save_checkpoint,
    setup_device,
    setup_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = [
    "get_base_args",
    "merge_config",
    "setup_seed",
    "setup_device",
    "create_save_dir",
    "create_dataloaders",
    "flatten_lstm_parameters",
    "build_optimizer_and_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "build_model_input_dict",
    "BaseTrainer",
]


def get_base_args():
    """获取基础命令行参数"""
    parser = argparse.ArgumentParser(description="IMUHOI模块化训练")
    parser.add_argument("--cfg", type=str, default="configs/IMUHOI_train.yaml", help="配置文件路径")
    parser.add_argument("--seed", type=int, default=10, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=None, help="批量大小")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率（优先于配置文件）")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="预训练权重路径")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--no_trans", action="store_true", help="禁用根节点位移预测")
    parser.add_argument("--model_arch", type=str, choices=["rnn", "dit"], default="dit", help="选择模型架构(rnn/dit)")
    return parser


def merge_config(args):
    """合并配置文件和命令行参数"""
    with open(args.cfg, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg = edict(cfg)

    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epoch = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    cfg.pretrained_ckpt = args.pretrained_ckpt or getattr(cfg, "pretrained_ckpt", None)

    cfg.debug = args.debug
    cfg.no_trans = args.no_trans
    cfg.cfg_file = args.cfg
    cfg.model_arch = args.model_arch or getattr(cfg, "model_arch", "rnn")

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
    """创建数据加载器"""
    if project_root is None:
        project_root = PROJECT_ROOT

    train_datasets = getattr(cfg, "train_datasets", None)
    if train_datasets is None:
        train_datasets = list(cfg.datasets.keys())

    train_paths = []
    test_paths = []

    for ds_name in train_datasets:
        if ds_name not in cfg.datasets:
            print(f"警告: 数据集 {ds_name} 未在配置中找到")
            continue

        ds_cfg = cfg.datasets[ds_name]
        if cfg.debug and hasattr(ds_cfg, "debug_path"):
            train_paths.append(os.path.join(project_root, ds_cfg.debug_path))
            test_paths.append(os.path.join(project_root, ds_cfg.debug_path))
        else:
            train_paths.append(os.path.join(project_root, ds_cfg.train_path))
            if hasattr(ds_cfg, "test_path"):
                test_paths.append(os.path.join(project_root, ds_cfg.test_path))

    if not train_paths:
        raise ValueError("未找到可用的数据集配置")

    train_paths = [p for p in train_paths if os.path.exists(p)]
    test_paths = [p for p in test_paths if os.path.exists(p)]

    if not train_paths:
        raise ValueError("训练数据路径不存在")

    print(f"训练数据路径: {train_paths}")

    train_cfg = getattr(cfg, "train", {})
    test_cfg = getattr(cfg, "test", {})
    train_full_sequence = (
        train_cfg.get("full_sequence", False)
        if isinstance(train_cfg, dict)
        else getattr(train_cfg, "full_sequence", False)
    )
    test_full_sequence = (
        test_cfg.get("full_sequence", False)
        if isinstance(test_cfg, dict)
        else getattr(test_cfg, "full_sequence", False)
    )

    train_dataset = IMUDataset(
        data_dir=train_paths,
        window_size=cfg.train.window,
        debug=cfg.debug,
        simulate_imu_noise=True,
        full_sequence=bool(train_full_sequence),
        obj_points_sample_count=int(getattr(cfg, "mesh_downsample_points", 256)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"训练数据集大小: {len(train_dataset)}")

    test_loader = None
    if test_paths:
        test_dataset = IMUDataset(
            data_dir=test_paths,
            window_size=cfg.test.window,
            debug=cfg.debug,
            simulate_imu_noise=False,
            full_sequence=bool(test_full_sequence),
            obj_points_sample_count=int(getattr(cfg, "mesh_downsample_points", 256)),
        )
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                drop_last=False,
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
        gamma=cfg.gamma,
    )

    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    print(f"优化器参数: lr={lr}, weight_decay={cfg.weight_decay}")

    return optimizer, scheduler


class BaseTrainer:
    """基础训练器（带EMA与确定性评估采样）"""

    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(cfg.device)

        dit_cfg = getattr(cfg, "dit", {})

        def _get_cfg(name, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        self.inference_steps = _get_cfg("dit_inference_steps", None)
        self.inference_steps = int(self.inference_steps) if self.inference_steps is not None else None

        self.eval_ddim_steps = _get_cfg("eval_ddim_steps", self.inference_steps)
        self.eval_ddim_steps = int(self.eval_ddim_steps) if self.eval_ddim_steps is not None else None
        self.eval_sampler = str(_get_cfg("eval_sampler", _get_cfg("dit_inference_sampler", "ddim"))).lower()
        self.eval_eta = float(_get_cfg("eval_eta", 0.0))

        self.ema_decay = float(_get_cfg("ema_decay", 0.0))
        self.ema_model = None

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

        if cfg.use_multi_gpu:
            print(f"Wrapping model with DataParallel for GPUs: {cfg.gpus}")
            self.model = torch.nn.DataParallel(self.model, device_ids=cfg.gpus)

        flatten_lstm_parameters(self.model)

        self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.model, cfg)
        amp_dtype_cfg = str(getattr(cfg, "amp_dtype", "bf16")).lower()
        use_amp_cfg = bool(getattr(cfg, "use_amp", True))
        self.amp_enabled = bool(use_amp_cfg and self.device.type == "cuda")
        self.amp_dtype = None
        self.use_grad_scaler = False

        if self.amp_enabled:
            if amp_dtype_cfg in {"bf16", "bfloat16"}:
                bf16_supported = bool(
                    torch.cuda.is_available()
                    and hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()
                )
                if bf16_supported:
                    self.amp_dtype = torch.bfloat16
                else:
                    print("Warning: bf16 is not supported on this device, fallback to fp16.")
                    self.amp_dtype = torch.float16
                    self.use_grad_scaler = True
            elif amp_dtype_cfg in {"fp16", "float16", "half"}:
                self.amp_dtype = torch.float16
                self.use_grad_scaler = True
            else:
                print(f"Warning: unknown amp_dtype '{amp_dtype_cfg}', fallback to bf16.")
                bf16_supported = bool(
                    torch.cuda.is_available()
                    and hasattr(torch.cuda, "is_bf16_supported")
                    and torch.cuda.is_bf16_supported()
                )
                if bf16_supported:
                    self.amp_dtype = torch.bfloat16
                else:
                    self.amp_dtype = torch.float16
                    self.use_grad_scaler = True

        self.scaler = GradScaler(enabled=self.use_grad_scaler)
        if self.amp_enabled:
            amp_mode = "bf16" if self.amp_dtype == torch.bfloat16 else "fp16"
            print(f"AMP enabled: {amp_mode} (GradScaler={self.use_grad_scaler})")
        else:
            if use_amp_cfg and self.device.type != "cuda":
                print("AMP disabled: non-CUDA device.")
            else:
                print("AMP disabled.")

        if self.ema_decay > 0.0:
            self.ema_model = copy.deepcopy(self._unwrap_model(self.model)).to(self.device)
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
            print(f"EMA enabled (decay={self.ema_decay})")

        self.writer = None
        if cfg.use_tensorboard and not cfg.debug:
            if SummaryWriter is None:
                print("Warning: TensorBoard is unavailable, skipping SummaryWriter.")
            else:
                log_dir = os.path.join(cfg.save_dir, "tensorboard_logs")
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard logs: {log_dir}")

        self.best_loss = float("inf")
        self.n_iter = 0

    @staticmethod
    def _unwrap_model(model):
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    def _update_ema(self):
        if self.ema_model is None:
            return

        src_state = self._unwrap_model(self.model).state_dict()
        dst_state = self.ema_model.state_dict()

        with torch.no_grad():
            for key, value in dst_state.items():
                src_val = src_state[key].detach()
                if torch.is_floating_point(value):
                    value.mul_(self.ema_decay).add_(src_val, alpha=1.0 - self.ema_decay)
                else:
                    value.copy_(src_val)

    @contextmanager
    def _ema_scope(self):
        if self.ema_model is None:
            yield
            return

        target = self._unwrap_model(self.model)
        backup_state = copy.deepcopy(target.state_dict())
        target.load_state_dict(self.ema_model.state_dict(), strict=False)
        try:
            yield
        finally:
            target.load_state_dict(backup_state, strict=False)

    @contextmanager
    def _autocast_scope(self):
        if not self.amp_enabled or self.amp_dtype is None:
            yield
            return
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
            yield

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
        """默认的前向封装，可被子类重写"""
        gt_arg = batch if use_gt_targets else None
        if force_inference and hasattr(self.model, "inference"):
            try:
                return self.model.inference(
                    data_dict,
                    gt_targets=batch,
                    sample_steps=sample_steps,
                    sampler=sampler,
                    eta=eta,
                )
            except TypeError:
                return self.model.inference(data_dict, sample_steps=sample_steps, sampler=sampler, eta=eta)
        try:
            return self.model(data_dict, gt_targets=gt_arg)
        except TypeError:
            return self.model(data_dict)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0.0
        loss_components = {}

        train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in train_iter:
            data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast_scope():
                pred_dict = self.model_forward(data_dict, batch=batch)
                total_loss, losses, weighted_losses = self.loss_fn(pred_dict, batch, self.device)

            if self.use_grad_scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            self._update_ema()

            train_loss += total_loss.item()
            for key, value in weighted_losses.items():
                if isinstance(value, torch.Tensor):
                    loss_components[key] = loss_components.get(key, 0.0) + value.item()

            postfix = {"loss": total_loss.item()}
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.item() != 0.0:
                    postfix[key] = value.item()
            train_iter.set_postfix(postfix)

            if self.writer is not None:
                self.writer.add_scalar("train/total_loss", total_loss.item(), self.n_iter)
                for key, value in weighted_losses.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f"train/{key}", value.item(), self.n_iter)

            self.n_iter += 1

        train_loss /= max(len(self.train_loader), 1)
        for key in loss_components:
            loss_components[key] /= max(len(self.train_loader), 1)

        return train_loss, loss_components

    def evaluate(self, epoch):
        """评估"""
        if self.test_loader is None:
            return None, {}

        self.model.eval()
        test_loss = 0.0
        loss_components = {}

        with self._ema_scope():
            with torch.no_grad():
                test_iter = tqdm(self.test_loader, desc=f"Test {epoch}", leave=False)

                for batch in test_iter:
                    data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=False)
                    with self._autocast_scope():
                        pred_dict = self.model_forward(
                            data_dict,
                            batch=batch,
                            use_gt_targets=False,
                            force_inference=True,
                            sample_steps=self.eval_ddim_steps,
                            sampler=self.eval_sampler,
                            eta=self.eval_eta,
                        )

                        if hasattr(self.loss_fn, "compute_test_loss"):
                            total_loss, losses = self.loss_fn.compute_test_loss(pred_dict, batch, self.device)
                        else:
                            total_loss, losses, _ = self.loss_fn(pred_dict, batch, self.device)

                    test_loss += total_loss.item()
                    for key, value in losses.items():
                        if isinstance(value, torch.Tensor):
                            loss_components[key] = loss_components.get(key, 0.0) + value.item()

        test_loss /= max(len(self.test_loader), 1)
        for key in loss_components:
            loss_components[key] /= max(len(self.test_loader), 1)

        return test_loss, loss_components

    def train(self):
        """完整训练循环"""
        max_epoch = self.cfg.epoch

        for epoch in range(max_epoch):
            train_loss, train_components = self.train_epoch(epoch)
            print(f"\rEpoch {epoch}, Train Loss: {train_loss:.4f}", end="")

            if epoch % 20 == 0 and self.test_loader is not None:
                test_loss, test_components = self.evaluate(epoch)

                if test_loss is not None:
                    print(f", Test Loss: {test_loss:.4f}")

                    if test_loss < self.best_loss:
                        self.best_loss = test_loss
                        save_path = os.path.join(self.cfg.save_dir, "best.pt")
                        additional_info = {"test_components": test_components}
                        if self.ema_model is not None:
                            additional_info["ema_state_dict"] = self.ema_model.state_dict()
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            epoch,
                            save_path,
                            test_loss,
                            additional_info,
                        )
                        print(f"新的最佳测试损失: {self.best_loss:.4f}")

                    if self.writer is not None:
                        self.writer.add_scalar("test/total_loss", test_loss, self.n_iter)
                        for key, value in test_components.items():
                            self.writer.add_scalar(f"test/{key}", value, self.n_iter)
            else:
                print()

            self.scheduler.step()

        final_path = os.path.join(self.cfg.save_dir, "final.pt")
        additional_info = {}
        if self.ema_model is not None:
            additional_info["ema_state_dict"] = self.ema_model.state_dict()
        save_checkpoint(self.model, self.optimizer, max_epoch - 1, final_path, train_loss, additional_info)

        if self.writer is not None:
            self.writer.close()

        return self.model
