"""Training helpers for the Mamba path."""
from __future__ import annotations

import argparse
import os
import random
import sys
import types
from datetime import datetime

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import _REDUCED_POSE_NAMES, _SENSOR_POS_INDICES, _SENSOR_VEL_NAMES
from utils.rotation_conversions import matrix_to_rotation_6d

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:
    SummaryWriter = None

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
    parser = argparse.ArgumentParser(description="IMUHOI Mamba module training")
    parser.add_argument("--cfg", type=str, default="configs/IMUHOI_train_mamba.yaml", help="配置文件路径")
    parser.add_argument("--seed", type=int, default=10, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=None, help="批量大小")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率（优先于配置文件）")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="预训练权重路径")
    parser.add_argument("--resume_dir", type=str, default=None, help="从已有训练输出目录恢复训练")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--no_trans", action="store_true", help="禁用根节点位移预测")
    parser.add_argument("--model_arch", type=str, choices=["rnn", "dit", "mamba"], default="mamba", help="选择模型架构")
    return parser


def merge_config(args):
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
    cfg.resume_dir = args.resume_dir or getattr(cfg, "resume_dir", None)
    cfg.debug = args.debug
    cfg.no_trans = args.no_trans
    cfg.cfg_file = args.cfg
    cfg.model_arch = args.model_arch or "mamba"

    if args.debug:
        cfg.batch_size = min(cfg.batch_size, 4)
        cfg.num_workers = min(getattr(cfg, "num_workers", 0), 2)
    return cfg


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device(cfg):
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"可用GPU数量: {available_gpus}")
        if hasattr(cfg, "gpus") and cfg.gpus:
            valid_gpus = [gpu for gpu in cfg.gpus if gpu < available_gpus]
            if len(valid_gpus) != len(cfg.gpus):
                print(f"警告: 配置的GPU {cfg.gpus} 中部分不可用，使用可用GPU: {valid_gpus}")
                cfg.gpus = valid_gpus
            cfg.use_multi_gpu = getattr(cfg, "use_multi_gpu", True) and len(cfg.gpus) > 1
            cfg.device = f"cuda:{cfg.gpus[0]}"
            torch.cuda.set_device(cfg.gpus[0])
        else:
            cfg.gpus = [0]
            cfg.use_multi_gpu = False
            cfg.device = "cuda:0"
    else:
        print("CUDA不可用，使用CPU")
        cfg.device = "cpu"
        cfg.use_multi_gpu = False
    return cfg


def create_save_dir(cfg, module_name):
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
    if cfg.debug:
        run_name = f"{run_name}_debug"
    save_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    cfg.save_dir = save_dir
    return save_dir


def create_dataloaders(cfg, project_root=None):
    _ensure_pytorch3d_transforms()
    from dataset.dataset_IMUHOI import IMUDataset

    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    train_datasets = getattr(cfg, "train_datasets", None) or list(cfg.datasets.keys())
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

    train_paths = [p for p in train_paths if os.path.exists(p)]
    test_paths = [p for p in test_paths if os.path.exists(p)]
    if not train_paths:
        raise ValueError("训练数据路径不存在")

    print(f"训练数据路径: {train_paths}")
    train_dataset = IMUDataset(data_dir=train_paths, window_size=cfg.train.window, debug=cfg.debug)
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
        test_dataset = IMUDataset(data_dir=test_paths, window_size=cfg.test.window, debug=cfg.debug)
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


def _ensure_pytorch3d_transforms():
    """Install a tiny transforms fallback for environments without PyTorch3D."""
    try:
        import pytorch3d.transforms  # noqa: F401
        return
    except Exception:
        pass

    import utils.rotation_conversions as rot

    pytorch3d_mod = types.ModuleType("pytorch3d")
    transforms_mod = types.ModuleType("pytorch3d.transforms")
    transforms_mod.rotation_6d_to_matrix = rot.rotation_6d_to_matrix
    transforms_mod.matrix_to_rotation_6d = rot.matrix_to_rotation_6d
    transforms_mod.matrix_to_axis_angle = rot.matrix_to_axis_angle
    pytorch3d_mod.transforms = transforms_mod
    sys.modules["pytorch3d"] = pytorch3d_mod
    sys.modules["pytorch3d.transforms"] = transforms_mod


def flatten_lstm_parameters(module):
    for child in module.children():
        if isinstance(child, torch.nn.LSTM):
            child.flatten_parameters()
        else:
            flatten_lstm_parameters(child)


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


def _select_state_dict(checkpoint, use_ema=True):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if use_ema and checkpoint.get("ema_state_dict") is not None:
        return checkpoint["ema_state_dict"]
    return checkpoint.get("module_state_dict", checkpoint.get("model_state_dict", checkpoint))


def _load_model_from_checkpoint(model, checkpoint, strict=True, use_ema=True):
    state_dict = _select_state_dict(checkpoint, use_ema=use_ema)
    model.load_state_dict(state_dict, strict=strict)


def _torch_load_checkpoint(checkpoint_path, map_location):
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


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
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
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


def load_checkpoint(model, checkpoint_path, device, strict=True, use_ema=True):
    checkpoint = _torch_load_checkpoint(checkpoint_path, map_location=device)
    _load_model_from_checkpoint(model, checkpoint, strict=strict, use_ema=use_ema)
    print(f"加载检查点: {checkpoint_path}")
    return checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0


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


def _ensure_bt(tensor, shape, device, dtype):
    if not isinstance(tensor, torch.Tensor):
        return torch.zeros(*shape, device=device, dtype=dtype)
    out = tensor.to(device=device)
    if out.dim() == len(shape) - 1:
        out = out.unsqueeze(0)
    if out.shape[0] == 1 and shape[0] > 1:
        out = out.expand(shape[0], *out.shape[1:])
    if tuple(out.shape) != tuple(shape):
        return torch.zeros(*shape, device=device, dtype=dtype)
    return out.to(dtype=dtype)


def build_model_input_dict(batch, cfg, device, add_noise=True):
    human_imu = batch["human_imu"].to(device)
    batch_size, seq_len = human_imu.shape[:2]
    dtype = human_imu.dtype

    skip_extra_noise = batch.get("imu_noise_applied", False)
    skip_noise = bool(skip_extra_noise) if not isinstance(skip_extra_noise, torch.Tensor) else bool(skip_extra_noise.flatten()[0].item())
    imu_noise_std = float(getattr(cfg, "imu_noise_std", 0.1))
    if add_noise and imu_noise_std > 0.0 and not skip_noise:
        human_imu = human_imu + torch.randn_like(human_imu) * imu_noise_std

    sensor_vel_root = _ensure_bt(
        batch.get("sensor_vel_root"),
        (batch_size, seq_len, len(_SENSOR_VEL_NAMES), 3),
        device,
        dtype,
    )
    sensor_vel_glb = _ensure_bt(
        batch.get("sensor_vel_glb"),
        (batch_size, seq_len, len(_SENSOR_POS_INDICES), 3),
        device,
        dtype,
    )
    trans = _ensure_bt(batch.get("trans"), (batch_size, seq_len, 3), device, dtype)

    ori_root_reduced = batch.get("ori_root_reduced")
    if isinstance(ori_root_reduced, torch.Tensor):
        ori_root_reduced = ori_root_reduced.to(device=device, dtype=dtype)
        if ori_root_reduced.dim() == 4:
            ori_root_reduced = ori_root_reduced.unsqueeze(0)
        if ori_root_reduced.shape[0] == 1 and batch_size > 1:
            ori_root_reduced = ori_root_reduced.expand(batch_size, *ori_root_reduced.shape[1:])
        if ori_root_reduced.shape[:3] == (batch_size, seq_len, len(_REDUCED_POSE_NAMES)):
            p_init = matrix_to_rotation_6d(ori_root_reduced[:, 0].reshape(-1, 3, 3)).reshape(
                batch_size,
                len(_REDUCED_POSE_NAMES),
                6,
            )
        else:
            p_init = torch.zeros(batch_size, len(_REDUCED_POSE_NAMES), 6, device=device, dtype=dtype)
    else:
        p_init = torch.zeros(batch_size, len(_REDUCED_POSE_NAMES), 6, device=device, dtype=dtype)

    obj_imu = batch.get("obj_imu")
    if isinstance(obj_imu, torch.Tensor):
        obj_imu = obj_imu.to(device=device, dtype=dtype)
    else:
        obj_imu = torch.zeros(batch_size, seq_len, int(getattr(cfg, "obj_imu_dim", 9)), device=device, dtype=dtype)
    obj_noise_std = float(getattr(cfg, "obj_imu_noise_std", 0.1))
    if add_noise and obj_noise_std > 0.0 and not skip_noise:
        obj_imu = obj_imu + torch.randn_like(obj_imu) * obj_noise_std

    obj_vel = _ensure_bt(batch.get("obj_vel"), (batch_size, seq_len, 3), device, dtype)
    obj_trans = _ensure_bt(batch.get("obj_trans"), (batch_size, seq_len, 3), device, dtype)

    lhand_contact = batch.get("lhand_contact")
    rhand_contact = batch.get("rhand_contact")
    obj_contact = batch.get("obj_contact")
    if isinstance(lhand_contact, torch.Tensor):
        lhand_first = lhand_contact.to(device=device, dtype=dtype)
        if lhand_first.dim() == 1:
            lhand_first = lhand_first.unsqueeze(0)
        lhand_first = lhand_first[:, 0]
    else:
        lhand_first = torch.zeros(batch_size, device=device, dtype=dtype)
    if isinstance(rhand_contact, torch.Tensor):
        rhand_first = rhand_contact.to(device=device, dtype=dtype)
        if rhand_first.dim() == 1:
            rhand_first = rhand_first.unsqueeze(0)
        rhand_first = rhand_first[:, 0]
    else:
        rhand_first = torch.zeros(batch_size, device=device, dtype=dtype)
    if isinstance(obj_contact, torch.Tensor):
        obj_first = obj_contact.to(device=device, dtype=dtype)
        if obj_first.dim() == 1:
            obj_first = obj_first.unsqueeze(0)
        obj_first = obj_first[:, 0]
    else:
        obj_first = torch.zeros(batch_size, device=device, dtype=dtype)

    has_object_raw = batch.get("has_object")
    if isinstance(has_object_raw, torch.Tensor):
        has_object = has_object_raw.to(device=device, dtype=torch.bool)
        if has_object.dim() == 0:
            has_object = has_object.view(1)
        if has_object.shape[0] == 1 and batch_size > 1:
            has_object = has_object.expand(batch_size)
    elif isinstance(has_object_raw, (bool, int)):
        has_object = torch.full((batch_size,), bool(has_object_raw), dtype=torch.bool, device=device)
    else:
        has_object = torch.ones(batch_size, dtype=torch.bool, device=device)

    obj_points_count = int(getattr(cfg, "mesh_downsample_points", 256))
    obj_points_canonical_raw = batch.get("obj_points_canonical")
    if isinstance(obj_points_canonical_raw, torch.Tensor):
        obj_points_canonical = obj_points_canonical_raw.to(device=device, dtype=dtype)
        if obj_points_canonical.dim() == 2:
            obj_points_canonical = obj_points_canonical.unsqueeze(0)
        if obj_points_canonical.shape[0] == 1 and batch_size > 1:
            obj_points_canonical = obj_points_canonical.expand(batch_size, -1, -1)
        if obj_points_canonical.dim() != 3 or obj_points_canonical.shape[0] != batch_size or obj_points_canonical.shape[-1] != 3:
            obj_points_canonical = torch.zeros(batch_size, obj_points_count, 3, device=device, dtype=dtype)
    else:
        obj_points_canonical = torch.zeros(batch_size, obj_points_count, 3, device=device, dtype=dtype)

    return {
        "human_imu": human_imu,
        "obj_imu": obj_imu,
        "v_init": sensor_vel_root[:, 0],
        "p_init": p_init,
        "trans_init": trans[:, 0],
        "trans_gt": trans,
        "hand_vel_glb_init": sensor_vel_glb[:, 0, -2:],
        "obj_trans_init": obj_trans[:, 0],
        "obj_vel_init": obj_vel[:, 0],
        "contact_init": torch.stack((lhand_first, rhand_first, obj_first), dim=-1),
        "has_object": has_object,
        "obj_name": batch.get("obj_name"),
        "obj_points_canonical": obj_points_canonical,
        "seq_file": batch.get("seq_file"),
        "window_start": batch.get("window_start"),
        "window_end": batch.get("window_end"),
    }


def build_optimizer_and_scheduler(model, cfg):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lr = cfg.lr * (len(cfg.gpus) if getattr(cfg, "use_multi_gpu", False) else 1)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
    print(f"优化器参数: lr={lr}, weight_decay={cfg.weight_decay}")
    return optimizer, scheduler


class BaseTrainer:
    """Simple supervised trainer."""

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
        self.best_loss = float("inf")
        self.n_iter = 0

        pretrained_ckpt = getattr(cfg, "pretrained_ckpt", None)
        if self.resume_path:
            print(f"Resume checkpoint: {self.resume_path}")
            self.resume_checkpoint = _torch_load_checkpoint(self.resume_path, map_location=self.device)
            _load_model_from_checkpoint(self.model, self.resume_checkpoint, strict=False, use_ema=False)
            print(f"恢复模型权重: {self.resume_path}")
        elif pretrained_ckpt:
            if os.path.exists(pretrained_ckpt):
                try:
                    load_checkpoint(self.model, pretrained_ckpt, self.device, strict=False)
                except Exception as exc:
                    print(f"警告：加载预训练权重失败 {pretrained_ckpt}: {exc}")
            else:
                print(f"警告：预训练权重文件不存在: {pretrained_ckpt}")
        if self.resume_path and pretrained_ckpt:
            print("检测到 --resume_dir，忽略 --pretrained_ckpt；resume 会恢复断点权重。")

        if getattr(cfg, "use_multi_gpu", False):
            print(f"Wrapping model with DataParallel for GPUs: {cfg.gpus}")
            self.model = torch.nn.DataParallel(self.model, device_ids=cfg.gpus)

        flatten_lstm_parameters(self.model)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(self.model, cfg)
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        if self.resume_checkpoint is not None:
            self._restore_training_state(self.resume_checkpoint)

        self.writer = None
        if getattr(cfg, "use_tensorboard", False) and not cfg.debug:
            if SummaryWriter is None:
                print("TensorBoard不可用，跳过日志写入")
            else:
                log_dir = os.path.join(cfg.save_dir, "tensorboard_logs")
                self.writer = SummaryWriter(log_dir=log_dir)
                print(f"TensorBoard logs: {log_dir}")

    def _restore_training_state(self, checkpoint):
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Resume checkpoint must be a dict: {self.resume_path}")

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        else:
            print("警告：resume checkpoint 缺少 optimizer_state_dict，将只恢复权重。")

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        else:
            print("警告：resume checkpoint 缺少 scheduler_state_dict，学习率调度器将从当前配置重新开始。")

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

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

    def model_forward(self, data_dict, batch=None):
        return self.model(data_dict)

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0
        loss_components = {}
        train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in train_iter:
            data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)
            self.optimizer.zero_grad(set_to_none=True)
            pred_dict = self.model_forward(data_dict, batch=batch)
            total_loss, losses, weighted_losses = self.loss_fn(pred_dict, batch, self.device)

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
        if self.test_loader is None:
            return None, {}
        self.model.eval()
        test_loss = 0.0
        loss_components = {}
        with torch.no_grad():
            test_iter = tqdm(self.test_loader, desc=f"Test {epoch}", leave=False)
            for batch in test_iter:
                data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=False)
                pred_dict = self.model_forward(data_dict, batch=batch)
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
        max_epoch = self.cfg.epoch
        last_train_loss = 0.0
        if self.start_epoch >= max_epoch:
            print(f"Resume checkpoint already reached epoch {self.start_epoch - 1}; target epochs={max_epoch}.")
            if self.writer is not None:
                self.writer.close()
            return self.model

        for epoch in range(self.start_epoch, max_epoch):
            last_train_loss, train_components = self.train_epoch(epoch)
            print(f"\rEpoch {epoch}, Train Loss: {last_train_loss:.4f}", end="")

            if epoch % 10 == 0 and self.test_loader is not None:
                test_loss, test_components = self.evaluate(epoch)
                if test_loss is not None:
                    print(f", Test Loss: {test_loss:.4f}")
                    if test_loss < self.best_loss:
                        self.best_loss = test_loss
                        save_path = os.path.join(self.cfg.save_dir, "best.pt")
                        save_checkpoint(
                            self.model,
                            self.optimizer,
                            epoch,
                            save_path,
                            test_loss,
                            {
                                "test_components": test_components,
                                "best_loss": self.best_loss,
                                "n_iter": self.n_iter,
                            },
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                        )
                        print(f"新的最佳测试损失: {self.best_loss:.4f}")
                    if self.writer is not None:
                        self.writer.add_scalar("test/total_loss", test_loss, self.n_iter)
                        for key, value in test_components.items():
                            self.writer.add_scalar(f"test/{key}", value, self.n_iter)
            else:
                print()

            self.scheduler.step()
            last_path = os.path.join(self.cfg.save_dir, "last.pt")
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                last_path,
                last_train_loss,
                {
                    "train_components": train_components,
                    "best_loss": self.best_loss,
                    "n_iter": self.n_iter,
                },
                scheduler=self.scheduler,
                scaler=self.scaler,
            )

        final_path = os.path.join(self.cfg.save_dir, "final.pt")
        save_checkpoint(
            self.model,
            self.optimizer,
            max_epoch - 1,
            final_path,
            last_train_loss,
            {"best_loss": self.best_loss, "n_iter": self.n_iter},
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        if self.writer is not None:
            self.writer.close()
        return self.model
