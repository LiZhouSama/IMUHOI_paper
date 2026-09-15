"""Train IMUHOI comparison baselines.

Example:
    conda activate SAGE
    python -m Comparisons.train_comparison --method transpose --debug --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import yaml
from easydict import EasyDict as edict
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comparisons.common.dataset import IMUDataset
from Comparisons.common.adapters import (
    TIP_ACC_SUM_WIN_LEN,
    adapt_batch,
    tensor_to_device,
)
from Comparisons.common.losses import DIPLoss, GlobalPoseLoss, TIPLoss, TransPoseLoss
from Comparisons.dip18 import DIP18HOIModel, DIP18ReferenceHOIModel
from Comparisons.globalpose import GlobalPoseHOIModel
from Comparisons.tip import TIPHOIModel, TIPReferenceHOIModel
from Comparisons.tip.sbp_cache import attach_tip_sbp_from_cache
from Comparisons.transpose import TransPoseHOIModel
from utils.utils import setup_seed


METHODS = ("dip18", "tip", "transpose", "globalpose")
COMPARISON_PROTOCOL_VERSION = 2
OBJECT_OUTPUT_MODE = "velocity"


try:
    _BooleanOptionalAction = argparse.BooleanOptionalAction
except AttributeError:  # Python 3.8 compatibility
    class _BooleanOptionalAction(argparse.Action):
        """Backport argparse's ``BooleanOptionalAction`` for Python 3.8."""

        def __init__(self, option_strings, dest, default=None, **kwargs):
            expanded = []
            for option in option_strings:
                expanded.append(option)
                if option.startswith("--"):
                    expanded.append("--no-" + option[2:])
            super().__init__(expanded, dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))


METHOD_TRAINING_DEFAULTS = {
    "dip18": {
        "epochs": 80,
        "batch_size": 16,
        "lr": 1e-4,
        "optimizer": "adam",
        "scheduler": "dip_exponential",
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "window_size": 120,
        "obj_loss_weight": 1.0,
    },
    "tip": {
        "epochs": 1100,
        "batch_size": 256,
        "lr": 1e-4,
        "optimizer": "adam",
        "scheduler": "tip_cosine",
        "weight_decay": 1e-4,
        "grad_clip": 5.0,
        # Keep the historical extension default for compatibility; official
        # TIP overrides this to 120 in load_cfg below.
        "window_size": 40,
        "obj_loss_weight": 100.0,
    },
    "transpose": {
        "epochs": 300,
        "batch_size": 128,
        "lr": 2e-3,
        "optimizer": "adam",
        "scheduler": "none",
        "weight_decay": 0.0,
        "grad_clip": 0.0,
        "window_size": 120,
        "obj_loss_weight": 1.0,
    },
    "globalpose": {
        "epochs": 100,
        "batch_size": 16,
        "lr": 1e-4,
        "optimizer": "adamw",
        "scheduler": "none",
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        # The official loader splits at up to 240 frames and keeps tails >= 60.
        # IMUHOI's fixed-window loader uses 120 so short processed sequences remain usable.
        "window_size": 120,
        "obj_loss_weight": 1.0,
    },
}
TIP_N_HEADS = 16
TIP_STATE_NOISE = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-method comparison baselines on IMUHOI")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--cfg", default="configs/IMUHOI_train_rnn.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default=None)
    parser.add_argument(
        "--scheduler",
        choices=("none", "dip_exponential", "tip_cosine"),
        default=None,
    )
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=None,
        help="Global gradient norm; <= 0 disables clipping",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Training/validation window; defaults to the method reference protocol",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_sbps", type=int, default=5, help="TIP SBP count; labels are masked if absent")
    parser.add_argument("--tip_n_heads", type=int, default=TIP_N_HEADS)
    parser.add_argument(
        "--tip_variant",
        "--tip-variant",
        choices=("legacy", "official"),
        default="official",
        help="TIP protocol; official matches the released transformer-inertial-poser checkpoint.",
    )
    parser.add_argument(
        "--transpose_variant",
        "--transpose-variant",
        choices=("legacy", "official"),
        default="official",
        help="TransPose protocol; official matches the released OMOMO object checkpoint.",
    )
    parser.add_argument(
        "--globalpose_variant",
        "--globalpose-variant",
        choices=("legacy", "official"),
        default="official",
        help="GlobalPose object protocol; official uses the released 243+9 path.",
    )
    parser.add_argument("--tip_state_noise", type=float, default=TIP_STATE_NOISE)
    parser.add_argument(
        "--tip_sbp_cache_root",
        default="Comparisons/tip/sbp_cache",
        help="TIP sidecar SBP cache root; only used by --method tip",
    )
    parser.add_argument(
        "--obj_loss_weight",
        type=float,
        default=None,
        help="Override the method reference extension's object-loss scale",
    )
    parser.add_argument("--save_dir", default="outputs/Comparisons")
    parser.add_argument("--run_name", default=None, help="Optional fixed run directory name under --save_dir")
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.05,
        help="Fraction of training files held out for checkpoint selection; 0 uses cfg test data",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Seed for the deterministic train/validation file split",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Run validation every N epochs (also runs at epoch 1 and the final epoch)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Write resumable last.pt every N epochs (also writes on validation/final epochs)",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Validation checks without improvement; 0 disables early stopping",
    )
    parser.add_argument("--resume", default=None, help="Resume model, optimizer, and scheduler from a checkpoint")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Reference human-model weights used only when starting a new run",
    )
    parser.add_argument(
        "--freeze_reference_human",
        "--freeze-reference-human",
        action=_BooleanOptionalAction,
        default=None,
        help=(
            "For GlobalPose with --pretrained, freeze PL/IK/VR and optimize "
            "only the object head. Defaults to enabled for the official variant."
        ),
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument(
        "--max_train_sequences",
        type=int,
        default=None,
        help="Limit sequence files before dataset preload (useful for smoke tests)",
    )
    parser.add_argument(
        "--max_eval_sequences",
        type=int,
        default=None,
        help="Limit validation sequence files before dataset preload",
    )
    parser.add_argument("--max_train_batches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max_eval_batches", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_cfg(args: argparse.Namespace):
    if args.n_sbps < 0:
        raise ValueError("--n_sbps must be non-negative")
    if args.method == "tip":
        if args.tip_n_heads <= 0 or 256 % args.tip_n_heads != 0:
            raise ValueError("--tip_n_heads must be a positive divisor of 256")
        if args.tip_state_noise < 0:
            raise ValueError("--tip_state_noise must be non-negative")
        if args.tip_variant == "official":
            # The public object checkpoint has no SBP channels.
            args.n_sbps = 0
    if args.freeze_reference_human is None:
        args.freeze_reference_human = bool(
            args.method == "globalpose"
            and args.globalpose_variant == "official"
            and args.pretrained is not None
        )
    if args.freeze_reference_human and args.method != "globalpose":
        raise ValueError("--freeze-reference-human is only supported for GlobalPose")
    if args.freeze_reference_human and args.pretrained is None:
        raise ValueError("--freeze-reference-human requires --pretrained")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be in [0, 1)")
    if args.eval_every <= 0 or args.save_every <= 0:
        raise ValueError("--eval_every and --save_every must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience must be non-negative")
    for name in ("max_train_sequences", "max_eval_sequences"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be positive")
    with open(args.cfg, "r") as f:
        cfg = edict(yaml.safe_load(f))
    method_defaults = METHOD_TRAINING_DEFAULTS[args.method]
    if args.obj_loss_weight is None:
        args.obj_loss_weight = float(method_defaults["obj_loss_weight"])
    if args.obj_loss_weight < 0:
        raise ValueError("--obj_loss_weight must be non-negative")
    cfg.epochs = int(method_defaults["epochs"])
    cfg.batch_size = method_defaults["batch_size"]
    cfg.lr = method_defaults["lr"]
    cfg.comparison_optimizer = method_defaults["optimizer"]
    cfg.comparison_scheduler = method_defaults["scheduler"]
    cfg.weight_decay = method_defaults["weight_decay"]
    cfg.comparison_grad_clip = method_defaults["grad_clip"]
    cfg.train.window = int(method_defaults["window_size"])
    cfg.test.window = int(method_defaults["window_size"])
    if args.method == "tip" and args.tip_variant == "official" and args.window_size is None:
        cfg.train.window = 120
        cfg.test.window = 120
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.optimizer is not None:
        cfg.comparison_optimizer = args.optimizer
    if args.scheduler is not None:
        cfg.comparison_scheduler = args.scheduler
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.grad_clip is not None:
        cfg.comparison_grad_clip = args.grad_clip
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.window_size is not None:
        if args.window_size <= 1:
            raise ValueError("--window_size must be greater than 1")
        cfg.train.window = int(args.window_size)
        cfg.test.window = int(args.window_size)
    cfg.debug = bool(args.debug)
    cfg.tip_variant = str(args.tip_variant)
    cfg.transpose_variant = str(args.transpose_variant)
    cfg.globalpose_variant = str(args.globalpose_variant)
    cfg.comparison_max_train_sequences = args.max_train_sequences
    cfg.comparison_max_eval_sequences = args.max_eval_sequences
    if cfg.debug:
        cfg.batch_size = min(int(cfg.batch_size), 4)
        cfg.num_workers = 0
        if cfg.comparison_max_train_sequences is None:
            cfg.comparison_max_train_sequences = 100
        if cfg.comparison_max_eval_sequences is None:
            cfg.comparison_max_eval_sequences = 100
    return cfg


def _to_plain(value):
    if isinstance(value, dict):
        return {key: _to_plain(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _dataset_paths(cfg, split: str) -> list[str]:
    train_datasets = getattr(cfg, "train_datasets", None) or list(cfg.datasets.keys())
    paths = []
    for ds_name in train_datasets:
        if ds_name not in cfg.datasets:
            continue
        ds_cfg = cfg.datasets[ds_name]
        rel = getattr(ds_cfg, f"{split}_path", None)
        if split == "train" and cfg.debug and hasattr(ds_cfg, "debug_path"):
            debug_path = PROJECT_ROOT / ds_cfg.debug_path
            if debug_path.exists():
                rel = ds_cfg.debug_path
        if rel is None:
            continue
        path = PROJECT_ROOT / rel
        if path.exists():
            paths.append(str(path))
    return paths


def _sequence_files(paths: list[str]) -> list[str]:
    files = []
    for path in paths:
        files.extend(str(file_path.resolve()) for file_path in Path(path).glob("*.pt"))
    return sorted(files)


def split_training_files(
    files: list[str], validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Create a deterministic file-level holdout without touching test data."""
    files = sorted(files)
    if validation_fraction <= 0.0 or len(files) < 2:
        return files, []
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    validation = set(shuffled[:validation_count])
    return (
        [path for path in files if path not in validation],
        [path for path in files if path in validation],
    )


def create_loader(
    cfg,
    split: str,
    method: str,
    *,
    sequence_paths: list[str] | None = None,
    training: bool | None = None,
) -> DataLoader | None:
    paths = _dataset_paths(cfg, split)
    if not paths:
        return None
    is_training = split == "train" if training is None else bool(training)
    window_cfg = cfg.train if is_training else cfg.test
    window_size = int(window_cfg.window)
    if method == "tip" and getattr(cfg, "tip_variant", "official") == "legacy":
        # Legacy extension windows include context for its rolling acc-sum.
        window_size += TIP_ACC_SUM_WIN_LEN
    if sequence_paths is None:
        limit = (
            cfg.comparison_max_train_sequences
            if is_training
            else cfg.comparison_max_eval_sequences
        )
        if limit is not None:
            sequence_paths = _sequence_files(paths)[: int(limit)]
    dataset = IMUDataset(
        data_dir=paths,
        sequence_paths=sequence_paths,
        window_size=window_size,
        debug=bool(cfg.debug),
        obj_points_sample_count=int(getattr(cfg, "mesh_downsample_points", 256)),
        simulate_imu_noise=is_training,
        resolve_bimanual_contact_conflicts=bool(getattr(cfg, "resolve_bimanual_contact_conflicts", False)),
        random_window=is_training,
        body_model_path=(
            getattr(cfg, "body_model_path", None)
            if method == "globalpose"
            and getattr(cfg, "globalpose_variant", "official") == "official"
            else None
        ),
        imu_protocol=(
            "dip18_reference"
            if method == "dip18"
            else "tip_reference"
            if method == "tip" and getattr(cfg, "tip_variant", "official") == "official"
            else "transpose_reference"
            if method == "transpose" and getattr(cfg, "transpose_variant", "official") == "official"
            else "globalpose_reference"
            if method == "globalpose" and getattr(cfg, "globalpose_variant", "official") == "official"
            else "imuhoi"
        ),
    )
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=is_training,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=(is_training and len(dataset) >= int(cfg.batch_size)),
        persistent_workers=int(cfg.num_workers) > 0,
    )


def build_model_and_loss(args: argparse.Namespace, device: torch.device):
    if args.method == "dip18":
        # RefCodes/dip18/my_simple/train_obj.py uses w_human=10 and trains the
        # released raw-matrix human branch together with object velocity.
        return DIP18ReferenceHOIModel().to(device), DIPLoss(
            pose_weight=10.0, obj_weight=args.obj_loss_weight
        )
    if args.method == "tip":
        if args.tip_variant == "official":
            return TIPReferenceHOIModel(
                n_heads=args.tip_n_heads,
                # Training follows the released p=0.8 history dropout.
                past_state_dropout=0.8,
            ).to(device), TIPLoss(n_sbps=0, obj_weight=args.obj_loss_weight)
        return TIPHOIModel(n_sbps=args.n_sbps, n_heads=args.tip_n_heads).to(device), TIPLoss(n_sbps=args.n_sbps, obj_weight=args.obj_loss_weight)
    if args.method == "transpose":
        return TransPoseHOIModel(
            transpose_reference=args.transpose_variant == "official"
        ).to(device), TransPoseLoss(obj_weight=args.obj_loss_weight)
    if args.method == "globalpose":
        model = GlobalPoseHOIModel(
            obj_imu_dim=9 if args.globalpose_variant == "official" else 12,
            object_bidirectional=False,
            object_with_init=True,
            human_condition_on_object=False,
        ).to(device)
        model.freeze_reference_human = bool(args.freeze_reference_human)
        return model, GlobalPoseLoss(obj_weight=args.obj_loss_weight)
    raise ValueError(args.method)


def _unwrap_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                checkpoint = state
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Pretrained checkpoint does not contain a state dict")
    state = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }
    if not state:
        raise ValueError("Pretrained checkpoint has no tensor parameters")
    return state


def initialize_from_reference(
    method: str,
    model: torch.nn.Module,
    checkpoint_path: str | os.PathLike,
) -> dict[str, object]:
    """Import only the published human model, leaving object heads trainable."""
    path = Path(checkpoint_path).resolve()
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    source = _unwrap_state_dict(raw)
    target = model.state_dict()
    loaded: list[str] = []

    if method == "dip18" and getattr(model, "dip_variant", "legacy") == "official":
        # The public OMOMO file is a complete PoserWithObject state_dict.  Load
        # it strictly so a missing or silently reshaped branch cannot produce a
        # seemingly valid but untrained comparison checkpoint.
        missing = sorted(set(target) - set(source))
        unexpected = sorted(set(source) - set(target))
        shape_errors = [
            key
            for key in set(source).intersection(target)
            if tuple(source[key].shape) != tuple(target[key].shape)
        ]
        if missing or unexpected or shape_errors:
            raise ValueError(
                "DIP official checkpoint is incompatible with the reference "
                f"architecture (missing={missing[:3]}, unexpected={unexpected[:3]}, "
                f"shape_errors={shape_errors[:3]})"
            )
        model.load_state_dict(source, strict=True)
        return {
            "path": str(path),
            "num_parameters_tensors": len(source),
            "loaded_keys": sorted(source),
            "object_parameters_initialized": True,
            "reference_protocol": "dip18_official_omomo",
        }

    if method == "tip":
        if getattr(model, "tip_reference", False):
            missing = sorted(set(target) - set(source))
            unexpected = sorted(set(source) - set(target))
            shape_errors = [
                key
                for key in set(source).intersection(target)
                if tuple(source[key].shape) != tuple(target[key].shape)
            ]
            if missing or unexpected or shape_errors:
                raise ValueError(
                    "TIP official checkpoint is incompatible with the reference "
                    f"architecture (missing={missing[:3]}, unexpected={unexpected[:3]}, "
                    f"shape_errors={shape_errors[:3]})"
                )
            model.load_state_dict(source, strict=True)
            return {
                "path": str(path),
                "num_parameters_tensors": len(source),
                "loaded_keys": sorted(source),
                "object_parameters_initialized": True,
                "reference_protocol": "tip_official_omomo",
            }
        required_shared = [
            key
            for key in source
            if key not in ("in_linear.weight", "linear.weight", "linear.bias")
        ]
        for key in required_shared:
            if key not in target or target[key].shape != source[key].shape:
                raise ValueError(
                    f"TIP reference parameter {key!r} is incompatible: "
                    f"source={tuple(source[key].shape)}, "
                    f"target={tuple(target[key].shape) if key in target else None}"
                )
            target[key].copy_(source[key])
            loaded.append(key)

        source_input = source.get("in_linear.weight")
        source_output_weight = source.get("linear.weight")
        source_output_bias = source.get("linear.bias")
        if source_input is None or source_output_weight is None or source_output_bias is None:
            raise ValueError("TIP reference is missing input/output projection weights")
        human_state_dim = int(model.human_state_dim)
        expected_source_input = int(model.input_size_imu) + human_state_dim
        if source_input.shape != (target["in_linear.weight"].shape[0], expected_source_input):
            raise ValueError(
                "TIP reference input width does not match human IMU + human state: "
                f"got {tuple(source_input.shape)}, expected second dimension {expected_source_input}"
            )
        if source_output_weight.shape[0] != human_state_dim:
            raise ValueError(
                "TIP reference output does not match the requested SBP count: "
                f"got {source_output_weight.shape[0]}, expected {human_state_dim}"
            )

        target_input = target["in_linear.weight"]
        target_input.zero_()
        imu_end = int(model.input_size_imu)
        object_imu_end = imu_end + int(model.obj_imu_dim)
        target_input[:, :imu_end].copy_(source_input[:, :imu_end])
        target_input[:, object_imu_end : object_imu_end + human_state_dim].copy_(
            source_input[:, imu_end:]
        )
        target["in_linear.bias"].copy_(source["in_linear.bias"])
        target["linear.weight"][:human_state_dim].copy_(source_output_weight)
        target["linear.bias"][:human_state_dim].copy_(source_output_bias)
        loaded.extend(
            ["in_linear.weight", "in_linear.bias", "linear.weight[:human]", "linear.bias[:human]"]
        )
    elif method == "transpose" and getattr(model, "transpose_reference", False):
        mapped_source = {}
        for key, value in source.items():
            if key.startswith("base."):
                key = key[len("base.") :]
            elif key.startswith("object_net.net."):
                key = "obj_head." + key[len("object_net.net.") :]
            mapped_source[key] = value
        missing = sorted(set(target) - set(mapped_source))
        unexpected = sorted(set(mapped_source) - set(target))
        shape_errors = [
            key
            for key in set(mapped_source).intersection(target)
            if tuple(mapped_source[key].shape) != tuple(target[key].shape)
        ]
        if missing or unexpected or shape_errors:
            raise ValueError(
                "TransPose official checkpoint is incompatible with the reference "
                f"architecture (missing={missing[:3]}, unexpected={unexpected[:3]}, "
                f"shape_errors={shape_errors[:3]})"
            )
        model.load_state_dict(mapped_source, strict=True)
        return {
            "path": str(path),
            "num_parameters_tensors": len(mapped_source),
            "loaded_keys": sorted(mapped_source),
            "object_parameters_initialized": True,
            "reference_protocol": "transpose_official_omomo",
        }
    else:
        if method == "transpose":
            remap = lambda key: key
            expected_prefixes = ("pose_s1.", "pose_s2.", "pose_s3.", "tran_b1.", "tran_b2.")
        elif method == "globalpose":
            remap = lambda key: key.replace("iknet.net1.", "iknet1.").replace(
                "iknet.net2.", "iknet2."
            )
            expected_prefixes = ("plnet.", "iknet.net1.", "iknet.net2.", "vrnet.")
        elif method == "dip18":
            raise ValueError(
                "Legacy DIP cannot import a PyTorch reference checkpoint; use "
                "the official 22-joint OMOMO checkpoint with DIP18ReferenceHOIModel"
            )
        else:
            raise ValueError(
                "DIP reference TensorFlow weights are not available in a compatible "
                "PyTorch checkpoint; omit --pretrained for dip18"
            )
        reference_keys = [
            key for key in source if key.startswith(expected_prefixes)
        ]
        if not reference_keys:
            raise ValueError(f"No {method} human parameters found in {path}")
        for source_key in reference_keys:
            target_key = remap(source_key)
            if target_key not in target or target[target_key].shape != source[source_key].shape:
                raise ValueError(
                    f"{method} reference parameter {source_key!r} is incompatible: "
                    f"source={tuple(source[source_key].shape)}, "
                    f"target={tuple(target[target_key].shape) if target_key in target else None}"
                )
            target[target_key].copy_(source[source_key])
            loaded.append(target_key)

    model.load_state_dict(target, strict=True)
    return {
        "path": str(path),
        "num_parameters_tensors": len(loaded),
        "loaded_keys": loaded,
        "object_parameters_initialized": False,
    }


def build_scheduler(cfg, optimizer: torch.optim.Optimizer):
    """Build the per-optimizer-step schedule used by the reference training code."""
    name = str(cfg.comparison_scheduler).lower()
    if name == "none":
        return None
    if name == "dip_exponential":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 0.96 ** (float(step) / 2000.0),
        )
    if name == "tip_cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.epochs) + 850,
        )
    raise ValueError(f"Unsupported scheduler: {cfg.comparison_scheduler!r}")


def model_forward(
    method: str,
    model: torch.nn.Module,
    adapted: dict[str, torch.Tensor],
    tip_state_noise: float = TIP_STATE_NOISE,
) -> dict[str, torch.Tensor]:
    if method == "dip18":
        return model(adapted["imu"], adapted["obj_imu"])
    if method == "tip":
        if getattr(model, "tip_reference", False):
            state_input = adapted["state_target"][:, :-1]
            state_noise = (torch.rand_like(state_input) - 0.5) * (
                2.0 * float(tip_state_noise)
            )
            return model(
                adapted["imu"][:, :-1],
                (state_input + state_noise).nan_to_num(0.0),
            )
        # TIP learns the transition state[t] -> state[t+1] from IMU[t].
        state_input = adapted["state_target"][:, :-1]
        state_noise = (torch.rand_like(state_input) - 0.5) * (
            2.0 * float(tip_state_noise)
        )
        state_input = (state_input + state_noise).nan_to_num(0.0)
        object_state = adapted["obj_vel_target"][:, :-1]
        object_noise = (torch.rand_like(object_state) - 0.5) * (
            2.0 * float(tip_state_noise)
        )
        return model(
            adapted["imu"][:, :-1],
            state_input,
            adapted["obj_imu"][:, :-1],
            prev_obj_velocity=(object_state + object_noise).nan_to_num(0.0),
        )
    if method == "transpose":
        return model(adapted["imu"], adapted["obj_imu"])
    if method == "globalpose":
        return model(
            adapted["x"],
            adapted["obj_imu"],
            apply_gravity_correction=True,
            column_r6d=True,
            pl_init=adapted["pl_init"],
            vr_init=adapted["vr_init"],
        )
    raise ValueError(method)


def model_inference_forward(
    method: str,
    model: torch.nn.Module,
    adapted: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if method == "tip":
        if getattr(model, "tip_reference", False):
            return model.infer_autoregressive(
                adapted["imu"],
                window_size=40,
                initial_state=adapted["initial_state"],
                emit_initial_state=True,
            )
        return model.infer_autoregressive(
            adapted["imu"],
            adapted["obj_imu"],
            window_size=40,
            initial_state=adapted["initial_state"],
            emit_initial_state=True,
        )
    return model_forward(method, model, adapted)


def run_epoch(
    method: str,
    model: torch.nn.Module,
    loss_fn: Callable,
    loader: DataLoader,
    device: torch.device,
    n_sbps: int,
    tip_sbp_cache_root: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    grad_clip: float = 0.0,
    tip_state_noise: float = TIP_STATE_NOISE,
    max_batches: int | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    train = optimizer is not None
    amp_enabled = bool(use_amp and device.type == "cuda")
    if train and scaler is None:
        scaler = GradScaler(enabled=amp_enabled)
    model.train(train)
    if method == "globalpose" and getattr(model, "freeze_reference_human", False):
        # ``model.train()`` would otherwise reactivate dropout in the imported
        # official RNNs, making the frozen human baseline nondeterministic and
        # changing its outputs while the object head is being optimized.
        for module_name in ("plnet", "iknet1", "iknet2", "vrnet"):
            getattr(model, module_name).eval()
        model.obj_head.train(train)
    totals: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(
        tqdm(loader, desc="train" if train else "eval", leave=False)
    ):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = tensor_to_device(batch, device)
        if method == "tip":
            batch = attach_tip_sbp_from_cache(batch, tip_sbp_cache_root, device=device)
        adapted = adapt_batch(
            batch,
            method=method,
            n_sbps=n_sbps,
            protocol_version=COMPARISON_PROTOCOL_VERSION,
            tip_history_context=(
                TIP_ACC_SUM_WIN_LEN - 1
                if method == "tip" and not getattr(model, "tip_reference", False)
                else 0
            ),
            dip_variant=("official" if method == "dip18" else "legacy"),
            tip_variant=("official" if getattr(model, "tip_reference", False) else "legacy"),
            transpose_variant=(
                "official" if getattr(model, "transpose_reference", False) else "legacy"
            ),
            globalpose_obj_imu_dim=(
                int(getattr(model, "obj_imu_dim", 12))
                if method == "globalpose"
                else 12
            ),
            globalpose_init_mode=(
                # RefCodes/GlobalPose's RNNWithInitDataset supplies the first
                # target frame to both PL and VR during train/validation. The
                # deployable zero-VR path is selected only by eval_comparison.
                "reference_train" if method == "globalpose" else "deploy"
            ),
        )
        with autocast(enabled=amp_enabled):
            output = (
                model_forward(
                    method,
                    model,
                    adapted,
                    tip_state_noise=tip_state_noise,
                )
                if train
                else model_inference_forward(method, model, adapted)
            )
            loss_dict = loss_fn(output, adapted)
            loss = loss_dict["loss"]
        if train:
            optimizer.zero_grad(set_to_none=True)
            assert scaler is not None
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_succeeded = scaler.get_scale() >= scale_before
            if scheduler is not None and optimizer_step_succeeded:
                scheduler.step()
        count += 1
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Resume checkpoint does not contain a model state: {path}")
    return checkpoint


def _project_relative_paths(paths: list[str]) -> list[str]:
    relative = []
    for value in paths:
        path = Path(value).resolve()
        try:
            relative.append(str(path.relative_to(PROJECT_ROOT)))
        except ValueError:
            relative.append(str(path))
    return relative


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    seed = int(getattr(cfg, "seed", 42))
    validation_seed = seed if args.validation_seed is None else args.validation_seed
    setup_seed(seed)
    device = torch.device(
        getattr(cfg, "device", "cuda:0") if torch.cuda.is_available() else "cpu"
    )
    use_amp = bool(args.amp and device.type == "cuda")

    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is not None and args.pretrained is not None:
        raise ValueError("--resume and --pretrained are mutually exclusive")
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    if resume_path is not None and args.run_name is None:
        save_dir = resume_path.parent
        run_name = save_dir.name
    else:
        run_name = args.run_name or f"{args.method}_{datetime.now().strftime('%m%d%H%M')}"
        if cfg.debug and args.run_name is None:
            run_name += "_debug"
        save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = save_dir / "split_manifest.json"
    if resume_path is not None and manifest_path.exists():
        with open(manifest_path, "r") as handle:
            existing_manifest = json.load(handle)

        def _manifest_paths(values: list[str]) -> list[str]:
            resolved = []
            for value in values:
                path = Path(value)
                if not path.is_absolute():
                    path = PROJECT_ROOT / path
                if not path.exists():
                    raise FileNotFoundError(
                        f"A sequence from the saved split no longer exists: {path}"
                    )
                resolved.append(str(path.resolve()))
            return resolved

        training_files = _manifest_paths(existing_manifest["training_files"])
        validation_files = _manifest_paths(existing_manifest["validation_files"])
        validation_seed = int(existing_manifest["seed"])
        args.validation_fraction = float(
            existing_manifest["validation_fraction"]
        )
        validation_source = str(existing_manifest["validation_source"])
    else:
        training_roots = _dataset_paths(cfg, "train")
        all_training_files = _sequence_files(training_roots)
        if cfg.comparison_max_train_sequences is not None:
            all_training_files = all_training_files[
                : int(cfg.comparison_max_train_sequences)
            ]
        training_files, validation_files = split_training_files(
            all_training_files,
            validation_fraction=float(args.validation_fraction),
            seed=int(validation_seed),
        )
        if cfg.comparison_max_eval_sequences is not None and validation_files:
            validation_files = validation_files[
                : int(cfg.comparison_max_eval_sequences)
            ]
        validation_source = (
            "held_out_training_files"
            if validation_files
            else "configured_test_path"
        )

    train_loader = create_loader(
        cfg,
        "train",
        args.method,
        sequence_paths=training_files,
        training=True,
    )
    if train_loader is None:
        raise RuntimeError("No training data found for comparison training")
    if validation_files:
        eval_loader = create_loader(
            cfg,
            "train",
            args.method,
            sequence_paths=validation_files,
            training=False,
        )
    else:
        eval_loader = create_loader(cfg, "test", args.method, training=False)

    split_manifest = {
        "seed": int(validation_seed),
        "validation_fraction": float(args.validation_fraction),
        "validation_source": validation_source,
        "training_files": _project_relative_paths(training_files),
        "validation_files": _project_relative_paths(validation_files),
    }
    with open(manifest_path, "w") as handle:
        json.dump(split_manifest, handle, indent=2)

    model, loss_fn = build_model_and_loss(args, device)
    pretrained_info = None
    if args.pretrained is not None:
        pretrained_path = Path(args.pretrained)
        if not pretrained_path.is_absolute():
            pretrained_path = PROJECT_ROOT / pretrained_path
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained reference weights not found: {pretrained_path}"
            )
        pretrained_info = initialize_from_reference(
            args.method, model, pretrained_path
        )
        print(
            f"initialized {pretrained_info['num_parameters_tensors']} human "
            f"parameter tensors from {pretrained_path}"
        )
    if args.freeze_reference_human:
        # Build the optimizer only after disabling gradients, otherwise AdamW
        # would still update the imported human network despite its name.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("obj_head."))
        trainable = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("Reference-human freeze left no trainable parameters")
        print(
            f"freezing GlobalPose human network; trainable tensors: {len(trainable)}"
        )
    optimizer_class = (
        torch.optim.AdamW
        if str(cfg.comparison_optimizer).lower() == "adamw"
        else torch.optim.Adam
    )
    optimizer = optimizer_class(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(getattr(cfg, "lr", 4e-4)),
        weight_decay=float(getattr(cfg, "weight_decay", 1e-5)),
    )
    scheduler = build_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=use_amp)

    best_eval = float("inf")
    start_epoch = 1
    checks_without_improvement = 0
    if resume_path is not None:
        resume = _load_checkpoint(resume_path)
        if str(resume.get("method", args.method)).lower() != args.method:
            raise ValueError(
                f"Resume checkpoint method is {resume.get('method')!r}, not {args.method!r}"
            )
        if int(resume.get("comparison_protocol_version", -1)) != COMPARISON_PROTOCOL_VERSION:
            raise ValueError("Resume checkpoint protocol does not match corrected protocol v2")
        if int(resume.get("n_sbps", args.n_sbps)) != args.n_sbps:
            raise ValueError("Resume checkpoint n_sbps does not match --n_sbps")
        model.load_state_dict(resume["model"], strict=True)
        if resume.get("optimizer") is None:
            raise ValueError("Resume checkpoint has no optimizer state; use last.pt")
        optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None:
            if resume.get("scheduler") is None:
                raise ValueError("Resume checkpoint has no scheduler state")
            scheduler.load_state_dict(resume["scheduler"])
        if resume.get("scaler") is not None:
            scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        best_eval = float(resume.get("best_eval", float("inf")))
        checks_without_improvement = int(
            resume.get("checks_without_improvement", 0)
        )
        pretrained_info = resume.get("pretrained")
        print(f"resumed {resume_path} at epoch {start_epoch}")

    config_payload = {
        "method": args.method,
        "source_cfg": args.cfg,
        "run_name": run_name,
        "n_sbps": args.n_sbps,
        "tip_sbp_cache_root": args.tip_sbp_cache_root,
        "obj_loss_weight": args.obj_loss_weight,
        "comparison_protocol_version": COMPARISON_PROTOCOL_VERSION,
        "dip_variant": "official" if args.method == "dip18" else None,
        "tip_variant": args.tip_variant if args.method == "tip" else None,
        "transpose_variant": args.transpose_variant if args.method == "transpose" else None,
        "globalpose_variant": args.globalpose_variant if args.method == "globalpose" else None,
        "globalpose_train_init_mode": (
            "reference_train" if args.method == "globalpose" else None
        ),
        "object_output_mode": OBJECT_OUTPUT_MODE,
        "object_velocity_frame": (
            "initial_root"
            if args.method == "tip"
            else "current_root"
            if args.method == "globalpose"
            else "world"
        ),
        "object_head_causal": args.method in ("transpose", "globalpose"),
        "human_condition_on_object": args.method == "tip",
        "tip_n_heads": args.tip_n_heads if args.method == "tip" else None,
        "tip_state_noise": args.tip_state_noise if args.method == "tip" else None,
        "tip_object_state_feedback": (
            False if args.method == "tip" and args.tip_variant == "official"
            else args.method == "tip"
        ),
        "window_size": int(cfg.train.window),
        "num_training_files": len(training_files),
        "num_validation_files": len(validation_files),
        "validation_source": validation_source,
        "validation_fraction": float(args.validation_fraction),
        "validation_seed": int(validation_seed),
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "early_stopping_patience": args.early_stopping_patience,
        "amp": use_amp,
        "pretrained": pretrained_info,
        "freeze_reference_human": bool(args.freeze_reference_human),
        "max_train_sequences": cfg.comparison_max_train_sequences,
        "max_eval_sequences": cfg.comparison_max_eval_sequences,
        "optimizer": str(cfg.comparison_optimizer),
        "scheduler": str(cfg.comparison_scheduler),
        "grad_clip": float(cfg.comparison_grad_clip),
        "cfg": _to_plain(cfg),
    }
    with open(save_dir / "config.yaml", "w") as handle:
        yaml.safe_dump(
            config_payload,
            handle,
            sort_keys=False,
            allow_unicode=True,
        )

    def checkpoint_payload(
        epoch: int,
        metrics: dict[str, float],
        *,
        include_training_state: bool,
    ) -> dict:
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "best_eval": best_eval,
            "checks_without_improvement": checks_without_improvement,
            "method": args.method,
            "n_sbps": args.n_sbps,
            "comparison_protocol_version": COMPARISON_PROTOCOL_VERSION,
            "dip_variant": "official" if args.method == "dip18" else None,
            "tip_variant": args.tip_variant if args.method == "tip" else None,
            "transpose_variant": args.transpose_variant if args.method == "transpose" else None,
            "globalpose_variant": (
                args.globalpose_variant if args.method == "globalpose" else None
            ),
            "globalpose_train_init_mode": (
                "reference_train" if args.method == "globalpose" else None
            ),
            "object_output_mode": OBJECT_OUTPUT_MODE,
            "object_velocity_frame": config_payload["object_velocity_frame"],
            "object_head_causal": args.method in ("transpose", "globalpose"),
            "human_condition_on_object": args.method == "tip",
            "tip_n_heads": args.tip_n_heads if args.method == "tip" else None,
            "tip_state_noise": args.tip_state_noise if args.method == "tip" else None,
            "tip_object_state_feedback": (
                False if args.method == "tip" and args.tip_variant == "official"
                else args.method == "tip"
            ),
            "window_size": int(cfg.train.window),
            "num_training_files": len(training_files),
            "num_validation_files": len(validation_files),
            "validation_source": validation_source,
            "validation_fraction": float(args.validation_fraction),
            "validation_seed": int(validation_seed),
            "optimizer_name": str(cfg.comparison_optimizer),
            "scheduler_name": str(cfg.comparison_scheduler),
            "grad_clip": float(cfg.comparison_grad_clip),
            "amp": use_amp,
            "pretrained": pretrained_info,
            "freeze_reference_human": bool(args.freeze_reference_human),
            "model_kwargs": (
                {
                    "n_sbps": args.n_sbps,
                    "n_heads": args.tip_n_heads,
                    "object_state_feedback": (
                        False if args.tip_variant == "official" else True
                    ),
                    "tip_variant": args.tip_variant,
                }
                if args.method == "tip"
                else {"human_condition_on_object": False}
                if args.method == "dip18"
                else {
                    "object_bidirectional": False,
                    "object_condition_on_full": False,
                    "human_condition_on_object": False,
                    "transpose_variant": args.transpose_variant,
                }
                if args.method == "transpose"
                else {
                    "object_bidirectional": False,
                    "object_with_init": True,
                    "human_condition_on_object": False,
                    "obj_imu_dim": 9 if args.globalpose_variant == "official" else 12,
                    "globalpose_variant": args.globalpose_variant,
                }
                if args.method == "globalpose"
                else {}
            ),
        }
        if include_training_state:
            payload.update(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "scaler": scaler.state_dict(),
                }
            )
        return payload

    epochs = int(getattr(cfg, "epochs", 1))
    history_path = save_dir / "history.jsonl"
    if start_epoch > epochs:
        print(f"checkpoint already reached epoch {start_epoch - 1}; target is {epochs}")
        return

    last_epoch = start_epoch - 1
    last_train_metrics: dict[str, float] = {}
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        train_metrics = run_epoch(
            args.method,
            model,
            loss_fn,
            train_loader,
            device,
            args.n_sbps,
            tip_sbp_cache_root=args.tip_sbp_cache_root,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=float(cfg.comparison_grad_clip),
            tip_state_noise=args.tip_state_noise,
            max_batches=args.max_train_batches,
            scaler=scaler,
            use_amp=use_amp,
        )
        last_train_metrics = train_metrics
        print(f"epoch {epoch:04d} train {train_metrics}")

        should_validate = eval_loader is not None and (
            epoch == 1 or epoch % args.eval_every == 0 or epoch == epochs
        )
        eval_metrics = None
        improved = False
        if should_validate:
            with torch.no_grad():
                eval_metrics = run_epoch(
                    args.method,
                    model,
                    loss_fn,
                    eval_loader,
                    device,
                    args.n_sbps,
                    tip_sbp_cache_root=args.tip_sbp_cache_root,
                    max_batches=args.max_eval_batches,
                    use_amp=use_amp,
                )
            print(f"epoch {epoch:04d} eval {eval_metrics}")
            if eval_metrics.get("loss", float("inf")) < best_eval:
                best_eval = float(eval_metrics["loss"])
                checks_without_improvement = 0
                improved = True
                _atomic_torch_save(
                    checkpoint_payload(
                        epoch, eval_metrics, include_training_state=False
                    ),
                    save_dir / "best.pt",
                )
            else:
                checks_without_improvement += 1
        elif eval_loader is None and train_metrics["loss"] < best_eval:
            best_eval = float(train_metrics["loss"])
            improved = True
            _atomic_torch_save(
                checkpoint_payload(
                    epoch, train_metrics, include_training_state=False
                ),
                save_dir / "best.pt",
            )

        history_record = {
            "epoch": epoch,
            "train": train_metrics,
            "eval": eval_metrics,
            "improved": improved,
            "best_eval": best_eval,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "amp_scale": scaler.get_scale(),
        }
        with open(history_path, "a") as handle:
            handle.write(json.dumps(history_record, allow_nan=False) + "\n")

        should_save_last = (
            epoch % args.save_every == 0 or should_validate or epoch == epochs
        )
        if should_save_last:
            _atomic_torch_save(
                checkpoint_payload(
                    epoch, train_metrics, include_training_state=True
                ),
                save_dir / "last.pt",
            )
        if (
            should_validate
            and args.early_stopping_patience > 0
            and checks_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early stopping after {checks_without_improvement} validation checks "
                "without improvement"
            )
            break

    if not (save_dir / "last.pt").exists() or last_epoch % args.save_every != 0:
        _atomic_torch_save(
            checkpoint_payload(
                last_epoch, last_train_metrics, include_training_state=True
            ),
            save_dir / "last.pt",
        )
    print(f"saved comparison run to {save_dir}")


if __name__ == "__main__":
    main()
