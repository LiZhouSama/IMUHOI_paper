"""Train IMUHOI comparison baselines.

Example:
    conda activate SAGE
    python -m Comparisons.train_comparison --method transpose --debug --epochs 1
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_IMUHOI import IMUDataset
from Comparisons.common.adapters import adapt_batch, tensor_to_device
from Comparisons.common.losses import DIPLoss, GlobalPoseLoss, TIPLoss, TransPoseLoss
from Comparisons.dip18 import DIP18HOIModel
from Comparisons.globalpose import GlobalPoseHOIModel
from Comparisons.tip import TIPHOIModel
from Comparisons.tip.sbp_cache import attach_tip_sbp_from_cache
from Comparisons.transpose import TransPoseHOIModel
from utils.utils import setup_seed


METHODS = ("dip18", "tip", "transpose", "globalpose")
METHOD_TRAINING_DEFAULTS = {
    "dip18": {"batch_size": 160, "lr": 1e-3},
    "transpose": {"batch_size": 200, "lr": 2e-3},
    "tip": {"batch_size": 256, "lr": 1e-4},
    "globalpose": {"batch_size": 200, "lr": 3e-3},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-method comparison baselines on IMUHOI")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--cfg", default="configs/IMUHOI_train_rnn.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_sbps", type=int, default=5, help="TIP SBP count; labels are masked if absent")
    parser.add_argument(
        "--tip_sbp_cache_root",
        default="Comparisons/tip/sbp_cache",
        help="TIP sidecar SBP cache root; only used by --method tip",
    )
    parser.add_argument("--obj_loss_weight", type=float, default=10.0)
    parser.add_argument("--save_dir", default="outputs/Comparisons")
    parser.add_argument("--run_name", default=None, help="Optional fixed run directory name under --save_dir")
    return parser.parse_args()


def load_cfg(args: argparse.Namespace):
    with open(args.cfg, "r") as f:
        cfg = edict(yaml.safe_load(f))
    method_defaults = METHOD_TRAINING_DEFAULTS[args.method]
    cfg.batch_size = method_defaults["batch_size"]
    cfg.lr = method_defaults["lr"]
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    cfg.debug = bool(args.debug)
    if cfg.debug:
        cfg.batch_size = min(int(cfg.batch_size), 4)
        cfg.num_workers = 0
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
        if cfg.debug and hasattr(ds_cfg, "debug_path"):
            rel = ds_cfg.debug_path
        else:
            rel = getattr(ds_cfg, f"{split}_path", None)
        if rel is None:
            continue
        path = PROJECT_ROOT / rel
        if path.exists():
            paths.append(str(path))
    return paths


def create_loader(cfg, split: str) -> DataLoader | None:
    paths = _dataset_paths(cfg, split)
    if not paths:
        return None
    window_cfg = cfg.train if split == "train" else cfg.test
    dataset = IMUDataset(
        data_dir=paths,
        window_size=int(window_cfg.window),
        debug=bool(cfg.debug),
        obj_points_sample_count=int(getattr(cfg, "mesh_downsample_points", 256)),
        simulate_imu_noise=(split == "train"),
        resolve_bimanual_contact_conflicts=bool(getattr(cfg, "resolve_bimanual_contact_conflicts", False)),
    )
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=(split == "train"),
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        drop_last=(split == "train"),
    )


def build_model_and_loss(args: argparse.Namespace, device: torch.device):
    if args.method == "dip18":
        return DIP18HOIModel().to(device), DIPLoss(obj_weight=args.obj_loss_weight)
    if args.method == "tip":
        return TIPHOIModel(n_sbps=args.n_sbps).to(device), TIPLoss(n_sbps=args.n_sbps, obj_weight=args.obj_loss_weight)
    if args.method == "transpose":
        return TransPoseHOIModel().to(device), TransPoseLoss(obj_weight=args.obj_loss_weight)
    if args.method == "globalpose":
        return GlobalPoseHOIModel().to(device), GlobalPoseLoss(obj_weight=args.obj_loss_weight)
    raise ValueError(args.method)


def model_forward(method: str, model: torch.nn.Module, adapted: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if method == "dip18":
        return model(adapted["imu"], adapted["obj_imu"])
    if method == "tip":
        return model(adapted["imu"], adapted["prev_state"], adapted["obj_imu"])
    if method == "transpose":
        return model(adapted["imu"], adapted["obj_imu"])
    if method == "globalpose":
        return model(adapted["x"], adapted["obj_imu"])
    raise ValueError(method)


def run_epoch(
    method: str,
    model: torch.nn.Module,
    loss_fn: Callable,
    loader: DataLoader,
    device: torch.device,
    n_sbps: int,
    tip_sbp_cache_root: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    for batch in tqdm(loader, desc="train" if train else "eval", leave=False):
        batch = tensor_to_device(batch, device)
        if method == "tip":
            batch = attach_tip_sbp_from_cache(batch, tip_sbp_cache_root, device=device)
        adapted = adapt_batch(batch, method=method, n_sbps=n_sbps)
        output = model_forward(method, model, adapted)
        loss_dict = loss_fn(output, adapted)
        loss = loss_dict["loss"]
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count += 1
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    setup_seed(int(getattr(cfg, "seed", 42)))
    device = torch.device(getattr(cfg, "device", "cuda:0") if torch.cuda.is_available() else "cpu")

    train_loader = create_loader(cfg, "train")
    if train_loader is None:
        raise RuntimeError("No training data found for comparison training")
    test_loader = create_loader(cfg, "test")

    model, loss_fn = build_model_and_loss(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(getattr(cfg, "lr", 4e-4)),
        weight_decay=float(getattr(cfg, "weight_decay", 1e-5)),
    )

    run_name = args.run_name or f"{args.method}_{datetime.now().strftime('%m%d%H%M')}"
    if cfg.debug and args.run_name is None:
        run_name += "_debug"
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "method": args.method,
                "source_cfg": args.cfg,
                "n_sbps": args.n_sbps,
                "tip_sbp_cache_root": args.tip_sbp_cache_root,
                "obj_loss_weight": args.obj_loss_weight,
                "cfg": _to_plain(cfg),
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    best_eval = float("inf")
    epochs = int(getattr(cfg, "epochs", 1))
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            args.method,
            model,
            loss_fn,
            train_loader,
            device,
            args.n_sbps,
            tip_sbp_cache_root=args.tip_sbp_cache_root,
            optimizer=optimizer,
        )
        print(f"epoch {epoch:04d} train {train_metrics}")
        if test_loader is not None:
            with torch.no_grad():
                eval_metrics = run_epoch(
                    args.method,
                    model,
                    loss_fn,
                    test_loader,
                    device,
                    args.n_sbps,
                    tip_sbp_cache_root=args.tip_sbp_cache_root,
                )
            print(f"epoch {epoch:04d} eval {eval_metrics}")
            if eval_metrics["loss"] < best_eval:
                best_eval = eval_metrics["loss"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": eval_metrics}, save_dir / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": train_metrics}, save_dir / "last.pt")

    print(f"saved comparison run to {save_dir}")


if __name__ == "__main__":
    main()
