"""Export GlobalPose comparison inference artifacts for Windows physics refine.

The official GlobalPose physics stage depends on the Windows-only prebuilt
carticulate package.  This script runs the IMUHOI GlobalPoseHOI neural path on
Linux and stores per-sequence tensors that are sufficient for a Windows-side
physics wrapper to reconstruct the PL/IK/VR outputs and aligned IMU streams.

Example:
    python -m Comparisons.export_globalpose_for_physics \
        --checkpoint outputs/Comparisons/globalpose_xxx/best.pt \
        --dataset omomo --split test \
        --output_dir outputs/Comparisons/globalpose_physics_export/omomo_test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import FRAME_RATE
from dataset.dataset_IMUHOI import IMUDataset
from Comparisons.common.adapters import (
    ROOT_LAST_ORDER_FROM_IMUHOI,
    _root_last_global_imu,
    adapt_batch,
    tensor_to_device,
)
from Comparisons.common.geometry import rotation_angular_velocity
from Comparisons.globalpose import GlobalPoseHOIModel
from utils.rotation_conversions import rotation_6d_to_matrix
from utils.utils import setup_seed


DEFAULT_OUTPUT_DIR = "outputs/Comparisons/globalpose_physics_export"
RAW_TENSOR_KEYS = (
    "human_imu",
    "obj_imu",
    "trans",
    "pose",
    "position_global",
    "rotation_global",
    "root_vel",
    "sensor_vel_root",
    "sensor_vel_glb",
    "obj_trans",
    "obj_rot",
    "obj_scale",
    "obj_vel",
    "lfoot_contact",
    "rfoot_contact",
    "lhand_contact",
    "rhand_contact",
    "obj_contact",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Linux GlobalPoseHOI inference tensors for Windows carticulate physics refine."
    )
    parser.add_argument("--cfg", default="configs/IMUHOI_train_rnn.yaml")
    parser.add_argument("--checkpoint", default=None, help="GlobalPoseHOI checkpoint from train_comparison.py")
    parser.add_argument("--allow_random_weights", action="store_true", help="Only for smoke tests; do not use for eval")
    parser.add_argument("--dataset", default="omomo", help="Dataset key under cfg.datasets")
    parser.add_argument("--split", choices=("train", "test", "debug"), default="test")
    parser.add_argument(
        "--data_dir",
        action="append",
        default=None,
        help="Override cfg dataset path. Can be passed multiple times.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=float(FRAME_RATE))
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Export one deterministic-seed random window per sequence instead of full sequences.",
    )
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Use dataset debug mode and cap sequence count")
    parser.add_argument("--include_object_points", action="store_true")
    return parser.parse_args()


def load_cfg(path: str | Path) -> edict:
    with open(path, "r") as f:
        return edict(yaml.safe_load(f))


def resolve_data_dirs(args: argparse.Namespace, cfg: edict) -> list[str]:
    if args.data_dir:
        paths = []
        for path in args.data_dir:
            data_path = Path(path)
            if not data_path.is_absolute():
                data_path = PROJECT_ROOT / data_path
            paths.append(str(data_path.resolve()))
        return paths

    if not hasattr(cfg, "datasets") or args.dataset not in cfg.datasets:
        available = sorted(getattr(cfg, "datasets", {}).keys())
        raise KeyError(f"Dataset {args.dataset!r} not found in config. Available: {available}")

    ds_cfg = cfg.datasets[args.dataset]
    path_key = "debug_path" if args.split == "debug" else f"{args.split}_path"
    if not hasattr(ds_cfg, path_key):
        raise KeyError(f"Dataset {args.dataset!r} does not define {path_key}")
    data_path = Path(getattr(ds_cfg, path_key))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    return [str(data_path.resolve())]


def load_model(args: argparse.Namespace, device: torch.device) -> GlobalPoseHOIModel:
    model = GlobalPoseHOIModel().to(device)
    if args.checkpoint is None:
        if not args.allow_random_weights:
            raise ValueError("--checkpoint is required unless --allow_random_weights is set")
        print("WARNING: exporting with random GlobalPoseHOI weights; this is only useful for smoke tests.")
        return model

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded GlobalPoseHOI checkpoint: {checkpoint_path}")
    return model


def create_loader(args: argparse.Namespace, cfg: edict, data_dirs: list[str]) -> DataLoader:
    window_size = args.window_size
    if window_size is None:
        split_cfg = cfg.train if args.split == "train" else cfg.test
        window_size = int(getattr(split_cfg, "window", 120))

    dataset = IMUDataset(
        data_dir=data_dirs,
        window_size=int(window_size),
        debug=bool(args.debug),
        full_sequence=not bool(args.windowed),
        obj_points_sample_count=int(getattr(cfg, "mesh_downsample_points", 256)),
        simulate_imu_noise=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No valid sequences found under: {data_dirs}")
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def tensor_to_cpu(value: torch.Tensor, squeeze_batch: bool = True) -> torch.Tensor:
    out = value.detach().cpu()
    if squeeze_batch and out.dim() > 0 and out.shape[0] == 1:
        out = out[0]
    return out.contiguous()


def batch_scalar(batch: dict[str, Any], key: str, default: Any = None) -> Any:
    value = batch.get(key, default)
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        if value.shape[0] == 1:
            return value[0].tolist()
        return value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def safe_filename(seq_name: str, seq_path: str, window_start: int, window_end: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(seq_name)).strip("._")
    if not safe:
        safe = "sequence"
    digest = hashlib.sha1(str(seq_path).encode("utf-8")).hexdigest()[:10]
    return f"{safe}_{window_start:06d}_{window_end:06d}_{digest}.pt"


def normalize_vector(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def split_globalpose_input(x: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "x": tensor_to_cpu(x),
        "a_rb": tensor_to_cpu(x[..., :18].reshape(x.shape[0], x.shape[1], 6, 3)),
        "w_rb": tensor_to_cpu(x[..., 18:36].reshape(x.shape[0], x.shape[1], 6, 3)),
        "r_rb": tensor_to_cpu(x[..., 36:81].reshape(x.shape[0], x.shape[1], 5, 3, 3)),
        "g_r0": tensor_to_cpu(x[..., 81:84]),
    }


def build_export_record(
    batch: dict[str, Any],
    adapted: dict[str, torch.Tensor],
    output: dict[str, torch.Tensor],
    sensor_acc_world: torch.Tensor,
    sensor_w_world: torch.Tensor,
    sensor_rot_world: torch.Tensor,
    args: argparse.Namespace,
    cfg_path: Path,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    seq_name = str(batch_scalar(batch, "seq_name", "unknown"))
    seq_path = str(batch_scalar(batch, "seq_path", ""))
    window_start = int(batch_scalar(batch, "window_start", 0))
    window_end = int(batch_scalar(batch, "window_end", adapted["x"].shape[1]))

    ik2_rotmat = rotation_6d_to_matrix(output["ik2"].reshape(-1, 6)).reshape(
        output["ik2"].shape[0], output["ik2"].shape[1], 15, 3, 3
    )
    stationary_prob = output["vr"][..., 4:].sigmoid()

    raw = {key: tensor_to_cpu(batch[key]) for key in RAW_TENSOR_KEYS if torch.is_tensor(batch.get(key))}
    if args.include_object_points and torch.is_tensor(batch.get("obj_points_canonical")):
        raw["obj_points_canonical"] = tensor_to_cpu(batch["obj_points_canonical"])

    record = {
        "meta": {
            "format": "imuhoi_globalpose_physics_export_v1",
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "seq_name": seq_name,
            "seq_path": seq_path,
            "seq_file": str(batch_scalar(batch, "seq_file", "")),
            "window_start": window_start,
            "window_end": window_end,
            "full_sequence": not bool(args.windowed),
            "dataset": args.dataset,
            "split": args.split,
            "cfg": str(cfg_path),
            "checkpoint": checkpoint_path,
            "fps": float(args.fps),
            "root_last_order_from_imuhoi": list(ROOT_LAST_ORDER_FROM_IMUHOI),
            "has_object": bool(batch_scalar(batch, "has_object", False)),
            "obj_name": str(batch_scalar(batch, "obj_name", "")),
        },
        "inputs": {
            **split_globalpose_input(adapted["x"]),
            "obj_imu_12d": tensor_to_cpu(adapted["obj_imu"]),
            "sensor_acc_world": tensor_to_cpu(sensor_acc_world),
            "sensor_ang_vel_world": tensor_to_cpu(sensor_w_world),
            "sensor_rot_world": tensor_to_cpu(sensor_rot_world),
        },
        "pred": {
            "human": tensor_to_cpu(output["human"]),
            "pl": tensor_to_cpu(output["pl"]),
            "ik1": tensor_to_cpu(output["ik1"]),
            "ik2_6d": tensor_to_cpu(output["ik2"]),
            "ik2_rotmat": tensor_to_cpu(ik2_rotmat),
            "vr": tensor_to_cpu(output["vr"]),
            "stationary_prob": tensor_to_cpu(stationary_prob),
            "g_r1": tensor_to_cpu(normalize_vector(output["pl"][..., 15:18])),
            "g_r2": tensor_to_cpu(normalize_vector(output["ik1"][..., 69:72])),
            "obj_trans": tensor_to_cpu(output["obj_trans"]),
        },
        "targets": {
            "globalpose_target": tensor_to_cpu(adapted["target"]),
            "obj_trans": tensor_to_cpu(adapted["obj_trans"]),
            "has_object": tensor_to_cpu(adapted["has_object"], squeeze_batch=False),
        },
        "raw": raw,
    }
    return record


def record_summary(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    def shape_of(container: dict[str, Any], key: str) -> list[int] | None:
        value = container.get(key)
        return list(value.shape) if torch.is_tensor(value) else None

    return {
        "file": str(path),
        "seq_name": record["meta"]["seq_name"],
        "seq_path": record["meta"]["seq_path"],
        "window_start": record["meta"]["window_start"],
        "window_end": record["meta"]["window_end"],
        "has_object": record["meta"]["has_object"],
        "x_shape": shape_of(record["inputs"], "x"),
        "pl_shape": shape_of(record["pred"], "pl"),
        "ik1_shape": shape_of(record["pred"], "ik1"),
        "ik2_shape": shape_of(record["pred"], "ik2_6d"),
        "vr_shape": shape_of(record["pred"], "vr"),
    }


def main() -> None:
    args = parse_args()
    setup_seed(int(args.seed))
    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = load_cfg(cfg_path)
    data_dirs = resolve_data_dirs(args, cfg)

    device_name = args.device or getattr(cfg, "device", "cuda:0")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    loader = create_loader(args, cfg, data_dirs)
    model = load_model(args, device)
    model.eval()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_arg = Path(args.checkpoint)
        if not checkpoint_arg.is_absolute():
            checkpoint_arg = PROJECT_ROOT / checkpoint_arg
        checkpoint_path = str(checkpoint_arg.resolve())

    manifest = {
        "format": "imuhoi_globalpose_physics_export_manifest_v1",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "cfg": str(cfg_path),
        "checkpoint": checkpoint_path,
        "dataset": args.dataset,
        "split": args.split,
        "data_dirs": data_dirs,
        "output_dir": str(output_dir),
        "full_sequence": not bool(args.windowed),
        "window_size": int(args.window_size or getattr(cfg.test if args.split != "train" else cfg.train, "window", 120)),
        "fps": float(args.fps),
        "files": [],
    }

    exported = 0
    processed = 0
    skipped = 0
    limit = args.max_sequences
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="export_globalpose")):
            if limit is not None and processed >= limit:
                break
            processed += 1

            batch = tensor_to_device(batch, device)
            adapted = adapt_batch(batch, method="globalpose", fps=float(args.fps))
            sensor_acc_world, sensor_rot_world = _root_last_global_imu(batch, fps=float(args.fps))
            sensor_w_world = rotation_angular_velocity(sensor_rot_world, dt=1.0 / float(args.fps))
            output = model(adapted["x"], adapted["obj_imu"])

            seq_name = str(batch_scalar(batch, "seq_name", f"batch_{batch_idx}"))
            seq_path = str(batch_scalar(batch, "seq_path", ""))
            window_start = int(batch_scalar(batch, "window_start", 0))
            window_end = int(batch_scalar(batch, "window_end", adapted["x"].shape[1]))
            out_path = output_dir / safe_filename(seq_name, seq_path, window_start, window_end)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            record = build_export_record(
                batch=batch,
                adapted=adapted,
                output=output,
                sensor_acc_world=sensor_acc_world,
                sensor_w_world=sensor_w_world,
                sensor_rot_world=sensor_rot_world,
                args=args,
                cfg_path=cfg_path,
                checkpoint_path=checkpoint_path,
            )
            torch.save(record, out_path)
            manifest["files"].append(record_summary(out_path, record))
            exported += 1

    manifest["num_exported"] = exported
    manifest["num_processed"] = processed
    manifest["num_skipped_existing"] = skipped
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Exported {exported} sequences to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} existing files; pass --overwrite to regenerate them.")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
