"""Diagnose hand-contact F1 calibration.

This script compares:
  1. pred_hand_contact_prob       (unconditional: object_prob * hand_cond_prob)
  2. pred_hand_contact_prob_cond  (conditional hand contact probability)

It also sweeps left/right thresholds to check whether the default 0.5 cutoff is
the main source of low F1.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_IMUHOI import IMUDataset
from eval_IMUHOI import (
    _apply_cli_module_overrides,
    _filter_pipeline_module_paths,
    _infer_model_kind,
    get_default_dataset_config,
)
from model import load_model
from utils.utils import build_model_input_dict, load_config


Record = Dict[str, np.ndarray]


def _as_dict(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "items"):
        return dict(value)
    return {}


def _get_cfg_value(container, key: str, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _resolve_path(path_like) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _resolve_data_runs(args, config) -> List[Tuple[str, Path]]:
    if args.test_data_dir:
        return [("custom", _resolve_path(args.test_data_dir))]

    default_config = get_default_dataset_config(args.no_trans)
    if args.dataset:
        if args.dataset in default_config:
            return [(args.dataset, _resolve_path(default_config[args.dataset]["data_dir"]))]

        datasets_cfg = _as_dict(getattr(config, "datasets", {}))
        dataset_cfg = _as_dict(datasets_cfg.get(args.dataset))
        test_path = dataset_cfg.get("test_path")
        if test_path:
            return [(args.dataset, _resolve_path(test_path))]

        valid = sorted(set(default_config.keys()) | set(datasets_cfg.keys()))
        raise ValueError(f"Unknown dataset '{args.dataset}'. Valid keys: {valid}")

    return [(name, _resolve_path(cfg["data_dir"])) for name, cfg in default_config.items()]


def _build_module_paths(config, args, model_arch: str) -> Optional[Dict[str, str]]:
    module_paths = None
    pretrained = getattr(config, "pretrained_modules", None)
    if pretrained:
        module_paths = dict(pretrained)

    module_paths = _apply_cli_module_overrides(module_paths, args)
    module_paths = dict(module_paths) if isinstance(module_paths, dict) else {}

    if args.velocity_contact_ckpt:
        module_paths["velocity_contact"] = args.velocity_contact_ckpt
    if args.object_trans_ckpt:
        module_paths["object_trans"] = args.object_trans_ckpt
    if args.interaction_ckpt:
        module_paths["interaction"] = args.interaction_ckpt

    if model_arch in {"dit", "mamba"}:
        module_paths = _filter_pipeline_module_paths(module_paths)

    return module_paths or None


def _move_batch_to_device(batch, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _run_inference(model, data_dict, batch_device, model_kind: str, interaction_human_source: str):
    source = str(interaction_human_source).lower()
    interaction_use_human_pred = source != "gt"

    if model_kind == "mix":
        return model.inference(data_dict, gt_targets=batch_device)

    if model_kind == "pipeline":
        has_separate_vc = hasattr(model, "velocity_contact_module")
        return model.inference(
            data_dict,
            gt_targets=batch_device,
            use_object_data=not has_separate_vc,
            compute_fk=False,
            interaction_use_human_pred=interaction_use_human_pred,
        )

    if model_kind == "interaction":
        return model.inference(data_dict, hp_out=None, gt_targets=batch_device)

    if hasattr(model, "inference"):
        try:
            return model.inference(
                data_dict,
                gt_targets=batch_device,
                use_object_data=True,
                compute_fk=False,
                interaction_use_human_pred=interaction_use_human_pred,
            )
        except TypeError:
            return model.inference(data_dict, gt_targets=batch_device)

    return model(data_dict, use_object_data=True, compute_fk=False)


def _tensor_to_np(value: torch.Tensor, sample_idx: int, min_dim: int) -> Optional[np.ndarray]:
    if not isinstance(value, torch.Tensor):
        return None
    sample = value[sample_idx].detach().float().cpu()
    if sample.dim() < min_dim:
        return None
    return sample.numpy()


def _extract_cond_prob(pred_dict: Dict) -> Optional[torch.Tensor]:
    cond = pred_dict.get("pred_hand_contact_prob_cond")
    if isinstance(cond, torch.Tensor):
        return cond
    logits = pred_dict.get("pred_hand_contact_logits_cond")
    if isinstance(logits, torch.Tensor):
        return torch.sigmoid(logits)
    return None


def collect_records(model, data_loader, config, device, args) -> List[Record]:
    model.eval()
    model_kind = _infer_model_kind(model)
    records: List[Record] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch_device = _move_batch_to_device(batch, device)
            try:
                data_dict = build_model_input_dict(batch_device, config, device, add_noise=False)
            except Exception as exc:
                print(f"[Warn] Failed to build model input at batch {batch_idx}: {exc}")
                continue

            try:
                pred_dict = _run_inference(
                    model,
                    data_dict,
                    batch_device,
                    model_kind=model_kind,
                    interaction_human_source=args.interaction_human_source,
                )
            except Exception as exc:
                print(f"[Warn] Inference failed at batch {batch_idx}: {exc}")
                continue

            uncond = pred_dict.get("pred_hand_contact_prob")
            if uncond is None:
                uncond = pred_dict.get("contact_prob_pred")
            cond = _extract_cond_prob(pred_dict)

            if not isinstance(uncond, torch.Tensor) or uncond.shape[-1] < 2:
                print(f"[Warn] Batch {batch_idx} has no usable pred_hand_contact_prob.")
                continue

            gt_l = batch_device.get("lhand_contact")
            gt_r = batch_device.get("rhand_contact")
            gt_o = batch_device.get("obj_contact")
            if not all(isinstance(x, torch.Tensor) for x in (gt_l, gt_r)):
                print(f"[Warn] Batch {batch_idx} has no hand contact labels.")
                continue

            batch_size = uncond.shape[0]
            for sample_idx in range(batch_size):
                pred_uncond = _tensor_to_np(uncond, sample_idx, min_dim=2)
                pred_cond = _tensor_to_np(cond, sample_idx, min_dim=2) if cond is not None else None
                l_np = gt_l[sample_idx].detach().bool().cpu().numpy().astype(np.int64)
                r_np = gt_r[sample_idx].detach().bool().cpu().numpy().astype(np.int64)
                o_np = (
                    gt_o[sample_idx].detach().bool().cpu().numpy().astype(np.int64)
                    if isinstance(gt_o, torch.Tensor)
                    else np.zeros_like(l_np)
                )

                if pred_uncond is None:
                    continue
                seq_len = min(len(l_np), len(r_np), pred_uncond.shape[0])
                if pred_cond is not None:
                    seq_len = min(seq_len, pred_cond.shape[0])
                if seq_len <= 0:
                    continue

                record: Record = {
                    "gt_left": l_np[:seq_len],
                    "gt_right": r_np[:seq_len],
                    "gt_object": o_np[:seq_len],
                    "uncond": pred_uncond[:seq_len],
                }
                if pred_cond is not None and pred_cond.shape[-1] >= 2:
                    record["cond"] = pred_cond[:seq_len]
                records.append(record)

            if args.max_batches is not None and (batch_idx + 1) >= args.max_batches:
                break
            if (batch_idx + 1) % args.progress_every == 0:
                print(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

    return records


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.unique(y_true).size <= 1:
        return float("nan")
    return float(f1_score(y_true.astype(np.int64), y_pred.astype(np.int64)))


def _mean_valid(values: Iterable[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def sequence_mean_f1(records: List[Record], source: str, hand: str, threshold: float) -> float:
    idx = 0 if hand == "left" else 1
    gt_key = f"gt_{hand}"
    scores = []
    for rec in records:
        prob = rec.get(source)
        if prob is None or prob.shape[-1] <= idx:
            continue
        pred = prob[:, idx] > threshold
        scores.append(_safe_f1(rec[gt_key], pred))
    return _mean_valid(scores)


def global_f1(records: List[Record], source: str, hand: str, threshold: float) -> float:
    idx = 0 if hand == "left" else 1
    gt_key = f"gt_{hand}"
    y_true = []
    y_pred = []
    for rec in records:
        prob = rec.get(source)
        if prob is None or prob.shape[-1] <= idx:
            continue
        y_true.append(rec[gt_key])
        y_pred.append((prob[:, idx] > threshold).astype(np.int64))
    if not y_true:
        return float("nan")
    return _safe_f1(np.concatenate(y_true), np.concatenate(y_pred))


def object_f1(records: List[Record], threshold: float = 0.5) -> Tuple[float, float]:
    seq_scores = []
    y_true = []
    y_pred = []
    for rec in records:
        prob = rec.get("uncond")
        if prob is None:
            continue
        if prob.shape[-1] >= 3:
            pred = prob[:, 2] > threshold
        else:
            pred = (prob[:, 0] > threshold) | (prob[:, 1] > threshold)
        gt = rec["gt_object"]
        seq_scores.append(_safe_f1(gt, pred))
        y_true.append(gt)
        y_pred.append(pred.astype(np.int64))
    global_score = _safe_f1(np.concatenate(y_true), np.concatenate(y_pred)) if y_true else float("nan")
    return _mean_valid(seq_scores), global_score


def threshold_values(start: float, end: float, step: float) -> np.ndarray:
    values = []
    current = start
    while current <= end + (step * 0.5):
        values.append(round(current, 6))
        current += step
    return np.asarray(values, dtype=np.float64)


def best_threshold(records: List[Record], source: str, hand: str, thresholds: np.ndarray) -> Tuple[float, float, float, float]:
    best_t = float("nan")
    best_seq = float("nan")
    best_global = float("nan")
    baseline_seq = sequence_mean_f1(records, source, hand, 0.5)

    for threshold in thresholds:
        seq_score = sequence_mean_f1(records, source, hand, float(threshold))
        if math.isnan(seq_score):
            continue
        if math.isnan(best_seq) or seq_score > best_seq:
            best_t = float(threshold)
            best_seq = seq_score
            best_global = global_f1(records, source, hand, float(threshold))

    gain = best_seq - baseline_seq if not (math.isnan(best_seq) or math.isnan(baseline_seq)) else float("nan")
    return best_t, best_seq, best_global, gain


def _fmt(value: float) -> str:
    return f"{value:.4f}" if not math.isnan(value) else "NaN"


def print_report(records: List[Record], args) -> None:
    thresholds = threshold_values(args.threshold_start, args.threshold_end, args.threshold_step)
    has_cond = any("cond" in rec for rec in records)
    total_frames = sum(len(rec["gt_left"]) for rec in records)

    print(f"\nCollected sequences: {len(records)} | frames: {total_frames}")
    print("\n--- F1 at threshold 0.5 ---")
    for source, label in (("uncond", "pred_hand_contact_prob"), ("cond", "pred_hand_contact_prob_cond")):
        if source == "cond" and not has_cond:
            print(f"{label}: unavailable")
            continue
        l_seq = sequence_mean_f1(records, source, "left", 0.5)
        r_seq = sequence_mean_f1(records, source, "right", 0.5)
        l_global = global_f1(records, source, "left", 0.5)
        r_global = global_f1(records, source, "right", 0.5)
        print(f"{label} Left : seq_mean={_fmt(l_seq)} | global={_fmt(l_global)}")
        print(f"{label} Right: seq_mean={_fmt(r_seq)} | global={_fmt(r_global)}")

    obj_seq, obj_global = object_f1(records, threshold=0.5)
    print(f"Object from pred_hand_contact_prob: seq_mean={_fmt(obj_seq)} | global={_fmt(obj_global)}")

    if has_cond:
        print("\n--- Product Effect at threshold 0.5 (cond - uncond, seq_mean) ---")
        for hand in ("left", "right"):
            cond_score = sequence_mean_f1(records, "cond", hand, 0.5)
            uncond_score = sequence_mean_f1(records, "uncond", hand, 0.5)
            delta = cond_score - uncond_score if not (math.isnan(cond_score) or math.isnan(uncond_score)) else float("nan")
            print(f"{hand.capitalize():5s}: {_fmt(delta)}")

    print(f"\n--- Threshold Sweep [{args.threshold_start}, {args.threshold_end}] step={args.threshold_step} ---")
    for source, label in (("uncond", "pred_hand_contact_prob"), ("cond", "pred_hand_contact_prob_cond")):
        if source == "cond" and not has_cond:
            continue
        print(f"\n{label}")
        for hand in ("left", "right"):
            best_t, best_seq, best_global, gain = best_threshold(records, source, hand, thresholds)
            base_seq = sequence_mean_f1(records, source, hand, 0.5)
            print(
                f"{hand.capitalize():5s}: "
                f"base@0.5={_fmt(base_seq)} | "
                f"best_seq={_fmt(best_seq)} @ t={_fmt(best_t)} | "
                f"best_global={_fmt(best_global)} | "
                f"gain={_fmt(gain)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose hand contact F1 and threshold calibration.")
    parser.add_argument("--config", type=str, default="configs/IMUHOI_train_rnn.yaml", help="Config path.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset key from eval defaults or config.datasets.")
    parser.add_argument("--test_data_dir", type=str, default=None, help="Override test data directory.")
    parser.add_argument("--model_arch", type=str, choices=["rnn", "dit", "mamba", "mamba_simple"], default=None)
    parser.add_argument("--hp_ckpt", type=str, default=None, help="Override human_pose checkpoint.")
    parser.add_argument("--velocity_contact_ckpt", type=str, default=None, help="Override velocity_contact checkpoint.")
    parser.add_argument("--object_trans_ckpt", type=str, default=None, help="Override object_trans checkpoint.")
    parser.add_argument("--interaction_ckpt", type=str, default=None, help="Override interaction checkpoint.")
    parser.add_argument("--no_trans", action="store_true", help="Use noTrans mode when loading the model.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument("--threshold_start", type=float, default=0.2)
    parser.add_argument("--threshold_end", type=float, default=0.8)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument(
        "--interaction_human_source",
        type=str,
        default="pred",
        choices=["pred", "gt"],
        help="For interaction-style models: use predicted or GT human context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    if args.model_arch is not None:
        config.model_arch = args.model_arch
    config.num_workers = args.num_workers
    if not hasattr(config, "debug"):
        config.debug = False

    model_arch = str(getattr(config, "model_arch", "rnn")).lower()
    module_paths = _build_module_paths(config, args, model_arch)

    runs = _resolve_data_runs(args, config)
    model = load_model(config, device, no_trans=args.no_trans, module_paths=module_paths)

    for dataset_name, data_path in runs:
        if not data_path.exists():
            print(f"[Skip] {dataset_name}: data path does not exist: {data_path}")
            continue

        test_cfg = _as_dict(getattr(config, "test", {}))
        train_cfg = _as_dict(getattr(config, "train", {}))
        test_window = _get_cfg_value(test_cfg, "window", _get_cfg_value(train_cfg, "window", 60))

        print(f"\n=== Contact diagnostics: {dataset_name} ===")
        print(f"Device: {device}")
        print(f"Model arch: {model_arch}")
        print(f"Data: {data_path}")
        print(f"Window: {test_window}")

        dataset = IMUDataset(
            data_dir=str(data_path),
            window_size=test_window,
            debug=config.get("debug", False),
            simulate_imu_noise=False,
            full_sequence=True,
            resolve_bimanual_contact_conflicts=config.get("resolve_bimanual_contact_conflicts", True),
        )
        if len(dataset) == 0:
            print(f"[Skip] {dataset_name}: dataset is empty.")
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        start = time.time()
        records = collect_records(model, loader, config, device, args)
        print_report(records, args)
        print(f"\nElapsed: {time.time() - start:.2f}s")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
