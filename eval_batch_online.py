#!/usr/bin/env python3
"""Batched online evaluation for the RNN IMUHOI pipeline."""
import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_IMUHOI import IMUDataset
from eval_IMUHOI import (
    _apply_cli_module_overrides,
    _build_dataset_runs,
    _filter_pipeline_module_paths,
    _object_contact_capabilities,
    _select_path,
    _set_eval_seed,
    _unwrap_model,
    evaluate_model,
    get_default_dataset_config,
)
from model.rnn.imuhoi_model import load_model
from model.rnn.online import (
    TimeDictAccumulator,
    infer_batch_seq,
    merge_latest_context,
    resolve_online_window,
    select_time_context,
    take_latest_frame,
    update_data_inits_from_history,
)
from utils.utils import build_model_input_dict, load_config, load_smpl_model


TEMPORAL_SAMPLE_KEYS = {
    "human_imu",
    "obj_imu",
    "trans",
    "ori_root_reduced",
    "obj_trans",
    "obj_rot",
    "obj_scale",
    "root_vel",
    "obj_vel",
    "sensor_vel_root",
    "sensor_vel_glb",
    "lhand_contact",
    "rhand_contact",
    "obj_contact",
    "interaction_start",
    "interaction_end",
    "interaction_start_gauss",
    "interaction_end_gauss",
    "lfoot_contact",
    "rfoot_contact",
    "lhand_obj_direction",
    "rhand_obj_direction",
    "lhand_lb",
    "rhand_lb",
    "position_global",
    "rotation_global",
    "pose",
}


class _CachedSequenceDataset(Dataset):
    def __init__(self, samples: Sequence[Dict[str, Any]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


class IMUHOIModelCached(torch.nn.Module):
    """Evaluation shim that reuses eval_IMUHOI metrics with precomputed outputs."""

    def __init__(self, real_model: torch.nn.Module, pred_cache: Dict[str, Dict[str, Any]], device: torch.device):
        super().__init__()
        self.real_model = real_model
        self.pred_cache = pred_cache
        self.device = device
        core = _unwrap_model(real_model)
        self.human_pose_module = getattr(core, "human_pose_module", None)
        self.velocity_contact_module = getattr(core, "velocity_contact_module", None)
        self.object_trans_module = getattr(core, "object_trans_module", None)

    def _lookup_key(self, gt_targets: Optional[Dict[str, Any]], data_dict: Dict[str, Any]) -> str:
        for container in (gt_targets, data_dict):
            if not isinstance(container, dict):
                continue
            value = container.get("_cache_key") or container.get("seq_file")
            if isinstance(value, (list, tuple)):
                value = value[0]
            if isinstance(value, torch.Tensor):
                value = value.item()
            if value is not None:
                key = str(value)
                if key in self.pred_cache:
                    return key
        raise KeyError("No cached prediction key found for metric batch.")

    def inference(self, data_dict: Dict[str, Any], gt_targets: Optional[Dict[str, Any]] = None, **_) -> Dict[str, Any]:
        key = self._lookup_key(gt_targets, data_dict)
        return _move_to_device(self.pred_cache[key], self.device)

    def forward(self, data_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.inference(data_dict, **kwargs)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _detach_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    return value


def _slice_batch_value(value: Any, row: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == int(batch_size):
        return value[row:row + 1].detach()
    if isinstance(value, dict):
        return {key: _slice_batch_value(item, row, batch_size) for key, item in value.items()}
    return value


def _split_batch_dict(output: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
    return [
        {key: _slice_batch_value(value, row, batch_size) for key, value in output.items()}
        for row in range(batch_size)
    ]


def _stack_dicts(dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not dicts:
        return {}
    result: Dict[str, Any] = {}
    common_keys = [key for key in dicts[0].keys() if all(key in item for item in dicts)]
    for key in common_keys:
        values = [item[key] for item in dicts]
        if all(isinstance(value, torch.Tensor) for value in values):
            first = values[0]
            if all(value.shape[1:] == first.shape[1:] and value.dtype == first.dtype for value in values):
                result[key] = torch.cat(values, dim=0)
            continue
        if all(isinstance(value, dict) for value in values):
            nested = _stack_dicts(values)  # type: ignore[arg-type]
            if nested:
                result[key] = nested
            continue
        result[key] = values[0]
    return result


def _sample_seq_len(sample: Dict[str, Any]) -> int:
    human_imu = sample.get("human_imu")
    if not isinstance(human_imu, torch.Tensor):
        raise ValueError("Sample is missing tensor human_imu.")
    return int(human_imu.shape[0])


def _slice_sample_value(key: str, value: Any, seq_len: int, start: int, end: int) -> Any:
    if key in TEMPORAL_SAMPLE_KEYS and isinstance(value, torch.Tensor):
        if value.dim() >= 1 and int(value.shape[0]) == int(seq_len):
            return value[start:end]
    return value


def _collate_sample_slices(
    samples: Sequence[Dict[str, Any]],
    starts: Sequence[int],
    ends: Sequence[int],
) -> Dict[str, Any]:
    if not samples:
        return {}
    batch: Dict[str, Any] = {}
    keys: List[str] = []
    seen = set()
    for sample in samples:
        for key in sample.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)

    for key in keys:
        values = []
        present = True
        for sample, start, end in zip(samples, starts, ends):
            if key not in sample:
                present = False
                break
            seq_len = _sample_seq_len(sample)
            values.append(_slice_sample_value(key, sample[key], seq_len, int(start), int(end)))
        if not present:
            continue
        if all(isinstance(value, torch.Tensor) for value in values):
            try:
                batch[key] = torch.stack(values, dim=0)
                continue
            except RuntimeError:
                pass
        if all(isinstance(value, bool) for value in values):
            batch[key] = torch.tensor(values, dtype=torch.bool)
            continue
        if all(isinstance(value, (int, np.integer)) for value in values):
            batch[key] = torch.tensor(values, dtype=torch.long)
            continue
        if all(isinstance(value, (float, np.floating)) for value in values):
            batch[key] = torch.tensor(values, dtype=torch.float32)
            continue
        batch[key] = list(values)
    return batch


def _merge_stage_context(prefix: Dict[str, Any], latest: Dict[str, Any]) -> Dict[str, Any]:
    stage_prefix = {key: prefix[key] for key in latest.keys() if key in prefix}
    return merge_latest_context(stage_prefix, latest)


def _run_forward_batch(
    model,
    config,
    device: torch.device,
    samples: Sequence[Dict[str, Any]],
    use_object_data: bool,
    compute_fk: bool,
    refine_human: bool,
) -> Dict[str, Any]:
    seq_lens = [_sample_seq_len(sample) for sample in samples]
    if len(set(seq_lens)) != 1:
        raise ValueError("Forward batch requires equal sequence lengths.")
    raw_batch = _collate_sample_slices(samples, [0] * len(samples), seq_lens)
    data_dict = build_model_input_dict(raw_batch, config, device, add_noise=False)
    return model.forward(
        data_dict,
        use_object_data=use_object_data,
        compute_fk=compute_fk,
        refine_human=refine_human,
    )


def _run_batched_online_group(
    model,
    config,
    device: torch.device,
    samples: Sequence[Dict[str, Any]],
    online_window: int,
    use_object_data: bool,
    compute_fk: bool,
    refine_human: bool,
    desc: str,
) -> List[Dict[str, Any]]:
    if not samples:
        return []

    outputs: List[Optional[Dict[str, Any]]] = [None] * len(samples)
    seq_lens = [_sample_seq_len(sample) for sample in samples]

    short_by_len: Dict[int, List[int]] = defaultdict(list)
    long_positions = []
    for pos, seq_len in enumerate(seq_lens):
        if seq_len <= online_window:
            short_by_len[seq_len].append(pos)
        else:
            long_positions.append(pos)

    for same_len, positions in short_by_len.items():
        same_samples = [samples[pos] for pos in positions]
        out = _run_forward_batch(
            model,
            config,
            device,
            same_samples,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
        )
        for pos, row_out in zip(positions, _split_batch_dict(out, len(positions))):
            outputs[pos] = _detach_to_cpu(row_out)

    if long_positions:
        long_samples = [samples[pos] for pos in long_positions]
        long_lens = [seq_lens[pos] for pos in long_positions]
        warmup_batch = _collate_sample_slices(
            long_samples,
            [0] * len(long_samples),
            [online_window] * len(long_samples),
        )
        warmup_data = build_model_input_dict(warmup_batch, config, device, add_noise=False)
        warmup_out = model.forward(
            warmup_data,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
        )
        warmup_rows = _split_batch_dict(warmup_out, len(long_positions))
        accumulators = {
            local_pos: TimeDictAccumulator(row_out, total_seq_len=long_lens[local_pos])
            for local_pos, row_out in enumerate(warmup_rows)
        }

        max_len = max(long_lens)
        step_iter = range(online_window + 1, max_len + 1)
        for end in tqdm(step_iter, desc=desc, unit="step", leave=False, dynamic_ncols=True):
            active_local = [idx for idx, seq_len in enumerate(long_lens) if seq_len >= end]
            if not active_local:
                continue
            start = end - online_window
            active_samples = [long_samples[idx] for idx in active_local]
            window_batch = _collate_sample_slices(
                active_samples,
                [start] * len(active_samples),
                [end] * len(active_samples),
            )
            window_data = build_model_input_dict(window_batch, config, device, add_noise=False)
            histories = [accumulators[idx].current() for idx in active_local]
            history = _stack_dicts(histories)
            window_data = update_data_inits_from_history(window_data, history, index=start - 1)
            active_batch, window_len = infer_batch_seq(window_data)
            prefix = select_time_context(history, start, end - 1)

            hp_out_raw = model.human_pose_module.forward(model._build_hp_input_dict(window_data))
            hp_latest = take_latest_frame(hp_out_raw, active_batch, window_len)
            hp_context = _merge_stage_context(prefix, hp_latest)

            vc_input_dict = {
                "human_imu": window_data["human_imu"],
                "obj_imu": window_data["obj_imu"],
                "hand_vel_glb_init": window_data["hand_vel_glb_init"],
                "obj_vel_init": window_data["obj_vel_init"],
                "obj_trans_init": window_data["obj_trans_init"],
                "contact_init": window_data.get("contact_init"),
                "hp_out": hp_context,
            }
            vc_out_raw = model.velocity_contact_module.forward(vc_input_dict, hp_out=hp_context)
            vc_latest = take_latest_frame(vc_out_raw, active_batch, window_len)
            vc_context = _merge_stage_context(prefix, vc_latest)

            results: Dict[str, Any] = {}
            results.update(hp_context)
            results.update(vc_context)

            has_object = window_data.get("has_object")
            if use_object_data and (has_object is None or has_object.any()):
                obj_prefix = prefix.get("pred_obj_trans")
                if isinstance(obj_prefix, torch.Tensor) and obj_prefix.shape[1] > 0:
                    ot_obj_trans_init = obj_prefix[:, 0]
                else:
                    ot_obj_trans_init = window_data["obj_trans_init"]
                ot_out = model.object_trans_module.forward(
                    hp_context["pred_hand_glb_pos"],
                    vc_context["pred_hand_contact_prob"],
                    ot_obj_trans_init,
                    obj_imu=window_data["obj_imu"],
                    human_imu=window_data["human_imu"],
                    obj_vel_input=vc_context["pred_obj_vel"],
                    contact_init=window_data.get("contact_init"),
                    has_object_mask=has_object,
                    human_pose_input=hp_context.get("p_pred"),
                    root_trans_input=hp_context.get("root_trans_pred"),
                    obj_points_canonical=window_data.get("obj_points_canonical"),
                    obj_rot_gt=window_data.get("obj_rot_gt"),
                    obj_trans_gt=window_data.get("obj_trans_gt"),
                    obj_scale_gt=window_data.get("obj_scale_gt"),
                    enable_refine=refine_human,
                    known_obj_trans_prefix=obj_prefix,
                )
                results.update(ot_out)
                if refine_human:
                    model._promote_refined_human_outputs(results, ot_out, window_data)
                if compute_fk:
                    model._compute_fk_output(results, vc_context, window_data, ot_obj_trans_init)

            results["has_object"] = has_object
            latest = take_latest_frame(results, active_batch, window_len)
            latest_rows = _split_batch_dict(latest, active_batch)
            for local_pos, row_latest in zip(active_local, latest_rows):
                accumulators[local_pos].append(row_latest)

        for local_pos, original_pos in enumerate(long_positions):
            outputs[original_pos] = _detach_to_cpu(accumulators[local_pos].current())

    missing = [idx for idx, output in enumerate(outputs) if output is None]
    if missing:
        raise RuntimeError(f"Batched online inference missed outputs for local positions: {missing}")
    return [output for output in outputs if output is not None]


def _batched_online_predict_dataset(
    model,
    dataset: IMUDataset,
    config,
    device: torch.device,
    online_batch_size: int,
    online_window: int,
    use_object_data: bool,
    compute_fk: bool,
    refine_human: bool,
    sort_desc: bool,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    order = list(range(len(dataset)))
    order.sort(key=lambda idx: int(dataset.sequence_info[idx]["seq_len"]), reverse=sort_desc)
    pred_cache: Dict[str, Dict[str, Any]] = {}
    sample_cache: List[Dict[str, Any]] = []

    progress = tqdm(
        range(0, len(order), online_batch_size),
        desc="BatchOnline",
        unit="batch",
        dynamic_ncols=True,
    )
    for group_id, offset in enumerate(progress):
        group_indices = order[offset:offset + online_batch_size]
        samples = []
        keys = []
        seq_lens = []
        for dataset_idx in group_indices:
            info = dataset.sequence_info[dataset_idx]
            key = str(Path(info["file_path"]).resolve())
            sample = dict(dataset[dataset_idx])
            sample["_cache_key"] = key
            samples.append(sample)
            sample_cache.append(sample)
            keys.append(key)
            seq_lens.append(_sample_seq_len(sample))

        progress.set_postfix_str(
            f"seq_len {min(seq_lens)}-{max(seq_lens)} | n={len(samples)}",
            refresh=True,
        )
        outputs = _run_batched_online_group(
            model,
            config,
            device,
            samples,
            online_window=online_window,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
            desc=f"online {group_id + 1}",
        )
        for key, output in zip(keys, outputs):
            pred_cache[key] = output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred_cache, sample_cache


def _format_metric(results: Dict[str, float], key: str) -> str:
    value = results.get(key, float("nan"))
    return f"{value:.4f}" if not np.isnan(value) else "NaN"


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RNN IMUHOI with cross-sequence batched online inference.")
    parser.add_argument("--config", type=str, default="configs/IMUHOI_train_rnn.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--smpl_model_path", type=str, default="datasets/smpl_models/smplh/male/model.npz")
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=12, help="Kept for dataset parity; metric pass uses cached samples.")
    parser.add_argument("--online_batch_size", "--online-batch-size", type=int, default=64)
    parser.add_argument("--online_window", "--online-window", type=int, default=None)
    parser.add_argument("--sort_asc", action="store_true", help="Sort sequence lengths ascending instead of descending.")
    parser.add_argument("--no_trans", action="store_true")
    parser.add_argument("--no_eval_objects", action="store_true")
    parser.add_argument("--compare_3", action="store_true")
    parser.add_argument("--hp_ckpt", type=str, default=None)
    parser.add_argument("--interaction_ckpt", type=str, default=None)
    parser.add_argument("--velocity_contact_ckpt", type=str, default=None)
    parser.add_argument("--object_trans_ckpt", type=str, default=None)
    parser.add_argument("--hand_contact_threshold", type=float, default=None)
    parser.add_argument("--object_contact_threshold", type=float, default=None)
    parser.add_argument("--ablate_vc_boundary", action="store_true")
    parser.add_argument("--ablate_ot_obs_encoder", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--interaction_human_source", type=str, default="pred", choices=["pred", "gt"])
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument("--enable_ot_refine", dest="ot_refine", action="store_true", default=None)
    refine_group.add_argument("--disable_ot_refine", dest="ot_refine", action="store_false")
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    if args.online_batch_size < 1:
        raise ValueError("--online_batch_size must be >= 1")

    seed = _set_eval_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Model arch: rnn")
    print(f"Mode: {'noTrans' if args.no_trans else 'Normal'}")
    print(f"Batch online size: {args.online_batch_size}")
    print(
        "Ablations: "
        f"vc_boundary={'zero' if args.ablate_vc_boundary else 'normal'} | "
        f"ot_obs_encoder={'zero' if args.ablate_ot_obs_encoder else 'normal'}"
    )
    print(f"Eval seed: {seed if seed is not None else 'unset'}")

    default_config = get_default_dataset_config(args.no_trans)
    try:
        dataset_runs = _build_dataset_runs(args, default_config)
    except Exception as exc:
        print(f"[BatchEval] Argument error: {exc}")
        return 2

    all_results = {}
    for dataset_name, dataset_cfg in dataset_runs:
        print(f"\n=== Batch-online evaluating dataset: {dataset_name} ===")
        config = load_config(args.config)
        config.model_arch = "rnn"
        config.ablate_vc_boundary = bool(args.ablate_vc_boundary)
        config.ablate_ot_obs_encoder = bool(args.ablate_ot_obs_encoder)
        ot_refine = (
            bool(args.ot_refine)
            if args.ot_refine is not None
            else bool(getattr(config, "enable_ot_refine", False))
        )
        config.enable_ot_refine = ot_refine
        if args.num_workers is not None:
            config.num_workers = args.num_workers
        if not hasattr(config, "debug"):
            config.debug = False

        hand_contact_threshold = float(
            args.hand_contact_threshold
            if args.hand_contact_threshold is not None
            else getattr(config, "hand_contact_threshold", 0.5)
        )
        object_contact_threshold = float(
            args.object_contact_threshold
            if args.object_contact_threshold is not None
            else getattr(config, "object_contact_threshold", 0.5)
        )

        modules_override = dataset_cfg.get("modules")
        module_paths = dict(config.pretrained_modules) if getattr(config, "pretrained_modules", None) else None
        if modules_override:
            config_copy = copy.deepcopy(config)
            from eval_IMUHOI import _apply_module_overrides

            _apply_module_overrides(config_copy, modules_override)
            module_paths = dict(config_copy.pretrained_modules) if getattr(config_copy, "pretrained_modules", None) else module_paths
        module_paths = _apply_cli_module_overrides(module_paths, args)
        module_paths = _filter_pipeline_module_paths(module_paths)
        has_velocity_contact_ckpt, has_object_trans_ckpt = _object_contact_capabilities(module_paths)

        if args.smpl_model_path:
            config.body_model_path = args.smpl_model_path
        smpl_model_path = config.get("body_model_path", "datasets/smpl_models/smplh/neutral/model.npz")
        try:
            smpl_model = load_smpl_model(smpl_model_path, device)
        except FileNotFoundError as exc:
            print(f"[BatchEval] Skipping '{dataset_name}': {exc}")
            continue

        data_dir_default = dataset_cfg.get("data_dir")
        if data_dir_default is None and not args.test_data_dir:
            print(f"[BatchEval] Skipping '{dataset_name}' (no dataset directory configured).")
            continue
        base_data_path = data_dir_default if data_dir_default is not None else Path(args.test_data_dir).expanduser()
        data_override = args.test_data_dir if args.dataset else None
        data_path = _select_path(data_override, base_data_path)
        if not data_path.exists():
            print(f"[BatchEval] Skipping '{dataset_name}' (data not found at {data_path}).")
            continue

        print(f"Loading test dataset from: {data_path}")
        test_window = config.test.get("window", config.train.get("window", 60))
        test_dataset = IMUDataset(
            data_dir=str(data_path),
            window_size=test_window,
            debug=config.get("debug", False),
            simulate_imu_noise=False,
            min_obj_contact_frames=0,
            full_sequence=True,
        )
        if len(test_dataset) == 0:
            print(f"[BatchEval] Skipping '{dataset_name}' (dataset is empty).")
            continue

        model = load_model(config, device, no_trans=args.no_trans, module_paths=module_paths)
        model = _unwrap_model(model)
        model.eval()

        evaluate_contacts = (not args.no_eval_objects) and has_velocity_contact_ckpt
        evaluate_objects = (not args.no_eval_objects) and has_velocity_contact_ckpt
        run_object_branch = evaluate_objects and has_object_trans_ckpt
        if not args.no_eval_objects:
            if not has_velocity_contact_ckpt:
                print("[BatchEval] No velocity_contact checkpoint loaded; skipping contact/object metrics.")
            elif not has_object_trans_ckpt:
                print("[BatchEval] No object_trans checkpoint loaded; object branch is disabled.")

        online_window = resolve_online_window(config, args.online_window)
        compute_fk = bool(args.compare_3 and run_object_branch)
        print(f"Dataset size: {len(test_dataset)}")
        print(f"Online window: {online_window}")
        print(f"OT human refine: {'enabled' if ot_refine else 'disabled'}")

        eval_start = time.time()
        with torch.inference_mode():
            pred_cache, sample_cache = _batched_online_predict_dataset(
                model,
                test_dataset,
                config,
                device,
                online_batch_size=args.online_batch_size,
                online_window=online_window,
                use_object_data=run_object_branch,
                compute_fk=compute_fk,
                refine_human=ot_refine,
                sort_desc=not args.sort_asc,
            )

        metric_model = IMUHOIModelCached(model, pred_cache, device).eval()
        metric_loader = DataLoader(
            _CachedSequenceDataset(sample_cache),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
        results = evaluate_model(
            metric_model,
            smpl_model,
            metric_loader,
            config,
            device,
            no_trans=args.no_trans,
            evaluate_objects=evaluate_objects,
            evaluate_contacts=evaluate_contacts,
            run_object_branch=run_object_branch,
            compare_three=args.compare_3,
            interaction_human_source=args.interaction_human_source,
            ot_refine=ot_refine,
            hand_contact_threshold=hand_contact_threshold,
            object_contact_threshold=object_contact_threshold,
            inference_mode="online",
            online_window=online_window,
        )
        eval_duration = time.time() - eval_start
        all_results[dataset_name] = results

        print("\n--- Batch Online Evaluation Results ---")
        print(f"MPJPE (cm):                     {_format_metric(results, 'mpjpe')}")
        print(f"MPJRE (deg):                    {_format_metric(results, 'mpjre_angle')}")
        if not args.no_trans:
            print(f"Root Trans Error (cm):          {_format_metric(results, 'root_trans_err')}")
            print(f"Stage1 Root Trans Error (cm):   {_format_metric(results, 'stage1_root_trans_err')}")
        else:
            print(f"Hand Vel Error L (m/s):         {_format_metric(results, 'hand_vel_err_lhand')}")
            print(f"Hand Vel Error R (m/s):         {_format_metric(results, 'hand_vel_err_rhand')}")
            print(f"Hand Vel Error Avg (m/s):       {_format_metric(results, 'hand_vel_err_avg')}")
        print(f"Jitter (mm/frame^2):            {_format_metric(results, 'jitter')}")
        if evaluate_objects:
            print("\n--- Object Translation Errors ---")
            print(f"Fusion (cm):                    {_format_metric(results, 'obj_trans_err_fusion')}")
            print(f"FK (cm):                        {_format_metric(results, 'obj_trans_err_fk')}")
            print(f"IMU (cm):                       {_format_metric(results, 'obj_trans_err_imu')}")
            print("\n--- HOI Errors ---")
            print(f"Fusion (cm):                    {_format_metric(results, 'hoi_err_fusion')}")
            print(f"FK (cm):                        {_format_metric(results, 'hoi_err_fk')}")
            print(f"IMU (cm):                       {_format_metric(results, 'hoi_err_imu')}")
        if evaluate_contacts:
            print("\n--- Contact Prediction F1 ---")
            print(f"Left Hand:                      {_format_metric(results, 'contact_f1_lhand')}")
            print(f"Right Hand:                     {_format_metric(results, 'contact_f1_rhand')}")
            print(f"Object:                         {_format_metric(results, 'contact_f1_obj')}")
        print(f"\n评估耗时: {eval_duration:.2f}秒")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": args.config,
            "inference_mode": "batch_online",
            "online_window": args.online_window,
            "resolved_online_window": resolve_online_window(load_config(args.config), args.online_window),
            "online_batch_size": args.online_batch_size,
            "ablate_vc_boundary": bool(args.ablate_vc_boundary),
            "ablate_ot_obs_encoder": bool(args.ablate_ot_obs_encoder),
            "seed": seed,
            "results": all_results,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[BatchEval] Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
