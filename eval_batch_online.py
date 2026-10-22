#!/usr/bin/env python3
"""Batched online evaluation for the RNN IMUHOI pipeline."""
import argparse
import copy
import json
import sys
import time
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
    _set_eval_seed,
    _unwrap_model,
    evaluate_model,
    get_default_dataset_config,
)
from model.rnn.imuhoi_model import load_model
from model.rnn.object_trans_IMUHOI import is_imuhoi_object_trans_arch
from model.rnn.velocity_contact import (
    VELOCITY_CONTACT_ARCH_CHOICES,
    normalize_velocity_contact_arch,
)
from model.rnn.online import (
    resolve_online_lookahead,
    resolve_online_window,
    slice_time_dict,
)
from utils.utils import apply_eval_imu_noise_to_sequence, build_model_input_dict, load_config
from utils.hoi_metrics import hoi_metric_metadata
from utils.dataset_config import (
    get_full_sequence_test_path,
    resolve_dataset_path,
    resolve_evaluation_body_model_path,
)


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


def _pad_sample_to_length(sample: Dict[str, Any], target_len: int) -> Dict[str, Any]:
    """Repeat-pad temporal fields for batched strict-lookahead inference.

    A sample with length ``T`` only publishes targets through ``T-D-1``.  Those
    targets read input frames no later than ``T-1``, so padding after ``T`` can
    never affect the retained output.  Repeat padding keeps rotations and other
    structured inputs valid while allowing differently sized sequences to run
    in one GPU batch.
    """
    seq_len = _sample_seq_len(sample)
    if seq_len >= target_len:
        return dict(sample)
    if seq_len <= 0:
        raise ValueError("Cannot pad an empty online sequence.")

    padded = dict(sample)
    pad_len = int(target_len) - seq_len
    for key, value in sample.items():
        if (
            key in TEMPORAL_SAMPLE_KEYS
            and isinstance(value, torch.Tensor)
            and value.dim() >= 1
            and int(value.shape[0]) == seq_len
        ):
            repeat_shape = (pad_len,) + tuple(value.shape[1:])
            pad = value[-1:].expand(repeat_shape).clone()
            padded[key] = torch.cat((value, pad), dim=0)
    return padded


def _distance_to_transition(labels: np.ndarray) -> np.ndarray:
    """Frames from each frame to the nearest label transition (inf if none)."""
    labels = np.asarray(labels, dtype=bool)
    boundaries = np.flatnonzero(np.diff(labels.astype(np.int8)) != 0) + 0.5
    if boundaries.size == 0:
        return np.full(labels.shape[0], np.inf)
    frames = np.arange(labels.shape[0], dtype=np.float64)
    return np.abs(frames[:, None] - boundaries[None, :]).min(axis=1)


def _build_vc_oracles(
    raw_batch: Dict[str, Any],
    spec: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Diagnostic GT substitutes for VelocityContact outputs (NaN = keep prediction).

    ``spec``: contact in {None, all, hands, move}; region in {all, boundary,
    interior} relative to +-boundary_frames around each channel's GT
    transitions; shift delays the GT labels by that many frames; obj_vel
    replaces the predicted object velocity with GT.
    """
    oracles = {}
    channels = {"all": (0, 1, 2), "hands": (0, 1), "move": (2,)}.get(spec.get("contact"))
    if channels:
        gt = torch.stack(
            [raw_batch[key].bool() for key in ("lhand_contact", "rhand_contact", "obj_contact")], dim=-1
        ).cpu().numpy()
        shift = int(spec.get("shift", 0))
        if shift > 0:
            gt = np.concatenate((np.repeat(gt[:, :1], shift, axis=1), gt[:, :-shift]), axis=1)
        elif shift < 0:
            gt = np.concatenate((gt[:, -shift:], np.repeat(gt[:, -1:], -shift, axis=1)), axis=1)
        region = str(spec.get("region", "all"))
        radius = float(spec.get("boundary_frames", 10))
        oracle = np.full(gt.shape, np.nan, dtype=np.float32)
        for row in range(gt.shape[0]):
            for channel in channels:
                labels = gt[row, :, channel]
                if region == "all":
                    keep = np.ones(labels.shape[0], dtype=bool)
                else:
                    near = _distance_to_transition(labels) <= radius
                    keep = near if region == "boundary" else ~near
                oracle[row, keep, channel] = labels[keep].astype(np.float32)
        oracles["oracle_vc_contact"] = torch.from_numpy(oracle).to(device=device, dtype=dtype)
    if spec.get("obj_vel"):
        oracles["oracle_vc_obj_vel"] = raw_batch["obj_vel"].to(device=device, dtype=dtype)
    return oracles


def _run_batched_online_group(
    model,
    config,
    device: torch.device,
    samples: Sequence[Dict[str, Any]],
    online_window: int,
    online_lookahead: int,
    use_object_data: bool,
    compute_fk: bool,
    refine_human: bool,
    desc: str,
    feedback_kwargs: Optional[Dict[str, Any]] = None,
    online_chunk: int = 1,
) -> List[Dict[str, Any]]:
    if not samples:
        return []

    seq_lens = [_sample_seq_len(sample) for sample in samples]
    # The first online call consumes W+D frames at once.  Padding a shorter
    # warmup would leak repeated tail frames into every retained warmup output,
    # so sequences that do not exceed W+D must be grouped by exact length.
    # Longer sequences are safe to repeat-pad: every retained target only
    # reads real frames through its own strict endpoint T-1.
    initial_context_len = int(online_window) + int(online_lookahead)
    grouped_positions: List[List[int]] = []
    long_positions = [
        idx for idx, seq_len in enumerate(seq_lens) if seq_len > initial_context_len
    ]
    if long_positions:
        grouped_positions.append(long_positions)
    short_by_len: Dict[int, List[int]] = {}
    for idx, seq_len in enumerate(seq_lens):
        if seq_len <= initial_context_len:
            short_by_len.setdefault(seq_len, []).append(idx)
    grouped_positions.extend(short_by_len.values())
    if len(grouped_positions) > 1:
        outputs: List[Optional[Dict[str, Any]]] = [None] * len(samples)
        for group_idx, positions in enumerate(grouped_positions, start=1):
            group_outputs = _run_batched_online_group(
                model,
                config,
                device,
                [samples[idx] for idx in positions],
                online_window=online_window,
                online_lookahead=online_lookahead,
                use_object_data=use_object_data,
                compute_fk=compute_fk,
                refine_human=refine_human,
                desc=f"{desc}.{group_idx}",
                feedback_kwargs=feedback_kwargs,
                online_chunk=online_chunk,
            )
            for idx, output in zip(positions, group_outputs):
                outputs[idx] = output
        return [output for output in outputs if output is not None]

    max_len = max(seq_lens)
    padded_samples = [_pad_sample_to_length(sample, max_len) for sample in samples]
    raw_batch = _collate_sample_slices(
        padded_samples,
        [0] * len(padded_samples),
        [max_len] * len(padded_samples),
    )
    data_dict = build_model_input_dict(raw_batch, config, device, add_noise=False)
    oracle_spec = getattr(config, "eval_oracle_vc", None)
    if oracle_spec:
        data_dict.update(_build_vc_oracles(raw_batch, oracle_spec, device, data_dict["human_imu"].dtype))
    init_frame = int(getattr(config, "eval_contact_init_frame", 0) or 0)
    if init_frame > 0:
        # The full-sequence splits label hand contact 0 at frame 0 for every
        # sequence (raw labels too; the segment splits do not), so the frame-0
        # initial contact state is an artifact; read it from a later frame.
        data_dict["contact_init"] = torch.stack(
            [raw_batch[key][:, init_frame].float() for key in ("lhand_contact", "rhand_contact", "obj_contact")],
            dim=-1,
        ).to(device=device, dtype=data_dict["contact_init"].dtype)
    if getattr(config, "eval_vc_contact_init", None) == "gt":
        data_dict["oracle_vc_contact_init"] = torch.stack(
            [raw_batch[key].bool() for key in ("lhand_contact", "rhand_contact", "obj_contact")], dim=-1
        ).to(device=device, dtype=data_dict["human_imu"].dtype)
    max_publish_len = max(max_len - int(online_lookahead), 0)
    frame_progress = tqdm(
        total=max_publish_len,
        desc=desc,
        unit="frame",
        leave=False,
        dynamic_ncols=True,
    )

    def _update_progress(completed: int, _total: int) -> None:
        frame_progress.update(max(0, int(completed) - frame_progress.n))

    try:
        out = model.inference(
            data_dict,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
            inference_mode="online",
            online_window=online_window,
            online_lookahead=online_lookahead,
            online_progress_callback=_update_progress,
            online_chunk=online_chunk,
            # Diagnostics only: feedback_mode / feedback_perturbation /
            # feedback_injection_frame for the ObjectTrans state loop.
            **(feedback_kwargs or {}),
        )
    finally:
        frame_progress.close()

    outputs = []
    for seq_len, row_out in zip(seq_lens, _split_batch_dict(out, len(samples))):
        publish_len = max(seq_len - int(online_lookahead), 0)
        row_out = slice_time_dict(
            row_out,
            0,
            publish_len,
            batch_size=1,
            seq_len=max_publish_len,
        )
        outputs.append(_detach_to_cpu(row_out))
    return outputs


def _batched_online_predict_dataset(
    model,
    dataset: IMUDataset,
    config,
    device: torch.device,
    online_batch_size: int,
    online_window: int,
    online_lookahead: int,
    use_object_data: bool,
    compute_fk: bool,
    refine_human: bool,
    sort_desc: bool,
    eval_imu_noise: bool,
    eval_imu_noise_seed: Optional[int],
    feedback_kwargs: Optional[Dict[str, Any]] = None,
    online_chunk: int = 1,
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
    for offset in progress:
        group_indices = order[offset:offset + online_batch_size]
        samples = []
        keys = []
        seq_lens = []
        for dataset_idx in group_indices:
            info = dataset.sequence_info[dataset_idx]
            key = str(Path(info["file_path"]).resolve())
            sample = dict(dataset[dataset_idx])
            sample["_cache_key"] = key
            if eval_imu_noise:
                sample = apply_eval_imu_noise_to_sequence(
                    sample,
                    config,
                    seed=eval_imu_noise_seed,
                    sequence_key=key,
                )
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
            online_lookahead=online_lookahead,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
            desc=f"OnlineFrames {offset // online_batch_size + 1}",
            feedback_kwargs=feedback_kwargs,
            online_chunk=online_chunk,
        )
        for key, output in zip(keys, outputs):
            pred_cache[key] = output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred_cache, sample_cache


def _format_metric(results: Dict[str, float], key: str) -> str:
    value = results.get(key, float("nan"))
    return f"{value:.4f}" if not np.isnan(value) else "NaN"


OBJECT_TRANS_ARCH_CHOICES = (
    "current",
    "v2",
    "v3",
    "legacy",
    "legacy_mesh",
    "mergeHeads_obs",
    "imuhoi",
    "imuhoi_4heads",
)


def _normalize_object_trans_arch(value: str) -> str:
    """Normalize CLI/config spelling while retaining the legacy display name."""
    normalized = str(value).strip().lower()
    if normalized == "mergeheads_obs":
        return "mergeHeads_obs"
    return normalized


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RNN IMUHOI with cross-sequence batched online inference.")
    parser.add_argument("--config", type=str, default="configs/IMUHOI_train_rnn.yaml")
    parser.add_argument("--dataset", type=str, default="omomo", help="Dataset key under config.datasets (default: omomo).")
    parser.add_argument(
        "--sequence_path",
        "--sequence-path",
        action="append",
        default=None,
        help="Evaluate an explicit PT sequence path; repeat to select multiple files.",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        default=None,
        help="Deprecated and ignored: joint metrics use model/dataset canonical joints directly.",
    )
    parser.add_argument("--num_workers", type=int, default=12, help="Kept for dataset parity; metric pass uses cached samples.")
    parser.add_argument(
        "--online_batch_size",
        "--online-batch-size",
        type=int,
        default=64,
        help=(
            "Sequences per padded online batch. Each batch also reports frame-level progress."
        ),
    )
    parser.add_argument(
        "--online_window",
        "--online-window",
        type=int,
        default=None,
        help="Base online window; defaults to test.window from the config.",
    )
    parser.add_argument(
        "--online_chunk",
        "--online-chunk",
        type=int,
        default=16,
        help=(
            "Target frames whose HumanPose/VelocityContact windows run as one batch "
            "(1 = one window per frame; capped at online_window - 1)"
        ),
    )
    parser.add_argument(
        "--online_lookahead",
        "--online-lookahead",
        type=int,
        default=10,
        help="Future frames appended after the legacy online update frame (strict tail).",
    )
    parser.add_argument("--sort_asc", action="store_true", help="Sort sequence lengths ascending instead of descending.")
    parser.add_argument("--no_trans", action="store_true")
    parser.add_argument("--no_objects", action="store_true")
    parser.add_argument("--compare_3", action="store_true")
    parser.add_argument("--hp_ckpt", type=str, default=None)
    parser.add_argument("--interaction_ckpt", type=str, default=None)
    parser.add_argument("--vc_ckpt", type=str, default=None)
    parser.add_argument("--ot_ckpt", type=str, default=None)
    parser.add_argument(
        "--object_trans_arch",
        "--object-trans-arch",
        type=_normalize_object_trans_arch,
        choices=OBJECT_TRANS_ARCH_CHOICES,
        default=None,
        help=(
            "ObjectTrans架构（不区分大小写）；legacy为无prior多head，"
            "legacy_mesh为ObsEncoder多head，mergeHeads_obs为共享head+ObsEncoder，"
            "v2为固定VC权重，v3为不确定性感知的分层FK融合，"
            "imuhoi为EgoIMU原始三路融合，"
            "imuhoi_4heads为增加静止头的四路融合。"
        ),
    )
    parser.add_argument(
        "--velocity_contact_arch",
        "--velocity-contact-arch",
        type=normalize_velocity_contact_arch,
        choices=VELOCITY_CONTACT_ARCH_CHOICES,
        default='current',
        help=(
            "VelocityContact架构（不区分大小写）；legacy不包含interaction "
            "boundary模块，v2为velocity→obj_move→direct hand-contact级联，"
            "imuhoi为EgoIMU VC。"
        ),
    )
    parser.add_argument(
        "--object_trans_state_feedback",
        choices=("none", "fused"),
        default=None,
        help="Override OT feedback mode for a shared-head checkpoint.",
    )
    parser.add_argument(
        "--human_pose_smoothing_half_width",
        type=int,
        default=None,
        help="Override the HumanPose output smoothing half-width in frames (0 disables; must be <= the online lookahead).",
    )
    parser.add_argument(
        "--online_obj_vel_init_diff_frames",
        type=int,
        default=None,
        help="Initialise the VC object velocity from the published object trajectory differenced over N frames (0 = VC's own output).",
    )
    parser.add_argument(
        "--object_trans_nocontact_lock_vertical",
        action="store_true",
        default=None,
        help="Inference-only: the OT IMU-integration branch (object moving without hand contact) keeps the previous height (y).",
    )
    parser.add_argument(
        "--contact_init_frame",
        type=int,
        default=0,
        help="Frame whose GT contact labels form the initial contact state (0 = protocol default). "
        "The full-sequence splits carry a frame-0 hand-contact label artifact (always 0).",
    )
    parser.add_argument(
        "--eval_vc_contact_init",
        choices=("gt", "neutral", "zero"),
        default=None,
        help="Diagnostic: VelocityContact's per-window contact initializer (online), instead of the "
        "published contact at the window start.",
    )
    parser.add_argument(
        "--oracle_vc_contact",
        choices=("all", "hands", "move"),
        default=None,
        help="Diagnostic: replace these VC contact channels with GT labels inside the online loop.",
    )
    parser.add_argument(
        "--oracle_vc_region",
        choices=("all", "boundary", "interior"),
        default="all",
        help="Diagnostic: apply the contact oracle only within (boundary) or beyond (interior) "
        "--oracle_vc_boundary_frames of each GT transition.",
    )
    parser.add_argument("--oracle_vc_boundary_frames", type=int, default=10)
    parser.add_argument(
        "--oracle_vc_shift",
        type=int,
        default=0,
        help="Diagnostic: delay the oracle GT contact labels by this many frames (negative = earlier).",
    )
    parser.add_argument(
        "--oracle_vc_obj_vel",
        action="store_true",
        help="Diagnostic: replace the VC object velocity with GT inside the online loop.",
    )
    parser.add_argument("--hand_contact_threshold", type=float, default=None)
    parser.add_argument("--object_contact_threshold", type=float, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--sequence_records_json",
        "--sequence-records-json",
        type=str,
        default=None,
        help="Save per-sequence object/HOI metrics and diagnostic factors as JSON.",
    )
    parser.add_argument("--seed", type=int, default=42)
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--simulate_imu_noise",
        "--simulate-imu-noise",
        action="store_true",
        help=(
            "Enable Dataset.simulate_imu_noise for full-sequence evaluation. "
            "Default: disabled."
        ),
    )
    noise_group.add_argument(
        "--eval_imu_noise",
        "--eval-imu-noise",
        action="store_true",
        help="Apply one fixed split-Gaussian IMU noise draw per full sequence before online slicing.",
    )
    parser.add_argument(
        "--simulate_imu_noise_seed",
        "--simulate-imu-noise-seed",
        type=int,
        default=42,
        help="Base seed for deterministic Dataset IMU noise; sequence path is mixed in.",
    )
    parser.add_argument(
        "--eval_imu_noise_seed",
        "--eval-imu-noise-seed",
        type=int,
        default=42,
        help="Base seed for sequence-level noisy eval; sequence path is mixed in for stable per-sequence noise.",
    )
    parser.add_argument("--interaction_human_source", type=str, default="pred", choices=["pred", "gt"])
    refine_group = parser.add_mutually_exclusive_group()
    refine_group.add_argument("--enable_ot_refine", dest="ot_refine", action="store_true", default=None)
    refine_group.add_argument("--disable_ot_refine", dest="ot_refine", action="store_false")
    return parser


def main() -> int:
    args = _make_parser().parse_args()
    if args.smpl_model_path:
        print("Warning: ignoring deprecated --smpl_model_path; no eval BodyModel is loaded.")
    if args.online_batch_size < 1:
        raise ValueError("--online_batch_size must be >= 1")

    seed = _set_eval_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Model arch: rnn")
    print(f"Mode: {'noTrans' if args.no_trans else 'Normal'}")
    print(f"Batch online size: {args.online_batch_size}")
    print("Batch scheduler: sorted repeat-padding + strict per-sequence crop")
    print(f"VelocityContact architecture: {args.velocity_contact_arch or 'config/default'}")
    print(f"Eval seed: {seed if seed is not None else 'unset'}")
    print(
        "Eval IMU noise: "
        f"{'enabled' if args.eval_imu_noise else 'disabled'}"
        + (f" | base_seed={args.eval_imu_noise_seed}" if args.eval_imu_noise else "")
    )
    print(
        "simulate_imu_noise: "
        f"{'enabled' if args.simulate_imu_noise else 'disabled'}"
        + (f" | base_seed={args.simulate_imu_noise_seed}" if args.simulate_imu_noise else "")
        + " (sequence-level)"
    )

    try:
        config_for_dataset_lookup = load_config(args.config)
        default_config = get_default_dataset_config(config=config_for_dataset_lookup)
        dataset_runs = _build_dataset_runs(args, default_config)
    except Exception as exc:
        print(f"[BatchEval] Argument error: {exc}")
        return 2

    all_results = {}
    all_sequence_records = {}
    model_body_models = {}
    resolved_ot_state_feedback = None
    resolved_hp_smoothing = 0
    resolved_obj_vel_diff = 0
    resolved_vertical_lock = False
    resolved_ot_arch = None
    for dataset_name, dataset_cfg in dataset_runs:
        print(f"\n=== Batch-online evaluating dataset: {dataset_name} ===")
        config = load_config(args.config)
        config.model_arch = "rnn"
        if args.object_trans_arch is not None:
            config.object_trans_arch = args.object_trans_arch
        resolved_ot_arch = _normalize_object_trans_arch(
            getattr(config, "object_trans_arch", "current")
        )
        config.object_trans_arch = resolved_ot_arch
        if args.velocity_contact_arch is not None:
            config.velocity_contact_arch = args.velocity_contact_arch
        resolved_vc_arch = normalize_velocity_contact_arch(
            getattr(config, "velocity_contact_arch", "current")
        )
        config.velocity_contact_arch = resolved_vc_arch
        if args.object_trans_state_feedback is not None:
            config.object_trans_state_feedback = args.object_trans_state_feedback == "fused"
        resolved_ot_state_feedback = bool(getattr(config, "object_trans_state_feedback", False))
        ot_refine = False if is_imuhoi_object_trans_arch(
            resolved_ot_arch
        ) else (
            bool(args.ot_refine)
            if args.ot_refine is not None
            else bool(getattr(config, "enable_ot_refine", False))
        )
        config.enable_ot_refine = ot_refine
        if args.human_pose_smoothing_half_width is not None:
            config.human_pose_smoothing_half_width = int(args.human_pose_smoothing_half_width)
        if args.online_obj_vel_init_diff_frames is not None:
            config.online_obj_vel_init_diff_frames = int(args.online_obj_vel_init_diff_frames)
        if args.object_trans_nocontact_lock_vertical:
            config.object_trans_nocontact_lock_vertical = True
        if args.contact_init_frame:
            config.eval_contact_init_frame = int(args.contact_init_frame)
            print(f"Initial contact state read from frame {args.contact_init_frame}")
        if args.eval_vc_contact_init:
            config.eval_vc_contact_init = args.eval_vc_contact_init
            print(f"DIAGNOSTIC VC window contact_init: {args.eval_vc_contact_init}")
        if args.oracle_vc_contact or args.oracle_vc_obj_vel:
            config.eval_oracle_vc = {
                "contact": args.oracle_vc_contact,
                "region": args.oracle_vc_region,
                "boundary_frames": args.oracle_vc_boundary_frames,
                "shift": args.oracle_vc_shift,
                "obj_vel": bool(args.oracle_vc_obj_vel),
            }
            print(f"DIAGNOSTIC VC oracle: {config.eval_oracle_vc}")
        resolved_hp_smoothing = int(getattr(config, "human_pose_smoothing_half_width", 0) or 0)
        resolved_obj_vel_diff = int(getattr(config, "online_obj_vel_init_diff_frames", 0) or 0)
        resolved_vertical_lock = bool(getattr(config, "object_trans_nocontact_lock_vertical", False))
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

        data_dir_default = get_full_sequence_test_path(dataset_cfg)
        if data_dir_default is None:
            print(f"[BatchEval] Skipping '{dataset_name}' (no dataset directory configured).")
            continue
        try:
            data_path = resolve_dataset_path(data_dir_default, PROJECT_ROOT)
        except ValueError as exc:
            print(f"[BatchEval] Skipping '{dataset_name}': {exc}")
            continue
        if not data_path.exists():
            print(f"[BatchEval] Skipping '{dataset_name}' (data not found at {data_path}).")
            continue

        print(f"Loading test dataset from: {data_path}")
        test_window = config.test.get("window", config.train.get("window", 60))
        test_dataset = IMUDataset(
            data_dir=str(data_path),
            sequence_paths=args.sequence_path,
            window_size=test_window,
            debug=config.get("debug", False),
            # full_sequence=True makes simulate_imu_noise operate once per sequence,
            # before the online evaluator creates its rolling windows.
            simulate_imu_noise=bool(args.simulate_imu_noise),
            simulate_imu_noise_seed=args.simulate_imu_noise_seed,
            min_obj_contact_frames=0,
            full_sequence=True,
            resolve_bimanual_contact_conflicts=config.get("resolve_bimanual_contact_conflicts", False),
            require_canonical_skeleton=bool(getattr(config, "require_canonical_skeleton", False)),
        )
        if len(test_dataset) == 0:
            print(f"[BatchEval] Skipping '{dataset_name}' (dataset is empty).")
            continue

        model_body_path, model_body_type, model_body_source = resolve_evaluation_body_model_path(
            config, dataset_cfg, test_dataset.sequence_info,
        )
        config.body_model_path = model_body_path
        model_body_models[dataset_name] = {
            "path": model_body_path,
            "type": model_body_type,
            "source": model_body_source,
            "loaded_by_evaluator": False,
        }
        print(
            "Model canonical BodyModel: "
            f"{model_body_type} from {model_body_source} ({model_body_path})"
        )
        print("Joint metrics: model pred_joints_global vs dataset canonical position_global")

        model = load_model(
            config,
            device,
            no_trans=args.no_trans,
            module_paths=module_paths,
            object_trans_arch=args.object_trans_arch,
            velocity_contact_arch=args.velocity_contact_arch,
        )
        model = _unwrap_model(model)
        model.eval()

        evaluate_contacts = (not args.no_objects) and has_velocity_contact_ckpt
        evaluate_objects = (not args.no_objects) and has_velocity_contact_ckpt
        run_object_branch = evaluate_objects and has_object_trans_ckpt
        if not args.no_objects:
            if not has_velocity_contact_ckpt:
                print("[BatchEval] No velocity_contact checkpoint loaded; skipping contact/object metrics.")
            elif not has_object_trans_ckpt:
                print("[BatchEval] No object_trans checkpoint loaded; object branch is disabled.")

        online_window = resolve_online_window(config, args.online_window)
        online_lookahead = resolve_online_lookahead(config, args.online_lookahead)
        online_chunk = int(
            args.online_chunk
            if args.online_chunk is not None
            else getattr(config, "online_chunk", 1)
        )
        config.online_chunk = online_chunk
        compute_fk = bool(args.compare_3 and run_object_branch)
        print(f"Dataset size: {len(test_dataset)}")
        print(f"Online window: {online_window}")
        print(f"Online lookahead: {online_lookahead} (strict tail)")
        print(f"Online chunk: {online_chunk} window(s) per batched HP/VC forward")
        print(
            "OT predictor feedback: "
            f"{'fused' if bool(getattr(config, 'object_trans_state_feedback', False)) else 'none'}"
        )
        print(f"VC architecture: {resolved_vc_arch}")
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
                online_lookahead=online_lookahead,
                use_object_data=run_object_branch,
                compute_fk=compute_fk,
                refine_human=ot_refine,
                sort_desc=not args.sort_asc,
                eval_imu_noise=bool(args.eval_imu_noise),
                eval_imu_noise_seed=args.eval_imu_noise_seed,
                online_chunk=online_chunk,
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
        sequence_records = [] if args.sequence_records_json else None
        results = evaluate_model(
            metric_model,
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
            online_lookahead=online_lookahead,
            sequence_records=sequence_records,
        )
        eval_duration = time.time() - eval_start
        all_results[dataset_name] = results
        if sequence_records is not None:
            all_sequence_records[dataset_name] = sequence_records
            print(f"Per-sequence diagnostic records: {len(sequence_records)}")

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
            print(f"Fusion raw (cm):                {_format_metric(results, 'obj_trans_err_fusion_raw')}")
            print(f"Fusion refine/published (cm):   {_format_metric(results, 'obj_trans_err_fusion_refine')}")
            print(f"FK (cm):                        {_format_metric(results, 'obj_trans_err_fk')}")
            print(f"IMU (cm):                       {_format_metric(results, 'obj_trans_err_imu')}")
            print("\n--- HOI Errors ---")
            print(f"Fusion published (cm):          {_format_metric(results, 'hoi_err_fusion')}")
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
        output_config = load_config(args.config)
        output_online_window = resolve_online_window(
            output_config,
            args.online_window,
        )
        output_online_lookahead = resolve_online_lookahead(
            output_config,
            args.online_lookahead,
        )
        payload = {
            "config": args.config,
            **hoi_metric_metadata(config),
            "inference_mode": "batch_online",
            "online_window": output_online_window,
            "resolved_online_window": output_online_window,
            "online_lookahead": output_online_lookahead,
            "resolved_online_lookahead": output_online_lookahead,
            "online_batch_size": args.online_batch_size,
            "online_chunk": int(args.online_chunk) if args.online_chunk is not None else 1,
            "object_trans_state_feedback": resolved_ot_state_feedback,
            "human_pose_smoothing_half_width": resolved_hp_smoothing,
            "online_obj_vel_init_diff_frames": resolved_obj_vel_diff,
            "object_trans_nocontact_lock_vertical": resolved_vertical_lock,
            "eval_oracle_vc": getattr(config, "eval_oracle_vc", None),
            "eval_vc_contact_init": getattr(config, "eval_vc_contact_init", None),
            "contact_init_frame": int(getattr(config, "eval_contact_init_frame", 0) or 0),
            "object_trans_arch": resolved_ot_arch or _normalize_object_trans_arch(
                getattr(load_config(args.config), "object_trans_arch", "current")
            ),
            "velocity_contact_arch": args.velocity_contact_arch or getattr(load_config(args.config), "velocity_contact_arch", "current"),
            "seed": seed,
            "simulate_imu_noise": bool(args.simulate_imu_noise),
            "simulate_imu_noise_seed": (
                args.simulate_imu_noise_seed if args.simulate_imu_noise else None
            ),
            "eval_imu_noise": bool(args.eval_imu_noise),
            "eval_imu_noise_seed": args.eval_imu_noise_seed if args.eval_imu_noise else None,
            "joint_metric_source": "pred_joints_global_vs_canonical_position_global",
            "model_body_models": model_body_models,
            "results": all_results,
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[BatchEval] Results saved to: {output_path}")

    if args.sequence_records_json:
        records_path = Path(args.sequence_records_json).expanduser()
        if not records_path.is_absolute():
            records_path = (Path.cwd() / records_path).resolve()
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_payload = {
            "config": args.config,
            "protocol": {
                "inference_mode": "batch_online",
                "online_window": resolve_online_window(
                    load_config(args.config), args.online_window
                ),
                "online_lookahead": resolve_online_lookahead(
                    load_config(args.config), args.online_lookahead
                ),
                "online_batch_size": args.online_batch_size,
                "seed": seed,
                "simulate_imu_noise": bool(args.simulate_imu_noise),
                "simulate_imu_noise_seed": (
                    args.simulate_imu_noise_seed if args.simulate_imu_noise else None
                ),
                "eval_imu_noise": bool(args.eval_imu_noise),
                "eval_imu_noise_seed": (
                    args.eval_imu_noise_seed if args.eval_imu_noise else None
                ),
                "velocity_contact_arch": args.velocity_contact_arch,
                "object_trans_arch": resolved_ot_arch,
                "object_trans_state_feedback": resolved_ot_state_feedback,
                "object_trans_nocontact_lock_vertical": resolved_vertical_lock,
                "human_pose_checkpoint": args.hp_ckpt,
                "velocity_contact_checkpoint": args.vc_ckpt,
                "object_trans_checkpoint": args.ot_ckpt,
                "sequence_paths": args.sequence_path,
                "joint_metric_source": "pred_joints_global_vs_canonical_position_global",
                "model_body_models": model_body_models,
            },
            "aggregate_results": all_results,
            "datasets": all_sequence_records,
        }
        records_path.write_text(
            json.dumps(records_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[BatchEval] Per-sequence diagnostics saved to: {records_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
