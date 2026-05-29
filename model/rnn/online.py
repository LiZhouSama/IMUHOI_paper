"""Utilities for RNN offline/online inference."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch


InferenceOutput = Dict[str, Any]


def normalize_inference_mode(inference_mode: str) -> str:
    mode = str(inference_mode or "offline").lower()
    if mode not in {"offline", "online"}:
        raise ValueError(f"inference_mode must be 'offline' or 'online', got {inference_mode!r}")
    return mode


def resolve_online_window(cfg: Any = None, online_window: Optional[int] = None, default: int = 120) -> int:
    if online_window is not None:
        return max(int(online_window), 1)

    for section_name in ("test", "train"):
        section = getattr(cfg, section_name, None) if cfg is not None else None
        value = None
        if isinstance(section, dict):
            value = section.get("window")
        elif section is not None:
            value = getattr(section, "window", None)
        if value is not None:
            return max(int(value), 1)
    return int(default)


def infer_batch_seq(data_dict: Dict[str, Any], key: str = "human_imu") -> Tuple[int, int]:
    value = data_dict.get(key)
    if not isinstance(value, torch.Tensor) or value.dim() < 2:
        raise ValueError(f"{key} must be a tensor with batch/time dimensions")
    return int(value.shape[0]), int(value.shape[1])


def is_temporal_tensor(value: Any, batch_size: int, seq_len: int) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.dim() >= 2
        and int(value.shape[0]) == int(batch_size)
        and int(value.shape[1]) == int(seq_len)
    )


def slice_time_value(value: Any, start: int, end: int, batch_size: int, seq_len: int) -> Any:
    if is_temporal_tensor(value, batch_size, seq_len):
        return value[:, start:end]
    if isinstance(value, dict):
        return {k: slice_time_value(v, start, end, batch_size, seq_len) for k, v in value.items()}
    return value


def slice_time_dict(data: Dict[str, Any], start: int, end: int, batch_size: int, seq_len: int) -> Dict[str, Any]:
    return {key: slice_time_value(value, start, end, batch_size, seq_len) for key, value in data.items()}


def append_stream_data(
    previous: Optional[Dict[str, Any]],
    current: Dict[str, Any],
    sequence_key: str = "human_imu",
) -> Tuple[Dict[str, Any], int]:
    """Append a streaming chunk to buffered model inputs.

    Non-temporal tensors such as init vectors are preserved from the previous
    buffer so a recomputed warmup sequence still uses the original start state.
    """
    if not previous:
        return dict(current), 0

    batch_size, prev_seq = infer_batch_seq(previous, key=sequence_key)
    _, cur_seq = infer_batch_seq(current, key=sequence_key)
    merged: Dict[str, Any] = {}
    keys = set(previous.keys()) | set(current.keys())
    for key in keys:
        prev_value = previous.get(key)
        cur_value = current.get(key)
        if is_temporal_tensor(prev_value, batch_size, prev_seq) and is_temporal_tensor(cur_value, batch_size, cur_seq):
            merged[key] = torch.cat((prev_value, cur_value), dim=1)
        elif key in previous:
            merged[key] = prev_value
        else:
            merged[key] = cur_value
    return merged, prev_seq


def take_latest_frame(output: InferenceOutput, batch_size: int, seq_len: int) -> InferenceOutput:
    return slice_time_dict(output, seq_len - 1, seq_len, batch_size, seq_len)


def _temporal_value_for_concat(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() >= 2


def concat_time_dicts(chunks: Iterable[InferenceOutput]) -> InferenceOutput:
    chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
    if not chunks:
        return {}

    result: InferenceOutput = {}
    keys = set()
    for chunk in chunks:
        keys.update(chunk.keys())

    for key in keys:
        values = [chunk[key] for chunk in chunks if key in chunk]
        if not values:
            continue
        if all(isinstance(value, dict) for value in values):
            result[key] = concat_time_dicts(values)  # type: ignore[arg-type]
            continue
        if all(_temporal_value_for_concat(value) for value in values):
            first = values[0]
            assert isinstance(first, torch.Tensor)
            if all(
                isinstance(value, torch.Tensor)
                and value.dim() == first.dim()
                and value.shape[0] == first.shape[0]
                and value.shape[2:] == first.shape[2:]
                for value in values
            ):
                result[key] = torch.cat(values, dim=1)
                continue
        result[key] = values[-1]
    return result


def select_time_context(history: InferenceOutput, start: int, end: int) -> InferenceOutput:
    if not history:
        return {}
    batch_size, seq_len = _infer_output_batch_seq(history)
    return slice_time_dict(history, start, end, batch_size, seq_len)


def _infer_output_batch_seq(output: InferenceOutput) -> Tuple[int, int]:
    for value in output.values():
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            return int(value.shape[0]), int(value.shape[1])
        if isinstance(value, dict):
            try:
                return _infer_output_batch_seq(value)
            except ValueError:
                pass
    raise ValueError("No temporal tensor found in output")


def _history_tensor(output: InferenceOutput, key: str, index: int = -1) -> Optional[torch.Tensor]:
    value = output.get(key)
    if isinstance(value, torch.Tensor) and value.dim() >= 2:
        return value[:, index].detach()
    return None


def update_data_inits_from_history(data: Dict[str, Any], history: InferenceOutput, index: int = -1) -> Dict[str, Any]:
    """Refresh init tensors from the latest known predictions."""
    if not history:
        return data

    updated = dict(data)

    value = _history_tensor(history, "v_pred", index)
    if value is not None:
        updated["v_init"] = value.reshape(value.shape[0], -1, 3)

    value = _history_tensor(history, "p_pred", index)
    if value is not None:
        updated["p_init"] = value.reshape(value.shape[0], -1, 6)

    value = _history_tensor(history, "root_trans_pred", index)
    if value is not None:
        updated["trans_init"] = value

    value = _history_tensor(history, "pred_hand_glb_vel", index)
    if value is not None:
        updated["hand_vel_glb_init"] = value.reshape(value.shape[0], 2, 3)

    value = _history_tensor(history, "pred_obj_vel", index)
    if value is not None:
        updated["obj_vel_init"] = value

    value = _history_tensor(history, "pred_hand_contact_prob", index)
    if value is not None:
        updated["contact_init"] = value[..., :3]

    value = _history_tensor(history, "pred_obj_trans", index)
    if value is not None:
        updated["obj_trans_init"] = value

    return updated


def merge_latest_context(prefix: InferenceOutput, latest: InferenceOutput) -> InferenceOutput:
    if not prefix:
        return latest
    return concat_time_dicts([prefix, latest])
