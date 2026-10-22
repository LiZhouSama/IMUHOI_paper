"""Utilities for RNN offline/online inference."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch


InferenceOutput = Dict[str, Any]


_SEQUENCE_LENGTH_KEYS = (
    "human_imu",
    "hand_positions",
    "p_pred",
    "v_pred",
    "root_trans_pred",
    "pred_hand_glb_pos",
    "pred_palm_glb_pos",
    "pred_hand_contact_prob",
    "pred_obj_vel",
    "pred_obj_trans",
)

# Keys whose tensors are per-sequence metadata with a small fixed second
# dimension (2, 3, N points, ...).  The shape-based temporal heuristic
# misclassifies them whenever the window length happens to coincide with that
# dimension, silently slicing init vectors and priors, so list them explicitly.
NON_TEMPORAL_KEYS = frozenset(
    {
        "v_init",
        "p_init",
        "trans_init",
        "prev_foot_pos",
        "hand_vel_glb_init",
        "obj_vel_init",
        "obj_trans_init",
        "contact_init",
        "has_object",
        "obj_points_canonical",
        "lb_prior_quantiles",
        "init_lhand_oe_ho",
        "init_rhand_oe_ho",
        "init_lhand_lb",
        "init_rhand_lb",
        "gating_smoothing_applied",
    }
)


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


def resolve_online_lookahead(
    cfg: Any = None,
    online_lookahead: Optional[int] = None,
    default: int = 5,
) -> int:
    """Resolve the fixed-lag online delay in frames.

    The legacy base window is preserved and ``D`` frames are appended after
    its update frame as future context.  A delay of ``D`` therefore publishes
    frame ``t`` only after frames through ``t + D`` have arrived.  Strict end
    handling emits ``max(T - D, 0)`` frames for a sequence of length ``T``;
    the unseen tail is never padded or flushed with less context.  Setting
    ``D=0`` recovers the original warmup and sliding-window behavior.
    """
    if online_lookahead is not None:
        return max(int(online_lookahead), 0)

    test_cfg = getattr(cfg, "test", None) if cfg is not None else None
    if isinstance(test_cfg, dict):
        value = test_cfg.get("online_lookahead")
    else:
        value = getattr(test_cfg, "online_lookahead", None) if test_cfg is not None else None
    if value is None and cfg is not None:
        value = getattr(cfg, "online_lookahead", None)
    return max(int(default if value is None else value), 0)


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
    return {
        key: value if key in NON_TEMPORAL_KEYS else slice_time_value(value, start, end, batch_size, seq_len)
        for key, value in data.items()
    }


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


def _iter_named_tensors(output: InferenceOutput):
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            yield key, value
        elif isinstance(value, dict):
            yield from _iter_named_tensors(value)


def _valid_sequence_candidate(value: Any) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.dim() >= 2
        and int(value.shape[0]) > 0
        and int(value.shape[1]) > 0
    )


def concat_time_dicts(chunks: Iterable[InferenceOutput]) -> InferenceOutput:
    chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
    if not chunks:
        return {}

    chunk_seq = []
    for chunk in chunks:
        try:
            chunk_seq.append(_infer_output_batch_seq(chunk))
        except ValueError:
            chunk_seq.append(None)

    result: InferenceOutput = {}
    keys = []
    seen_keys = set()
    for chunk in chunks:
        for key in chunk.keys():
            if key not in seen_keys:
                seen_keys.add(key)
                keys.append(key)

    for key in keys:
        values_with_seq = [
            (chunk[key], seq_info)
            for chunk, seq_info in zip(chunks, chunk_seq)
            if key in chunk
        ]
        values = [value for value, _ in values_with_seq]
        if not values:
            continue
        if key in NON_TEMPORAL_KEYS:
            result[key] = values[-1]
            continue
        if all(isinstance(value, dict) for value in values):
            result[key] = concat_time_dicts(values)  # type: ignore[arg-type]
            continue
        if all(
            isinstance(value, torch.Tensor)
            and seq_info is not None
            and is_temporal_tensor(value, seq_info[0], seq_info[1])
            for value, seq_info in values_with_seq
        ):
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


class _TemporalBuffer:
    def __init__(self, value: torch.Tensor, total_seq_len: int, seq_len: int):
        self.buffer = value.new_empty((value.shape[0], int(total_seq_len), *value.shape[2:]))
        self.length = 0
        self.append(value, seq_len)

    def can_append(self, value: Any) -> bool:
        return (
            isinstance(value, torch.Tensor)
            and value.dim() == self.buffer.dim()
            and value.shape[0] == self.buffer.shape[0]
            and value.shape[2:] == self.buffer.shape[2:]
            and value.device == self.buffer.device
            and value.dtype == self.buffer.dtype
            and self.length + int(value.shape[1]) <= self.buffer.shape[1]
        )

    def append(self, value: torch.Tensor, seq_len: Optional[int] = None) -> None:
        append_len = int(value.shape[1] if seq_len is None else seq_len)
        end = self.length + append_len
        self.buffer[:, self.length:end].copy_(value[:, :append_len])
        self.length = end

    def current(self) -> torch.Tensor:
        return self.buffer[:, :self.length]


class TimeDictAccumulator:
    """Append latest-frame inference dicts without rebuilding full history tensors."""

    def __init__(
        self,
        initial: InferenceOutput,
        total_seq_len: int,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
    ):
        if batch_size is None or seq_len is None:
            batch_size, seq_len = _infer_output_batch_seq(initial)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.total_seq_len = int(total_seq_len)
        self._values: Dict[str, Any] = {
            key: self._init_value(key, value, self.seq_len)
            for key, value in initial.items()
        }

    def _init_value(self, key: str, value: Any, seq_len: int) -> Any:
        if key in NON_TEMPORAL_KEYS:
            return value
        if is_temporal_tensor(value, self.batch_size, seq_len):
            return _TemporalBuffer(value, self.total_seq_len, seq_len)
        if isinstance(value, dict):
            return TimeDictAccumulator(
                value,
                self.total_seq_len,
                batch_size=self.batch_size,
                seq_len=seq_len,
            )
        return value

    def append(self, latest: InferenceOutput) -> InferenceOutput:
        if not isinstance(latest, dict):
            return self.current()
        try:
            _, append_len = _infer_output_batch_seq(latest)
        except ValueError:
            append_len = 0

        for key, value in latest.items():
            if key not in self._values:
                # Match concat_time_dicts semantics for keys that first appear
                # after warmup: keep the latest value instead of pretending a
                # full absolute-time history exists.
                self._values[key] = value
                continue

            current_value = self._values[key]
            if isinstance(current_value, _TemporalBuffer):
                if current_value.can_append(value):
                    current_value.append(value)
                else:
                    # Replacing the buffer silently truncates accumulated
                    # history; make the (unexpected) shape/dtype drift visible.
                    print(
                        f"Warning: TimeDictAccumulator dropped buffered history for '{key}' "
                        "(incoming value is incompatible with the temporal buffer)."
                    )
                    self._values[key] = value
                continue

            if isinstance(current_value, TimeDictAccumulator) and isinstance(value, dict):
                current_value.append(value)
                continue

            self._values[key] = value

        if append_len > 0:
            self.seq_len += append_len
        return self.current()

    def current(self) -> InferenceOutput:
        out: InferenceOutput = {}
        for key, value in self._values.items():
            if isinstance(value, _TemporalBuffer):
                out[key] = value.current()
            elif isinstance(value, TimeDictAccumulator):
                out[key] = value.current()
            else:
                out[key] = value
        return out


def select_time_context(history: InferenceOutput, start: int, end: int) -> InferenceOutput:
    if not history:
        return {}
    batch_size, seq_len = _infer_output_batch_seq(history)
    return slice_time_dict(history, start, end, batch_size, seq_len)


def _infer_output_batch_seq(output: InferenceOutput) -> Tuple[int, int]:
    tensors = list(_iter_named_tensors(output))
    for preferred_key in _SEQUENCE_LENGTH_KEYS:
        for key, value in tensors:
            if (
                key == preferred_key
                and isinstance(value, torch.Tensor)
                and value.dim() >= 2
                and int(value.shape[0]) > 0
            ):
                return int(value.shape[0]), int(value.shape[1])

    counts = {}
    for _, value in tensors:
        if _valid_sequence_candidate(value):
            shape = (int(value.shape[0]), int(value.shape[1]))
            counts[shape] = counts.get(shape, 0) + 1
    if counts:
        return max(counts, key=lambda shape: (counts[shape], shape[1]))
    raise ValueError("No temporal tensor found in output")


def _history_tensor(output: InferenceOutput, key: str, index: int = -1) -> Optional[torch.Tensor]:
    value = output.get(key)
    if isinstance(value, torch.Tensor) and value.dim() >= 2:
        return value[:, index].detach()
    return None


def _published_object_velocity(
    history: InferenceOutput, index: int, frames: int, fps: float
) -> Optional[torch.Tensor]:
    """Backward difference of the published object position ending at ``index``."""
    trans = history.get("pred_obj_trans")
    if not isinstance(trans, torch.Tensor) or trans.dim() != 3:
        return None
    length = trans.shape[1]
    end = index if index >= 0 else length + index
    span = min(int(frames), end)
    if end >= length or span <= 0:
        return None
    return ((trans[:, end] - trans[:, end - span]) * (fps / span)).detach()


def update_data_inits_from_history(
    data: Dict[str, Any],
    history: InferenceOutput,
    index: int = -1,
    contact_index: Optional[int] = None,
    trans_index: Optional[int] = None,
    obj_vel_diff_frames: int = 0,
    fps: float = 30.0,
    hp_init_smoothing=None,
) -> Dict[str, Any]:
    """Refresh init tensors from known predictions.

    ``index`` keeps predecessor-frame semantics for recurrent initializers.
    ``contact_index`` and ``trans_index`` can independently select the first
    frame of the current window, because those values represent frame-0 state.

    ``obj_vel_diff_frames`` > 0 initialises the object velocity from the
    PUBLISHED object trajectory (backward difference over that many frames)
    instead of VelocityContact's own previous output.  The velocity RNN
    integrates the object IMU from its initialiser, so feeding it its own
    output makes the velocity error a random walk over the sequence; the
    published object is anchored by the FK/static branches, which turns that
    loop into a leaky integrator.
    """
    if not history:
        return data

    updated = dict(data)

    # ``hp_init_smoothing(history_tensor, index, is_rotation)`` de-noises the
    # recurrent HumanPose initialisers from the already published frames around
    # ``index``.  Every published frame comes from its own window pass, whose
    # offset is inherited from its initialiser; white noise in the initialiser
    # chain therefore survives in the published pose forever.
    value = _history_tensor(history, "v_pred", index)
    if value is not None:
        if callable(hp_init_smoothing):
            value = hp_init_smoothing(history["v_pred"], index, False)
        updated["v_init"] = value.reshape(value.shape[0], -1, 3)

    value = _history_tensor(history, "p_pred", index)
    if value is not None:
        if callable(hp_init_smoothing):
            value = hp_init_smoothing(history["p_pred"], index, True)
        updated["p_init"] = value.reshape(value.shape[0], -1, 6)

    resolved_trans_index = index if trans_index is None else trans_index
    value = _history_tensor(history, "root_trans_pred", resolved_trans_index)
    if value is not None:
        updated["trans_init"] = value

    value = _history_tensor(history, "pred_foot_pos", index)
    if value is not None:
        updated["prev_foot_pos"] = value

    value = _history_tensor(history, "pred_hand_glb_vel", index)
    if value is not None:
        updated["hand_vel_glb_init"] = value.reshape(value.shape[0], 2, 3)

    value = _history_tensor(history, "pred_obj_vel", index)
    if value is not None:
        updated["obj_vel_init"] = value
    if int(obj_vel_diff_frames) > 0:
        diff_velocity = _published_object_velocity(history, index, int(obj_vel_diff_frames), float(fps))
        if diff_velocity is not None:
            if value is not None:
                diff_velocity = diff_velocity.to(dtype=value.dtype)
            updated["obj_vel_init"] = diff_velocity

    resolved_contact_index = index if contact_index is None else contact_index
    for contact_key in ("pred_hand_contact_prob_frame", "pred_hand_contact_prob"):
        value = _history_tensor(history, contact_key, resolved_contact_index)
        if value is not None:
            updated["contact_init"] = value[..., :3]
            break

    value = _history_tensor(history, "pred_obj_trans", index)
    if value is not None:
        updated["obj_trans_init"] = value

    return updated


def merge_latest_context(prefix: InferenceOutput, latest: InferenceOutput) -> InferenceOutput:
    if not prefix:
        return latest
    return concat_time_dicts([prefix, latest])
