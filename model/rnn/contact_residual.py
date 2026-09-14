"""Small, label-isolated residuals on frozen online HP/VC/OT predictions.

Only two sensor fields with a ``gt_`` prefix are observations: ``gt_obj_imu``
and ``gt_human_imu``. All other labels and dump metadata are deliberately
unread. No acceleration integration, human correction, or loss lives here.

Build features once per published sequence, then slice them into GRU chunks
while carrying the returned state. For streaming feature construction, prepend
the previous observed frame to each new chunk and discard that context frame's
features. The builder has only one-frame lookback and no sequence statistics.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from utils.object_fusion import build_contact_fusion_weights
from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix

__all__ = [
    "CONTACT_RESIDUAL_FEATURE_DIM",
    "CONTACT_RESIDUAL_FEATURE_SLICES",
    "CONTACT_RESIDUAL_TEMPORAL_SLICE",
    "build_contact_residual_features",
    "build_contact_residual_feature_mask",
    "CausalContactResidual",
    "PointwiseContactResidual",
    "apply_contact_residual",
]

# Fixed dataset units, not normalization fitted on a sequence or evaluation set.
_FRAME_RATE = 30.0
_POSITION_SCALE = 1.0  # m
_ACCELERATION_SCALE = 10.0  # m/s^2
_VELOCITY_SCALE = 2.0  # m/s
_FEATURE_CLIP = 5.0
_ROOT_SENSOR = 0
_FOREARM_SENSORS = (4, 5)
_IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
_FEATURE_BLOCKS = (
    ("object_from_root", 3),
    ("palms_from_root", 6),
    ("object_from_palms", 6),
    ("candidate_offsets", 6),
    ("candidate_disagreement", 3),
    ("bones", 6),
    ("proposal_offsets", 6),
    ("bone_alpha", 2),
    ("contact_prob", 3),
    ("gating_weights", 4),
    ("object_acc", 3),
    ("forearm_relative_acc", 6),
    ("root_relative_rotation", 6),
    ("forearm_relative_rotation", 12),
    ("relative_geometry_velocity", 9),
    ("bone_velocity", 6),
    ("contact_change", 3),
    ("object_rotation_change", 6),
    ("raw_published_offset", 3),
)
CONTACT_RESIDUAL_FEATURE_SLICES: dict[str, slice] = {}
CONTACT_RESIDUAL_FEATURE_DIM = 0
for _name, _width in _FEATURE_BLOCKS:
    CONTACT_RESIDUAL_FEATURE_SLICES[_name] = slice(
        CONTACT_RESIDUAL_FEATURE_DIM, CONTACT_RESIDUAL_FEATURE_DIM + _width
    )
    CONTACT_RESIDUAL_FEATURE_DIM += _width
CONTACT_RESIDUAL_TEMPORAL_SLICE = slice(72, 96)


def build_contact_residual_feature_mask(feature_dim: int, ablation: str = "full") -> torch.Tensor:
    """Fixed, checkpointed feature ablations applied AFTER train normalization."""
    mask = torch.ones(feature_dim, dtype=torch.float32)
    if ablation == "full":
        return mask
    if feature_dim != CONTACT_RESIDUAL_FEATURE_DIM:
        raise ValueError("Structured ablation requires the current feature schema.")
    if ablation == "no_publication":
        mask[CONTACT_RESIDUAL_FEATURE_SLICES["raw_published_offset"]] = 0
    elif ablation == "no_history":
        mask[CONTACT_RESIDUAL_TEMPORAL_SLICE] = 0
    elif ablation == "no_extra_imu":
        for key in ("object_acc", "forearm_relative_acc", "root_relative_rotation", "forearm_relative_rotation"):
            mask[CONTACT_RESIDUAL_FEATURE_SLICES[key]] = 0
    else:
        raise ValueError(f"Unsupported feature ablation: {ablation}")
    return mask

def _read_tensor(
    record: Mapping[str, Any],
    key: str,
    base: torch.Tensor,
    tail_shape: tuple[int, ...],
    *,
    required: bool = False,
    check_finite: bool = True,
) -> torch.Tensor | None:
    value = record.get(key)
    if value is None:
        if required:
            raise KeyError(f"Contact residual requires observation/prediction {key!r}")
        return None
    value = torch.as_tensor(value, device=base.device, dtype=torch.float32).detach()
    seq_len = base.shape[0]
    if not tail_shape and value.shape == (seq_len, 1):
        value = value[:, 0]
    expected = (seq_len, *tail_shape)
    if tuple(value.shape) != expected:
        raise ValueError(f"{key} must have shape {expected}, got {tuple(value.shape)}")
    if check_finite and not torch.isfinite(value).all():
        raise ValueError(f"{key} contains non-finite values")
    return value


def _safe_rotation(d6: torch.Tensor) -> torch.Tensor:
    """Use the repository's row-based 6D convention; invalid rows -> identity."""
    finite = torch.isfinite(d6).all(dim=-1)
    clean = torch.nan_to_num(d6, nan=0.0, posinf=0.0, neginf=0.0)
    clean = clean / clean.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    first = F.normalize(clean[..., :3], dim=-1, eps=1e-6)
    second = clean[..., 3:] - (first * clean[..., 3:]).sum(-1, keepdim=True) * first
    usable = finite & (clean[..., :3].norm(dim=-1) > 1e-6) & (second.norm(dim=-1) > 1e-6)
    identity = clean.new_tensor(_IDENTITY_6D)
    return rotation_6d_to_matrix(torch.where(usable[..., None], clean, identity))


def _rotate(rotation: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tij,t...j->t...i", rotation, vectors)


def _backward_difference(value: torch.Tensor) -> torch.Tensor:
    # The first observed frame has no predecessor, not an assumed zero position.
    return torch.cat((torch.zeros_like(value[:1]), value[1:] - value[:-1]), dim=0)


@torch.no_grad()
def build_contact_residual_features(record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return detached float32 features [T,99] and frozen geometry for one dump.

    Required: gt_obj_imu [T,9], gt_human_imu [T,6,9], pred_hand_contact_prob
    [T,3], pred_palm_glb_pos [T,2,3] (FINAL palms), root_trans_pred [T,3]
    (fallback stage1_root_trans_pred), and either pred_obj_trans_published or
    pred_obj_trans [T,3]. Published object positions always take precedence.
    The builder cannot reconstruct published positions: the caller must cache
    them using the formal evaluator. Missing raw positions default to base_obj,
    making the local raw-minus-published feature zero.

    Optional per-hand defaults (lhand/rhand, fixed widths even when absent):
      * pred_*_obj_trans: base_obj (zero candidate offset).
      * pred_*_obj_vector: pred_*_obj_direction * pred_*_lb if both exist,
        otherwise R_obj^T (base_obj - final_palm).
      * pred_*_bone_proposal: the corresponding bone (zero proposal offset).
      * pred_*_bone_gate: one (no temporal latch, as in v2).
      * gating_weights: the fixed contact-to-four-way-fusion mapping. Supplied
        nonnegative weights are normalized; an all-zero row uses that fallback.

    Spatial vectors are in the noisy observed object frame, positions / 1 m,
    accelerations / 10 m/s^2 and local-vector backward differences * 30 / 2 m/s.
    These local derivatives include object-frame rotation, not world velocity.
    Contact differences are per frame; rotation changes are 6D(R_prev^T R)-6D(I).
    All feature columns are clipped to [-5,5]; returned geometry is NOT clipped.
    Bone vectors/proposals are already object-local and are not rotated again.
    Invalid/degenerate 6D orientations use identity, never a GT rotation fallback.
    Non-finite accelerations or geometry and misaligned sequence lengths fail.
    """
    base_key = "pred_obj_trans_published"
    if record.get(base_key) is None:
        base_key = "pred_obj_trans"
    if record.get(base_key) is None:
        raise KeyError("Contact residual requires pred_obj_trans_published or pred_obj_trans")
    base_obj = torch.as_tensor(record[base_key], dtype=torch.float32).detach()
    if base_obj.ndim != 2 or base_obj.shape[-1] != 3:
        raise ValueError(f"{base_key} must have shape [T,3], got {tuple(base_obj.shape)}")
    if not torch.isfinite(base_obj).all():
        raise ValueError(f"{base_key} contains non-finite values")
    raw_obj = _read_tensor(record, "pred_obj_trans", base_obj, (3,))
    if raw_obj is None:
        raw_obj = base_obj
    pred_palm = _read_tensor(record, "pred_palm_glb_pos", base_obj, (2, 3), required=True)
    root_key = "root_trans_pred"
    if record.get(root_key) is None:
        root_key = "stage1_root_trans_pred"
    root = _read_tensor(record, root_key, base_obj, (3,), required=True)
    contact = _read_tensor(record, "pred_hand_contact_prob", base_obj, (3,), required=True)
    contact = contact.clamp(0.0, 1.0)
    obj_imu = _read_tensor(record, "gt_obj_imu", base_obj, (9,), required=True, check_finite=False)
    human_imu = _read_tensor(
        record, "gt_human_imu", base_obj, (6, 9), required=True, check_finite=False
    )
    if not torch.isfinite(obj_imu[..., :3]).all() or not torch.isfinite(human_imu[..., :3]).all():
        raise ValueError("IMU accelerations contain non-finite values")

    obj_rotation = _safe_rotation(obj_imu[:, 3:9])
    obj_inverse = obj_rotation.transpose(-1, -2)
    root_rotation = _safe_rotation(human_imu[:, _ROOT_SENSOR, 3:9])
    root_relative_rotation = obj_inverse @ root_rotation
    forearm_root = _safe_rotation(human_imu[:, _FOREARM_SENSORS, 3:9])
    # Non-root orientations and accelerations in the dump are ROOT-relative.
    forearm_relative_rotation = root_relative_rotation[:, None] @ forearm_root
    forearm_acc_world = _rotate(
        root_rotation,
        human_imu[:, _FOREARM_SENSORS, :3] + human_imu[:, _ROOT_SENSOR, None, :3],
    )
    object_from_root = _rotate(obj_inverse, base_obj - root)
    palms_from_root = _rotate(obj_inverse, pred_palm - root[:, None])
    object_from_palms = _rotate(obj_inverse, base_obj[:, None] - pred_palm)

    candidates, bones, proposals, alphas = [], [], [], []
    for hand_index, hand in enumerate(("lhand", "rhand")):
        candidate = _read_tensor(record, f"pred_{hand}_obj_trans", base_obj, (3,))
        candidates.append(base_obj if candidate is None else candidate)
        bone = _read_tensor(record, f"pred_{hand}_obj_vector", base_obj, (3,))
        if bone is None:
            direction = _read_tensor(record, f"pred_{hand}_obj_direction", base_obj, (3,))
            length = _read_tensor(record, f"pred_{hand}_lb", base_obj, ())
            bone = (
                direction * length[:, None]
                if direction is not None and length is not None
                else object_from_palms[:, hand_index]
            )
        bones.append(bone)
        proposal = _read_tensor(record, f"pred_{hand}_bone_proposal", base_obj, (3,))
        proposals.append(bone if proposal is None else proposal)
        alpha = _read_tensor(record, f"pred_{hand}_bone_gate", base_obj, ())
        alphas.append(base_obj.new_ones(base_obj.shape[0]) if alpha is None else alpha.clamp(0.0, 1.0))
    candidates = torch.stack(candidates, dim=1)
    bones = torch.stack(bones, dim=1)
    proposals = torch.stack(proposals, dim=1)
    alphas = torch.stack(alphas, dim=1)
    weights_default = build_contact_fusion_weights(contact)
    weights = _read_tensor(record, "gating_weights", base_obj, (4,))
    if weights is None:
        weights = weights_default
    else:
        weights = weights.clamp_min(0.0)
        total = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(total > 1e-6, weights / total.clamp_min(1e-6), weights_default)

    previous_rotation = torch.cat((obj_rotation[:1], obj_rotation[:-1]), dim=0)
    rotation_change = matrix_to_rotation_6d(previous_rotation.transpose(-1, -2) @ obj_rotation)
    rotation_change = rotation_change - obj_rotation.new_tensor(_IDENTITY_6D)
    # Avoid roundoff in R(0)^T R(0) masquerading as a first-frame motion signal.
    rotation_change = torch.cat((torch.zeros_like(rotation_change[:1]), rotation_change[1:]), dim=0)
    relative_geometry = torch.cat((object_from_root, object_from_palms.flatten(1)), dim=-1)
    blocks = (
        object_from_root / _POSITION_SCALE,
        palms_from_root.flatten(1) / _POSITION_SCALE,
        object_from_palms.flatten(1) / _POSITION_SCALE,
        _rotate(obj_inverse, candidates - base_obj[:, None]).flatten(1) / _POSITION_SCALE,
        _rotate(obj_inverse, candidates[:, 0] - candidates[:, 1]) / _POSITION_SCALE,
        bones.flatten(1) / _POSITION_SCALE,
        (proposals - bones).flatten(1) / _POSITION_SCALE,
        alphas,
        contact,
        weights,
        _rotate(obj_inverse, obj_imu[:, :3]) / _ACCELERATION_SCALE,
        _rotate(obj_inverse, forearm_acc_world - obj_imu[:, None, :3]).flatten(1) / _ACCELERATION_SCALE,
        matrix_to_rotation_6d(root_relative_rotation),
        matrix_to_rotation_6d(forearm_relative_rotation).flatten(1),
        _backward_difference(relative_geometry) * (_FRAME_RATE / _VELOCITY_SCALE),
        _backward_difference(bones).flatten(1) * (_FRAME_RATE / _VELOCITY_SCALE),
        _backward_difference(contact),
        rotation_change,
        _rotate(obj_inverse, raw_obj - base_obj) / _POSITION_SCALE,
    )
    features = torch.cat(blocks, dim=-1).clamp(-_FEATURE_CLIP, _FEATURE_CLIP)
    return {
        "features": features,
        "obj_rotation": obj_rotation,
        "contact_prob": contact,
        "base_obj": base_obj,
        "pred_palm": pred_palm,
    }


def _validate_capacity(feature_dim: int, max_correction: float) -> None:
    if not isinstance(feature_dim, int) or isinstance(feature_dim, bool) or feature_dim < 1:
        raise ValueError("feature_dim must be a positive integer")
    if not math.isfinite(max_correction) or max_correction < 0.0:
        raise ValueError("max_correction must be finite and nonnegative")


def _bound_correction(raw: torch.Tensor, max_correction: float) -> torch.Tensor:
    # Linear -> tanh, with an additional radial projection (not a per-axis cap).
    direction = torch.tanh(raw)
    return max_correction * (direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0))


def _check_features(features: torch.Tensor, feature_dim: int) -> None:
    if features.ndim != 3 or features.shape[-1] != feature_dim:
        raise ValueError(f"features must have shape [B,T,{feature_dim}], got {tuple(features.shape)}")


class CausalContactResidual(nn.Module):
    """Unidirectional GRU with a zero-initialized, L2-bounded local residual.

    State is the ordinary GRU tensor [num_layers,B,hidden_dim]. Carry it across
    ALL chunks of a sequence, and reset only at a real sequence boundary. No
    state is cached or detached implicitly (the trainer owns truncated BPTT).
    Exact chunk/whole parity applies in eval mode, or with dropout=0 in train.
    The default dtype is float32; explicit .to(dtype=...) is also supported.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        max_correction: float = 0.20,
        dropout: float = 0.0,
    ):
        super().__init__()
        _validate_capacity(feature_dim, max_correction)
        for name, value in (("hidden_dim", hidden_dim), ("num_layers", num_layers)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be finite and in [0,1)")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_correction = float(max_correction)
        self.gru = nn.GRU(
            feature_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=False, dropout=dropout if num_layers > 1 else 0.0,
            dtype=torch.float32,
        )
        self.head = nn.Linear(hidden_dim, 3, dtype=torch.float32)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _check_features(features, self.feature_dim)
        features = features.to(dtype=self.head.weight.dtype)
        if state is not None and tuple(state.shape) != (self.num_layers, features.shape[0], self.hidden_dim):
            raise ValueError("state must have shape [num_layers,B,hidden_dim]")
        if features.shape[1] == 0:
            if state is None:
                state = features.new_zeros(self.num_layers, features.shape[0], self.hidden_dim)
            return features.new_zeros(features.shape[0], 0, 3), state
        hidden, state = self.gru(features, state)
        return _bound_correction(self.head(hidden), self.max_correction), state


class PointwiseContactResidual(nn.Module):
    """Zero-init Linear -> tanh residual without learned recurrent memory.

    Uses the same L2 bound as the GRU and always returns None for state. Shared
    features still contain observed backward differences; mask columns 72:96
    for the additional no-explicit-differences ablation, if desired.
    """

    def __init__(self, feature_dim: int, max_correction: float = 0.20):
        super().__init__()
        _validate_capacity(feature_dim, max_correction)
        self.feature_dim = feature_dim
        self.max_correction = float(max_correction)
        self.head = nn.Linear(feature_dim, 3, dtype=torch.float32)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, features: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        _check_features(features, self.feature_dim)
        features = features.to(dtype=self.head.weight.dtype)
        return _bound_correction(self.head(features), self.max_correction), None


def apply_contact_residual(
    base_obj: torch.Tensor,
    obj_rotation: torch.Tensor,
    delta_local: torch.Tensor,
    gain: float = 1.0,
) -> torch.Tensor:
    """Return base_obj + gain * R_obj @ delta_local, with ordinary broadcasting.

    Accepts [T,3] or [B,T,3] geometry (rotations end in [3,3]). No smoothing,
    contact masking, integration, or label access. The head's L2 bound is before
    gain, so abs(gain)>1 increases it. Device/dtype follow the residual tensor;
    casts preserve gradients. The input tensors are never modified.
    """
    if base_obj.shape[-1:] != (3,) or delta_local.shape[-1:] != (3,):
        raise ValueError("base_obj and delta_local must end in three coordinates")
    if obj_rotation.shape[-2:] != (3, 3):
        raise ValueError("obj_rotation must end in [3,3]")
    if not math.isfinite(gain):
        raise ValueError("gain must be finite")
    delta_world = torch.matmul(obj_rotation.to(delta_local), delta_local.unsqueeze(-1)).squeeze(-1)
    return base_obj.to(delta_local) + float(gain) * delta_world
