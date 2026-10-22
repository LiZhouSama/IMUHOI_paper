"""VC-refine: correct VelocityContact's contacts with the first-pass OT/HP geometry.

VelocityContact decides contact from IMU and pose evidence alone.  After a
first ObjectTrans pass the window also holds what that decision *implies*:
where the object would be if hand s held it (the state-free FK candidate), how
fast that candidate moves against the object IMU, how far the hand is from the
fused object state, whether the two hands' candidates agree.  VC-refine reads
those consistency signals over the whole online window (a bidirectional GRU;
the window ends at the 10-frame online lookahead) and adds residual logits to
VC's factorized contact logits:

    z_move' = z_move + d_move,   z_s|move' = z_s|move + d_s,
    p_move' = sigmoid(z_move'),  p_s' = p_move' * sigmoid(z_s|move').

The output head is zero-initialised, so an untrained VC-refine reproduces VC
exactly.  Features split into state-free ones (FK candidate motion, palm
motion, object IMU, VC outputs) and state-dependent ones (hand -> fused object
geometry, candidate innovation), the latter read the first-pass fused
trajectory, i.e. the published history online.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix

from configs import FRAME_RATE

# Output channel order follows VelocityContact: (left | move, right | move, move).
CONTACT_LOGIT_CLAMP = 12.0
POSITION_CLAMP = 2.0
DISTANCE_CLAMP = 3.0


def _backward_rate(values: torch.Tensor) -> torch.Tensor:
    """Per-frame backward difference times the frame rate; frame 0 copies frame 1."""
    if values.shape[1] < 2:
        return torch.zeros_like(values)
    step = (values[:, 1:] - values[:, :-1]) * FRAME_RATE
    return torch.cat((step[:, :1], step), dim=1)


def _object_frame(obj_rotm: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """World vectors [B,T,3] -> object frame."""
    return torch.einsum("btji,btj->bti", obj_rotm, vectors)


def _vector_and_norm(vector: torch.Tensor):
    return vector.clamp(-POSITION_CLAMP, POSITION_CLAMP), vector.norm(dim=-1, keepdim=True).clamp(max=DISTANCE_CLAMP)


def _speed(value: torch.Tensor) -> torch.Tensor:
    """log1p-compressed speed [B,T,1] of a [B,T,3] velocity (one-frame jumps reach metres per second)."""
    return torch.log1p(value.norm(dim=-1, keepdim=True))


def build_vc_refine_features(
    vc_logits: torch.Tensor,
    obj_vel: torch.Tensor,
    obj_imu: torch.Tensor,
    hands: torch.Tensor,
    fk_candidates: torch.Tensor,
    fused: torch.Tensor,
    state_init: torch.Tensor,
    bimanual_candidate: torch.Tensor = None,
    static_gate: torch.Tensor = None,
) -> torch.Tensor:
    """Per-frame consistency features [B,T,FEATURE_DIM] (see VCRefineModule.FEATURE_DIM).

    ``fused`` is the first-pass object trajectory of the window and
    ``state_init`` the object state before its first frame; the state a frame's
    decision is judged against is the fused object of the *previous* frame.
    """
    batch_size, seq_len = vc_logits.shape[:2]
    dtype = torch.float32
    vc_logits = vc_logits.to(dtype)
    obj_vel = obj_vel.to(dtype)
    hands = hands.to(dtype).reshape(batch_size, seq_len, 2, 3)
    fk_candidates = fk_candidates.to(dtype).reshape(batch_size, seq_len, 2, 3)
    fused = fused.to(dtype)
    obj_rotm = rotation_6d_to_matrix(obj_imu[..., 3:9].to(dtype).reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
    state_prev = torch.cat((state_init.to(dtype).unsqueeze(1), fused[:, :-1]), dim=1)

    logits = vc_logits.clamp(-CONTACT_LOGIT_CLAMP, CONTACT_LOGIT_CLAMP)
    p_move = torch.sigmoid(logits[..., 2:3])
    probs = torch.cat((p_move * torch.sigmoid(logits[..., :2]), p_move), dim=-1)

    rel_rot = obj_rotm[:, :-1].transpose(-1, -2) @ obj_rotm[:, 1:]
    cos = ((rel_rot.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rot_rate = torch.acos(cos) * FRAME_RATE
    rot_rate = torch.cat((rot_rate[:, :1], rot_rate), dim=1).unsqueeze(-1) if seq_len > 1 else torch.zeros(
        batch_size, seq_len, 1, device=obj_vel.device, dtype=dtype
    )
    gate = (
        static_gate.to(dtype).reshape(batch_size, seq_len, 1)
        if isinstance(static_gate, torch.Tensor)
        else torch.zeros(batch_size, seq_len, 1, device=obj_vel.device, dtype=dtype)
    )
    obj_vel_local, _ = _vector_and_norm(_object_frame(obj_rotm, obj_vel))
    parts = [
        logits / CONTACT_LOGIT_CLAMP,
        probs,
        obj_vel_local,
        _speed(obj_vel),
        torch.log1p(obj_imu[..., :3].to(dtype).norm(dim=-1, keepdim=True)),
        torch.log1p(rot_rate),
        gate,
    ]
    for hand in range(2):
        palm = hands[:, :, hand]
        candidate = fk_candidates[:, :, hand]
        observed = _object_frame(obj_rotm, state_prev - palm)
        bone = _object_frame(obj_rotm, candidate - palm)
        innovation = _object_frame(obj_rotm, candidate - state_prev)
        candidate_vel = _backward_rate(candidate)
        palm_vel = _backward_rate(palm)
        for vector in (observed, bone, innovation):
            clipped, norm = _vector_and_norm(vector)
            parts += [clipped, norm]
        parts += [
            _speed(candidate_vel - obj_vel),
            _speed(candidate_vel),
            _speed(palm_vel - obj_vel),
            _speed(palm_vel),
        ]
    closure, closure_norm = _vector_and_norm(_object_frame(obj_rotm, fk_candidates[:, :, 0] - fk_candidates[:, :, 1]))
    span = (hands[:, :, 0] - hands[:, :, 1]).norm(dim=-1, keepdim=True).clamp(max=DISTANCE_CLAMP)
    if isinstance(bimanual_candidate, torch.Tensor):
        pair, pair_norm = _vector_and_norm(_object_frame(obj_rotm, bimanual_candidate.to(dtype) - state_prev))
    else:
        pair = torch.zeros_like(closure)
        pair_norm = torch.zeros_like(closure_norm)
    fused_vel = (fused - state_prev) * FRAME_RATE
    parts += [closure, closure_norm, span, pair, pair_norm, _speed(fused_vel), _speed(fused_vel - obj_vel)]
    return torch.cat(parts, dim=-1)


class VCRefineModule(nn.Module):
    """Residual contact-logit correction from window-level OT/HP consistency."""

    FEATURE_DIM = 56
    ARCH_VERSION = 1

    def __init__(self, cfg):
        super().__init__()
        hidden = int(getattr(cfg, "cascade_vcr_hidden_dim", 128))
        layers = int(getattr(cfg, "cascade_vcr_num_layers", 2))
        dropout = float(getattr(cfg, "cascade_vcr_dropout", 0.2))
        self.input_norm = nn.LayerNorm(self.FEATURE_DIM)
        self.encoder = nn.Sequential(nn.Linear(self.FEATURE_DIM, hidden), nn.GELU(), nn.Dropout(dropout))
        self.temporal = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("_cascade_vcr_arch_version", torch.tensor(self.ARCH_VERSION, dtype=torch.int64))

    def forward(
        self,
        vc_logits: torch.Tensor,
        obj_vel: torch.Tensor,
        obj_imu: torch.Tensor,
        hands: torch.Tensor,
        fk_candidates: torch.Tensor,
        fused: torch.Tensor,
        state_init: torch.Tensor,
        bimanual_candidate: torch.Tensor = None,
        static_gate: torch.Tensor = None,
    ):
        """Refined VelocityContact contact outputs (same keys/shapes as VC)."""
        # The features and the recurrent core run in FP32: PyTorch 1.13 has no
        # BF16 CUDA GRU kernel, and the inputs come from frozen stages anyway.
        with torch.autocast(device_type=vc_logits.device.type, enabled=False):
            features = build_vc_refine_features(
                vc_logits.detach(),
                obj_vel.detach(),
                obj_imu.detach(),
                hands.detach(),
                fk_candidates.detach(),
                fused.detach(),
                state_init.detach(),
                bimanual_candidate=None if bimanual_candidate is None else bimanual_candidate.detach(),
                static_gate=None if static_gate is None else static_gate.detach(),
            )
            encoded = self.encoder(self.input_norm(features))
            hidden, _ = self.temporal(encoded)
            delta = self.head(hidden)
            logits = vc_logits.float() + delta
            cond_logits, move_logits = logits[..., :2], logits[..., 2:3]
            move_prob = torch.sigmoid(move_logits)
            cond_prob = torch.sigmoid(cond_logits)
            marginal = move_prob * cond_prob
        dtype = vc_logits.dtype
        return {
            "pred_hand_contact_logits": logits.to(dtype),
            "pred_hand_contact_prob": torch.cat((marginal, move_prob), dim=-1).to(dtype),
            "pred_obj_move_prob": move_prob.to(dtype),
            "pred_hand_contact_prob_cond": cond_prob.to(dtype),
            "pred_hand_contact_logits_cond": cond_logits.to(dtype),
            "pred_hand_contact_prob_marginal": marginal.to(dtype),
            "cascade_vcr_delta_logits": delta.to(dtype),
        }

    def validate_checkpoint_state_dict(self, state_dict):
        version = state_dict.get("_cascade_vcr_arch_version") if isinstance(state_dict, dict) else None
        if not isinstance(version, torch.Tensor) or int(version.item()) != self.ARCH_VERSION:
            raise ValueError("Checkpoint is not a cascade VC-refine v1 state dict")


# Keys VC-refine replaces in the VelocityContact output.
REFINED_CONTACT_KEYS = (
    "pred_hand_contact_logits",
    "pred_hand_contact_prob",
    "pred_obj_move_prob",
    "pred_hand_contact_prob_cond",
    "pred_hand_contact_logits_cond",
    "pred_hand_contact_prob_marginal",
)

__all__ = ["VCRefineModule", "build_vc_refine_features", "REFINED_CONTACT_KEYS"]
