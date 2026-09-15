"""Event-anchored grasp state for ObjectTrans (``object_trans_grasp_latent``).

The hand->object bone (object-frame vector from the palm to the object centre)
is treated as a *segment state* instead of a per-frame regression target:

* onset   : when a hand's contact probability rises through ``on_threshold`` the
            latent is (re-)anchored by fusing the observed geometry
            ``R_t^T (p_{t-1} - h_t)`` -- exact when the published object is right,
            drift-inherited when it is not -- with a state-free posture prior
            predicted from hands / IMUs / contact alone.  The mixing weight alpha
            is predicted from their disagreement and the IMU rest / contact-gap
            ages, so a state that contradicts the posture prior can be distrusted.
* inside  : ``g_t = g_{t-1} + delta_t`` with a learned increment (zero at init,
            so the module starts as a hard latch and learns the slow non-rigid
            re-settling of the grasp).  With ``modes`` a per-hand rigidity
            probability scales the increment towards zero (rigid grasp) or one
            (sliding contact).
* release : the latent is dropped when the contact probability falls through
            ``off_threshold``; a free hand's FK proposal follows the prior.

The published object position therefore enters the grasp geometry only at
contact onsets: a wrong object state cannot be amplified frame by frame, and
every onset is a fresh anchor.  The latent, the active flags and the ages are
published per frame (``STATE_KEYS``) so online windows continue the segment
instead of re-deriving it from the published trajectory.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RNNWithInit


class GraspLatent(nn.Module):
    """Per-hand grasp latent with onset anchoring, learned increments and release."""

    STATE_KEYS = ("grasp_latent", "grasp_active", "grasp_age")
    STATE_DIMS = {"grasp_latent": 6, "grasp_active": 2, "grasp_age": 2}
    FUSION_FEATURE_DIM = 15
    FEEDBACK_FEATURE_DIM = 17
    LENGTH_EPS = 1e-4

    def __init__(
        self,
        n_prediction_input,
        hidden_dim,
        n_layers,
        dropout,
        lb_quantile_dim,
        length_init,
        on_threshold=0.5,
        off_threshold=0.25,
        age_norm=60.0,
        increment_scale=0.1,
        fusion_hidden=32,
        fusion_bias=3.0,
        modes=False,
        increment_mode="integrate",
        residual_bound=0.10,
    ):
        super().__init__()
        # integrate : g_t = g_{t-1} + delta_t (a persistent integrator; on
        #             streams longer than the training window the increments
        #             random-walk: OMOMO FK HOI grew 8.1 -> 12.8 cm over a
        #             segment vs 7.6 -> 10.0 for per-frame regression).
        # residual  : the state is the onset ANCHOR, held constant; the bone
        #             is anchor + bound * tanh(Linear(hidden)) -- a bounded,
        #             non-integrated deviation, so the long-horizon drift of
        #             the grasp geometry is capped by construction.
        if increment_mode not in {"integrate", "residual"}:
            raise ValueError("increment_mode must be integrate or residual")
        self.increment_mode = str(increment_mode)
        self.residual_bound = float(residual_bound)
        if not 0.0 <= float(off_threshold) <= float(on_threshold) <= 1.0:
            raise ValueError("grasp thresholds must satisfy 0 <= off <= on <= 1")
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.age_norm = float(age_norm)
        self.increment_scale = float(increment_scale)
        self.modes_enabled = bool(modes)
        # State-free posture prior: the predictor's inputs minus every
        # object-state-derived quantity (init geometry, feedback), so it stays
        # valid when the published object has drifted.
        self.prior_head = RNNWithInit(
            n_input=n_prediction_input,
            n_output=8,  # left dir(3)+len(1), right dir(3)+len(1)
            n_hidden=hidden_dim,
            n_init=3 + lb_quantile_dim,  # initial contact state + l_b prior quantiles
            n_rnn_layer=n_layers,
            bidirectional=False,
            dropout=dropout,
        )
        length_bias = math.log(math.expm1(max(float(length_init), 2.0 * self.LENGTH_EPS) - self.LENGTH_EPS))
        with torch.no_grad():
            self.prior_head.linear2.bias[3] = length_bias
            self.prior_head.linear2.bias[7] = length_bias
        # Onset fusion: alpha = sigmoid(MLP(z)) is the weight of the observed
        # geometry; the bias starts it near 1 (trust the published object).
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.FUSION_FEATURE_DIM),
            nn.Linear(self.FUSION_FEATURE_DIM, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, 1),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.constant_(self.fusion[-1].bias, float(fusion_bias))
        # In-segment increment (object frame, metres per frame) from the
        # predictor's top-layer hidden state; zero at init = hard latch.
        self.increment = nn.Linear(hidden_dim, 6)
        nn.init.zeros_(self.increment.weight)
        nn.init.zeros_(self.increment.bias)
        if self.modes_enabled:
            self.mode_head = nn.Linear(hidden_dim, 2)  # per-hand rigid-grasp logit
            nn.init.zeros_(self.mode_head.weight)
            nn.init.zeros_(self.mode_head.bias)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _unit(x, eps=1e-8):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def split_latent(latent):
        """[B,2,3] -> (directions [B,2,3], lengths [B,2])."""
        lengths = latent.norm(dim=-1)
        return latent / lengths.clamp_min(1e-8).unsqueeze(-1), lengths

    def prior(self, prior_input, prior_init, obj_rotm, hands):
        """State-free prior bone [B,T,2,3] (object frame) and its world position [B,T,2,3]."""
        out, _ = self.prior_head.forward_with_state(prior_input, x_init=prior_init)
        out = out.to(obj_rotm.dtype)
        directions = torch.stack((self._unit(out[..., 0:3]), self._unit(out[..., 4:7])), dim=2)
        lengths = torch.stack((F.softplus(out[..., 3]), F.softplus(out[..., 7])), dim=2) + self.LENGTH_EPS
        bone = directions * lengths.unsqueeze(-1)
        position = hands + torch.einsum("btij,bthj->bthi", obj_rotm, bone)
        return bone, position

    def causal_ages(self, rest_flag, contact_prob):
        """IMU-rest run length and frames since any hand contact, causal within the window ([B,T] each)."""
        batch_size, seq_len = rest_flag.shape
        dtype = contact_prob.dtype
        contact_any = contact_prob[..., :2].amax(dim=-1) > self.on_threshold
        rest_age = torch.zeros(batch_size, seq_len, dtype=dtype, device=contact_prob.device)
        gap_age = torch.zeros_like(rest_age)
        rest_run = torch.zeros(batch_size, dtype=dtype, device=contact_prob.device)
        gap_run = torch.zeros_like(rest_run)
        for t in range(seq_len):
            rest_run = torch.where(rest_flag[:, t], rest_run + 1.0, torch.zeros_like(rest_run))
            gap_run = torch.where(contact_any[:, t], torch.zeros_like(gap_run), gap_run + 1.0)
            rest_age[:, t] = rest_run
            gap_age[:, t] = gap_run
        return rest_age, gap_age

    def fusion_features(self, observed, prior, rest_flag, rest_age, gap_age, contact_prob, other_active):
        """[B,2,15] onset consistency features per hand.

        [observed(3), prior(3), observed-prior(3), |observed-prior|, rest_flag,
         rest_age/norm, gap_age/norm, contact_prob(hand), other hand active]
        """
        dtype = observed.dtype
        diff = observed - prior

        def per_hand(value):
            return value.to(dtype).view(-1, 1, 1).expand(-1, 2, 1)

        return torch.cat(
            (
                observed,
                prior,
                diff,
                diff.norm(dim=-1, keepdim=True),
                per_hand(rest_flag),
                per_hand((rest_age / self.age_norm).clamp(max=2.0)),
                per_hand((gap_age / self.age_norm).clamp(max=2.0)),
                contact_prob[..., :2].to(dtype).unsqueeze(-1),
                other_active.to(dtype).unsqueeze(-1),
            ),
            dim=-1,
        )

    def anchor(self, observed, prior, rest_flag, rest_age, gap_age, contact_prob, other_active):
        """Fused onset bone [B,2,3] and the observed-geometry weight alpha [B,2]."""
        z = self.fusion_features(observed, prior, rest_flag, rest_age, gap_age, contact_prob, other_active)
        alpha = torch.sigmoid(self.fusion(z).to(observed.dtype))
        return alpha * observed + (1.0 - alpha) * prior, alpha.squeeze(-1)

    # ------------------------------------------------------------------ state
    def initial_state(self, observed0, prior0, rest_flag0, rest_age0, gap_age0, contact_prob0):
        """State before frame 0 of a window without published history.

        Hands whose contact probability is already above ``on_threshold`` are
        anchored from the window-init geometry (fused with the prior); the
        others carry the prior.  Returns latent [B,2,3], active [B,2] (bool),
        age [B,2] and the onset alpha [B,2] (zero for inactive hands).
        """
        active = contact_prob0[..., :2] > self.on_threshold
        fused, alpha = self.anchor(observed0, prior0, rest_flag0, rest_age0, gap_age0, contact_prob0, active.flip(-1))
        latent = torch.where(active.unsqueeze(-1), fused, prior0)
        # Ages count frames *processed* inside the segment: 0 before frame 0.
        age = torch.zeros_like(latent[..., 0])
        return latent, active, age, alpha * active.to(alpha.dtype)

    def step(self, hidden, latent_prev, active_prev, age_prev, contact_prob, observed, prior, rest_flag, rest_age, gap_age):
        """One causal update of the per-hand latent.

        hidden      [B,H]   predictor top-layer hidden state after this frame
        latent_prev [B,2,3] active_prev [B,2] bool   age_prev [B,2]
        contact_prob[B,3]   observed/prior [B,2,3] object-frame bones at this frame
        """
        probs = contact_prob[..., :2]
        active = torch.where(active_prev, probs > self.off_threshold, probs > self.on_threshold)
        onset = active & ~active_prev
        fused, alpha = self.anchor(observed, prior, rest_flag, rest_age, gap_age, contact_prob, active_prev.flip(-1))
        raw = self.increment(hidden).to(latent_prev.dtype).view(-1, 2, 3)
        rigid = rigid_logit = None
        if self.modes_enabled:
            rigid_logit = self.mode_head(hidden).to(latent_prev.dtype)
            rigid = torch.sigmoid(rigid_logit)
        if self.increment_mode == "residual":
            # norm-bounded deviation: |delta| = bound * tanh(|raw|) <= bound
            norm = raw.norm(dim=-1, keepdim=True)
            delta = raw * (self.residual_bound * torch.tanh(norm) / norm.clamp_min(1e-6))
            if rigid is not None:
                delta = delta * (1.0 - rigid).unsqueeze(-1)
            # state = anchor (constant inside the segment); bone = anchor + bounded residual
            latent = torch.where(onset.unsqueeze(-1), fused, torch.where(active.unsqueeze(-1), latent_prev, prior))
            bone = torch.where(active.unsqueeze(-1), latent + delta, prior)
        else:
            delta = raw * self.increment_scale
            if rigid is not None:
                delta = delta * (1.0 - rigid).unsqueeze(-1)
            continued = latent_prev + delta
            latent = torch.where(onset.unsqueeze(-1), fused, torch.where(active.unsqueeze(-1), continued, prior))
            bone = latent
        age = torch.where(
            active,
            torch.where(onset, torch.ones_like(age_prev), age_prev + 1.0),
            torch.zeros_like(age_prev),
        )
        return {
            "latent": latent,
            "bone": bone,
            "active": active,
            "age": age,
            "alpha": alpha * onset.to(alpha.dtype),
            "onset": onset,
            "delta": delta,
            "rigid": rigid,
            "rigid_logit": rigid_logit,
        }

    def feedback_features(self, latent, active, age, prev_delta_obj, prev_gate):
        """[B,17] predictor feedback built from the latent instead of the published object."""
        dtype = latent.dtype
        return torch.cat(
            (
                latent.flatten(1),
                active.to(dtype),
                (age / self.age_norm).clamp(max=2.0).to(dtype),
                prev_delta_obj.to(dtype),
                prev_gate.to(dtype),
            ),
            dim=-1,
        )
