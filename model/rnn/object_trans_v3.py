"""ObjectTrans v3: the v2 fusion with observable grasp geometry and a learned soft latch.

What the OMOMO error decomposition (``outputs/.../ot_v3_diagnosis.md``) says about
the object branch, with the hands the pipeline actually produces:

* the object error is dominated by the hand error inside contact segments
  (7 of 14 cm); the per-segment bone drift the first v3 modelled is worth
  about 1 cm and only shows with ground-truth hands;
* a one-shot anchor latched at contact onset is worse than v2's per-frame
  regression (onset bone error 10.0 vs 8.8 cm), because the latch keeps every
  onset mistake for the whole segment and cannot follow a re-grasp that the
  contact probability never reports;
* 38% of the error growth happens on frames where the object is not moving:
  false contacts drag it and the soft static weight lets the IMU branch leak.

This module therefore keeps v2's per-frame bone regression and adds three
things that the evidence supports:

1. ``R_obj^T R_forearm`` for both hands (fully observable from the IMUs) as
   head input, plus the swing of that quantity since the current contact
   started.  The re-orientation of the forearm against the object is what
   predicts how the palm-to-object bone changes.
2. A learned soft latch.  The head proposes a bone every frame and a gate
   ``alpha`` in ``[gate_floor, 1]``; the effective bone is an exponential
   moving average ``bone = bone + alpha * (proposal - bone)`` that is reset to
   the proposal at every contact onset.  ``alpha = 1`` recovers v2 exactly, a
   small ``alpha`` is the segment latch, and the head decides per frame (for
   example opening the gate when a re-grasp changes the grasp geometry).
3. The object-static hard gate of the base fusion
   (``object_trans_static_gate``), which pins the fused position to the
   previous frame whenever the object IMU or the VC motion head says the
   object is not moving.

Online sliding-window inference hands the published latch state back to each
window (``SEGMENT_STATE_OUTPUT_KEYS``) so a window that starts inside a
segment continues it instead of re-deriving it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from configs import FRAME_RATE, _SENSOR_NAMES
from utils.object_fusion import build_contact_fusion_weights
from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix

from .base import RNNWithInit
from .object_trans_v2 import V2ObjectTransModule

_ROOT_SENSOR_INDEX = _SENSOR_NAMES.index("Root")
_FOREARM_SENSOR_INDICES = (
    _SENSOR_NAMES.index("LeftForeArm"),
    _SENSOR_NAMES.index("RightForeArm"),
)
_IDENTITY_ROTATION_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class V3ObjectTransModule(V2ObjectTransModule):
    """v2 fusion + forearm/object geometry + learned soft latch."""

    ARCH_VERSION = 3
    # Per hand: bone direction (3), bone length (1), latch gate logit (1).
    HAND_OUTPUT_DIM = 5
    # Two hands x 6D for R_obj^T R_forearm.
    REL_ROT_FEATURE_DIM = 12
    # Published per-frame outputs that fully describe the latch state after
    # that frame.  Online inference hands them back as a known prefix so a
    # sliding window continues the segment instead of re-deriving it.
    SEGMENT_STATE_OUTPUT_KEYS = (
        "pred_lhand_obj_vector",
        "pred_rhand_obj_vector",
        "segment_contact_active",
        "segment_rel_rot6d",
    )

    def __init__(self, cfg):
        super().__init__(cfg)

        n_prediction_input = self.n_prediction_input + self.REL_ROT_FEATURE_DIM
        predictor = self.object_trans_head
        self.object_trans_head = RNNWithInit(
            n_input=n_prediction_input,
            n_output=2 * self.HAND_OUTPUT_DIM,
            n_hidden=predictor.n_hidden,
            n_init=int(predictor.init_net[0].in_features),
            n_rnn_layer=predictor.n_rnn_layer,
            bidirectional=False,
            dropout=float(getattr(cfg, "object_trans_dropout", 0.2)),
        )
        length_bias = math.log(
            math.expm1(self.object_trans_length_init - self.LENGTH_EPS)
        )
        # Gate logit bias: alpha starts close to v2 (fast tracking) and the
        # head learns where a slower latch pays off.
        self.gate_init_bias = float(getattr(cfg, "object_trans_v3_gate_init_bias", 1.0))
        with torch.no_grad():
            for hand_index in range(2):
                base = hand_index * self.HAND_OUTPUT_DIM
                self.object_trans_head.linear2.bias[base + 3] = length_bias
                self.object_trans_head.linear2.bias[base + 4] = self.gate_init_bias
            self.object_trans_arch_version.fill_(self.ARCH_VERSION)

        # Both per-step injections project into the prediction input, which just
        # grew by the relative-rotation block, so they have to be rebuilt.
        feedback_hidden_dim = int(getattr(cfg, "object_trans_feedback_hidden_dim", 64))
        self.feedback_embed = self._build_step_injection(
            self.feedback_feature_dim, feedback_hidden_dim, n_prediction_input
        )
        self.onset_embed = self._build_step_injection(
            self.REL_ROT_FEATURE_DIM, feedback_hidden_dim, n_prediction_input
        )
        self.n_prediction_input = n_prediction_input

        # Lower bound of the latch gate: the effective bone always tracks the
        # proposal with at least this rate, so no onset mistake is permanent.
        self.gate_floor = min(
            max(float(getattr(cfg, "object_trans_v3_gate_floor", 0.05)), 0.0), 1.0
        )
        # Online sliding-window inference: carry the published latch state
        # into each new window instead of re-initialising it from the
        # published trajectory.
        self.online_carry_segment_state = bool(
            getattr(cfg, "object_trans_v3_online_carry_segment_state", True)
        )
        self.contact_on_threshold = float(
            getattr(
                cfg,
                "object_trans_v3_contact_on_threshold",
                getattr(cfg, "hand_contact_threshold", 0.15),
            )
        )
        self.contact_off_threshold = float(
            getattr(
                cfg,
                "object_trans_v3_contact_off_threshold",
                max(self.contact_on_threshold * 0.5, 0.05),
            )
        )
        self._known_segment_state_prefix = None

    @staticmethod
    def _build_step_injection(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        """A per-step feature injection that starts as an exact no-op."""
        block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(block[-1].weight)
        nn.init.zeros_(block[-1].bias)
        return block

    # ------------------------------------------------------------------
    # Observable forearm <-> object geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _rotation_6d_to_matrix_safe(rotation_6d: torch.Tensor) -> torch.Tensor:
        """6D -> matrix, falling back to identity for absent/degenerate rows.

        A missing human IMU arrives as zeros, which would otherwise normalise
        into a zero matrix and silently corrupt the relative rotation.
        """
        flat = rotation_6d.reshape(-1, 6)
        usable = (
            torch.isfinite(flat).all(dim=-1)
            & (flat[:, 0:3].norm(dim=-1) > 1e-6)
            & (flat[:, 3:6].norm(dim=-1) > 1e-6)
        )
        identity = flat.new_tensor(_IDENTITY_ROTATION_6D).expand_as(flat)
        safe = torch.where(usable.unsqueeze(-1), flat, identity)
        return rotation_6d_to_matrix(safe).reshape(*rotation_6d.shape[:-1], 3, 3)

    def _forearm_object_rotations(
        self, human_imu: torch.Tensor, obj_rotm: torch.Tensor
    ) -> torch.Tensor:
        """``R_obj^T @ R_forearm`` for both hands, straight from the IMUs.

        Non-root sensors store root-relative orientation, so the forearm is
        recovered as ``R_root @ (R_root^T R_forearm)`` before being expressed in
        the object frame.
        """
        batch_size, seq_len = obj_rotm.shape[:2]
        if (
            human_imu.dim() != 4
            or human_imu.shape[2] <= max(_FOREARM_SENSOR_INDICES)
            or human_imu.shape[3] < 9
        ):
            return (
                torch.eye(3, device=obj_rotm.device, dtype=obj_rotm.dtype)
                .expand(batch_size, seq_len, 2, 3, 3)
                .clone()
            )
        root_rotm = self._rotation_6d_to_matrix_safe(
            human_imu[:, :, _ROOT_SENSOR_INDEX, 3:9]
        )
        obj_rotm_transposed = obj_rotm.transpose(-1, -2)
        relative = []
        for sensor_index in _FOREARM_SENSOR_INDICES:
            forearm_root = self._rotation_6d_to_matrix_safe(
                human_imu[:, :, sensor_index, 3:9]
            )
            relative.append(obj_rotm_transposed @ (root_rotm @ forearm_root))
        return torch.stack(relative, dim=2)

    def _prepare_prediction_context(
        self,
        hand_positions,
        pred_hand_contact_prob,
        obj_trans_init,
        *,
        human_imu=None,
        **kwargs,
    ):
        context = super()._prepare_prediction_context(
            hand_positions,
            pred_hand_contact_prob,
            obj_trans_init,
            human_imu=human_imu,
            **kwargs,
        )
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        prepared_human_imu = self._prepare_human_imu(
            human_imu, batch_size, seq_len, context["device"], context["dtype"]
        )
        relative = self._forearm_object_rotations(
            prepared_human_imu, context["obj_rotm"]
        )
        context["forearm_obj_rotm"] = relative
        context["prediction_input"] = torch.cat(
            (
                context["prediction_input"],
                matrix_to_rotation_6d(relative).reshape(
                    batch_size, seq_len, self.REL_ROT_FEATURE_DIM
                ),
            ),
            dim=-1,
        )
        return context

    # ------------------------------------------------------------------
    # Segment / latch state
    # ------------------------------------------------------------------
    def _init_scan_state(self, context):
        state = super()._init_scan_state(context)
        initial_bone = torch.stack(
            (
                context["l_oe0"] * context["l_lb0"],
                context["r_oe0"] * context["r_lb0"],
            ),
            dim=1,
        )
        state.update(
            {
                "bone": initial_bone,
                "segment_rel_rotm": context["forearm_obj_rotm"][:, 0],
                "hand_contact_active": context["pred_hand_contact_prob"][:, 0, :2]
                >= self.contact_on_threshold,
            }
        )
        return state

    def _decode_bone_proposal(self, prediction_out):
        """Per-hand bone proposal [B,2,3] and latch gate alpha [B,2]."""
        hands = prediction_out.reshape(-1, 2, self.HAND_OUTPUT_DIM)
        proposal = self._unit_vector(hands[:, :, 0:3]) * self._softplus_positive(
            hands[:, :, 3]
        ).unsqueeze(-1)
        gate = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(hands[:, :, 4])
        return proposal, gate

    def _decode_prediction_output(self, prediction_out, contact_prob=None):
        raise NotImplementedError(
            "V3ObjectTransModule decodes bone proposals and latch gates inside "
            "_predict_step; _decode_prediction_output belongs to the per-frame "
            "FK head used by v1/v2."
        )

    # ------------------------------------------------------------------
    # Published latch state as a known prefix (online sliding window)
    # ------------------------------------------------------------------
    def forward(self, *args, known_segment_state_prefix=None, **kwargs):
        """v2 forward plus an optional published latch-state prefix."""
        self._known_segment_state_prefix = (
            known_segment_state_prefix if self.online_carry_segment_state else None
        )
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._known_segment_state_prefix = None

    def _prepare_known_segment_state(self, context):
        """Per-frame latch overrides ([B,T,...]) from a published prefix, or None."""
        prefix = getattr(self, "_known_segment_state_prefix", None)
        if not isinstance(prefix, dict):
            return None
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        dims = {
            "pred_lhand_obj_vector": 3,
            "pred_rhand_obj_vector": 3,
            "segment_contact_active": 2,
            "segment_rel_rot6d": 12,
        }
        known = {}
        valid = None
        for key, dim in dims.items():
            values, mask = self._prepare_known_sequence(
                prefix.get(key), batch_size, seq_len, dim, device, dtype, None
            )
            known[key] = values
            valid = mask if valid is None else (valid & mask)
        if valid is None or not bool(valid.any().item()):
            return None
        known["mask"] = valid
        return known

    def _apply_known_segment_state(self, next_state, known, time_index):
        """Replace the latch state after ``time_index`` with the published one."""
        mask = known["mask"][:, time_index]
        if not bool(mask.any().item()):
            return next_state
        bone = torch.stack(
            (
                known["pred_lhand_obj_vector"][:, time_index],
                known["pred_rhand_obj_vector"][:, time_index],
            ),
            dim=1,
        )
        rel_rotm = rotation_6d_to_matrix(
            known["segment_rel_rot6d"][:, time_index].reshape(-1, 6)
        ).reshape(-1, 2, 3, 3)
        active = known["segment_contact_active"][:, time_index] > 0.5
        out = dict(next_state)
        out["bone"] = torch.where(mask.view(-1, 1, 1), bone, next_state["bone"])
        out["segment_rel_rotm"] = torch.where(
            mask.view(-1, 1, 1, 1), rel_rotm, next_state["segment_rel_rotm"]
        )
        out["hand_contact_active"] = torch.where(
            mask.view(-1, 1), active, next_state["hand_contact_active"]
        )
        return out

    def _scan_predictor(
        self,
        context,
        known_obj_trans_prefix=None,
        known_obj_trans_prefix_mask=None,
        known_gating_weights_prefix=None,
    ):
        """Always scan step by step.

        The base class runs the head over the whole sequence at once whenever
        state feedback is off, which is only sound while the head's input has no
        recurrent dependence.  v3 injects the swing measured against the
        latched segment reference, so that dependence always exists.
        """
        state = self._init_scan_state(context)
        known_pos, known_mask = self._prepare_known_sequence(
            known_obj_trans_prefix,
            context["batch_size"],
            context["seq_len"],
            3,
            context["device"],
            context["dtype"],
            known_obj_trans_prefix_mask,
        )
        known_weights, known_weights_mask = self._prepare_known_sequence(
            known_gating_weights_prefix,
            context["batch_size"],
            context["seq_len"],
            4,
            context["device"],
            context["dtype"],
            None,
        )
        known_segment = self._prepare_known_segment_state(context)
        collected = {}
        for time_index in range(context["seq_len"]):
            frame_output, state = self._predict_step(
                context,
                time_index,
                state,
                known_obj_trans=known_pos[:, time_index],
                known_mask=known_mask[:, time_index],
                known_weights=known_weights[:, time_index],
                known_weights_mask=known_weights_mask[:, time_index],
            )
            if known_segment is not None:
                state = self._apply_known_segment_state(state, known_segment, time_index)
            for key, value in frame_output.items():
                collected.setdefault(key, []).append(value)
        return {
            key: torch.stack(values, dim=1) for key, values in collected.items()
        }, state

    # ------------------------------------------------------------------
    # One frame
    # ------------------------------------------------------------------
    def _predict_step(
        self,
        context,
        time_index,
        state,
        known_obj_trans=None,
        known_mask=None,
        known_weights=None,
        known_weights_mask=None,
        prediction_out=None,
        next_rnn_state=None,
    ):
        step_state = state
        feedback_mode = context.get("feedback_mode", "normal")
        if feedback_mode == "gt_prev":
            step_state = dict(state)
            step_state["prev_obj_trans"] = context["feedback_prev_obj_trans"][
                :, time_index
            ]
            step_state["prev_prev_obj_trans"] = context[
                "feedback_prev_prev_obj_trans"
            ][:, time_index]
            step_state["prev_obj_delta"] = context["feedback_prev_obj_delta"][
                :, time_index
            ]
        elif feedback_mode == "perturb":
            injection_mask = context["feedback_injection_mask"][:, time_index]
            if bool(injection_mask.any().item()):
                step_state = dict(state)
                step_state["prev_obj_trans"] = torch.where(
                    injection_mask.unsqueeze(-1),
                    state["prev_obj_trans"]
                    + context["feedback_perturbation"][:, time_index],
                    state["prev_obj_trans"],
                )

        batch_size = context["batch_size"]
        obj_rotm = context["obj_rotm"][:, time_index]
        hand_positions = context["hand_positions"][:, time_index]
        contact_prob = context["pred_hand_contact_prob"][:, time_index]
        relative = context["forearm_obj_rotm"][:, time_index]

        step_input = context["prediction_input"][:, time_index]
        # How far the forearm has re-oriented against the object since the
        # current contact started: the observable that predicts bone drift.
        observed_swing = relative @ step_state["segment_rel_rotm"].transpose(-1, -2)
        step_input = step_input + self.onset_embed(
            matrix_to_rotation_6d(observed_swing).reshape(
                batch_size, self.REL_ROT_FEATURE_DIM
            )
        )
        if self._state_feedback_enabled():
            step_input = step_input + self.feedback_embed(
                self._feedback_features(
                    obj_rotm,
                    hand_positions[:, 0],
                    hand_positions[:, 1],
                    step_state,
                )
            )

        if prediction_out is None:
            prediction_out, rnn_state = self.object_trans_head.step(
                step_input, state["rnn_state"]
            )
        else:
            rnn_state = (
                state["rnn_state"] if next_rnn_state is None else next_rnn_state
            )
        proposal, gate = self._decode_bone_proposal(prediction_out)

        previous_active = step_state["hand_contact_active"]
        contact_active = torch.where(
            previous_active,
            contact_prob[:, :2] >= self.contact_off_threshold,
            contact_prob[:, :2] >= self.contact_on_threshold,
        )
        contact_onset = contact_active & ~previous_active
        segment_rel_rotm = torch.where(
            contact_onset.view(batch_size, 2, 1, 1), relative, step_state["segment_rel_rotm"]
        )
        # Soft latch: a contact onset resets the bone to the proposal, then the
        # head's gate decides how quickly the bone follows new proposals.
        alpha = torch.where(contact_onset, torch.ones_like(gate), gate).unsqueeze(-1)
        previous_bone = step_state["bone"]
        bone = previous_bone + alpha * (proposal - previous_bone)

        vectors_world = torch.einsum("bij,bhj->bhi", obj_rotm, bone)
        fk_positions = hand_positions + vectors_world
        l_pos_fk = fk_positions[:, 0]
        r_pos_fk = fk_positions[:, 1]

        weights = build_contact_fusion_weights(contact_prob)
        weights = self._apply_static_gate(weights, context, time_index)
        previous_pos = step_state["prev_obj_trans"]
        imu_integrated_pos = (
            previous_pos + context["obj_vel_input"][:, time_index] / FRAME_RATE
        )
        static_pos = previous_pos
        fused_model_pos = (
            weights[:, 0:1] * l_pos_fk
            + weights[:, 1:2] * r_pos_fk
            + weights[:, 2:3] * imu_integrated_pos
            + weights[:, 3:4] * static_pos
        )
        if known_mask is None:
            known_mask = torch.zeros(
                batch_size, device=previous_pos.device, dtype=torch.bool
            )
        if isinstance(known_obj_trans, torch.Tensor):
            known_obj_trans = known_obj_trans.to(
                device=previous_pos.device, dtype=previous_pos.dtype
            )
            if not self.training:
                known_obj_trans = known_obj_trans.detach()
            fused_pos = torch.where(
                known_mask.unsqueeze(-1), known_obj_trans, fused_model_pos
            )
        else:
            fused_pos = fused_model_pos

        effective_weights = weights
        if isinstance(known_weights, torch.Tensor) and isinstance(
            known_weights_mask, torch.Tensor
        ):
            known_weights = known_weights.to(
                device=weights.device, dtype=weights.dtype
            )
            feedback_mask = known_mask & known_weights_mask.to(
                device=weights.device, dtype=torch.bool
            )
            effective_weights = torch.where(
                feedback_mask.unsqueeze(-1), known_weights, weights
            )

        frames_seen = int(step_state.get("frames_seen", 0))
        velocity = (
            (fused_pos - previous_pos) * FRAME_RATE
            if frames_seen > 0
            else torch.zeros_like(fused_pos)
        )
        acceleration = (
            (fused_pos - 2.0 * previous_pos + step_state["prev_prev_obj_trans"])
            * (FRAME_RATE ** 2)
            if frames_seen > 1
            else torch.zeros_like(fused_pos)
        )
        next_state = {
            "rnn_state": rnn_state,
            "prev_obj_trans": fused_pos,
            "prev_prev_obj_trans": previous_pos,
            "prev_obj_delta": fused_pos - previous_pos,
            "prev_gating_weights": effective_weights,
            "prev_obj_rotm": obj_rotm,
            "frames_seen": frames_seen + 1,
            "bone": bone,
            "segment_rel_rotm": segment_rel_rotm,
            "hand_contact_active": contact_active,
        }
        frame_output = {
            "pred_obj_trans": fused_pos,
            "gating_weights": effective_weights,
            "gating_weights_raw": weights,
            "pred_obj_vel_from_posdiff": velocity,
            "pred_obj_acc_from_posdiff": acceleration,
            "pred_lhand_obj_direction": self._unit_vector(bone[:, 0]),
            "pred_rhand_obj_direction": self._unit_vector(bone[:, 1]),
            "pred_lhand_lb": bone[:, 0].norm(dim=-1),
            "pred_rhand_lb": bone[:, 1].norm(dim=-1),
            "pred_lhand_obj_trans": l_pos_fk,
            "pred_rhand_obj_trans": r_pos_fk,
            "pred_imu_obj_trans": imu_integrated_pos,
            "pred_static_obj_trans": static_pos,
            "pred_lhand_obj_vector": bone[:, 0],
            "pred_rhand_obj_vector": bone[:, 1],
            "pred_lhand_bone_proposal": proposal[:, 0],
            "pred_rhand_bone_proposal": proposal[:, 1],
            "pred_lhand_bone_gate": alpha[:, 0, 0],
            "pred_rhand_bone_gate": alpha[:, 1, 0],
            "segment_contact_active": contact_active.to(dtype=fused_pos.dtype),
            "segment_rel_rot6d": matrix_to_rotation_6d(
                segment_rel_rotm.reshape(-1, 3, 3)
            ).reshape(batch_size, self.REL_ROT_FEATURE_DIM),
            "lhand_fk_out": prediction_out[:, : self.HAND_OUTPUT_DIM],
            "rhand_fk_out": prediction_out[:, self.HAND_OUTPUT_DIM :],
        }
        return frame_output, next_state

    def _build_results(self, context, prediction, enable_refine=True):
        results = super()._build_results(
            context, prediction, enable_refine=enable_refine
        )
        results["ot_hand_positions"] = context["hand_positions"]
        results["forearm_obj_rotm"] = context["forearm_obj_rotm"]
        sample_mask = self._resolve_object_mask(
            context["has_object_mask"],
            context["batch_size"],
            context["device"],
            context["dtype"],
        )
        if sample_mask is not None:
            time_mask = sample_mask.view(context["batch_size"], 1, 1)
            for key in (
                "pred_lhand_obj_vector",
                "pred_rhand_obj_vector",
                "pred_lhand_bone_proposal",
                "pred_rhand_bone_proposal",
            ):
                results[key] = results[key] * time_mask
        results["object_trans_v3"] = True
        return results

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        output = V2ObjectTransModule.empty_output(batch_size, seq_len, device)
        zeros_vector = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_head = torch.zeros(
            batch_size, seq_len, V3ObjectTransModule.HAND_OUTPUT_DIM, device=device
        )
        output.update(
            {
                "pred_lhand_obj_vector": zeros_vector.clone(),
                "pred_rhand_obj_vector": zeros_vector.clone(),
                "pred_lhand_bone_proposal": zeros_vector.clone(),
                "pred_rhand_bone_proposal": zeros_vector.clone(),
                "pred_lhand_bone_gate": torch.ones(batch_size, seq_len, device=device),
                "pred_rhand_bone_gate": torch.ones(batch_size, seq_len, device=device),
                "segment_contact_active": torch.zeros(
                    batch_size, seq_len, 2, device=device
                ),
                "segment_rel_rot6d": torch.tensor(
                    _IDENTITY_ROTATION_6D * 2, device=device
                ).expand(batch_size, seq_len, V3ObjectTransModule.REL_ROT_FEATURE_DIM).clone(),
                "lhand_fk_out": zeros_head.clone(),
                "rhand_fk_out": zeros_head.clone(),
                "ot_hand_positions": torch.zeros(
                    batch_size, seq_len, 2, 3, device=device
                ),
                "forearm_obj_rotm": torch.eye(3, device=device)
                .expand(batch_size, seq_len, 2, 3, 3)
                .clone(),
                "object_trans_v3": True,
            }
        )
        return output


ObjectTransModuleV3 = V3ObjectTransModule

__all__ = ["V3ObjectTransModule", "ObjectTransModuleV3"]
