"""Batched advance of a teacher-forced window prefix for the ObjectTrans scans.

Online sliding-window inference hands each window the object trajectory and the
gate weights it already published for the frames before the target.  On those
frames the fused position *is* the published one, so everything the predictor's
recurrence carries forward is known before the scan starts and only the LSTM
hidden state still has to be walked through them.  One batched call does that,
instead of one Python step plus one single-frame kernel launch per frame.

The mixin keeps the shared pieces; each ObjectTrans branch adds its own
structural guard, extra predictor inputs and state entries.
"""
from typing import Any, Dict, Optional

import torch


class KnownPrefixScan:
    """Shared machinery for skipping a teacher-forced prefix's Python loop."""

    # Branches that keep their own per-frame latent recurrence leave this False
    # so that ``forward`` never forwards the hints to their scan.
    SUPPORTS_FAST_KNOWN_PREFIX = False

    @staticmethod
    def _object_frame_vector_sequence(obj_rotm: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Time-batched ``_object_frame_vector`` for [B,T,3,3] x [B,T,3]."""
        return torch.einsum("btji,btj->bti", obj_rotm, vector)

    @staticmethod
    def _known_prefix_length(known_mask, known_weights_mask) -> int:
        """Leading frames that are teacher-forced for *every* batch row.

        Both the position and the gating prefix must be known: a frame with a
        known position but unknown weights falls back to the model's own gate
        output, which re-introduces the recurrent dependence the batched prefix
        exists to avoid.
        """
        if not isinstance(known_mask, torch.Tensor) or not isinstance(known_weights_mask, torch.Tensor):
            return 0
        joint = known_mask & known_weights_mask.to(device=known_mask.device, dtype=torch.bool)
        if not bool(joint.any()):
            return 0
        return int(joint.to(torch.int32).cumprod(dim=1).sum(dim=1).min().item())

    def _fast_known_prefix_supported(self, context: Dict[str, Any], prefix_len: int) -> bool:
        """Structural guard shared by every branch.

        A diagnostic feedback mode rewrites the state a frame is conditioned on,
        so the prefix is no longer a function of the published trajectory.

        The hard static gate needs no guard: it only rewrites the fusion
        weights, and on a teacher-forced frame both the fused position and the
        published gate replace what the gate produced, so it never reaches the
        recurrence.  Its own value comes from the window's IMU and contact
        context, not from the predictor state.
        """
        return (
            prefix_len >= 2
            and context.get("feedback_mode", "normal") == "normal"
        )

    def _known_prefix_recurrent_sequences(self, context, state, known_pos, known_weights, prefix_len):
        """The recurrence carriers of a teacher-forced prefix, as [B,k,...]."""
        fused = known_pos[:, :prefix_len]
        init_pos = context["obj_trans_init"]
        prev_pos = torch.cat((init_pos.unsqueeze(1), fused[:, :-1]), dim=1)
        prev_delta = torch.cat(
            (torch.zeros_like(init_pos).unsqueeze(1), fused[:, :-1] - prev_pos[:, :-1]), dim=1
        )
        prev_weights = torch.cat(
            (state["prev_gating_weights"].unsqueeze(1), known_weights[:, :prefix_len - 1]), dim=1
        )
        return fused, prev_pos, prev_delta, prev_weights

    def _known_prefix_feedback_input(self, context, prev_pos, prev_delta, prev_weights, prefix_len):
        """Time-batched counterpart of ``_feedback_features`` + ``feedback_embed``."""
        obj_rotm = context["obj_rotm"][:, :prefix_len]
        features = torch.cat(
            (
                self._object_frame_vector_sequence(
                    obj_rotm, prev_pos - context["lhand_position"][:, :prefix_len]
                ),
                self._object_frame_vector_sequence(
                    obj_rotm, prev_pos - context["rhand_position"][:, :prefix_len]
                ),
                self._object_frame_vector_sequence(obj_rotm, prev_delta),
                prev_weights,
            ),
            dim=-1,
        )
        return self.feedback_embed(features)

    def _batched_known_prefix(
        self,
        context,
        state,
        known_pos,
        known_weights,
        prefix_len,
        extra_input: Optional[torch.Tensor] = None,
        state_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Return the state as it would be after ``prefix_len`` sequential steps."""
        fused, prev_pos, prev_delta, prev_weights = self._known_prefix_recurrent_sequences(
            context, state, known_pos, known_weights, prefix_len
        )
        step_input = context["prediction_input"][:, :prefix_len]
        if self._state_feedback_enabled():
            step_input = step_input + self._known_prefix_feedback_input(
                context, prev_pos, prev_delta, prev_weights, prefix_len
            )
        if extra_input is not None:
            step_input = step_input + extra_input
        _, rnn_state = self.object_trans_head.forward_with_state(
            step_input, state=state["rnn_state"]
        )
        state = dict(state)
        state.update(
            {
                "rnn_state": rnn_state,
                "prev_obj_trans": fused[:, -1],
                "prev_prev_obj_trans": prev_pos[:, -1],
                "prev_obj_delta": fused[:, -1] - prev_pos[:, -1],
                "prev_gating_weights": known_weights[:, prefix_len - 1],
                "prev_obj_rotm": context["obj_rotm"][:, prefix_len - 1],
                "frames_seen": int(state.get("frames_seen", 0)) + prefix_len,
            }
        )
        if state_overrides:
            state.update(state_overrides)
        return state

    def _resolve_known_prefix(self, context, known_mask, known_weights_mask, scan_end):
        """Prefix length to skip, or 0 when the sequential scan has to run.

        At least one frame always stays sequential, so that the caller's
        published frame comes from a real predictor step.
        """
        seq_len = int(context["seq_len"])
        last_needed = seq_len if scan_end is None else min(int(scan_end), seq_len)
        prefix_len = min(
            self._known_prefix_length(known_mask, known_weights_mask), last_needed - 1
        )
        if prefix_len < 1 or not self._fast_known_prefix_supported(context, prefix_len):
            return 0
        return prefix_len

    @staticmethod
    def _restore_window_shape(prediction, context, known_pos, known_weights, scan_start, scan_stop):
        """Re-expand a partial scan to the full window length.

        Only the published frames are ever read from this dict, but the window
        shape has to survive so that downstream time slicing and the causal
        refine head keep working.  Skipped prefix frames carry their published
        position and gate, which the refine head consumes; the remaining keys
        and the trimmed tail carry placeholders that no caller publishes.
        """
        batch_size = context["batch_size"]
        seq_len = int(context["seq_len"])
        restored = {}
        for key, value in prediction.items():
            head = None
            if scan_start > 0:
                if key == "pred_obj_trans":
                    head = known_pos[:, :scan_start]
                elif key == "gating_weights":
                    head = known_weights[:, :scan_start]
                else:
                    head = value.new_zeros((batch_size, scan_start) + tuple(value.shape[2:]))
            full = value if head is None else torch.cat((head, value), dim=1)
            if full.shape[1] < seq_len:
                tail = full[:, -1:].expand(
                    (batch_size, seq_len - int(full.shape[1])) + tuple(full.shape[2:])
                )
                full = torch.cat((full, tail), dim=1)
            restored[key] = full
        return restored
