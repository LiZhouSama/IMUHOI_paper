"""Feedback-free ObjectTrans v2 for the cascade refine architecture.

The cascade (HP -> VC -> OT -> VC-refine -> OT-refine -> HP-refine) drops the
OT state feedback, so the shared predictor LSTM reads observed inputs only and
its per-frame outputs no longer depend on the fused object.  Two consequences
shape this subclass:

* the whole window's predictor outputs come from one batched LSTM call, and the
  causal fusion scan only has to visit the frames a caller needs (online: the
  published frame, plus whatever the caller asks for after it);
* the bimanual-closure gate reads the predictor's top-layer hidden state of the
  *current* frame.  The stock no-feedback path of
  ``ObjectTransModule._scan_predictor`` never steps the LSTM inside its fusion
  loop and hands every frame the initial state instead, while the online path
  steps it properly.  Training and inference would disagree, so this scan
  passes each frame its own hidden state.

The checkpoint layout is the v2 one (the unused feedback MLP keeps its
zero-initialised weights), so ``ARCH_VERSION`` stays 2.
"""
from __future__ import annotations

import torch

from model.rnn.object_trans_v2 import V2ObjectTransModule


class CascadeObjectTransModule(V2ObjectTransModule):
    """ObjectTrans v2 (fixed VC-driven fusion) without state feedback."""

    def __init__(self, cfg):
        # V2ObjectTransModule.__init__ tests ``type(self) is V2ObjectTransModule``
        # to keep v3 (a different scan) from inheriting the FK filter and the
        # closure.  This class is v2 with only the scan rewritten, so it is built
        # as a v2 instance and re-typed afterwards; no other v2 method checks
        # the type.
        self.__class__ = V2ObjectTransModule
        try:
            V2ObjectTransModule.__init__(self, cfg)
        finally:
            self.__class__ = CascadeObjectTransModule
        if self._state_feedback_enabled():
            raise ValueError("CascadeObjectTransModule requires object_trans_state_feedback: false")
        if self.grasp_latent_enabled or self.bigrasp_enabled or self.state_trust_enabled:
            raise ValueError("CascadeObjectTransModule supports the plain v2 predictor only")

    # ------------------------------------------------------------------ predictor
    def predictor_sequence(self, context):
        """Decoded predictor outputs [B,T,8], top-layer hidden states [B,T,H] and the final LSTM state.

        Mirrors ``RNN.forward_with_state`` (including its FP32 recurrent core
        under BF16 autocast) but keeps the per-frame hidden sequence.
        """
        head = self.object_trans_head
        x = head.encode_input(context["prediction_input"])
        state = head.initial_state(context["prediction_init"])
        if x.is_cuda and x.dtype == torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                hidden, final_state = head.rnn(x.float(), tuple(value.float() for value in state))
        else:
            hidden, final_state = head.rnn(x, state)
        return head.linear2(hidden), hidden, final_state

    def fk_candidate_sequence(self, context, prediction_sequence):
        """State-free per-hand FK candidates [B,T,2,3] for every window frame (before the v2 filter)."""
        l_fk, r_fk = prediction_sequence[..., :4], prediction_sequence[..., 4:8]
        obj_rotm = context["obj_rotm"]
        candidates = []
        for fk_out, hand in ((l_fk, context["lhand_position"]), (r_fk, context["rhand_position"])):
            direction = self._unit_vector(fk_out[..., :3]).to(obj_rotm.dtype)
            length = self._softplus_positive(fk_out[..., 3]).to(obj_rotm.dtype)
            world = torch.einsum("btij,btj->bti", obj_rotm, direction)
            candidates.append(hand + world * length.unsqueeze(-1))
        return torch.stack(candidates, dim=2)

    # ----------------------------------------------------------------------- scan
    def _filter_prefix_known(self, context, prefix_len):
        if not self.fk_filter_enabled:
            return True
        known = context.get("v2_filter_known")
        return known is not None and bool(known["mask"][:, :prefix_len].all().item())

    def _teacher_forced_state(self, context, state, known_pos, known_weights, prefix_len):
        """Scan state after ``prefix_len`` teacher-forced frames (no LSTM stepping needed)."""
        fused = known_pos[:, :prefix_len]
        previous = fused[:, -2] if prefix_len >= 2 else context["obj_trans_init"]
        state = dict(state)
        state.update(
            prev_obj_trans=fused[:, -1],
            prev_prev_obj_trans=previous,
            prev_obj_delta=fused[:, -1] - previous,
            prev_gating_weights=known_weights[:, prefix_len - 1],
            prev_obj_rotm=context["obj_rotm"][:, prefix_len - 1],
            frames_seen=int(state.get("frames_seen", 0)) + prefix_len,
        )
        if self.fk_filter_enabled:
            known, last, batch_size = context["v2_filter_known"], prefix_len - 1, context["batch_size"]
            state.update(
                v2_fk_filter_pos=known["v2_fk_filter_pos"][:, last].reshape(batch_size, 2, 3),
                v2_fk_filter_vel=known["v2_fk_filter_vel"][:, last].reshape(batch_size, 2, 3),
                v2_fk_filter_active=known["v2_fk_filter_active"][:, last],
            )
        return state

    def _scan_predictor(
        self,
        context,
        known_obj_trans_prefix=None,
        known_obj_trans_prefix_mask=None,
        known_gating_weights_prefix=None,
        fast_known_prefix: bool = False,
        scan_end: int = None,
    ):
        batch_size, seq_len = context["batch_size"], int(context["seq_len"])
        device, dtype = context["device"], context["dtype"]
        prediction_sequence, hidden, final_rnn_state = self.predictor_sequence(context)
        context["cascade_fk_candidates"] = self.fk_candidate_sequence(context, prediction_sequence)

        known_pos, known_mask = self._prepare_known_sequence(
            known_obj_trans_prefix, batch_size, seq_len, 3, device, dtype, known_obj_trans_prefix_mask
        )
        known_weights, known_weights_mask = self._prepare_known_sequence(
            known_gating_weights_prefix, batch_size, seq_len, 4, device, dtype, None
        )
        state = self._init_scan_state(context)
        scan_stop = seq_len if scan_end is None else min(int(scan_end), seq_len)
        scan_start = 0
        if fast_known_prefix and scan_stop > 1:
            prefix_len = min(self._known_prefix_length(known_mask, known_weights_mask), scan_stop - 1)
            if prefix_len >= 1 and self._filter_prefix_known(context, prefix_len):
                state = self._teacher_forced_state(context, state, known_pos, known_weights, prefix_len)
                scan_start = prefix_len

        collected = {}
        for time_index in range(scan_start, scan_stop):
            # (h, c) placeholder whose top layer is this frame's hidden state:
            # only the closure gate reads it (rnn_state[0][-1]).
            frame_hidden = hidden[:, time_index].unsqueeze(0)
            frame_output, state = self._predict_step(
                context,
                time_index,
                state,
                known_obj_trans=known_pos[:, time_index],
                known_mask=known_mask[:, time_index],
                known_weights=known_weights[:, time_index],
                known_weights_mask=known_weights_mask[:, time_index],
                prediction_out=prediction_sequence[:, time_index],
                next_rnn_state=(frame_hidden, frame_hidden),
            )
            for key, value in frame_output.items():
                collected.setdefault(key, []).append(value)
        prediction = {key: torch.stack(values, dim=1) for key, values in collected.items()}
        if scan_start > 0 or scan_stop < seq_len:
            prediction = self._restore_window_shape(
                prediction, context, known_pos, known_weights, scan_start, scan_stop
            )
        state = dict(state)
        state["rnn_state"] = final_rnn_state
        return prediction, state

    def _build_results(self, context, prediction, enable_refine=True):
        results = super()._build_results(context, prediction, enable_refine=enable_refine)
        # State-free quantities of every window frame, read by VC-refine.
        candidates = context.get("cascade_fk_candidates")
        if isinstance(candidates, torch.Tensor):
            results["cascade_fk_candidates"] = candidates
        closure = context.get("closure")
        if isinstance(closure, dict) and isinstance(closure.get("candidate"), torch.Tensor):
            results["cascade_bimanual_candidate"] = closure["candidate"]
        static_gate = context.get("static_gate")
        if isinstance(static_gate, torch.Tensor):
            results["cascade_static_gate"] = static_gate.to(context["dtype"])
        return results


__all__ = ["CascadeObjectTransModule"]
