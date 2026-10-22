"""Cascade refine pipeline: HP -> VC -> OT -> VC-refine -> OT-refine -> HP-refine.

Stages (all but VC-refine reuse existing modules):

* HP      : HumanPoseModule (stage 1, frozen).
* VC      : VelocityContact v5 (frozen); its contacts are the *first pass*.
* OT      : ObjectTrans v2 without state feedback (``CascadeObjectTransModule``).
            Online it fuses the published frame from the published history (the
            teacher-forced prefix) and continues through the lookahead, so that
            VC-refine sees the fused trajectory of the whole window.
* VC-refine: residual contact correction from the first-pass OT/HP geometry.
* OT-refine: the same OT weights re-run with the refined contacts.
            ``e2`` (default) re-fuses the whole window, so a contact corrected in
            hindsight also undoes the drag it caused.  The window starts from the
            previous window's OT-refine trajectory (the "carry"): window t starts
            at frame t-59 from the value window t-1 re-fused there, and the first
            pass teacher-forces window t-1's re-fused trajectory.  The trajectory
            is therefore one chain that advances a frame per window (a fixed-lag
            smoother).  ``e2p`` starts each window from the value *published* at
            its first frame instead; P(t) then depends on P(t-59) only, so the
            publication splits into 59 interleaved chains that drift apart (kept
            to reproduce that failure).  ``e1`` keeps the published prefix and
            re-fuses the published frame only.
* HP-refine: the OT refine head on the OT-refine output (output-only).

Online history: contacts, object position, fusion weights and the v2 filter
state are the refined (published) values; ``e2`` reads the OT start state from
the carry instead.  HumanPose initialisers keep reading the stage-1 history by
default (``cascade_hp_history: stage1``); ``refined`` feeds the refined
pose/root back, as in the original proposal.

Publication.  The evaluator re-fuses the object with the refined palms
(``eval_IMUHOI._build_published_object_trans``) by replaying the four-way
fusion frame after frame from its own previous output.  That replay is exact
for the stock protocol and for ``e1``, but ``e2``/``e2p`` re-fuse each window from
its start, which a frame-to-frame replay cannot express.  The cascade therefore
applies the same hand correction itself, inside each window
(``cascade_published_obj_trans``), publishes that trajectory as
``pred_obj_trans`` and hides the candidate keys the evaluator replays, so the
evaluator scores the cascade's own publication.  The uncorrected trajectory the
loop runs on is kept as ``cascade_raw_obj_trans``.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from model.rnn.imuhoi_model import IMUHOIModel
from model.rnn.online import (
    TimeDictAccumulator,
    concat_time_dicts,
    infer_batch_seq,
    select_time_context,
    slice_time_dict,
)
from .ot_nofb import CascadeObjectTransModule
from .vc_refine import REFINED_CONTACT_KEYS, VCRefineModule

CASCADE_MODES = ("e1", "e2", "e2p")
HP_HISTORY_SOURCES = ("stage1", "refined")
# Candidate keys the evaluator replays; renamed on the cascade's final output.
EVALUATOR_REPLAY_KEYS = ("pred_static_obj_trans", "pred_imu_obj_trans")
# Per-frame values ``e2`` carries from one window's OT-refine pass to the next
# (plus the OT module's SEGMENT_STATE_OUTPUT_KEYS).
CARRY_KEYS = ("pred_obj_trans", "gating_weights", "pred_hand_contact_prob", "cascade_published_obj_trans")


def publication_chain(
    start: torch.Tensor,
    weights: torch.Tensor,
    left_fk: torch.Tensor,
    right_fk: torch.Tensor,
    imu_pos: torch.Tensor,
    static_pos: torch.Tensor,
    palm_delta: torch.Tensor,
) -> torch.Tensor:
    """Four-way fusion re-run with hand-corrected FK candidates.

    Same recursion as ``eval_IMUHOI._build_published_object_trans`` (batched):
    x_j = w0 (FK_L + dh_L) + w1 (FK_R + dh_R) + w2 (x_{j-1} + imu_j - static_j)
          + w3 x_{j-1}, starting from x_{-1} = ``start``.
    Shapes: start [B,3], weights [B,T,4], candidates [B,T,3], palm_delta [B,T,2,3].
    """
    left = left_fk + palm_delta[:, :, 0]
    right = right_fk + palm_delta[:, :, 1]
    imu_delta = imu_pos - static_pos
    previous = start
    outputs = []
    for frame in range(weights.shape[1]):
        w = weights[:, frame]
        previous = (
            w[:, 0:1] * left[:, frame]
            + w[:, 1:2] * right[:, frame]
            + w[:, 2:3] * (previous + imu_delta[:, frame])
            + w[:, 3:4] * previous
        )
        outputs.append(previous)
    return torch.stack(outputs, dim=1)


class CascadeIMUHOIModel(IMUHOIModel):
    """IMUHOI RNN pipeline with the VC-refine / OT-refine / HP-refine cascade."""

    def __init__(self, cfg, device, no_trans: bool = False, velocity_contact_arch: Optional[str] = None):
        super().__init__(cfg, device, no_trans=no_trans, object_trans_arch="v2", velocity_contact_arch=velocity_contact_arch)
        self.object_trans_module = CascadeObjectTransModule(cfg)
        self.vc_refine_module = VCRefineModule(cfg)
        self.cascade_mode = str(getattr(cfg, "cascade_refit_mode", "e2")).strip().lower()
        if self.cascade_mode not in CASCADE_MODES:
            raise ValueError(f"cascade_refit_mode must be one of {CASCADE_MODES}, got {self.cascade_mode!r}")
        self.cascade_hp_history = str(getattr(cfg, "cascade_hp_history", "stage1")).strip().lower()
        if self.cascade_hp_history not in HP_HISTORY_SOURCES:
            raise ValueError(f"cascade_hp_history must be one of {HP_HISTORY_SOURCES}")
        # False = identity VC-refine (the cascade then reduces to the feedback-free baseline).
        self.vc_refine_enabled = bool(getattr(cfg, "cascade_vc_refine", True))

    # ------------------------------------------------------------- stage helpers
    def run_vc_refine(self, vc_first: Dict[str, torch.Tensor], ot_first: Dict[str, torch.Tensor],
                      hands: torch.Tensor, obj_imu: torch.Tensor, state_init: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Refined contact outputs over the window (identity when VC-refine is disabled)."""
        if not self.vc_refine_enabled:
            return {key: vc_first[key] for key in REFINED_CONTACT_KEYS if key in vc_first}
        return self.vc_refine_module(
            vc_first["pred_hand_contact_logits"],
            vc_first["pred_obj_vel"],
            obj_imu,
            hands,
            ot_first["cascade_fk_candidates"],
            ot_first["pred_obj_trans"],
            state_init,
            bimanual_candidate=ot_first.get("cascade_bimanual_candidate"),
            static_gate=ot_first.get("cascade_static_gate"),
        )

    def _ot_common_kwargs(self, data, hp_context, obj_vel, lb_prior_quantiles):
        return dict(
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=obj_vel,
            contact_init=data.get("contact_init"),
            has_object_mask=data.get("has_object"),
            human_pose_input=hp_context.get("p_pred"),
            root_trans_input=hp_context.get("root_trans_pred"),
            lb_prior_quantiles=lb_prior_quantiles,
        )

    def cascade_window(
        self,
        data: Dict[str, torch.Tensor],
        hp_out: Dict[str, torch.Tensor],
        vc_first: Dict[str, torch.Tensor],
        obj_trans_init: torch.Tensor,
        lb_prior_quantiles: Optional[torch.Tensor] = None,
        refine_human: bool = False,
        first_pass_grad: bool = False,
    ):
        """Offline cascade over one window (training and the online warmup).

        Returns (first-pass OT, refined contacts, OT-refine output).  The first
        pass runs without gradients unless ``first_pass_grad``; the OT-refine pass
        keeps the graph so a position loss reaches VC-refine through the fixed
        fusion weights and the predictor's contact input.
        """
        hands = self._resolve_ot_hand_positions(hp_out)
        common = self._ot_common_kwargs(data, hp_out, vc_first["pred_obj_vel"], lb_prior_quantiles)
        with torch.set_grad_enabled(first_pass_grad and torch.is_grad_enabled()):
            ot_first = self.object_trans_module(
                hands, vc_first["pred_hand_contact_prob"], obj_trans_init, enable_refine=False, **common
            )
        refined = self.run_vc_refine(vc_first, ot_first, hands, data["obj_imu"], obj_trans_init)
        ot_refined = self.object_trans_module(
            hands, refined["pred_hand_contact_prob"], obj_trans_init, enable_refine=refine_human, **common
        )
        return ot_first, refined, ot_refined

    # ------------------------------------------------------------------ outputs
    @staticmethod
    def _palm_delta(results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Refined minus stage-1 palms [B,T,2,3] (zeros without HP-refine)."""
        refined = results.get("pred_palm_glb_pos")
        stage1 = results.get("stage1_pred_palm_glb_pos")
        if isinstance(refined, torch.Tensor) and isinstance(stage1, torch.Tensor) and refined.shape == stage1.shape:
            return refined - stage1
        reference = results["pred_obj_trans"]
        return reference.new_zeros(*reference.shape[:2], 2, 3)

    @staticmethod
    def _merge_refined_contacts(vc_out: Dict[str, torch.Tensor], refined: Dict[str, torch.Tensor]):
        merged = dict(vc_out)
        merged.update({key: refined[key] for key in REFINED_CONTACT_KEYS if key in refined})
        merged.pop("pred_hand_contact_prob_frame", None)
        merged["cascade_vc_first_pass_prob"] = vc_out["pred_hand_contact_prob"]
        delta = refined.get("cascade_vcr_delta_logits")
        if isinstance(delta, torch.Tensor):
            merged["cascade_vcr_delta_logits"] = delta
        return merged

    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
        refine_human: Optional[bool] = None,
        feedback_control=None,
    ) -> Dict[str, torch.Tensor]:
        if feedback_control is not None:
            raise ValueError("The cascade has no OT state feedback to diagnose")
        if compute_fk:
            raise ValueError("compute_fk is not supported by the cascade")
        refine_human = self._resolve_refine_human(refine_human)
        hp_out = self.human_pose_module(self._build_hp_input_dict(data_dict))
        results = dict(hp_out)
        vc_out = self.velocity_contact_module(
            {
                "human_imu": data_dict["human_imu"],
                "obj_imu": data_dict["obj_imu"],
                "hand_vel_glb_init": data_dict["hand_vel_glb_init"],
                "obj_vel_init": data_dict["obj_vel_init"],
                "obj_trans_init": data_dict["obj_trans_init"],
                "contact_init": data_dict.get("contact_init"),
                "hp_out": hp_out,
            },
            hp_out=hp_out,
        )
        vc_out = self._apply_vc_segment_contact_if_enabled(vc_out, data_dict.get("contact_init"))
        vc_out = self._apply_vc_oracle(vc_out, data_dict)
        has_object = data_dict.get("has_object")
        if use_object_data and (has_object is None or has_object.any()):
            lb_prior = self._predict_lb_prior(hp_out, vc_out, data_dict["obj_imu"])
            _, refined, ot_refined = self.cascade_window(
                data_dict, hp_out, vc_out, data_dict["obj_trans_init"], lb_prior, refine_human=refine_human
            )
            results.update(self._merge_refined_contacts(vc_out, refined))
            results.update(ot_refined)
            if refine_human:
                self._promote_refined_human_outputs(results, ot_refined, data_dict)
            palm_delta = self._palm_delta(results)
            results["cascade_palm_delta"] = palm_delta
            results["cascade_published_obj_trans"] = publication_chain(
                results["pred_static_obj_trans"][:, 0],
                results["gating_weights"],
                results["pred_lhand_obj_trans"],
                results["pred_rhand_obj_trans"],
                results["pred_imu_obj_trans"],
                results["pred_static_obj_trans"],
                palm_delta,
            )
        else:
            results.update(vc_out)
        results["has_object"] = has_object
        return results

    # ------------------------------------------------------------------- online
    def _stage1_history_for_inits(self, history):
        if getattr(self, "cascade_hp_history", "stage1") == "refined":
            return history
        return IMUHOIModel._stage1_history_for_inits(history)

    def _initial_carry(self, history):
        """``e2`` carry from the warmup: one offline cascade pass, already a single chain."""
        keys = CARRY_KEYS + tuple(self.object_trans_module.SEGMENT_STATE_OUTPUT_KEYS)
        return {"start": 0, **{key: history.get(key) for key in keys}}

    def _cascade_online_frame(self, plan, history, batch_size, refine_human, carry=None):
        """Run OT -> VC-refine -> OT-refine -> HP-refine for one published frame.

        Returns the published frame and the ``e2`` carry for the next window
        (None for the other modes).
        """
        target, start = plan["target"], plan["start"]
        window_data, window_len, target_local = plan["window_data"], plan["window_len"], plan["target_local"]
        prefix = select_time_context(history, start, target)
        hp_context = plan["hp_context"]
        vc_raw = self._apply_vc_segment_contact_if_enabled(plan["vc_out_raw"], window_data.get("contact_init"))
        vc_raw = self._apply_vc_oracle(vc_raw, window_data)
        vc_current = slice_time_dict(vc_raw, target_local, window_len, batch_size, window_len)
        vc_context = self._merge_stage_context(prefix, vc_current)

        results = dict(hp_context)
        has_object = window_data.get("has_object")
        hands = self._resolve_ot_hand_positions(hp_context)
        lb_prior = self._predict_lb_prior(hp_context, vc_context, window_data["obj_imu"])
        if lb_prior is None and self._online_lb_stats_enabled():
            lb_prior = self._compute_online_lb_stats(history)
        segment_keys = self.object_trans_module.SEGMENT_STATE_OUTPUT_KEYS
        first_contact = vc_context["pred_hand_contact_prob"]
        if self.cascade_mode == "e2":
            # The OT state of frames [start, target) is window t-1's re-fused trajectory.
            if carry is None:
                raise ValueError("cascade mode e2 needs the previous window's carry")
            shift = start - carry["start"]
            span = slice(shift, shift + target_local)
            state = {key: value[:, span] if isinstance(value, torch.Tensor) else value
                     for key, value in carry.items() if key != "start"}
            if state["pred_obj_trans"].shape[1] != target_local:
                raise ValueError(f"e2 carry covers {state['pred_obj_trans'].shape[1]} frames, need {target_local}")
            first_contact = torch.cat(
                (state["pred_hand_contact_prob"], first_contact[:, target_local:]), dim=1
            )
        else:
            state = prefix
        obj_prefix, gate_prefix = state["pred_obj_trans"], state["gating_weights"]
        ot_init = obj_prefix[:, 0]
        segment_prefix = {key: state.get(key) for key in segment_keys}
        common = self._ot_common_kwargs(window_data, hp_context, vc_context["pred_obj_vel"], lb_prior)

        # First pass: fuse the published frame from the history (e2: the carried
        # re-fused trajectory) and continue through the lookahead (VC-refine
        # reads the whole window).
        ot_first = self.object_trans_module(
            hands, first_contact, ot_init, enable_refine=False,
            known_obj_trans_prefix=obj_prefix, known_gating_weights_prefix=gate_prefix,
            known_segment_state_prefix=segment_prefix, fast_known_prefix=True, scan_end=None, **common,
        )
        refined = self.run_vc_refine(vc_raw, ot_first, hands, window_data["obj_imu"], ot_init)

        whole_window = self.cascade_mode in ("e2", "e2p")
        if whole_window:
            # Whole-window re-fusion from the window start (e2: the carried state,
            # e2p: the value published at the start frame).
            contact = refined["pred_hand_contact_prob"]
            known_pos, known_gate = obj_prefix[:, :1], gate_prefix[:, :1]
            known_segment = {key: value[:, :1] if isinstance(value, torch.Tensor) else value
                             for key, value in segment_prefix.items()}
        else:
            contact = torch.cat(
                (prefix["pred_hand_contact_prob"], refined["pred_hand_contact_prob"][:, target_local:]), dim=1
            )
            known_pos, known_gate, known_segment = obj_prefix, gate_prefix, segment_prefix
        ot_refined = self.object_trans_module(
            hands, contact, ot_init, enable_refine=refine_human,
            known_obj_trans_prefix=known_pos, known_gating_weights_prefix=known_gate,
            known_segment_state_prefix=known_segment, fast_known_prefix=True, scan_end=target_local + 1, **common,
        )
        results.update(self._merge_refined_contacts(vc_context, refined))
        results["cascade_vc_first_pass_prob"] = torch.cat(
            (prefix["cascade_vc_first_pass_prob"], vc_raw["pred_hand_contact_prob"][:, target_local:]), dim=1
        )
        results.update(ot_refined)
        if refine_human:
            self._promote_refined_human_outputs(results, ot_refined, window_data, derive=True)

        palm_delta = self._palm_delta(results)
        results["cascade_palm_delta"] = palm_delta
        if whole_window:
            # The palm corrections of past frames are their published (fixed)
            # values, so a frame enters every window's chain with the same input.
            chain_start, first = state["cascade_published_obj_trans"][:, 0], 1
            history_delta = prefix["cascade_palm_delta"]
            frame_delta = torch.cat((history_delta[:, first:], palm_delta[:, target_local:target_local + 1]), dim=1)
        else:
            chain_start, first = prefix["cascade_published_obj_trans"][:, -1], target_local
            frame_delta = palm_delta[:, target_local:target_local + 1]
        stop = target_local + 1
        published = publication_chain(
            chain_start,
            results["gating_weights"][:, first:stop],
            results["pred_lhand_obj_trans"][:, first:stop],
            results["pred_rhand_obj_trans"][:, first:stop],
            results["pred_imu_obj_trans"][:, first:stop],
            results["pred_static_obj_trans"][:, first:stop],
            frame_delta,
        )
        results["cascade_published_obj_trans"] = torch.cat(
            (prefix["cascade_published_obj_trans"], published[:, -1:],
             published[:, -1:].expand(batch_size, window_len - stop, 3)), dim=1
        )
        results["has_object"] = has_object
        next_carry = None
        if self.cascade_mode == "e2":
            # This window's re-fused frames [start, target] start the next window.
            next_carry = {"start": start}
            for key in ("pred_obj_trans", "gating_weights") + tuple(segment_keys):
                value = ot_refined.get(key)
                next_carry[key] = value[:, :stop] if isinstance(value, torch.Tensor) else None
            next_carry["pred_hand_contact_prob"] = refined["pred_hand_contact_prob"][:, :stop]
            next_carry["cascade_published_obj_trans"] = torch.cat((chain_start.unsqueeze(1), published), dim=1)
        return slice_time_dict(results, target_local, target_local + 1, batch_size, window_len), next_carry

    def _inference_online_sequence(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
        refine_human: Optional[bool] = None,
        online_window: int = 120,
        online_lookahead: int = 5,
        progress_callback=None,
        feedback_mode: str = "normal",
        feedback_perturbation=None,
        feedback_injection_frame: Optional[int] = None,
        online_chunk: int = 1,
    ) -> Dict[str, torch.Tensor]:
        if str(feedback_mode or "normal").lower() != "normal":
            raise ValueError("The cascade has no OT state feedback to diagnose")
        if compute_fk or not use_object_data:
            raise ValueError("The cascade online loop needs the object branch and no FK baseline")
        refine_human = self._resolve_refine_human(refine_human)
        batch_size, seq_len = infer_batch_seq(data_dict)
        window, lookahead = int(online_window), int(online_lookahead)
        publish_len = max(seq_len - lookahead, 0)
        if publish_len == 0:
            template = self.forward(data_dict, refine_human=refine_human)
            return self._finalize_publication(slice_time_dict(template, 0, 0, batch_size, seq_len))

        warmup_publish_len = min(window, publish_len)
        warmup_input_len = warmup_publish_len + lookahead
        warmup_data = slice_time_dict(data_dict, 0, warmup_input_len, batch_size, seq_len)
        warmup = self.forward(warmup_data, refine_human=refine_human)
        history = slice_time_dict(warmup, 0, warmup_publish_len, batch_size, warmup_input_len)
        history_keys = set(history)
        history_acc = TimeDictAccumulator(history, publish_len)
        history = history_acc.current()
        if callable(progress_callback):
            progress_callback(warmup_publish_len, publish_len)

        carry = self._initial_carry(history) if self.cascade_mode == "e2" else None
        chunk = max(1, min(int(online_chunk), max(1, window - 1)))
        for chunk_start in range(warmup_publish_len, publish_len, chunk):
            plans = self._prepare_online_chunk(
                data_dict, history, list(range(chunk_start, min(chunk_start + chunk, publish_len))),
                window, lookahead, batch_size, seq_len, "normal", None, None,
            )
            for plan in plans:
                frame, carry = self._cascade_online_frame(plan, history, batch_size, refine_human, carry)
                missing = history_keys - set(frame)
                if missing:
                    raise KeyError(f"cascade frame lacks history keys: {sorted(missing)}")
                history = history_acc.append({key: frame[key] for key in history_keys})
                if callable(progress_callback):
                    progress_callback(plan["target"] + 1, publish_len)
        output = history_acc.current()
        if refine_human:
            # Same deferred derivation as the stock loop: identical per-frame FK,
            # and the root velocity from the published (not per-window) roots.
            self._derive_refined_human_outputs(output, data_dict, chunk_rows=self.DERIVE_CHUNK_ROWS)
        return self._finalize_publication(output)

    @staticmethod
    def _finalize_publication(output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = dict(output)
        published = output.get("cascade_published_obj_trans")
        if isinstance(published, torch.Tensor):
            output["cascade_raw_obj_trans"] = output["pred_obj_trans"]
            output["pred_obj_trans"] = published
            for key in EVALUATOR_REPLAY_KEYS:
                if key in output:
                    output[key.replace("pred_", "cascade_", 1)] = output.pop(key)
        return output


def build_cascade_model(cfg, device, module_paths: Dict[str, str], kind: str = "cascade",
                        no_trans: bool = False, velocity_contact_arch: Optional[str] = None):
    """Build and load a cascade model (``cascade``) or the feedback-free OT baseline (``b0``)."""
    from utils.utils import flatten_lstm_parameters

    if kind == "cascade":
        model = CascadeIMUHOIModel(cfg, device, no_trans=no_trans, velocity_contact_arch=velocity_contact_arch)
    elif kind == "b0":
        model = IMUHOIModel(cfg, device, no_trans=no_trans, object_trans_arch="v2",
                            velocity_contact_arch=velocity_contact_arch)
        model.object_trans_module = CascadeObjectTransModule(cfg)
    else:
        raise ValueError(f"unknown cascade model kind {kind!r}")
    model = model.to(device)
    model.load_pretrained_modules({key: value for key, value in module_paths.items() if value})
    flatten_lstm_parameters(model)
    model.eval()
    return model


__all__ = ["CascadeIMUHOIModel", "build_cascade_model", "publication_chain", "CASCADE_MODES", "HP_HISTORY_SOURCES"]
