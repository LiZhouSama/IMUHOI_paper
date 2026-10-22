"""ObjectTrans v2 with fixed VC-driven fusion weights."""
import math

import torch

from .base import RNNWithInit
from .object_trans import ObjectTransModule as BaseObjectTransModule
from utils.object_fusion import build_contact_fusion_weights


class V2ObjectTransModule(BaseObjectTransModule):
    """Current ObjectTrans pipeline without a trainable fusion head."""

    # v2 reuses the base scan.  Its fixed fusion only feeds the frame's own
    # gate, and the one recurrence it adds, the FK alpha-beta filter, is
    # published per frame and restored on teacher-forced frames, so the
    # batched known prefix applies (see _batched_known_prefix below).
    SUPPORTS_FAST_KNOWN_PREFIX = True
    ARCH_VERSION = 2

    def __init__(self, cfg):
        if type(self) is not V2ObjectTransModule and bool(getattr(cfg, 'object_trans_bimanual_closure', False)):
            raise ValueError('Bimanual closure with fixed v2 fusion is not supported by v3')
        super().__init__(cfg)
        # Default only for v2 itself: v3 inherits this class but keeps its own
        # candidate/state recurrence and must not inherit the filter.
        is_v2 = type(self) is V2ObjectTransModule
        self.fk_filter_alpha = float(getattr(cfg, 'v2_fk_filter_alpha', .35)) if is_v2 else 1.0
        self.fk_filter_beta = float(getattr(cfg, 'v2_fk_filter_beta', .05)) if is_v2 else 0.0
        self.fk_filter_enabled = self.fk_filter_alpha < 1.0
        if not 0 < self.fk_filter_alpha <= 1 or not 0 <= self.fk_filter_beta <= 1:
            raise ValueError('Invalid v2 FK filter gain')
        if self.fk_filter_enabled:
            if type(self) is not V2ObjectTransModule:
                raise ValueError('FK candidate filter is supported only for v2, not subclasses')
            if self.feedback_source != 'published' or self.grasp_latent_enabled or self.bigrasp_enabled:
                raise ValueError('FK candidate filter requires v2 with published feedback and no grasp latent or bigrasp')
            self.SEGMENT_STATE_OUTPUT_KEYS = ('v2_fk_filter_pos', 'v2_fk_filter_vel', 'v2_fk_filter_active')

        predictor_hidden = self.object_trans_head.n_hidden
        predictor_layers = self.object_trans_head.n_rnn_layer
        predictor_dropout = float(getattr(cfg, "object_trans_dropout", 0.2))
        n_init = int(self.object_trans_head.init_net[0].in_features)
        self.object_trans_head = RNNWithInit(
            n_input=self.n_prediction_input,
            n_output=8,
            n_hidden=predictor_hidden,
            n_init=n_init,
            n_rnn_layer=predictor_layers,
            bidirectional=False,
            dropout=predictor_dropout,
        )
        length_bias = math.log(
            math.expm1(self.object_trans_length_init - self.LENGTH_EPS)
        )
        with torch.no_grad():
            self.object_trans_head.linear2.bias[3] = length_bias
            self.object_trans_head.linear2.bias[7] = length_bias
            self.object_trans_arch_version.fill_(self.ARCH_VERSION)

    def _closure_weights5(self, weights, gate_logits, bimanual_logit):
        del gate_logits  # v2 has no trainable four-way gate
        pair_mass = torch.sigmoid(bimanual_logit).unsqueeze(-1)
        return torch.cat((weights * (1.0 - pair_mass), pair_mass), dim=-1)

    def _closure_uses_effective_candidates(self):
        return self.bimanual_closure_enabled

    def _init_scan_state(self, context):
        state = super()._init_scan_state(context)
        if self.fk_filter_enabled:
            initial = context['obj_trans_init'][:, None].expand(-1, 2, -1)
            state['v2_fk_filter_pos'] = initial.clone()
            state['v2_fk_filter_vel'] = torch.zeros_like(initial)
            state['v2_fk_filter_active'] = initial.new_zeros(initial.shape[:2])
            prefix = context.get('known_segment_state_prefix')
            if isinstance(prefix, dict):
                known = {}
                for key, dim in (('v2_fk_filter_pos', 6), ('v2_fk_filter_vel', 6), ('v2_fk_filter_active', 2)):
                    values, mask = self._prepare_known_sequence(
                        prefix.get(key), context['batch_size'], context['seq_len'], dim,
                        context['device'], context['dtype'], context.get('known_segment_mask'))
                    known[key] = values
                    known['mask'] = mask if 'mask' not in known else known['mask'] & mask
                context['v2_filter_known'] = known
        return state

    def _fast_known_prefix_supported(self, context, prefix_len):
        """The filter state must be published over the whole prefix.

        Otherwise the filter would evolve from the head's own FK proposals,
        which the batched prefix never computes.
        """
        if not super()._fast_known_prefix_supported(context, prefix_len):
            return False
        if not self.fk_filter_enabled:
            return True
        known = context.get('v2_filter_known')
        return known is not None and bool(known['mask'][:, :prefix_len].all().item())

    def _batched_known_prefix(self, context, state, known_pos, known_weights, prefix_len, **kwargs):
        if self.fk_filter_enabled:
            # On a teacher-forced frame the filter state is overwritten by the
            # published one, so after the prefix it is simply the last one.
            known, last = context['v2_filter_known'], prefix_len - 1
            batch_size = context['batch_size']
            overrides = dict(kwargs.pop('state_overrides', None) or {})
            overrides.update(
                v2_fk_filter_pos=known['v2_fk_filter_pos'][:, last].reshape(batch_size, 2, 3),
                v2_fk_filter_vel=known['v2_fk_filter_vel'][:, last].reshape(batch_size, 2, 3),
                v2_fk_filter_active=known['v2_fk_filter_active'][:, last],
            )
            kwargs['state_overrides'] = overrides
        return super()._batched_known_prefix(context, state, known_pos, known_weights, prefix_len, **kwargs)

    def _predict_step(self, context, time_index, state, **kwargs):
        output, next_state = super()._predict_step(context, time_index, state, **kwargs)
        if not self.fk_filter_enabled:
            return output, next_state
        raw = torch.stack((output['pred_lhand_obj_trans'], output['pred_rhand_obj_trans']), dim=1)
        prob = context['pred_hand_contact_prob'][:, time_index, :2]
        before = state['v2_fk_filter_active'] > .5
        active = torch.where(before, prob >= .3, prob >= .7)
        keep = active & before
        predicted = state['v2_fk_filter_pos'] + state['v2_fk_filter_vel'] / 30.0
        innovation = raw - predicted
        filtered = torch.where(keep[..., None], predicted + self.fk_filter_alpha * innovation, raw)
        velocity = torch.where(keep[..., None], state['v2_fk_filter_vel'] + self.fk_filter_beta * 30.0 * innovation, torch.zeros_like(raw))
        active = active.to(raw.dtype)
        known = context.get('v2_filter_known')
        if known is not None:
            valid = known['mask'][:, time_index]
            filtered = torch.where(valid[:, None, None], known['v2_fk_filter_pos'][:, time_index].reshape_as(raw), filtered)
            velocity = torch.where(valid[:, None, None], known['v2_fk_filter_vel'][:, time_index].reshape_as(raw), velocity)
            active = torch.where(valid[:, None], known['v2_fk_filter_active'][:, time_index], active)
        w = output['gating_weights']
        fused = ((w[:, :2, None] * filtered).sum(1)
                 + w[:, 2:3] * output['pred_imu_obj_trans'] + w[:, 3:4] * output['pred_static_obj_trans'])
        known_pos, mask = kwargs.get('known_obj_trans'), kwargs.get('known_mask')
        if isinstance(known_pos, torch.Tensor) and isinstance(mask, torch.Tensor):
            fused = torch.where(mask[:, None], known_pos.to(fused), fused)
        previous = state['prev_obj_trans']
        next_state.update(prev_obj_trans=fused, prev_obj_delta=fused-previous,
                          v2_fk_filter_pos=filtered, v2_fk_filter_vel=velocity, v2_fk_filter_active=active)
        output.update(pred_obj_trans=fused, pred_lhand_obj_trans=filtered[:, 0], pred_rhand_obj_trans=filtered[:, 1],
                      v2_fk_filter_pos=filtered.flatten(1), v2_fk_filter_vel=velocity.flatten(1), v2_fk_filter_active=active)
        output['pred_obj_vel_from_posdiff'] = (fused-previous)*30 if state['frames_seen'] else torch.zeros_like(fused)
        output['pred_obj_acc_from_posdiff'] = (fused-2*previous+state['prev_prev_obj_trans'])*900 if state['frames_seen'] > 1 else torch.zeros_like(fused)
        return output, next_state

    def _decode_prediction_output(self, prediction_out, contact_prob=None):
        if contact_prob is None:
            raise ValueError("V2ObjectTransModule requires VC contact probabilities")
        l_fk_out, r_fk_out = prediction_out.split((4, 4), dim=-1)
        weights = build_contact_fusion_weights(contact_prob)
        return l_fk_out, r_fk_out, weights, None

    def _build_results(self, context, prediction, enable_refine=True):
        results = super()._build_results(context, prediction, enable_refine=enable_refine)
        results["gating_is_fixed"] = True
        return results

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        output = BaseObjectTransModule.empty_output(batch_size, seq_len, device)
        output.pop("gate_logits", None)
        output["gating_is_fixed"] = True
        return output


ObjectTransModule = V2ObjectTransModule

__all__ = ["V2ObjectTransModule", "ObjectTransModule"]