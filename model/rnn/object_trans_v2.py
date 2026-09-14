"""ObjectTrans v2 with fixed VC-driven fusion weights."""
import math

import torch

from .base import RNNWithInit
from .object_trans import ObjectTransModule as BaseObjectTransModule
from utils.object_fusion import build_contact_fusion_weights


class V2ObjectTransModule(BaseObjectTransModule):
    """Current ObjectTrans pipeline without a trainable fusion head."""

    ARCH_VERSION = 2

    def __init__(self, cfg):
        super().__init__(cfg)

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