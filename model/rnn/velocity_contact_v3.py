"""Single-factor v3 VC ablation: boundary + bidirectional obj_move."""
from __future__ import annotations

import torch

from .base import RNNWithInit
from .velocity_contact_ablation import (
    IMUHOIConditionalVelocityContactModule,
)


class V3VelocityContactModule(IMUHOIConditionalVelocityContactModule):
    """Experiment 4 plus a bidirectional object-movement network.

    This intentionally inherits the complete ``imuhoi_cond_boundary`` path.
    The only changed module is ``obj_move_net``: its LSTM is bidirectional.
    Hand/object velocity heads, boundary features, hand-contact network,
    conditional probability factorization, losses, and online behavior are
    otherwise identical to Conditional + boundary.
    """

    # The public name remains v3, but old v3 checkpoints represented a
    # different topology and must not be partially loaded.
    ARCH_VERSION = 32
    requires_hp = True

    def __init__(self, cfg):
        super().__init__(cfg, arch="imuhoi_cond_boundary")
        self.arch = "v3"
        self.requires_hp = True
        self.obj_move_bidirectional = True

        contact_hidden = max(
            int(getattr(cfg, "velocity_hidden_dim", 128)) // 2,
            32,
        )
        dropout = float(getattr(cfg, "velocity_dropout", 0.2))
        del self.obj_move_net
        self.obj_move_net = RNNWithInit(
            n_input=self.obj_move_input_dim,
            n_output=1,
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=True,
            dropout=dropout,
        )
        self.register_buffer(
            "_v3_arch_version",
            torch.tensor(self.ARCH_VERSION, dtype=torch.int64),
        )

    def forward(
        self,
        data_dict: dict,
        hp_out: dict = None,
        velocity_only: bool = False,
    ):
        """Expose the downstream-consumed hand-contact marginal for its BCE."""
        out = super().forward(
            data_dict,
            hp_out=hp_out,
            velocity_only=velocity_only,
        )
        hand_contact_prob = out.get("pred_hand_contact_prob")
        if not velocity_only and isinstance(hand_contact_prob, torch.Tensor):
            out["pred_hand_contact_prob_marginal"] = hand_contact_prob[..., :2]
        return out

    def validate_checkpoint_state_dict(self, state_dict):
        """Reject fixed-lookahead/shared-head and old v3 checkpoints."""
        if not isinstance(state_dict, dict):
            raise ValueError("VC v3 checkpoint must contain a state_dict.")
        normalized = {
            str(key[7:] if str(key).startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        version = normalized.get("_v3_arch_version")
        if (
            not isinstance(version, torch.Tensor)
            or version.numel() != 1
            or int(version.item()) != self.ARCH_VERSION
        ):
            raise ValueError(
                "Checkpoint is not marked as VC v3 boundary+bidir-obj_move; "
                "retrain or select the matching --velocity_contact_arch."
            )
        super().validate_checkpoint_state_dict(normalized)


VelocityContactModule = V3VelocityContactModule

__all__ = ["V3VelocityContactModule", "VelocityContactModule"]
