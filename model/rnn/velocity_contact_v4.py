"""Single-factor v4 VC ablation: v3 plus hand-object geometry features."""
from __future__ import annotations

import torch

from .base import RNNWithInit
from .velocity_contact_ablation import IMUHOIConditionalVelocityContactModule


class V4VelocityContactModule(IMUHOIConditionalVelocityContactModule):
    """Experiment 4 with v3's bidirectional object-movement head.

    v4 is v3 plus the existing contact geometry features. Boundary encoding,
    conditional contact factorization, velocity heads, losses, and online
    inference remain unchanged.
    """

    ARCH_VERSION = 41
    requires_hp = True

    def __init__(self, cfg):
        super().__init__(cfg, arch="imuhoi_cond_boundary")
        self.arch = "v4"
        self.requires_hp = True
        self.use_contact_geom_features = True
        self.obj_move_bidirectional = True

        hidden_dim = int(getattr(cfg, "velocity_hidden_dim", 128))
        contact_hidden = max(hidden_dim // 2, 32)
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

        self.hand_contact_input_dim = (
            self.base_contact_input_dim
            + self.boundary_hidden_dim
            + self.contact_geom_dim
        )
        del self.hand_contact_net
        self.hand_contact_net = RNNWithInit(
            n_input=self.hand_contact_input_dim,
            n_output=2,
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )

        self.register_buffer(
            "_v4_arch_version",
            torch.tensor(self.ARCH_VERSION, dtype=torch.int64),
        )

    def validate_checkpoint_state_dict(self, state_dict):
        """Reject v3/geometry-free checkpoints for the v4 topology."""
        if not isinstance(state_dict, dict):
            raise ValueError("VC v4 checkpoint must contain a state_dict.")
        normalized = {
            str(key[7:] if str(key).startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        version = normalized.get("_v4_arch_version")
        if (
            not isinstance(version, torch.Tensor)
            or version.numel() != 1
            or int(version.item()) != self.ARCH_VERSION
        ):
            raise ValueError(
                "Checkpoint is not marked as VC v4 boundary+geometry+bidir-"
                "obj_move; retrain or select the matching --velocity_contact_arch."
            )
        IMUHOIConditionalVelocityContactModule.validate_checkpoint_state_dict(
            self,
            normalized,
        )


VelocityContactModule = V4VelocityContactModule

__all__ = ["V4VelocityContactModule", "VelocityContactModule"]
