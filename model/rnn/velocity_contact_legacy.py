"""Legacy velocity/contact predictor without the interaction-boundary branch."""
from __future__ import annotations

import torch
import torch.nn as nn

from configs import FRAME_RATE, _SENSOR_NAMES
from .base import RNNWithInit
from .velocity_contact import VelocityContactModule


class LegacyVelocityContactModule(nn.Module):
    """Original VC architecture with IMU-only hand-contact prediction.

    The online inference protocol and tensor preparation helpers are shared
    with the current implementation, but this class has its own parameter
    initialization and forward path.  In particular, it never creates the
    current model's body-group encoders, boundary RNN, or boundary head.
    """

    _REMOVED_MERGED_CONTACT_PREFIXES = VelocityContactModule._REMOVED_MERGED_CONTACT_PREFIXES

    # Reuse only architecture-independent helpers and the online protocol.
    # These methods dispatch to this class's ``forward`` implementation and do
    # not access any boundary-specific attributes.
    _denormalize_imu = VelocityContactModule._denormalize_imu
    _prepare_init_vector = VelocityContactModule._prepare_init_vector
    _slice_hp_out = VelocityContactModule._slice_hp_out
    _inference_online_sequence = VelocityContactModule._inference_online_sequence
    inference = VelocityContactModule.inference
    should_adapt_checkpoint_hand_contact_input = (
        VelocityContactModule.should_adapt_checkpoint_hand_contact_input
    )
    enable_legacy_hand_contact_input = VelocityContactModule.enable_legacy_hand_contact_input
    validate_checkpoint_state_dict = VelocityContactModule.validate_checkpoint_state_dict

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.obj_imu_dim = getattr(cfg, "obj_imu_dim", self.imu_dim)
        if self.obj_imu_dim != self.imu_dim:
            raise ValueError("Shared velocity predictor requires obj_imu_dim to match imu_dim.")
        hidden_dim = getattr(cfg, "velocity_hidden_dim", 128)
        num_layers = getattr(cfg, "velocity_num_layers", 2)
        dropout = getattr(cfg, "velocity_dropout", 0.2)
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.adapt_checkpoint_hand_contact_input = bool(
            getattr(cfg, "velocity_contact_adapt_checkpoint_input", False)
        )

        self.vel_net = RNNWithInit(
            n_input=self.imu_dim,
            n_output=3,
            n_hidden=hidden_dim,
            n_init=3,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        contact_input_dim = 2 * self.imu_dim + self.obj_imu_dim
        contact_hidden = max(hidden_dim // 2, 32)
        self.obj_move_net = RNNWithInit(
            n_input=contact_input_dim,
            n_output=1,
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )
        self.legacy_hand_contact_input_dim = contact_input_dim
        self.base_hand_contact_input_dim = contact_input_dim
        self.geom_hand_contact_input_dim = contact_input_dim
        self.contact_geom_dim = 0
        self.hand_contact_input_mode = "legacy"
        self.hand_contact_net = RNNWithInit(
            n_input=contact_input_dim,
            n_output=2,
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )

        self.left_hand_sensor = _SENSOR_NAMES.index("LeftForeArm")
        self.right_hand_sensor = _SENSOR_NAMES.index("RightForeArm")

    def forward(self, data_dict: dict, hp_out: dict = None):
        """Predict velocity and contact outputs from IMU streams only."""
        del hp_out
        human_imu = data_dict["human_imu"]
        obj_imu = data_dict.get("obj_imu")
        hand_vel_init = data_dict["hand_vel_glb_init"]
        obj_vel_init = data_dict["obj_vel_init"]
        contact_init = data_dict.get("contact_init")
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B, T, num_imu, imu_dim], got {human_imu.shape}")
        batch_size, seq_len, _, _ = human_imu.shape
        device = human_imu.device
        dtype = human_imu.dtype

        if obj_imu is None:
            obj_imu = torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        elif obj_imu.dim() == 4:
            obj_imu = obj_imu.reshape(batch_size, seq_len, -1)
        elif obj_imu.dim() != 3:
            raise ValueError(f"obj_imu must be [B, T, obj_dim], got {obj_imu.shape}")
        if obj_imu.shape[-1] != self.imu_dim:
            raise ValueError(
                f"Shared velocity net expects obj_imu feature dim {self.imu_dim}, "
                f"got {obj_imu.shape[-1]}"
            )

        if hand_vel_init.dim() != 3 or hand_vel_init.shape[1:] != (2, 3):
            raise ValueError(f"hand_vel_init must be [B,2,3], got {hand_vel_init.shape}")
        l_hand_vel_init = hand_vel_init[:, 0, :]
        r_hand_vel_init = hand_vel_init[:, 1, :]

        if obj_vel_init.dim() == 1:
            obj_vel_init_vec = obj_vel_init.unsqueeze(0).expand(batch_size, -1)
        elif obj_vel_init.dim() == 2:
            obj_vel_init_vec = obj_vel_init
        else:
            raise ValueError(f"obj_vel_init must be [B,3] or [3], got {obj_vel_init.shape}")

        human_imu_denorm = self._denormalize_imu(human_imu)
        l_hand_imu = human_imu_denorm[:, :, self.left_hand_sensor, :]
        r_hand_imu = human_imu_denorm[:, :, self.right_hand_sensor, :]
        l_hand_vel = self.vel_net((l_hand_imu, l_hand_vel_init))
        r_hand_vel = self.vel_net((r_hand_imu, r_hand_vel_init))
        hand_vel = torch.stack((l_hand_vel, r_hand_vel), dim=2)
        obj_vel = self.vel_net((obj_imu, obj_vel_init_vec))

        hand_imu_feat = human_imu_denorm[:, :, [self.left_hand_sensor, self.right_hand_sensor], :].reshape(
            batch_size, seq_len, -1
        )
        contact_input = torch.cat((hand_imu_feat, obj_imu), dim=-1)

        if contact_init is None:
            contact_init_vec = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        elif contact_init.dim() == 1:
            contact_init_vec = contact_init.unsqueeze(0).expand(batch_size, -1)
        else:
            contact_init_vec = contact_init
        if contact_init is not None and contact_init_vec.shape[-1] > 3:
            contact_init_vec = contact_init_vec[..., :3]
        if contact_init_vec.shape[-1] != 3:
            raise ValueError(f"contact_init must have last dim 3, got {contact_init_vec.shape}")

        obj_move_logits = self.obj_move_net((contact_input, contact_init_vec))
        obj_move_prob = torch.sigmoid(obj_move_logits)
        hand_contact_logits = self.hand_contact_net((contact_input, contact_init_vec))
        hand_contact_prob_cond = torch.sigmoid(hand_contact_logits)
        hand_contact_prob = obj_move_prob * hand_contact_prob_cond

        return {
            "pred_hand_glb_vel": hand_vel,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": torch.cat((hand_contact_logits, obj_move_logits), dim=-1),
            "pred_hand_contact_prob": torch.cat((hand_contact_prob, obj_move_prob), dim=-1),
            "pred_obj_move_prob": obj_move_prob,
            "pred_hand_contact_prob_cond": hand_contact_prob_cond,
            "pred_hand_contact_logits_cond": hand_contact_logits,
        }

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        return {
            "pred_hand_glb_vel": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "pred_obj_vel": zeros_pos.clone(),
            "pred_hand_contact_logits": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_hand_contact_prob": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_move_prob": torch.zeros(batch_size, seq_len, 1, device=device),
            "pred_hand_contact_prob_cond": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_hand_contact_logits_cond": torch.zeros(batch_size, seq_len, 2, device=device),
        }


# Keep the module-level import pattern parallel to the other RNN architecture
# files while exposing the explicit class name for architecture selection.
VelocityContactModule = LegacyVelocityContactModule

__all__ = ["LegacyVelocityContactModule", "VelocityContactModule"]
