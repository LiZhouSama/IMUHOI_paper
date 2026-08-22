"""Causal velocity-first VelocityContact v2 architecture."""
from __future__ import annotations

import torch
import torch.nn as nn

from configs import _SENSOR_NAMES
from .base import RNNWithInit
from .velocity_contact import VelocityContactModule as CurrentVelocityContactModule


class V2VelocityContactModule(nn.Module):
    """Predict contact through a causal velocity -> object -> hand cascade.

    The shared legacy velocity predictor first estimates both hand velocities
    and object velocity.  A causal object-move RNN consumes those nine velocity
    values.  A second causal RNN consumes the same velocities plus the object
    probability and directly predicts left/right hand-contact probabilities.

    HumanPose outputs are intentionally not used by VC v2.  They remain part of
    the complete HP/VC/OT pipeline and are consumed by ObjectTrans.
    """

    ARCH_VERSION = 2
    requires_hp = False

    _slice_hp_out = CurrentVelocityContactModule._slice_hp_out
    _inference_online_sequence = (
        CurrentVelocityContactModule._inference_online_sequence
    )
    inference = CurrentVelocityContactModule.inference
    _denormalize_imu = CurrentVelocityContactModule._denormalize_imu

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_human_imus = int(
            getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        )
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(getattr(cfg, "obj_imu_dim", self.imu_dim))
        if self.obj_imu_dim != self.imu_dim:
            raise ValueError(
                "V2 shared velocity predictor requires obj_imu_dim to match "
                "imu_dim."
            )

        hidden_dim = int(getattr(cfg, "velocity_hidden_dim", 128))
        num_layers = int(getattr(cfg, "velocity_num_layers", 2))
        dropout = float(getattr(cfg, "velocity_dropout", 0.2))
        contact_hidden = max(hidden_dim // 2, 32)

        self.vel_net = RNNWithInit(
            n_input=self.imu_dim,
            n_output=3,
            n_hidden=hidden_dim,
            n_init=3,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        self.velocity_feature_dim = 3 + 3 + 3
        self.obj_move_net = RNNWithInit(
            n_input=self.velocity_feature_dim,
            n_output=1,
            n_hidden=contact_hidden,
            n_init=1,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )
        self.hand_contact_input_dim = self.velocity_feature_dim + 1
        self.hand_contact_net = RNNWithInit(
            n_input=self.hand_contact_input_dim,
            n_output=2,
            n_hidden=contact_hidden,
            n_init=2,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )

        self.left_hand_sensor = _SENSOR_NAMES.index("LeftForeArm")
        self.right_hand_sensor = _SENSOR_NAMES.index("RightForeArm")
        self.register_buffer(
            "_v2_arch_version",
            torch.tensor(self.ARCH_VERSION, dtype=torch.int64),
        )

    @staticmethod
    def _prepare_init(value, batch_size, dim, device, dtype, name):
        if not isinstance(value, torch.Tensor):
            return torch.zeros(batch_size, dim, device=device, dtype=dtype)
        out = value.to(device=device, dtype=dtype)
        if out.dim() == 1:
            out = out.unsqueeze(0).expand(batch_size, -1)
        if out.dim() == 3:
            out = out.reshape(batch_size, -1)
        if out.shape != (batch_size, dim):
            raise ValueError(
                f"{name} must resolve to [B,{dim}], got {tuple(out.shape)}"
            )
        return out

    def validate_checkpoint_state_dict(self, state_dict):
        """Reject checkpoints from other VC topologies before non-strict load."""
        if not isinstance(state_dict, dict):
            raise ValueError("VC v2 checkpoint must contain a state_dict.")
        normalized = {
            key[7:] if str(key).startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        version = normalized.get("_v2_arch_version")
        if (
            not isinstance(version, torch.Tensor)
            or version.numel() != 1
            or int(version.item()) != self.ARCH_VERSION
        ):
            raise ValueError(
                "Checkpoint is not marked as VelocityContact v2; retrain or "
                "select the matching --velocity_contact_arch."
            )
        expected = self.state_dict()
        missing = [key for key in expected if key not in normalized]
        mismatched = [
            key
            for key, value in expected.items()
            if key in normalized
            and getattr(normalized[key], "shape", None) != value.shape
        ]
        if missing or mismatched:
            details = []
            if missing:
                details.append(f"missing keys: {missing[:3]}")
            if mismatched:
                details.append(f"shape mismatches: {mismatched[:3]}")
            raise ValueError(
                "Checkpoint is incompatible with VelocityContact v2 ("
                + "; ".join(details)
                + ")."
            )

    def forward(
        self,
        data_dict: dict,
        hp_out: dict = None,
        velocity_only: bool = False,
    ):
        del hp_out
        human_imu = data_dict["human_imu"]
        obj_imu = data_dict.get("obj_imu")
        if human_imu.dim() != 4:
            raise ValueError(
                "human_imu must be [B,T,num_imu,imu_dim], "
                f"got {tuple(human_imu.shape)}"
            )
        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype
        if human_imu.shape[2:] != (self.num_human_imus, self.imu_dim):
            raise ValueError(
                "human_imu sensor shape does not match VC v2: expected "
                f"{(self.num_human_imus, self.imu_dim)}, got "
                f"{tuple(human_imu.shape[2:])}"
            )

        if obj_imu is None:
            obj_imu = torch.zeros(
                batch_size,
                seq_len,
                self.obj_imu_dim,
                device=device,
                dtype=dtype,
            )
        else:
            obj_imu = obj_imu.to(device=device, dtype=dtype)
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(batch_size, seq_len, -1)
            if obj_imu.shape != (batch_size, seq_len, self.obj_imu_dim):
                raise ValueError(
                    f"obj_imu must be [B,T,{self.obj_imu_dim}], "
                    f"got {tuple(obj_imu.shape)}"
                )

        hand_vel_init = self._prepare_init(
            data_dict.get("hand_vel_glb_init"),
            batch_size,
            6,
            device,
            dtype,
            "hand_vel_glb_init",
        ).reshape(batch_size, 2, 3)
        obj_vel_init = self._prepare_init(
            data_dict.get("obj_vel_init"),
            batch_size,
            3,
            device,
            dtype,
            "obj_vel_init",
        )
        contact_init = self._prepare_init(
            data_dict.get("contact_init"),
            batch_size,
            3,
            device,
            dtype,
            "contact_init",
        )

        human_imu_denorm = self._denormalize_imu(human_imu)
        left_imu = human_imu_denorm[:, :, self.left_hand_sensor, :]
        right_imu = human_imu_denorm[:, :, self.right_hand_sensor, :]
        left_vel = self.vel_net((left_imu, hand_vel_init[:, 0]))
        right_vel = self.vel_net((right_imu, hand_vel_init[:, 1]))
        hand_vel = torch.stack((left_vel, right_vel), dim=2)
        obj_vel = self.vel_net((obj_imu, obj_vel_init))

        velocity_features = torch.cat((left_vel, right_vel, obj_vel), dim=-1)
        output = {
            "pred_hand_glb_vel": hand_vel,
            "pred_hand_glb_vel_flat": velocity_features[..., :6],
            "pred_obj_vel": obj_vel,
        }
        if velocity_only:
            return output

        obj_move_logits = self.obj_move_net(
            (velocity_features, contact_init[:, 2:3])
        )
        obj_move_prob = torch.sigmoid(obj_move_logits)
        hand_contact_input = torch.cat(
            (velocity_features, obj_move_prob), dim=-1
        )
        hand_contact_logits = self.hand_contact_net(
            (hand_contact_input, contact_init[:, :2])
        )
        hand_contact_prob = torch.sigmoid(hand_contact_logits)

        output.update(
            {
                "pred_hand_contact_logits": torch.cat(
                    (hand_contact_logits, obj_move_logits), dim=-1
                ),
                "pred_hand_contact_prob": torch.cat(
                    (hand_contact_prob, obj_move_prob), dim=-1
                ),
                "pred_hand_contact_logits_direct": hand_contact_logits,
                "pred_hand_contact_prob_direct": hand_contact_prob,
                "pred_obj_move_logits": obj_move_logits,
                "pred_obj_move_prob": obj_move_prob,
            }
        )
        return output

    @staticmethod
    def _apply_segment_contact(output, contact_init=None):
        del contact_init
        return output

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_hands = torch.zeros(batch_size, seq_len, 2, device=device)
        return {
            "pred_hand_glb_vel": torch.zeros(
                batch_size, seq_len, 2, 3, device=device
            ),
            "pred_hand_glb_vel_flat": torch.zeros(
                batch_size, seq_len, 6, device=device
            ),
            "pred_obj_vel": zeros_pos.clone(),
            "pred_hand_contact_logits": zeros_pos.clone(),
            "pred_hand_contact_prob": zeros_pos.clone(),
            "pred_hand_contact_logits_direct": zeros_hands.clone(),
            "pred_hand_contact_prob_direct": zeros_hands.clone(),
            "pred_obj_move_logits": torch.zeros(
                batch_size, seq_len, 1, device=device
            ),
            "pred_obj_move_prob": torch.zeros(
                batch_size, seq_len, 1, device=device
            ),
        }


VelocityContactModule = V2VelocityContactModule

__all__ = ["V2VelocityContactModule", "VelocityContactModule"]
