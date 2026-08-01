"""EgoIMU IMUHOI velocity/contact architecture with the current online API."""
from __future__ import annotations

import torch
import torch.nn as nn

from configs import _SENSOR_NAMES
from .base import RNNWithInit
from .velocity_contact import VelocityContactModule as CurrentVelocityContactModule


class IMUHOIVelocityContactModule(nn.Module):
    """Original IMUHOI VC network.

    Hand velocity is predicted from all human IMUs, object velocity has an
    independent predictor, and the three contact channels are predicted
    jointly from hand/object IMUs plus the predicted velocities.  Human-pose
    outputs are intentionally ignored.
    """

    _slice_hp_out = CurrentVelocityContactModule._slice_hp_out
    _inference_online_sequence = CurrentVelocityContactModule._inference_online_sequence
    inference = CurrentVelocityContactModule.inference

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_human_imus = int(
            getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        )
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(getattr(cfg, "obj_imu_dim", 9))
        hidden_dim = int(getattr(cfg, "velocity_hidden_dim", 128))
        num_layers = int(getattr(cfg, "velocity_num_layers", 2))
        dropout = float(getattr(cfg, "velocity_dropout", 0.2))

        self.hand_vel_net = RNNWithInit(
            n_input=self.num_human_imus * self.imu_dim,
            n_output=6,
            n_hidden=hidden_dim,
            n_init=6,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )
        self.obj_vel_net = RNNWithInit(
            n_input=self.obj_imu_dim,
            n_output=3,
            n_hidden=hidden_dim,
            n_init=3,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        contact_input_dim = 2 * self.imu_dim + self.obj_imu_dim + 6 + 3
        contact_hidden = max(hidden_dim // 2, 32)
        self.contact_net = RNNWithInit(
            n_input=contact_input_dim,
            n_output=3,
            n_hidden=contact_hidden,
            n_init=6,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )

        self.left_hand_sensor = _SENSOR_NAMES.index("LeftForeArm")
        self.right_hand_sensor = _SENSOR_NAMES.index("RightForeArm")

    def validate_checkpoint_state_dict(self, state_dict):
        """Reject checkpoints from the current/legacy VC architectures."""
        if not isinstance(state_dict, dict):
            raise ValueError("IMUHOI VC checkpoint must contain a state_dict.")
        normalized = {
            key[7:] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
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
                "Checkpoint is incompatible with the EgoIMU IMUHOI velocity/"
                "contact architecture (" + "; ".join(details) + "). "
                "Pass an IMUHOI VC checkpoint via --vc_ckpt."
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

    @staticmethod
    def _prepare_contact_init(
        contact_init,
        obj_vel_init,
        batch_size,
        device,
        dtype,
    ):
        """Build the original six-dimensional contact RNN initializer.

        The current dataset contract stores the three contact states only.
        EgoIMU initializes the remaining three dimensions with object velocity.
        Existing six-dimensional checkpoints/inputs remain accepted.
        """
        if isinstance(contact_init, torch.Tensor):
            out = contact_init.to(device=device, dtype=dtype)
            if out.dim() == 1:
                out = out.unsqueeze(0).expand(batch_size, -1)
            if out.shape == (batch_size, 6):
                return out
            if out.shape == (batch_size, 3):
                return torch.cat((out, obj_vel_init), dim=-1)
            if out.shape[-1] > 6:
                return out[..., :6]
            raise ValueError(
                "contact_init must have 3 or 6 features for IMUHOI VC, "
                f"got {tuple(out.shape)}"
            )
        zeros = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        return torch.cat((zeros, obj_vel_init), dim=-1)

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
                "human_imu sensor shape does not match the configured IMUHOI VC: "
                f"expected {(self.num_human_imus, self.imu_dim)}, "
                f"got {tuple(human_imu.shape[2:])}"
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
        )
        obj_vel_init = self._prepare_init(
            data_dict.get("obj_vel_init"),
            batch_size,
            3,
            device,
            dtype,
            "obj_vel_init",
        )

        # Match EgoIMU exactly: human IMUs are converted back to their
        # world/root-composed representation before the human velocity heads.
        human_imu_denorm = CurrentVelocityContactModule._denormalize_imu(
            self, human_imu
        )
        human_imu_flat = human_imu_denorm.reshape(batch_size, seq_len, -1)
        hand_vel_flat = self.hand_vel_net((human_imu_flat, hand_vel_init))
        hand_vel = hand_vel_flat.view(batch_size, seq_len, 2, 3)
        obj_vel = self.obj_vel_net((obj_imu, obj_vel_init))

        if velocity_only:
            return {
                "pred_hand_glb_vel": hand_vel,
                "pred_hand_glb_vel_flat": hand_vel_flat,
                "pred_obj_vel": obj_vel,
            }

        hand_imu_feat = human_imu_denorm[
            :,
            :,
            [self.left_hand_sensor, self.right_hand_sensor],
            :,
        ].reshape(batch_size, seq_len, -1)
        contact_input = torch.cat(
            (hand_imu_feat, obj_imu, hand_vel_flat, obj_vel),
            dim=-1,
        )
        contact_init = self._prepare_contact_init(
            data_dict.get("contact_init"),
            obj_vel_init,
            batch_size,
            device,
            dtype,
        )
        contact_logits = self.contact_net((contact_input, contact_init))
        contact_prob = torch.sigmoid(contact_logits)

        return {
            "pred_hand_glb_vel": hand_vel,
            "pred_hand_glb_vel_flat": hand_vel_flat,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": contact_logits,
            "pred_hand_contact_prob": contact_prob,
        }

    @staticmethod
    def _apply_segment_contact(output, contact_init=None):
        # The original IMUHOI VC has no endpoint branch to decode.
        del contact_init
        return output

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
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
        }


VelocityContactModule = IMUHOIVelocityContactModule

__all__ = ["IMUHOIVelocityContactModule", "VelocityContactModule"]
