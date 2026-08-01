"""EgoIMU IMUHOI three-branch ObjectTrans model with current online inference."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

from configs import FRAME_RATE, _SENSOR_NAMES
from .base import RNNWithInit
from .object_trans_legacy import LegacyObjectTransModule


class IMUHOIObjectTransModule(nn.Module):
    """Original left-FK/right-FK/IMU-integration fusion architecture."""

    _slice_optional_time = staticmethod(
        LegacyObjectTransModule._slice_optional_time
    )
    _inference_online_sequence = LegacyObjectTransModule._inference_online_sequence
    inference = LegacyObjectTransModule.inference

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(getattr(cfg, "obj_imu_dim", self.imu_dim))
        self.num_human_imus = int(
            getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        )
        hidden_dim_multiplier = int(getattr(cfg, "hidden_dim_multiplier", 1))

        self.gating_prior_beta = float(getattr(cfg, "gating_prior_beta", 5.0))
        self.gating_temperature = float(
            getattr(cfg, "gating_temperature", 5.0)
        )
        self.gating_smoothing_enabled = bool(
            getattr(cfg, "gating_smoothing_enabled", False)
        )
        self.gating_smoothing_alpha = float(
            getattr(cfg, "gating_smoothing_alpha", 0.6)
        )
        self.gating_max_change = float(
            getattr(cfg, "gating_max_change", 0.25)
        )

        n_fk_branch_input = (
            6 + 3 + 1 + self.obj_imu_dim + self.imu_dim + 3 + 3
        )
        self.lhand_fk_head = RNNWithInit(
            n_input=n_fk_branch_input,
            n_output=4,
            n_hidden=128 * hidden_dim_multiplier,
            n_init=4,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        self.rhand_fk_head = RNNWithInit(
            n_input=n_fk_branch_input,
            n_output=4,
            n_hidden=128 * hidden_dim_multiplier,
            n_init=4,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        self.gating_head = RNNWithInit(
            n_input=9,
            n_output=3,
            n_hidden=64 * hidden_dim_multiplier,
            n_init=6,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=0.2,
        )

    def validate_checkpoint_state_dict(self, state_dict):
        """Reject checkpoints containing a different ObjectTrans topology."""
        if not isinstance(state_dict, dict):
            raise ValueError("IMUHOI ObjectTrans checkpoint must contain a state_dict.")
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
                "Checkpoint is incompatible with the EgoIMU IMUHOI "
                "ObjectTrans architecture (" + "; ".join(details) + ")."
            )

    def online_mode(self) -> str:
        return "window"

    @staticmethod
    def _unit_vector(value: torch.Tensor, eps: float = 1e-8):
        return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _softplus_positive(value: torch.Tensor):
        return F.softplus(value) + 1e-4

    @staticmethod
    def _pad_or_trim_last_dim(value: torch.Tensor, out_dim: int):
        if value.shape[-1] == out_dim:
            return value
        if value.shape[-1] > out_dim:
            return value[..., :out_dim]
        pad = value.new_zeros(*value.shape[:-1], out_dim - value.shape[-1])
        return torch.cat((value, pad), dim=-1)

    def _prepare_obj_imu(
        self, value, batch_size, seq_len, device, dtype
    ):
        if not isinstance(value, torch.Tensor):
            return torch.zeros(
                batch_size,
                seq_len,
                self.obj_imu_dim,
                device=device,
                dtype=dtype,
            )
        value = value.to(device=device, dtype=dtype)
        if value.dim() == 4:
            value = value.reshape(batch_size, seq_len, -1)
        if value.dim() == 2:
            value = value.unsqueeze(0)
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1, -1)
        if value.dim() != 3 or value.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                "obj_imu must preserve the ObjectTrans batch/time dimensions, "
                f"got {tuple(value.shape)}"
            )
        return self._pad_or_trim_last_dim(value, self.obj_imu_dim)

    def _prepare_human_imu(
        self, value, batch_size, seq_len, device, dtype
    ):
        if not isinstance(value, torch.Tensor):
            return torch.zeros(
                batch_size,
                seq_len,
                self.num_human_imus,
                self.imu_dim,
                device=device,
                dtype=dtype,
            )
        value = value.to(device=device, dtype=dtype)
        if value.dim() == 3:
            value = value.reshape(
                batch_size, seq_len, self.num_human_imus, self.imu_dim
            )
        if value.dim() != 4 or value.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                "human_imu must preserve the ObjectTrans batch/time dimensions, "
                f"got {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _rot6d_delta(rot6d: torch.Tensor):
        batch_size, seq_len = rot6d.shape[:2]
        rotm = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(
            batch_size, seq_len, 3, 3
        )
        relative = torch.matmul(
            rotm[:, 1:].transpose(-1, -2), rotm[:, :-1]
        )
        delta = matrix_to_axis_angle(relative.reshape(-1, 3, 3)).reshape(
            batch_size, seq_len - 1, 3
        )
        return F.pad(delta, (0, 0, 1, 0))

    def _compute_init_dir_len(self, hand_pos, obj_rotm, obj_pos):
        vector_world = obj_pos - hand_pos
        length = vector_world.norm(dim=-1, keepdim=True)
        direction_world = self._unit_vector(vector_world)
        direction_object = torch.bmm(
            obj_rotm.transpose(-1, -2), direction_world.unsqueeze(-1)
        ).squeeze(-1)
        return direction_object, length

    @staticmethod
    def _prepare_contact_init(
        contact_init,
        obj_vel_first,
        batch_size,
        device,
        dtype,
    ):
        if isinstance(contact_init, torch.Tensor):
            out = contact_init.to(device=device, dtype=dtype)
            if out.dim() == 1:
                out = out.unsqueeze(0).expand(batch_size, -1)
            if out.shape == (batch_size, 6):
                return out
            if out.shape == (batch_size, 3):
                return torch.cat((out, obj_vel_first), dim=-1)
            if out.shape[-1] > 6:
                return out[..., :6]
            raise ValueError(
                "contact_init must have 3 or 6 features for IMUHOI OT, "
                f"got {tuple(out.shape)}"
            )
        zeros = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        return torch.cat((zeros, obj_vel_first), dim=-1)

    def _smooth_gating_weights(self, weights: torch.Tensor):
        if (
            self.training
            or not self.gating_smoothing_enabled
            or weights.shape[1] < 2
        ):
            return weights
        smoothed = weights.clone()
        previous = weights[:, 0]
        for frame in range(1, weights.shape[1]):
            current = weights[:, frame]
            transition = previous.argmax(-1) != current.argmax(-1)
            blended = (
                self.gating_smoothing_alpha * previous
                + (1.0 - self.gating_smoothing_alpha) * current
            )
            if self.gating_max_change > 0:
                change = blended - previous
                norm = change.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                scale = (self.gating_max_change / norm).clamp_max(1.0)
                blended = previous + change * scale
            frame_weights = torch.where(
                transition.unsqueeze(-1), blended, current
            )
            frame_weights = frame_weights / frame_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            smoothed[:, frame] = frame_weights
            previous = frame_weights
        return smoothed

    def forward(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        obj_imu: torch.Tensor = None,
        human_imu: torch.Tensor = None,
        obj_vel_input: torch.Tensor = None,
        contact_init: torch.Tensor = None,
        has_object_mask: torch.Tensor = None,
        known_obj_trans_prefix: torch.Tensor = None,
        **_,
    ):
        if hand_positions.dim() == 3:
            batch_size, seq_len = hand_positions.shape[:2]
            hand_positions = hand_positions.reshape(
                batch_size, seq_len, 2, 3
            )
        elif hand_positions.dim() == 4:
            batch_size, seq_len = hand_positions.shape[:2]
        else:
            raise ValueError(
                f"Unexpected hand_positions shape {tuple(hand_positions.shape)}"
            )
        device = hand_positions.device
        dtype = hand_positions.dtype
        left_hand = hand_positions[:, :, 0]
        right_hand = hand_positions[:, :, 1]

        obj_imu = self._prepare_obj_imu(
            obj_imu, batch_size, seq_len, device, dtype
        )
        human_imu = self._prepare_human_imu(
            human_imu, batch_size, seq_len, device, dtype
        )
        if obj_vel_input is None:
            obj_vel_input = torch.zeros(
                batch_size, seq_len, 3, device=device, dtype=dtype
            )
        else:
            obj_vel_input = obj_vel_input.to(device=device, dtype=dtype)

        obj_rot6d = obj_imu[:, :, 3:9]
        obj_rotm = rotation_6d_to_matrix(
            obj_rot6d.reshape(-1, 6)
        ).reshape(batch_size, seq_len, 3, 3)
        obj_rot_delta = self._rot6d_delta(obj_rot6d)
        left_imu = human_imu[
            :, :, _SENSOR_NAMES.index("LeftForeArm"), :
        ]
        right_imu = human_imu[
            :, :, _SENSOR_NAMES.index("RightForeArm"), :
        ]

        p_left = pred_hand_contact_prob[:, :, 0:1]
        p_right = pred_hand_contact_prob[:, :, 1:2]
        left_input = torch.cat(
            (
                obj_rot6d,
                left_hand,
                p_left,
                obj_imu,
                left_imu,
                obj_vel_input,
                obj_rot_delta,
            ),
            dim=-1,
        )
        right_input = torch.cat(
            (
                obj_rot6d,
                right_hand,
                p_right,
                obj_imu,
                right_imu,
                obj_vel_input,
                obj_rot_delta,
            ),
            dim=-1,
        )

        left_dir0, left_len0 = self._compute_init_dir_len(
            left_hand[:, 0], obj_rotm[:, 0], obj_trans_init
        )
        right_dir0, right_len0 = self._compute_init_dir_len(
            right_hand[:, 0], obj_rotm[:, 0], obj_trans_init
        )
        left_out = self.lhand_fk_head(
            (left_input, torch.cat((left_dir0, left_len0), dim=-1))
        )
        right_out = self.rhand_fk_head(
            (right_input, torch.cat((right_dir0, right_len0), dim=-1))
        )
        left_dir = self._unit_vector(left_out[:, :, :3])
        right_dir = self._unit_vector(right_out[:, :, :3])
        left_len = self._softplus_positive(left_out[:, :, 3])
        right_len = self._softplus_positive(right_out[:, :, 3])

        flat_rotm = obj_rotm.reshape(batch_size * seq_len, 3, 3)
        left_world = torch.bmm(
            flat_rotm,
            left_dir.reshape(batch_size * seq_len, 3, 1),
        ).reshape(batch_size, seq_len, 3)
        right_world = torch.bmm(
            flat_rotm,
            right_dir.reshape(batch_size * seq_len, 3, 1),
        ).reshape(batch_size, seq_len, 3)
        left_fk = left_hand + left_world * left_len.unsqueeze(-1)
        right_fk = right_hand + right_world * right_len.unsqueeze(-1)

        gating_input = torch.cat(
            (
                pred_hand_contact_prob,
                obj_vel_input,
                obj_imu[:, :, :3],
            ),
            dim=-1,
        )
        gating_init = self._prepare_contact_init(
            contact_init,
            obj_vel_input[:, 0],
            batch_size,
            device,
            dtype,
        )
        gate_logits = self.gating_head((gating_input, gating_init))
        prior_imu = 1.0 - torch.maximum(
            p_left.squeeze(-1), p_right.squeeze(-1)
        )
        prior = torch.stack(
            (p_left.squeeze(-1), p_right.squeeze(-1), prior_imu),
            dim=-1,
        )
        gate_logits_with_prior = (
            gate_logits
            + self.gating_prior_beta * torch.log(prior.clamp_min(1e-6))
        )
        weights_raw = F.softmax(
            gate_logits_with_prior / self.gating_temperature,
            dim=-1,
        )
        weights = self._smooth_gating_weights(weights_raw)

        prefix = None
        prefix_len = 0
        if isinstance(known_obj_trans_prefix, torch.Tensor):
            prefix = known_obj_trans_prefix.to(device=device, dtype=dtype)
            if prefix.dim() == 2:
                prefix = prefix.unsqueeze(1)
            if prefix.dim() == 3 and prefix.shape[0] == batch_size:
                prefix_len = min(prefix.shape[1], seq_len)

        fused_steps = []
        imu_steps = []
        dt = 1.0 / FRAME_RATE
        for frame in range(seq_len):
            if frame < prefix_len:
                fused_steps.append(prefix[:, frame])
                imu_steps.append(prefix[:, frame])
                continue
            previous = fused_steps[-1] if frame > 0 else obj_trans_init
            imu_position = previous + obj_vel_input[:, frame] * dt
            imu_steps.append(imu_position)
            fused_steps.append(
                weights[:, frame, 0:1] * left_fk[:, frame]
                + weights[:, frame, 1:2] * right_fk[:, frame]
                + weights[:, frame, 2:3] * imu_position
            )
        fused_pos = torch.stack(fused_steps, dim=1)
        imu_pos = torch.stack(imu_steps, dim=1)

        velocity = torch.zeros_like(fused_pos)
        acceleration = torch.zeros_like(fused_pos)
        if seq_len > 1:
            velocity = torch.cat(
                (
                    velocity[:, :1],
                    (fused_pos[:, 1:] - fused_pos[:, :-1]) * FRAME_RATE,
                ),
                dim=1,
            )
        if seq_len > 2:
            acceleration = torch.cat(
                (
                    acceleration[:, :2],
                    (
                        fused_pos[:, 2:]
                        - 2.0 * fused_pos[:, 1:-1]
                        + fused_pos[:, :-2]
                    )
                    * (FRAME_RATE**2),
                ),
                dim=1,
            )

        if has_object_mask is not None:
            mask = has_object_mask.to(
                device=device, dtype=dtype
            ).reshape(batch_size, 1, 1)
            scalar_mask = mask.squeeze(-1)
            batch_mask = mask.reshape(batch_size)
            fused_pos = fused_pos * mask
            imu_pos = imu_pos * mask
            velocity = velocity * mask
            acceleration = acceleration * mask
            weights = weights * mask
            weights_raw = weights_raw * mask
            gate_logits = gate_logits * mask
            left_fk = left_fk * mask
            right_fk = right_fk * mask
            left_dir = left_dir * mask
            right_dir = right_dir * mask
            left_len = left_len * scalar_mask
            right_len = right_len * scalar_mask
            left_dir0 = left_dir0 * scalar_mask
            right_dir0 = right_dir0 * scalar_mask
            left_len0 = left_len0.squeeze(-1) * batch_mask
            right_len0 = right_len0.squeeze(-1) * batch_mask
        else:
            left_len0 = left_len0.squeeze(-1)
            right_len0 = right_len0.squeeze(-1)

        return {
            "pred_obj_trans": fused_pos,
            "gating_weights": weights,
            "gating_weights_raw": weights_raw,
            "gate_logits": gate_logits,
            "gate_logits_with_prior": gate_logits_with_prior,
            "gating_contact_prob": pred_hand_contact_prob,
            "pred_obj_vel_from_posdiff": velocity,
            "pred_obj_acc_from_posdiff": acceleration,
            "obj_vel_input": obj_vel_input,
            "obj_vel_corrected": obj_vel_input,
            "pred_lhand_obj_direction": left_dir,
            "pred_rhand_obj_direction": right_dir,
            "pred_lhand_lb": left_len,
            "pred_rhand_lb": right_len,
            "pred_lhand_obj_trans": left_fk,
            "pred_rhand_obj_trans": right_fk,
            "pred_imu_obj_trans": imu_pos,
            "init_lhand_oe_ho": left_dir0,
            "init_rhand_oe_ho": right_dir0,
            "init_lhand_lb": left_len0,
            "init_rhand_lb": right_len0,
            "gating_smoothing_applied": (
                not self.training and self.gating_smoothing_enabled
            ),
        }

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_scalar = torch.zeros(batch_size, seq_len, device=device)
        zeros_weights = torch.zeros(batch_size, seq_len, 3, device=device)
        return {
            "pred_obj_trans": zeros_pos.clone(),
            "gating_weights": zeros_weights.clone(),
            "gating_weights_raw": zeros_weights.clone(),
            "gate_logits": zeros_weights.clone(),
            "gate_logits_with_prior": zeros_weights.clone(),
            "pred_obj_vel_from_posdiff": zeros_pos.clone(),
            "pred_obj_acc_from_posdiff": zeros_pos.clone(),
            "obj_vel_input": zeros_pos.clone(),
            "obj_vel_corrected": zeros_pos.clone(),
            "pred_lhand_obj_direction": zeros_pos.clone(),
            "pred_rhand_obj_direction": zeros_pos.clone(),
            "pred_lhand_lb": zeros_scalar.clone(),
            "pred_rhand_lb": zeros_scalar.clone(),
            "pred_lhand_obj_trans": zeros_pos.clone(),
            "pred_rhand_obj_trans": zeros_pos.clone(),
            "pred_imu_obj_trans": zeros_pos.clone(),
            "gating_contact_prob": zeros_weights.clone(),
            "init_lhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_rhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_lhand_lb": torch.zeros(batch_size, device=device),
            "init_rhand_lb": torch.zeros(batch_size, device=device),
            "gating_smoothing_applied": False,
        }


ObjectTransModule = IMUHOIObjectTransModule

__all__ = ["IMUHOIObjectTransModule", "ObjectTransModule"]
