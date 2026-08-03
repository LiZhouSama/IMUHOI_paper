"""Losses used by the original EgoIMU IMUHOI VC and ObjectTrans models."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from configs import _SENSOR_POS_INDICES
from utils.rotation_conversions import rotation_6d_to_matrix
from utils.human_pose import select_wrist_positions


def _masked_mse(pred, target, mask, zero):
    if mask is None:
        return F.mse_loss(pred, target)
    mask = mask.bool()
    if not mask.any():
        return zero.clone()
    return F.mse_loss(pred[mask], target[mask])


def _masked_mean(values, mask, zero):
    if mask is None:
        return values.mean()
    mask = mask.bool()
    if not mask.any():
        return zero.clone()
    return values[mask].mean()


def _has_object_mask(batch, batch_size, seq_len, device):
    value = batch.get("has_object")
    if value is None:
        return torch.ones(
            batch_size, seq_len, device=device, dtype=torch.bool
        )
    value = torch.as_tensor(value, device=device, dtype=torch.bool)
    if value.dim() == 0:
        value = value.reshape(1)
    if value.dim() == 1:
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size)
        return value.reshape(batch_size, 1).expand(batch_size, seq_len)
    if value.shape[:2] != (batch_size, seq_len):
        value = value.reshape(batch_size, -1)[:, :1]
        return value.expand(batch_size, seq_len)
    return value


class IMUHOIVelocityContactLoss:
    """Original MSE velocity losses plus joint three-channel contact BCE."""

    LOSS_KEYS = {"obj_vel", "hand_vel", "hand_contact"}
    TEST_LOSS_KEYS = set(LOSS_KEYS)
    DEFAULT_WEIGHTS = {
        "obj_vel": 1.0,
        "hand_vel": 1.0,
        "hand_contact": 1.0,
    }

    def __init__(self, weights=None):
        self.weights = weights or {}

    def __call__(
        self,
        pred_dict,
        batch,
        device,
        active_loss_keys=None,
    ):
        return self.compute_loss(
            pred_dict,
            batch,
            device,
            active_loss_keys=active_loss_keys,
        )

    def compute_loss(
        self,
        pred_dict,
        batch,
        device,
        active_loss_keys=None,
    ):
        human_imu = batch["human_imu"].to(device)
        batch_size, seq_len = human_imu.shape[:2]
        dtype = human_imu.dtype
        zero = human_imu.new_tensor(0.0)
        losses = {key: zero.clone() for key in self.LOSS_KEYS}

        obj_vel = batch.get("obj_vel")
        if isinstance(obj_vel, torch.Tensor):
            obj_vel = obj_vel.to(device=device, dtype=dtype)
        else:
            obj_vel = torch.zeros(
                batch_size, seq_len, 3, device=device, dtype=dtype
            )

        sensor_vel = batch.get("sensor_vel_glb")
        if isinstance(sensor_vel, torch.Tensor):
            sensor_vel = sensor_vel.to(device=device, dtype=dtype)
            if sensor_vel.dim() == 3:
                sensor_vel = sensor_vel.unsqueeze(0).expand(
                    batch_size, -1, -1, -1
                )
        else:
            sensor_vel = torch.zeros(
                batch_size,
                seq_len,
                len(_SENSOR_POS_INDICES),
                3,
                device=device,
                dtype=dtype,
            )

        contacts = []
        for key in ("lhand_contact", "rhand_contact", "obj_contact"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                value = value.to(device=device, dtype=dtype)
            else:
                value = torch.zeros(
                    batch_size, seq_len, device=device, dtype=dtype
                )
            contacts.append(value)
        contact_target = torch.stack(contacts, dim=-1)

        if isinstance(pred_dict.get("pred_obj_vel"), torch.Tensor):
            losses["obj_vel"] = F.mse_loss(
                pred_dict["pred_obj_vel"], obj_vel
            )
        if isinstance(pred_dict.get("pred_hand_glb_vel"), torch.Tensor):
            losses["hand_vel"] = F.mse_loss(
                pred_dict["pred_hand_glb_vel"], sensor_vel[:, :, -2:]
            )
        if isinstance(pred_dict.get("pred_hand_contact_prob"), torch.Tensor):
            losses["hand_contact"] = F.binary_cross_entropy(
                pred_dict["pred_hand_contact_prob"].clamp(1e-6, 1.0 - 1e-6),
                contact_target,
            )

        active = (
            set(self.LOSS_KEYS)
            if active_loss_keys is None
            else set(active_loss_keys)
        )
        unknown = active - self.LOSS_KEYS
        if unknown:
            raise ValueError(
                f"Unknown IMUHOI VC loss keys: {sorted(unknown)}"
            )
        weighted = {}
        total = zero.clone()
        for key, value in losses.items():
            if key not in active:
                continue
            weighted[key] = value * float(self.weights.get(key, 1.0))
            total = total + weighted[key]
        return total, losses, weighted

    def compute_test_loss(self, pred_dict, batch, device):
        total, _, weighted = self.compute_loss(pred_dict, batch, device)
        return total, weighted


class IMUHOIObjectTransLoss:
    """Original wrist-based IMUHOI ObjectTrans supervision."""

    LOSS_KEYS = {
        "obj_trans",
        "lhand_obj_direction",
        "rhand_obj_direction",
        "lhand_lb",
        "rhand_lb",
        "hoi_error_l",
        "hoi_error_r",
        "obj_vel_cons",
        "obj_acc_cons",
        "gate_weak",
    }
    TEST_LOSS_KEYS = {
        "lhand_obj_direction",
        "rhand_obj_direction",
        "lhand_lb",
        "rhand_lb",
        "hoi_error_l",
        "hoi_error_r",
    }
    MODEL_SELECTION_KEYS = set(TEST_LOSS_KEYS)
    DEFAULT_WEIGHTS = {
        "obj_trans": 0.1,
        "lhand_obj_direction": 0.1,
        "rhand_obj_direction": 0.1,
        "lhand_lb": 0.1,
        "rhand_lb": 0.1,
        "hoi_error_l": 0.2,
        "hoi_error_r": 0.2,
        "obj_vel_cons": 0.1,
        "obj_acc_cons": 0.01,
        "gate_weak": 0.0,
    }

    def __init__(self, weights=None):
        self.weights = weights or {}

    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)

    @staticmethod
    def _build_gate_weak_target(contact_prob, eps=1e-6):
        """Soft target for [left FK, right FK, IMU motion, static hold]."""
        p_left = contact_prob[..., 0:1].clamp(0.0, 1.0)
        p_right = contact_prob[..., 1:2].clamp(0.0, 1.0)
        p_move = contact_prob[..., 2:3].clamp(0.0, 1.0)
        p_left_cond = (p_left / p_move.clamp_min(eps)).clamp(0.0, 1.0)
        p_right_cond = (p_right / p_move.clamp_min(eps)).clamp(0.0, 1.0)
        no_hand_cond = (1.0 - p_left_cond) * (1.0 - p_right_cond)
        scores = torch.cat(
            (
                p_left,
                p_right,
                p_move * no_hand_cond,
                1.0 - p_move,
            ),
            dim=-1,
        ).clamp_min(eps)
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _wrist_geometry_targets(
        batch,
        obj_trans,
        position_global,
        batch_size,
        seq_len,
        device,
        dtype,
    ):
        wrists = select_wrist_positions(position_global)
        vector_world = obj_trans.unsqueeze(2) - wrists
        lengths = vector_world.norm(dim=-1)
        direction_world = vector_world / lengths.unsqueeze(-1).clamp_min(1e-8)

        obj_rot = batch.get("obj_rot")
        if isinstance(obj_rot, torch.Tensor):
            obj_rot = obj_rot.to(device=device, dtype=dtype)
        else:
            obj_imu = batch.get("obj_imu")
            obj_rot = (
                obj_imu[..., 3:9].to(device=device, dtype=dtype)
                if isinstance(obj_imu, torch.Tensor)
                else None
            )
        if obj_rot is None:
            eye = torch.eye(3, device=device, dtype=dtype)
            rotm = eye.reshape(1, 1, 3, 3).expand(
                batch_size, seq_len, 3, 3
            )
        elif obj_rot.shape[-1] == 6:
            rotm = rotation_6d_to_matrix(
                obj_rot.reshape(-1, 6)
            ).reshape(batch_size, seq_len, 3, 3)
        elif obj_rot.shape[-2:] == (3, 3):
            rotm = obj_rot.reshape(batch_size, seq_len, 3, 3)
        else:
            raise ValueError(
                f"Unsupported object rotation shape {tuple(obj_rot.shape)}"
            )

        direction_object = torch.matmul(
            rotm.transpose(-1, -2).unsqueeze(2),
            direction_world.unsqueeze(-1),
        ).squeeze(-1)
        return wrists, lengths, direction_object

    def compute_loss(self, pred_dict, batch, device):
        human_imu = batch["human_imu"].to(device)
        batch_size, seq_len = human_imu.shape[:2]
        dtype = human_imu.dtype
        zero = human_imu.new_tensor(0.0)
        losses = {key: zero.clone() for key in self.LOSS_KEYS}

        obj_trans = batch.get("obj_trans")
        if isinstance(obj_trans, torch.Tensor):
            obj_trans = obj_trans.to(device=device, dtype=dtype)
        else:
            obj_trans = torch.zeros(
                batch_size, seq_len, 3, device=device, dtype=dtype
            )
        obj_vel = batch.get("obj_vel")
        if isinstance(obj_vel, torch.Tensor):
            obj_vel = obj_vel.to(device=device, dtype=dtype)
        else:
            obj_vel = torch.zeros_like(obj_trans)
        obj_imu = batch.get("obj_imu")
        if isinstance(obj_imu, torch.Tensor):
            obj_acc = obj_imu[..., :3].to(device=device, dtype=dtype)
        else:
            obj_acc = torch.zeros_like(obj_trans)

        obj_mask = _has_object_mask(
            batch, batch_size, seq_len, device
        )
        if not obj_mask.any():
            weighted = {key: zero.clone() for key in self.LOSS_KEYS}
            return zero.clone(), losses, weighted

        if isinstance(pred_dict.get("pred_obj_trans"), torch.Tensor):
            losses["obj_trans"] = _masked_mse(
                pred_dict["pred_obj_trans"],
                obj_trans,
                obj_mask,
                zero,
            )
        if isinstance(
            pred_dict.get("pred_obj_vel_from_posdiff"), torch.Tensor
        ):
            losses["obj_vel_cons"] = _masked_mse(
                pred_dict["pred_obj_vel_from_posdiff"],
                obj_vel,
                obj_mask,
                zero,
            )
        if isinstance(
            pred_dict.get("pred_obj_acc_from_posdiff"), torch.Tensor
        ):
            losses["obj_acc_cons"] = _masked_mse(
                pred_dict["pred_obj_acc_from_posdiff"],
                obj_acc,
                obj_mask,
                zero,
            )

        gate_contact_source = None
        contact_values = []
        for key in ("lhand_contact", "rhand_contact", "obj_contact"):
            value = batch.get(key)
            if not isinstance(value, torch.Tensor):
                contact_values = []
                break
            value = value.to(device=device, dtype=dtype)
            if value.shape != (batch_size, seq_len):
                contact_values = []
                break
            contact_values.append(value)
        if len(contact_values) == 3:
            gate_contact_source = torch.stack(contact_values, dim=-1)
        elif isinstance(
            pred_dict.get("gating_contact_prob"), torch.Tensor
        ):
            candidate = pred_dict["gating_contact_prob"]
            if candidate.shape == (batch_size, seq_len, 3):
                gate_contact_source = candidate.to(device=device, dtype=dtype)

        gating_weights = pred_dict.get("gating_weights_raw")
        if (
            gate_contact_source is not None
            and isinstance(gating_weights, torch.Tensor)
            and gating_weights.shape == (batch_size, seq_len, 4)
        ):
            target = self._build_gate_weak_target(
                gate_contact_source
            ).detach()
            smoothing = float(
                self.weights.get("gate_weak_target_smoothing", 0.1)
            )
            smoothing = min(max(smoothing, 0.0), 1.0)
            if smoothing > 0.0:
                target = (
                    (1.0 - smoothing) * target
                    + smoothing / target.shape[-1]
                )
            gate_ce = -(
                target
                * torch.log(
                    gating_weights.to(device=device, dtype=dtype).clamp_min(1e-6)
                )
            ).sum(dim=-1)
            losses["gate_weak"] = _masked_mean(gate_ce, obj_mask, zero)

        position_global = batch.get("position_global")
        if isinstance(position_global, torch.Tensor):
            position_global = position_global.to(
                device=device, dtype=dtype
            )
            _, lengths, directions = self._wrist_geometry_targets(
                batch,
                obj_trans,
                position_global,
                batch_size,
                seq_len,
                device,
                dtype,
            )
            left_contact = batch.get("lhand_contact")
            right_contact = batch.get("rhand_contact")
            left_contact = (
                left_contact.to(device=device, dtype=torch.bool)
                if isinstance(left_contact, torch.Tensor)
                else torch.zeros_like(obj_mask)
            )
            right_contact = (
                right_contact.to(device=device, dtype=torch.bool)
                if isinstance(right_contact, torch.Tensor)
                else torch.zeros_like(obj_mask)
            )
            masks = (left_contact & obj_mask, right_contact & obj_mask)
            sides = (
                ("lhand", 0, masks[0]),
                ("rhand", 1, masks[1]),
            )
            for side, index, mask in sides:
                pred_length = pred_dict.get(f"pred_{side}_lb")
                pred_direction = pred_dict.get(
                    f"pred_{side}_obj_direction"
                )
                if isinstance(pred_length, torch.Tensor):
                    losses[f"{side}_lb"] = _masked_mse(
                        pred_length,
                        lengths[:, :, index],
                        mask,
                        zero,
                    )
                if isinstance(pred_direction, torch.Tensor):
                    losses[f"{side}_obj_direction"] = _masked_mse(
                        pred_direction,
                        directions[:, :, index],
                        mask,
                        zero,
                    )
                if (
                    isinstance(pred_length, torch.Tensor)
                    and isinstance(pred_direction, torch.Tensor)
                ):
                    target_vector = (
                        directions[:, :, index]
                        * lengths[:, :, index].unsqueeze(-1)
                    )
                    pred_vector = pred_direction * pred_length.unsqueeze(-1)
                    error = (pred_vector - target_vector).norm(dim=-1)
                    suffix = "l" if side == "lhand" else "r"
                    losses[f"hoi_error_{suffix}"] = _masked_mean(
                        error, mask, zero
                    )

        weighted = {}
        total = zero.clone()
        for key, value in losses.items():
            default_weight = 0.0 if key == "gate_weak" else 1.0
            weighted[key] = value * float(
                self.weights.get(key, default_weight)
            )
            total = total + weighted[key]
        return total, losses, weighted

    def compute_test_loss(self, pred_dict, batch, device):
        _, _, weighted = self.compute_loss(pred_dict, batch, device)
        test_losses = {
            key: value
            for key, value in weighted.items()
            if key in self.TEST_LOSS_KEYS
        }
        total = (
            sum(test_losses.values())
            if test_losses
            else batch["human_imu"].to(device).new_tensor(0.0)
        )
        return total, test_losses


__all__ = [
    "IMUHOIVelocityContactLoss",
    "IMUHOIObjectTransLoss",
]
