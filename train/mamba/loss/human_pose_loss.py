"""Loss for the Mamba human pose module."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from configs import _REDUCED_POSE_NAMES, _SENSOR_VEL_NAMES
from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix


class HumanPoseLoss:
    """Supervised Stage-1 loss for dual-temporal hierarchical Mamba."""

    LOSS_KEYS = (
        "vel_root",
        "pose_reduced",
        "root_vel_local",
        "root_vel",
        "root_trans",
        "hand_pos",
        "fk_joint",
        "contact",
        "smooth",
    )
    TEST_LOSS_KEYS = ("pose_reduced", "root_trans", "fk_joint", "hand_pos")

    _WEIGHT_ALIASES = {
        "vel_root": ("vel_root", "simple_vel"),
        "pose_reduced": ("pose_reduced", "simple_pose"),
        "root_vel_local": ("root_vel_local", "simple_root_vel_local"),
        "root_vel": ("root_vel", "simple_root_vel"),
        "root_trans": ("root_trans", "simple_root_trans"),
        "hand_pos": ("hand_pos",),
        "fk_joint": ("fk_joint", "fk"),
        "contact": ("contact", "foot_contact", "contact_logits"),
        "smooth": ("smooth", "vel_smooth"),
    }

    _DEFAULT_WEIGHTS = {
        "vel_root": 1.0,
        "pose_reduced": 10.0,
        "root_vel_local": 1.0,
        "root_vel": 1.0,
        "root_trans": 1.0,
        "hand_pos": 1.0,
        "fk_joint": 1.0,
        "contact": 1.0,
        "smooth": 0.5,
    }

    def __init__(self, weights=None, no_trans: bool = False):
        self.weights = weights or {}
        self.no_trans = bool(no_trans)

    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)

    def _weight(self, key: str) -> float:
        for alias in self._WEIGHT_ALIASES.get(key, (key,)):
            if alias in self.weights:
                return float(self.weights[alias])
        return self._DEFAULT_WEIGHTS[key]

    @staticmethod
    def _expand_bt(value, batch_size: int, seq_len: int, trailing_shape, device, dtype):
        shape = (batch_size, seq_len) if trailing_shape is None else (batch_size, seq_len, *trailing_shape)
        if not isinstance(value, torch.Tensor):
            return None
        out = value.to(device=device, dtype=dtype)
        if out.dim() == len(shape) - 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, *out.shape[1:])
        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return None
        if trailing_shape is not None and tuple(out.shape[2:]) != tuple(trailing_shape):
            return None
        return out

    def _target_pose_6d(self, batch, batch_size: int, seq_len: int, device, dtype):
        ori_root_reduced = self._expand_bt(
            batch.get("ori_root_reduced"),
            batch_size,
            seq_len,
            (len(_REDUCED_POSE_NAMES), 3, 3),
            device,
            dtype,
        )
        if ori_root_reduced is None:
            return None
        return matrix_to_rotation_6d(ori_root_reduced.reshape(-1, 3, 3)).reshape(
            batch_size,
            seq_len,
            len(_REDUCED_POSE_NAMES),
            6,
        )

    def compute_loss(self, pred_dict, batch, device):
        human_imu = batch["human_imu"].to(device)
        dtype = human_imu.dtype
        batch_size, seq_len = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)
        losses = {key: zero.clone() for key in self.LOSS_KEYS}

        root_ori_gt = rotation_6d_to_matrix(human_imu[:, :, 0, -6:])
        trans_gt = self._expand_bt(batch.get("trans"), batch_size, seq_len, (3,), device, dtype)
        if trans_gt is None:
            trans_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        root_vel_gt = self._expand_bt(batch.get("root_vel"), batch_size, seq_len, (3,), device, dtype)
        if root_vel_gt is None:
            root_vel_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        root_vel_local_gt = torch.matmul(root_ori_gt.transpose(-1, -2), root_vel_gt.unsqueeze(-1)).squeeze(-1)

        sensor_vel_root_gt = self._expand_bt(
            batch.get("sensor_vel_root"),
            batch_size,
            seq_len,
            (len(_SENSOR_VEL_NAMES), 3),
            device,
            dtype,
        )
        if sensor_vel_root_gt is not None and isinstance(pred_dict.get("v_pred"), torch.Tensor):
            v_pred = pred_dict["v_pred"].to(device=device, dtype=dtype).reshape(batch_size, seq_len, -1, 3)
            vel_indices = [0, 1, 2, 3, 0, 3, 4, 5]
            target_vel = sensor_vel_root_gt[:, :, vel_indices, :]
            if v_pred.shape[2] == target_vel.shape[2]:
                losses["vel_root"] = F.mse_loss(v_pred, target_vel)

        pose_gt_6d = self._target_pose_6d(batch, batch_size, seq_len, device, dtype)
        if pose_gt_6d is not None and isinstance(pred_dict.get("p_pred"), torch.Tensor):
            p_pred = pred_dict["p_pred"].to(device=device, dtype=dtype).reshape(
                batch_size,
                seq_len,
                len(_REDUCED_POSE_NAMES),
                6,
            )
            losses["pose_reduced"] = F.mse_loss(p_pred, pose_gt_6d)

            if seq_len > 1:
                losses["smooth"] = F.mse_loss(p_pred[:, 1:] - p_pred[:, :-1], pose_gt_6d[:, 1:] - pose_gt_6d[:, :-1])

        if not self.no_trans:
            if isinstance(pred_dict.get("root_vel_local_pred"), torch.Tensor):
                losses["root_vel_local"] = F.mse_loss(
                    pred_dict["root_vel_local_pred"].to(device=device, dtype=dtype),
                    root_vel_local_gt,
                )
            if isinstance(pred_dict.get("root_vel_pred"), torch.Tensor):
                losses["root_vel"] = F.mse_loss(pred_dict["root_vel_pred"].to(device=device, dtype=dtype), root_vel_gt)
            if isinstance(pred_dict.get("root_trans_pred"), torch.Tensor):
                root_trans_pred = pred_dict["root_trans_pred"].to(device=device, dtype=dtype)
                losses["root_trans"] = F.mse_loss(root_trans_pred, trans_gt)
                if seq_len > 1:
                    losses["smooth"] = losses["smooth"] + F.mse_loss(
                        root_trans_pred[:, 1:] - root_trans_pred[:, :-1],
                        trans_gt[:, 1:] - trans_gt[:, :-1],
                    )

        position_global_gt = self._expand_bt(batch.get("position_global"), batch_size, seq_len, None, device, dtype)
        if position_global_gt is None and isinstance(batch.get("position_global"), torch.Tensor):
            position_global_gt = batch["position_global"].to(device=device, dtype=dtype)
            if position_global_gt.dim() == 3:
                position_global_gt = position_global_gt.unsqueeze(0)
            if position_global_gt.shape[0] == 1 and batch_size > 1:
                position_global_gt = position_global_gt.expand(batch_size, -1, -1, -1)

        if (
            isinstance(pred_dict.get("pred_hand_glb_pos"), torch.Tensor)
            and isinstance(position_global_gt, torch.Tensor)
            and position_global_gt.shape[0] == batch_size
            and position_global_gt.shape[1] == seq_len
            and position_global_gt.shape[2] > 21
        ):
            hand_gt = torch.stack((position_global_gt[:, :, 20], position_global_gt[:, :, 21]), dim=2)
            losses["hand_pos"] = F.mse_loss(pred_dict["pred_hand_glb_pos"].to(device=device, dtype=dtype), hand_gt)

        if (
            isinstance(pred_dict.get("pred_joints_global"), torch.Tensor)
            and isinstance(position_global_gt, torch.Tensor)
            and position_global_gt.shape[0] == batch_size
            and position_global_gt.shape[1] == seq_len
        ):
            pred_joints = pred_dict["pred_joints_global"].to(device=device, dtype=dtype)
            nj = min(pred_joints.shape[2], position_global_gt.shape[2])
            if nj > 0:
                losses["fk_joint"] = F.mse_loss(pred_joints[:, :, :nj], position_global_gt[:, :, :nj])

        contact_pred = pred_dict.get("contact_pred", pred_dict.get("b_pred"))
        lfoot = self._expand_bt(batch.get("lfoot_contact"), batch_size, seq_len, (), device, dtype)
        rfoot = self._expand_bt(batch.get("rfoot_contact"), batch_size, seq_len, (), device, dtype)
        if isinstance(contact_pred, torch.Tensor) and lfoot is not None and rfoot is not None:
            contact_target = torch.stack((lfoot, rfoot), dim=-1).clamp(0.0, 1.0)
            losses["contact"] = F.binary_cross_entropy_with_logits(
                contact_pred.to(device=device, dtype=dtype),
                contact_target,
            )

        total_loss = zero.clone()
        weighted_losses = {}
        for key in self.LOSS_KEYS:
            weight = self._weight(key)
            if self.no_trans and key in {"root_vel_local", "root_vel", "root_trans"}:
                weight = 0.0
            weighted = losses[key] * weight
            weighted_losses[key] = weighted
            total_loss = total_loss + weighted
        return total_loss, losses, weighted_losses

    def compute_test_loss(self, pred_dict, batch, device):
        _, losses, _ = self.compute_loss(pred_dict, batch, device)
        test_losses = {k: losses[k] for k in self.TEST_LOSS_KEYS}
        if self.no_trans:
            test_losses["root_trans"] = losses["root_trans"].new_tensor(0.0)
        return sum(test_losses.values()), test_losses

    @classmethod
    def get_loss_keys(cls):
        return list(cls.LOSS_KEYS)
