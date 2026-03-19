"""
HumanPose loss for inpainting-guided diffusion training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

from configs import _SENSOR_VEL_NAMES, _REDUCED_POSE_NAMES


class HumanPoseLoss:
    """DiffusionPoser-style objective composition for Stage-1 pose model."""

    LOSS_KEYS = {
        "simple_vel",
        "simple_pose",
        "simple_root_vel_local",
        "simple_root_vel",
        "simple_root_trans",
        "vel_smooth",
        "fk_joint",
        "drift",
        "foot_slide",
        "hand_pos",
        "diffusion_x0",
    }

    TEST_LOSS_KEYS = {
        "simple_pose",
        "simple_root_trans",
        "fk_joint",
        "drift",
    }

    def __init__(self, weights=None, no_trans=False):
        self.weights = weights or {}
        self.no_trans = no_trans

    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)

    def _weight(self, key: str, default: float = 1.0) -> float:
        return float(self.weights.get(key, default))

    @staticmethod
    def _masked_l2(values: torch.Tensor, mask: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        if mask.sum() == 0:
            return zero.clone()
        return (values[mask] ** 2).mean()

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        if mask is None:
            return F.mse_loss(pred, target)
        if mask.dtype != torch.bool:
            mask = mask > 0.5
        if mask.shape != pred.shape:
            try:
                mask = mask.expand_as(pred)
            except RuntimeError:
                return F.mse_loss(pred, target)
        if mask.sum() == 0:
            return zero.clone()
        diff = (pred - target) ** 2
        return diff[mask].mean()

    @staticmethod
    def _normalize_frame_mask(
        frame_mask: torch.Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(frame_mask, torch.Tensor):
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        mask = frame_mask.to(device=device, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        if mask.shape[0] != batch_size or mask.shape[1] != seq_len:
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        return mask

    def compute_loss(self, pred_dict, batch, device):
        human_imu = batch["human_imu"].to(device)
        dtype = human_imu.dtype
        bs, seq = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)

        losses = {key: zero.clone() for key in self.LOSS_KEYS}
        diffusion_aux = pred_dict.get("diffusion_aux", {})
        rollout_frame_mask = diffusion_aux.get("rollout_frame_mask") if isinstance(diffusion_aux, dict) else None
        rollout_frame_mask = self._normalize_frame_mask(
            rollout_frame_mask,
            batch_size=bs,
            seq_len=seq,
            device=device,
        )
        rollout_pair_mask = None
        if seq > 1:
            rollout_pair_mask = rollout_frame_mask[:, 1:] & rollout_frame_mask[:, :-1]

        root_ori_gt = rotation_6d_to_matrix(human_imu[:, :, 0, -6:])

        trans_gt = batch.get("trans")
        if isinstance(trans_gt, torch.Tensor):
            trans_gt = trans_gt.to(device=device, dtype=dtype)
        else:
            trans_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

        root_vel_gt = batch.get("root_vel")
        if isinstance(root_vel_gt, torch.Tensor):
            root_vel_gt = root_vel_gt.to(device=device, dtype=dtype)
        else:
            root_vel_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

        root_vel_local_gt = root_ori_gt.transpose(-1, -2).matmul(root_vel_gt.unsqueeze(-1)).squeeze(-1)

        sensor_vel_root_gt = batch.get("sensor_vel_root")
        if isinstance(sensor_vel_root_gt, torch.Tensor):
            sensor_vel_root_gt = sensor_vel_root_gt.to(device=device, dtype=dtype)
            if sensor_vel_root_gt.dim() == 3:
                sensor_vel_root_gt = sensor_vel_root_gt.unsqueeze(0).expand(bs, -1, -1, -1)
        else:
            sensor_vel_root_gt = torch.zeros(bs, seq, len(_SENSOR_VEL_NAMES), 3, device=device, dtype=dtype)

        ori_root_reduced_gt = batch.get("ori_root_reduced")
        if isinstance(ori_root_reduced_gt, torch.Tensor):
            ori_root_reduced_gt = ori_root_reduced_gt.to(device=device, dtype=dtype)
        else:
            ori_root_reduced_gt = None

        position_global_gt = batch.get("position_global")
        if isinstance(position_global_gt, torch.Tensor):
            position_global_gt = position_global_gt.to(device=device, dtype=dtype)
        else:
            position_global_gt = None

        vel_indices = [0, 1, 2, 3, 0, 3, 4, 5]
        target_vel = sensor_vel_root_gt[:, :, vel_indices, :]
        frame_mask_v = rollout_frame_mask.unsqueeze(-1).unsqueeze(-1)
        pair_mask_v = rollout_pair_mask.unsqueeze(-1).unsqueeze(-1) if rollout_pair_mask is not None else None

        if "v_pred" in pred_dict:
            v_pred = pred_dict["v_pred"].view(bs, seq, -1, 3)
            losses["simple_vel"] = self._masked_mse(v_pred, target_vel, frame_mask_v, zero)

            if seq > 1:
                dv_pred = v_pred[:, 1:] - v_pred[:, :-1]
                dv_gt = target_vel[:, 1:] - target_vel[:, :-1]
                losses["vel_smooth"] = losses["vel_smooth"] + self._masked_mse(dv_pred, dv_gt, pair_mask_v, zero)

        pose_gt_6d = None
        if "p_pred" in pred_dict and ori_root_reduced_gt is not None:
            pose_gt_6d = matrix_to_rotation_6d(
                ori_root_reduced_gt.reshape(-1, 3, 3)
            ).reshape(bs, seq, len(_REDUCED_POSE_NAMES), 6)
            p_pred = pred_dict["p_pred"].view(bs, seq, len(_REDUCED_POSE_NAMES), 6)
            losses["simple_pose"] = self._masked_mse(
                p_pred,
                pose_gt_6d,
                rollout_frame_mask.unsqueeze(-1).unsqueeze(-1),
                zero,
            )

            if seq > 1:
                dp_pred = p_pred[:, 1:] - p_pred[:, :-1]
                dp_gt = pose_gt_6d[:, 1:] - pose_gt_6d[:, :-1]
                losses["vel_smooth"] = losses["vel_smooth"] + self._masked_mse(
                    dp_pred,
                    dp_gt,
                    pair_mask_v,
                    zero,
                )

        if not self.no_trans:
            if "root_vel_local_pred" in pred_dict:
                losses["simple_root_vel_local"] = self._masked_mse(
                    pred_dict["root_vel_local_pred"],
                    root_vel_local_gt,
                    rollout_frame_mask.unsqueeze(-1),
                    zero,
                )
                if seq > 1:
                    drv_pred = pred_dict["root_vel_local_pred"][:, 1:] - pred_dict["root_vel_local_pred"][:, :-1]
                    drv_gt = root_vel_local_gt[:, 1:] - root_vel_local_gt[:, :-1]
                    losses["vel_smooth"] = losses["vel_smooth"] + self._masked_mse(
                        drv_pred,
                        drv_gt,
                        rollout_pair_mask.unsqueeze(-1) if rollout_pair_mask is not None else None,
                        zero,
                    )

            if "root_vel_pred" in pred_dict:
                losses["simple_root_vel"] = self._masked_mse(
                    pred_dict["root_vel_pred"],
                    root_vel_gt,
                    rollout_frame_mask.unsqueeze(-1),
                    zero,
                )

            if "root_trans_pred" in pred_dict:
                root_trans_pred = pred_dict["root_trans_pred"]
                losses["simple_root_trans"] = self._masked_mse(
                    root_trans_pred,
                    trans_gt,
                    rollout_frame_mask.unsqueeze(-1),
                    zero,
                )

                start_idx = diffusion_aux.get("rollout_start_idx") if isinstance(diffusion_aux, dict) else None
                if isinstance(start_idx, torch.Tensor):
                    start_indices = start_idx.to(device=device, dtype=torch.long).view(-1)
                    if start_indices.numel() == 1 and bs > 1:
                        start_indices = start_indices.expand(bs)
                    elif start_indices.numel() != bs:
                        start_indices = start_indices[:1].expand(bs)
                elif isinstance(start_idx, int):
                    start_indices = torch.full((bs,), start_idx, device=device, dtype=torch.long)
                else:
                    start_indices = rollout_frame_mask.to(dtype=torch.long).argmax(dim=1)
                start_indices = start_indices.clamp(0, seq - 1)

                batch_ids = torch.arange(bs, device=device, dtype=torch.long)
                ref_pred = root_trans_pred[batch_ids, start_indices]
                ref_gt = trans_gt[batch_ids, start_indices]
                rel_pred = root_trans_pred - ref_pred.unsqueeze(1)
                rel_gt = trans_gt - ref_gt.unsqueeze(1)
                losses["drift"] = self._masked_mse(
                    rel_pred,
                    rel_gt,
                    rollout_frame_mask.unsqueeze(-1),
                    zero,
                )

        if "pred_joints_global" in pred_dict and position_global_gt is not None:
            pred_joints_global = pred_dict["pred_joints_global"]
            num_joints = min(pred_joints_global.shape[2], position_global_gt.shape[2])
            if num_joints > 0:
                losses["fk_joint"] = self._masked_mse(
                    pred_joints_global[:, :, :num_joints, :],
                    position_global_gt[:, :, :num_joints, :],
                    rollout_frame_mask.unsqueeze(-1).unsqueeze(-1),
                    zero,
                )

        if "pred_hand_glb_pos" in pred_dict and position_global_gt is not None:
            if position_global_gt.shape[2] > 21:
                hand_pos_gt = torch.stack([
                    position_global_gt[:, :, 20, :],
                    position_global_gt[:, :, 21, :],
                ], dim=2)
                losses["hand_pos"] = self._masked_mse(
                    pred_dict["pred_hand_glb_pos"],
                    hand_pos_gt,
                    rollout_frame_mask.unsqueeze(-1).unsqueeze(-1),
                    zero,
                )

        if "pred_joints_global" in pred_dict and seq > 1:
            pred_joints_global = pred_dict["pred_joints_global"]
            foot_ids = [7, 8]
            foot_vel = (pred_joints_global[:, 1:, foot_ids, :] - pred_joints_global[:, :-1, foot_ids, :])
            foot_speed = torch.norm(foot_vel, dim=-1)  # [B, T-1, 2]

            lfoot_contact = batch.get("lfoot_contact")
            rfoot_contact = batch.get("rfoot_contact")
            if isinstance(lfoot_contact, torch.Tensor) and isinstance(rfoot_contact, torch.Tensor):
                lfoot_contact = lfoot_contact.to(device=device, dtype=dtype)
                rfoot_contact = rfoot_contact.to(device=device, dtype=dtype)
                if lfoot_contact.dim() == 1:
                    lfoot_contact = lfoot_contact.unsqueeze(0).expand(bs, -1)
                if rfoot_contact.dim() == 1:
                    rfoot_contact = rfoot_contact.unsqueeze(0).expand(bs, -1)
                contact = torch.stack([lfoot_contact[:, 1:], rfoot_contact[:, 1:]], dim=-1) > 0.5
                if rollout_pair_mask is not None:
                    contact = contact & rollout_pair_mask.unsqueeze(-1)
                losses["foot_slide"] = self._masked_l2(foot_speed, contact, zero)

        x0_target = diffusion_aux.get("x0_target") if isinstance(diffusion_aux, dict) else None
        x0_projected = diffusion_aux.get("x0_projected") if isinstance(diffusion_aux, dict) else None
        unknown_mask = diffusion_aux.get("unknown_mask") if isinstance(diffusion_aux, dict) else None

        if isinstance(x0_target, torch.Tensor) and isinstance(x0_projected, torch.Tensor):
            x0_target = x0_target.to(device=device, dtype=dtype)
            x0_projected = x0_projected.to(device=device, dtype=dtype)
            active_mask = None
            if isinstance(unknown_mask, torch.Tensor):
                unknown_mask = unknown_mask.to(device=device, dtype=torch.bool)
                if unknown_mask.shape == x0_target.shape:
                    active_mask = unknown_mask
            if active_mask is None:
                active_mask = rollout_frame_mask.unsqueeze(-1).expand_as(x0_target)
            if active_mask.sum() > 0:
                diff = x0_projected - x0_target
                losses["diffusion_x0"] = (diff[active_mask] ** 2).mean()
            else:
                losses["diffusion_x0"] = zero.clone()

        total_loss = zero.clone()
        weighted_losses = {}

        weight_map = {
            "simple_vel": self._weight("simple_vel", default=1.0),
            "simple_pose": self._weight("simple_pose", default=1.0),
            "simple_root_vel_local": self._weight("simple_root_vel_local", default=1.0),
            "simple_root_vel": self._weight("simple_root_vel", default=1.0),
            "simple_root_trans": self._weight("simple_root_trans", default=1.0),
            "vel_smooth": self._weight("vel_smooth", default=0.5),
            "fk_joint": self._weight("fk_joint", default=1.0),
            "drift": self._weight("drift", default=1.0),
            "foot_slide": self._weight("foot_slide", default=0.1),
            "hand_pos": self._weight("hand_pos", default=1.0),
            "diffusion_x0": self._weight("diffusion_x0", default=1.0),
        }

        for key, loss in losses.items():
            weight = weight_map.get(key, 1.0)
            if self.no_trans and key in {"simple_root_vel_local", "simple_root_vel", "simple_root_trans", "drift"}:
                weight = 0.0
            weighted = loss * weight
            weighted_losses[key] = weighted
            total_loss = total_loss + weighted

        return total_loss, losses, weighted_losses

    def compute_test_loss(self, pred_dict, batch, device):
        total_loss, losses, _ = self.compute_loss(pred_dict, batch, device)
        test_losses = {k: v for k, v in losses.items() if k in self.TEST_LOSS_KEYS}
        if not test_losses:
            return total_loss, losses
        test_total = sum(test_losses.values())
        return test_total, test_losses

    @classmethod
    def get_loss_keys(cls):
        return list(cls.LOSS_KEYS)
