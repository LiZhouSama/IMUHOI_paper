"""
HumanPose loss for full-window diffusion training.

Loss terms (and only these five):
- L_simple
- L_vel
- L_FK
- L_drift
- L_slide
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d


class HumanPoseLoss:
    """Stage-1 loss composition for full-window denoising."""

    LOSS_KEYS = ("simple", "vel", "fk", "drift", "slide")
    TEST_LOSS_KEYS = ("mpjre_mse", "root_trans_err_mm", "jitter_local_fk_mm", "root_jitter_mm")

    _WEIGHT_ALIASES = {
        "simple": ("simple", "L_simple", "diffusion_x0", "simple_pose"),
        "vel": ("vel", "L_vel", "vel_smooth"),
        "fk": ("fk", "L_FK", "fk_joint"),
        "drift": ("drift", "L_drift"),
        "slide": ("slide", "L_slide", "foot_slide"),
    }

    def __init__(self, weights=None, test_metric_weights=None, no_trans: bool = False):
        self.weights = weights or {}
        self.test_metric_weights = test_metric_weights or {}
        self.no_trans = bool(no_trans)

    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)

    def _weight(self, key: str, default: float = 1.0) -> float:
        aliases = self._WEIGHT_ALIASES.get(key, (key,))
        for name in aliases:
            if name in self.weights:
                return float(self.weights[name])
        return float(default)

    @staticmethod
    def _to_bt(value, batch_size: int, seq_len: int, trailing_shape, device, dtype, default: float = 0.0):
        shape = (batch_size, seq_len, *trailing_shape)
        if not isinstance(value, torch.Tensor):
            return torch.full(shape, float(default), device=device, dtype=dtype)
        out = value.to(device=device, dtype=dtype)
        if out.dim() == len(shape) - 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, *out.shape[1:])
        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return torch.full(shape, float(default), device=device, dtype=dtype)
        if len(trailing_shape) > 0 and tuple(out.shape[2:]) != tuple(trailing_shape):
            return torch.full(shape, float(default), device=device, dtype=dtype)
        return out

    def compute_loss(self, pred_dict, batch, device):
        human_imu = batch["human_imu"].to(device)
        dtype = human_imu.dtype
        batch_size, seq_len = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)

        losses = {k: zero.clone() for k in self.LOSS_KEYS}

        diffusion_aux = pred_dict.get("diffusion_aux", {}) if isinstance(pred_dict, dict) else {}
        x_pred = pred_dict.get("x_pred") if isinstance(pred_dict, dict) else None
        x_target = diffusion_aux.get("x0_target") if isinstance(diffusion_aux, dict) else None

        if isinstance(x_pred, torch.Tensor) and isinstance(x_target, torch.Tensor):
            x_pred = x_pred.to(device=device, dtype=dtype)
            x_target = x_target.to(device=device, dtype=dtype)
            if x_pred.shape == x_target.shape:
                losses["simple"] = F.mse_loss(x_pred, x_target)

        r_pred = pred_dict.get("R_pred_6d") if isinstance(pred_dict, dict) else None
        if isinstance(r_pred, torch.Tensor) and isinstance(x_target, torch.Tensor):
            r_pred = r_pred.to(device=device, dtype=dtype)
            rot_dim = r_pred.shape[2] * 6
            if x_target.shape[-1] >= rot_dim:
                r_gt = x_target[..., :rot_dim].reshape_as(r_pred)
                if seq_len > 1:
                    dr_pred = r_pred[:, 1:] - r_pred[:, :-1]
                    dr_gt = r_gt[:, 1:] - r_gt[:, :-1]
                    losses["vel"] = F.mse_loss(dr_pred, dr_gt)

        pred_joints_local = pred_dict.get("pred_joints_local") if isinstance(pred_dict, dict) else None
        gt_joints_local = pred_dict.get("gt_joints_local") if isinstance(pred_dict, dict) else None
        if not isinstance(gt_joints_local, torch.Tensor):
            position_global = batch.get("position_global")
            trans = batch.get("trans")
            if isinstance(position_global, torch.Tensor) and isinstance(trans, torch.Tensor):
                position_global = position_global.to(device=device, dtype=dtype)
                trans = trans.to(device=device, dtype=dtype)
                if position_global.dim() == 3:
                    position_global = position_global.unsqueeze(0)
                if trans.dim() == 2:
                    trans = trans.unsqueeze(0)
                if position_global.shape[0] == 1 and batch_size > 1:
                    position_global = position_global.expand(batch_size, -1, -1, -1)
                if trans.shape[0] == 1 and batch_size > 1:
                    trans = trans.expand(batch_size, -1, -1)
                if position_global.shape[0] == batch_size and trans.shape[0] == batch_size:
                    gt_joints_local = position_global - trans.unsqueeze(2)

        if isinstance(pred_joints_local, torch.Tensor) and isinstance(gt_joints_local, torch.Tensor):
            pred_joints_local = pred_joints_local.to(device=device, dtype=dtype)
            gt_joints_local = gt_joints_local.to(device=device, dtype=dtype)
            nj = min(pred_joints_local.shape[2], gt_joints_local.shape[2])
            if nj > 0:
                losses["fk"] = F.mse_loss(pred_joints_local[:, :, :nj], gt_joints_local[:, :, :nj])

        delta_p_pred = pred_dict.get("delta_p_pred") if isinstance(pred_dict, dict) else None
        trans_gt = batch.get("trans")
        if (
            (not self.no_trans)
            and isinstance(delta_p_pred, torch.Tensor)
            and delta_p_pred.shape[-1] >= 2
            and isinstance(trans_gt, torch.Tensor)
        ):
            delta_p_pred = delta_p_pred.to(device=device, dtype=dtype)
            trans_gt = trans_gt.to(device=device, dtype=dtype)
            if trans_gt.dim() == 2:
                trans_gt = trans_gt.unsqueeze(0)
            if trans_gt.shape[0] == 1 and batch_size > 1:
                trans_gt = trans_gt.expand(batch_size, -1, -1)
            if trans_gt.shape[0] == batch_size and trans_gt.shape[1] == seq_len:
                xz_init = trans_gt[:, 0, [0, 2]]
                xz_pred = torch.cumsum(delta_p_pred[..., :2], dim=1) + xz_init.unsqueeze(1)
                losses["drift"] = F.mse_loss(xz_pred[:, -1], trans_gt[:, -1, [0, 2]])

        p_y_pred = pred_dict.get("p_y_pred") if isinstance(pred_dict, dict) else None
        b_prob_pred = pred_dict.get("b_prob_pred") if isinstance(pred_dict, dict) else None
        if (
            (not self.no_trans)
            and isinstance(pred_joints_local, torch.Tensor)
            and isinstance(delta_p_pred, torch.Tensor)
            and isinstance(b_prob_pred, torch.Tensor)
            and seq_len > 1
        ):
            pred_joints_local = pred_joints_local.to(device=device, dtype=dtype)
            delta_p_pred = delta_p_pred.to(device=device, dtype=dtype)
            b_prob_pred = b_prob_pred.to(device=device, dtype=dtype)

            if pred_joints_local.shape[2] > 8:
                foot = pred_joints_local[:, :, [7, 8], :]  # [B,T,2,3]
                foot_delta = foot[:, 1:] - foot[:, :-1]  # [B,T-1,2,3]

                delta2 = delta_p_pred[:, :-1, :2]  # [B,T-1,2]
                delta3 = torch.zeros(batch_size, seq_len - 1, 1, 3, device=device, dtype=dtype)
                delta3[:, :, :, 0] = delta2[..., 0:1]
                delta3[:, :, :, 2] = delta2[..., 1:2]
                if isinstance(p_y_pred, torch.Tensor):
                    p_y_pred = p_y_pred.to(device=device, dtype=dtype)
                    if p_y_pred.dim() == 2:
                        p_y_pred = p_y_pred.unsqueeze(-1)
                    if p_y_pred.shape[0] == 1 and batch_size > 1:
                        p_y_pred = p_y_pred.expand(batch_size, -1, -1)
                    if p_y_pred.shape[0] == batch_size and p_y_pred.shape[1] == seq_len and p_y_pred.shape[-1] >= 1:
                        delta_p_y = p_y_pred[:, 1:, 0] - p_y_pred[:, :-1, 0]  # [B,T-1]
                        delta3[:, :, :, 1] = delta_p_y.unsqueeze(-1)

                contact_w = b_prob_pred[:, :-1, :2].clamp(0.0, 1.0).unsqueeze(-1)  # [B,T-1,2,1]
                slide_vec = contact_w * (foot_delta + delta3)
                losses["slide"] = (slide_vec ** 2).mean()

        weighted_losses = {}
        total_loss = zero.clone()

        weight_defaults = {
            "simple": 1.0,
            "vel": 1.0,
            "fk": 1.0,
            "drift": 0.1,
            "slide": 1,
        }

        for key in self.LOSS_KEYS:
            w = self._weight(key, default=weight_defaults[key])
            if self.no_trans and key in {"drift", "slide"}:
                w = 0.0
            weighted = losses[key] * w
            weighted_losses[key] = weighted
            total_loss = total_loss + weighted

        return total_loss, losses, weighted_losses

    def compute_test_loss(self, pred_dict, batch, device):
        human_imu = batch["human_imu"].to(device)
        dtype = human_imu.dtype
        batch_size, seq_len = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)

        metrics = {k: zero.clone() for k in self.TEST_LOSS_KEYS}

        r_pred = pred_dict.get("R_pred_6d") if isinstance(pred_dict, dict) else None
        rot_gt = batch.get("rotation_global")
        if isinstance(r_pred, torch.Tensor) and isinstance(rot_gt, torch.Tensor):
            r_pred = r_pred.to(device=device, dtype=dtype)
            rot_gt = rot_gt.to(device=device, dtype=dtype)
            if rot_gt.dim() == 4:
                rot_gt = rot_gt.unsqueeze(0)
            if rot_gt.shape[0] == 1 and batch_size > 1:
                rot_gt = rot_gt.expand(batch_size, -1, -1, -1, -1)
            if rot_gt.shape[0] == batch_size and rot_gt.shape[1] == seq_len and rot_gt.shape[-2:] == (3, 3):
                joints_pred = r_pred.shape[2] if r_pred.dim() == 4 else 0
                joints_gt = rot_gt.shape[2]
                nj = min(joints_pred, joints_gt)
                if nj > 0:
                    gt_6d = matrix_to_rotation_6d(rot_gt[:, :, :nj].reshape(-1, 3, 3)).reshape(batch_size, seq_len, nj, 6)
                    metrics["mpjre_mse"] = F.mse_loss(r_pred[:, :, :nj], gt_6d)

        root_trans_pred = pred_dict.get("root_trans_pred") if isinstance(pred_dict, dict) else None
        if isinstance(root_trans_pred, torch.Tensor):
            root_trans_pred = root_trans_pred.to(device=device, dtype=dtype)
        trans_gt = batch.get("trans")
        if (not self.no_trans) and isinstance(root_trans_pred, torch.Tensor) and isinstance(trans_gt, torch.Tensor):
            trans_gt = trans_gt.to(device=device, dtype=dtype)
            if trans_gt.dim() == 2:
                trans_gt = trans_gt.unsqueeze(0)
            if trans_gt.shape[0] == 1 and batch_size > 1:
                trans_gt = trans_gt.expand(batch_size, -1, -1)
            if root_trans_pred.shape[0] == batch_size and root_trans_pred.shape[1] == seq_len and trans_gt.shape[:2] == (batch_size, seq_len):
                metrics["root_trans_err_mm"] = torch.linalg.norm(root_trans_pred - trans_gt, dim=-1).mean() * 1000.0

        pred_joints_local = pred_dict.get("pred_joints_local") if isinstance(pred_dict, dict) else None
        if isinstance(pred_joints_local, torch.Tensor):
            pred_joints_local = pred_joints_local.to(device=device, dtype=dtype)
            if pred_joints_local.shape[0] == batch_size and pred_joints_local.shape[1] == seq_len and seq_len > 2:
                acc_local = pred_joints_local[:, 2:] - 2.0 * pred_joints_local[:, 1:-1] + pred_joints_local[:, :-2]
                metrics["jitter_local_fk_mm"] = torch.linalg.norm(acc_local, dim=-1).mean() * 1000.0

        if (not self.no_trans) and isinstance(root_trans_pred, torch.Tensor) and seq_len > 2:
            if root_trans_pred.shape[0] == batch_size and root_trans_pred.shape[1] == seq_len:
                acc_root = root_trans_pred[:, 2:] - 2.0 * root_trans_pred[:, 1:-1] + root_trans_pred[:, :-2]
                metrics["root_jitter_mm"] = torch.linalg.norm(acc_root, dim=-1).mean() * 1000.0

        default_test_weights = {
            "mpjre_mse": 1.0,
            "root_trans_err_mm": 1.0,
            "jitter_local_fk_mm": 1.0,
            "root_jitter_mm": 1.0,
        }

        test_total = zero.clone()
        for key in self.TEST_LOSS_KEYS:
            weight = float(self.test_metric_weights.get(key, default_test_weights[key]))
            if self.no_trans and key in {"root_trans_err_mm", "root_jitter_mm"}:
                weight = 0.0
            test_total = test_total + metrics[key] * weight

        return test_total, metrics

    @classmethod
    def get_loss_keys(cls):
        return list(cls.LOSS_KEYS)
