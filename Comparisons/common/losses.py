"""Loss functions matching the comparison baseline protocols."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def object_position_loss(pred: torch.Tensor, target: torch.Tensor, has_object: torch.Tensor) -> torch.Tensor:
    """Masked object translation MSE."""
    if has_object is None:
        return F.mse_loss(pred, target)
    mask = has_object.to(device=pred.device, dtype=pred.dtype).view(-1, 1, 1)
    denom = mask.sum() * pred.shape[1] * pred.shape[2]
    if denom.item() == 0:
        return pred.sum() * 0.0
    return (((pred - target) ** 2) * mask).sum() / denom.clamp_min(1.0)


def gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Isotropic Gaussian NLL used by DIP-style pose reconstruction."""
    log_sigma = log_sigma.clamp(min=-7.0, max=5.0)
    inv_var = torch.exp(-2.0 * log_sigma)
    return 0.5 * (((target - mu) ** 2) * inv_var + 2.0 * log_sigma).mean()


class DIPLoss(torch.nn.Module):
    def __init__(self, pose_weight: float = 1.0, obj_weight: float = 10.0):
        super().__init__()
        self.pose_weight = pose_weight
        self.obj_weight = obj_weight

    def forward(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pose = gaussian_nll(output["pose_mu"], output["pose_log_sigma"], batch["pose_target"])
        obj = object_position_loss(output["obj_trans"], batch["obj_trans"], batch["has_object"])
        total = self.pose_weight * pose + self.obj_weight * obj
        return {"loss": total, "pose_nll": pose.detach(), "obj_pos": obj.detach()}


def tip_pose_root_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TIP's pose/root velocity loss split."""
    pose_loss = ((pred[..., :108] - target[..., :108]) ** 2).mean() * 100.0
    root_xy_loss = ((pred[..., 108:110] - target[..., 108:110]) ** 2).mean() * 6.0
    root_z_loss = ((pred[..., 110:111] - target[..., 110:111]) ** 2).mean() * 12.0
    return pose_loss + root_xy_loss + root_z_loss


def tip_constraint_loss(pred_sbp: torch.Tensor, target_sbp: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """TIP SBP/contact constraint loss. Skips samples without SBP labels."""
    if pred_sbp.numel() == 0:
        return pred_sbp.sum() * 0.0
    valid = valid_mask & (~torch.isnan(target_sbp).any(dim=-1))
    if not valid.any():
        return pred_sbp.sum() * 0.0
    pred = pred_sbp[valid]
    target = target_sbp[valid]
    n_c = pred.shape[-1] // 4
    loss = pred.sum() * 0.0
    for i in range(n_c):
        start = 4 * i
        cls = F.binary_cross_entropy_with_logits(pred[:, start : start + 1], target[:, start : start + 1])
        reg = ((pred[:, start + 1 : start + 4] - target[:, start + 1 : start + 4] * 5.0) ** 2).mean()
        loss = loss + cls + reg * 4.0
    return loss / max(n_c, 1) * 2.5


def jerk_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 4:
        return x.sum() * 0.0
    jerk = x[:, 3:] - 3 * x[:, 2:-1] + 3 * x[:, 1:-2] - x[:, :-3]
    return (jerk ** 2).mean() * 100.0


class TIPLoss(torch.nn.Module):
    def __init__(self, n_sbps: int = 5, obj_weight: float = 10.0, jerk_weight: float = 1.0):
        super().__init__()
        self.n_sbps = n_sbps
        self.obj_weight = obj_weight
        self.jerk_weight = jerk_weight

    def forward(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        human_dim = 111 + self.n_sbps * 4
        human_pred = output["state"][:, :, :human_dim]
        target = batch["state_target"]
        pose_root = tip_pose_root_loss(human_pred[:, :, :111], target[:, :, :111])
        sbp = tip_constraint_loss(
            human_pred[:, :, 111:],
            target[:, :, 111:],
            batch["sbp_valid"],
        )
        jerk = jerk_loss(human_pred[:, :, :108])
        obj = object_position_loss(output["obj_trans"], batch["obj_trans"], batch["has_object"])
        total = pose_root + sbp + self.jerk_weight * jerk + self.obj_weight * obj
        return {
            "loss": total,
            "pose_root": pose_root.detach(),
            "sbp": sbp.detach(),
            "jerk": jerk.detach(),
            "obj_pos": obj.detach(),
        }


class TransPoseLoss(torch.nn.Module):
    def __init__(self, obj_weight: float = 10.0):
        super().__init__()
        self.obj_weight = obj_weight

    def forward(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        leaf = F.mse_loss(output["leaf_pos"], batch["leaf_target"])
        full = F.mse_loss(output["full_pos"], batch["full_target"])
        pose = F.mse_loss(output["pose"], batch["pose_target"])
        contact = F.binary_cross_entropy_with_logits(output["contact_logits"], batch["contact_target"])
        root_vel = F.mse_loss(output["root_vel"], batch["root_vel_target"])
        obj = object_position_loss(output["obj_trans"], batch["obj_trans"], batch["has_object"])
        total = leaf + full + pose + contact + root_vel + self.obj_weight * obj
        return {
            "loss": total,
            "leaf": leaf.detach(),
            "full": full.detach(),
            "pose": pose.detach(),
            "contact": contact.detach(),
            "root_vel": root_vel.detach(),
            "obj_pos": obj.detach(),
        }


class GlobalPoseLoss(torch.nn.Module):
    def __init__(self, obj_weight: float = 10.0):
        super().__init__()
        self.obj_weight = obj_weight

    def forward(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = output["human"]
        target = batch["target"]
        pl = F.mse_loss(pred[:, :, :18], target[:, :, :18])
        ik1 = F.mse_loss(pred[:, :, 18:90], target[:, :, 18:90])
        ik2 = F.mse_loss(pred[:, :, 90:180], target[:, :, 90:180])
        vr = F.mse_loss(pred[:, :, 180:189], target[:, :, 180:189])
        obj = object_position_loss(output["obj_trans"], batch["obj_trans"], batch["has_object"])
        total = pl + ik1 + ik2 + vr + self.obj_weight * obj
        return {
            "loss": total,
            "pl": pl.detach(),
            "ik1": ik1.detach(),
            "ik2": ik2.detach(),
            "vr": vr.detach(),
            "obj_pos": obj.detach(),
        }

