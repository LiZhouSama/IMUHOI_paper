"""Loss functions matching the comparison baseline protocols."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def object_motion_loss(pred: torch.Tensor, target: torch.Tensor, has_object: torch.Tensor) -> torch.Tensor:
    """Masked object-motion MSE (velocity for corrected checkpoints)."""
    if has_object is None:
        return F.mse_loss(pred, target)
    mask = has_object.to(device=pred.device, dtype=pred.dtype).view(-1, 1, 1)
    denom = mask.sum() * pred.shape[1] * pred.shape[2]
    if denom.item() == 0:
        return pred.sum() * 0.0
    return (((pred - target) ** 2) * mask).sum() / denom.clamp_min(1.0)


def object_motion_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    has_object: torch.Tensor,
) -> torch.Tensor:
    """Masked object-motion Smooth-L1 used by the GlobalPose extension."""
    if has_object is None:
        return F.smooth_l1_loss(pred, target)
    mask = has_object.to(device=pred.device, dtype=pred.dtype).view(-1, 1, 1)
    denom = mask.sum() * pred.shape[1] * pred.shape[2]
    if denom.item() == 0:
        return pred.sum() * 0.0
    values = F.smooth_l1_loss(pred, target, reduction="none")
    return (values * mask).sum() / denom.clamp_min(1.0)


def gaussian_nll(mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """DIP's diagonal Gaussian NLL, summed over pose channels."""
    variance = sigma.square().clamp_min(1e-6)
    element_nll = 0.5 * (
        torch.log(2.0 * math.pi * variance) + (target - mu).square() / variance
    )
    return element_nll.sum(dim=-1).mean()


class DIPLoss(torch.nn.Module):
    def __init__(self, pose_weight: float = 1.0, obj_weight: float = 1.0):
        super().__init__()
        self.pose_weight = float(pose_weight)
        self.obj_weight = float(obj_weight)

    def forward(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # The released OMOMO DIP checkpoint regresses raw 3x3 matrices with an
        # MSE objective.  The legacy adapter used a diagonal Gaussian head;
        # applying its NLL to the reference model makes the scale and gradients
        # incompatible with the published training protocol.
        is_reference = bool(output.get("dip_reference", False))
        if is_reference:
            pose = F.mse_loss(output["pose"], batch["pose_target"])
        else:
            pose = gaussian_nll(
                output["pose_mu"], output["pose_sigma"], batch["pose_target"]
            )
        obj_velocity = object_motion_loss(
            output["obj_trans"], batch["obj_vel_target"], batch["has_object"]
        )
        if is_reference:
            # Official DIP trains the object branch on velocity only; position
            # is obtained by integration at inference time.
            obj_position = output["obj_position_relative"].sum() * 0.0
        else:
            obj_position = object_motion_loss(
                output["obj_position_relative"],
                batch["obj_trans_target"],
                batch["has_object"],
            )
        total = self.pose_weight * pose + self.obj_weight * (
            obj_velocity + obj_position
        )
        return {
            "loss": total,
            "pose_loss": pose.detach(),
            "pose_mse": pose.detach() if is_reference else pose.detach() * 0.0,
            "pose_nll": pose.detach(),
            "obj_velocity": obj_velocity.detach(),
            "obj_position": obj_position.detach(),
        }


def tip_pose_root_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TIP's pose/root velocity loss split."""
    pose_loss = ((pred[..., :108] - target[..., :108]) ** 2).mean() * 100.0
    # TIP is Z-up. After mapping its state to IMUHOI's Y-up convention, x/z
    # are horizontal and y remains the vertically weighted component.
    horizontal = pred[..., (108, 110)] - target[..., (108, 110)]
    vertical = pred[..., 109:110] - target[..., 109:110]
    return pose_loss + horizontal.square().mean() * 6.0 + vertical.square().mean() * 12.0


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
    def __init__(self, n_sbps: int = 5, obj_weight: float = 100.0, jerk_weight: float = 1.0):
        super().__init__()
        self.n_sbps = n_sbps
        self.obj_weight = obj_weight
        self.jerk_weight = jerk_weight

    def forward(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if bool(output.get("tip_reference", False)) or batch.get("tip_variant") == "official":
            pred = output["state"]
            target = batch["state_target"]
            if bool(output.get("includes_initial_state", False)):
                pred = pred[:, 1:]
                target = target[:, 1:]
            elif pred.shape[1] == target.shape[1] - 1:
                target = target[:, 1:]
            if pred.shape[:2] != target.shape[:2]:
                raise ValueError(
                    "Official TIP output/target lengths do not match: "
                    f"{tuple(pred.shape)} vs {tuple(target.shape)}"
                )
            pose_root = tip_pose_root_loss(pred[:, :, :111], target[:, :, :111])
            jerk = jerk_loss(pred[:, :, :108])
            obj = object_motion_loss(
                pred[:, :, -3:], target[:, :, -3:], batch["has_object"]
            )
            total = pose_root + self.jerk_weight * jerk + self.obj_weight * obj
            return {
                "loss": total,
                "pose_root": pose_root.detach(),
                "sbp": pred.sum() * 0.0,
                "jerk": jerk.detach(),
                "obj_pos": obj.detach(),
            }
        human_dim = 111 + self.n_sbps * 4
        human_pred = output["state"][:, :, :human_dim]
        obj_pred = output["obj_trans"]
        target = batch["state_target"]
        sbp_valid_mask = batch["sbp_valid"]
        obj_target = batch["obj_vel_target"]
        if bool(output.get("includes_initial_state", False)):
            if human_pred.shape[1] != target.shape[1]:
                raise ValueError(
                    "TIP output marked with an initial state must match target length"
                )
            human_pred = human_pred[:, 1:]
            obj_pred = obj_pred[:, 1:]
            target = target[:, 1:]
            sbp_valid_mask = sbp_valid_mask[:, 1:]
            obj_target = obj_target[:, 1:]
        elif human_pred.shape[1] == target.shape[1] - 1:
            target = target[:, 1:]
            sbp_valid_mask = sbp_valid_mask[:, 1:]
            obj_target = obj_target[:, 1:]
        elif human_pred.shape[1] != target.shape[1]:
            raise ValueError(
                "TIP output length must equal target length or target length - 1, "
                f"got {human_pred.shape[1]} and {target.shape[1]}"
            )
        pose_root = tip_pose_root_loss(human_pred[:, :, :111], target[:, :, :111])
        sbp = tip_constraint_loss(
            human_pred[:, :, 111:],
            target[:, :, 111:],
            sbp_valid_mask,
        )
        jerk = jerk_loss(human_pred[:, :, :108])
        obj = object_motion_loss(obj_pred, obj_target, batch["has_object"])
        total = pose_root + sbp + self.jerk_weight * jerk + self.obj_weight * obj
        return {
            "loss": total,
            "pose_root": pose_root.detach(),
            "sbp": sbp.detach(),
            "jerk": jerk.detach(),
            "obj_pos": obj.detach(),
        }


class TransPoseLoss(torch.nn.Module):
    def __init__(
        self,
        obj_weight: float = 1.0,
        velocity_scales: tuple[int, ...] = (1, 3, 9, 27),
    ):
        super().__init__()
        self.obj_weight = obj_weight
        self.velocity_scales = velocity_scales

    def _multi_scale_velocity_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        losses = []
        batch_size, seq_len, channels = pred.shape
        for scale in self.velocity_scales:
            blocks = seq_len // scale
            if scale <= 0 or blocks == 0:
                continue
            end = blocks * scale
            pred_block = pred[:, :end].reshape(
                batch_size, blocks, scale, channels
            ).mean(dim=2)
            target_block = target[:, :end].reshape(
                batch_size, blocks, scale, channels
            ).mean(dim=2)
            losses.append(
                ((pred_block - target_block).square().sum(dim=-1) * scale).mean()
            )
        if not losses:
            return pred.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if batch.get("transpose_variant") == "official":
            # The released TransPose checkpoint is supervised with the
            # original multi-branch targets.  Its model wrapper exposes the
            # same branch tensors, but translation/object outputs are already
            # velocities in the reference units.
            leaf = F.mse_loss(output["leaf_pos"], batch["leaf_target"])
            full = F.mse_loss(output["full_pos"], batch["full_target"])
            pose = F.mse_loss(output["pose"], batch["pose_target"])
            contact = F.binary_cross_entropy_with_logits(
                output["contact_logits"], batch["contact_target"]
            )
            root_vel_local = self._multi_scale_velocity_loss(
                output["root_velocity_local"], batch["root_vel_local_target"]
            )
            root_vel_world = F.mse_loss(
                output["root_velocity_world"], batch["root_vel_world_target"]
            )
            root_position = F.mse_loss(
                output["root_position_relative"], batch["root_position_target"]
            )
            obj_velocity = object_motion_loss(
                output["obj_trans"], batch["obj_vel_target"], batch["has_object"]
            )
            obj_position = object_motion_loss(
                output["obj_position_relative"],
                batch["obj_trans_target"],
                batch["has_object"],
            )
            total = (
                leaf + full + 5.0 * pose + contact + 0.05 * root_vel_local
                + root_vel_world + root_position
                + self.obj_weight * (0.1 * obj_velocity + 0.5 * obj_position)
            )
            return {
                "loss": total,
                "leaf": leaf.detach(), "full": full.detach(),
                "pose": pose.detach(), "contact": contact.detach(),
                "root_vel_local": root_vel_local.detach(),
                "root_vel_world": root_vel_world.detach(),
                "root_position": root_position.detach(),
                "obj_velocity": obj_velocity.detach(),
                "obj_position": obj_position.detach(),
            }
        leaf = F.mse_loss(output["leaf_pos"], batch["leaf_target"])
        full = F.mse_loss(output["full_pos"], batch["full_target"])
        pose = F.mse_loss(output["pose"], batch["pose_target"])
        contact = F.binary_cross_entropy_with_logits(output["contact_logits"], batch["contact_target"])
        root_vel_local = self._multi_scale_velocity_loss(
            output["root_velocity_local"], batch["root_vel_local_target"]
        )
        root_vel_world = F.mse_loss(
            output["root_velocity_world"], batch["root_vel_world_target"]
        )
        root_position = F.mse_loss(
            output["root_position_relative"], batch["root_position_target"]
        )
        obj_velocity = object_motion_loss(
            output["obj_trans"], batch["obj_vel_target"], batch["has_object"]
        )
        obj_position = object_motion_loss(
            output["obj_position_relative"],
            batch["obj_trans_target"],
            batch["has_object"],
        )
        total = (
            leaf
            + full
            + 5.0 * pose
            + contact
            + 0.05 * root_vel_local
            + root_vel_world
            + root_position
            + self.obj_weight * (0.1 * obj_velocity + 0.5 * obj_position)
        )
        return {
            "loss": total,
            "leaf": leaf.detach(),
            "full": full.detach(),
            "pose": pose.detach(),
            "contact": contact.detach(),
            "root_vel_local": root_vel_local.detach(),
            "root_vel_world": root_vel_world.detach(),
            "root_position": root_position.detach(),
            "obj_velocity": obj_velocity.detach(),
            "obj_position": obj_position.detach(),
        }


class GlobalPoseLoss(torch.nn.Module):
    def __init__(self, obj_weight: float = 1.0):
        super().__init__()
        self.obj_weight = obj_weight

    def forward(self, output: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = output["human"]
        target = batch["target"]
        pl_position = F.smooth_l1_loss(pred[:, :, :15], target[:, :, :15])
        pl_gravity = self._direction_loss(pred[:, :, 15:18], target[:, :, 15:18])
        ik_position = F.smooth_l1_loss(pred[:, :, 18:87], target[:, :, 18:87])
        ik_gravity = self._direction_loss(pred[:, :, 87:90], target[:, :, 87:90])
        ik_rotation = F.mse_loss(pred[:, :, 90:180], target[:, :, 90:180])
        vr_velocity = F.smooth_l1_loss(pred[:, :, 180:184], target[:, :, 180:184])
        vr_stationary = F.binary_cross_entropy_with_logits(
            pred[:, :, 184:189], target[:, :, 184:189]
        )
        obj = object_motion_smooth_l1_loss(
            output["obj_trans"], batch["obj_vel_target"], batch["has_object"]
        )
        total = (
            pl_position
            + 0.5 * pl_gravity
            + ik_position
            + 0.5 * ik_gravity
            + ik_rotation
            + 0.5 * vr_velocity
            + vr_stationary
            + self.obj_weight * obj
        )
        return {
            "loss": total,
            "pl_position": pl_position.detach(),
            "pl_gravity": pl_gravity.detach(),
            "ik_position": ik_position.detach(),
            "ik_gravity": ik_gravity.detach(),
            "ik_rotation": ik_rotation.detach(),
            "vr_velocity": vr_velocity.detach(),
            "vr_stationary": vr_stationary.detach(),
            "obj_pos": obj.detach(),
        }

    @staticmethod
    def _direction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return (1.0 - (pred * target).sum(dim=-1)).mean()
