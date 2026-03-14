"""
Loss for merged interaction module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, zero: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return zero.clone()
    return F.mse_loss(pred[mask_bool], target[mask_bool])


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, zero: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return zero.clone()
    return values[mask_bool].mean()


class InteractionLoss:
    """Main losses: obj_trans + contact logits; others auxiliary."""

    LOSS_KEYS = {
        "obj_trans",
        "contact_logits",
        "interaction_boundary",
        "interaction_order",
        "geometry_dir_l",
        "geometry_dir_r",
        "geometry_len_l",
        "geometry_len_r",
        "hoi_error_l",
        "hoi_error_r",
        "obj_vel_cons",
        "obj_acc_cons",
        "diffusion_eps",
    }

    TEST_LOSS_KEYS = {
        "obj_trans",
        "contact_logits",
        "interaction_boundary",
        "geometry_dir_l",
        "geometry_dir_r",
        "geometry_len_l",
        "geometry_len_r",
        "hoi_error_l",
        "hoi_error_r",
    }

    def __init__(self, weights=None):
        self.weights = weights or {}

    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)

    @staticmethod
    def _prepare_bt(tensor, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype, default=0.0):
        if isinstance(tensor, torch.Tensor):
            out = tensor.to(device=device, dtype=dtype)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            if out.shape[0] == 1 and batch_size > 1:
                out = out.expand(batch_size, *out.shape[1:])
            return out
        return torch.full((batch_size, seq_len), float(default), device=device, dtype=dtype)

    def compute_loss(self, pred_dict, batch, device):
        human_imu = batch["human_imu"].to(device)
        dtype = human_imu.dtype
        bs, seq = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)

        losses = {key: zero.clone() for key in self.LOSS_KEYS}

        obj_trans_gt = batch.get("obj_trans")
        if isinstance(obj_trans_gt, torch.Tensor):
            obj_trans_gt = obj_trans_gt.to(device=device, dtype=dtype)
        else:
            obj_trans_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

        obj_vel_gt = batch.get("obj_vel")
        if isinstance(obj_vel_gt, torch.Tensor):
            obj_vel_gt = obj_vel_gt.to(device=device, dtype=dtype)
        else:
            obj_vel_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

        obj_imu_gt = batch.get("obj_imu")
        if isinstance(obj_imu_gt, torch.Tensor):
            obj_imu_gt = obj_imu_gt.to(device=device, dtype=dtype)
        else:
            obj_imu_gt = None

        has_object = batch.get("has_object")
        if isinstance(has_object, torch.Tensor):
            has_object_mask = has_object.to(device=device, dtype=torch.bool)
            if has_object_mask.dim() == 0:
                has_object_mask = has_object_mask.view(1)
            if has_object_mask.dim() == 1:
                if has_object_mask.shape[0] == 1 and bs > 1:
                    has_object_mask = has_object_mask.expand(bs)
                has_object_mask = has_object_mask.unsqueeze(1).expand(bs, seq)
        else:
            has_object_mask = torch.ones(bs, seq, device=device, dtype=torch.bool)

        if has_object_mask.shape[0] != bs:
            has_object_mask = has_object_mask[0].view(1, -1).expand(bs, seq)

        obj_mask = has_object_mask
        obj_mask_f = obj_mask.float()

        pred_obj_trans = pred_dict.get("pred_obj_trans")
        if isinstance(pred_obj_trans, torch.Tensor):
            losses["obj_trans"] = _masked_mse(pred_obj_trans, obj_trans_gt, obj_mask, zero)

        contact_logits = pred_dict.get("pred_hand_contact_logits")
        if isinstance(contact_logits, torch.Tensor):
            lhand = self._prepare_bt(batch.get("lhand_contact"), bs, seq, device, dtype)
            rhand = self._prepare_bt(batch.get("rhand_contact"), bs, seq, device, dtype)
            objc = self._prepare_bt(batch.get("obj_contact"), bs, seq, device, dtype)
            contact_gt = torch.stack([lhand, rhand, objc], dim=-1).clamp(0.0, 1.0)
            bce = F.binary_cross_entropy_with_logits(contact_logits, contact_gt, reduction="none")
            denom = obj_mask_f.sum().clamp_min(1e-6)
            losses["contact_logits"] = (bce * obj_mask_f.unsqueeze(-1)).sum() / denom

        boundary_logits = pred_dict.get("pred_interaction_boundary_logits")
        boundary_prob = pred_dict.get("pred_interaction_boundary_prob")
        start_gt = self._prepare_bt(batch.get("interaction_start_gauss", batch.get("interaction_start")), bs, seq, device, dtype)
        end_gt = self._prepare_bt(batch.get("interaction_end_gauss", batch.get("interaction_end")), bs, seq, device, dtype)
        boundary_gt = torch.stack([start_gt, end_gt], dim=-1)

        if isinstance(boundary_logits, torch.Tensor):
            boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_gt, reduction="none")
            losses["interaction_boundary"] = (boundary_loss * obj_mask_f.unsqueeze(-1)).sum() / obj_mask_f.sum().clamp_min(1e-6)
            boundary_prob_for_order = torch.sigmoid(boundary_logits)
        elif isinstance(boundary_prob, torch.Tensor):
            boundary_loss = F.binary_cross_entropy(boundary_prob, boundary_gt, reduction="none")
            losses["interaction_boundary"] = (boundary_loss * obj_mask_f.unsqueeze(-1)).sum() / obj_mask_f.sum().clamp_min(1e-6)
            boundary_prob_for_order = boundary_prob
        else:
            boundary_prob_for_order = None

        if isinstance(boundary_prob_for_order, torch.Tensor):
            t = torch.arange(seq, device=device, dtype=dtype).view(1, seq)
            start_prob = boundary_prob_for_order[..., 0]
            end_prob = boundary_prob_for_order[..., 1]
            start_mass = start_prob.sum(dim=1)
            end_mass = end_prob.sum(dim=1)
            valid = (start_mass > 1e-3) & (end_mass > 1e-3)
            if valid.any():
                start_time = (start_prob[valid] * t).sum(dim=1) / start_mass[valid]
                end_time = (end_prob[valid] * t).sum(dim=1) / end_mass[valid]
                order_loss = F.relu(start_time - end_time)
                losses["interaction_order"] = order_loss.mean()

        pred_vel = pred_dict.get("pred_obj_vel_from_posdiff")
        if isinstance(pred_vel, torch.Tensor):
            losses["obj_vel_cons"] = _masked_mse(pred_vel, obj_vel_gt, obj_mask, zero)

        pred_acc = pred_dict.get("pred_obj_acc_from_posdiff")
        if isinstance(pred_acc, torch.Tensor) and isinstance(obj_imu_gt, torch.Tensor):
            losses["obj_acc_cons"] = _masked_mse(pred_acc, obj_imu_gt[:, :, :3], obj_mask, zero)

        pred_hand_pos = pred_dict.get("pred_hand_glb_pos")
        if isinstance(pred_obj_trans, torch.Tensor) and isinstance(pred_hand_pos, torch.Tensor):
            pred_hand_pos = pred_hand_pos.to(device=device, dtype=dtype)
            rel_l = pred_obj_trans - pred_hand_pos[:, :, 0, :]
            rel_r = pred_obj_trans - pred_hand_pos[:, :, 1, :]

            pred_len_l = torch.norm(rel_l, dim=-1)
            pred_len_r = torch.norm(rel_r, dim=-1)
            pred_dir_l = rel_l / pred_len_l.unsqueeze(-1).clamp_min(1e-6)
            pred_dir_r = rel_r / pred_len_r.unsqueeze(-1).clamp_min(1e-6)

            gt_dir_l = batch.get("lhand_obj_direction")
            gt_dir_r = batch.get("rhand_obj_direction")
            gt_len_l = batch.get("lhand_lb")
            gt_len_r = batch.get("rhand_lb")

            if isinstance(gt_dir_l, torch.Tensor):
                gt_dir_l = gt_dir_l.to(device=device, dtype=dtype)
            if isinstance(gt_dir_r, torch.Tensor):
                gt_dir_r = gt_dir_r.to(device=device, dtype=dtype)
            if isinstance(gt_len_l, torch.Tensor):
                gt_len_l = gt_len_l.to(device=device, dtype=dtype)
            if isinstance(gt_len_r, torch.Tensor):
                gt_len_r = gt_len_r.to(device=device, dtype=dtype)

            lhand_contact = self._prepare_bt(batch.get("lhand_contact"), bs, seq, device, dtype) > 0.5
            rhand_contact = self._prepare_bt(batch.get("rhand_contact"), bs, seq, device, dtype) > 0.5
            mask_l = lhand_contact & obj_mask
            mask_r = rhand_contact & obj_mask

            if isinstance(gt_dir_l, torch.Tensor):
                losses["geometry_dir_l"] = _masked_mse(pred_dir_l, gt_dir_l, mask_l, zero)
            if isinstance(gt_dir_r, torch.Tensor):
                losses["geometry_dir_r"] = _masked_mse(pred_dir_r, gt_dir_r, mask_r, zero)
            if isinstance(gt_len_l, torch.Tensor):
                losses["geometry_len_l"] = _masked_mse(pred_len_l, gt_len_l, mask_l, zero)
            if isinstance(gt_len_r, torch.Tensor):
                losses["geometry_len_r"] = _masked_mse(pred_len_r, gt_len_r, mask_r, zero)

            if isinstance(gt_dir_l, torch.Tensor) and isinstance(gt_len_l, torch.Tensor):
                vec_pred_l = pred_dir_l * pred_len_l.unsqueeze(-1)
                vec_gt_l = gt_dir_l * gt_len_l.unsqueeze(-1)
                losses["hoi_error_l"] = _masked_mean(torch.norm(vec_pred_l - vec_gt_l, dim=-1), mask_l, zero)
            if isinstance(gt_dir_r, torch.Tensor) and isinstance(gt_len_r, torch.Tensor):
                vec_pred_r = pred_dir_r * pred_len_r.unsqueeze(-1)
                vec_gt_r = gt_dir_r * gt_len_r.unsqueeze(-1)
                losses["hoi_error_r"] = _masked_mean(torch.norm(vec_pred_r - vec_gt_r, dim=-1), mask_r, zero)

        diffusion_aux = pred_dict.get("diffusion_aux", {})
        eps_pred = diffusion_aux.get("eps_pred") if isinstance(diffusion_aux, dict) else None
        noise = diffusion_aux.get("noise") if isinstance(diffusion_aux, dict) else None
        if isinstance(eps_pred, torch.Tensor) and isinstance(noise, torch.Tensor):
            losses["diffusion_eps"] = F.mse_loss(eps_pred, noise)

        total_loss = zero.clone()
        weighted_losses = {}
        for key, loss in losses.items():
            weight = float(self.weights.get(key, 1.0))
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
