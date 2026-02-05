"""
VelocityContactModule loss.
"""
import torch
import torch.nn.functional as F

from configs import _SENSOR_POS_INDICES


class VelocityContactLoss:
    """Loss computation for velocity/contact and interaction boundary."""

    LOSS_KEYS = {
        'obj_vel',
        'hand_vel',
        'obj_move',
        'hand_contact_cond',
        'interaction_boundary',
        'interaction_order',
    }

    def __init__(self, weights=None):
        """
        Args:
            weights: dict of loss weights (defaults to 1.0 if missing).
                - hand_contact: weight for hand_contact_cond
                - interaction_boundary: weight for boundary focal loss
                - interaction_order: weight for start-before-end penalty (default 0)
                - boundary_focal_alpha / boundary_focal_gamma: focal params
        """
        self.weights = weights or {}

    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)

    @staticmethod
    def _focal_binary(pred, target, alpha=0.25, gamma=2.0, eps=1e-6, mask=None):
        """Binary focal loss on probabilities."""
        pred = pred.clamp(min=eps, max=1 - eps)
        pos_term = -alpha * (1 - pred) ** gamma * target * torch.log(pred)
        neg_term = -(1 - alpha) * pred ** gamma * (1 - target) * torch.log(1 - pred)
        loss = pos_term + neg_term
        if mask is not None:
            loss = loss * mask
            denom = mask.sum().clamp_min(eps)
            return loss.sum() / denom
        return loss.mean()

    def compute_loss(self, pred_dict, batch, device):
        human_imu = batch['human_imu'].to(device)
        dtype = human_imu.dtype
        bs, seq = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)

        losses = {key: zero.clone() for key in self.LOSS_KEYS}

        # GT
        sensor_vel_glb_gt = batch.get('sensor_vel_glb')
        if isinstance(sensor_vel_glb_gt, torch.Tensor):
            sensor_vel_glb_gt = sensor_vel_glb_gt.to(device)
            if sensor_vel_glb_gt.dim() == 3:
                sensor_vel_glb_gt = sensor_vel_glb_gt.unsqueeze(0).expand(bs, -1, -1, -1)
        else:
            sensor_vel_glb_gt = torch.zeros(bs, seq, len(_SENSOR_POS_INDICES), 3, device=device, dtype=dtype)

        obj_vel_gt = batch.get('obj_vel')
        if isinstance(obj_vel_gt, torch.Tensor):
            obj_vel_gt = obj_vel_gt.to(device)
        else:
            obj_vel_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

        lhand_contact_gt = batch.get('lhand_contact')
        if isinstance(lhand_contact_gt, torch.Tensor):
            lhand_contact_gt = lhand_contact_gt.to(device).bool()
        else:
            lhand_contact_gt = torch.zeros(bs, seq, device=device, dtype=torch.bool)

        rhand_contact_gt = batch.get('rhand_contact')
        if isinstance(rhand_contact_gt, torch.Tensor):
            rhand_contact_gt = rhand_contact_gt.to(device).bool()
        else:
            rhand_contact_gt = torch.zeros(bs, seq, device=device, dtype=torch.bool)

        obj_contact_gt = batch.get('obj_contact')
        if isinstance(obj_contact_gt, torch.Tensor):
            obj_contact_gt = obj_contact_gt.to(device).bool()
        else:
            obj_contact_gt = torch.zeros(bs, seq, device=device, dtype=torch.bool)

        has_object_mask = batch.get('has_object')
        if has_object_mask is not None:
            has_object_mask = torch.as_tensor(has_object_mask, device=device, dtype=dtype).view(bs, 1, 1)

        # obj vel
        if 'pred_obj_vel' in pred_dict:
            losses['obj_vel'] = F.mse_loss(pred_dict['pred_obj_vel'], obj_vel_gt)

        # hand vel
        if 'pred_hand_glb_vel' in pred_dict:
            hand_indices = [-2, -1]
            gt_hand_vel = sensor_vel_glb_gt[:, :, hand_indices, :]
            losses['hand_vel'] = F.mse_loss(pred_dict['pred_hand_glb_vel'], gt_hand_vel)

        # obj move (use logits for numerical stability)
        obj_move_pred = pred_dict.get('pred_obj_move_prob')
        contact_logits = pred_dict.get('pred_hand_contact_logits')
        obj_move_logits = None
        if contact_logits is not None and contact_logits.shape[-1] >= 3:
            obj_move_logits = contact_logits[..., 2:3]
        if obj_move_logits is not None:
            losses['obj_move'] = F.binary_cross_entropy_with_logits(
                obj_move_logits, obj_contact_gt.float().unsqueeze(-1)
            )
        elif obj_move_pred is not None:
            # fallback to prob if logits missing
            losses['obj_move'] = F.binary_cross_entropy(
                obj_move_pred, obj_contact_gt.float().unsqueeze(-1)
            )

        # conditional hand contact (only when object moving)
        hand_contact_prob = pred_dict.get('pred_hand_contact_prob_cond')
        hand_logits_cond = pred_dict.get('pred_hand_contact_logits_cond')
        if hand_logits_cond is None:
            if contact_logits is not None and contact_logits.shape[-1] >= 2:
                hand_logits_cond = contact_logits[..., :2]
        if hand_contact_prob is None:
            if hand_logits_cond is not None:
                hand_contact_prob = torch.sigmoid(hand_logits_cond)
        if hand_logits_cond is not None:
            hand_contact_gt_2 = torch.stack(
                [lhand_contact_gt.float(), rhand_contact_gt.float()], dim=-1
            )
            cond_loss = F.binary_cross_entropy_with_logits(
                hand_logits_cond, hand_contact_gt_2.float(), reduction='none'
            )
            mask = obj_contact_gt.float()
            losses['hand_contact_cond'] = (
                (cond_loss * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1e-6)
                if mask.sum() > 0
                else zero.clone()
            )
        elif hand_contact_prob is not None:
            # fallback to prob path if logits missing
            hand_contact_gt_2 = torch.stack(
                [lhand_contact_gt.float(), rhand_contact_gt.float()], dim=-1
            )
            cond_loss = F.binary_cross_entropy(hand_contact_prob, hand_contact_gt_2.float(), reduction='none')
            mask = obj_contact_gt.float()
            losses['hand_contact_cond'] = (
                (cond_loss * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1e-6)
                if mask.sum() > 0
                else zero.clone()
            )

        # interaction boundary focal loss
        boundary_pred = pred_dict.get('pred_interaction_boundary_prob')
        if boundary_pred is not None:
            start_gt = batch.get('interaction_start_gauss', batch.get('interaction_start'))
            end_gt = batch.get('interaction_end_gauss', batch.get('interaction_end'))
            if isinstance(start_gt, torch.Tensor):
                start_gt = start_gt.to(device=device, dtype=dtype)
            else:
                start_gt = torch.zeros(bs, seq, device=device, dtype=dtype)
            if isinstance(end_gt, torch.Tensor):
                end_gt = end_gt.to(device=device, dtype=dtype)
            else:
                end_gt = torch.zeros(bs, seq, device=device, dtype=dtype)
            if start_gt.dim() == 2:
                start_gt = start_gt
            if end_gt.dim() == 2:
                end_gt = end_gt
            boundary_gt = torch.stack([start_gt, end_gt], dim=-1)
            alpha = float(self.weights.get('boundary_focal_alpha', 0.25))
            gamma = float(self.weights.get('boundary_focal_gamma', 2.0))
            mask = has_object_mask if has_object_mask is not None else None
            losses['interaction_boundary'] = self._focal_binary(boundary_pred, boundary_gt, alpha=alpha, gamma=gamma, mask=mask)

            # order penalty: start should be before end (soft expectation)
            t = torch.arange(seq, device=device, dtype=dtype).view(1, seq)
            start_prob = boundary_pred[..., 0]
            end_prob = boundary_pred[..., 1]
            start_mass = start_prob.sum(dim=1)
            end_mass = end_prob.sum(dim=1)
            valid_mask = (start_mass > 1e-3) & (end_mass > 1e-3)
            if valid_mask.any():
                start_time = (start_prob[valid_mask] * t).sum(dim=1) / start_mass[valid_mask]
                end_time = (end_prob[valid_mask] * t).sum(dim=1) / end_mass[valid_mask]
                order_loss = F.relu(start_time - end_time)
                if has_object_mask is not None:
                    mask_flat = has_object_mask.view(bs)[valid_mask]
                    if mask_flat.numel() > 0 and mask_flat.sum() > 0:
                        order_loss = (order_loss * mask_flat).sum() / mask_flat.sum()
                    else:
                        order_loss = zero.clone()
                else:
                    order_loss = order_loss.mean()
                losses['interaction_order'] = order_loss
            else:
                losses['interaction_order'] = zero.clone()

        # weighted sum
        total_loss = zero.clone()
        weighted_losses = {}
        for key, loss in losses.items():
            weight = self.weights.get(key, 1.0)
            weighted_losses[key] = loss * weight
            total_loss = total_loss + weighted_losses[key]

        return total_loss, losses, weighted_losses

    @classmethod
    def get_loss_keys(cls):
        """Return loss keys."""
        return list(cls.LOSS_KEYS)
