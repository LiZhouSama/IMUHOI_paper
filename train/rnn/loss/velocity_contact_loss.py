"""
VelocityContactModule loss.
"""
import torch
import torch.nn.functional as F

from configs import _SENSOR_POS_INDICES


class VelocityContactLoss:
    """Loss computation for velocity, dense contact and contact endpoints."""

    LOSS_KEYS = {
        'obj_vel',
        'hand_vel',
        'obj_move',
        'hand_contact_cond',
        'contact_endpoint',
        'contact_endpoint_order',
        'endpoint_contact_consistency',
    }

    TEST_LOSS_KEYS = {
        'obj_vel',
        'obj_move',
        'hand_contact_cond',
        'contact_endpoint',
        'endpoint_contact_consistency',
    }

    def __init__(self, weights=None):
        """
        Args:
            weights: dict of loss weights (defaults to 1.0 if missing).
                - hand_contact: weight for hand_contact_cond
                - contact_endpoint: endpoint heatmap focal loss
                - endpoint_contact_consistency: dense endpoint/contact agreement
        """
        self.weights = weights or {}

    def __call__(self, pred_dict, batch, device, active_loss_keys=None):
        return self.compute_loss(
            pred_dict,
            batch,
            device,
            active_loss_keys=active_loss_keys,
        )

    @staticmethod
    def _focal_binary(pred, target, alpha=0.25, gamma=2.0, eps=1e-6, mask=None):
        """Binary focal loss on probabilities."""
        pred = pred.clamp(min=eps, max=1 - eps)
        pos_term = -alpha * (1 - pred) ** gamma * target * torch.log(pred)
        neg_term = -(1 - alpha) * pred ** gamma * (1 - target) * torch.log(1 - pred)
        loss = pos_term + neg_term
        if mask is not None:
            mask = mask.to(device=loss.device, dtype=loss.dtype)
            while mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(loss)
            loss = loss * mask
            denom = mask.sum().clamp_min(eps)
            return loss.sum() / denom
        return loss.mean()

    @staticmethod
    def _soft_order_penalty(start_prob, end_prob, has_object_mask, zero):
        """Penalize the soft-expected start time exceeding the end time."""
        bs, seq = start_prob.shape[:2]
        t = torch.arange(seq, device=start_prob.device, dtype=start_prob.dtype).view(1, seq)
        start_mass = start_prob.sum(dim=1)
        end_mass = end_prob.sum(dim=1)
        valid_mask = (start_mass > 1e-3) & (end_mass > 1e-3)
        if not valid_mask.any():
            return zero.clone()
        start_time = (start_prob[valid_mask] * t).sum(dim=1) / start_mass[valid_mask]
        end_time = (end_prob[valid_mask] * t).sum(dim=1) / end_mass[valid_mask]
        order_loss = F.relu(start_time - end_time)
        if has_object_mask is not None:
            mask_flat = has_object_mask.view(bs)[valid_mask]
            if mask_flat.numel() > 0 and mask_flat.sum() > 0:
                return (order_loss * mask_flat).sum() / mask_flat.sum()
            return zero.clone()
        return order_loss.mean()

    def compute_loss(self, pred_dict, batch, device, active_loss_keys=None):
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
            pos_weight = float(self.weights.get('hand_contact_pos_weight', 1.0))
            neg_weight = float(self.weights.get('hand_contact_neg_weight', 1.0))
            hard_neg_weight = float(self.weights.get('hand_contact_hard_neg_weight', neg_weight))
            elem_weight = torch.where(
                hand_contact_gt_2 > 0.5,
                torch.full_like(hand_contact_gt_2, pos_weight),
                torch.full_like(hand_contact_gt_2, neg_weight),
            )
            hard_neg_mask = (
                obj_contact_gt
                & (~lhand_contact_gt)
                & (~rhand_contact_gt)
            ).unsqueeze(-1)
            elem_weight = torch.where(
                hard_neg_mask & (hand_contact_gt_2 <= 0.5),
                torch.full_like(elem_weight, hard_neg_weight),
                elem_weight,
            )
            valid_weight = elem_weight * mask.unsqueeze(-1)
            losses['hand_contact_cond'] = (
                (cond_loss * valid_weight).sum() / valid_weight.sum().clamp_min(1e-6)
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
            pos_weight = float(self.weights.get('hand_contact_pos_weight', 1.0))
            neg_weight = float(self.weights.get('hand_contact_neg_weight', 1.0))
            hard_neg_weight = float(self.weights.get('hand_contact_hard_neg_weight', neg_weight))
            elem_weight = torch.where(
                hand_contact_gt_2 > 0.5,
                torch.full_like(hand_contact_gt_2, pos_weight),
                torch.full_like(hand_contact_gt_2, neg_weight),
            )
            hard_neg_mask = (
                obj_contact_gt
                & (~lhand_contact_gt)
                & (~rhand_contact_gt)
            ).unsqueeze(-1)
            elem_weight = torch.where(
                hard_neg_mask & (hand_contact_gt_2 <= 0.5),
                torch.full_like(elem_weight, hard_neg_weight),
                elem_weight,
            )
            valid_weight = elem_weight * mask.unsqueeze(-1)
            losses['hand_contact_cond'] = (
                (cond_loss * valid_weight).sum() / valid_weight.sum().clamp_min(1e-6)
                if mask.sum() > 0
                else zero.clone()
            )

        # contact segment endpoint heatmaps
        # 6 channels: (l_start, l_end, r_start, r_end, obj_start, obj_end)
        endpoint_logits = pred_dict.get('pred_contact_endpoint_logits')
        if endpoint_logits is not None:
            endpoint_gt = batch.get('contact_endpoint_gauss')
            if isinstance(endpoint_gt, torch.Tensor):
                endpoint_gt = endpoint_gt.to(device=device, dtype=dtype)
            else:
                endpoint_gt = torch.zeros(
                    bs, seq, endpoint_logits.shape[-1], device=device, dtype=dtype
                )
            endpoint_prob = torch.sigmoid(endpoint_logits)
            alpha = float(self.weights.get('endpoint_focal_alpha', 0.25))
            gamma = float(self.weights.get('endpoint_focal_gamma', 2.0))
            losses['contact_endpoint'] = self._focal_binary(
                endpoint_prob, endpoint_gt, alpha=alpha, gamma=gamma, mask=has_object_mask
            )

            num_pairs = endpoint_prob.shape[-1] // 2
            order_terms = []
            for pair in range(num_pairs):
                order_terms.append(self._soft_order_penalty(
                    endpoint_prob[..., 2 * pair],
                    endpoint_prob[..., 2 * pair + 1],
                    has_object_mask,
                    zero,
                ))
            if order_terms:
                losses['contact_endpoint_order'] = sum(order_terms) / len(order_terms)

        # Endpoint-recovered dense state should agree with the original
        # per-frame contact prediction. Detaching the dense prediction makes it
        # a teacher for this term, avoiding a mutual-collapse shortcut.
        endpoint_contact = pred_dict.get('pred_contact_from_endpoints_prob')
        frame_contact = pred_dict.get(
            'pred_hand_contact_prob_frame',
            pred_dict.get('pred_hand_contact_prob'),
        )
        if (
            isinstance(endpoint_contact, torch.Tensor)
            and isinstance(frame_contact, torch.Tensor)
            and endpoint_contact.shape == frame_contact.shape
            and seq > 1
        ):
            endpoint_contact = endpoint_contact[:, 1:].clamp(1e-6, 1.0 - 1e-6)
            frame_teacher = frame_contact[:, 1:].detach().clamp(1e-6, 1.0 - 1e-6)
            consistency = F.binary_cross_entropy(
                endpoint_contact,
                frame_teacher,
                reduction='none',
            )
            if has_object_mask is not None:
                consistency_mask = has_object_mask.expand(
                    bs,
                    seq - 1,
                    endpoint_contact.shape[-1],
                )
                losses['endpoint_contact_consistency'] = (
                    (consistency * consistency_mask).sum()
                    / consistency_mask.sum().clamp_min(1e-6)
                )
            else:
                losses['endpoint_contact_consistency'] = consistency.mean()

        # weighted sum
        if active_loss_keys is None:
            active_loss_keys = self.LOSS_KEYS
        else:
            active_loss_keys = set(active_loss_keys)
            unknown_keys = active_loss_keys - self.LOSS_KEYS
            if unknown_keys:
                raise ValueError(
                    f"Unknown VelocityContact loss keys: {sorted(unknown_keys)}"
                )
        total_loss = zero.clone()
        weighted_losses = {}
        for key, loss in losses.items():
            if key not in active_loss_keys:
                continue
            weight = self.weights.get(key, 1.0)
            weighted_losses[key] = loss * weight
            total_loss = total_loss + weighted_losses[key]

        return total_loss, losses, weighted_losses

    def compute_test_loss(self, pred_dict, batch, device):
        """Test loss tracks only downstream-consumed VC outputs.

        Stage3 consumes ``pred_obj_vel`` and ``pred_hand_contact_prob``.  The
        latter is built from conditional hand-contact logits plus object-move
        probability. Test loss keeps those terms together with endpoint
        heatmaps and endpoint/contact consistency, excluding hand velocity.
        """
        _, _, weighted_losses = self.compute_loss(pred_dict, batch, device)
        test_losses = {
            key: value
            for key, value in weighted_losses.items()
            if key in self.TEST_LOSS_KEYS
        }
        if test_losses:
            total_loss = sum(test_losses.values())
        else:
            human_imu = batch['human_imu'].to(device)
            total_loss = human_imu.new_tensor(0.0)
        return total_loss, test_losses
    
    @classmethod
    def get_loss_keys(cls):
        """Return loss keys."""
        return list(cls.LOSS_KEYS)
