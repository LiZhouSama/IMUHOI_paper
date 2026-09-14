"""
VelocityContactModule loss.
"""
import torch
import torch.nn.functional as F

from configs import _SENSOR_POS_INDICES
from model.rnn.velocity_contact_v4 import DEFAULT_ANTICIPATION_HORIZON


# bfloat16 keeps ~8 mantissa bits, so ``sigmoid`` rounds to exactly 1.0 once the
# logit passes ~8 (and to 0.0 symmetrically).  ``binary_cross_entropy`` then sees
# a saturated probability: the loss is capped at 100, but ``d/dp`` is 1/(1-p),
# which is ~1e12 at p == 1.  One confidently wrong frame is enough to push that
# through ``clip_grad_norm_`` and poison the next forward with non-finite
# activations -- which is what killed VC-v4 at epoch 170, long after it had
# converged enough to saturate.  Clamping into the open interval bounds the
# gradient at 1/eps and is a no-op everywhere else.
_BCE_PROBABILITY_EPS = 1e-6


def _binary_cross_entropy_probability(input_tensor, target, reduction="mean"):
    """BCE on probabilities, made safe when the caller uses CUDA AMP."""
    if input_tensor.device.type == "cuda":
        with torch.autocast(device_type="cuda", enabled=False):
            return F.binary_cross_entropy(
                input_tensor.float().clamp(
                    _BCE_PROBABILITY_EPS, 1.0 - _BCE_PROBABILITY_EPS
                ),
                target.float(),
                reduction=reduction,
            )
    return F.binary_cross_entropy(
        input_tensor.clamp(_BCE_PROBABILITY_EPS, 1.0 - _BCE_PROBABILITY_EPS),
        target,
        reduction=reduction,
    )


class VelocityContactLoss:
    """Loss computation for velocity, dense contact and contact endpoints."""

    LOSS_KEYS = {
        'obj_vel',
        'hand_vel',
        'obj_move',
        'hand_contact_cond',
        'hand_contact_marginal',
        'contact_endpoint',
        'contact_endpoint_order',
        'endpoint_contact_consistency',
        'contact_anticipation',
    }

    TEST_LOSS_KEYS = {
        'obj_vel',
        'obj_move',
        'hand_contact_cond',
        'hand_contact_marginal',
        'contact_endpoint',
        'endpoint_contact_consistency',
    }

    def __init__(self, weights=None, anticipation_horizon=None):
        """
        Args:
            weights: dict of loss weights (defaults to 1.0 if missing).
                - hand_contact_cond: conditional hand-contact BCE
                - hand_contact_marginal: proper BCE on the downstream-consumed
                  P(object move) * P(hand contact | object move)
                - contact_endpoint: endpoint heatmap focal loss
                - endpoint_contact_consistency: dense endpoint/contact agreement
                - contact_anticipation: focal loss on "a contact transition
                  occurs within the next k frames"
            anticipation_horizon: k, in frames. Must match the horizon the
                anticipation head was built for (``v4_anticipation_horizon``).
        """
        self.weights = weights or {}
        self.anticipation_horizon = max(
            1,
            int(
                DEFAULT_ANTICIPATION_HORIZON
                if anticipation_horizon is None
                else anticipation_horizon
            ),
        )

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
    def _future_transition_target(endpoint_gt, horizon):
        """Target and validity for "a transition occurs in the next k frames".

        Human pose leads the contact labels: reach and retract are visible
        before the object starts or stops moving, so a per-frame target
        penalizes the pose prior for firing early.  Pooling the hard endpoint
        labels over ``(t, t + horizon]`` turns that lead into the signal.

        Frames whose horizon runs past the window end are invalid rather than
        negative — a transition may sit just outside the window.
        """
        batch_size, seq_len, channels = endpoint_gt.shape
        horizon = int(horizon)
        shifted = torch.cat(
            (
                endpoint_gt[:, 1:],
                endpoint_gt.new_zeros(batch_size, horizon, channels),
            ),
            dim=1,
        )
        target = F.max_pool1d(
            shifted.transpose(1, 2),
            kernel_size=horizon,
            stride=1,
        ).transpose(1, 2)
        frames = torch.arange(seq_len, device=endpoint_gt.device)
        valid = (frames <= seq_len - 1 - horizon).view(1, seq_len).expand(
            batch_size, seq_len
        )
        return target, valid

    @staticmethod
    def _soft_order_penalty(start_prob, end_prob, has_object_mask, zero):
        """Penalize the soft-expected start time exceeding the end time."""
        batch_size, seq_len = start_prob.shape[:2]
        frame_indices = torch.arange(
            seq_len, device=start_prob.device, dtype=start_prob.dtype
        ).view(1, seq_len)
        start_mass = start_prob.sum(dim=1)
        end_mass = end_prob.sum(dim=1)
        valid_mask = (start_mass > 1e-3) & (end_mass > 1e-3)
        start_time = (start_prob * frame_indices).sum(dim=1) / start_mass.clamp_min(1e-3)
        end_time = (end_prob * frame_indices).sum(dim=1) / end_mass.clamp_min(1e-3)
        order_loss = torch.where(valid_mask, F.relu(start_time - end_time), 0.0)
        weights = valid_mask.to(dtype=order_loss.dtype)
        if has_object_mask is not None:
            weights = weights * has_object_mask.view(batch_size)
        weight_sum = weights.sum()
        denominator = torch.where(weight_sum > 0, weight_sum, 1.0)
        return torch.where(weight_sum > 0, (order_loss * weights).sum() / denominator, zero)

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

        # v3 emits this mask so contact supervision only uses frames with the
        # complete fixed-lookahead context. Other architectures retain their
        # previous all-frame behavior.
        contact_valid_mask = pred_dict.get('pred_contact_future_valid_mask')
        if isinstance(contact_valid_mask, torch.Tensor):
            contact_valid_mask = contact_valid_mask.to(device=device).bool()
            if contact_valid_mask.dim() == 3 and contact_valid_mask.shape[-1] == 1:
                contact_valid_mask = contact_valid_mask[..., 0]
            if contact_valid_mask.shape != (bs, seq):
                raise ValueError(
                    'pred_contact_future_valid_mask must be [B,T] or [B,T,1], '
                    f'got {tuple(contact_valid_mask.shape)}'
                )
        else:
            contact_valid_mask = None

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
            obj_move_element = F.binary_cross_entropy_with_logits(
                obj_move_logits,
                obj_contact_gt.float().unsqueeze(-1),
                reduction='none',
            )
            if contact_valid_mask is None:
                losses['obj_move'] = obj_move_element.mean()
            elif contact_valid_mask.any():
                losses['obj_move'] = obj_move_element[
                    contact_valid_mask.unsqueeze(-1)
                ].mean()
        elif obj_move_pred is not None:
            # fallback to prob if logits missing
            obj_move_element = _binary_cross_entropy_probability(
                obj_move_pred,
                obj_contact_gt.float().unsqueeze(-1),
                reduction='none',
            )
            if contact_valid_mask is None:
                losses['obj_move'] = obj_move_element.mean()
            elif contact_valid_mask.any():
                losses['obj_move'] = obj_move_element[
                    contact_valid_mask.unsqueeze(-1)
                ].mean()

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
            if contact_valid_mask is not None:
                mask = mask * contact_valid_mask.to(dtype=mask.dtype)
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
            cond_loss = _binary_cross_entropy_probability(
                hand_contact_prob, hand_contact_gt_2.float(), reduction='none'
            )
            mask = obj_contact_gt.float()
            if contact_valid_mask is not None:
                mask = mask * contact_valid_mask.to(dtype=mask.dtype)
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

        # v3 explicitly opts into supervising the marginal probability that OT
        # and evaluation consume. Conditional BCE alone cannot calibrate the
        # product when object-move confidence is imperfect.
        hand_contact_marginal = pred_dict.get(
            'pred_hand_contact_prob_marginal'
        )
        if isinstance(hand_contact_marginal, torch.Tensor):
            hand_contact_gt_2 = torch.stack(
                [lhand_contact_gt.float(), rhand_contact_gt.float()], dim=-1
            )
            if hand_contact_marginal.shape != hand_contact_gt_2.shape:
                raise ValueError(
                    'pred_hand_contact_prob_marginal must be [B,T,2], '
                    f'got {tuple(hand_contact_marginal.shape)}'
                )
            marginal_element = _binary_cross_entropy_probability(
                hand_contact_marginal.clamp(1e-6, 1.0 - 1e-6),
                hand_contact_gt_2,
                reduction='none',
            )
            if contact_valid_mask is None:
                losses['hand_contact_marginal'] = marginal_element.mean()
            elif contact_valid_mask.any():
                marginal_mask = contact_valid_mask.unsqueeze(-1).expand_as(
                    marginal_element
                )
                losses['hand_contact_marginal'] = marginal_element[
                    marginal_mask
                ].mean()

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

        # Contact anticipation from pose-only boundary features.
        anticipation_logits = pred_dict.get('pred_contact_anticipation_logits')
        endpoint_hard_gt = batch.get('contact_endpoint')
        # Without the hard endpoint labels the target would be all-negative,
        # which trains the head to never anticipate while still producing a
        # healthy-looking decreasing loss. Leaving the term at exactly zero
        # makes the missing label visible in the logs instead.
        if isinstance(anticipation_logits, torch.Tensor) and isinstance(
            endpoint_hard_gt, torch.Tensor
        ):
            channels = int(anticipation_logits.shape[-1])
            endpoint_hard_gt = endpoint_hard_gt.to(device=device, dtype=dtype)
            if endpoint_hard_gt.shape != (bs, seq, channels):
                raise ValueError(
                    "contact_endpoint must be [B,T,"
                    f"{channels}] to supervise the anticipation head, got "
                    f"{tuple(endpoint_hard_gt.shape)}"
                )
            anticipation_gt, horizon_valid = self._future_transition_target(
                endpoint_hard_gt, self.anticipation_horizon
            )
            anticipation_mask = horizon_valid.to(dtype=dtype)
            if has_object_mask is not None:
                anticipation_mask = anticipation_mask * has_object_mask.view(bs, 1)
            if contact_valid_mask is not None:
                anticipation_mask = anticipation_mask * contact_valid_mask.to(
                    dtype=dtype
                )
            if anticipation_mask.sum() > 0:
                losses['contact_anticipation'] = self._focal_binary(
                    torch.sigmoid(anticipation_logits),
                    anticipation_gt,
                    alpha=float(self.weights.get('anticipation_focal_alpha', 0.25)),
                    gamma=float(self.weights.get('anticipation_focal_gamma', 2.0)),
                    mask=anticipation_mask,
                )

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
            consistency = _binary_cross_entropy_probability(
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

        Stage3 consumes ``pred_obj_vel`` and ``pred_hand_contact_prob``. The
        latter is built from conditional hand-contact logits plus object-move
        probability. Test loss keeps the conditional and optional marginal
        supervision together with endpoint terms, excluding hand velocity.
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


class V2VelocityContactLoss:
    """Loss for the direct-contact, velocity-first VC v2 cascade.

    Unlike ``VelocityContactLoss``, the two hand channels are unconditional:
    they are supervised on every frame.  The object channel remains a separate
    object-move/contact target and is also an input to the hand-contact head.
    """

    LOSS_KEYS = {"obj_vel", "hand_vel", "obj_move", "hand_contact"}
    TEST_LOSS_KEYS = {"obj_vel", "obj_move", "hand_contact"}

    def __init__(self, weights=None):
        self.weights = weights or {}

    def __call__(self, pred_dict, batch, device, active_loss_keys=None):
        return self.compute_loss(
            pred_dict,
            batch,
            device,
            active_loss_keys=active_loss_keys,
        )

    @staticmethod
    def _target(batch, key, batch_size, seq_len, device, dtype):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

    def compute_loss(self, pred_dict, batch, device, active_loss_keys=None):
        human_imu = batch["human_imu"].to(device)
        dtype = human_imu.dtype
        batch_size, seq_len = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)
        losses = {key: zero.clone() for key in self.LOSS_KEYS}

        obj_vel_gt = batch.get("obj_vel")
        if isinstance(obj_vel_gt, torch.Tensor):
            obj_vel_gt = obj_vel_gt.to(device=device, dtype=dtype)
        else:
            obj_vel_gt = torch.zeros(
                batch_size, seq_len, 3, device=device, dtype=dtype
            )
        sensor_vel_gt = batch.get("sensor_vel_glb")
        if isinstance(sensor_vel_gt, torch.Tensor):
            sensor_vel_gt = sensor_vel_gt.to(device=device, dtype=dtype)
            if sensor_vel_gt.dim() == 3:
                sensor_vel_gt = sensor_vel_gt.unsqueeze(0).expand(
                    batch_size, -1, -1, -1
                )
        else:
            sensor_vel_gt = torch.zeros(
                batch_size,
                seq_len,
                len(_SENSOR_POS_INDICES),
                3,
                device=device,
                dtype=dtype,
            )

        left_gt = self._target(
            batch, "lhand_contact", batch_size, seq_len, device, dtype
        )
        right_gt = self._target(
            batch, "rhand_contact", batch_size, seq_len, device, dtype
        )
        obj_gt = self._target(
            batch, "obj_contact", batch_size, seq_len, device, dtype
        )
        hand_gt = torch.stack((left_gt, right_gt), dim=-1)

        pred_obj_vel = pred_dict.get("pred_obj_vel")
        if isinstance(pred_obj_vel, torch.Tensor):
            losses["obj_vel"] = F.mse_loss(pred_obj_vel, obj_vel_gt)
        pred_hand_vel = pred_dict.get("pred_hand_glb_vel")
        if isinstance(pred_hand_vel, torch.Tensor):
            losses["hand_vel"] = F.mse_loss(
                pred_hand_vel, sensor_vel_gt[:, :, -2:, :]
            )

        obj_logits = pred_dict.get("pred_obj_move_logits")
        if not isinstance(obj_logits, torch.Tensor):
            contact_logits = pred_dict.get("pred_hand_contact_logits")
            if (
                isinstance(contact_logits, torch.Tensor)
                and contact_logits.shape[-1] >= 3
            ):
                obj_logits = contact_logits[..., 2:3]
        if isinstance(obj_logits, torch.Tensor):
            losses["obj_move"] = F.binary_cross_entropy_with_logits(
                obj_logits, obj_gt.unsqueeze(-1)
            )
        else:
            obj_prob = pred_dict.get("pred_obj_move_prob")
            if isinstance(obj_prob, torch.Tensor):
                losses["obj_move"] = _binary_cross_entropy_probability(
                    obj_prob.clamp(1e-6, 1.0 - 1e-6),
                    obj_gt.unsqueeze(-1),
                )

        hand_logits = pred_dict.get("pred_hand_contact_logits_direct")
        if not isinstance(hand_logits, torch.Tensor):
            contact_logits = pred_dict.get("pred_hand_contact_logits")
            if (
                isinstance(contact_logits, torch.Tensor)
                and contact_logits.shape[-1] >= 2
            ):
                hand_logits = contact_logits[..., :2]
        if isinstance(hand_logits, torch.Tensor):
            element_loss = F.binary_cross_entropy_with_logits(
                hand_logits, hand_gt, reduction="none"
            )
        else:
            hand_prob = pred_dict.get("pred_hand_contact_prob_direct")
            if not isinstance(hand_prob, torch.Tensor):
                combined_prob = pred_dict.get("pred_hand_contact_prob")
                if (
                    isinstance(combined_prob, torch.Tensor)
                    and combined_prob.shape[-1] >= 2
                ):
                    hand_prob = combined_prob[..., :2]
            element_loss = (
                _binary_cross_entropy_probability(
                    hand_prob.clamp(1e-6, 1.0 - 1e-6),
                    hand_gt,
                    reduction="none",
                )
                if isinstance(hand_prob, torch.Tensor)
                else None
            )
        if element_loss is not None:
            pos_weight = float(
                self.weights.get("v2_hand_contact_pos_weight", 1.0)
            )
            neg_weight = float(
                self.weights.get("v2_hand_contact_neg_weight", 1.0)
            )
            hard_neg_weight = float(
                self.weights.get(
                    "v2_hand_contact_hard_neg_weight", neg_weight
                )
            )
            element_weight = torch.where(
                hand_gt > 0.5,
                torch.full_like(hand_gt, pos_weight),
                torch.full_like(hand_gt, neg_weight),
            )
            hard_negative = (
                (obj_gt > 0.5)
                & (left_gt <= 0.5)
                & (right_gt <= 0.5)
            ).unsqueeze(-1)
            element_weight = torch.where(
                hard_negative & (hand_gt <= 0.5),
                torch.full_like(element_weight, hard_neg_weight),
                element_weight,
            )
            losses["hand_contact"] = (
                element_loss * element_weight
            ).sum() / element_weight.sum().clamp_min(1e-6)

        active = (
            set(self.LOSS_KEYS)
            if active_loss_keys is None
            else set(active_loss_keys)
        )
        unknown = active - self.LOSS_KEYS
        if unknown:
            raise ValueError(f"Unknown VC v2 loss keys: {sorted(unknown)}")
        weighted = {}
        total = zero.clone()
        for key, value in losses.items():
            if key not in active:
                continue
            weighted[key] = value * float(self.weights.get(key, 1.0))
            total = total + weighted[key]
        return total, losses, weighted

    def compute_test_loss(self, pred_dict, batch, device):
        _, _, weighted = self.compute_loss(pred_dict, batch, device)
        selected = {
            key: value
            for key, value in weighted.items()
            if key in self.TEST_LOSS_KEYS
        }
        total = (
            sum(selected.values())
            if selected
            else batch["human_imu"].to(device).new_tensor(0.0)
        )
        return total, selected

    @classmethod
    def get_loss_keys(cls):
        return list(cls.LOSS_KEYS)
