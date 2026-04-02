"""
Unified loss for IMUHOI Mix DiT model.

Train loss terms:
- simple
- vel
- fk
- drift
- slide
- vel_obj
- jitter_obj

Test metrics:
- mpjre_mse
- root_trans_err_mm
- jitter_local_fk_mm
- root_jitter_mm
- obj_pos_mse
- obj_pos_jitter_mm
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d


class IMUHOIMixLoss:
    """Combined HumanPose + Interaction loss with single simple term."""

    LOSS_KEYS = ("simple", "vel", "fk", "drift", "slide", "vel_obj", "jitter_obj")
    TEST_LOSS_KEYS = (
        "mpjre_mse",
        "root_trans_err_mm",
        "jitter_local_fk_mm",
        "root_jitter_mm",
        "obj_pos_mse",
        "obj_pos_jitter_mm",
    )

    _WEIGHT_ALIASES = {
        "simple": ("simple", "L_simple", "diffusion_x0", "simple_pose", "Loss_simple"),
        "vel": ("vel", "L_vel", "vel_smooth"),
        "fk": ("fk", "L_FK", "fk_joint"),
        "drift": ("drift", "L_drift"),
        "slide": ("slide", "L_slide", "foot_slide"),
        "vel_obj": ("vel_obj", "Loss_vel_obj", "L_vel_obj", "drift_obj", "Loss_drift_obj", "L_drift_obj"),
        "jitter_obj": ("jitter_obj", "Loss_jitter_obj", "L_jitter_obj"),
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
    def _prepare_has_object_mask(has_object, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if has_object is None:
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

        if isinstance(has_object, torch.Tensor):
            mask = has_object.to(device=device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(has_object, device=device, dtype=torch.bool)

        if mask.dim() == 0:
            mask = mask.view(1)
        if mask.dim() == 1:
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size)
            if mask.shape[0] != batch_size:
                mask = mask[:1].expand(batch_size)
            mask = mask.view(batch_size, 1).expand(batch_size, seq_len)
        elif mask.dim() == 2:
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size, mask.shape[1])
            if mask.shape[0] != batch_size:
                mask = mask[:1].expand(batch_size, mask.shape[1])
            if mask.shape[1] == 1 and seq_len > 1:
                mask = mask.expand(batch_size, seq_len)
            elif mask.shape[1] != seq_len:
                mask = mask[:, :1].expand(batch_size, seq_len)
        else:
            mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

        return mask

    @staticmethod
    def _integrate_obj_trans(delta_p_obj_pred: torch.Tensor, obj_trans_init: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(delta_p_obj_pred, dim=1) + obj_trans_init.unsqueeze(1)

    def compute_loss(self, pred_dict, batch, device):
        human_imu = batch["human_imu"].to(device)
        dtype = human_imu.dtype
        batch_size, seq_len = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)

        losses = {k: zero.clone() for k in self.LOSS_KEYS}

        diffusion_aux = pred_dict.get("diffusion_aux", {}) if isinstance(pred_dict, dict) else {}
        x_pred = pred_dict.get("x_pred") if isinstance(pred_dict, dict) else None
        x_target = diffusion_aux.get("x0_target") if isinstance(diffusion_aux, dict) else None

        # Single simple term for full mix feature vector.
        if isinstance(x_pred, torch.Tensor) and isinstance(x_target, torch.Tensor):
            x_pred = x_pred.to(device=device, dtype=dtype)
            x_target = x_target.to(device=device, dtype=dtype)
            if x_pred.shape == x_target.shape:
                losses["simple"] = F.mse_loss(x_pred, x_target)

        # Human pose velocity smooth term.
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

        # Human FK term.
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

        # Human root drift / slide terms.
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
                foot = pred_joints_local[:, :, [7, 8], :]
                foot_delta = foot[:, 1:] - foot[:, :-1]

                delta2 = delta_p_pred[:, :-1, :2]
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
                        delta_p_y = p_y_pred[:, 1:, 0] - p_y_pred[:, :-1, 0]
                        delta3[:, :, :, 1] = delta_p_y.unsqueeze(-1)

                contact_w = b_prob_pred[:, :-1, :2].clamp(0.0, 1.0).unsqueeze(-1)
                slide_vec = contact_w * (foot_delta + delta3)
                losses["slide"] = (slide_vec ** 2).mean()

        # Object terms (masked by has_object).
        has_object_mask = self._prepare_has_object_mask(batch.get("has_object"), batch_size, seq_len, device=device)
        pred_obj_trans = pred_dict.get("pred_obj_trans") if isinstance(pred_dict, dict) else None
        p_obj_pred = pred_dict.get("p_obj_pred") if isinstance(pred_dict, dict) else None
        delta_p_obj_pred = pred_dict.get("delta_p_obj_pred") if isinstance(pred_dict, dict) else None
        pred_obj_vel = pred_dict.get("pred_obj_vel") if isinstance(pred_dict, dict) else None

        if not isinstance(pred_obj_trans, torch.Tensor) and isinstance(p_obj_pred, torch.Tensor):
            pred_obj_trans = p_obj_pred

        if not isinstance(pred_obj_trans, torch.Tensor) and isinstance(delta_p_obj_pred, torch.Tensor):
            delta_p_obj_pred = delta_p_obj_pred.to(device=device, dtype=dtype)
            if delta_p_obj_pred.dim() == 2:
                delta_p_obj_pred = delta_p_obj_pred.unsqueeze(0)

            obj_trans_init = pred_dict.get("obj_trans_init") if isinstance(pred_dict, dict) else None
            if isinstance(obj_trans_init, torch.Tensor):
                obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
                if obj_trans_init.dim() == 1:
                    obj_trans_init = obj_trans_init.unsqueeze(0)
                if obj_trans_init.shape[0] == 1 and batch_size > 1:
                    obj_trans_init = obj_trans_init.expand(batch_size, -1)
            else:
                obj_trans_gt_for_init = batch.get("obj_trans")
                if isinstance(obj_trans_gt_for_init, torch.Tensor):
                    obj_trans_gt_for_init = obj_trans_gt_for_init.to(device=device, dtype=dtype)
                    if obj_trans_gt_for_init.dim() == 2:
                        obj_trans_gt_for_init = obj_trans_gt_for_init.unsqueeze(0)
                    if obj_trans_gt_for_init.shape[0] == 1 and batch_size > 1:
                        obj_trans_gt_for_init = obj_trans_gt_for_init.expand(batch_size, -1, -1)
                    if obj_trans_gt_for_init.shape[0] == batch_size and obj_trans_gt_for_init.shape[1] > 0:
                        obj_trans_init = obj_trans_gt_for_init[:, 0]

            if isinstance(obj_trans_init, torch.Tensor) and obj_trans_init.shape[0] == batch_size:
                pred_obj_trans = self._integrate_obj_trans(delta_p_obj_pred, obj_trans_init)

        if isinstance(pred_obj_trans, torch.Tensor):
            pred_obj_trans = pred_obj_trans.to(device=device, dtype=dtype)
            if pred_obj_trans.dim() == 2:
                pred_obj_trans = pred_obj_trans.unsqueeze(0)

        if isinstance(pred_obj_vel, torch.Tensor):
            pred_obj_vel = pred_obj_vel.to(device=device, dtype=dtype)
            if pred_obj_vel.dim() == 2:
                pred_obj_vel = pred_obj_vel.unsqueeze(0)
        elif isinstance(pred_obj_trans, torch.Tensor):
            pred_obj_vel = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
            if pred_obj_trans.shape[:2] == (batch_size, seq_len) and seq_len > 1:
                pred_obj_vel[:, 1:] = pred_obj_trans[:, 1:] - pred_obj_trans[:, :-1]
                pred_obj_vel[:, 0] = pred_obj_vel[:, 1]

        obj_trans_gt = batch.get("obj_trans")
        if isinstance(obj_trans_gt, torch.Tensor):
            obj_trans_gt = obj_trans_gt.to(device=device, dtype=dtype)
            if obj_trans_gt.dim() == 2:
                obj_trans_gt = obj_trans_gt.unsqueeze(0)
            if obj_trans_gt.shape[0] == 1 and batch_size > 1:
                obj_trans_gt = obj_trans_gt.expand(batch_size, -1, -1)

        obj_vel_gt = batch.get("obj_vel")
        if isinstance(obj_vel_gt, torch.Tensor):
            obj_vel_gt = obj_vel_gt.to(device=device, dtype=dtype)
            if obj_vel_gt.dim() == 2:
                obj_vel_gt = obj_vel_gt.unsqueeze(0)
            if obj_vel_gt.shape[0] == 1 and batch_size > 1:
                obj_vel_gt = obj_vel_gt.expand(batch_size, -1, -1)
        elif isinstance(obj_trans_gt, torch.Tensor):
            obj_vel_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
            if obj_trans_gt.shape[:2] == (batch_size, seq_len) and seq_len > 1:
                obj_vel_gt[:, 1:] = obj_trans_gt[:, 1:] - obj_trans_gt[:, :-1]
                obj_vel_gt[:, 0] = obj_vel_gt[:, 1]

        if (
            isinstance(pred_obj_vel, torch.Tensor)
            and isinstance(obj_vel_gt, torch.Tensor)
            and pred_obj_vel.shape[:2] == (batch_size, seq_len)
            and obj_vel_gt.shape[:2] == (batch_size, seq_len)
        ):
            vel_mask = has_object_mask.unsqueeze(-1).expand(-1, -1, 3)
            if vel_mask.any():
                losses["vel_obj"] = F.mse_loss(pred_obj_vel[vel_mask], obj_vel_gt[vel_mask])

        if isinstance(pred_obj_trans, torch.Tensor):
            if pred_obj_trans.shape[:2] == (batch_size, seq_len) and seq_len > 2:
                acc = pred_obj_trans[:, 2:] - 2.0 * pred_obj_trans[:, 1:-1] + pred_obj_trans[:, :-2]
                center_mask = has_object_mask[:, 2:]
                if center_mask.any():
                    losses["jitter_obj"] = (acc[center_mask] ** 2).mean()

        weighted_losses = {}
        total_loss = zero.clone()

        defaults = {
            "simple": 1.0,
            "vel": 1.0,
            "fk": 1.0,
            "drift": 0.1,
            "slide": 1.0,
            "vel_obj": 1.0,
            "jitter_obj": 1.0,
        }

        for key in self.LOSS_KEYS:
            w = self._weight(key, default=defaults[key])
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

        # Human test metrics.
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

        # Object test metrics.
        has_object_mask = self._prepare_has_object_mask(batch.get("has_object"), batch_size, seq_len, device=device)

        pred_obj_trans = pred_dict.get("pred_obj_trans") if isinstance(pred_dict, dict) else None
        if isinstance(pred_obj_trans, torch.Tensor):
            pred_obj_trans = pred_obj_trans.to(device=device, dtype=dtype)
            if pred_obj_trans.dim() == 2:
                pred_obj_trans = pred_obj_trans.unsqueeze(0)

        if not isinstance(pred_obj_trans, torch.Tensor):
            p_obj_pred = pred_dict.get("p_obj_pred") if isinstance(pred_dict, dict) else None
            if isinstance(p_obj_pred, torch.Tensor):
                pred_obj_trans = p_obj_pred.to(device=device, dtype=dtype)
                if pred_obj_trans.dim() == 2:
                    pred_obj_trans = pred_obj_trans.unsqueeze(0)

        if not isinstance(pred_obj_trans, torch.Tensor):
            delta_p_obj_pred = pred_dict.get("delta_p_obj_pred") if isinstance(pred_dict, dict) else None
            obj_trans_init = pred_dict.get("obj_trans_init") if isinstance(pred_dict, dict) else None
            if isinstance(delta_p_obj_pred, torch.Tensor):
                delta_p_obj_pred = delta_p_obj_pred.to(device=device, dtype=dtype)
                if delta_p_obj_pred.dim() == 2:
                    delta_p_obj_pred = delta_p_obj_pred.unsqueeze(0)

                if isinstance(obj_trans_init, torch.Tensor):
                    obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
                    if obj_trans_init.dim() == 1:
                        obj_trans_init = obj_trans_init.unsqueeze(0)
                    if obj_trans_init.shape[0] == 1 and batch_size > 1:
                        obj_trans_init = obj_trans_init.expand(batch_size, -1)
                else:
                    obj_trans_init = None

                if not isinstance(obj_trans_init, torch.Tensor):
                    obj_trans_gt = batch.get("obj_trans")
                    if isinstance(obj_trans_gt, torch.Tensor):
                        obj_trans_gt = obj_trans_gt.to(device=device, dtype=dtype)
                        if obj_trans_gt.dim() == 2:
                            obj_trans_gt = obj_trans_gt.unsqueeze(0)
                        if obj_trans_gt.shape[0] == 1 and batch_size > 1:
                            obj_trans_gt = obj_trans_gt.expand(batch_size, -1, -1)
                        if obj_trans_gt.shape[0] == batch_size and obj_trans_gt.shape[1] > 0:
                            obj_trans_init = obj_trans_gt[:, 0]

                if isinstance(obj_trans_init, torch.Tensor) and obj_trans_init.shape[0] == batch_size:
                    pred_obj_trans = self._integrate_obj_trans(delta_p_obj_pred, obj_trans_init)

        obj_trans_gt = batch.get("obj_trans")
        if isinstance(pred_obj_trans, torch.Tensor) and isinstance(obj_trans_gt, torch.Tensor):
            obj_trans_gt = obj_trans_gt.to(device=device, dtype=dtype)
            if obj_trans_gt.dim() == 2:
                obj_trans_gt = obj_trans_gt.unsqueeze(0)
            if obj_trans_gt.shape[0] == 1 and batch_size > 1:
                obj_trans_gt = obj_trans_gt.expand(batch_size, -1, -1)

            if pred_obj_trans.shape[:2] == (batch_size, seq_len) and obj_trans_gt.shape[:2] == (batch_size, seq_len):
                valid = has_object_mask.unsqueeze(-1).expand(-1, -1, 3)
                if valid.any():
                    sq_err = (pred_obj_trans - obj_trans_gt) ** 2
                    metrics["obj_pos_mse"] = sq_err[valid].mean() 

                if seq_len > 2:
                    acc = pred_obj_trans[:, 2:] - 2.0 * pred_obj_trans[:, 1:-1] + pred_obj_trans[:, :-2]
                    triplet_mask = has_object_mask[:, 2:] & has_object_mask[:, 1:-1] & has_object_mask[:, :-2]
                    if triplet_mask.any():
                        metrics["obj_pos_jitter_mm"] = torch.linalg.norm(acc[triplet_mask], dim=-1).mean() * 1000.0

        default_test_weights = {
            "mpjre_mse": 1.0,
            "root_trans_err_mm": 1.0,
            "jitter_local_fk_mm": 1.0,
            "root_jitter_mm": 1.0,
            "obj_pos_mse": 1.0,
            "obj_pos_jitter_mm": 1.0,
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
