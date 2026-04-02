"""
Interaction loss for full-window diffusion training.

Loss terms:
- Loss_simple
- Loss_vel_obj
- Loss_jitter_obj
- Loss_align
- Loss_code_cls
- Loss_commit
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class InteractionLoss:
    """Stage-2 loss composition for full-window denoising."""

    LOSS_KEYS = ("simple", "vel_obj", "jitter_obj", "align", "code_cls", "commit")
    TEST_LOSS_KEYS = ("obj_pos_rmse_mm", "obj_pos_jitter_mm")

    _WEIGHT_ALIASES = {
        "simple": ("simple", "Loss_simple", "L_simple"),
        "vel_obj": ("vel_obj", "Loss_vel_obj", "L_vel_obj", "drift_obj", "Loss_drift_obj", "L_drift_obj"),
        "jitter_obj": ("jitter_obj", "Loss_jitter_obj", "L_jitter_obj"),
        "align": ("align", "Loss_align", "L_align"),
        "code_cls": ("code_cls", "Loss_code_cls", "L_code_cls"),
        "commit": ("commit", "Loss_commit", "L_commit"),
    }

    def __init__(self, weights=None, test_metric_weights=None):
        self.weights = weights or {}
        self.test_metric_weights = test_metric_weights or {}

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

        losses = {key: zero.clone() for key in self.LOSS_KEYS}

        diffusion_aux = pred_dict.get("diffusion_aux", {}) if isinstance(pred_dict, dict) else {}
        x_pred = pred_dict.get("x_pred") if isinstance(pred_dict, dict) else None
        x_target = diffusion_aux.get("x0_target") if isinstance(diffusion_aux, dict) else None

        if isinstance(x_pred, torch.Tensor) and isinstance(x_target, torch.Tensor):
            x_pred = x_pred.to(device=device, dtype=dtype)
            x_target = x_target.to(device=device, dtype=dtype)
            if x_pred.shape == x_target.shape:
                losses["simple"] = F.mse_loss(x_pred, x_target)

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

        # Object prior/codebook losses.
        object_prior_aux = pred_dict.get("object_prior_aux") if isinstance(pred_dict, dict) else None
        if isinstance(object_prior_aux, dict):
            z_e_obs = object_prior_aux.get("z_e_obs")
            z_e_mesh = object_prior_aux.get("z_e_mesh")
            z_q_mesh = object_prior_aux.get("z_q_mesh")
            code_idx_mesh = object_prior_aux.get("code_idx_mesh")
            code_logits_obs = object_prior_aux.get("code_logits_obs")
            mesh_valid_mask = object_prior_aux.get("mesh_valid_mask")
            vq_beta = object_prior_aux.get("vq_beta")

            sample_has_object = has_object_mask.any(dim=1)
            if isinstance(mesh_valid_mask, torch.Tensor):
                valid_mask = mesh_valid_mask.to(device=device, dtype=torch.bool)
            else:
                valid_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
            valid_mask = valid_mask & sample_has_object

            if (
                isinstance(z_e_obs, torch.Tensor)
                and isinstance(z_q_mesh, torch.Tensor)
                and z_e_obs.shape == z_q_mesh.shape
                and z_e_obs.dim() == 2
                and z_e_obs.shape[0] == batch_size
                and valid_mask.any()
            ):
                z_e_obs = z_e_obs.to(device=device, dtype=dtype)
                z_q_mesh = z_q_mesh.to(device=device, dtype=dtype)
                losses["align"] = F.mse_loss(z_e_obs[valid_mask], z_q_mesh.detach()[valid_mask])

            if (
                isinstance(code_logits_obs, torch.Tensor)
                and isinstance(code_idx_mesh, torch.Tensor)
                and code_logits_obs.dim() == 2
                and code_idx_mesh.dim() == 1
                and code_logits_obs.shape[0] == batch_size
                and code_idx_mesh.shape[0] == batch_size
            ):
                code_logits_obs = code_logits_obs.to(device=device, dtype=dtype)
                code_idx_mesh = code_idx_mesh.to(device=device, dtype=torch.long)
                cls_valid = valid_mask & (code_idx_mesh >= 0)
                if cls_valid.any():
                    losses["code_cls"] = F.cross_entropy(code_logits_obs[cls_valid], code_idx_mesh[cls_valid])

            if (
                isinstance(z_e_mesh, torch.Tensor)
                and isinstance(z_q_mesh, torch.Tensor)
                and z_e_mesh.shape == z_q_mesh.shape
                and z_e_mesh.dim() == 2
                and z_e_mesh.shape[0] == batch_size
                and valid_mask.any()
            ):
                z_e_mesh = z_e_mesh.to(device=device, dtype=dtype)
                z_q_mesh = z_q_mesh.to(device=device, dtype=dtype)
                if isinstance(vq_beta, torch.Tensor):
                    beta = float(vq_beta.detach().mean().item())
                elif isinstance(vq_beta, (float, int)):
                    beta = float(vq_beta)
                else:
                    beta = 0.25
                beta = max(beta, 0.0)
                losses["commit"] = (
                    F.mse_loss(z_e_mesh.detach()[valid_mask], z_q_mesh[valid_mask])
                    + beta * F.mse_loss(z_e_mesh[valid_mask], z_q_mesh.detach()[valid_mask])
                )

        weighted_losses = {}
        total_loss = zero.clone()

        defaults = {
            "simple": 1.0,
            "vel_obj": 1.0,
            "jitter_obj": 1.0,
            "align": 1.0,
            "code_cls": 1.0,
            "commit": 0.1,
        }

        for key in self.LOSS_KEYS:
            w = self._weight(key, default=defaults[key])
            weighted = losses[key] * w
            weighted_losses[key] = weighted
            total_loss = total_loss + weighted

        return total_loss, losses, weighted_losses

    def compute_test_loss(self, pred_dict, batch, device):
        human_imu = batch["human_imu"].to(device)
        dtype = human_imu.dtype
        batch_size, seq_len = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)

        metrics = {key: zero.clone() for key in self.TEST_LOSS_KEYS}

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
        if (
            isinstance(pred_obj_trans, torch.Tensor)
            and isinstance(obj_trans_gt, torch.Tensor)
        ):
            obj_trans_gt = obj_trans_gt.to(device=device, dtype=dtype)
            if obj_trans_gt.dim() == 2:
                obj_trans_gt = obj_trans_gt.unsqueeze(0)
            if obj_trans_gt.shape[0] == 1 and batch_size > 1:
                obj_trans_gt = obj_trans_gt.expand(batch_size, -1, -1)

            if pred_obj_trans.shape[:2] == (batch_size, seq_len) and obj_trans_gt.shape[:2] == (batch_size, seq_len):
                valid = has_object_mask.unsqueeze(-1).expand(-1, -1, 3)
                if valid.any():
                    sq_err = (pred_obj_trans - obj_trans_gt) ** 2
                    mse_mm2 = sq_err[valid].mean() * (1000.0 ** 2)
                    metrics["obj_pos_rmse_mm"] = torch.sqrt(torch.clamp(mse_mm2, min=0.0))

                if seq_len > 2:
                    acc = pred_obj_trans[:, 2:] - 2.0 * pred_obj_trans[:, 1:-1] + pred_obj_trans[:, :-2]
                    triplet_mask = has_object_mask[:, 2:] & has_object_mask[:, 1:-1] & has_object_mask[:, :-2]
                    if triplet_mask.any():
                        metrics["obj_pos_jitter_mm"] = torch.linalg.norm(acc[triplet_mask], dim=-1).mean() * 1000.0

        default_test_weights = {
            "obj_pos_rmse_mm": 1.0,
            "obj_pos_jitter_mm": 1.0,
        }

        test_total = zero.clone()
        for key in self.TEST_LOSS_KEYS:
            # Backward-compat: accept legacy key `obj_pos_mse` as RMSE weight.
            if key == "obj_pos_rmse_mm" and key not in self.test_metric_weights:
                weight = float(self.test_metric_weights.get("obj_pos_mse", default_test_weights[key]))
            else:
                weight = float(self.test_metric_weights.get(key, default_test_weights[key]))
            test_total = test_total + metrics[key] * weight

        return test_total, metrics

    @classmethod
    def get_loss_keys(cls):
        return list(cls.LOSS_KEYS)
