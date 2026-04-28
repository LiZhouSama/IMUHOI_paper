"""Loss for the Mamba InteractionModule."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from configs import _REDUCED_POSE_NAMES, _SENSOR_POS_INDICES
from utils.rotation_conversions import matrix_to_rotation_6d


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, zero: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    mask = mask.to(device=pred.device, dtype=torch.bool)
    if mask.shape != pred.shape:
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(pred)
    if not mask.any():
        return zero.clone()
    return F.mse_loss(pred[mask], target[mask])


class InteractionLoss:
    """Supervised interaction loss matching the unified Mamba decoder heads."""

    LOSS_KEYS = (
        "hand_vel",
        "obj_vel",
        "obj_move",
        "hand_contact",
        "foot_contact",
        "obj_trans",
        "lhand_obj_direction",
        "rhand_obj_direction",
        "lhand_lb",
        "rhand_lb",
        "obj_vel_cons",
        "obj_acc_cons",
        "refine_pose",
        "refine_root_trans",
        "refine_hand_pos",
        "refine_fk_joint",
        "align",
        "commit",
        "code_cls",
        "obj_smooth",
        "refine_smooth",
    )
    TEST_LOSS_KEYS = ("obj_trans", "refine_fk_joint", "refine_hand_pos")

    _DEFAULT_WEIGHTS = {
        "hand_vel": 1.0,
        "obj_vel": 1.0,
        "obj_move": 1.0,
        "hand_contact": 1.0,
        "foot_contact": 1.0,
        "obj_trans": 1.0,
        "lhand_obj_direction": 1.0,
        "rhand_obj_direction": 1.0,
        "lhand_lb": 1.0,
        "rhand_lb": 1.0,
        "obj_vel_cons": 0.5,
        "obj_acc_cons": 0.2,
        "refine_pose": 5.0,
        "refine_root_trans": 0.5,
        "refine_hand_pos": 2.0,
        "refine_fk_joint": 1.0,
        "align": 1.0,
        "commit": 0.1,
        "code_cls": 0.5,
        "obj_smooth": 0.3,
        "refine_smooth": 0.3,
    }

    _ALIASES = {
        "hand_contact": ("hand_contact", "hand_contact_cond"),
        "foot_contact": ("foot_contact", "interaction_boundary"),
        "align": ("align", "Loss_align", "L_align"),
        "commit": ("commit", "Loss_commit", "L_commit"),
        "code_cls": ("code_cls", "Loss_code_cls", "L_code_cls"),
    }

    def __init__(self, weights=None):
        self.weights = weights or {}

    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)

    def _weight(self, key: str) -> float:
        for name in self._ALIASES.get(key, (key,)):
            if name in self.weights:
                return float(self.weights[name])
        return float(self._DEFAULT_WEIGHTS[key])

    @staticmethod
    def _expand_bt(value, batch_size: int, seq_len: int, trailing_shape, device, dtype):
        if trailing_shape is None:
            shape = None
        else:
            shape = (batch_size, seq_len) if trailing_shape == () else (batch_size, seq_len, *trailing_shape)
        if not isinstance(value, torch.Tensor):
            return None
        out = value.to(device=device, dtype=dtype)
        if shape is None:
            if out.dim() == 3:
                out = out.unsqueeze(0)
            if out.shape[0] == 1 and batch_size > 1:
                out = out.expand(batch_size, *out.shape[1:])
            if out.shape[0] != batch_size or out.shape[1] != seq_len:
                return None
            return out
        if out.dim() == len(shape) - 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, *out.shape[1:])
        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return None
        if trailing_shape != () and tuple(out.shape[2:]) != tuple(trailing_shape):
            return None
        return out

    @staticmethod
    def _has_object_mask(value, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        if isinstance(value, torch.Tensor):
            mask = value.to(device=device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(value, device=device, dtype=torch.bool)
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
            if mask.shape[1] == 1 and seq_len > 1:
                mask = mask.expand(batch_size, seq_len)
            elif mask.shape[1] != seq_len:
                mask = mask[:, :1].expand(batch_size, seq_len)
        else:
            mask = mask.reshape(batch_size, -1)[:, :1].expand(batch_size, seq_len)
        return mask

    @staticmethod
    def _second_diff(x: torch.Tensor) -> torch.Tensor:
        return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]

    def _target_reduced_pose(self, batch, batch_size: int, seq_len: int, device, dtype):
        reduced = self._expand_bt(batch.get("ori_root_reduced"), batch_size, seq_len, (len(_REDUCED_POSE_NAMES), 3, 3), device, dtype)
        if reduced is None:
            return None
        return matrix_to_rotation_6d(reduced.reshape(-1, 3, 3)).reshape(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6)

    def compute_loss(self, pred_dict, batch, device):
        human_imu = batch["human_imu"].to(device)
        dtype = human_imu.dtype
        batch_size, seq_len = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)
        losses = {key: zero.clone() for key in self.LOSS_KEYS}

        has_object_mask = self._has_object_mask(batch.get("has_object"), batch_size, seq_len, device=device)

        sensor_vel_glb = self._expand_bt(batch.get("sensor_vel_glb"), batch_size, seq_len, (len(_SENSOR_POS_INDICES), 3), device, dtype)
        if sensor_vel_glb is not None and isinstance(pred_dict.get("pred_hand_glb_vel"), torch.Tensor):
            gt_hand_vel = sensor_vel_glb[:, :, -2:]
            losses["hand_vel"] = F.mse_loss(pred_dict["pred_hand_glb_vel"].to(device=device, dtype=dtype), gt_hand_vel)

        obj_vel_gt = self._expand_bt(batch.get("obj_vel"), batch_size, seq_len, (3,), device, dtype)
        if obj_vel_gt is not None and isinstance(pred_dict.get("pred_obj_vel"), torch.Tensor):
            losses["obj_vel"] = _masked_mse(pred_dict["pred_obj_vel"].to(device=device, dtype=dtype), obj_vel_gt, has_object_mask, zero)

        lhand_contact = self._expand_bt(batch.get("lhand_contact"), batch_size, seq_len, (), device, dtype)
        rhand_contact = self._expand_bt(batch.get("rhand_contact"), batch_size, seq_len, (), device, dtype)
        obj_contact = self._expand_bt(batch.get("obj_contact"), batch_size, seq_len, (), device, dtype)
        if obj_contact is None:
            obj_contact = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

        obj_move_logits = pred_dict.get("pred_obj_move_logits")
        if not isinstance(obj_move_logits, torch.Tensor):
            logits_all = pred_dict.get("pred_hand_contact_logits")
            if isinstance(logits_all, torch.Tensor) and logits_all.shape[-1] >= 3:
                obj_move_logits = logits_all[..., 2:3]
        if isinstance(obj_move_logits, torch.Tensor):
            losses["obj_move"] = F.binary_cross_entropy_with_logits(
                obj_move_logits.to(device=device, dtype=dtype),
                obj_contact.unsqueeze(-1).clamp(0.0, 1.0),
            )

        hand_logits = pred_dict.get("pred_hand_contact_logits_cond")
        if not isinstance(hand_logits, torch.Tensor):
            logits_all = pred_dict.get("pred_hand_contact_logits")
            if isinstance(logits_all, torch.Tensor) and logits_all.shape[-1] >= 2:
                hand_logits = logits_all[..., :2]
        if isinstance(hand_logits, torch.Tensor) and lhand_contact is not None and rhand_contact is not None:
            target = torch.stack((lhand_contact, rhand_contact), dim=-1).clamp(0.0, 1.0)
            raw = F.binary_cross_entropy_with_logits(hand_logits.to(device=device, dtype=dtype), target, reduction="none")
            mask = (obj_contact > 0.5) & has_object_mask
            losses["hand_contact"] = raw[mask].mean() if mask.any() else zero.clone()

        foot_logits = pred_dict.get("pred_foot_contact_logits")
        if isinstance(foot_logits, torch.Tensor):
            lfoot = self._expand_bt(batch.get("lfoot_contact"), batch_size, seq_len, (), device, dtype)
            rfoot = self._expand_bt(batch.get("rfoot_contact"), batch_size, seq_len, (), device, dtype)
            if lfoot is not None and rfoot is not None:
                target = torch.stack((lfoot, rfoot), dim=-1).clamp(0.0, 1.0)
                losses["foot_contact"] = F.binary_cross_entropy_with_logits(foot_logits.to(device=device, dtype=dtype), target)
            else:
                start = self._expand_bt(batch.get("interaction_start_gauss", batch.get("interaction_start")), batch_size, seq_len, (), device, dtype)
                end = self._expand_bt(batch.get("interaction_end_gauss", batch.get("interaction_end")), batch_size, seq_len, (), device, dtype)
                if start is not None and end is not None:
                    target = torch.stack((start, end), dim=-1).clamp(0.0, 1.0)
                    losses["foot_contact"] = F.binary_cross_entropy_with_logits(foot_logits.to(device=device, dtype=dtype), target)

        obj_trans_gt = self._expand_bt(batch.get("obj_trans"), batch_size, seq_len, (3,), device, dtype)
        if obj_trans_gt is not None and isinstance(pred_dict.get("pred_obj_trans"), torch.Tensor):
            pred_obj = pred_dict["pred_obj_trans"].to(device=device, dtype=dtype)
            losses["obj_trans"] = _masked_mse(pred_obj, obj_trans_gt, has_object_mask, zero)
            if seq_len > 2:
                obj_acc = self._second_diff(pred_obj)
                obj_acc_mask = has_object_mask[:, 2:] & has_object_mask[:, 1:-1] & has_object_mask[:, :-2]
                losses["obj_smooth"] = (obj_acc[obj_acc_mask] ** 2).mean() if obj_acc_mask.any() else zero.clone()

        if obj_vel_gt is not None and isinstance(pred_dict.get("pred_obj_vel_from_posdiff"), torch.Tensor):
            losses["obj_vel_cons"] = _masked_mse(
                pred_dict["pred_obj_vel_from_posdiff"].to(device=device, dtype=dtype),
                obj_vel_gt,
                has_object_mask,
                zero,
            )

        obj_imu = self._expand_bt(batch.get("obj_imu"), batch_size, seq_len, (9,), device, dtype)
        if obj_imu is not None and isinstance(pred_dict.get("pred_obj_acc_from_posdiff"), torch.Tensor):
            losses["obj_acc_cons"] = _masked_mse(
                pred_dict["pred_obj_acc_from_posdiff"].to(device=device, dtype=dtype),
                obj_imu[..., :3],
                has_object_mask,
                zero,
            )

        l_dir_gt = self._expand_bt(batch.get("lhand_obj_direction"), batch_size, seq_len, (3,), device, dtype)
        r_dir_gt = self._expand_bt(batch.get("rhand_obj_direction"), batch_size, seq_len, (3,), device, dtype)
        l_len_gt = self._expand_bt(batch.get("lhand_lb"), batch_size, seq_len, (), device, dtype)
        r_len_gt = self._expand_bt(batch.get("rhand_lb"), batch_size, seq_len, (), device, dtype)
        mask_l = (lhand_contact > 0.5) & has_object_mask if lhand_contact is not None else has_object_mask
        mask_r = (rhand_contact > 0.5) & has_object_mask if rhand_contact is not None else has_object_mask
        if l_dir_gt is not None and isinstance(pred_dict.get("pred_lhand_obj_direction"), torch.Tensor):
            losses["lhand_obj_direction"] = _masked_mse(pred_dict["pred_lhand_obj_direction"].to(device=device, dtype=dtype), l_dir_gt, mask_l, zero)
        if r_dir_gt is not None and isinstance(pred_dict.get("pred_rhand_obj_direction"), torch.Tensor):
            losses["rhand_obj_direction"] = _masked_mse(pred_dict["pred_rhand_obj_direction"].to(device=device, dtype=dtype), r_dir_gt, mask_r, zero)
        if l_len_gt is not None and isinstance(pred_dict.get("pred_lhand_lb"), torch.Tensor):
            losses["lhand_lb"] = _masked_mse(pred_dict["pred_lhand_lb"].to(device=device, dtype=dtype), l_len_gt, mask_l, zero)
        if r_len_gt is not None and isinstance(pred_dict.get("pred_rhand_lb"), torch.Tensor):
            losses["rhand_lb"] = _masked_mse(pred_dict["pred_rhand_lb"].to(device=device, dtype=dtype), r_len_gt, mask_r, zero)

        pose_gt = self._target_reduced_pose(batch, batch_size, seq_len, device, dtype)
        if pose_gt is not None and isinstance(pred_dict.get("refined_pose"), torch.Tensor):
            refined_pose = pred_dict["refined_pose"].to(device=device, dtype=dtype)
            losses["refine_pose"] = F.mse_loss(refined_pose, pose_gt)
            if seq_len > 2:
                losses["refine_smooth"] = (self._second_diff(refined_pose) ** 2).mean()

        trans_gt = self._expand_bt(batch.get("trans"), batch_size, seq_len, (3,), device, dtype)
        if trans_gt is not None and isinstance(pred_dict.get("refined_trans"), torch.Tensor):
            losses["refine_root_trans"] = F.mse_loss(pred_dict["refined_trans"].to(device=device, dtype=dtype), trans_gt)

        position_global = self._expand_bt(batch.get("position_global"), batch_size, seq_len, None, device, dtype)
        if position_global is None and isinstance(batch.get("position_global"), torch.Tensor):
            position_global = batch["position_global"].to(device=device, dtype=dtype)
            if position_global.dim() == 3:
                position_global = position_global.unsqueeze(0)
            if position_global.shape[0] == 1 and batch_size > 1:
                position_global = position_global.expand(batch_size, -1, -1, -1)
        if isinstance(position_global, torch.Tensor) and position_global.shape[:2] == (batch_size, seq_len):
            if isinstance(pred_dict.get("refined_hand_glb_pos"), torch.Tensor) and position_global.shape[2] > 21:
                hand_gt = torch.stack((position_global[:, :, 20], position_global[:, :, 21]), dim=2)
                losses["refine_hand_pos"] = F.mse_loss(pred_dict["refined_hand_glb_pos"].to(device=device, dtype=dtype), hand_gt)
            if isinstance(pred_dict.get("refined_joints_global"), torch.Tensor):
                pred_joints = pred_dict["refined_joints_global"].to(device=device, dtype=dtype)
                nj = min(pred_joints.shape[2], position_global.shape[2])
                if nj > 0:
                    losses["refine_fk_joint"] = F.mse_loss(pred_joints[:, :, :nj], position_global[:, :, :nj])

        prior_aux = pred_dict.get("object_prior_aux") if isinstance(pred_dict, dict) else None
        if isinstance(prior_aux, dict):
            mesh_valid = prior_aux.get("mesh_valid_mask")
            if isinstance(mesh_valid, torch.Tensor):
                valid = mesh_valid.to(device=device, dtype=torch.bool) & has_object_mask.any(dim=1)
            else:
                valid = torch.zeros(batch_size, device=device, dtype=torch.bool)

            z_e_obs = prior_aux.get("z_e_obs")
            z_q_mesh = prior_aux.get("z_q_mesh")
            if isinstance(z_e_obs, torch.Tensor) and isinstance(z_q_mesh, torch.Tensor) and z_e_obs.shape == z_q_mesh.shape and valid.any():
                losses["align"] = F.mse_loss(z_e_obs.to(device=device, dtype=dtype)[valid], z_q_mesh.to(device=device, dtype=dtype).detach()[valid])

            code_logits_obs = prior_aux.get("code_logits_obs")
            code_idx_mesh = prior_aux.get("code_idx_mesh")
            if isinstance(code_logits_obs, torch.Tensor) and isinstance(code_idx_mesh, torch.Tensor):
                cls_valid = valid & (code_idx_mesh.to(device=device) >= 0)
                if cls_valid.any():
                    losses["code_cls"] = F.cross_entropy(
                        code_logits_obs.to(device=device, dtype=dtype)[cls_valid],
                        code_idx_mesh.to(device=device, dtype=torch.long)[cls_valid],
                    )

            beta_raw = prior_aux.get("vq_beta", 0.25)
            beta = float(beta_raw.detach().mean().item()) if isinstance(beta_raw, torch.Tensor) else float(beta_raw)
            commit_terms = []
            for e_key, q_key in (("z_e_mesh", "z_q_mesh"), ("z_e_obs", "z_q_obs")):
                z_e = prior_aux.get(e_key)
                z_q = prior_aux.get(q_key)
                if isinstance(z_e, torch.Tensor) and isinstance(z_q, torch.Tensor) and z_e.shape == z_q.shape:
                    use_mask = valid if e_key == "z_e_mesh" else has_object_mask.any(dim=1)
                    if use_mask.any():
                        z_e = z_e.to(device=device, dtype=dtype)
                        z_q = z_q.to(device=device, dtype=dtype)
                        commit_terms.append(
                            F.mse_loss(z_e.detach()[use_mask], z_q[use_mask])
                            + max(beta, 0.0) * F.mse_loss(z_e[use_mask], z_q.detach()[use_mask])
                        )
            if commit_terms:
                losses["commit"] = sum(commit_terms) / len(commit_terms)

        total_loss = zero.clone()
        weighted_losses = {}
        for key in self.LOSS_KEYS:
            weighted = losses[key] * self._weight(key)
            weighted_losses[key] = weighted
            total_loss = total_loss + weighted
        return total_loss, losses, weighted_losses

    def compute_test_loss(self, pred_dict, batch, device):
        _, losses, _ = self.compute_loss(pred_dict, batch, device)
        test_losses = {key: losses[key] for key in self.TEST_LOSS_KEYS}
        return sum(test_losses.values()), test_losses

    @classmethod
    def get_loss_keys(cls):
        return list(cls.LOSS_KEYS)
