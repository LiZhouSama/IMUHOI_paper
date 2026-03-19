"""
Unified interaction diffusion module (merged VC + OT).

Main outputs:
- pred_obj_trans
- pred_hand_contact_logits/prob (left, right, object_move)
Auxiliary output:
- pred_interaction_boundary_logits/prob
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix

from .base import ConditionalDiT
from configs import FRAME_RATE, _SENSOR_NAMES


class InteractionModule(nn.Module):
    """Merged interaction head for object translation + contact prediction."""

    def __init__(self, cfg):
        super().__init__()
        self.num_joints = int(getattr(cfg, "num_joints", 24))
        self.num_human_imus = int(getattr(cfg, "num_human_imus", len(_SENSOR_NAMES)))
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(getattr(cfg, "obj_imu_dim", self.imu_dim))
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))

        self.obj_trans_dim = 3
        self.contact_dim = 3
        self.boundary_dim = 2

        self.human_imu_dim = self.num_human_imus * self.imu_dim
        self.pose_dim = self.num_joints * 6
        self.cond_dim = (
            self.human_imu_dim
            + self.obj_imu_dim
            + 6  # hand positions
            + 3  # root velocity
            + self.pose_dim
            + 3  # obj_trans_init
            + 3  # contact_init
        )

        dit_cfg = getattr(cfg, "dit", {})
        interaction_cfg = getattr(cfg, "interaction", {})

        def _dit_param(name, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        def _interaction_param(name, default):
            if isinstance(interaction_cfg, dict) and name in interaction_cfg:
                return interaction_cfg[name]
            return getattr(cfg, name, default)

        max_seq_len = _dit_param("dit_max_seq_len", getattr(getattr(cfg, "train", {}), "window", 256))
        self.use_diffusion_noise = bool(_dit_param("dit_use_noise", True))
        self.dit_formulation = str(_dit_param("formulation", "standard")).lower()
        if self.dit_formulation not in {"standard", "residual"}:
            raise ValueError(f"Unsupported dit formulation: {self.dit_formulation}")
        self.head_mode = str(_interaction_param("head_mode", "unified")).lower()
        if self.head_mode not in {"unified", "split"}:
            raise ValueError(f"Unsupported interaction head mode: {self.head_mode}")

        if self.head_mode == "split":
            self.target_dim = self.obj_trans_dim
        else:
            self.target_dim = self.obj_trans_dim + self.contact_dim + self.boundary_dim

        self.inference_steps = _dit_param("dit_inference_steps", None)
        self.inference_steps = int(self.inference_steps) if self.inference_steps is not None else None
        self.inference_sampler = str(_dit_param("dit_inference_sampler", "ddim")).lower()
        self.inference_eta = float(_dit_param("dit_inference_eta", 0.0))

        self.dit = ConditionalDiT(
            target_dim=self.target_dim,
            cond_dim=self.cond_dim,
            d_model=_dit_param("dit_d_model", 256),
            nhead=_dit_param("dit_nhead", 8),
            num_layers=_dit_param("dit_num_layers", 6),
            dim_feedforward=_dit_param("dit_dim_feedforward", 1024),
            dropout=_dit_param("dit_dropout", 0.1),
            max_seq_len=max_seq_len,
            timesteps=_dit_param("dit_timesteps", 1000),
            use_time_embed=_dit_param("dit_use_time_embed", True),
        )

        if self.head_mode == "split":
            cls_hidden = int(_interaction_param("cls_hidden_dim", _dit_param("dit_d_model", 256)))
            self.cls_in_proj = nn.Linear(self.cond_dim + self.obj_trans_dim, cls_hidden)
            self.cls_temporal = nn.Sequential(
                nn.Conv1d(cls_hidden, cls_hidden, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(cls_hidden, cls_hidden, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.contact_head = nn.Conv1d(cls_hidden, self.contact_dim, kernel_size=1)
            self.boundary_head = nn.Conv1d(cls_hidden, self.boundary_dim, kernel_size=1)

    def _denormalize_imu(self, human_imu: torch.Tensor) -> torch.Tensor:
        """Convert relative IMU to world-frame aligned representation."""
        batch_size, seq_len = human_imu.shape[:2]

        human_imu_acc = human_imu[:, :, :, :3]
        human_imu_ori = human_imu[:, :, :, 3:9]
        human_imu_ori_6d = human_imu_ori.reshape(-1, 6)
        human_imu_ori_mat = rotation_6d_to_matrix(human_imu_ori_6d).reshape(batch_size, seq_len, self.num_human_imus, 3, 3)

        r0_t = human_imu_ori_mat[:, :, 0].transpose(-1, -2)
        acc_world = torch.matmul(human_imu_acc, r0_t)
        acc0_world = acc_world[:, :, :1, :]
        acc_rest_mix = acc_world[:, :, 1:, :] + acc0_world
        human_imu_acc_denorm = torch.cat([acc0_world, acc_rest_mix], dim=2)

        human_imu_ori_denorm = torch.cat(
            [
                human_imu_ori_mat[:, :, :1],
                human_imu_ori_mat[:, :, :1].matmul(human_imu_ori_mat[:, :, 1:]),
            ],
            dim=2,
        )

        from pytorch3d.transforms import matrix_to_rotation_6d

        human_imu_ori_denorm_6d = matrix_to_rotation_6d(human_imu_ori_denorm)
        return torch.cat([human_imu_acc_denorm, human_imu_ori_denorm_6d], dim=-1)

    @staticmethod
    def _ensure_bt_tensor(value, batch_size: int, seq_len: int, last_dim: int, device: torch.device, dtype: torch.dtype):
        if not isinstance(value, torch.Tensor):
            return torch.zeros(batch_size, seq_len, last_dim, device=device, dtype=dtype)
        tensor = value.to(device=device, dtype=dtype)
        if tensor.dim() == 2 and last_dim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.dim() == 2 and tensor.shape[-1] == last_dim:
            tensor = tensor.unsqueeze(1).expand(batch_size, seq_len, last_dim)
        if tensor.dim() == 1 and tensor.shape[0] == last_dim:
            tensor = tensor.view(1, 1, last_dim).expand(batch_size, seq_len, last_dim)
        if tensor.dim() == 3 and tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1, -1)
        if tensor.shape[0] != batch_size or tensor.shape[1] != seq_len or tensor.shape[2] != last_dim:
            return torch.zeros(batch_size, seq_len, last_dim, device=device, dtype=dtype)
        return tensor

    @staticmethod
    def _prepare_has_object_mask(has_object_mask, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype):
        if has_object_mask is None:
            return None
        mask = torch.as_tensor(has_object_mask, device=device)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.dim() == 0:
            mask = mask.view(1)
        if mask.dim() == 1:
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size)
            mask = mask.view(batch_size, 1).expand(batch_size, seq_len)
        elif mask.dim() == 2:
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size, seq_len)
        else:
            raise ValueError(f"has_object_mask must be [B] or [B,T], got {mask.shape}")
        return mask.to(device=device, dtype=dtype)

    def _prepare_inputs(self, data_dict: dict, hp_out: dict):
        human_imu = data_dict["human_imu"]
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len, _, _ = human_imu.shape
        device = human_imu.device
        dtype = human_imu.dtype

        obj_imu = data_dict.get("obj_imu")
        if obj_imu is None:
            obj_imu = torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        else:
            obj_imu = obj_imu.to(device=device, dtype=dtype)
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(batch_size, seq_len, -1)
            if obj_imu.dim() != 3 or obj_imu.shape[-1] != self.obj_imu_dim:
                raise ValueError(f"obj_imu must be [B,T,{self.obj_imu_dim}], got {obj_imu.shape}")

        if hp_out is None:
            raise ValueError("hp_out is required for InteractionModule")

        hand_pos = hp_out.get("pred_hand_glb_pos")
        if isinstance(hand_pos, torch.Tensor):
            hand_pos = hand_pos.to(device=device, dtype=dtype)
            if hand_pos.dim() == 3:
                hand_pos = hand_pos.view(batch_size, seq_len, 2, 3)
            if hand_pos.shape[0] == 1 and batch_size > 1:
                hand_pos = hand_pos.expand(batch_size, -1, -1, -1)
            if hand_pos.shape[1] != seq_len:
                hand_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)
        else:
            hand_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)

        root_vel = hp_out.get("root_vel_pred")
        root_vel = self._ensure_bt_tensor(root_vel, batch_size, seq_len, 3, device, dtype)

        pose_6d = hp_out.get("pred_full_pose_6d")
        if isinstance(pose_6d, torch.Tensor):
            pose_6d = pose_6d.to(device=device, dtype=dtype)
            if pose_6d.dim() == 3:
                pose_6d = pose_6d.view(batch_size, seq_len, self.num_joints, 6)
            if pose_6d.shape[0] == 1 and batch_size > 1:
                pose_6d = pose_6d.expand(batch_size, -1, -1, -1)
            joints_now = pose_6d.shape[2]
            if joints_now > self.num_joints:
                pose_6d = pose_6d[:, :, : self.num_joints]
            elif joints_now < self.num_joints:
                pad = torch.zeros(batch_size, seq_len, self.num_joints - joints_now, 6, device=device, dtype=dtype)
                pose_6d = torch.cat((pose_6d, pad), dim=2)
        else:
            pose_6d = torch.zeros(batch_size, seq_len, self.num_joints, 6, device=device, dtype=dtype)

        obj_trans_init = data_dict.get("obj_trans_init")
        if isinstance(obj_trans_init, torch.Tensor):
            obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
            if obj_trans_init.dim() == 1:
                obj_trans_init = obj_trans_init.unsqueeze(0).expand(batch_size, -1)
            if obj_trans_init.dim() == 2 and obj_trans_init.shape[0] == 1 and batch_size > 1:
                obj_trans_init = obj_trans_init.expand(batch_size, -1)
        else:
            obj_trans_init = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        contact_init = data_dict.get("contact_init")
        if isinstance(contact_init, torch.Tensor):
            contact_init = contact_init.to(device=device, dtype=dtype)
            if contact_init.dim() == 1:
                contact_init = contact_init.unsqueeze(0).expand(batch_size, -1)
            if contact_init.dim() == 2 and contact_init.shape[0] == 1 and batch_size > 1:
                contact_init = contact_init.expand(batch_size, -1)
            if contact_init.shape[-1] < 3:
                pad = torch.zeros(batch_size, 3 - contact_init.shape[-1], device=device, dtype=dtype)
                contact_init = torch.cat((contact_init, pad), dim=-1)
            contact_init = contact_init[:, :3]
        else:
            contact_init = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        human_imu_denorm = self._denormalize_imu(human_imu).reshape(batch_size, seq_len, -1)
        pose_feat = pose_6d.reshape(batch_size, seq_len, -1)
        obj_trans_init_seq = obj_trans_init.unsqueeze(1).expand(batch_size, seq_len, 3)
        contact_init_seq = contact_init.unsqueeze(1).expand(batch_size, seq_len, 3)

        cond = torch.cat(
            [
                human_imu_denorm,
                obj_imu,
                hand_pos.reshape(batch_size, seq_len, -1),
                root_vel,
                pose_feat,
                obj_trans_init_seq,
                contact_init_seq,
            ],
            dim=-1,
        )

        if self.head_mode == "split":
            x_seed = obj_trans_init_seq
        else:
            x_seed = torch.cat(
                [
                    obj_trans_init_seq,
                    contact_init_seq,
                    torch.zeros(batch_size, seq_len, self.boundary_dim, device=device, dtype=dtype),
                ],
                dim=-1,
            )

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "hand_pos": hand_pos,
            "has_object_mask": self._prepare_has_object_mask(
                data_dict.get("has_object"), batch_size, seq_len, device=device, dtype=dtype
            ),
        }
        return cond, x_seed, context

    def _build_x_start(self, gt_targets: dict, context: dict) -> torch.Tensor:
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]

        obj_trans_gt = gt_targets.get("obj_trans") if isinstance(gt_targets, dict) else None
        obj_trans_gt = self._ensure_bt_tensor(obj_trans_gt, batch_size, seq_len, 3, device, dtype)

        def _prepare_contact(key: str) -> torch.Tensor:
            val = gt_targets.get(key) if isinstance(gt_targets, dict) else None
            if isinstance(val, torch.Tensor):
                val = val.to(device=device, dtype=dtype)
                if val.dim() == 1:
                    val = val.unsqueeze(0)
                if val.shape[0] == 1 and batch_size > 1:
                    val = val.expand(batch_size, -1)
                if val.dim() == 2:
                    return val
            return torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

        lhand_contact = _prepare_contact("lhand_contact")
        rhand_contact = _prepare_contact("rhand_contact")
        obj_contact = _prepare_contact("obj_contact")
        contact_gt = torch.stack([lhand_contact, rhand_contact, obj_contact], dim=-1).clamp(0.0, 1.0)

        def _prepare_boundary(key_a: str, key_b: str) -> torch.Tensor:
            val = gt_targets.get(key_a, gt_targets.get(key_b)) if isinstance(gt_targets, dict) else None
            if isinstance(val, torch.Tensor):
                val = val.to(device=device, dtype=dtype)
                if val.dim() == 1:
                    val = val.unsqueeze(0)
                if val.shape[0] == 1 and batch_size > 1:
                    val = val.expand(batch_size, -1)
                if val.dim() == 2:
                    return val
            return torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

        start_gt = _prepare_boundary("interaction_start_gauss", "interaction_start")
        end_gt = _prepare_boundary("interaction_end_gauss", "interaction_end")
        boundary_gt = torch.stack([start_gt, end_gt], dim=-1)

        return torch.cat([obj_trans_gt, contact_gt, boundary_gt], dim=-1)

    def _predict_contact_boundary(self, cond: torch.Tensor, pred_obj_trans: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_in = torch.cat([cond, pred_obj_trans], dim=-1)
        h = self.cls_in_proj(cls_in).transpose(1, 2)
        h = self.cls_temporal(h)
        contact_logits = self.contact_head(h).transpose(1, 2)
        boundary_logits = self.boundary_head(h).transpose(1, 2)
        return contact_logits, boundary_logits

    def _decode_outputs(
        self,
        pred_obj_trans: torch.Tensor,
        contact_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        context: dict,
    ) -> dict:
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]

        contact_prob = torch.sigmoid(contact_logits)
        boundary_prob = torch.sigmoid(boundary_logits)

        pred_obj_vel = torch.zeros_like(pred_obj_trans)
        pred_obj_acc = torch.zeros_like(pred_obj_trans)
        if seq_len > 1:
            pred_obj_vel[:, 1:] = (pred_obj_trans[:, 1:] - pred_obj_trans[:, :-1]) * self.fps
        if seq_len > 2:
            pred_obj_acc[:, 2:] = (pred_obj_trans[:, 2:] - 2 * pred_obj_trans[:, 1:-1] + pred_obj_trans[:, :-2]) * (self.fps ** 2)

        has_object_mask = context.get("has_object_mask")
        if has_object_mask is not None:
            mask = has_object_mask.unsqueeze(-1)
            pred_obj_trans = pred_obj_trans * mask
            pred_obj_vel = pred_obj_vel * mask
            pred_obj_acc = pred_obj_acc * mask
            contact_logits = contact_logits * mask
            contact_prob = contact_prob * mask
            boundary_logits = boundary_logits * mask
            boundary_prob = boundary_prob * mask

        return {
            "pred_obj_trans": pred_obj_trans,
            "pred_obj_vel": pred_obj_vel,
            "pred_obj_vel_from_posdiff": pred_obj_vel,
            "pred_obj_acc_from_posdiff": pred_obj_acc,
            "pred_hand_contact_logits": contact_logits,
            "pred_hand_contact_prob": contact_prob,
            "pred_obj_move_prob": contact_prob[..., 2:3],
            "pred_hand_contact_prob_cond": contact_prob[..., :2],
            "pred_hand_contact_logits_cond": contact_logits[..., :2],
            "pred_interaction_boundary_logits": boundary_logits,
            "pred_interaction_boundary_prob": boundary_prob,
            "pred_hand_glb_pos": context["hand_pos"],
            "has_object": (context["has_object_mask"] > 0.5) if context.get("has_object_mask") is not None else None,
        }

    def forward(
        self,
        data_dict: dict,
        hp_out: dict,
        gt_targets: dict | None = None,
    ) -> dict:
        cond, x_seed, context = self._prepare_inputs(data_dict, hp_out)

        if self.training and gt_targets is None:
            raise ValueError("InteractionModule training requires gt_targets")

        add_noise = bool(self.training and self.use_diffusion_noise)

        if self.head_mode == "split":
            x_gt_abs = self._build_x_start(gt_targets, context)[..., : self.obj_trans_dim] if gt_targets is not None else x_seed
            if self.dit_formulation == "residual":
                x_start = x_gt_abs - x_seed
                pred_residual, aux = self.dit(cond, x_start=x_start, add_noise=add_noise)
                pred_obj_trans = x_seed + pred_residual
                aux["x_seed"] = x_seed
                aux["residual_target"] = x_start
                aux["residual_pred"] = pred_residual
            else:
                pred_obj_trans, aux = self.dit(cond, x_start=x_gt_abs, add_noise=add_noise)
            contact_logits, boundary_logits = self._predict_contact_boundary(cond, pred_obj_trans)
        else:
            x_gt_abs = self._build_x_start(gt_targets, context) if gt_targets is not None else x_seed
            if self.dit_formulation == "residual":
                x_start = x_gt_abs - x_seed
                pred_residual, aux = self.dit(cond, x_start=x_start, add_noise=add_noise)
                pred_feats = x_seed + pred_residual
                aux["x_seed"] = x_seed
                aux["residual_target"] = x_start
                aux["residual_pred"] = pred_residual
            else:
                pred_feats, aux = self.dit(cond, x_start=x_gt_abs, add_noise=add_noise)
            pred_obj_trans = pred_feats[..., : self.obj_trans_dim]
            contact_logits = pred_feats[..., self.obj_trans_dim : self.obj_trans_dim + self.contact_dim]
            boundary_logits = pred_feats[..., self.obj_trans_dim + self.contact_dim :]

        outputs = self._decode_outputs(pred_obj_trans, contact_logits, boundary_logits, context)
        outputs["diffusion_aux"] = aux
        return outputs

    @torch.no_grad()
    def inference(
        self,
        data_dict: dict,
        hp_out: dict,
        sample_steps: int | None = None,
        sampler: str | None = None,
        eta: float | None = None,
    ) -> dict:
        cond, x_seed, context = self._prepare_inputs(data_dict, hp_out)

        steps = self.inference_steps if sample_steps is None else int(sample_steps)
        if steps is None:
            steps = self.dit.timesteps
        sampler_name = self.inference_sampler if sampler is None else str(sampler).lower()
        eta_val = self.inference_eta if eta is None else float(eta)

        if self.head_mode == "split":
            if self.dit_formulation == "residual":
                pred_residual = self.dit.sample(cond=cond, x_start=None, steps=steps, sampler=sampler_name, eta=eta_val)
                pred_obj_trans = x_seed + pred_residual
            else:
                pred_obj_trans = self.dit.sample(cond=cond, x_start=x_seed, steps=steps, sampler=sampler_name, eta=eta_val)
            contact_logits, boundary_logits = self._predict_contact_boundary(cond, pred_obj_trans)
        else:
            if self.dit_formulation == "residual":
                pred_residual = self.dit.sample(cond=cond, x_start=None, steps=steps, sampler=sampler_name, eta=eta_val)
                pred_feats = x_seed + pred_residual
            else:
                pred_feats = self.dit.sample(cond=cond, x_start=x_seed, steps=steps, sampler=sampler_name, eta=eta_val)
            pred_obj_trans = pred_feats[..., : self.obj_trans_dim]
            contact_logits = pred_feats[..., self.obj_trans_dim : self.obj_trans_dim + self.contact_dim]
            boundary_logits = pred_feats[..., self.obj_trans_dim + self.contact_dim :]

        outputs = self._decode_outputs(pred_obj_trans, contact_logits, boundary_logits, context)
        outputs["diffusion_aux"] = {
            "inference_steps": steps,
            "sampler": sampler_name,
            "eta": eta_val,
            "formulation": self.dit_formulation,
            "head_mode": self.head_mode,
        }
        return outputs

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_contact = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_boundary = torch.zeros(batch_size, seq_len, 2, device=device)
        return {
            "pred_obj_trans": zeros_pos,
            "pred_obj_vel": zeros_pos,
            "pred_obj_vel_from_posdiff": zeros_pos,
            "pred_obj_acc_from_posdiff": zeros_pos,
            "pred_hand_contact_logits": zeros_contact,
            "pred_hand_contact_prob": zeros_contact,
            "pred_obj_move_prob": zeros_contact[..., 2:3],
            "pred_hand_contact_prob_cond": zeros_contact[..., :2],
            "pred_hand_contact_logits_cond": zeros_contact[..., :2],
            "pred_interaction_boundary_logits": zeros_boundary,
            "pred_interaction_boundary_prob": zeros_boundary,
            "pred_hand_glb_pos": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "has_object": None,
            "diffusion_aux": {},
        }
