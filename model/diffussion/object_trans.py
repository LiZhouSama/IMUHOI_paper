"""
DiT-based ObjectTransModule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix

from .base import ConditionalDiT
from configs import FRAME_RATE, _SENSOR_NAMES


class ObjectTransModule(nn.Module):
    """Diffusion Transformer replacement for Stage-3 object translation."""

    def __init__(self, cfg):
        super().__init__()
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.obj_imu_dim = getattr(cfg, "obj_imu_dim", self.imu_dim)
        self.num_joints = getattr(cfg, "num_joints", 24)
        self.target_dim = 3 + 3 + 3 + 1 + 1  # obj trans, dir L, dir R, len L, len R
        self.human_imu_cond_dim = self.num_human_imus * self.imu_dim
        self.contact_init_dim = 3
        cond_dim = (
            6  # hand positions
            + 3  # contact prob
            + self.obj_imu_dim
            + 3  # obj vel input
            + self.human_imu_cond_dim
            + self.contact_init_dim
        )

        dit_cfg = getattr(cfg, "dit", {})

        def _dit_param(name, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        max_seq_len = _dit_param("dit_max_seq_len", getattr(getattr(cfg, "train", {}), "window", 256))
        self.use_diffusion_noise = bool(_dit_param("dit_use_noise", False))
        self.dit_x_start_mode = str(_dit_param("dit_x_start_mode", "gt")).lower()
        self.inference_steps = _dit_param("dit_inference_steps", None)
        if self.inference_steps is not None:
            self.inference_steps = int(self.inference_steps)
        self.teacher_force_noise_std = float(_dit_param("teacher_force_noise_std", getattr(cfg, "teacher_force_noise_std", 0.01)))
        self.dit = ConditionalDiT(
            target_dim=self.target_dim,
            cond_dim=cond_dim,
            d_model=_dit_param("dit_d_model", 256),
            nhead=_dit_param("dit_nhead", 8),
            num_layers=_dit_param("dit_num_layers", 6),
            dim_feedforward=_dit_param("dit_dim_feedforward", 1024),
            dropout=_dit_param("dit_dropout", 0.1),
            max_seq_len=max_seq_len,
            timesteps=_dit_param("dit_timesteps", 1000),
            use_time_embed=_dit_param("dit_use_time_embed", True),
        )

    @staticmethod
    def _unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return x / norm

    @staticmethod
    def _softplus_positive(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-4

    def _prepare_inputs(
        self,
        hand_positions: torch.Tensor | None,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        obj_imu: torch.Tensor | None = None,
        human_imu: torch.Tensor | None = None,
        obj_vel_input: torch.Tensor | None = None,
        contact_init: torch.Tensor | None = None,
        has_object_mask: torch.Tensor | None = None,
        gt_targets: dict | None = None,
        teacher_forcing: bool = False,
    ):
        """Shared conditioning + seeds for both training and inference."""
        if hand_positions is None and not (teacher_forcing and gt_targets is not None):
            raise ValueError("hand_positions cannot be None")
        if hand_positions is not None and hand_positions.dim() == 3:
            bs, seq_len, _ = hand_positions.shape
            hand_positions = hand_positions.view(bs, seq_len, 2, 3)
        elif hand_positions is not None and hand_positions.dim() == 4:
            bs, seq_len = hand_positions.shape[:2]
        elif hand_positions is None:
            pos_gt_tf = gt_targets.get("position_global") if isinstance(gt_targets, dict) else None
            if isinstance(pos_gt_tf, torch.Tensor):
                if pos_gt_tf.dim() == 3:
                    pos_gt_tf = pos_gt_tf.unsqueeze(0)
                bs, seq_len = pos_gt_tf.shape[:2]
                hand_positions = pos_gt_tf[:, :, [20, 21], :]
            else:
                raise ValueError("hand_positions cannot be None and GT is not available for teacher forcing")
        else:
            raise ValueError(f"Unexpected hand_positions shape {hand_positions.shape}")

        device = hand_positions.device
        dtype = hand_positions.dtype

        def _prepare_gt_tensor(val, expected_dim):
            if not isinstance(val, torch.Tensor):
                return None
            tensor = val.to(device=device, dtype=dtype)
            if tensor.dim() == len(expected_dim) - 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] == 1 and bs > 1:
                tensor = tensor.expand(bs, *tensor.shape[1:])
            if tensor.shape[0] != bs:
                return None
            return tensor

        use_tf_cond = teacher_forcing and self.training and gt_targets is not None
        hand_positions_cond = hand_positions
        contact_prob_cond = pred_hand_contact_prob
        obj_vel_cond = obj_vel_input
        if use_tf_cond:
            pos_gt = _prepare_gt_tensor(
                gt_targets.get("position_global") if isinstance(gt_targets, dict) else None, (bs, seq_len, self.num_joints, 3)
            )
            if pos_gt is not None and pos_gt.shape[1] == seq_len:
                hand_positions_cond = pos_gt[:, :, [20, 21], :]
            l_contact_gt = _prepare_gt_tensor(gt_targets.get("lhand_contact") if isinstance(gt_targets, dict) else None, (bs, seq_len))
            r_contact_gt = _prepare_gt_tensor(gt_targets.get("rhand_contact") if isinstance(gt_targets, dict) else None, (bs, seq_len))
            obj_contact_gt = _prepare_gt_tensor(gt_targets.get("obj_contact") if isinstance(gt_targets, dict) else None, (bs, seq_len))
            if l_contact_gt is not None and r_contact_gt is not None and obj_contact_gt is not None:
                contact_prob_cond = torch.stack(
                    [l_contact_gt.float(), r_contact_gt.float(), obj_contact_gt.float()],
                    dim=-1,
                )
                if self.teacher_force_noise_std > 0:
                    contact_prob_cond = (contact_prob_cond + torch.randn_like(contact_prob_cond) * self.teacher_force_noise_std).clamp(
                        0.0, 1.0
                    )
            obj_vel_gt = _prepare_gt_tensor(gt_targets.get("obj_vel") if isinstance(gt_targets, dict) else None, (bs, seq_len, 3))
            if obj_vel_gt is not None:
                if self.teacher_force_noise_std > 0:
                    obj_vel_gt = obj_vel_gt + torch.randn_like(obj_vel_gt) * self.teacher_force_noise_std
                obj_vel_cond = obj_vel_gt
            if self.teacher_force_noise_std > 0 and hand_positions_cond is not None:
                hand_positions_cond = hand_positions_cond + torch.randn_like(hand_positions_cond) * self.teacher_force_noise_std

        hand_positions = hand_positions_cond
        lhand_position = hand_positions[:, :, 0, :]
        rhand_position = hand_positions[:, :, 1, :]

        obj_trans_init_vec = obj_trans_init.to(device=device, dtype=dtype)
        if obj_trans_init_vec.dim() == 1:
            obj_trans_init_vec = obj_trans_init_vec.unsqueeze(0).expand(bs, -1)
        elif obj_trans_init_vec.dim() == 2 and obj_trans_init_vec.shape[0] == 1 and bs > 1:
            obj_trans_init_vec = obj_trans_init_vec.expand(bs, -1)

        if obj_imu is None:
            obj_imu = torch.zeros(bs, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        else:
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(bs, seq_len, -1)
            if obj_imu.shape[-1] != self.obj_imu_dim:
                raise ValueError(f"obj_imu feature dim mismatch: expected {self.obj_imu_dim}, got {obj_imu.shape[-1]}")

        if human_imu is None:
            human_imu = torch.zeros(bs, seq_len, self.human_imu_cond_dim, device=device, dtype=dtype)
        else:
            if human_imu.dim() == 4:
                human_imu = human_imu.reshape(bs, seq_len, -1)
            if human_imu.shape[-1] != self.human_imu_cond_dim:
                raise ValueError(
                    f"human_imu feature dim mismatch: expected {self.human_imu_cond_dim}, got {human_imu.shape[-1]}"
                )

        if obj_vel_cond is None:
            obj_vel_input = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)
        else:
            obj_vel_input = obj_vel_cond

        if contact_init is None:
            contact_init_seq = torch.zeros(bs, seq_len, self.contact_init_dim, device=device, dtype=dtype)
        else:
            if contact_init.dim() == 1:
                contact_init_seq = contact_init.view(1, 1, -1).expand(bs, seq_len, -1)
            elif contact_init.dim() == 2:
                contact_init_seq = contact_init.unsqueeze(1).expand(bs, seq_len, -1)
            else:
                contact_init_seq = contact_init
            if contact_init_seq.shape[-1] != self.contact_init_dim:
                raise ValueError(f"contact_init last dim expected {self.contact_init_dim}, got {contact_init_seq.shape[-1]}")

        contact_prob = contact_prob_cond
        if contact_prob.dim() == 2:
            contact_prob = contact_prob.unsqueeze(-1)
        if contact_prob.shape[-1] < 3:
            pad = torch.zeros(bs, seq_len, 3 - contact_prob.shape[-1], device=device, dtype=dtype)
            contact_prob = torch.cat((contact_prob, pad), dim=-1)
        pL = contact_prob[:, :, 0:1]
        pR = contact_prob[:, :, 1:2]
        p_move = contact_prob[:, :, 2:3]

        obj_rot = obj_imu[:, :, 3:9]
        obj_rotm = rotation_6d_to_matrix(obj_rot.reshape(-1, 6)).reshape(bs, seq_len, 3, 3)

        cond = torch.cat(
            (
                hand_positions.reshape(bs, seq_len, -1),
                contact_prob,
                obj_imu,
                obj_vel_input,
                human_imu,
                contact_init_seq,
            ),
            dim=-1,
        )

        pos_seed = obj_trans_init_vec.unsqueeze(1).expand(bs, seq_len, 3)
        vec_l0_world = obj_trans_init_vec - lhand_position[:, 0, :]
        vec_r0_world = obj_trans_init_vec - rhand_position[:, 0, :]
        rot0_T = obj_rotm[:, 0].transpose(-1, -2)
        vec_l0_obj = torch.bmm(rot0_T, vec_l0_world.unsqueeze(-1)).squeeze(-1)
        vec_r0_obj = torch.bmm(rot0_T, vec_r0_world.unsqueeze(-1)).squeeze(-1)
        dir_l0 = self._unit_vector(vec_l0_obj).unsqueeze(1)
        dir_r0 = self._unit_vector(vec_r0_obj).unsqueeze(1)
        len_l0 = vec_l0_world.norm(dim=-1, keepdim=True).unsqueeze(1)
        len_r0 = vec_r0_world.norm(dim=-1, keepdim=True).unsqueeze(1)
        dir_l_seed = dir_l0.expand(bs, seq_len, 3)
        dir_r_seed = dir_r0.expand(bs, seq_len, 3)
        len_l_seed = len_l0.expand(bs, seq_len, 1)
        len_r_seed = len_r0.expand(bs, seq_len, 1)

        x_seed = torch.cat((pos_seed, dir_l_seed, dir_r_seed, len_l_seed, len_r_seed), dim=-1)

        context = {
            "batch_size": bs,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "obj_rotm": obj_rotm,
            "lhand_position": lhand_position,
            "rhand_position": rhand_position,
            "contact_prob": contact_prob,
            "has_object_mask": has_object_mask,
            "obj_vel_input": obj_vel_input,
            "obj_trans_init": obj_trans_init_vec,
            "dir_l_seed": dir_l_seed,
            "dir_r_seed": dir_r_seed,
            "len_l_seed": len_l_seed,
            "len_r_seed": len_r_seed,
        }
        return cond, x_seed, context

    def _build_x_start_from_gt(self, gt_targets: dict, context: dict):
        """Construct x_start from ground truth signals for training."""
        bs = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        obj_rotm = context["obj_rotm"]
        dir_l_seed = context["dir_l_seed"]
        dir_r_seed = context["dir_r_seed"]
        len_l_seed = context["len_l_seed"]
        len_r_seed = context["len_r_seed"]

        obj_trans_gt = gt_targets.get("obj_trans") if isinstance(gt_targets, dict) else None
        if isinstance(obj_trans_gt, torch.Tensor):
            obj_trans_gt = obj_trans_gt.to(device=device, dtype=dtype)
            if obj_trans_gt.dim() == 2:
                obj_trans_gt = obj_trans_gt.unsqueeze(0)
            if obj_trans_gt.shape[0] == 1 and bs > 1:
                obj_trans_gt = obj_trans_gt.expand(bs, -1, -1)
        if obj_trans_gt is None or obj_trans_gt.shape[0] != bs:
            obj_trans_gt = context["obj_trans_init"].unsqueeze(1).expand(bs, seq_len, 3)

        lhand_dir_gt = gt_targets.get("lhand_obj_direction") if isinstance(gt_targets, dict) else None
        rhand_dir_gt = gt_targets.get("rhand_obj_direction") if isinstance(gt_targets, dict) else None

        def _prepare_dir(val):
            if isinstance(val, torch.Tensor):
                val = val.to(device=device, dtype=dtype)
                if val.dim() == 2:
                    val = val.unsqueeze(0)
                if val.shape[0] == 1 and bs > 1:
                    val = val.expand(bs, -1, -1)
                return val
            return None

        lhand_dir_gt = _prepare_dir(lhand_dir_gt)
        rhand_dir_gt = _prepare_dir(rhand_dir_gt)

        position_global_gt = gt_targets.get("position_global") if isinstance(gt_targets, dict) else None
        len_l_gt = None
        len_r_gt = None
        lhand_lb_gt = gt_targets.get("lhand_lb") if isinstance(gt_targets, dict) else None
        rhand_lb_gt = gt_targets.get("rhand_lb") if isinstance(gt_targets, dict) else None
        if isinstance(lhand_lb_gt, torch.Tensor):
            lhand_lb_gt = lhand_lb_gt.to(device=device, dtype=dtype)
            if lhand_lb_gt.dim() == 1:
                lhand_lb_gt = lhand_lb_gt.unsqueeze(0)
            if lhand_lb_gt.shape[0] == 1 and bs > 1:
                lhand_lb_gt = lhand_lb_gt.expand(bs, -1)
            len_l_gt = lhand_lb_gt.unsqueeze(-1)
        if isinstance(rhand_lb_gt, torch.Tensor):
            rhand_lb_gt = rhand_lb_gt.to(device=device, dtype=dtype)
            if rhand_lb_gt.dim() == 1:
                rhand_lb_gt = rhand_lb_gt.unsqueeze(0)
            if rhand_lb_gt.shape[0] == 1 and bs > 1:
                rhand_lb_gt = rhand_lb_gt.expand(bs, -1)
            len_r_gt = rhand_lb_gt.unsqueeze(-1)
        if isinstance(position_global_gt, torch.Tensor):
            position_global_gt = position_global_gt.to(device=device, dtype=dtype)
            if position_global_gt.dim() == 3:
                position_global_gt = position_global_gt.unsqueeze(0)
            if position_global_gt.shape[0] == 1 and bs > 1:
                position_global_gt = position_global_gt.expand(bs, -1, -1, -1)
            lhand_pos_gt = position_global_gt[:, :, 20, :]
            rhand_pos_gt = position_global_gt[:, :, 21, :]
            len_l_gt = (obj_trans_gt - lhand_pos_gt).norm(dim=-1, keepdim=True)
            len_r_gt = (obj_trans_gt - rhand_pos_gt).norm(dim=-1, keepdim=True)
            if lhand_dir_gt is None:
                vec_l_gt = obj_trans_gt - lhand_pos_gt
                lhand_dir_gt = torch.bmm(obj_rotm.transpose(-1, -2), vec_l_gt.unsqueeze(-1)).squeeze(-1)
                lhand_dir_gt = self._unit_vector(lhand_dir_gt)
            if rhand_dir_gt is None:
                vec_r_gt = obj_trans_gt - rhand_pos_gt
                rhand_dir_gt = torch.bmm(obj_rotm.transpose(-1, -2), vec_r_gt.unsqueeze(-1)).squeeze(-1)
                rhand_dir_gt = self._unit_vector(rhand_dir_gt)

        if lhand_dir_gt is None:
            lhand_dir_gt = dir_l_seed
        if rhand_dir_gt is None:
            rhand_dir_gt = dir_r_seed
        if len_l_gt is None:
            len_l_gt = len_l_seed
        if len_r_gt is None:
            len_r_gt = len_r_seed

        return torch.cat((obj_trans_gt, lhand_dir_gt, rhand_dir_gt, len_l_gt, len_r_gt), dim=-1)

    def _decode_outputs(self, pred_feats: torch.Tensor, context: dict):
        """Decode transformer outputs into structured predictions."""
        bs = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        has_object_mask = context["has_object_mask"]
        obj_vel_input = context["obj_vel_input"]

        offset = 0
        pred_obj_trans = pred_feats[..., offset : offset + 3]
        offset += 3
        pred_lhand_dir = self._unit_vector(pred_feats[..., offset : offset + 3])
        offset += 3
        pred_rhand_dir = self._unit_vector(pred_feats[..., offset : offset + 3])
        offset += 3
        pred_lhand_lb = self._softplus_positive(pred_feats[..., offset : offset + 1])
        offset += 1
        pred_rhand_lb = self._softplus_positive(pred_feats[..., offset : offset + 1])

        vel_from_pos = torch.zeros_like(pred_obj_trans)
        acc_from_pos = torch.zeros_like(pred_obj_trans)
        if seq_len > 1:
            vel_from_pos[:, 1:] = (pred_obj_trans[:, 1:] - pred_obj_trans[:, :-1]) * FRAME_RATE
        if seq_len > 2:
            acc_from_pos[:, 2:] = (pred_obj_trans[:, 2:] - 2 * pred_obj_trans[:, 1:-1] + pred_obj_trans[:, :-2]) * (
                FRAME_RATE**2
            )

        if has_object_mask is not None:
            if has_object_mask.dim() == 1:
                mask_bt = has_object_mask.view(bs, 1).expand(bs, seq_len)
            elif has_object_mask.dim() == 2:
                mask_bt = has_object_mask
                if mask_bt.shape[0] == 1 and bs > 1:
                    mask_bt = mask_bt.expand(bs, seq_len)
            else:
                raise ValueError(f"has_object_mask must be [B] or [B,T], got {has_object_mask.shape}")

            mask = mask_bt.to(dtype=dtype, device=device).unsqueeze(-1)
            pred_obj_trans = pred_obj_trans * mask
            pred_lhand_dir = pred_lhand_dir * mask
            pred_rhand_dir = pred_rhand_dir * mask
            pred_lhand_lb = pred_lhand_lb * mask
            pred_rhand_lb = pred_rhand_lb * mask
            vel_from_pos = vel_from_pos * mask
            acc_from_pos = acc_from_pos * mask

        zeros_weights = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)

        return {
            "pred_obj_trans": pred_obj_trans,
            "pred_lhand_obj_direction": pred_lhand_dir,
            "pred_rhand_obj_direction": pred_rhand_dir,
            "pred_lhand_lb": pred_lhand_lb.squeeze(-1),
            "pred_rhand_lb": pred_rhand_lb.squeeze(-1),
            "pred_obj_vel_from_posdiff": vel_from_pos,
            "pred_obj_acc_from_posdiff": acc_from_pos,
            "gating_weights": zeros_weights,
            "gating_weights_raw": zeros_weights,
            "obj_vel_input": obj_vel_input,
        }

    def forward(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        obj_imu: torch.Tensor = None,
        human_imu: torch.Tensor = None,
        obj_vel_input: torch.Tensor = None,
        contact_init: torch.Tensor = None,
        has_object_mask: torch.Tensor = None,
        gt_targets: dict | None = None,
        teacher_forcing: bool = False,
    ):
        cond, x_seed, context = self._prepare_inputs(
            hand_positions,
            pred_hand_contact_prob,
            obj_trans_init,
            obj_imu=obj_imu,
            human_imu=human_imu,
            obj_vel_input=obj_vel_input,
            contact_init=contact_init,
            has_object_mask=has_object_mask,
            gt_targets=gt_targets,
            teacher_forcing=teacher_forcing,
        )

        using_gt_xstart = self.training and self.dit_x_start_mode == "gt"
        if using_gt_xstart and gt_targets is None:
            raise ValueError("gt_targets is required during training when dit_x_start_mode='gt'")

        x_start = self._build_x_start_from_gt(gt_targets, context) if (using_gt_xstart and gt_targets is not None) else x_seed
        add_noise = self.training and (self.use_diffusion_noise or (using_gt_xstart and gt_targets is not None))
        pred_feats, aux = self.dit(cond, x_start=x_start, add_noise=add_noise)

        outputs = self._decode_outputs(pred_feats, context)
        outputs["diffusion_aux"] = aux
        return outputs

    @torch.no_grad()
    def inference(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        obj_imu: torch.Tensor = None,
        human_imu: torch.Tensor = None,
        obj_vel_input: torch.Tensor = None,
        contact_init: torch.Tensor = None,
        has_object_mask: torch.Tensor = None,
        sample_steps: int | None = None,
    ):
        """纯推理接口，固定使用采样路径且不接收gt_targets。"""
        cond, x_seed, context = self._prepare_inputs(
            hand_positions,
            pred_hand_contact_prob,
            obj_trans_init,
            obj_imu=obj_imu,
            human_imu=human_imu,
            obj_vel_input=obj_vel_input,
            contact_init=contact_init,
            has_object_mask=has_object_mask,
            gt_targets=None,
            teacher_forcing=False,
        )
        steps = sample_steps if sample_steps is not None else self.inference_steps
        pred_feats = self.dit.sample(cond=cond, x_start=x_seed, steps=steps)
        outputs = self._decode_outputs(pred_feats, context)
        outputs["diffusion_aux"] = {"inference_steps": steps if steps is not None else self.dit.timesteps}
        return outputs

    # training/inference logic rewritten above

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_dir = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_scalar = torch.zeros(batch_size, seq_len, device=device)
        zeros_weights = torch.zeros(batch_size, seq_len, 3, device=device)
        return {
            "pred_obj_trans": zeros_pos,
            "pred_obj_vel_from_posdiff": zeros_pos,
            "pred_obj_acc_from_posdiff": zeros_pos,
            "obj_vel_input": zeros_pos,
            "pred_lhand_obj_direction": zeros_dir,
            "pred_rhand_obj_direction": zeros_dir,
            "pred_lhand_lb": zeros_scalar,
            "pred_rhand_lb": zeros_scalar,
            "gating_weights": zeros_weights,
            "gating_weights_raw": zeros_weights,
            "diffusion_aux": {},
        }
