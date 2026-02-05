"""
Object translation with FK fusion + learned gating (DiT produces FK params).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from .base import ConditionalDiT
from configs import FRAME_RATE, _SENSOR_NAMES


class ObjectTransModuleFK(nn.Module):
    """
    DiT predicts FK directions/lengths; a learned MLP gating fuses FK and IMU integration.
    """

    def __init__(self, cfg):
        super().__init__()
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.obj_imu_dim = getattr(cfg, "obj_imu_dim", self.imu_dim)

        self.human_imu_cond_dim = self.num_human_imus * self.imu_dim
        self.contact_init_dim = 3

        # DiT predicts: l_dir(3) + r_dir(3) + l_len(1) + r_len(1)
        self.target_dim = 8

        cond_dim = (
            6  # hand positions
            + 3  # contact prob
            + self.obj_imu_dim
            + 3  # obj vel input
            + self.human_imu_cond_dim
            + self.contact_init_dim
            + 3  # obj rot delta
        )

        dit_cfg = getattr(cfg, "dit", {})

        def _dit_param(name, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        max_seq_len = _dit_param("dit_max_seq_len", getattr(getattr(cfg, "train", {}), "window", 256))
        self.use_diffusion_noise = bool(_dit_param("dit_use_noise", False))
        self.dit_x_start_mode = str(_dit_param("dit_x_start_mode", "gt")).lower()
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

        gating_hidden = _dit_param("ot_gating_hidden", 64)
        self.gating_temperature = float(_dit_param("ot_gating_temperature", 1.0))
        self.gating_prior_beta = float(_dit_param("ot_gating_prior_beta", 5.0))
        gating_in_dim = 3 + 3 + 3  # contact prob + obj vel + obj imu acc
        self.gating_net = nn.Sequential(
            nn.Linear(gating_in_dim, gating_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gating_hidden, gating_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gating_hidden, 3),
        )
        self.vel_static_threshold = float(getattr(cfg, "vel_static_threshold", 0.3))
        self.vel_min_hand_speed = float(getattr(cfg, "vel_min_hand_speed", 0.02))

    @staticmethod
    def _unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return x / norm

    @staticmethod
    def _softplus_positive(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-4

    @staticmethod
    def _rot6d_delta(rot6d: torch.Tensor) -> torch.Tensor:
        """Compute axis-angle delta between successive rotations."""
        bs, seq_len, _ = rot6d.shape
        if seq_len <= 1:
            return torch.zeros(bs, seq_len, 3, device=rot6d.device, dtype=rot6d.dtype)
        R = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(bs, seq_len, 3, 3)
        rel = torch.matmul(R[:, 1:].transpose(-1, -2), R[:, :-1])
        aa = matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(bs, seq_len - 1, 3)
        aa = F.pad(aa, (0, 0, 1, 0))
        return aa

    @staticmethod
    def _compute_hand_velocity(hand_pos: torch.Tensor) -> torch.Tensor:
        """Finite-difference hand velocity in world frame."""
        vel = torch.zeros_like(hand_pos)
        if hand_pos.size(1) > 1:
            vel[:, 1:] = (hand_pos[:, 1:] - hand_pos[:, :-1]) * FRAME_RATE
        return vel

    def _correct_obj_velocity(
        self,
        v_imu: torch.Tensor,
        v_lhand: torch.Tensor,
        v_rhand: torch.Tensor,
        p_left: torch.Tensor,
        p_right: torch.Tensor,
        p_move: torch.Tensor,
    ) -> torch.Tensor:
        """
        Copy of RNN velocity correction: blend IMU velocity with hand motion when in contact.
        """

        static_factor = torch.clamp(p_move / self.vel_static_threshold, 0, 1)

        def direction_correct(v_obj, v_hand, p_contact):
            v_hand_speed = v_hand.norm(dim=-1, keepdim=True)
            hand_moving = (v_hand_speed > self.vel_min_hand_speed).float()
            v_hand_dir = v_hand / v_hand_speed.clamp_min(1e-6)
            proj_scalar = (v_obj * v_hand_dir).sum(dim=-1, keepdim=True)
            v_corrected = v_obj - proj_scalar * v_hand_dir + v_hand
            w = p_contact * hand_moving
            return v_obj * (1 - w) + v_corrected * w

        v_lcorr = direction_correct(v_imu, v_lhand, p_left)
        v_rcorr = direction_correct(v_imu, v_rhand, p_right)

        total_p = p_left + p_right + 1e-6
        v_contact_corrected = (p_left / total_p) * v_lcorr + (p_right / total_p) * v_rcorr

        return static_factor * v_contact_corrected

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
    ):
        if hand_positions is None:
            raise ValueError("hand_positions cannot be None")
        if hand_positions.dim() == 3:
            bs, seq_len, _ = hand_positions.shape
            hand_positions = hand_positions.view(bs, seq_len, 2, 3)
        elif hand_positions.dim() == 4:
            bs, seq_len = hand_positions.shape[:2]
        else:
            raise ValueError(f"Unexpected hand_positions shape {hand_positions.shape}")

        device = hand_positions.device
        dtype = hand_positions.dtype
        lhand_position = hand_positions[:, :, 0, :]
        rhand_position = hand_positions[:, :, 1, :]

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

        if obj_vel_input is None:
            obj_vel_input = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)

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

        contact_prob = pred_hand_contact_prob
        if contact_prob.dim() == 2:
            contact_prob = contact_prob.unsqueeze(-1)
        if contact_prob.shape[-1] < 3:
            pad = torch.zeros(bs, seq_len, 3 - contact_prob.shape[-1], device=device, dtype=dtype)
            contact_prob = torch.cat((contact_prob, pad), dim=-1)
        pL = contact_prob[:, :, 0:1]
        pR = contact_prob[:, :, 1:2]
        p_move = contact_prob[:, :, 2:3]

        obj_rot = obj_imu[:, :, 3:9]
        obj_rot_delta = self._rot6d_delta(obj_rot)
        lhand_vel = self._compute_hand_velocity(lhand_position)
        rhand_vel = self._compute_hand_velocity(rhand_position)
        obj_vel_input = self._correct_obj_velocity(obj_vel_input, lhand_vel, rhand_vel, pL, pR, p_move)
        obj_rotm = rotation_6d_to_matrix(obj_rot.reshape(-1, 6)).reshape(bs, seq_len, 3, 3)

        cond = torch.cat(
            (
                hand_positions.reshape(bs, seq_len, -1),
                contact_prob,
                obj_imu,
                obj_vel_input,
                human_imu,
                contact_init_seq,
                obj_rot_delta,
            ),
            dim=-1,
        )

        # Seeds for FK params
        obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
        vec_l0_world = obj_trans_init - lhand_position[:, 0, :]
        vec_r0_world = obj_trans_init - rhand_position[:, 0, :]
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
        x_seed = torch.cat((dir_l_seed, dir_r_seed, len_l_seed, len_r_seed), dim=-1)

        x_start = x_seed
        use_gt_xstart = self.dit_x_start_mode == "gt" and gt_targets is not None
        if use_gt_xstart:
            tensor_dtype = dtype
            lhand_dir_gt = gt_targets.get("lhand_obj_direction") if isinstance(gt_targets, dict) else None
            rhand_dir_gt = gt_targets.get("rhand_obj_direction") if isinstance(gt_targets, dict) else None
            def _prepare_dir(val):
                if isinstance(val, torch.Tensor):
                    val = val.to(device=device, dtype=tensor_dtype)
                    if val.dim() == 2:
                        val = val.unsqueeze(0)
                    if val.shape[0] == 1 and bs > 1:
                        val = val.expand(bs, -1, -1)
                    return val
                return None
            lhand_dir_gt = _prepare_dir(lhand_dir_gt)
            rhand_dir_gt = _prepare_dir(rhand_dir_gt)

            obj_trans_gt = gt_targets.get("obj_trans") if isinstance(gt_targets, dict) else None
            if isinstance(obj_trans_gt, torch.Tensor):
                obj_trans_gt = obj_trans_gt.to(device=device, dtype=tensor_dtype)
                if obj_trans_gt.dim() == 2:
                    obj_trans_gt = obj_trans_gt.unsqueeze(0)
                if obj_trans_gt.shape[0] == 1 and bs > 1:
                    obj_trans_gt = obj_trans_gt.expand(bs, -1, -1)

            position_global_gt = gt_targets.get("position_global") if isinstance(gt_targets, dict) else None
            len_l_gt = None
            len_r_gt = None
            # prefer dataset-precomputed lengths if available
            lhand_lb_gt = gt_targets.get("lhand_lb") if isinstance(gt_targets, dict) else None
            rhand_lb_gt = gt_targets.get("rhand_lb") if isinstance(gt_targets, dict) else None
            if isinstance(lhand_lb_gt, torch.Tensor):
                lhand_lb_gt = lhand_lb_gt.to(device=device, dtype=tensor_dtype)
                if lhand_lb_gt.dim() == 1:
                    lhand_lb_gt = lhand_lb_gt.unsqueeze(0)
                if lhand_lb_gt.shape[0] == 1 and bs > 1:
                    lhand_lb_gt = lhand_lb_gt.expand(bs, -1)
            if isinstance(rhand_lb_gt, torch.Tensor):
                rhand_lb_gt = rhand_lb_gt.to(device=device, dtype=tensor_dtype)
                if rhand_lb_gt.dim() == 1:
                    rhand_lb_gt = rhand_lb_gt.unsqueeze(0)
                if rhand_lb_gt.shape[0] == 1 and bs > 1:
                    rhand_lb_gt = rhand_lb_gt.expand(bs, -1)
            if isinstance(lhand_lb_gt, torch.Tensor):
                len_l_gt = lhand_lb_gt.unsqueeze(-1)
            if isinstance(rhand_lb_gt, torch.Tensor):
                len_r_gt = rhand_lb_gt.unsqueeze(-1)
            if isinstance(position_global_gt, torch.Tensor) and isinstance(obj_trans_gt, torch.Tensor):
                position_global_gt = position_global_gt.to(device=device, dtype=tensor_dtype)
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

            x_start = torch.cat((lhand_dir_gt, rhand_dir_gt, len_l_gt, len_r_gt), dim=-1)

        add_noise = self.training and self.use_diffusion_noise
        pred_feats, aux = self.dit(cond, x_start=x_start, add_noise=add_noise)

        offset = 0
        pred_lhand_dir = self._unit_vector(pred_feats[..., offset : offset + 3])
        offset += 3
        pred_rhand_dir = self._unit_vector(pred_feats[..., offset : offset + 3])
        offset += 3
        pred_lhand_lb = self._softplus_positive(pred_feats[..., offset : offset + 1])
        offset += 1
        pred_rhand_lb = self._softplus_positive(pred_feats[..., offset : offset + 1])

        # FK positions
        obj_rotm_flat = obj_rotm.reshape(bs * seq_len, 3, 3)
        l_dir_world = torch.bmm(obj_rotm_flat, pred_lhand_dir.reshape(bs * seq_len, 3, 1)).reshape(bs, seq_len, 3)
        r_dir_world = torch.bmm(obj_rotm_flat, pred_rhand_dir.reshape(bs * seq_len, 3, 1)).reshape(bs, seq_len, 3)
        l_pos_fk = lhand_position + l_dir_world * pred_lhand_lb
        r_pos_fk = rhand_position + r_dir_world * pred_rhand_lb

        # IMU integration path
        dt = 1.0 / FRAME_RATE
        pos_imu = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)
        pos_imu[:, 0] = obj_trans_init
        if seq_len > 1:
            vel_dt = obj_vel_input * dt
            cumsum_vel = vel_dt.cumsum(dim=1)
            pos_imu[:, 1:] = obj_trans_init.unsqueeze(1) + cumsum_vel[:, 1:]

        # Gating weights (softmax + prior)
        gating_input = torch.cat((contact_prob, obj_vel_input, obj_imu[:, :, :3]), dim=-1)
        gating_logits = self.gating_net(gating_input)
        prior_im = 1.0 - p_move.squeeze(-1)
        prior = torch.stack([pL.squeeze(-1), pR.squeeze(-1), prior_im], dim=-1)
        gating_logits = gating_logits + self.gating_prior_beta * torch.log(prior + 1e-6)
        gating_weights_raw = F.softmax(gating_logits / self.gating_temperature, dim=-1)
        gating_weights = gating_weights_raw

        fused_pos = (
            gating_weights[..., 0:1] * l_pos_fk
            + gating_weights[..., 1:2] * r_pos_fk
            + gating_weights[..., 2:3] * pos_imu
        )

        vel_from_pos = torch.zeros_like(fused_pos)
        acc_from_pos = torch.zeros_like(fused_pos)
        if seq_len > 1:
            vel_from_pos[:, 1:] = (fused_pos[:, 1:] - fused_pos[:, :-1]) * FRAME_RATE
        if seq_len > 2:
            acc_from_pos[:, 2:] = (fused_pos[:, 2:] - 2 * fused_pos[:, 1:-1] + fused_pos[:, :-2]) * (FRAME_RATE**2)

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
            fused_pos = fused_pos * mask
            l_pos_fk = l_pos_fk * mask
            r_pos_fk = r_pos_fk * mask
            pred_lhand_dir = pred_lhand_dir * mask
            pred_rhand_dir = pred_rhand_dir * mask
            pred_lhand_lb = pred_lhand_lb * mask
            pred_rhand_lb = pred_rhand_lb * mask
            vel_from_pos = vel_from_pos * mask
            acc_from_pos = acc_from_pos * mask
            gating_weights = gating_weights * mask
            gating_weights_raw = gating_weights_raw * mask

        return {
            "pred_obj_trans": fused_pos,
            "pred_lhand_obj_direction": pred_lhand_dir,
            "pred_rhand_obj_direction": pred_rhand_dir,
            "pred_lhand_lb": pred_lhand_lb.squeeze(-1),
            "pred_rhand_lb": pred_rhand_lb.squeeze(-1),
            "pred_obj_vel_from_posdiff": vel_from_pos,
            "pred_obj_acc_from_posdiff": acc_from_pos,
            "obj_vel_input": obj_vel_input,
            "obj_vel_corrected": obj_vel_input,
            "gating_weights": gating_weights,
            "gating_weights_raw": gating_weights_raw,
            "pred_lhand_obj_trans": l_pos_fk,
            "pred_rhand_obj_trans": r_pos_fk,
            "diffusion_aux": aux,
        }

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_dir = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_scalar = torch.zeros(batch_size, seq_len, device=device)
        zeros_weights = torch.zeros(batch_size, seq_len, 3, device=device)
        return {
            "pred_obj_trans": zeros_pos,
            "gating_weights": zeros_weights,
            "gating_weights_raw": zeros_weights,
            "pred_obj_vel_from_posdiff": zeros_pos,
            "pred_obj_acc_from_posdiff": zeros_pos,
            "obj_vel_input": zeros_pos,
            "obj_vel_corrected": zeros_pos,
            "pred_lhand_obj_direction": zeros_dir,
            "pred_rhand_obj_direction": zeros_dir,
            "pred_lhand_lb": zeros_scalar,
            "pred_rhand_lb": zeros_scalar,
            "pred_lhand_obj_trans": zeros_pos,
            "pred_rhand_obj_trans": zeros_pos,
            "diffusion_aux": {},
        }
