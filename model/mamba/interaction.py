"""Unified Mamba interaction module for object/contact estimation and pose refinement."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import FRAME_RATE, _REDUCED_POSE_NAMES, _SENSOR_NAMES
from utils.human_pose import (
    compute_smpl_joints_from_global,
    ensure_body_model_device,
    integrate_root_velocity,
    load_body_model,
    reduced_root_pose_to_full_global,
)
from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix

from .base import CausalMovingAverage, CrossScaleFusion, InputStem, MambaBlock, RMSNorm, TemporalMambaEncoder


def _as_mapping(value):
    return value if isinstance(value, dict) else {}


def _cfg_lookup(cfg, name: str, default):
    """Read interaction config from nested Mamba sections, then legacy top-level keys."""
    mamba_cfg = getattr(cfg, "mamba", {})
    candidates = []
    if isinstance(mamba_cfg, dict):
        candidates.append(_as_mapping(mamba_cfg.get("interaction", {})))
        candidates.append(mamba_cfg)
    else:
        candidates.append(_as_mapping(getattr(mamba_cfg, "interaction", {})))
        candidates.append(mamba_cfg)

    candidates.append(_as_mapping(getattr(cfg, "mamba_interaction", {})))
    candidates.append(_as_mapping(getattr(cfg, "interaction", {})))
    for container in candidates:
        if isinstance(container, dict) and name in container:
            return container[name]
        if hasattr(container, name):
            return getattr(container, name)

    legacy_name = f"mamba_interaction_{name}"
    if hasattr(cfg, legacy_name):
        return getattr(cfg, legacy_name)
    if hasattr(cfg, name):
        return getattr(cfg, name)
    return default


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class _GRUPriorEncoder(nn.Module):
    """Temporal GRU encoder that maps [B,T,C] to one sequence token."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float):
        super().__init__()
        layers = max(int(layers), 1)
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=float(dropout) if layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        _, h_n = self.gru(h)
        return self.out_proj(h_n[-1])


class _MeshPriorEncoder(nn.Module):
    """PointNet per-frame encoder plus GRU temporal aggregation."""

    def __init__(self, point_dim: int, aux_dim: int, hidden_dim: int, out_dim: int, layers: int, dropout: float):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.frame_fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=max(int(layers), 1),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, points_world: torch.Tensor, aux_seq: torch.Tensor) -> torch.Tensor:
        point_token = self.point_mlp(points_world).amax(dim=2)
        aux_token = self.aux_mlp(aux_seq)
        frame_token = self.frame_fuse(torch.cat((point_token, aux_token), dim=-1))
        _, h_n = self.temporal(frame_token)
        return self.out_proj(h_n[-1])


class InteractionModule(nn.Module):
    """Stage-2 Mamba interaction module.

    This combines the previous velocity/contact and object-translation stages
    and adds a residual human-pose refinement head conditioned on the predicted
    object state.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_human_imus = int(getattr(cfg, "num_human_imus", len(_SENSOR_NAMES)))
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = max(int(getattr(cfg, "obj_imu_dim", self.imu_dim)), 9)
        self.num_joints = int(getattr(cfg, "num_joints", 24))
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.hand_joint_indices = (20, 21)

        hidden_dim = int(_cfg_lookup(cfg, "hidden_dim", getattr(cfg, "human_pose_hidden", 256)))
        self.hidden_dim = hidden_dim
        dropout = float(_cfg_lookup(cfg, "dropout", 0.1))
        fast_layers = int(_cfg_lookup(cfg, "fast_layers", 3))
        slow_layers = int(_cfg_lookup(cfg, "slow_layers", 3))
        fast_kernel = int(_cfg_lookup(cfg, "fast_kernel", 3))
        slow_kernel = int(_cfg_lookup(cfg, "slow_kernel", 9))
        slow_dilation = int(_cfg_lookup(cfg, "slow_dilation", 2))
        slow_ma_window = int(_cfg_lookup(cfg, "slow_ma_window", 5))

        self.code_dim = int(_cfg_lookup(cfg, "object_code_dim", getattr(cfg, "object_code_dim", 128)))
        self.num_codes = int(_cfg_lookup(cfg, "num_object_codes", getattr(cfg, "num_object_codes", 128)))
        prior_hidden = int(_cfg_lookup(cfg, "prior_hidden_dim", getattr(cfg, "prior_encoder_hidden_dim", 256)))
        prior_layers = int(_cfg_lookup(cfg, "prior_layers", getattr(cfg, "prior_encoder_layers", 2)))
        prior_dropout = float(_cfg_lookup(cfg, "prior_dropout", getattr(cfg, "prior_encoder_dropout", 0.1)))
        self.mesh_downsample_points = int(_cfg_lookup(cfg, "mesh_downsample_points", getattr(cfg, "mesh_downsample_points", 256)))
        self.vq_commit_beta = float(_cfg_lookup(cfg, "vq_commit_beta", getattr(cfg, "vq_commit_beta", 0.25)))

        mode_probs_cfg = _cfg_lookup(cfg, "cond_mode_probs", getattr(cfg, "cond_mode_probs", [0.4, 0.4, 0.2]))
        if not isinstance(mode_probs_cfg, (list, tuple)) or len(mode_probs_cfg) != 3:
            mode_probs_cfg = [0.4, 0.4, 0.2]
        cond_mode_probs = torch.tensor(mode_probs_cfg, dtype=torch.float32)
        if float(cond_mode_probs.sum().item()) <= 0.0:
            cond_mode_probs = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)
        self.register_buffer("cond_mode_probs", cond_mode_probs / cond_mode_probs.sum().clamp_min(1e-8), persistent=False)

        self.gating_prior_beta = float(_cfg_lookup(cfg, "gating_prior_beta", 5.0))
        self.gating_temperature = float(_cfg_lookup(cfg, "gating_temperature", 5.0))
        self.gating_smoothing_enabled = bool(_cfg_lookup(cfg, "gating_smoothing_enabled", False))
        self.gating_smoothing_alpha = float(_cfg_lookup(cfg, "gating_smoothing_alpha", 0.6))
        self.gating_max_change = float(_cfg_lookup(cfg, "gating_max_change", 0.25))
        self.vel_static_threshold = float(_cfg_lookup(cfg, "vel_static_threshold", 0.3))
        self.vel_min_hand_speed = float(_cfg_lookup(cfg, "vel_min_hand_speed", 0.02))
        self.refine_gt_epochs = int(_cfg_lookup(cfg, "refine_gt_epochs", 50))
        self.current_epoch = 0

        input_dim = self.num_human_imus * self.imu_dim + self.obj_imu_dim + self.num_joints * 6 + 6 + 3 + 3
        self.stem = InputStem(input_dim, hidden_dim, dropout=dropout, conv_kernel=3)
        self.fast_encoder = TemporalMambaEncoder(hidden_dim, fast_layers, conv_kernel=fast_kernel, dropout=dropout, dilation=1)
        self.slow_filter = CausalMovingAverage(slow_ma_window)
        self.slow_encoder = TemporalMambaEncoder(
            hidden_dim,
            slow_layers,
            conv_kernel=slow_kernel,
            dropout=dropout,
            dilation=slow_dilation,
        )
        self.cross_scale_fusion = CrossScaleFusion(hidden_dim, dropout=dropout, use_fusion_block=True)

        obs_dim = self.num_joints * 6 + self.obj_imu_dim + 6
        mesh_aux_dim = self.num_joints * 6 + 6
        self.obs_encoder = _GRUPriorEncoder(obs_dim, prior_hidden, self.code_dim, prior_layers, prior_dropout)
        self.mesh_encoder = _MeshPriorEncoder(3, mesh_aux_dim, prior_hidden, self.code_dim, prior_layers, prior_dropout)
        self.object_codebook = nn.Embedding(self.num_codes, self.code_dim)
        nn.init.normal_(self.object_codebook.weight, mean=0.0, std=0.02)
        self.null_object_code = nn.Parameter(torch.zeros(1, self.code_dim))
        self.code_proj = nn.Linear(self.code_dim, hidden_dim)
        self.code_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cond_norm = RMSNorm(hidden_dim)

        self.hand_vel_head = _make_mlp(hidden_dim, hidden_dim, 6, dropout)
        self.obj_vel_head = _make_mlp(hidden_dim, hidden_dim, 3, dropout)
        self.foot_contact_head = nn.Linear(hidden_dim, 2)
        self.hand_contact_head = nn.Linear(hidden_dim, 2)
        self.obj_move_head = nn.Linear(hidden_dim, 1)

        fk_input_dim = hidden_dim + 6 + 3 + 1 + self.obj_imu_dim
        self.lhand_fk_head = _make_mlp(fk_input_dim, hidden_dim, 4, dropout)
        self.rhand_fk_head = _make_mlp(fk_input_dim, hidden_dim, 4, dropout)
        self.gating_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 3))

        refine_input_dim = hidden_dim + 3 + 6 + 2 + 3
        self.refine_proj = nn.Sequential(nn.Linear(refine_input_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout))
        self.refine_block = MambaBlock(hidden_dim, conv_kernel=3, dropout=dropout)
        self.refine_norm = RMSNorm(hidden_dim)
        self.pose_refine_head = nn.Linear(hidden_dim, len(_REDUCED_POSE_NAMES) * 6)
        self.root_refine_head = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.pose_refine_head.weight)
        nn.init.zeros_(self.pose_refine_head.bias)
        nn.init.zeros_(self.root_refine_head.weight)
        nn.init.zeros_(self.root_refine_head.bias)

        body_model_path = getattr(cfg, "body_model_path", None)
        if body_model_path is None:
            raise ValueError("body_model_path is not set")
        self.body_model = load_body_model(body_model_path, num_betas=16)
        self.body_model_device = torch.device("cpu")

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    @staticmethod
    def _unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _positive(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-4

    @staticmethod
    def _to_bt(value, batch_size: int, seq_len: int, trailing_shape, device, dtype, default: float = 0.0):
        shape = (batch_size, seq_len) if trailing_shape == () else (batch_size, seq_len, *trailing_shape)
        if not isinstance(value, torch.Tensor):
            return torch.full(shape, float(default), device=device, dtype=dtype)
        out = value.to(device=device, dtype=dtype)
        if out.dim() == len(shape) - 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, *out.shape[1:])
        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return torch.full(shape, float(default), device=device, dtype=dtype)
        if trailing_shape != () and tuple(out.shape[2:]) != tuple(trailing_shape):
            return torch.full(shape, float(default), device=device, dtype=dtype)
        return out

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
            mask = mask.reshape(batch_size, -1)
            mask = mask[:, :1].expand(batch_size, seq_len)
        return mask

    def _prepare_obj_imu(self, obj_imu, batch_size: int, seq_len: int, device, dtype) -> torch.Tensor:
        if not isinstance(obj_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        out = obj_imu.to(device=device, dtype=dtype)
        if out.dim() == 4:
            out = out.reshape(batch_size, seq_len, -1)
        if out.dim() != 3:
            return torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, -1, -1)
        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        if out.shape[-1] < self.obj_imu_dim:
            pad = torch.zeros(batch_size, seq_len, self.obj_imu_dim - out.shape[-1], device=device, dtype=dtype)
            out = torch.cat((out, pad), dim=-1)
        elif out.shape[-1] > self.obj_imu_dim:
            out = out[..., : self.obj_imu_dim]
        return out

    def _resolve_full_human_pose(self, hp_out: Optional[Dict], gt_targets: Optional[Dict], batch_size, seq_len, device, dtype):
        pose = hp_out.get("pred_full_pose_6d") if isinstance(hp_out, dict) else None
        if isinstance(pose, torch.Tensor):
            pose = pose.to(device=device, dtype=dtype)
            if pose.dim() == 3 and pose.shape[-1] == self.num_joints * 6:
                pose = pose.view(batch_size, seq_len, self.num_joints, 6)
            if pose.dim() == 4 and pose.shape[0] == batch_size and pose.shape[1] == seq_len:
                if pose.shape[2] > self.num_joints:
                    pose = pose[:, :, : self.num_joints]
                elif pose.shape[2] < self.num_joints:
                    pad = torch.zeros(batch_size, seq_len, self.num_joints - pose.shape[2], 6, device=device, dtype=dtype)
                    pose = torch.cat((pose, pad), dim=2)
                return pose

        rot_gt = gt_targets.get("rotation_global") if isinstance(gt_targets, dict) else None
        if isinstance(rot_gt, torch.Tensor):
            rot_gt = rot_gt.to(device=device, dtype=dtype)
            if rot_gt.dim() == 4:
                rot_gt = rot_gt.unsqueeze(0)
            if rot_gt.shape[0] == 1 and batch_size > 1:
                rot_gt = rot_gt.expand(batch_size, -1, -1, -1, -1)
            if rot_gt.shape[0] == batch_size and rot_gt.shape[1] == seq_len and rot_gt.shape[-2:] == (3, 3):
                if rot_gt.shape[2] > self.num_joints:
                    rot_gt = rot_gt[:, :, : self.num_joints]
                elif rot_gt.shape[2] < self.num_joints:
                    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
                    pad = eye.expand(batch_size, seq_len, self.num_joints - rot_gt.shape[2], -1, -1)
                    rot_gt = torch.cat((rot_gt, pad), dim=2)
                return matrix_to_rotation_6d(rot_gt.reshape(-1, 3, 3)).reshape(batch_size, seq_len, self.num_joints, 6)

        return torch.zeros(batch_size, seq_len, self.num_joints, 6, device=device, dtype=dtype)

    def _resolve_reduced_pose(self, hp_out: Optional[Dict], gt_targets: Optional[Dict], batch_size, seq_len, device, dtype):
        pose = hp_out.get("p_pred") if isinstance(hp_out, dict) else None
        if isinstance(pose, torch.Tensor):
            pose = pose.to(device=device, dtype=dtype)
            if pose.dim() == 4:
                pose = pose.reshape(batch_size, seq_len, -1)
            if pose.dim() == 3 and pose.shape[:2] == (batch_size, seq_len) and pose.shape[-1] == len(_REDUCED_POSE_NAMES) * 6:
                return pose

        reduced = gt_targets.get("ori_root_reduced") if isinstance(gt_targets, dict) else None
        if isinstance(reduced, torch.Tensor):
            reduced = reduced.to(device=device, dtype=dtype)
            if reduced.dim() == 4:
                reduced = reduced.unsqueeze(0)
            if reduced.shape[0] == 1 and batch_size > 1:
                reduced = reduced.expand(batch_size, -1, -1, -1, -1)
            if reduced.shape[:3] == (batch_size, seq_len, len(_REDUCED_POSE_NAMES)) and reduced.shape[-2:] == (3, 3):
                return matrix_to_rotation_6d(reduced.reshape(-1, 3, 3)).reshape(batch_size, seq_len, -1)
        return torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6, device=device, dtype=dtype)

    def _resolve_hand_pos(self, hp_out: Optional[Dict], gt_targets: Optional[Dict], batch_size, seq_len, device, dtype):
        hand_pos = hp_out.get("pred_hand_glb_pos") if isinstance(hp_out, dict) else None
        if isinstance(hand_pos, torch.Tensor):
            hand_pos = hand_pos.to(device=device, dtype=dtype)
            if hand_pos.dim() == 3 and hand_pos.shape[-1] == 6:
                hand_pos = hand_pos.view(batch_size, seq_len, 2, 3)
            if hand_pos.dim() == 4 and hand_pos.shape[:2] == (batch_size, seq_len) and hand_pos.shape[2:] == (2, 3):
                return hand_pos

        position_global = gt_targets.get("position_global") if isinstance(gt_targets, dict) else None
        if isinstance(position_global, torch.Tensor):
            position_global = position_global.to(device=device, dtype=dtype)
            if position_global.dim() == 3:
                position_global = position_global.unsqueeze(0)
            if position_global.shape[0] == 1 and batch_size > 1:
                position_global = position_global.expand(batch_size, -1, -1, -1)
            if position_global.shape[0] == batch_size and position_global.shape[1] == seq_len and position_global.shape[2] > 21:
                return torch.stack((position_global[:, :, 20], position_global[:, :, 21]), dim=2)
        return torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)

    def _assemble_features(self, data_dict: Dict, hp_out: Optional[Dict], gt_targets: Optional[Dict]) -> Dict[str, torch.Tensor]:
        human_imu = data_dict["human_imu"]
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")
        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        if human_imu.shape[2] > self.num_human_imus:
            human_imu = human_imu[:, :, : self.num_human_imus]
        elif human_imu.shape[2] < self.num_human_imus:
            pad = torch.zeros(batch_size, seq_len, self.num_human_imus - human_imu.shape[2], human_imu.shape[3], device=device, dtype=dtype)
            human_imu = torch.cat((human_imu, pad), dim=2)
        if human_imu.shape[3] > self.imu_dim:
            human_imu = human_imu[..., : self.imu_dim]
        elif human_imu.shape[3] < self.imu_dim:
            pad = torch.zeros(batch_size, seq_len, self.num_human_imus, self.imu_dim - human_imu.shape[3], device=device, dtype=dtype)
            human_imu = torch.cat((human_imu, pad), dim=-1)

        obj_imu = self._prepare_obj_imu(data_dict.get("obj_imu"), batch_size, seq_len, device, dtype)
        human_pose = self._resolve_full_human_pose(hp_out, gt_targets, batch_size, seq_len, device, dtype)
        p_pred_base = self._resolve_reduced_pose(hp_out, gt_targets, batch_size, seq_len, device, dtype)
        hand_pos = self._resolve_hand_pos(hp_out, gt_targets, batch_size, seq_len, device, dtype)

        root_vel = hp_out.get("root_vel_pred") if isinstance(hp_out, dict) else None
        if not isinstance(root_vel, torch.Tensor):
            root_vel = gt_targets.get("root_vel") if isinstance(gt_targets, dict) else None
        root_vel = self._to_bt(root_vel, batch_size, seq_len, (3,), device, dtype)

        root_trans = hp_out.get("root_trans_pred") if isinstance(hp_out, dict) else None
        if not isinstance(root_trans, torch.Tensor):
            root_trans = data_dict.get("trans_gt", gt_targets.get("trans") if isinstance(gt_targets, dict) else None)
        root_trans = self._to_bt(root_trans, batch_size, seq_len, (3,), device, dtype)

        trans_init = data_dict.get("trans_init")
        if isinstance(trans_init, torch.Tensor):
            trans_init = trans_init.to(device=device, dtype=dtype)
            if trans_init.dim() == 1:
                trans_init = trans_init.unsqueeze(0)
            if trans_init.shape[0] == 1 and batch_size > 1:
                trans_init = trans_init.expand(batch_size, -1)
            if trans_init.shape[0] != batch_size or trans_init.shape[-1] != 3:
                trans_init = root_trans[:, 0]
        else:
            trans_init = root_trans[:, 0]

        obj_trans_init = data_dict.get("obj_trans_init")
        if isinstance(obj_trans_init, torch.Tensor):
            obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
            if obj_trans_init.dim() == 3 and obj_trans_init.shape[1] == 1:
                obj_trans_init = obj_trans_init[:, 0]
            if obj_trans_init.dim() == 1:
                obj_trans_init = obj_trans_init.unsqueeze(0)
            if obj_trans_init.shape[0] == 1 and batch_size > 1:
                obj_trans_init = obj_trans_init.expand(batch_size, -1)
            if obj_trans_init.shape[0] != batch_size or obj_trans_init.shape[-1] != 3:
                obj_trans_init = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        else:
            obj_trans_init = torch.zeros(batch_size, 3, device=device, dtype=dtype)

        has_object_mask = self._prepare_has_object_mask(data_dict.get("has_object"), batch_size, seq_len, device)
        input_feat = torch.cat(
            (
                human_imu.reshape(batch_size, seq_len, -1),
                obj_imu,
                human_pose.reshape(batch_size, seq_len, -1),
                hand_pos.reshape(batch_size, seq_len, -1),
                root_vel,
                root_trans,
            ),
            dim=-1,
        )
        return {
            "input_feat": input_feat,
            "human_imu": human_imu,
            "obj_imu": obj_imu,
            "human_pose": human_pose.reshape(batch_size, seq_len, -1),
            "p_pred_base": p_pred_base,
            "hand_pos": hand_pos,
            "root_vel": root_vel,
            "root_trans": root_trans,
            "trans_init": trans_init,
            "obj_trans_init": obj_trans_init,
            "has_object_mask": has_object_mask,
        }

    def _quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        codebook = self.object_codebook.weight.to(device=z_e.device, dtype=z_e.dtype)
        dist = z_e.pow(2).sum(dim=1, keepdim=True) - 2.0 * z_e @ codebook.t() + codebook.pow(2).sum(dim=1).view(1, -1)
        code_idx = dist.argmin(dim=1)
        z_q = F.embedding(code_idx, codebook)
        logits = -dist
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q, z_q_st, code_idx, logits

    def _encode_observation_prior(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_seq = torch.cat((feats["human_pose"], feats["obj_imu"], feats["hand_pos"].reshape(feats["hand_pos"].shape[0], feats["hand_pos"].shape[1], -1)), dim=-1)
        z_e_obs = self.obs_encoder(obs_seq)
        z_q_obs, z_q_obs_st, code_idx_obs, logits_obs = self._quantize(z_e_obs)
        return {
            "z_e_obs": z_e_obs,
            "z_q_obs": z_q_obs,
            "z_q_obs_st": z_q_obs_st,
            "code_idx_obs": code_idx_obs,
            "code_logits_obs": logits_obs,
        }

    def _resolve_obj_rot_mats(self, obj_rot, batch_size: int, seq_len: int, device, dtype):
        if not isinstance(obj_rot, torch.Tensor):
            return None
        rot = obj_rot.to(device=device, dtype=dtype)
        if rot.dim() == 2 and rot.shape[-1] == 6:
            rot = rot.unsqueeze(0)
        if rot.dim() == 3 and rot.shape[-1] == 6:
            if rot.shape[0] == 1 and batch_size > 1:
                rot = rot.expand(batch_size, -1, -1)
            if rot.shape[:2] == (batch_size, seq_len):
                return rotation_6d_to_matrix(rot.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
        if rot.dim() == 3 and rot.shape[-2:] == (3, 3):
            rot = rot.unsqueeze(0)
        if rot.dim() == 4 and rot.shape[-2:] == (3, 3):
            if rot.shape[0] == 1 and batch_size > 1:
                rot = rot.expand(batch_size, -1, -1, -1)
            if rot.shape[:2] == (batch_size, seq_len):
                return rot
        return None

    def _encode_mesh_prior(self, data_dict: Dict, gt_targets: Optional[Dict], feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = feats["obj_imu"].shape[:2]
        device = feats["obj_imu"].device
        dtype = feats["obj_imu"].dtype
        z_e_mesh = torch.zeros(batch_size, self.code_dim, device=device, dtype=dtype)
        z_q_mesh = torch.zeros_like(z_e_mesh)
        z_q_mesh_st = torch.zeros_like(z_e_mesh)
        code_idx_mesh = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        logits_mesh = torch.zeros(batch_size, self.num_codes, device=device, dtype=dtype)
        valid_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

        if not isinstance(gt_targets, dict):
            return {
                "z_e_mesh": z_e_mesh,
                "z_q_mesh": z_q_mesh,
                "z_q_mesh_st": z_q_mesh_st,
                "code_idx_mesh": code_idx_mesh,
                "code_logits_mesh": logits_mesh,
                "mesh_valid_mask": valid_mask,
            }

        obj_points = data_dict.get("obj_points_canonical")
        if not isinstance(obj_points, torch.Tensor):
            return {
                "z_e_mesh": z_e_mesh,
                "z_q_mesh": z_q_mesh,
                "z_q_mesh_st": z_q_mesh_st,
                "code_idx_mesh": code_idx_mesh,
                "code_logits_mesh": logits_mesh,
                "mesh_valid_mask": valid_mask,
            }
        points = obj_points.to(device=device, dtype=dtype)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if points.shape[0] == 1 and batch_size > 1:
            points = points.expand(batch_size, -1, -1)
        if points.dim() != 3 or points.shape[0] != batch_size or points.shape[-1] != 3:
            return {
                "z_e_mesh": z_e_mesh,
                "z_q_mesh": z_q_mesh,
                "z_q_mesh_st": z_q_mesh_st,
                "code_idx_mesh": code_idx_mesh,
                "code_logits_mesh": logits_mesh,
                "mesh_valid_mask": valid_mask,
            }
        if points.shape[1] > self.mesh_downsample_points:
            idx = torch.linspace(0, points.shape[1] - 1, steps=self.mesh_downsample_points, device=device).long()
            points = points[:, idx]

        obj_rot_mats = self._resolve_obj_rot_mats(gt_targets.get("obj_rot"), batch_size, seq_len, device, dtype)
        if obj_rot_mats is None:
            obj_rot_mats = rotation_6d_to_matrix(feats["obj_imu"][..., 3:9].reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)

        obj_trans = self._to_bt(gt_targets.get("obj_trans"), batch_size, seq_len, (3,), device, dtype)
        obj_scale = self._to_bt(gt_targets.get("obj_scale"), batch_size, seq_len, (), device, dtype, default=1.0)
        has_object = feats["has_object_mask"].any(dim=1)
        nonzero_points = points.abs().sum(dim=(1, 2)) > 1e-8
        valid_mask = has_object & nonzero_points
        if not valid_mask.any():
            return {
                "z_e_mesh": z_e_mesh,
                "z_q_mesh": z_q_mesh,
                "z_q_mesh_st": z_q_mesh_st,
                "code_idx_mesh": code_idx_mesh,
                "code_logits_mesh": logits_mesh,
                "mesh_valid_mask": valid_mask,
            }

        valid_idx = torch.nonzero(valid_mask, as_tuple=False).flatten()
        points_valid = points[valid_idx]
        rot_valid = obj_rot_mats[valid_idx]
        trans_valid = obj_trans[valid_idx]
        scale_valid = obj_scale[valid_idx]
        point_seq = points_valid[:, None].expand(-1, seq_len, -1, -1) * scale_valid[:, :, None, None]
        points_world = torch.matmul(rot_valid[:, :, None], point_seq.unsqueeze(-1)).squeeze(-1) + trans_valid[:, :, None]
        aux_seq = torch.cat((feats["human_pose"][valid_idx], feats["hand_pos"][valid_idx].reshape(valid_idx.numel(), seq_len, -1)), dim=-1)
        z_e_valid = self.mesh_encoder(points_world, aux_seq)
        z_q_valid, z_q_st_valid, code_idx_valid, logits_valid = self._quantize(z_e_valid)

        z_e_mesh[valid_idx] = z_e_valid.to(dtype=z_e_mesh.dtype)
        z_q_mesh[valid_idx] = z_q_valid.to(dtype=z_q_mesh.dtype)
        z_q_mesh_st[valid_idx] = z_q_st_valid.to(dtype=z_q_mesh_st.dtype)
        code_idx_mesh[valid_idx] = code_idx_valid
        logits_mesh[valid_idx] = logits_valid.to(dtype=logits_mesh.dtype)
        return {
            "z_e_mesh": z_e_mesh,
            "z_q_mesh": z_q_mesh,
            "z_q_mesh_st": z_q_mesh_st,
            "code_idx_mesh": code_idx_mesh,
            "code_logits_mesh": logits_mesh,
            "mesh_valid_mask": valid_mask,
        }

    def _build_object_condition(self, data_dict: Dict, gt_targets: Optional[Dict], feats: Dict[str, torch.Tensor], training: bool):
        batch_size = feats["obj_imu"].shape[0]
        device = feats["obj_imu"].device
        dtype = feats["obj_imu"].dtype
        sample_has_object = feats["has_object_mask"].any(dim=1)

        obs = self._encode_observation_prior(feats)
        mesh = {
            "z_e_mesh": torch.zeros(batch_size, self.code_dim, device=device, dtype=dtype),
            "z_q_mesh": torch.zeros(batch_size, self.code_dim, device=device, dtype=dtype),
            "z_q_mesh_st": torch.zeros(batch_size, self.code_dim, device=device, dtype=dtype),
            "code_idx_mesh": torch.full((batch_size,), -1, device=device, dtype=torch.long),
            "code_logits_mesh": torch.zeros(batch_size, self.num_codes, device=device, dtype=dtype),
            "mesh_valid_mask": torch.zeros(batch_size, device=device, dtype=torch.bool),
        }
        if training:
            mesh = self._encode_mesh_prior(data_dict, gt_targets, feats)

        null_code = self.null_object_code.to(device=device, dtype=dtype).expand(batch_size, -1)
        if training:
            mode = torch.multinomial(self.cond_mode_probs.to(device=device), batch_size, replacement=True)
            mode = torch.where((mode == 0) & (~mesh["mesh_valid_mask"]), torch.ones_like(mode), mode)
            mode = torch.where(sample_has_object, mode, torch.full_like(mode, 2))
            cond = obs["z_q_obs_st"].clone()
            cond = torch.where(mode[:, None] == 0, mesh["z_q_mesh_st"], cond)
            cond = torch.where(mode[:, None] == 2, null_code, cond)
        else:
            mode = torch.full((batch_size,), -1, device=device, dtype=torch.long)
            cond = torch.where(sample_has_object[:, None], obs["z_q_obs_st"], null_code)

        code_idx_obs = torch.where(sample_has_object, obs["code_idx_obs"], torch.full_like(obs["code_idx_obs"], -1))
        aux = {
            **mesh,
            **obs,
            "code_idx_obs": code_idx_obs,
            "mode": mode,
            "cond": cond,
            "sample_has_object": sample_has_object,
            "vq_beta": torch.tensor(self.vq_commit_beta, device=device, dtype=dtype),
        }
        return cond, aux

    def _inject_object_condition(self, ctx: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        code_token = self.code_proj(cond).unsqueeze(1).expand(-1, ctx.shape[1], -1)
        gate = torch.sigmoid(self.code_gate(torch.cat((ctx, code_token), dim=-1)))
        return self.cond_norm(ctx + gate * code_token)

    def _encode_context(self, feats: Dict[str, torch.Tensor], cond: torch.Tensor) -> torch.Tensor:
        x = self.stem(feats["input_feat"])
        fast = self.fast_encoder(x)
        slow = self.slow_encoder(self.slow_filter(x))
        ctx = self.cross_scale_fusion(fast, slow)
        return self._inject_object_condition(ctx, cond)

    def _decode_contact_velocity(self, ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = ctx.shape[:2]
        hand_vel = self.hand_vel_head(ctx).reshape(batch_size, seq_len, 2, 3)
        obj_vel = self.obj_vel_head(ctx)
        foot_logits = self.foot_contact_head(ctx)
        hand_logits = self.hand_contact_head(ctx)
        obj_move_logits = self.obj_move_head(ctx)
        obj_move_prob = torch.sigmoid(obj_move_logits)
        hand_prob_cond = torch.sigmoid(hand_logits)
        hand_prob = hand_prob_cond * obj_move_prob
        contact_logits = torch.cat((hand_logits, obj_move_logits), dim=-1)
        contact_prob = torch.cat((hand_prob, obj_move_prob), dim=-1)
        return {
            "pred_hand_glb_vel": hand_vel,
            "pred_obj_vel": obj_vel,
            "pred_foot_contact_logits": foot_logits,
            "pred_foot_contact_prob": torch.sigmoid(foot_logits),
            "pred_interaction_boundary_logits": foot_logits,
            "pred_interaction_boundary_prob": torch.sigmoid(foot_logits),
            "pred_hand_contact_logits": contact_logits,
            "pred_hand_contact_prob": contact_prob,
            "pred_obj_move_logits": obj_move_logits,
            "pred_obj_move_prob": obj_move_prob,
            "pred_hand_contact_logits_cond": hand_logits,
            "pred_hand_contact_prob_cond": hand_prob_cond,
            "contact_prob_pred": hand_prob,
        }

    def _compute_hand_velocity(self, hand_pos: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(hand_pos)
        if hand_pos.shape[1] > 1:
            vel[:, 1:] = (hand_pos[:, 1:] - hand_pos[:, :-1]) * self.fps
            vel[:, 0] = vel[:, 1]
        return vel

    def _correct_obj_velocity(self, v_obj, v_lhand, v_rhand, p_left, p_right, p_move):
        static_factor = torch.clamp(p_move / max(self.vel_static_threshold, 1e-6), 0.0, 1.0)

        def direction_correct(v_base, v_hand, p_contact):
            speed = v_hand.norm(dim=-1, keepdim=True)
            moving = (speed > self.vel_min_hand_speed).to(dtype=v_base.dtype)
            hand_dir = v_hand / speed.clamp_min(1e-6)
            proj = (v_base * hand_dir).sum(dim=-1, keepdim=True)
            corrected = v_base - proj * hand_dir + v_hand
            weight = p_contact * moving
            return v_base * (1.0 - weight) + corrected * weight

        v_lcorr = direction_correct(v_obj, v_lhand, p_left)
        v_rcorr = direction_correct(v_obj, v_rhand, p_right)
        denom = p_left + p_right + 1e-6
        contact_corrected = (p_left / denom) * v_lcorr + (p_right / denom) * v_rcorr
        return static_factor * contact_corrected

    def _smooth_gating_weights(self, weights: torch.Tensor) -> torch.Tensor:
        if self.training or (not self.gating_smoothing_enabled) or weights.shape[1] < 2:
            return weights
        smoothed = weights.clone()
        prev = weights[:, 0]
        for t in range(1, weights.shape[1]):
            frame = weights[:, t].clone()
            changed = prev.argmax(dim=-1) != frame.argmax(dim=-1)
            if changed.any():
                frame[changed] = self.gating_smoothing_alpha * prev[changed] + (1.0 - self.gating_smoothing_alpha) * frame[changed]
                if self.gating_max_change > 0:
                    diff = frame[changed] - prev[changed]
                    norm = diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    scale = (self.gating_max_change / norm).clamp(max=1.0)
                    frame[changed] = prev[changed] + diff * scale
                frame[changed] = F.softmax(torch.log(frame[changed] + 1e-8) * self.gating_temperature, dim=-1)
            smoothed[:, t] = frame
            prev = frame
        return smoothed

    def _compute_init_dir_len(self, hand_pos_0, obj_rotm_0, obj_pos_0):
        vec_world = obj_pos_0 - hand_pos_0
        length = vec_world.norm(dim=-1, keepdim=True)
        unit_world = self._unit_vector(vec_world)
        local_dir = torch.bmm(obj_rotm_0.transpose(-1, -2), unit_world.unsqueeze(-1)).squeeze(-1)
        return local_dir, length.squeeze(-1)

    def _decode_object_trans(self, ctx: torch.Tensor, feats: Dict[str, torch.Tensor], contact_out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = ctx.shape[:2]
        device = ctx.device
        dtype = ctx.dtype
        hand_pos = feats["hand_pos"]
        lhand_pos = hand_pos[:, :, 0]
        rhand_pos = hand_pos[:, :, 1]
        obj_imu = feats["obj_imu"]
        obj_rot6d = obj_imu[..., 3:9]
        obj_rotm = rotation_6d_to_matrix(obj_rot6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)

        contact_prob = contact_out["pred_hand_contact_prob"]
        p_left = contact_prob[..., 0:1]
        p_right = contact_prob[..., 1:2]
        p_move = contact_prob[..., 2:3]
        obj_vel = contact_out["pred_obj_vel"]

        l_fk_in = torch.cat((ctx, obj_rot6d, lhand_pos, p_left, obj_imu), dim=-1)
        r_fk_in = torch.cat((ctx, obj_rot6d, rhand_pos, p_right, obj_imu), dim=-1)
        l_fk = self.lhand_fk_head(l_fk_in)
        r_fk = self.rhand_fk_head(r_fk_in)
        l_dir = self._unit_vector(l_fk[..., :3])
        r_dir = self._unit_vector(r_fk[..., :3])
        l_len = self._positive(l_fk[..., 3])
        r_len = self._positive(r_fk[..., 3])

        obj_rotm_flat = obj_rotm.reshape(batch_size * seq_len, 3, 3)
        l_world = torch.bmm(obj_rotm_flat, l_dir.reshape(batch_size * seq_len, 3, 1)).reshape(batch_size, seq_len, 3)
        r_world = torch.bmm(obj_rotm_flat, r_dir.reshape(batch_size * seq_len, 3, 1)).reshape(batch_size, seq_len, 3)
        l_pos_fk = lhand_pos + l_world * l_len.unsqueeze(-1)
        r_pos_fk = rhand_pos + r_world * r_len.unsqueeze(-1)

        lhand_vel = self._compute_hand_velocity(lhand_pos)
        rhand_vel = self._compute_hand_velocity(rhand_pos)
        obj_vel_corrected = self._correct_obj_velocity(obj_vel, lhand_vel, rhand_vel, p_left, p_right, p_move)

        gate_logits = self.gating_head(ctx)
        prior_imu = 1.0 - p_move.squeeze(-1)
        prior = torch.stack((p_left.squeeze(-1), p_right.squeeze(-1), prior_imu), dim=-1)
        gate_logits = gate_logits + self.gating_prior_beta * torch.log(prior + 1e-6)
        weights_raw = F.softmax(gate_logits / max(self.gating_temperature, 1e-6), dim=-1)
        weights = self._smooth_gating_weights(weights_raw)

        obj_trans_init = feats["obj_trans_init"]
        fused_pos = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        dt = 1.0 / self.fps
        for t in range(seq_len):
            prev = fused_pos[:, t - 1] if t > 0 else obj_trans_init
            pos_imu = prev + obj_vel_corrected[:, t] * dt
            fused_pos[:, t] = (
                weights[:, t, 0:1] * l_pos_fk[:, t]
                + weights[:, t, 1:2] * r_pos_fk[:, t]
                + weights[:, t, 2:3] * pos_imu
            )

        vel_from_pos = torch.zeros_like(fused_pos)
        acc_from_pos = torch.zeros_like(fused_pos)
        if seq_len > 1:
            vel_from_pos[:, 1:] = (fused_pos[:, 1:] - fused_pos[:, :-1]) * self.fps
            vel_from_pos[:, 0] = vel_from_pos[:, 1]
        if seq_len > 2:
            acc_from_pos[:, 2:] = (fused_pos[:, 2:] - 2.0 * fused_pos[:, 1:-1] + fused_pos[:, :-2]) * (self.fps ** 2)

        mask = feats["has_object_mask"].to(device=device, dtype=dtype).unsqueeze(-1)
        fused_pos = fused_pos * mask
        vel_from_pos = vel_from_pos * mask
        acc_from_pos = acc_from_pos * mask
        weights = weights * mask
        weights_raw = weights_raw * mask
        l_pos_fk = l_pos_fk * mask
        r_pos_fk = r_pos_fk * mask
        l_dir = l_dir * mask
        r_dir = r_dir * mask
        l_len = l_len * mask.squeeze(-1)
        r_len = r_len * mask.squeeze(-1)

        l_oe0, l_lb0 = self._compute_init_dir_len(lhand_pos[:, 0], obj_rotm[:, 0], obj_trans_init)
        r_oe0, r_lb0 = self._compute_init_dir_len(rhand_pos[:, 0], obj_rotm[:, 0], obj_trans_init)
        sample_mask = feats["has_object_mask"].any(dim=1).to(device=device, dtype=dtype)
        l_oe0 = l_oe0 * sample_mask.unsqueeze(-1)
        r_oe0 = r_oe0 * sample_mask.unsqueeze(-1)
        l_lb0 = l_lb0 * sample_mask
        r_lb0 = r_lb0 * sample_mask

        return {
            "pred_obj_trans": fused_pos,
            "gating_weights": weights,
            "gating_weights_raw": weights_raw,
            "pred_obj_vel_from_posdiff": vel_from_pos,
            "pred_obj_acc_from_posdiff": acc_from_pos,
            "obj_vel_input": obj_vel,
            "obj_vel_corrected": obj_vel_corrected * mask,
            "bone_dir_pred": torch.stack((l_dir, r_dir), dim=2),
            "bone_len_pred": torch.stack((l_len, r_len), dim=2),
            "pred_lhand_obj_direction": l_dir,
            "pred_rhand_obj_direction": r_dir,
            "pred_lhand_lb": l_len,
            "pred_rhand_lb": r_len,
            "pred_lhand_obj_trans": l_pos_fk,
            "pred_rhand_obj_trans": r_pos_fk,
            "init_lhand_oe_ho": l_oe0,
            "init_rhand_oe_ho": r_oe0,
            "init_lhand_lb": l_lb0,
            "init_rhand_lb": r_lb0,
            "obj_trans_init": obj_trans_init,
            "has_object": feats["has_object_mask"],
            "gating_smoothing_applied": (not self.training) and self.gating_smoothing_enabled,
        }

    def _refine_obj_input(self, obj_out: Dict[str, torch.Tensor], gt_targets: Optional[Dict], feats: Dict[str, torch.Tensor]):
        pred_obj = obj_out["pred_obj_trans"].detach()
        if self.training and self.current_epoch < self.refine_gt_epochs and isinstance(gt_targets, dict):
            batch_size, seq_len = pred_obj.shape[:2]
            obj_gt = self._to_bt(gt_targets.get("obj_trans"), batch_size, seq_len, (3,), pred_obj.device, pred_obj.dtype)
            mask = feats["has_object_mask"].to(device=pred_obj.device, dtype=pred_obj.dtype).unsqueeze(-1)
            return obj_gt * mask + pred_obj * (1.0 - mask)
        return pred_obj

    def _compute_refined_pose_outputs(self, refined_pose: torch.Tensor, refined_root_vel: torch.Tensor, feats: Dict[str, torch.Tensor]):
        batch_size, seq_len = refined_pose.shape[:2]
        device = refined_pose.device
        dtype = refined_pose.dtype
        ensure_body_model_device(self, device)

        refined_part = refined_pose.reshape(batch_size, seq_len, len(_REDUCED_POSE_NAMES), 6)
        orientation_6d = feats["human_imu"][..., -6:]
        orientation_mat = rotation_6d_to_matrix(orientation_6d.reshape(-1, 6)).reshape(
            batch_size,
            seq_len,
            feats["human_imu"].shape[2],
            3,
            3,
        )
        full_flat = reduced_root_pose_to_full_global(
            refined_part.reshape(batch_size * seq_len, len(_REDUCED_POSE_NAMES), 6),
            orientation_mat.reshape(batch_size * seq_len, orientation_mat.shape[2], 3, 3),
            num_joints=self.num_joints,
        )
        full_rotmats = full_flat.reshape(batch_size, seq_len, self.num_joints, 3, 3)
        full_6d = matrix_to_rotation_6d(full_rotmats.reshape(-1, 3, 3)).reshape(batch_size, seq_len, self.num_joints, 6)
        joints_local = compute_smpl_joints_from_global(full_rotmats, self.body_model, num_joints=self.num_joints)
        if joints_local is None:
            joints_local = torch.zeros(batch_size, seq_len, self.num_joints, 3, device=device, dtype=dtype)

        trans_init = feats["trans_init"]
        refined_trans = integrate_root_velocity(refined_root_vel, self.fps, trans_init=trans_init)
        pred_joints_global = joints_local + refined_trans.unsqueeze(2)
        refined_hand_pos = torch.stack(
            (pred_joints_global[:, :, self.hand_joint_indices[0]], pred_joints_global[:, :, self.hand_joint_indices[1]]),
            dim=2,
        )
        return {
            "refined_pose": refined_pose,
            "refined_pose_part": refined_part,
            "refined_root_vel": refined_root_vel,
            "refined_trans": refined_trans,
            "refined_full_pose_rotmat": full_rotmats,
            "refined_full_pose_6d": full_6d,
            "refined_joints_local": joints_local,
            "refined_joints_global": pred_joints_global,
            "refined_hand_glb_pos": refined_hand_pos,
        }

    def _decode_pose_refinement(self, ctx: torch.Tensor, feats: Dict[str, torch.Tensor], contact_out: Dict[str, torch.Tensor], obj_out: Dict[str, torch.Tensor], gt_targets: Optional[Dict]):
        obj_for_refine = self._refine_obj_input(obj_out, gt_targets, feats)
        bone_dir = obj_out["bone_dir_pred"].reshape(ctx.shape[0], ctx.shape[1], -1)
        bone_len = obj_out["bone_len_pred"]
        contact_prob = contact_out["pred_hand_contact_prob"]
        refine_input = torch.cat((ctx, obj_for_refine, bone_dir, bone_len, contact_prob), dim=-1)
        h = self.refine_proj(refine_input)
        h = self.refine_norm(self.refine_block(h))
        pose_residual = self.pose_refine_head(h)
        root_residual = self.root_refine_head(h)
        refined_pose = feats["p_pred_base"] + pose_residual
        refined_root_vel = feats["root_vel"] + root_residual
        out = self._compute_refined_pose_outputs(refined_pose, refined_root_vel, feats)
        out["pose_residual"] = pose_residual
        out["root_vel_residual"] = root_residual
        out["refine_obj_input"] = obj_for_refine
        return out

    def forward(self, data_dict: Dict, hp_out: Optional[Dict] = None, gt_targets: Optional[Dict] = None):
        feats = self._assemble_features(data_dict, hp_out, gt_targets)
        cond, prior_aux = self._build_object_condition(data_dict, gt_targets, feats, training=bool(self.training))
        ctx = self._encode_context(feats, cond)
        contact_out = self._decode_contact_velocity(ctx)
        obj_out = self._decode_object_trans(ctx, feats, contact_out)
        refine_out = self._decode_pose_refinement(ctx, feats, contact_out, obj_out, gt_targets)

        outputs = {}
        outputs.update(contact_out)
        outputs.update(obj_out)
        outputs.update(refine_out)
        outputs["object_prior_aux"] = prior_aux
        outputs["ctx"] = ctx
        return outputs

    @torch.no_grad()
    def inference(self, data_dict: Dict, hp_out: Optional[Dict] = None, gt_targets: Optional[Dict] = None, **_):
        return self.forward(data_dict, hp_out=hp_out, gt_targets=gt_targets)

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device, num_joints: int = 24):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_dir = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_scalar = torch.zeros(batch_size, seq_len, device=device)
        return {
            "pred_hand_glb_vel": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "pred_obj_vel": zeros_pos.clone(),
            "pred_hand_contact_logits": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_hand_contact_prob": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_move_prob": torch.zeros(batch_size, seq_len, 1, device=device),
            "pred_foot_contact_logits": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_foot_contact_prob": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_interaction_boundary_logits": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_interaction_boundary_prob": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_obj_trans": zeros_pos.clone(),
            "pred_obj_vel_from_posdiff": zeros_pos.clone(),
            "pred_obj_acc_from_posdiff": zeros_pos.clone(),
            "pred_lhand_obj_direction": zeros_dir.clone(),
            "pred_rhand_obj_direction": zeros_dir.clone(),
            "pred_lhand_lb": zeros_scalar.clone(),
            "pred_rhand_lb": zeros_scalar.clone(),
            "refined_pose": torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6, device=device),
            "refined_trans": zeros_pos.clone(),
            "refined_joints_global": torch.zeros(batch_size, seq_len, num_joints, 3, device=device),
            "object_prior_aux": {},
        }
