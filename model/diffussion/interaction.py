"""
Stage-2 Interaction diffusion module.

Training:
- Full-window denoising only: x -> z_t -> x0_pred over all frames.
- No autoregressive rollout during training.

Inference:
- Autoregressive sliding-window inpainting.
- History frames are fixed.
- Current frame keeps observed dimensions fixed.
- Unknown dimensions are refined with configurable inpainting backend.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from .base import ConditionalDiT
from configs import FRAME_RATE, _SENSOR_NAMES


class _SequenceTransformerEncoder(nn.Module):
    """Transformer encoder that maps [B,T,C] to a single [B,D] token."""

    def __init__(
        self,
        *,
        in_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        out_dim: int,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(num_layers), 1))
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.out_proj(h)


class _MeshPriorEncoder(nn.Module):
    """PointNet frame encoder + temporal aggregator for mesh prior token."""

    def __init__(
        self,
        *,
        point_dim: int,
        aux_dim: int,
        d_model: int,
        out_dim: int,
        temporal_model: str = "gru",
        temporal_layers: int = 2,
        temporal_kernel_size: int = 3,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.aux_dim = int(max(aux_dim, 0))
        if self.aux_dim > 0:
            self.aux_mlp = nn.Sequential(
                nn.Linear(self.aux_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
            self.frame_fuse = nn.Linear(d_model * 2, d_model)
        else:
            self.aux_mlp = None
            self.frame_fuse = None
        self.frame_norm = nn.LayerNorm(d_model)

        self.temporal_model = str(temporal_model).lower()
        layers = max(int(temporal_layers), 1)
        if self.temporal_model == "gru":
            self.temporal_encoder = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=layers,
                batch_first=True,
                dropout=float(max(temporal_dropout, 0.0)) if layers > 1 else 0.0,
            )
        elif self.temporal_model == "tcn":
            kernel = int(max(1, temporal_kernel_size))
            if kernel % 2 == 0:
                kernel += 1
            tcn_blocks = []
            for _ in range(layers):
                tcn_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2),
                        nn.GELU(),
                        nn.Dropout(float(max(temporal_dropout, 0.0))),
                    )
                )
            self.temporal_encoder = nn.ModuleList(tcn_blocks)
        else:
            raise ValueError(f"Unsupported mesh temporal_model: {temporal_model}")
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, point_seq: torch.Tensor, aux_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        # point_seq: [B,T,N,3], aux_seq: [B,T,C_aux]
        frame_tokens = self.point_mlp(point_seq).amax(dim=2)
        if self.aux_mlp is not None:
            if aux_seq is None:
                raise ValueError("aux_seq must be provided when aux_dim > 0")
            if aux_seq.dim() != 3 or aux_seq.shape[:2] != frame_tokens.shape[:2]:
                raise ValueError(
                    f"aux_seq shape mismatch, got {aux_seq.shape}, expected [B={frame_tokens.shape[0]},T={frame_tokens.shape[1]},C]"
                )
            aux_tokens = self.aux_mlp(aux_seq)
            frame_tokens = self.frame_fuse(torch.cat([frame_tokens, aux_tokens], dim=-1))
        frame_tokens = self.frame_norm(frame_tokens)

        if self.temporal_model == "gru":
            _, h_n = self.temporal_encoder(frame_tokens)
            seq_token = h_n[-1]
        else:
            h = frame_tokens.transpose(1, 2)
            for block in self.temporal_encoder:
                h = h + block(h)
            temporal_tokens = h.transpose(1, 2)
            seq_token = temporal_tokens.mean(dim=1)
        return self.out_proj(seq_token)


class InteractionModule(nn.Module):
    """Diffusion Transformer for interaction reconstruction."""

    def __init__(self, cfg):
        super().__init__()

        self.num_joints = int(getattr(cfg, "num_joints", 24))
        self.num_human_imus = int(getattr(cfg, "num_human_imus", len(_SENSOR_NAMES)))
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(getattr(cfg, "obj_imu_dim", self.imu_dim))
        self.obj_imu_feat_dim = max(self.obj_imu_dim, 9)
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))

        self.hand_joint_indices = (20, 21)

        # x_frame = [R_{human+obj}, a_{human+obj}, p_hands, contact_prob, v_bone, l_bone, delta_p_obj]
        self.human_rot_dim = self.num_joints * 6
        self.obj_rot_dim = 6
        self.rot_dim = self.human_rot_dim + self.obj_rot_dim

        self.human_acc_dim = self.num_human_imus * 3
        self.obj_acc_dim = 3
        self.acc_dim = self.human_acc_dim + self.obj_acc_dim

        self.hands_dim = 6
        self.contact_dim = 2
        self.bone_dir_dim = 6
        self.bone_len_dim = 2
        self.delta_p_obj_dim = 3

        start = 0
        self.rot_slice = slice(start, start + self.rot_dim)
        start += self.rot_dim
        self.acc_slice = slice(start, start + self.acc_dim)
        start += self.acc_dim
        self.hands_slice = slice(start, start + self.hands_dim)
        start += self.hands_dim
        self.contact_slice = slice(start, start + self.contact_dim)
        start += self.contact_dim
        self.bone_dir_slice = slice(start, start + self.bone_dir_dim)
        start += self.bone_dir_dim
        self.bone_len_slice = slice(start, start + self.bone_len_dim)
        start += self.bone_len_dim
        self.delta_p_obj_slice = slice(start, start + self.delta_p_obj_dim)
        start += self.delta_p_obj_dim
        self.target_dim = start

        # Observed dims during inference: R_{human+obj}, a_{human+obj}, p_hands.
        observed_dim_mask = torch.zeros(self.target_dim, dtype=torch.bool)
        observed_dim_mask[self.rot_slice] = True
        observed_dim_mask[self.acc_slice] = True
        observed_dim_mask[self.hands_slice] = True
        unknown_dim_mask = ~observed_dim_mask
        self.register_buffer("observed_dim_mask", observed_dim_mask, persistent=False)
        self.register_buffer("unknown_dim_mask", unknown_dim_mask, persistent=False)

        train_cfg = getattr(cfg, "train", {})
        test_cfg = getattr(cfg, "test", {})
        train_window = train_cfg.get("window") if isinstance(train_cfg, dict) else getattr(train_cfg, "window", None)
        test_window = test_cfg.get("window") if isinstance(test_cfg, dict) else getattr(test_cfg, "window", None)
        self.window_size = int(test_window if test_window is not None else train_window)
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")

        dit_cfg = getattr(cfg, "dit", {})

        def _dit_param(name: str, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        self.use_diffusion_noise = bool(_dit_param("dit_use_noise", True))
        self.inference_steps = _dit_param("dit_inference_steps", None)
        self.inference_steps = int(self.inference_steps) if self.inference_steps is not None else None

        inference_type_raw = str(_dit_param("dit_inference_type", "sample_inpaint_x0")).lower()
        if inference_type_raw in {"sample_inpaint_x0", "inpaint_x0", "x0"}:
            self.inference_type = "sample_inpaint_x0"
        elif inference_type_raw in {"sample_inpaint", "inpaint", "diffusion"}:
            self.inference_type = "sample_inpaint"
        else:
            self.inference_type = "sample_inpaint_x0"

        self.inference_sampler = str(_dit_param("dit_inference_sampler", "ddim")).lower()
        if self.inference_sampler not in {"ddim", "ddpm"}:
            self.inference_sampler = "ddim"
        self.inference_eta = float(_dit_param("dit_inference_eta", 0.0))
        self.inference_eta = max(self.inference_eta, 0.0)

        self.test_use_gt_warmup = bool(_dit_param("dit_test_use_gt_warmup", True))
        test_warmup_len = _dit_param("dit_test_warmup_len", None)
        self.test_warmup_len = None if test_warmup_len is None else int(test_warmup_len)
        if self.test_warmup_len is not None and self.test_warmup_len < 0:
            raise ValueError(f"dit_test_warmup_len must be >= 0 or None, got {self.test_warmup_len}")

        prediction_type = str(_dit_param("dit_prediction_type", "x0")).lower()
        if prediction_type not in {"x0", "eps"}:
            prediction_type = "x0"

        # Object-prior condition settings.
        self.code_dim = int(getattr(cfg, "object_code_dim", 128))
        self.num_codes = int(getattr(cfg, "num_object_codes", 128))
        self.object_geo_root = str(getattr(cfg, "object_geo_root", "datasets/OMOMO/captured_objects"))
        self.mesh_downsample_points = int(max(1, getattr(cfg, "mesh_downsample_points", 256)))
        self.vq_commit_beta = float(getattr(cfg, "vq_commit_beta", 0.25))
        self.mesh_cond_dim = self.human_rot_dim + self.hands_dim
        self.obs_cond_dim = self.human_rot_dim + self.obj_imu_feat_dim + self.hands_dim

        prior_hidden = int(getattr(cfg, "prior_encoder_hidden_dim", 256))
        prior_heads = int(getattr(cfg, "prior_encoder_heads", 8))
        prior_layers = int(getattr(cfg, "prior_encoder_layers", 2))
        prior_dropout = float(getattr(cfg, "prior_encoder_dropout", 0.1))
        mesh_temporal_model = str(getattr(cfg, "mesh_temporal_model", "gru")).lower()
        mesh_temporal_layers = int(getattr(cfg, "mesh_temporal_layers", prior_layers))
        mesh_temporal_kernel_size = int(getattr(cfg, "mesh_temporal_kernel_size", 3))
        mesh_temporal_dropout = float(getattr(cfg, "mesh_temporal_dropout", prior_dropout))

        mode_probs_cfg = getattr(cfg, "cond_mode_probs", [0.4, 0.4, 0.2])
        if not isinstance(mode_probs_cfg, (list, tuple)) or len(mode_probs_cfg) != 3:
            mode_probs_cfg = [0.4, 0.4, 0.2]
        cond_mode_probs = torch.tensor(mode_probs_cfg, dtype=torch.float32)
        if float(cond_mode_probs.sum().item()) <= 0.0:
            cond_mode_probs = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)
        cond_mode_probs = cond_mode_probs / cond_mode_probs.sum().clamp_min(1e-8)
        self.register_buffer("cond_mode_probs", cond_mode_probs, persistent=False)

        self.mesh_encoder = _MeshPriorEncoder(
            point_dim=3,
            aux_dim=self.mesh_cond_dim,
            d_model=prior_hidden,
            out_dim=self.code_dim,
            temporal_model=mesh_temporal_model,
            temporal_layers=mesh_temporal_layers,
            temporal_kernel_size=mesh_temporal_kernel_size,
            temporal_dropout=mesh_temporal_dropout,
        )
        self.obs_encoder = _SequenceTransformerEncoder(
            in_dim=self.obs_cond_dim,
            d_model=prior_hidden,
            nhead=prior_heads,
            num_layers=prior_layers,
            dropout=prior_dropout,
            out_dim=self.code_dim,
        )

        self.object_codebook = nn.Embedding(self.num_codes, self.code_dim)
        nn.init.normal_(self.object_codebook.weight, mean=0.0, std=0.02)
        self.null_object_code = nn.Parameter(torch.zeros(1, self.code_dim))

        self._mesh_point_cloud_cache: Dict[str, torch.Tensor] = {}
        self._load_object_geometry_fn = None
        self._mesh_loader_failed = False

        self.dit = ConditionalDiT(
            target_dim=self.target_dim,
            cond_dim=self.code_dim,
            d_model=_dit_param("dit_d_model", 256),
            nhead=_dit_param("dit_nhead", 8),
            num_layers=_dit_param("dit_num_layers", 6),
            dim_feedforward=_dit_param("dit_dim_feedforward", 1024),
            dropout=_dit_param("dit_dropout", 0.1),
            max_seq_len=_dit_param("dit_max_seq_len", max(self.window_size, 256)),
            timesteps=_dit_param("dit_timesteps", 1000),
            use_time_embed=_dit_param("dit_use_time_embed", True),
            prediction_type=prediction_type,
        )

    @staticmethod
    def _to_bt(
        value,
        *,
        batch_size: int,
        seq_len: int,
        trailing_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        default: float = 0.0,
    ) -> torch.Tensor:
        shape = (batch_size, seq_len, *trailing_shape)
        if not isinstance(value, torch.Tensor):
            return torch.full(shape, float(default), device=device, dtype=dtype)

        out = value.to(device=device, dtype=dtype)
        expected_dim = 2 + len(trailing_shape)
        if out.dim() == expected_dim - 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, *out.shape[1:])

        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return torch.full(shape, float(default), device=device, dtype=dtype)

        if len(trailing_shape) > 0 and tuple(out.shape[2:]) != trailing_shape:
            return torch.full(shape, float(default), device=device, dtype=dtype)

        return out

    @staticmethod
    def _prepare_has_object_mask(
        has_object,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
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
            if mask.shape[1] == 1:
                mask = mask.expand(batch_size, seq_len)
            else:
                mask = mask[:, :seq_len]
                if mask.shape[1] < seq_len:
                    pad = mask[:, -1:].expand(batch_size, seq_len - mask.shape[1])
                    mask = torch.cat([mask, pad], dim=1)

        return mask

    def _prepare_obj_imu(self, obj_imu, *, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        target_dim = self.obj_imu_feat_dim
        if not isinstance(obj_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, target_dim, device=device, dtype=dtype)

        out = obj_imu.to(device=device, dtype=dtype)
        if out.dim() == 4:
            out = out.reshape(batch_size, seq_len, -1)
        if out.dim() != 3:
            return torch.zeros(batch_size, seq_len, target_dim, device=device, dtype=dtype)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, -1, -1)
        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return torch.zeros(batch_size, seq_len, target_dim, device=device, dtype=dtype)

        if out.shape[-1] < target_dim:
            pad = torch.zeros(batch_size, seq_len, target_dim - out.shape[-1], device=device, dtype=dtype)
            out = torch.cat([out, pad], dim=-1)
        elif out.shape[-1] > target_dim:
            out = out[..., :target_dim]

        return out

    def _prepare_human_imu(self, human_imu: torch.Tensor) -> torch.Tensor:
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        out = human_imu
        if out.shape[2] > self.num_human_imus:
            out = out[:, :, : self.num_human_imus]
        elif out.shape[2] < self.num_human_imus:
            pad = torch.zeros(
                batch_size,
                seq_len,
                self.num_human_imus - out.shape[2],
                out.shape[3],
                device=device,
                dtype=dtype,
            )
            out = torch.cat([out, pad], dim=2)

        if out.shape[3] < 9:
            pad = torch.zeros(batch_size, seq_len, self.num_human_imus, 9 - out.shape[3], device=device, dtype=dtype)
            out = torch.cat([out, pad], dim=3)
        elif out.shape[3] > 9:
            out = out[..., :9]

        return out

    def _resolve_obj_trans_init(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        obj_trans_init = data_dict.get("obj_trans_init")
        if isinstance(obj_trans_init, torch.Tensor):
            obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
            if obj_trans_init.dim() == 3 and obj_trans_init.size(1) == 1:
                obj_trans_init = obj_trans_init[:, 0]
            if obj_trans_init.dim() == 1:
                obj_trans_init = obj_trans_init.unsqueeze(0)
            if obj_trans_init.shape[0] == 1 and batch_size > 1:
                obj_trans_init = obj_trans_init.expand(batch_size, -1)
            if obj_trans_init.shape[0] == batch_size and obj_trans_init.shape[-1] == 3:
                return obj_trans_init

        if isinstance(gt_targets, dict) and isinstance(gt_targets.get("obj_trans"), torch.Tensor):
            obj_trans = gt_targets["obj_trans"].to(device=device, dtype=dtype)
            if obj_trans.dim() == 2:
                obj_trans = obj_trans.unsqueeze(0)
            if obj_trans.shape[0] == 1 and batch_size > 1:
                obj_trans = obj_trans.expand(batch_size, -1, -1)
            if obj_trans.shape[0] == batch_size and obj_trans.shape[1] > 0:
                return obj_trans[:, 0]

        return torch.zeros(batch_size, 3, device=device, dtype=dtype)

    def _resolve_human_rot6d(
        self,
        hp_out: Optional[Dict],
        gt_targets: Optional[Dict],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rot = hp_out.get("pred_full_pose_6d") if isinstance(hp_out, dict) else None
        if isinstance(rot, torch.Tensor):
            rot = rot.to(device=device, dtype=dtype)
            if rot.dim() == 3 and rot.shape[-1] == self.human_rot_dim:
                rot = rot.view(batch_size, seq_len, self.num_joints, 6)
            if rot.dim() == 4:
                if rot.shape[0] == 1 and batch_size > 1:
                    rot = rot.expand(batch_size, -1, -1, -1)
                if rot.shape[0] == batch_size and rot.shape[1] == seq_len:
                    joints_now = rot.shape[2]
                    if joints_now > self.num_joints:
                        rot = rot[:, :, : self.num_joints]
                    elif joints_now < self.num_joints:
                        pad = torch.zeros(batch_size, seq_len, self.num_joints - joints_now, 6, device=device, dtype=dtype)
                        rot = torch.cat([rot, pad], dim=2)
                    return rot

        rot_global = gt_targets.get("rotation_global") if isinstance(gt_targets, dict) else None
        if isinstance(rot_global, torch.Tensor):
            rot_global = rot_global.to(device=device, dtype=dtype)
            if rot_global.dim() == 4:
                rot_global = rot_global.unsqueeze(0)
            if rot_global.shape[0] == 1 and batch_size > 1:
                rot_global = rot_global.expand(batch_size, -1, -1, -1, -1)
            if rot_global.shape[0] == batch_size and rot_global.shape[1] == seq_len and rot_global.shape[-2:] == (3, 3):
                joints_now = rot_global.shape[2]
                if joints_now > self.num_joints:
                    rot_global = rot_global[:, :, : self.num_joints]
                elif joints_now < self.num_joints:
                    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
                    pad = eye.expand(batch_size, seq_len, self.num_joints - joints_now, -1, -1)
                    rot_global = torch.cat([rot_global, pad], dim=2)
                rot6d = matrix_to_rotation_6d(rot_global.reshape(-1, 3, 3)).reshape(batch_size, seq_len, self.num_joints, 6)
                return rot6d

        return torch.zeros(batch_size, seq_len, self.num_joints, 6, device=device, dtype=dtype)

    def _resolve_hand_positions(
        self,
        hp_out: Optional[Dict],
        gt_targets: Optional[Dict],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        hand_pos = hp_out.get("pred_hand_glb_pos") if isinstance(hp_out, dict) else None
        if isinstance(hand_pos, torch.Tensor):
            hand_pos = hand_pos.to(device=device, dtype=dtype)
            if hand_pos.dim() == 3 and hand_pos.shape[-1] == 6:
                hand_pos = hand_pos.view(batch_size, seq_len, 2, 3)
            if hand_pos.dim() == 4:
                if hand_pos.shape[0] == 1 and batch_size > 1:
                    hand_pos = hand_pos.expand(batch_size, -1, -1, -1)
                if hand_pos.shape[0] == batch_size and hand_pos.shape[1] == seq_len and hand_pos.shape[2:] == (2, 3):
                    return hand_pos

        position_global = gt_targets.get("position_global") if isinstance(gt_targets, dict) else None
        if isinstance(position_global, torch.Tensor):
            position_global = position_global.to(device=device, dtype=dtype)
            if position_global.dim() == 3:
                position_global = position_global.unsqueeze(0)
            if position_global.shape[0] == 1 and batch_size > 1:
                position_global = position_global.expand(batch_size, -1, -1, -1)
            if position_global.shape[0] == batch_size and position_global.shape[1] == seq_len and position_global.shape[2] > 21:
                return torch.stack(
                    [position_global[:, :, self.hand_joint_indices[0]], position_global[:, :, self.hand_joint_indices[1]]],
                    dim=2,
                )

        return torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)

    def _build_features(
        self,
        data_dict: Dict,
        hp_out: Optional[Dict],
        gt_targets: Optional[Dict],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        human_imu = self._prepare_human_imu(data_dict["human_imu"]).to(device=device, dtype=dtype)
        obj_imu = self._prepare_obj_imu(data_dict.get("obj_imu"), batch_size=batch_size, seq_len=seq_len, device=device, dtype=dtype)

        human_rot6d = self._resolve_human_rot6d(
            hp_out,
            gt_targets,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
        obj_rot6d = obj_imu[..., 3:9]
        human_pose = human_rot6d.reshape(batch_size, seq_len, -1)
        rot_human_obj = torch.cat([human_pose, obj_rot6d], dim=-1)

        human_acc = human_imu[..., :3].reshape(batch_size, seq_len, -1)
        obj_acc = obj_imu[..., :3]
        acc_human_obj = torch.cat([human_acc, obj_acc], dim=-1)

        hands = self._resolve_hand_positions(
            hp_out,
            gt_targets,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
        hands_flat = hands.reshape(batch_size, seq_len, -1)

        contact_prob = torch.stack(
            [
                self._to_bt(
                    gt_targets.get("lhand_contact") if isinstance(gt_targets, dict) else None,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rhand_contact") if isinstance(gt_targets, dict) else None,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)

        bone_dir = torch.cat(
            [
                self._to_bt(
                    gt_targets.get("lhand_obj_direction") if isinstance(gt_targets, dict) else None,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(3,),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rhand_obj_direction") if isinstance(gt_targets, dict) else None,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(3,),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
            ],
            dim=-1,
        )

        bone_len = torch.stack(
            [
                self._to_bt(
                    gt_targets.get("lhand_lb") if isinstance(gt_targets, dict) else None,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rhand_lb") if isinstance(gt_targets, dict) else None,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
            ],
            dim=-1,
        )

        obj_trans = self._to_bt(
            gt_targets.get("obj_trans") if isinstance(gt_targets, dict) else None,
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=(3,),
            device=device,
            dtype=dtype,
            default=0.0,
        )

        delta_p_obj = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        if seq_len > 1:
            delta_p_obj[:, 1:] = obj_trans[:, 1:] - obj_trans[:, :-1]

        return {
            "human_pose": human_pose,
            "obj_imu": obj_imu,
            "rot_human_obj": rot_human_obj,
            "acc_human_obj": acc_human_obj,
            "hands": hands,
            "hands_flat": hands_flat,
            "contact_prob": contact_prob,
            "bone_dir": bone_dir,
            "bone_len": bone_len,
            "obj_trans": obj_trans,
            "delta_p_obj": delta_p_obj,
        }

    @staticmethod
    def _as_string_list(value, *, batch_size: int) -> Tuple[List[str], bool]:
        if isinstance(value, str):
            return [value], batch_size == 1
        if isinstance(value, (list, tuple)):
            out = [str(v) for v in value]
            return out, len(out) == batch_size
        return [], False

    @staticmethod
    def _as_int_list(value, *, batch_size: int) -> Tuple[List[int], bool]:
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().view(-1).tolist()
            out = [int(v) for v in arr]
            return out, len(out) == batch_size
        if isinstance(value, (list, tuple)):
            out = [int(v) for v in value]
            return out, len(out) == batch_size
        if isinstance(value, (int, float)):
            return [int(value)], batch_size == 1
        return [], False

    def _get_load_object_geometry(self):
        if self._mesh_loader_failed:
            return None
        if self._load_object_geometry_fn is not None:
            return self._load_object_geometry_fn
        try:
            from process.preprocess import load_object_geometry
        except Exception:
            self._mesh_loader_failed = True
            return None
        self._load_object_geometry_fn = load_object_geometry
        return self._load_object_geometry_fn

    def _resolve_obj_rot_mats(
        self,
        obj_rot,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not isinstance(obj_rot, torch.Tensor):
            return None

        rot = obj_rot.to(device=device, dtype=dtype)
        if rot.dim() == 2 and rot.shape[-1] == 6:
            rot = rot.unsqueeze(0)
        if rot.dim() == 3 and rot.shape[-1] == 6:
            if rot.shape[0] == 1 and batch_size > 1:
                rot = rot.expand(batch_size, -1, -1)
            if rot.shape[0] != batch_size or rot.shape[1] != seq_len:
                return None
            return rotation_6d_to_matrix(rot.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)

        if rot.dim() == 3 and rot.shape[-2:] == (3, 3):
            rot = rot.unsqueeze(0)
        if rot.dim() == 4 and rot.shape[-2:] == (3, 3):
            if rot.shape[0] == 1 and batch_size > 1:
                rot = rot.expand(batch_size, -1, -1, -1)
            if rot.shape[0] == batch_size and rot.shape[1] == seq_len:
                return rot

        return None

    def _downsample_points(self, points: torch.Tensor) -> torch.Tensor:
        # points: [T,N,3]
        if points.dim() != 3 or points.shape[-1] != 3:
            return points
        num_points = int(points.shape[1])
        if num_points <= self.mesh_downsample_points:
            return points
        idx = torch.linspace(
            0,
            max(num_points - 1, 0),
            steps=self.mesh_downsample_points,
            device=points.device,
            dtype=torch.float32,
        ).long()
        return points[:, idx]

    def _quantize_to_codebook(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B,D]
        codebook = self.object_codebook.weight.to(device=z_e.device, dtype=z_e.dtype)  # [K,D]
        dist = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * z_e @ codebook.t()
            + codebook.pow(2).sum(dim=1, keepdim=True).t()
        )
        code_idx = dist.argmin(dim=1)
        z_q = F.embedding(code_idx, codebook)
        logits = -dist
        return z_q, code_idx, dist, logits

    def _encode_observation_prior(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_seq = torch.cat(
            [
                feats["human_pose"],
                feats["obj_imu"],
                feats["hands_flat"],
            ],
            dim=-1,
        )
        z_e_obs = self.obs_encoder(obs_seq)
        z_q_obs, code_idx_obs, _, logits_obs = self._quantize_to_codebook(z_e_obs)
        return {
            "z_e_obs": z_e_obs,
            "z_q_obs": z_q_obs,
            "code_idx_obs": code_idx_obs,
            "code_logits_obs": logits_obs,
        }

    def _encode_mesh_prior(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict],
        feats: Dict[str, torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        sample_has_object: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_e_mesh = torch.zeros(batch_size, self.code_dim, device=device, dtype=dtype)
        z_q_mesh = torch.zeros(batch_size, self.code_dim, device=device, dtype=dtype)
        code_idx_mesh = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        code_logits_mesh = torch.zeros(batch_size, self.num_codes, device=device, dtype=dtype)
        mesh_valid_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        mesh_dp_mismatch = torch.zeros(batch_size, device=device, dtype=torch.bool)

        if not isinstance(gt_targets, dict):
            return {
                "z_e_mesh": z_e_mesh,
                "z_q_mesh": z_q_mesh,
                "code_idx_mesh": code_idx_mesh,
                "code_logits_mesh": code_logits_mesh,
                "mesh_valid_mask": mesh_valid_mask,
                "mesh_dp_mismatch": mesh_dp_mismatch,
            }

        obj_names, obj_name_ok = self._as_string_list(data_dict.get("obj_name"), batch_size=batch_size)
        if not obj_name_ok:
            # Typical DataParallel behavior: non-tensor list metadata is replicated, not sharded.
            mesh_dp_mismatch.fill_(True)
            return {
                "z_e_mesh": z_e_mesh,
                "z_q_mesh": z_q_mesh,
                "code_idx_mesh": code_idx_mesh,
                "code_logits_mesh": code_logits_mesh,
                "mesh_valid_mask": mesh_valid_mask,
                "mesh_dp_mismatch": mesh_dp_mismatch,
            }

        seq_files, seq_ok = self._as_string_list(data_dict.get("seq_file"), batch_size=batch_size)
        if not seq_ok:
            seq_files = ["unknown_seq"] * batch_size
        starts, starts_ok = self._as_int_list(data_dict.get("window_start"), batch_size=batch_size)
        if not starts_ok:
            starts = [-1] * batch_size
        ends, ends_ok = self._as_int_list(data_dict.get("window_end"), batch_size=batch_size)
        if not ends_ok:
            ends = [-1] * batch_size

        obj_rot_mats = self._resolve_obj_rot_mats(
            gt_targets.get("obj_rot"),
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
        obj_trans = self._to_bt(
            gt_targets.get("obj_trans"),
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=(3,),
            device=device,
            dtype=dtype,
            default=0.0,
        )
        obj_scale = self._to_bt(
            gt_targets.get("obj_scale"),
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=(),
            device=device,
            dtype=dtype,
            default=1.0,
        )
        canonical_points_batch = None
        obj_points_canonical = data_dict.get("obj_points_canonical")
        if isinstance(obj_points_canonical, torch.Tensor):
            points = obj_points_canonical.to(device=device, dtype=dtype)
            if points.dim() == 2:
                points = points.unsqueeze(0)
            if points.shape[0] == 1 and batch_size > 1:
                points = points.expand(batch_size, -1, -1)
            if points.shape[0] == batch_size and points.dim() == 3 and points.shape[-1] == 3:
                canonical_points_batch = points

        if obj_rot_mats is None:
            return {
                "z_e_mesh": z_e_mesh,
                "z_q_mesh": z_q_mesh,
                "code_idx_mesh": code_idx_mesh,
                "code_logits_mesh": code_logits_mesh,
                "mesh_valid_mask": mesh_valid_mask,
                "mesh_dp_mismatch": mesh_dp_mismatch,
            }

        load_object_geometry = None

        valid_points = []
        valid_aux = []
        valid_batch_idx = []

        for b in range(batch_size):
            if not bool(sample_has_object[b].item()):
                continue
            obj_name = obj_names[b]
            if obj_name is None:
                continue
            obj_name = str(obj_name)
            if len(obj_name) == 0 or obj_name.lower() in {"none", "unknown", "unknown_object"}:
                continue

            points = None
            if canonical_points_batch is not None:
                canonical_points = canonical_points_batch[b]
                if canonical_points.dim() == 2 and canonical_points.shape[-1] == 3 and canonical_points.shape[0] > 0:
                    points_seq = canonical_points.unsqueeze(0).expand(seq_len, -1, -1)
                    points = torch.bmm(obj_rot_mats[b], points_seq.transpose(1, 2)).transpose(1, 2)
                    points = points * obj_scale[b].view(seq_len, 1, 1) + obj_trans[b].unsqueeze(1)

            if points is None:
                cache_key = f"{seq_files[b]}::{starts[b]}::{ends[b]}::{obj_name}"
                points_cpu = self._mesh_point_cloud_cache.get(cache_key)
                if points_cpu is None:
                    if load_object_geometry is None:
                        load_object_geometry = self._get_load_object_geometry()
                    if load_object_geometry is None:
                        continue
                    try:
                        transformed_verts, _ = load_object_geometry(
                            obj_name,
                            obj_rot_mats[b].detach().cpu(),
                            obj_trans[b].detach().cpu(),
                            obj_scale[b].detach().cpu(),
                            obj_geo_root=self.object_geo_root,
                            device="cpu",
                        )
                    except Exception:
                        transformed_verts = None
                    if (
                        isinstance(transformed_verts, torch.Tensor)
                        and transformed_verts.dim() == 3
                        and transformed_verts.shape[-1] == 3
                    ):
                        points_cpu = transformed_verts.detach().to(device="cpu", dtype=torch.float32)
                        self._mesh_point_cloud_cache[cache_key] = points_cpu

                if not isinstance(points_cpu, torch.Tensor):
                    continue
                points = points_cpu.to(device=device, dtype=dtype)

            if points.shape[0] != seq_len:
                if points.shape[0] > seq_len:
                    points = points[:seq_len]
                elif points.shape[0] > 0:
                    pad = points[-1:].expand(seq_len - points.shape[0], -1, -1)
                    points = torch.cat([points, pad], dim=0)
                else:
                    continue
            points = self._downsample_points(points)

            valid_points.append(points)
            valid_aux.append(torch.cat([feats["human_pose"][b], feats["hands_flat"][b]], dim=-1))
            valid_batch_idx.append(b)

        if len(valid_batch_idx) == 0:
            return {
                "z_e_mesh": z_e_mesh,
                "z_q_mesh": z_q_mesh,
                "code_idx_mesh": code_idx_mesh,
                "code_logits_mesh": code_logits_mesh,
                "mesh_valid_mask": mesh_valid_mask,
                "mesh_dp_mismatch": mesh_dp_mismatch,
            }

        point_tensor = torch.stack(valid_points, dim=0)
        aux_tensor = torch.stack(valid_aux, dim=0)
        z_e_valid = self.mesh_encoder(point_tensor, aux_tensor)
        z_q_valid, code_idx_valid, _, logits_valid = self._quantize_to_codebook(z_e_valid)

        # Under autocast, mesh branch may run in bf16/fp16 while accumulators stay fp32.
        # Align dtypes before indexed assignment to avoid dtype mismatch errors.
        z_e_valid = z_e_valid.to(dtype=z_e_mesh.dtype)
        z_q_valid = z_q_valid.to(dtype=z_q_mesh.dtype)
        logits_valid = logits_valid.to(dtype=code_logits_mesh.dtype)

        valid_idx_tensor = torch.as_tensor(valid_batch_idx, device=device, dtype=torch.long)
        z_e_mesh[valid_idx_tensor] = z_e_valid
        z_q_mesh[valid_idx_tensor] = z_q_valid
        code_idx_mesh[valid_idx_tensor] = code_idx_valid
        code_logits_mesh[valid_idx_tensor] = logits_valid
        mesh_valid_mask[valid_idx_tensor] = True

        return {
            "z_e_mesh": z_e_mesh,
            "z_q_mesh": z_q_mesh,
            "code_idx_mesh": code_idx_mesh,
            "code_logits_mesh": code_logits_mesh,
            "mesh_valid_mask": mesh_valid_mask,
            "mesh_dp_mismatch": mesh_dp_mismatch,
        }

    def _select_condition_mode(self, batch_size: int, device: torch.device) -> torch.Tensor:
        probs = self.cond_mode_probs.to(device=device)
        return torch.multinomial(probs, num_samples=batch_size, replacement=True)

    def _build_object_condition(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict],
        feats: Dict[str, torch.Tensor],
        context: Dict,
        *,
        training: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = int(context["batch_size"])
        seq_len = int(context["seq_len"])
        device = context["device"]
        dtype = context["dtype"]

        has_object_mask = context.get("has_object_mask")
        if isinstance(has_object_mask, torch.Tensor):
            sample_has_object = has_object_mask.to(device=device, dtype=torch.bool).any(dim=1)
        else:
            sample_has_object = torch.ones(batch_size, device=device, dtype=torch.bool)

        obs_prior = self._encode_observation_prior(feats)
        z_e_obs = obs_prior["z_e_obs"]
        z_q_obs = obs_prior["z_q_obs"]
        code_idx_obs = obs_prior["code_idx_obs"]
        code_logits_obs = obs_prior["code_logits_obs"]

        mesh_prior = {
            "z_e_mesh": torch.zeros(batch_size, self.code_dim, device=device, dtype=dtype),
            "z_q_mesh": torch.zeros(batch_size, self.code_dim, device=device, dtype=dtype),
            "code_idx_mesh": torch.full((batch_size,), -1, device=device, dtype=torch.long),
            "code_logits_mesh": torch.zeros(batch_size, self.num_codes, device=device, dtype=dtype),
            "mesh_valid_mask": torch.zeros(batch_size, device=device, dtype=torch.bool),
            "mesh_dp_mismatch": torch.zeros(batch_size, device=device, dtype=torch.bool),
        }
        if training:
            mesh_prior = self._encode_mesh_prior(
                data_dict,
                gt_targets,
                feats,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                dtype=dtype,
                sample_has_object=sample_has_object,
            )

        z_e_mesh = mesh_prior["z_e_mesh"]
        z_q_mesh = mesh_prior["z_q_mesh"]
        code_idx_mesh = mesh_prior["code_idx_mesh"]
        code_logits_mesh = mesh_prior["code_logits_mesh"]
        mesh_valid_mask = mesh_prior["mesh_valid_mask"]
        mesh_dp_mismatch = mesh_prior["mesh_dp_mismatch"]

        null_code = self.null_object_code.to(device=device, dtype=dtype).expand(batch_size, -1)

        if training:
            # 0: mesh teacher, 1: obs student, 2: null prior
            mode = self._select_condition_mode(batch_size, device)
            # If mesh teacher is unavailable, fallback to obs branch.
            mode = torch.where((mode == 0) & (~mesh_valid_mask), torch.ones_like(mode), mode)
            # No-object sequences always use null prior.
            mode = torch.where(sample_has_object, mode, torch.full_like(mode, 2))

            cond = z_q_obs.clone()
            cond = torch.where(mode[:, None] == 0, z_q_mesh, cond)
            cond = torch.where(mode[:, None] == 2, null_code, cond)
        else:
            mode = torch.full((batch_size,), -1, device=device, dtype=torch.long)
            cond = torch.where(sample_has_object[:, None], z_q_obs, null_code)

        code_idx_obs = torch.where(sample_has_object, code_idx_obs, torch.full_like(code_idx_obs, -1))

        cond_info = {
            "mode": mode,
            "cond": cond,
            "z_e_mesh": z_e_mesh,
            "z_q_mesh": z_q_mesh,
            "code_idx_mesh": code_idx_mesh,
            "code_logits_mesh": code_logits_mesh,
            "mesh_valid_mask": mesh_valid_mask,
            "mesh_dp_mismatch": mesh_dp_mismatch,
            "z_e_obs": z_e_obs,
            "z_q_obs": z_q_obs,
            "code_idx_obs": code_idx_obs,
            "code_logits_obs": code_logits_obs,
            "sample_has_object": sample_has_object,
            "vq_beta": torch.tensor(self.vq_commit_beta, device=device, dtype=dtype),
        }
        return cond, cond_info

    def _build_clean_window(self, data_dict: Dict, hp_out: Optional[Dict], gt_targets: Dict) -> Tuple[torch.Tensor, Dict]:
        if not isinstance(gt_targets, dict):
            raise ValueError("gt_targets must be provided for full-window diffusion training")

        human_imu = data_dict["human_imu"]
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        feats = self._build_features(
            data_dict,
            hp_out,
            gt_targets,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )

        x_clean = torch.cat(
            [
                feats["rot_human_obj"],
                feats["acc_human_obj"],
                feats["hands_flat"],
                feats["contact_prob"],
                feats["bone_dir"],
                feats["bone_len"],
                feats["delta_p_obj"],
            ],
            dim=-1,
        )

        obj_trans_init = self._resolve_obj_trans_init(
            data_dict,
            gt_targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        has_object_mask = self._prepare_has_object_mask(
            data_dict.get("has_object") if isinstance(data_dict, dict) else None,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "obj_trans_init": obj_trans_init,
            "obj_trans_gt": feats["obj_trans"],
            "has_object_mask": has_object_mask,
        }
        return x_clean, context

    def _build_observed_sequence(
        self,
        data_dict: Dict,
        hp_out: Optional[Dict],
        gt_targets: Optional[Dict],
    ) -> Tuple[torch.Tensor, Dict]:
        human_imu = data_dict["human_imu"]
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        feats = self._build_features(
            data_dict,
            hp_out,
            gt_targets,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )

        observed = torch.zeros(batch_size, seq_len, self.target_dim, device=device, dtype=dtype)
        observed[:, :, self.rot_slice] = feats["rot_human_obj"]
        observed[:, :, self.acc_slice] = feats["acc_human_obj"]
        observed[:, :, self.hands_slice] = feats["hands_flat"]

        obj_trans_init = self._resolve_obj_trans_init(
            data_dict,
            gt_targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        has_object_mask = self._prepare_has_object_mask(
            data_dict.get("has_object") if isinstance(data_dict, dict) else None,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "obj_trans_init": obj_trans_init,
            "obj_trans_gt": feats["obj_trans"] if isinstance(gt_targets, dict) else None,
            "has_object_mask": has_object_mask,
        }
        return observed, context

    def _decode_outputs(self, x_pred: torch.Tensor, context: Dict) -> Dict:
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        obj_trans_init = context["obj_trans_init"]

        rot_human_obj = x_pred[:, :, self.rot_slice]
        acc_human_obj = x_pred[:, :, self.acc_slice]
        hands = x_pred[:, :, self.hands_slice].reshape(batch_size, seq_len, 2, 3)

        contact_prob = x_pred[:, :, self.contact_slice]
        bone_dir = x_pred[:, :, self.bone_dir_slice].reshape(batch_size, seq_len, 2, 3)
        bone_len = x_pred[:, :, self.bone_len_slice].reshape(batch_size, seq_len, 2)
        delta_p_obj = x_pred[:, :, self.delta_p_obj_slice]

        has_object_mask = context.get("has_object_mask")
        if isinstance(has_object_mask, torch.Tensor):
            mask = has_object_mask.to(device=device, dtype=dtype).unsqueeze(-1)
            contact_prob = contact_prob * mask
            bone_dir = bone_dir * mask.unsqueeze(-1)
            bone_len = bone_len * mask
            delta_p_obj = delta_p_obj * mask

        pred_obj_trans = torch.cumsum(delta_p_obj, dim=1) + obj_trans_init.unsqueeze(1)
        if isinstance(has_object_mask, torch.Tensor):
            pred_obj_trans = pred_obj_trans * has_object_mask.to(device=device, dtype=dtype).unsqueeze(-1)

        pred_obj_vel = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        pred_obj_acc = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        if seq_len > 1:
            pred_obj_vel[:, 1:] = (pred_obj_trans[:, 1:] - pred_obj_trans[:, :-1]) * self.fps
            pred_obj_vel[:, 0] = pred_obj_vel[:, 1]
        if seq_len > 2:
            pred_obj_acc[:, 2:] = (
                pred_obj_trans[:, 2:] - 2.0 * pred_obj_trans[:, 1:-1] + pred_obj_trans[:, :-2]
            ) * (self.fps ** 2)

        out = {
            "x_pred": x_pred,
            "rot_human_obj_pred": rot_human_obj,
            "acc_human_obj_pred": acc_human_obj,
            "p_hands_pred": hands,
            "contact_prob_pred": contact_prob,
            "bone_dir_pred": bone_dir,
            "bone_len_pred": bone_len,
            "delta_p_obj_pred": delta_p_obj,
            "pred_obj_trans": pred_obj_trans,
            "pred_obj_vel": pred_obj_vel,
            "pred_obj_vel_from_posdiff": pred_obj_vel,
            "pred_obj_acc_from_posdiff": pred_obj_acc,
            "pred_lhand_obj_direction": bone_dir[:, :, 0],
            "pred_rhand_obj_direction": bone_dir[:, :, 1],
            "pred_lhand_lb": bone_len[:, :, 0],
            "pred_rhand_lb": bone_len[:, :, 1],
            "obj_trans_init": obj_trans_init,
            "has_object": has_object_mask,
        }

        if context.get("obj_trans_gt") is not None:
            out["gt_obj_trans"] = context["obj_trans_gt"]

        return out

    def forward(self, data_dict: Dict, hp_out: Optional[Dict] = None, gt_targets: Optional[Dict] = None):
        """Training forward: full-window denoising only."""
        if gt_targets is None:
            raise ValueError("InteractionModule forward requires gt_targets for training")

        x_clean, context = self._build_clean_window(data_dict, hp_out, gt_targets)
        add_noise = bool(self.training and self.use_diffusion_noise)

        feats = self._build_features(
            data_dict,
            hp_out,
            gt_targets,
            batch_size=context["batch_size"],
            seq_len=context["seq_len"],
            device=context["device"],
            dtype=context["dtype"],
        )
        cond, cond_info = self._build_object_condition(
            data_dict,
            gt_targets,
            feats,
            context,
            training=bool(self.training),
        )
        cond_seq = cond.unsqueeze(1).expand(context["batch_size"], context["seq_len"], self.code_dim)

        x0_pred, aux = self.dit(
            cond=cond_seq,
            x_start=x_clean,
            add_noise=add_noise,
        )

        outputs = self._decode_outputs(x0_pred, context)
        outputs["object_prior_aux"] = cond_info
        outputs["diffusion_aux"] = {
            **aux,
            "x0_target": x_clean,
            "x0_pred": x0_pred,
            "observed_dim_mask": self.observed_dim_mask,
            "unknown_dim_mask": self.unknown_dim_mask,
            "prediction_type": self.dit.prediction_type,
            "object_code_idx_pred": cond_info["code_idx_obs"],
            "object_code_emb_pred": cond_info["z_q_obs"],
            "cond_mode": cond_info["mode"],
        }
        return outputs

    def _autoregressive_inference(
        self,
        observed_seq: torch.Tensor,
        *,
        cond_seq: torch.Tensor,
        steps: int,
        inference_type: str,
        sampler: str,
        eta: float,
        warmup_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive sliding-window inference with inpainting."""
        batch_size, seq_len, feat_dim = observed_seq.shape
        device = observed_seq.device
        dtype = observed_seq.dtype

        if cond_seq.dim() != 3 or cond_seq.shape[0] != batch_size or cond_seq.shape[1] != seq_len:
            raise ValueError(f"cond_seq shape mismatch, got {cond_seq.shape}, expected [B={batch_size},T={seq_len},D]")

        window = self.window_size
        if window < 1:
            raise ValueError(f"window_size must be >= 1, got {window}")

        unknown_idx = torch.nonzero(self.unknown_dim_mask.to(device=device), as_tuple=False).flatten()
        unknown_dim = int(unknown_idx.numel())

        history = torch.zeros(batch_size, 0, feat_dim, device=device, dtype=dtype)
        if isinstance(warmup_seq, torch.Tensor):
            if warmup_seq.dim() != 3 or warmup_seq.shape[0] != batch_size or warmup_seq.shape[-1] != feat_dim:
                raise ValueError(
                    f"warmup_seq shape mismatch, got {warmup_seq.shape}, expected [B={batch_size},T_w,{feat_dim}]"
                )
            if warmup_seq.shape[1] >= seq_len:
                raise ValueError(
                    f"warmup_seq is too long for autoregressive rollout: T_w={warmup_seq.shape[1]} vs seq_len={seq_len}"
                )
            history = warmup_seq.to(device=device, dtype=dtype).clone()

        last_unknown = torch.zeros(batch_size, unknown_dim, device=device, dtype=dtype)
        if history.size(1) > 0 and unknown_dim > 0:
            last_unknown = history[:, -1, unknown_idx]

        for frame_idx in range(int(history.size(1)), seq_len):
            current = observed_seq[:, frame_idx].clone()
            if unknown_dim > 0:
                current[:, unknown_idx] = last_unknown

            if window > 1:
                hist = history[:, -(window - 1) :] if history.size(1) > 0 else history
                hist_len = int(hist.size(1))
                if hist_len < (window - 1):
                    pad_count = (window - 1) - hist_len
                    if hist_len > 0:
                        pad_src = hist[:, :1]
                    else:
                        pad_src = current.unsqueeze(1)
                    pad = pad_src.expand(batch_size, pad_count, feat_dim)
                    hist = torch.cat([pad, hist], dim=1)
                x_input = torch.cat([hist, current.unsqueeze(1)], dim=1)
            else:
                x_input = current.unsqueeze(1)

            if window > 1:
                start = max(0, frame_idx - window + 1)
                cond_window = cond_seq[:, start : frame_idx + 1]
                if cond_window.shape[1] < x_input.shape[1]:
                    pad = cond_window[:, :1].expand(batch_size, x_input.shape[1] - cond_window.shape[1], cond_window.shape[-1])
                    cond_window = torch.cat([pad, cond_window], dim=1)
            else:
                cond_window = cond_seq[:, frame_idx : frame_idx + 1]

            inpaint_mask = torch.ones_like(x_input, dtype=torch.bool)
            if unknown_dim > 0:
                inpaint_mask[:, -1, unknown_idx] = False

            if inference_type == "sample_inpaint":
                x_out = self.dit.sample_inpaint(
                    x_input=x_input,
                    inpaint_mask=inpaint_mask,
                    cond=cond_window,
                    x_start=x_input,
                    steps=steps,
                    sampler=sampler,
                    eta=eta,
                )
            else:
                x_out = self.dit.sample_inpaint_x0(
                    x_input=x_input,
                    inpaint_mask=inpaint_mask,
                    cond=cond_window,
                    steps=steps,
                )

            current_pred = x_out[:, -1]
            if unknown_dim > 0:
                last_unknown = current_pred[:, unknown_idx]
                current[:, unknown_idx] = last_unknown

            history = torch.cat([history, current.unsqueeze(1)], dim=1)

        return history

    @torch.no_grad()
    def inference(
        self,
        data_dict: Dict,
        hp_out: Optional[Dict] = None,
        gt_targets: Optional[Dict] = None,
        sample_steps: Optional[int] = None,
        sampler: Optional[str] = None,
        eta: Optional[float] = None,
    ):
        """Inference with autoregressive sliding-window inpainting."""
        observed_seq, context = self._build_observed_sequence(data_dict, hp_out, gt_targets)

        feats = self._build_features(
            data_dict,
            hp_out,
            gt_targets,
            batch_size=context["batch_size"],
            seq_len=context["seq_len"],
            device=context["device"],
            dtype=context["dtype"],
        )
        cond, cond_info = self._build_object_condition(
            data_dict,
            gt_targets,
            feats,
            context,
            training=False,
        )
        cond_seq = cond.unsqueeze(1).expand(context["batch_size"], context["seq_len"], self.code_dim)

        steps = self.inference_steps if sample_steps is None else int(sample_steps)
        if steps is None:
            steps = self.dit.timesteps
        steps = int(steps)

        sampler_name = self.inference_sampler if sampler is None else str(sampler).lower()
        eta_val = self.inference_eta if eta is None else float(eta)
        eta_val = max(eta_val, 0.0)
        if self.inference_type == "sample_inpaint" and sampler_name not in {"ddim", "ddpm"}:
            raise ValueError(f"sampler must be 'ddim' or 'ddpm', got {sampler_name}")

        warmup_seq = None
        warmup_len = 0
        seq_len = int(observed_seq.shape[1])
        if (
            (not self.training)
            and self.test_use_gt_warmup
            and isinstance(gt_targets, dict)
            and isinstance(gt_targets.get("obj_trans"), torch.Tensor)
        ):
            gt_clean_seq, _ = self._build_clean_window(data_dict, hp_out, gt_targets)
            max_warmup = max(seq_len, 1) - 1
            max_window_warmup = max(self.window_size - 1, 0)
            cfg_warmup = max_window_warmup if self.test_warmup_len is None else self.test_warmup_len
            warmup_len = min(max(cfg_warmup, 0), max_window_warmup, max_warmup)
            if warmup_len > 0:
                warmup_seq = gt_clean_seq[:, :warmup_len]

        pred_seq = self._autoregressive_inference(
            observed_seq,
            cond_seq=cond_seq,
            steps=steps,
            inference_type=self.inference_type,
            sampler=sampler_name,
            eta=eta_val,
            warmup_seq=warmup_seq,
        )

        outputs = self._decode_outputs(pred_seq, context)
        outputs["object_prior_aux"] = cond_info
        if self.inference_type == "sample_inpaint":
            aux_sampler = sampler_name
            aux_eta = eta_val
        else:
            aux_sampler = None
            aux_eta = None
        outputs["diffusion_aux"] = {
            "inference_steps": steps,
            "window_size": self.window_size,
            "warmup_len": int(warmup_len),
            "warmup_mode": "gt_prefix" if warmup_len > 0 else "none",
            "observed_dim_mask": self.observed_dim_mask,
            "unknown_dim_mask": self.unknown_dim_mask,
            "prediction_type": self.dit.prediction_type,
            "inference_mode": f"autoregressive_{self.inference_type}",
            "inference_type": self.inference_type,
            "sampler": aux_sampler,
            "eta": aux_eta,
            "object_code_idx_pred": cond_info["code_idx_obs"],
            "object_code_emb_pred": cond_info["z_q_obs"],
        }
        return outputs

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device, num_joints: int = 24, num_human_imus: int = 6):
        human_rot_dim = num_joints * 6
        target_dim = human_rot_dim + 6 + num_human_imus * 3 + 3 + 6 + 2 + 6 + 2 + 3
        return {
            "x_pred": torch.zeros(batch_size, seq_len, target_dim, device=device),
            "rot_human_obj_pred": torch.zeros(batch_size, seq_len, human_rot_dim + 6, device=device),
            "acc_human_obj_pred": torch.zeros(batch_size, seq_len, num_human_imus * 3 + 3, device=device),
            "p_hands_pred": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "contact_prob_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "bone_dir_pred": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "bone_len_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "delta_p_obj_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_trans": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_vel": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_vel_from_posdiff": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_acc_from_posdiff": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_lhand_obj_direction": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_rhand_obj_direction": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_lhand_lb": torch.zeros(batch_size, seq_len, device=device),
            "pred_rhand_lb": torch.zeros(batch_size, seq_len, device=device),
            "obj_trans_init": torch.zeros(batch_size, 3, device=device),
            "has_object": torch.ones(batch_size, seq_len, device=device, dtype=torch.bool),
            "diffusion_aux": {},
        }
