"""
ObjectTransModule: 预测物体位置
Stage 3 - 依赖Stage1和Stage2的输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from .base import RNN, RNNWithInit
from .online import (
    append_stream_data,
    concat_time_dicts,
    infer_batch_seq,
    normalize_inference_mode,
    resolve_online_window,
    select_time_context,
    slice_time_dict,
    slice_time_value,
    take_latest_frame,
)
from configs import FRAME_RATE, _REDUCED_POSE_NAMES, _SENSOR_NAMES


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.GELU(),
    )


def _gather_neighbors(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # x: [B,N,C], idx: [B,N,K]
    batch_size, num_points, channels = x.shape
    k = idx.shape[-1]
    x_expand = x.unsqueeze(1).expand(batch_size, num_points, num_points, channels)
    idx_expand = idx.unsqueeze(-1).expand(batch_size, num_points, k, channels)
    return torch.gather(x_expand, dim=2, index=idx_expand)


def _knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    k = min(max(int(k), 1), int(x.shape[1]))
    with torch.no_grad():
        dist = torch.cdist(x.float(), x.float())
        return dist.topk(k=k, dim=-1, largest=False).indices


class _EdgeConvBlock(nn.Module):
    """Small dependency-free EdgeConv block used by the RNN mesh prior."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, knn_source: torch.Tensor, k: int) -> torch.Tensor:
        idx = _knn_indices(knn_source.detach(), k)
        neigh = _gather_neighbors(x, idx)
        center = x.unsqueeze(2).expand_as(neigh)
        edge = torch.cat((center, neigh - center), dim=-1)
        return self.edge_mlp(edge).amax(dim=2)


class _DGCNNLiteEncoder(nn.Module):
    """DGCNN-lite point encoder with chunked kNN to keep memory bounded."""

    def __init__(self, hidden_dim: int, k: int, chunk_size: int = 64):
        super().__init__()
        mid_dim = max(hidden_dim // 2, 32)
        self.k = int(max(k, 1))
        self.chunk_size = int(max(chunk_size, 1))
        self.edge1 = _EdgeConvBlock(3, mid_dim)
        self.edge2 = _EdgeConvBlock(mid_dim, hidden_dim)
        self.out_proj = nn.Linear(mid_dim + hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _forward_chunk(self, points: torch.Tensor) -> torch.Tensor:
        f1 = self.edge1(points, points, self.k)
        f2 = self.edge2(f1, f1, self.k)
        return self.out_norm(self.out_proj(torch.cat((f1, f2), dim=-1)))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: [B,T,N,3] -> [B,T,N,H]
        batch_size, seq_len, num_points, _ = points.shape
        flat = points.reshape(batch_size * seq_len, num_points, 3)
        chunks = []
        for start in range(0, flat.shape[0], self.chunk_size):
            chunks.append(self._forward_chunk(flat[start:start + self.chunk_size]))
        return torch.cat(chunks, dim=0).reshape(batch_size, seq_len, num_points, -1)


class _ObsEncoder(nn.Module):
    """Observation prior encoder: IMU + human context -> per-frame interaction code."""

    def __init__(
        self,
        *,
        obj_imu_dim: int,
        pose_dim: int,
        hidden_dim: int,
        code_dim: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.in_dim = obj_imu_dim + pose_dim + 3 + 6
        self.in_proj = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        layers = max(int(layers), 1)
        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=float(dropout) if layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, code_dim)

    def forward(
        self,
        obj_imu: torch.Tensor,
        human_pose: torch.Tensor,
        root_trans: torch.Tensor,
        hand_positions: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            (
                obj_imu,
                human_pose,
                root_trans,
                hand_positions.reshape(hand_positions.shape[0], hand_positions.shape[1], -1),
            ),
            dim=-1,
        )
        h = self.in_proj(x)
        h, _ = self.temporal(h)
        return self.out_proj(h)


class _MeshPriorEncoder(nn.Module):
    """Privileged mesh teacher: canonical object geometry + human context -> per-frame code."""

    def __init__(
        self,
        *,
        pose_dim: int,
        hidden_dim: int,
        code_dim: int,
        layers: int,
        dropout: float,
        k: int,
        chunk_size: int,
    ):
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.point_encoder = _DGCNNLiteEncoder(hidden_dim=hidden_dim, k=k, chunk_size=chunk_size)
        self.human_mlp = _make_mlp(self.pose_dim + 3 + 6, hidden_dim, hidden_dim, dropout)
        heads = 8
        while hidden_dim % heads != 0 and heads > 1:
            heads //= 2
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.frame_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        layers = max(int(layers), 1)
        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=float(dropout) if layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_dim, code_dim)

    @staticmethod
    def _expand_bt(value, batch_size, seq_len, trailing_shape, device, dtype, default=0.0):
        shape = (batch_size, seq_len, *trailing_shape)
        if not isinstance(value, torch.Tensor):
            return torch.full(shape, float(default), device=device, dtype=dtype), False
        out = value.to(device=device, dtype=dtype)
        if out.dim() == len(shape) - 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, *out.shape[1:])
        if out.dim() != len(shape) or out.shape[:2] != (batch_size, seq_len):
            return torch.full(shape, float(default), device=device, dtype=dtype), False
        if tuple(out.shape[2:]) != tuple(trailing_shape):
            return torch.full(shape, float(default), device=device, dtype=dtype), False
        return out, True

    @staticmethod
    def _prepare_points(value, batch_size, device, dtype):
        if not isinstance(value, torch.Tensor):
            return None
        points = value.to(device=device, dtype=dtype)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if points.shape[0] == 1 and batch_size > 1:
            points = points.expand(batch_size, -1, -1)
        if points.dim() != 3 or points.shape[0] != batch_size or points.shape[-1] != 3:
            return None
        return points

    def forward(
        self,
        *,
        obj_points_canonical,
        obj_rot_gt,
        obj_trans_gt,
        obj_scale_gt,
        hand_positions: torch.Tensor,
        human_pose: torch.Tensor,
        root_trans: torch.Tensor,
        sample_has_object: torch.Tensor,
    ):
        batch_size, seq_len = hand_positions.shape[:2]
        device = hand_positions.device
        dtype = hand_positions.dtype
        zero_code = torch.zeros(batch_size, seq_len, self.out_proj.out_features, device=device, dtype=dtype)
        valid_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

        # GT object rotation/translation are intentionally ignored here. The
        # teacher should not expose pose information that the obs encoder lacks.
        points = self._prepare_points(obj_points_canonical, batch_size, device, dtype)
        obj_scale, _ = self._expand_bt(obj_scale_gt, batch_size, seq_len, (), device, dtype, default=1.0)
        if points is None:
            return zero_code, valid_mask

        nonzero_points = points.abs().sum(dim=(1, 2)) > 1e-8
        valid_mask = sample_has_object.to(device=device, dtype=torch.bool) & nonzero_points
        if not valid_mask.any():
            return zero_code, valid_mask

        points_seq = points[:, None] * obj_scale[:, :1].view(batch_size, 1, 1, 1)
        point_tokens = self.point_encoder(points_seq).expand(-1, seq_len, -1, -1)
        human_token = self.human_mlp(
            torch.cat((human_pose, root_trans, hand_positions.reshape(batch_size, seq_len, -1)), dim=-1)
        )

        bt = batch_size * seq_len
        query = human_token.reshape(bt, 1, -1)
        key_value = point_tokens.reshape(bt, point_tokens.shape[2], point_tokens.shape[3])
        attn_token, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
        attn_token = attn_token.reshape(batch_size, seq_len, -1)
        obj_token = point_tokens.amax(dim=2)
        fused = self.frame_fuse(torch.cat((human_token, attn_token, obj_token), dim=-1))
        temporal, _ = self.temporal(fused)
        code = self.out_proj(temporal)
        return code * valid_mask.to(dtype=dtype).view(batch_size, 1, 1), valid_mask


class ObjectTransModule(nn.Module):
    """
    物体位置预测模块
    基于手部位置、接触概率和物体IMU预测物体位置
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(max(getattr(cfg, "obj_imu_dim", self.imu_dim), 9))
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        hidden_dim_multiplier = getattr(cfg, "hidden_dim_multiplier", 1)

        # Gating网络参数
        self.gating_prior_beta = getattr(cfg, "gating_prior_beta", 5.0)
        self.gating_temperature = getattr(cfg, "gating_temperature", 5.0)
        self.gating_smoothing_enabled = getattr(cfg, "gating_smoothing_enabled", False)
        self.gating_smoothing_alpha = getattr(cfg, "gating_smoothing_alpha", 0.6)
        self.gating_max_change = getattr(cfg, "gating_max_change", 0.25)
        
        # 速度校正参数
        self.vel_static_threshold = getattr(cfg, "vel_static_threshold", 0.3)
        self.vel_min_hand_speed = getattr(cfg, "vel_min_hand_speed", 0.02)
        self.refine_pose_dim = len(_REDUCED_POSE_NAMES) * 6

        self.interaction_code_dim = int(getattr(cfg, "interaction_code_dim", getattr(cfg, "object_code_dim", 128)))
        prior_hidden_dim = int(getattr(cfg, "prior_encoder_hidden_dim", 256))
        prior_layers = int(getattr(cfg, "prior_encoder_layers", 2))
        prior_dropout = float(getattr(cfg, "prior_encoder_dropout", 0.1))
        dgcnn_k = int(getattr(cfg, "dgcnn_k", 16))
        dgcnn_chunk_size = int(getattr(cfg, "dgcnn_chunk_size", 64))
        mode_probs_cfg = getattr(cfg, "cond_mode_probs", [0.4, 0.4, 0.2])
        if not isinstance(mode_probs_cfg, (list, tuple)) or len(mode_probs_cfg) != 3:
            mode_probs_cfg = [0.4, 0.4, 0.2]
        cond_mode_probs = torch.tensor(mode_probs_cfg, dtype=torch.float32)
        if float(cond_mode_probs.sum().item()) <= 0.0:
            cond_mode_probs = torch.tensor([0.4, 0.4, 0.2], dtype=torch.float32)
        self.register_buffer("cond_mode_probs", cond_mode_probs / cond_mode_probs.sum().clamp_min(1e-8), persistent=False)

        self.mesh_prior_encoder = _MeshPriorEncoder(
            pose_dim=self.refine_pose_dim,
            hidden_dim=prior_hidden_dim,
            code_dim=self.interaction_code_dim,
            layers=prior_layers,
            dropout=prior_dropout,
            k=dgcnn_k,
            chunk_size=dgcnn_chunk_size,
        )
        self.obs_encoder = _ObsEncoder(
            obj_imu_dim=self.obj_imu_dim,
            pose_dim=self.refine_pose_dim,
            hidden_dim=prior_hidden_dim,
            code_dim=self.interaction_code_dim,
            layers=prior_layers,
            dropout=prior_dropout,
        )

        n_fk_branch_input = 6 + 3 + 1 + self.obj_imu_dim + self.imu_dim + 3 + 3 + self.interaction_code_dim
        n_gating_input = 9 + self.interaction_code_dim

        # 左手FK预测头
        self.lhand_fk_head = RNNWithInit(
            n_input=n_fk_branch_input,
            n_output=4,  # 方向(3) + 长度(1)
            n_hidden=128 * hidden_dim_multiplier,
            n_init=4,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        
        # 右手FK预测头
        self.rhand_fk_head = RNNWithInit(
            n_input=n_fk_branch_input,
            n_output=4,
            n_hidden=128 * hidden_dim_multiplier,
            n_init=4,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        
        # Gating网络
        self.gating_head = RNNWithInit(
            n_input=n_gating_input,
            n_output=3,  # 左手FK、右手FK、IMU积分
            n_hidden=64 * hidden_dim_multiplier,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=0.2,
        )

        n_refine_input = self.refine_pose_dim + 21
        self.refine_head = RNN(
            n_input=n_refine_input,
            n_output=self.refine_pose_dim + 3,
            n_hidden=128 * hidden_dim_multiplier,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        nn.init.zeros_(self.refine_head.linear2.weight)
        nn.init.zeros_(self.refine_head.linear2.bias)

    @staticmethod
    def _unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return x / norm

    @staticmethod
    def _softplus_positive(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-4

    def _smooth_gating_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """平滑gating权重，减少跳变"""
        if self.training or (not self.gating_smoothing_enabled) or weights.size(1) < 2:
            return weights

        LHAND_FK, RHAND_FK, IMU_BRANCH = 0, 1, 2
        smoothed_weights = weights.clone()
        prev_smoothed = weights[:, 0, :]

        for t in range(1, weights.size(1)):
            current_weights = weights[:, t, :]
            prev_dominant = prev_smoothed.argmax(dim=-1)
            curr_dominant = current_weights.argmax(dim=-1)
            frame_weights = current_weights.clone()

            transition_mask = prev_dominant != curr_dominant
            if transition_mask.any():
                for b in range(weights.size(0)):
                    if not transition_mask[b]:
                        continue
                    prev_dom = prev_dominant[b].item()
                    curr_dom = curr_dominant[b].item()
                    need_smoothing = (
                        (prev_dom == IMU_BRANCH and curr_dom in [LHAND_FK, RHAND_FK]) or
                        (prev_dom in [LHAND_FK, RHAND_FK] and curr_dom in [LHAND_FK, RHAND_FK])
                    )
                    if need_smoothing:
                        frame_weights[b, :] = (
                            self.gating_smoothing_alpha * prev_smoothed[b, :] +
                            (1.0 - self.gating_smoothing_alpha) * current_weights[b, :]
                        )
                        if self.gating_max_change > 0:
                            weight_change = frame_weights[b, :] - prev_smoothed[b, :]
                            change_norm = torch.norm(weight_change)
                            if change_norm > self.gating_max_change:
                                weight_change = weight_change * (self.gating_max_change / (change_norm + 1e-8))
                                frame_weights[b, :] = prev_smoothed[b, :] + weight_change
                        frame_weights[b, :] = F.softmax(
                            torch.log(frame_weights[b, :] + 1e-8) * self.gating_temperature, dim=-1
                        )
            smoothed_weights[:, t, :] = frame_weights
            prev_smoothed = frame_weights

        return smoothed_weights

    def _build_fk_inputs(self, obj_rot6d, hand_pos, hand_contact_scalar, obj_imu9, hand_imu9, obj_vel3, obj_rot_delta3, interaction_code):
        return torch.cat(
            [obj_rot6d, hand_pos, hand_contact_scalar, obj_imu9, hand_imu9, obj_vel3, obj_rot_delta3, interaction_code],
            dim=2,
        )

    def _build_gating_inputs(self, contact_prob3, obj_vel3, obj_imu_acc3, interaction_code):
        return torch.cat([contact_prob3, obj_vel3, obj_imu_acc3, interaction_code], dim=2)

    @staticmethod
    def _pad_or_trim_last_dim(x: torch.Tensor, out_dim: int) -> torch.Tensor:
        if x.shape[-1] == out_dim:
            return x
        if x.shape[-1] > out_dim:
            return x[..., :out_dim]
        pad_shape = (*x.shape[:-1], out_dim - x.shape[-1])
        return torch.cat((x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)), dim=-1)

    def _prepare_obj_imu(self, obj_imu: torch.Tensor, batch_size: int, seq_len: int, device, dtype) -> torch.Tensor:
        if not isinstance(obj_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        out = obj_imu.to(device=device, dtype=dtype)
        if out.dim() == 4:
            out = out.reshape(batch_size, seq_len, -1)
        if out.dim() == 2:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, -1, -1)
        if out.dim() != 3 or out.shape[:2] != (batch_size, seq_len):
            return torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        return self._pad_or_trim_last_dim(out, self.obj_imu_dim)

    def _prepare_human_imu(self, human_imu: torch.Tensor, batch_size: int, seq_len: int, device, dtype) -> torch.Tensor:
        if not isinstance(human_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, self.num_human_imus, self.imu_dim, device=device, dtype=dtype)
        out = human_imu.to(device=device, dtype=dtype)
        if out.dim() == 3 and out.shape[-1] == self.num_human_imus * self.imu_dim:
            out = out.view(batch_size, seq_len, self.num_human_imus, self.imu_dim)
        if out.dim() != 4 or out.shape[:2] != (batch_size, seq_len):
            return torch.zeros(batch_size, seq_len, self.num_human_imus, self.imu_dim, device=device, dtype=dtype)
        if out.shape[2] != self.num_human_imus or out.shape[3] != self.imu_dim:
            flat = out.reshape(batch_size, seq_len, -1)
            flat = self._pad_or_trim_last_dim(flat, self.num_human_imus * self.imu_dim)
            out = flat.view(batch_size, seq_len, self.num_human_imus, self.imu_dim)
        return out

    def _prepare_pose_feature(self, human_pose_input, batch_size, seq_len, device, dtype):
        pose = self._prepare_refine_pose(human_pose_input, batch_size, seq_len, device, dtype)
        if pose is None:
            return torch.zeros(batch_size, seq_len, self.refine_pose_dim, device=device, dtype=dtype)
        return pose

    @staticmethod
    def _prepare_root_feature(root_trans_input, batch_size, seq_len, device, dtype):
        if not isinstance(root_trans_input, torch.Tensor):
            return torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        root = root_trans_input.to(device=device, dtype=dtype)
        if root.dim() == 2:
            root = root.unsqueeze(1).expand(batch_size, seq_len, 3)
        if root.dim() == 3 and root.shape[:2] == (batch_size, seq_len) and root.shape[-1] == 3:
            return root
        return torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)

    @staticmethod
    def _sample_has_object(has_object_mask, batch_size, seq_len, device) -> torch.Tensor:
        if has_object_mask is None:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        if isinstance(has_object_mask, torch.Tensor):
            mask = has_object_mask.to(device=device, dtype=torch.bool)
        elif isinstance(has_object_mask, (bool, int)):
            return torch.full((batch_size,), bool(has_object_mask), device=device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(has_object_mask, device=device, dtype=torch.bool)
        if mask.dim() == 0:
            mask = mask.view(1)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, *mask.shape[1:])
        if mask.dim() == 1:
            return mask[:batch_size]
        if mask.shape[0] != batch_size:
            return mask.reshape(-1)[:1].expand(batch_size)
        return mask.reshape(batch_size, -1).any(dim=1)

    def _select_condition_mode(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.multinomial(self.cond_mode_probs.to(device=device), batch_size, replacement=True)

    def _build_interaction_condition(
        self,
        *,
        hand_positions,
        obj_imu,
        human_pose_input,
        root_trans_input,
        has_object_mask,
        obj_points_canonical,
        obj_rot_gt,
        obj_trans_gt,
        obj_scale_gt,
    ):
        batch_size, seq_len = hand_positions.shape[:2]
        device = hand_positions.device
        dtype = hand_positions.dtype
        human_pose = self._prepare_pose_feature(human_pose_input, batch_size, seq_len, device, dtype)
        root_trans = self._prepare_root_feature(root_trans_input, batch_size, seq_len, device, dtype)
        sample_has_object = self._sample_has_object(has_object_mask, batch_size, seq_len, device)

        obs_code = self.obs_encoder(obj_imu, human_pose, root_trans, hand_positions)
        mesh_code = torch.zeros_like(obs_code)
        mesh_valid_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if self.training:
            mesh_code, mesh_valid_mask = self.mesh_prior_encoder(
                obj_points_canonical=obj_points_canonical,
                obj_rot_gt=obj_rot_gt,
                obj_trans_gt=obj_trans_gt,
                obj_scale_gt=obj_scale_gt,
                hand_positions=hand_positions,
                human_pose=human_pose,
                root_trans=root_trans,
                sample_has_object=sample_has_object,
            )

        null_code = torch.zeros_like(obs_code)
        if self.training:
            mode = self._select_condition_mode(batch_size, device)
            mode = torch.where((mode == 0) & (~mesh_valid_mask), torch.ones_like(mode), mode)
            mode = torch.where(sample_has_object, mode, torch.full_like(mode, 2))
            interaction_code = obs_code.clone()
            interaction_code = torch.where(mode.view(batch_size, 1, 1) == 0, mesh_code, interaction_code)
            interaction_code = torch.where(mode.view(batch_size, 1, 1) == 2, null_code, interaction_code)
        else:
            mode = torch.where(
                sample_has_object,
                torch.ones(batch_size, device=device, dtype=torch.long),
                torch.full((batch_size,), 2, device=device, dtype=torch.long),
            )
            interaction_code = torch.where(sample_has_object.view(batch_size, 1, 1), obs_code, null_code)

        return interaction_code, {
            "mesh_code": mesh_code,
            "obs_code": obs_code,
            "mode": mode,
            "mesh_valid_mask": mesh_valid_mask,
            "sample_has_object": sample_has_object,
        }

    def _rot6d_delta(self, rot6d: torch.Tensor) -> torch.Tensor:
        B, T, _ = rot6d.shape
        R = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(B, T, 3, 3)
        rel = torch.matmul(R[:, 1:].transpose(-1, -2), R[:, :-1])
        aa = matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(B, T-1, 3)
        aa = F.pad(aa, (0, 0, 1, 0))
        return aa

    def _compute_hand_velocity(self, hand_pos: torch.Tensor) -> torch.Tensor:
        """从手部位置计算速度 [B, T, 3] -> [B, T, 3]"""
        vel = torch.zeros_like(hand_pos)
        if hand_pos.size(1) > 1:
            vel[:, 1:] = (hand_pos[:, 1:] - hand_pos[:, :-1]) * FRAME_RATE
        return vel

    def _correct_obj_velocity(
        self,
        v_imu: torch.Tensor,      # [B, T, 3]
        v_lhand: torch.Tensor,    # [B, T, 3]
        v_rhand: torch.Tensor,    # [B, T, 3]
        p_left: torch.Tensor,     # [B, T, 1]
        p_right: torch.Tensor,    # [B, T, 1]
        p_move: torch.Tensor,     # [B, T, 1]
    ) -> torch.Tensor:
        """
        两阶段速度校正：
        1. 静止校正：无接触时物体静止
        2. 接触方向校正：接触时物体速度在手方向上的分量=手速度
        """
        # === 阶段1：静止校正因子 ===
        # 以物体运动概率为依据
        static_factor = torch.clamp(p_move / self.vel_static_threshold, 0, 1)
        
        # === 阶段2：接触方向校正 ===
        def direction_correct(v_obj, v_hand, p_contact):
            """在手运动方向上校正物体速度"""
            v_hand_speed = v_hand.norm(dim=-1, keepdim=True)
            hand_moving = (v_hand_speed > self.vel_min_hand_speed).float()
            v_hand_dir = v_hand / v_hand_speed.clamp_min(1e-6)
            
            # v_obj在手方向上的投影
            proj_scalar = (v_obj * v_hand_dir).sum(dim=-1, keepdim=True)
            
            # 校正: 去掉v_obj在手方向的分量，换成手速度
            v_corrected = v_obj - proj_scalar * v_hand_dir + v_hand
            
            # 只有手在运动且有接触时才应用
            w = p_contact * hand_moving
            return v_obj * (1 - w) + v_corrected * w
        
        # 分别用左右手校正
        v_lcorr = direction_correct(v_imu, v_lhand, p_left)
        v_rcorr = direction_correct(v_imu, v_rhand, p_right)
        
        # 按接触概率融合
        total_p = p_left + p_right + 1e-6
        v_contact_corrected = (p_left / total_p) * v_lcorr + (p_right / total_p) * v_rcorr
        
        # 最终：静止因子控制整体幅度
        return static_factor * v_contact_corrected

    def _compute_init_dir_len(self, hand_pos_0, obj_rotm_0, obj_pos_0):
        vec_world = obj_pos_0 - hand_pos_0
        lb0 = vec_world.norm(dim=-1, keepdim=True)
        unit_world = self._unit_vector(vec_world)
        obj_Rt = obj_rotm_0.transpose(-1, -2)
        oe0 = torch.bmm(obj_Rt, unit_world.unsqueeze(-1)).squeeze(-1)
        return oe0, lb0

    def _prepare_refine_pose(self, human_pose_input, batch_size, seq_len, device, dtype):
        if not isinstance(human_pose_input, torch.Tensor):
            return None
        pose = human_pose_input.to(device=device, dtype=dtype)
        if pose.dim() == 4:
            pose = pose.reshape(batch_size, seq_len, -1)
        if pose.dim() == 3 and pose.shape[:2] == (batch_size, seq_len) and pose.shape[-1] == self.refine_pose_dim:
            return pose
        return None

    def _prepare_refine_root(self, root_trans_input, batch_size, seq_len, device, dtype):
        if not isinstance(root_trans_input, torch.Tensor):
            return None
        root = root_trans_input.to(device=device, dtype=dtype)
        if root.dim() == 2:
            root = root.unsqueeze(1).expand(batch_size, seq_len, 3)
        if root.dim() == 3 and root.shape[:2] == (batch_size, seq_len) and root.shape[-1] == 3:
            return root
        return None

    def _compute_human_refinement(
        self,
        human_pose_input,
        root_trans_input,
        fused_pos,
        obj_vel_corrected,
        weights,
        pred_hand_contact_prob,
        lhand_position,
        rhand_position,
        has_object_mask,
        enable_refine,
    ):
        if not enable_refine:
            return {}
        batch_size, seq_len = fused_pos.shape[:2]
        device = fused_pos.device
        dtype = fused_pos.dtype
        base_pose = self._prepare_refine_pose(human_pose_input, batch_size, seq_len, device, dtype)
        base_root = self._prepare_refine_root(root_trans_input, batch_size, seq_len, device, dtype)
        if base_pose is None or base_root is None:
            return {}

        refine_input = torch.cat(
            (
                fused_pos,
                obj_vel_corrected,
                weights,
                pred_hand_contact_prob,
                lhand_position,
                rhand_position,
                base_root,
                base_pose,
            ),
            dim=-1,
        )
        residual = self.refine_head(refine_input)
        pose_delta = residual[:, :, : self.refine_pose_dim]
        root_trans_delta = residual[:, :, self.refine_pose_dim :]
        if has_object_mask is not None:
            mask = has_object_mask
            if mask.dim() > 1:
                mask = mask.view(batch_size)
            mask = mask.to(device=device, dtype=dtype).view(batch_size, 1, 1)
            pose_delta = pose_delta * mask
            root_trans_delta = root_trans_delta * mask

        return {
            "pose_delta": pose_delta,
            "root_trans_delta": root_trans_delta,
            "refined_pose": base_pose + pose_delta,
            "refined_root_trans": base_root + root_trans_delta,
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
        human_pose_input: torch.Tensor = None,
        root_trans_input: torch.Tensor = None,
        obj_points_canonical: torch.Tensor = None,
        obj_rot_gt: torch.Tensor = None,
        obj_trans_gt: torch.Tensor = None,
        obj_scale_gt: torch.Tensor = None,
        enable_refine: bool = True,
        known_obj_trans_prefix: torch.Tensor = None,
    ):
        """
        前向传播
        
        Args:
            hand_positions: [B, T, 2, 3] 手部位置 (来自HumanPoseModule)
            pred_hand_contact_prob: [B, T, 3] 接触概率 (左/右无条件接触 + 物体运动，来自VelocityContactModule)
            obj_trans_init: [B, 3] 初始物体位置
            obj_imu: [B, T, 9] 物体IMU数据
            human_imu: [B, T, num_imu, imu_dim] 人体IMU数据
            obj_vel_input: [B, T, 3] 物体速度 (来自VelocityContactModule)
            contact_init: [B, 3] 初始接触状态
            has_object_mask: [B] 是否有物体
            human_pose_input: [B, T, num_reduced*6] 人体姿态基线，用于残差微调
            root_trans_input: [B, T, 3] 根节点位移基线，用于残差微调
            obj_points_canonical: [B, N, 3] 训练期物体canonical点云
            obj_scale_gt: 训练期mesh teacher可选scale
            obj_rot_gt/obj_trans_gt: 兼容旧调用，MeshPrior不再使用GT物体姿态
            enable_refine: 是否输出人体姿态/root trans微调结果
        
        Returns:
            dict: 包含预测的物体位置和相关信息
        """
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

        # 处理输入
        obj_imu = self._prepare_obj_imu(obj_imu, bs, seq_len, device, dtype)
        human_imu = self._prepare_human_imu(human_imu, bs, seq_len, device, dtype)

        obj_rot = obj_imu[:, :, 3:9]
        obj_rot_delta = self._rot6d_delta(obj_rot)
        obj_rotm = rotation_6d_to_matrix(obj_rot.reshape(-1, 6)).reshape(bs, seq_len, 3, 3)
        obj_imu_acc = obj_imu[:, :, :3]

        l_idx = _SENSOR_NAMES.index("LeftForeArm")
        r_idx = _SENSOR_NAMES.index("RightForeArm")
        lhand_imu9 = human_imu[:, :, l_idx, :]
        rhand_imu9 = human_imu[:, :, r_idx, :]

        if obj_vel_input is None:
            obj_vel_input = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)

        pL = pred_hand_contact_prob[:, :, 0:1]
        pR = pred_hand_contact_prob[:, :, 1:2]
        p_move = pred_hand_contact_prob[:, :, 2:3]

        # 计算手速度并校正物体速度
        lhand_vel = self._compute_hand_velocity(lhand_position)
        rhand_vel = self._compute_hand_velocity(rhand_position)
        obj_vel_corrected = self._correct_obj_velocity(obj_vel_input, lhand_vel, rhand_vel, pL, pR, p_move)

        interaction_code, interaction_prior_aux = self._build_interaction_condition(
            hand_positions=hand_positions,
            obj_imu=obj_imu,
            human_pose_input=human_pose_input,
            root_trans_input=root_trans_input,
            has_object_mask=has_object_mask,
            obj_points_canonical=obj_points_canonical,
            obj_rot_gt=obj_rot_gt,
            obj_trans_gt=obj_trans_gt,
            obj_scale_gt=obj_scale_gt,
        )

        # 构建FK输入
        fk_l_input = self._build_fk_inputs(obj_rot, lhand_position, pL, obj_imu, lhand_imu9, obj_vel_input, obj_rot_delta, interaction_code)
        fk_r_input = self._build_fk_inputs(obj_rot, rhand_position, pR, obj_imu, rhand_imu9, obj_vel_input, obj_rot_delta, interaction_code)

        # 计算初始方向和长度
        obj_pos_0 = obj_trans_init
        obj_R_0 = obj_rotm[:, 0, :, :]
        l_hand_0 = lhand_position[:, 0, :]
        r_hand_0 = rhand_position[:, 0, :]
        l_oe0, l_lb0 = self._compute_init_dir_len(l_hand_0, obj_R_0, obj_pos_0)
        r_oe0, r_lb0 = self._compute_init_dir_len(r_hand_0, obj_R_0, obj_pos_0)

        l_init_vec = torch.cat((l_oe0, l_lb0), dim=-1)
        r_init_vec = torch.cat((r_oe0, r_lb0), dim=-1)

        if contact_init is None:
            contact_init_vec = torch.zeros(bs, 3, device=device, dtype=dtype)
        else:
            if contact_init.dim() == 1:
                contact_init_vec = contact_init.unsqueeze(0).expand(bs, -1)
            else:
                contact_init_vec = contact_init
            if contact_init_vec.shape[-1] > 3:
                contact_init_vec = contact_init_vec[..., :3]
            if contact_init_vec.shape[-1] != 3:
                raise ValueError(f"contact_init must have last dim 3, got {contact_init_vec.shape}")

        # FK预测
        l_fk_out = self.lhand_fk_head((fk_l_input, l_init_vec))
        r_fk_out = self.rhand_fk_head((fk_r_input, r_init_vec))
        l_dir = self._unit_vector(l_fk_out[:, :, :3])
        r_dir = self._unit_vector(r_fk_out[:, :, :3])
        l_len = self._softplus_positive(l_fk_out[:, :, 3])
        r_len = self._softplus_positive(r_fk_out[:, :, 3])

        # 转换到世界坐标系
        obj_rotm_flat = obj_rotm.reshape(bs * seq_len, 3, 3)
        l_dir_world = torch.bmm(obj_rotm_flat, l_dir.reshape(bs * seq_len, 3, 1)).reshape(bs, seq_len, 3)
        r_dir_world = torch.bmm(obj_rotm_flat, r_dir.reshape(bs * seq_len, 3, 1)).reshape(bs, seq_len, 3)
        l_pos_fk = lhand_position + l_dir_world * l_len.unsqueeze(-1)
        r_pos_fk = rhand_position + r_dir_world * r_len.unsqueeze(-1)

        # Gating预测
        gating_input = self._build_gating_inputs(pred_hand_contact_prob, obj_vel_input, obj_imu_acc, interaction_code)
        gate_logits = self.gating_head((gating_input, contact_init_vec))
        prior_im = 1.0 - p_move.squeeze(-1)
        prior = torch.stack([pL.squeeze(-1), pR.squeeze(-1), prior_im], dim=-1)
        gate_logits = gate_logits + self.gating_prior_beta * torch.log(prior + 1e-6)
        weights_raw = F.softmax(gate_logits / self.gating_temperature, dim=-1)
        weights = self._smooth_gating_weights(weights_raw)

        # 融合位置（使用校正后的速度）。online模式下，窗口prefix来自已知历史结果，
        # 只预测窗口内尚未知的尾部帧。
        fused_pos = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)
        prefix_len = 0
        if isinstance(known_obj_trans_prefix, torch.Tensor):
            prefix = known_obj_trans_prefix.to(device=device, dtype=dtype)
            if prefix.dim() == 2:
                prefix = prefix.unsqueeze(1)
            if prefix.dim() == 3 and prefix.shape[0] == bs and prefix.shape[-1] == 3:
                prefix_len = min(int(prefix.shape[1]), seq_len)
                if prefix_len > 0:
                    fused_pos[:, :prefix_len, :] = prefix[:, :prefix_len, :]
        dt = 1.0 / FRAME_RATE
        for t in range(prefix_len, seq_len):
            prev_pos = fused_pos[:, t - 1, :] if t > 0 else obj_trans_init
            pos_imu_integrated = prev_pos + obj_vel_corrected[:, t, :] * dt
            fused_pos[:, t, :] = (
                weights[:, t, 0:1] * l_pos_fk[:, t, :] +
                weights[:, t, 1:2] * r_pos_fk[:, t, :] +
                weights[:, t, 2:3] * pos_imu_integrated
            )

        # 计算速度和加速度
        vel_from_pos = torch.zeros_like(fused_pos)
        acc_from_pos = torch.zeros_like(fused_pos)
        if seq_len > 1:
            vel_from_pos[:, 1:] = (fused_pos[:, 1:] - fused_pos[:, :-1]) * FRAME_RATE
        if seq_len > 2:
            acc_from_pos[:, 2:] = (fused_pos[:, 2:] - 2 * fused_pos[:, 1:-1] + fused_pos[:, :-2]) * (FRAME_RATE**2)

        # 应用mask
        if has_object_mask is not None:
            if has_object_mask.dim() > 1:
                has_object_mask = has_object_mask.view(bs)
            mask = has_object_mask.to(dtype=dtype, device=device).view(bs, 1, 1)
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
            l_oe0 = l_oe0 * mask.squeeze(-1)
            r_oe0 = r_oe0 * mask.squeeze(-1)
            l_lb0 = (l_lb0 * mask.squeeze(-1).unsqueeze(-1)).squeeze(-1)
            r_lb0 = (r_lb0 * mask.squeeze(-1).unsqueeze(-1)).squeeze(-1)
        else:
            l_lb0 = l_lb0.squeeze(-1)
            r_lb0 = r_lb0.squeeze(-1)

        results = {
            "pred_obj_trans": fused_pos,
            "gating_weights": weights,
            "gating_weights_raw": weights_raw,
            "pred_obj_vel_from_posdiff": vel_from_pos,
            "pred_obj_acc_from_posdiff": acc_from_pos,
            "obj_vel_input": obj_vel_input,
            "obj_vel_corrected": obj_vel_corrected,
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
            "gating_smoothing_applied": (not self.training) and self.gating_smoothing_enabled,
            "interaction_code": interaction_code,
            "interaction_prior_aux": interaction_prior_aux,
        }
        results.update(
            self._compute_human_refinement(
                human_pose_input,
                root_trans_input,
                fused_pos,
                obj_vel_corrected,
                weights,
                pred_hand_contact_prob,
                lhand_position,
                rhand_position,
                has_object_mask,
                enable_refine,
            )
        )
        return results

    @staticmethod
    def _slice_optional_time(value, start: int, end: int, batch_size: int, seq_len: int):
        return slice_time_value(value, start, end, batch_size, seq_len)

    def _inference_online_sequence(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        online_window: int,
        **kwargs,
    ):
        if hand_positions.dim() == 3:
            batch_size, seq_len, _ = hand_positions.shape
        elif hand_positions.dim() == 4:
            batch_size, seq_len = hand_positions.shape[:2]
        else:
            raise ValueError(f"Unexpected hand_positions shape {hand_positions.shape}")

        if seq_len <= online_window:
            return self.forward(
                hand_positions,
                pred_hand_contact_prob,
                obj_trans_init,
                **kwargs,
            )

        def _slice_kwargs(start: int, end: int):
            return {
                key: self._slice_optional_time(value, start, end, batch_size, seq_len)
                for key, value in kwargs.items()
                if key != "known_obj_trans_prefix"
            }

        warmup_len = int(online_window)
        warmup_out = self.forward(
            hand_positions[:, :warmup_len],
            pred_hand_contact_prob[:, :warmup_len],
            obj_trans_init,
            **_slice_kwargs(0, warmup_len),
        )
        history = warmup_out

        for end in range(warmup_len + 1, seq_len + 1):
            start = end - warmup_len
            prefix = select_time_context(history, start, end - 1).get("pred_obj_trans")
            if isinstance(prefix, torch.Tensor) and prefix.shape[1] > 0:
                step_obj_trans_init = prefix[:, 0]
            else:
                step_obj_trans_init = obj_trans_init
            window_out = self.forward(
                hand_positions[:, start:end],
                pred_hand_contact_prob[:, start:end],
                step_obj_trans_init,
                known_obj_trans_prefix=prefix,
                **_slice_kwargs(start, end),
            )
            latest = take_latest_frame(window_out, batch_size, end - start)
            history = concat_time_dicts([history, latest])

        return history

    def inference(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        inference_mode: str = "offline",
        online_window: int = None,
        online_state: dict = None,
        return_online_state: bool = False,
        **kwargs,
    ):
        mode = normalize_inference_mode(inference_mode)
        if mode == "offline":
            output = self.forward(
                hand_positions,
                pred_hand_contact_prob,
                obj_trans_init,
                **kwargs,
            )
            if return_online_state:
                return output, online_state or {}
            return output

        window = resolve_online_window(self.cfg, online_window)
        stream_dict = {
            "hand_positions": hand_positions,
            "pred_hand_contact_prob": pred_hand_contact_prob,
            **{key: value for key, value in kwargs.items() if key != "known_obj_trans_prefix"},
        }
        if isinstance(online_state, dict) and isinstance(online_state.get("data_dict"), dict):
            run_data, previous_len = append_stream_data(online_state["data_dict"], stream_dict, sequence_key="hand_positions")
            hand_positions = run_data["hand_positions"]
            pred_hand_contact_prob = run_data["pred_hand_contact_prob"]
            for key in stream_dict.keys():
                if key not in {"hand_positions", "pred_hand_contact_prob"}:
                    kwargs[key] = run_data.get(key)
        else:
            previous_len = 0
            run_data = stream_dict

        output = self._inference_online_sequence(
            hand_positions,
            pred_hand_contact_prob,
            obj_trans_init,
            window,
            **kwargs,
        )
        state = {"data_dict": run_data, "outputs": output}
        if return_online_state:
            if previous_len > 0:
                batch_size, seq_len = infer_batch_seq(run_data, key="hand_positions")
                output = slice_time_dict(output, previous_len, seq_len, batch_size, seq_len)
            return output, state
        return output

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        """返回空输出"""
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
            "init_lhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_rhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_lhand_lb": torch.zeros(batch_size, device=device),
            "init_rhand_lb": torch.zeros(batch_size, device=device),
            "gating_smoothing_applied": False,
        }
