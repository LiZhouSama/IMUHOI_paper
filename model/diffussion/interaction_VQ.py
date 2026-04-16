"""
VQ-based interaction diffusion modules.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PositionalEncoding, SinusoidalTimeEmbedding, _modulate, cosine_beta_schedule
from .base_cross import ConditionalCrossDiT
from .cond_VQ import CondVQModule
from .interaction import InteractionModule


class _DiscreteCrossBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.cond_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond_ctx: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(global_cond).chunk(9, dim=-1)

        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        q = _modulate(self.norm2(x), shift_cross, scale_cross)
        kv = self.cond_norm(cond_ctx)
        h, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = x + gate_cross.unsqueeze(1) * h

        h = _modulate(self.norm3(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class _DiscreteTokenDenoiser(nn.Module):
    def __init__(
        self,
        *,
        num_codebooks: int,
        codebook_size: int,
        cond_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.cond_dim = int(cond_dim)
        self.d_model = int(d_model)

        self.token_embeddings = nn.ModuleList([nn.Embedding(self.codebook_size, self.d_model) for _ in range(self.num_codebooks)])
        self.codebook_bias = nn.Parameter(torch.zeros(self.num_codebooks, self.d_model))
        self.cond_proj = nn.Linear(self.cond_dim, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=max_seq_len)
        self.cond_pos_encoding = PositionalEncoding(self.d_model, max_len=max_seq_len)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.d_model),
            nn.Linear(self.d_model, self.d_model * 4),
            nn.SiLU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )

        self.blocks = nn.ModuleList(
            [
                _DiscreteCrossBlock(self.d_model, nhead, mlp_ratio=4.0, dropout=dropout)
                for _ in range(max(int(num_layers), 1))
            ]
        )
        self.out_norm = nn.LayerNorm(self.d_model)
        self.heads = nn.ModuleList([nn.Linear(self.d_model, self.codebook_size) for _ in range(self.num_codebooks)])

    def forward(self, x_t: torch.Tensor, cond_seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.dim() != 3 or x_t.shape[-1] != self.num_codebooks:
            raise ValueError(f"x_t must be [B,L,{self.num_codebooks}], got {tuple(x_t.shape)}")
        if cond_seq.dim() != 3 or cond_seq.shape[0] != x_t.shape[0]:
            raise ValueError(f"cond_seq must be [B,Lc,{self.cond_dim}], got {tuple(cond_seq.shape)}")

        h = 0.0
        for codebook_idx, emb in enumerate(self.token_embeddings):
            h = h + emb(x_t[..., codebook_idx]) + self.codebook_bias[codebook_idx]
        h = h / float(max(self.num_codebooks, 1)) ** 0.5
        h = self.pos_encoding(h)

        cond_ctx = self.cond_pos_encoding(self.cond_proj(cond_seq))
        t_cond = self.time_embed(t) + cond_ctx.mean(dim=1)

        for blk in self.blocks:
            h = blk(h, cond_ctx, t_cond)
        h = self.out_norm(h)
        logits = torch.stack([head(h) for head in self.heads], dim=2)
        return logits


class ObsTokenDiffusionModule(nn.Module):
    """Stage-2 discrete diffusion over VQ token indices."""

    def __init__(self, cfg):
        super().__init__()
        self.tokenizer = CondVQModule(cfg)
        self.num_joints = int(getattr(cfg, "num_joints", 24))
        self.code_dim = int(getattr(cfg, "object_code_dim", 128))
        self.num_codebooks = int(getattr(cfg, "vq_num_codebooks", 4))
        self.codebook_size = int(getattr(cfg, "num_object_codes", getattr(cfg, "vq_codebook_size", 128)))
        self.downsample_stride = int(max(1, getattr(cfg, "vq_downsample_stride", 2)))
        self.obs_cond_dim = int(getattr(cfg, "vq_obs_cond_dim", getattr(cfg, "prior_encoder_hidden_dim", 256)))

        dit_cfg = getattr(cfg, "dit", {})

        def _dit_param(name: str, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        self.timesteps = int(getattr(cfg, "d3pm_timesteps", _dit_param("dit_timesteps", 1000)))
        self.sample_steps = getattr(cfg, "d3pm_sample_steps", None)
        self.sample_steps = int(self.sample_steps) if self.sample_steps is not None else None

        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float(), persistent=False)
        self.register_buffer("alphas", alphas.float(), persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float(), persistent=False)

        cond_in_dim = self.num_joints * 6 + 3 + 9
        self.condition_frame_encoder = nn.Sequential(
            nn.Linear(cond_in_dim, self.obs_cond_dim),
            nn.GELU(),
            nn.Linear(self.obs_cond_dim, self.obs_cond_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.obs_cond_dim,
            nhead=max(int(getattr(cfg, "vq_obs_nhead", 8)), 1),
            dim_feedforward=self.obs_cond_dim * 4,
            dropout=float(getattr(cfg, "vq_obs_dropout", 0.1)),
            activation="gelu",
            batch_first=True,
        )
        self.condition_temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=max(int(getattr(cfg, "vq_obs_layers", 2)), 1),
        )

        latent_max_len = int(max(_dit_param("dit_max_seq_len", 256), getattr(cfg, "train", {}).get("window", 60)))
        latent_max_len = max(latent_max_len, 8)
        latent_max_len = self.tokenizer.latent_length(latent_max_len)
        self.denoiser = _DiscreteTokenDenoiser(
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
            cond_dim=self.obs_cond_dim,
            d_model=int(getattr(cfg, "vq_obs_d_model", _dit_param("dit_d_model", 256))),
            nhead=int(getattr(cfg, "vq_obs_nhead", _dit_param("dit_nhead", 8))),
            num_layers=int(getattr(cfg, "vq_obs_layers", _dit_param("dit_num_layers", 6))),
            dropout=float(getattr(cfg, "vq_obs_dropout", _dit_param("dit_dropout", 0.1))),
            max_seq_len=latent_max_len,
        )

    @staticmethod
    def _prepare_has_object(value, batch_size: int, device: torch.device) -> torch.Tensor:
        if value is None:
            return torch.ones(batch_size, device=device, dtype=torch.bool)
        if isinstance(value, torch.Tensor):
            mask = value.to(device=device, dtype=torch.bool)
            if mask.dim() == 0:
                mask = mask.view(1)
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size)
            if mask.shape[0] != batch_size:
                mask = mask[:1].expand(batch_size)
            return mask
        return torch.as_tensor(value, device=device, dtype=torch.bool).view(-1)[:1].expand(batch_size)

    def _prepare_obj_imu(self, obj_imu, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not isinstance(obj_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, 9, device=device, dtype=dtype)
        out = obj_imu.to(device=device, dtype=dtype)
        if out.dim() == 4:
            out = out.reshape(batch_size, seq_len, -1)
        if out.dim() != 3:
            return torch.zeros(batch_size, seq_len, 9, device=device, dtype=dtype)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, -1, -1)
        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return torch.zeros(batch_size, seq_len, 9, device=device, dtype=dtype)
        if out.shape[-1] < 9:
            pad = torch.zeros(batch_size, seq_len, 9 - out.shape[-1], device=device, dtype=dtype)
            out = torch.cat([out, pad], dim=-1)
        elif out.shape[-1] > 9:
            out = out[..., :9]
        return out

    def _downsample_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.tokenizer._chunk_average(x)

    def _encode_condition_frames(self, human_pose6d: torch.Tensor, human_trans: torch.Tensor, obj_imu: torch.Tensor) -> torch.Tensor:
        if human_pose6d.dim() == 4:
            pose_flat = human_pose6d.reshape(human_pose6d.shape[0], human_pose6d.shape[1], -1)
        else:
            pose_flat = human_pose6d
        cond_frames = torch.cat([pose_flat, human_trans, obj_imu], dim=-1)
        cond_frames = self.condition_frame_encoder(cond_frames)
        cond_frames = self.condition_temporal_encoder(cond_frames)
        return self._downsample_sequence(cond_frames)

    def build_condition_sequence(self, human_pose6d: torch.Tensor, human_trans: torch.Tensor, obj_imu: torch.Tensor) -> torch.Tensor:
        return self._encode_condition_frames(human_pose6d, human_trans, obj_imu)

    def _run_tokenizer_eval(self, fn, *args, **kwargs):
        prev = self.tokenizer.training
        try:
            self.tokenizer.eval()
            with torch.no_grad():
                return fn(*args, **kwargs)
        finally:
            if prev:
                self.tokenizer.train()

    def freeze_tokenizer(self):
        self.tokenizer.eval()
        for param in self.tokenizer.parameters():
            param.requires_grad_(False)

    def freeze_all(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def soft_embed_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert denoiser logits to soft VQ embeddings via probability-weighted codebook lookup.
        logits: [B, L, num_codebooks, codebook_size]
        Returns: [B, L, code_dim]
        """
        probs = F.softmax(logits, dim=-1)
        parts = []
        for k, emb in enumerate(self.tokenizer.quantizer.codebooks):
            parts.append(torch.matmul(probs[:, :, k, :], emb.weight))
        return torch.cat(parts, dim=-1)

    def teacher_forced_forward(
        self,
        gt_indices: torch.Tensor,
        cond_seq: torch.Tensor,
        has_object: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """One-step teacher-forced denoising for joint training.
        gt_indices: [B, L, num_codebooks] from tokenizer
        cond_seq: [B, L_cond, obs_cond_dim] from condition encoder
        has_object: [B] bool
        Returns dict with soft_z_q, logits, target indices, timestep.
        """
        B, L = gt_indices.shape[:2]
        device = gt_indices.device
        t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)
        x_t = self.q_sample(gt_indices.detach(), t)
        logits = self.denoiser(x_t, cond_seq, t)
        soft_z_q = self.soft_embed_from_logits(logits)
        soft_z_q = soft_z_q * has_object.to(dtype=soft_z_q.dtype).view(B, 1, 1)
        return {
            "soft_z_q": soft_z_q,
            "logits": logits,
            "target": gt_indices,
            "t": t,
        }

    def tokenize_from_resolved(
        self,
        obj_points: torch.Tensor,
        human_pose6d: torch.Tensor,
        human_trans: torch.Tensor,
        obj_rot_mats: Optional[torch.Tensor] = None,
        obj_trans_world: Optional[torch.Tensor] = None,
        obj_scale: Optional[torch.Tensor] = None,
        differentiable: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if differentiable:
            # Direct call without no_grad — allows gradient flow through VQ
            return self.tokenizer.tokenize_from_resolved(
                obj_points, human_pose6d, human_trans,
                obj_rot_mats=obj_rot_mats,
                obj_trans_world=obj_trans_world,
                obj_scale=obj_scale,
            )
        return self._run_tokenizer_eval(
            self.tokenizer.tokenize_from_resolved,
            obj_points,
            human_pose6d,
            human_trans,
            obj_rot_mats=obj_rot_mats,
            obj_trans_world=obj_trans_world,
            obj_scale=obj_scale,
        )

    def _resolve_human_state_from_gt(self, gt_targets: Dict, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        pose6d = self.tokenizer._resolve_pose6d_from_gt(gt_targets, batch_size, seq_len, device, dtype)
        trans = self.tokenizer._resolve_trans(gt_targets, batch_size, seq_len, device, dtype)
        return pose6d, trans

    def _forward_probs_from_x0(self, x0_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat_x0 = x0_idx.reshape(-1)
        flat_t = t.view(-1, 1, 1).expand_as(x0_idx).reshape(-1)
        alpha_bar = self.alphas_cumprod[flat_t].to(device=x0_idx.device)
        base = ((1.0 - alpha_bar) / float(self.codebook_size)).unsqueeze(-1)
        probs = torch.full((flat_x0.shape[0], self.codebook_size), 0.0, device=x0_idx.device, dtype=torch.float32)
        probs = probs + base
        probs.scatter_add_(
            1,
            flat_x0.unsqueeze(-1),
            alpha_bar.unsqueeze(-1),
        )
        return probs.view(*x0_idx.shape, self.codebook_size)

    def q_sample(self, x0_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x0_idx.dtype != torch.long:
            x0_idx = x0_idx.long()
        keep_prob = self.alphas_cumprod[t].to(device=x0_idx.device, dtype=torch.float32).view(-1, 1, 1)
        keep_mask = torch.rand_like(x0_idx.float()) < keep_prob
        random_tokens = torch.randint(
            low=0,
            high=self.codebook_size,
            size=x0_idx.shape,
            device=x0_idx.device,
            dtype=torch.long,
        )
        return torch.where(keep_mask, x0_idx, random_tokens)

    def q_posterior(self, x0_idx: torch.Tensor, x_t_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat_x0 = x0_idx.reshape(-1)
        flat_xt = x_t_idx.reshape(-1)
        flat_t = t.view(-1, 1, 1).expand_as(x0_idx).reshape(-1)
        device = x0_idx.device

        posterior = torch.zeros(flat_x0.shape[0], self.codebook_size, device=device, dtype=torch.float32)
        mask_t0 = flat_t == 0
        if mask_t0.any():
            posterior[mask_t0] = F.one_hot(flat_x0[mask_t0], num_classes=self.codebook_size).float()

        mask_rest = ~mask_t0
        if mask_rest.any():
            t_rest = flat_t[mask_rest]
            x0_rest = flat_x0[mask_rest]
            xt_rest = flat_xt[mask_rest]
            alpha_t = self.alphas[t_rest].to(device=device, dtype=torch.float32)
            alpha_bar_prev = self.alphas_cumprod[(t_rest - 1).clamp_min(0)].to(device=device, dtype=torch.float32)
            base_prev = ((1.0 - alpha_bar_prev) / float(self.codebook_size)).unsqueeze(-1)
            base_step = ((1.0 - alpha_t) / float(self.codebook_size)).unsqueeze(-1)

            q_prev = base_prev.expand(-1, self.codebook_size).clone()
            q_prev.scatter_add_(1, x0_rest.unsqueeze(-1), alpha_bar_prev.unsqueeze(-1))

            q_trans = base_step.expand(-1, self.codebook_size).clone()
            q_trans.scatter_add_(1, xt_rest.unsqueeze(-1), alpha_t.unsqueeze(-1))

            post = q_prev * q_trans
            post = post / post.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            posterior[mask_rest] = post

        return posterior.view(*x0_idx.shape, self.codebook_size)

    def p_posterior(self, p_x0: torch.Tensor, x_t_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat_p = p_x0.reshape(-1, self.codebook_size)
        flat_xt = x_t_idx.reshape(-1)
        flat_t = t.view(-1, 1, 1).expand_as(x_t_idx).reshape(-1)
        device = p_x0.device
        posterior = torch.zeros_like(flat_p)

        mask_t0 = flat_t == 0
        if mask_t0.any():
            posterior[mask_t0] = flat_p[mask_t0]

        mask_rest = ~mask_t0
        if mask_rest.any():
            t_rest = flat_t[mask_rest]
            p_rest = flat_p[mask_rest]
            xt_rest = flat_xt[mask_rest]
            alpha_t = self.alphas[t_rest].to(device=device, dtype=torch.float32)
            alpha_bar_prev = self.alphas_cumprod[(t_rest - 1).clamp_min(0)].to(device=device, dtype=torch.float32)
            base_prev = (1.0 - alpha_bar_prev) / float(self.codebook_size)
            base_step = (1.0 - alpha_t) / float(self.codebook_size)
            trans_same = alpha_t + base_step
            trans_diff = base_step
            denom_same = base_prev + alpha_bar_prev * trans_same
            denom_diff = base_prev + alpha_bar_prev * trans_diff

            p_obs = p_rest.gather(1, xt_rest.unsqueeze(-1)).squeeze(-1)
            common_diff = base_prev * (p_obs / denom_same + (1.0 - p_obs) / denom_diff)
            sum_all = common_diff.unsqueeze(-1) + p_rest * (alpha_bar_prev / denom_diff).unsqueeze(-1)
            post = trans_diff.unsqueeze(-1) * sum_all

            same_val = trans_same * (
                p_obs * (base_prev + alpha_bar_prev) / denom_same + base_prev * (1.0 - p_obs) / denom_diff
            )
            post.scatter_(1, xt_rest.unsqueeze(-1), same_val.unsqueeze(-1))
            post = post / post.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            posterior[mask_rest] = post

        return posterior.view(*x_t_idx.shape, self.codebook_size)

    def _build_sampling_timesteps(self, steps: Optional[int] = None) -> torch.Tensor:
        if steps is None or steps >= self.timesteps:
            return torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long)
        steps = int(steps)
        t = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.float64)
        t = torch.round(t).to(torch.long)
        t = torch.unique_consecutive(t)
        if t[-1].item() != 0:
            t = torch.cat([t, torch.zeros(1, dtype=torch.long)], dim=0)
        return t

    def _sample_categorical(self, probs: torch.Tensor) -> torch.Tensor:
        flat = probs.reshape(-1, probs.shape[-1]).clamp_min(1e-8)
        flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
        return samples.view(*probs.shape[:-1])

    def _sample_from_condition(self, cond_seq: torch.Tensor, has_object: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        batch_size, latent_len = cond_seq.shape[:2]
        device = cond_seq.device
        x_t = torch.randint(
            low=0,
            high=self.codebook_size,
            size=(batch_size, latent_len, self.num_codebooks),
            device=device,
            dtype=torch.long,
        )
        schedule = self._build_sampling_timesteps(steps)

        for t_cur in schedule.tolist():
            t_batch = torch.full((batch_size,), int(t_cur), device=device, dtype=torch.long)
            logits = self.denoiser(x_t, cond_seq, t_batch)
            probs_x0 = torch.softmax(logits, dim=-1)
            if t_cur == 0:
                x_t = torch.argmax(probs_x0, dim=-1)
            else:
                probs_prev = self.p_posterior(probs_x0, x_t, t_batch)
                x_t = self._sample_categorical(probs_prev)

        if isinstance(has_object, torch.Tensor):
            x_t = torch.where(
                has_object.view(batch_size, 1, 1),
                x_t,
                torch.zeros_like(x_t),
            )
        return x_t

    @torch.no_grad()
    def sample_from_resolved(
        self,
        human_pose6d: torch.Tensor,
        human_trans: torch.Tensor,
        obj_imu: torch.Tensor,
        has_object: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        if human_pose6d.dim() == 4:
            batch_size, seq_len = human_pose6d.shape[:2]
            dtype = human_pose6d.dtype
            device = human_pose6d.device
        else:
            raise ValueError(f"human_pose6d must be [B,T,J,6], got {tuple(human_pose6d.shape)}")

        obj_imu = self._prepare_obj_imu(obj_imu, batch_size, seq_len, device, dtype)
        has_object = self._prepare_has_object(has_object, batch_size, device)
        cond_seq = self.build_condition_sequence(human_pose6d, human_trans, obj_imu)
        cond_seq = cond_seq * has_object.to(device=device, dtype=dtype).view(batch_size, 1, 1)

        prev = self.training
        try:
            self.eval()
            return self._sample_from_condition(cond_seq, has_object, steps=steps if steps is not None else self.sample_steps)
        finally:
            if prev:
                self.train()

    def forward(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict] = None,
        human_pose6d: Optional[torch.Tensor] = None,
        human_trans: Optional[torch.Tensor] = None,
    ):
        if gt_targets is None and (human_pose6d is None or human_trans is None):
            raise ValueError("ObsTokenDiffusionModule.forward requires gt_targets or resolved human state")

        if human_pose6d is None or human_trans is None:
            ref = gt_targets.get("trans")
            if not isinstance(ref, torch.Tensor):
                raise ValueError("gt_targets['trans'] is required")
            if ref.dim() == 2:
                ref = ref.unsqueeze(0)
            batch_size, seq_len = ref.shape[:2]
            device, dtype = self.tokenizer._infer_runtime_context(data_dict, gt_targets)
            human_pose6d, human_trans = self._resolve_human_state_from_gt(gt_targets, batch_size, seq_len, device, dtype)
        else:
            batch_size, seq_len = human_pose6d.shape[:2]
            device = human_pose6d.device
            dtype = human_pose6d.dtype

        obj_imu = self._prepare_obj_imu(data_dict.get("obj_imu"), batch_size, seq_len, device, dtype)
        has_object = self._prepare_has_object(data_dict.get("has_object"), batch_size, device)
        obj_points = self.tokenizer._prepare_obj_points(data_dict.get("obj_points_canonical"), batch_size, device, dtype)

        # Extract object transforms for world-space tokenization
        obj_rot_mats = None
        obj_trans_world = None
        obj_scale = None
        if isinstance(gt_targets, dict):
            obj_rot_mats = self.tokenizer._resolve_obj_rot_mats(
                gt_targets.get("obj_rot"), batch_size, seq_len, device, dtype,
            )
            obj_trans_raw = gt_targets.get("obj_trans")
            if isinstance(obj_trans_raw, torch.Tensor):
                obj_trans_world = obj_trans_raw.to(device=device, dtype=dtype)
                if obj_trans_world.dim() == 2:
                    obj_trans_world = obj_trans_world.unsqueeze(0)
                if obj_trans_world.shape[0] == 1 and batch_size > 1:
                    obj_trans_world = obj_trans_world.expand(batch_size, -1, -1)
                if obj_trans_world.shape[:2] != (batch_size, seq_len) or obj_trans_world.shape[-1] != 3:
                    obj_trans_world = None
            obj_scale_raw = gt_targets.get("obj_scale")
            if isinstance(obj_scale_raw, torch.Tensor):
                obj_scale = obj_scale_raw.to(device=device, dtype=dtype)
                if obj_scale.dim() == 1:
                    obj_scale = obj_scale.unsqueeze(0)
                if obj_scale.shape[0] == 1 and batch_size > 1:
                    obj_scale = obj_scale.expand(batch_size, -1)
                if obj_scale.shape[:2] != (batch_size, seq_len):
                    obj_scale = None

        tokenized = self.tokenize_from_resolved(
            obj_points, human_pose6d, human_trans,
            obj_rot_mats=obj_rot_mats,
            obj_trans_world=obj_trans_world,
            obj_scale=obj_scale,
        )
        x0_idx = tokenized["indices"].detach()
        cond_seq = self.build_condition_sequence(human_pose6d, human_trans, obj_imu)
        cond_seq = cond_seq * has_object.to(device=device, dtype=dtype).view(batch_size, 1, 1)

        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        x_t = self.q_sample(x0_idx, t)
        logits = self.denoiser(x_t, cond_seq, t)
        probs_x0 = torch.softmax(logits, dim=-1)
        q_posterior = self.q_posterior(x0_idx, x_t, t)
        p_posterior = self.p_posterior(probs_x0, x_t, t)

        return {
            "code_logits": logits,
            "code_probs": probs_x0,
            "code_target": x0_idx,
            "x_t_indices": x_t,
            "t": t,
            "q_posterior": q_posterior,
            "p_posterior": p_posterior,
            "sample_has_object": has_object,
            "cond_seq": cond_seq,
            "tokenizer_aux": {
                "perplexity": tokenized["perplexity"],
                "z_q": tokenized["z_q_detached"],
            },
        }

    @torch.no_grad()
    def inference(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict] = None,
        human_pose6d: Optional[torch.Tensor] = None,
        human_trans: Optional[torch.Tensor] = None,
    ):
        if human_pose6d is None or human_trans is None:
            if gt_targets is None:
                raise ValueError("ObsTokenDiffusionModule.inference requires gt_targets or resolved human state")
            ref = gt_targets.get("trans")
            if not isinstance(ref, torch.Tensor):
                raise ValueError("gt_targets['trans'] is required")
            if ref.dim() == 2:
                ref = ref.unsqueeze(0)
            batch_size, seq_len = ref.shape[:2]
            device, dtype = self.tokenizer._infer_runtime_context(data_dict, gt_targets)
            human_pose6d, human_trans = self._resolve_human_state_from_gt(gt_targets, batch_size, seq_len, device, dtype)
        obj_imu = self._prepare_obj_imu(data_dict.get("obj_imu"), human_pose6d.shape[0], human_pose6d.shape[1], human_pose6d.device, human_pose6d.dtype)
        has_object = self._prepare_has_object(data_dict.get("has_object"), human_pose6d.shape[0], human_pose6d.device)
        indices = self.sample_from_resolved(human_pose6d, human_trans, obj_imu, has_object=has_object)
        return {"sampled_indices": indices, "sample_has_object": has_object}


class InteractionVQModule(InteractionModule):
    """Stage-3 interaction diffusion conditioned on VQ token embeddings."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.interaction_variant = "vq"
        self.obs_token_diffusion = ObsTokenDiffusionModule(cfg)
        self.vq_cond_mode_probs = self.cond_mode_probs

        del self.mesh_encoder
        del self.obs_encoder
        del self.null_object_code

        dit_cfg = getattr(cfg, "dit", {})

        def _dit_param(name: str, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        train_cfg = getattr(cfg, "train", {})
        test_cfg = getattr(cfg, "test", {})
        train_window = train_cfg.get("window") if isinstance(train_cfg, dict) else getattr(train_cfg, "window", None)
        test_window = test_cfg.get("window") if isinstance(test_cfg, dict) else getattr(test_cfg, "window", None)
        max_seq_len = int(_dit_param("dit_max_seq_len", max(self.window_size, 256)))
        # Visualization/eval may run on full sequences that are longer than train/test windows.
        # Size cond positional encodings from the largest sequence length the DiT is configured to accept.
        max_cond_source_len = max(
            int(max_seq_len),
            int(test_window or 0),
            int(train_window or 0),
            int(self.window_size),
        )
        max_cond_len = self.obs_token_diffusion.tokenizer.latent_length(max_cond_source_len)

        self.dit = ConditionalCrossDiT(
            target_dim=self.target_dim,
            cond_dim=self.code_dim,
            d_model=_dit_param("dit_d_model", 256),
            nhead=_dit_param("dit_nhead", 8),
            num_layers=_dit_param("dit_num_layers", 6),
            dim_feedforward=_dit_param("dit_dim_feedforward", 1024),
            dropout=_dit_param("dit_dropout", 0.1),
            max_seq_len=max_seq_len,
            max_cond_len=max(max_cond_len, 8),
            timesteps=_dit_param("dit_timesteps", 1000),
            use_time_embed=_dit_param("dit_use_time_embed", True),
            prediction_type=str(_dit_param("dit_prediction_type", "x0")).lower(),
        )

    def _select_condition_mode(self, batch_size: int, device: torch.device) -> torch.Tensor:
        probs = self.vq_cond_mode_probs.to(device=device)
        return torch.multinomial(probs, num_samples=batch_size, replacement=True)

    def _resolve_vq_human_state(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict],
        feats: Dict[str, torch.Tensor],
        *,
        use_gt: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_gt and isinstance(gt_targets, dict):
            ref = gt_targets.get("trans")
            if isinstance(ref, torch.Tensor):
                if ref.dim() == 2:
                    ref = ref.unsqueeze(0)
                batch_size, seq_len = ref.shape[:2]
                device, dtype = self.obs_token_diffusion.tokenizer._infer_runtime_context(data_dict, gt_targets)
                pose6d = self.obs_token_diffusion.tokenizer._resolve_pose6d_from_gt(gt_targets, batch_size, seq_len, device, dtype)
                trans = self.obs_token_diffusion.tokenizer._resolve_trans(gt_targets, batch_size, seq_len, device, dtype)
                return pose6d, trans
        return feats["human_pose6d"], feats["human_trans"]

    def _resolve_obj_transforms(
        self,
        gt_targets: Optional[Dict],
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract object rotation matrices, translation, and scale from gt_targets."""
        if not isinstance(gt_targets, dict):
            return None, None, None
        tokenizer = self.obs_token_diffusion.tokenizer

        obj_rot_mats = tokenizer._resolve_obj_rot_mats(
            gt_targets.get("obj_rot"), batch_size, seq_len, device, dtype,
        )
        obj_trans_world = None
        obj_trans_raw = gt_targets.get("obj_trans")
        if isinstance(obj_trans_raw, torch.Tensor):
            obj_trans_world = obj_trans_raw.to(device=device, dtype=dtype)
            if obj_trans_world.dim() == 2:
                obj_trans_world = obj_trans_world.unsqueeze(0)
            if obj_trans_world.shape[0] == 1 and batch_size > 1:
                obj_trans_world = obj_trans_world.expand(batch_size, -1, -1)
            if obj_trans_world.shape[:2] != (batch_size, seq_len) or obj_trans_world.shape[-1] != 3:
                obj_trans_world = None

        obj_scale = None
        obj_scale_raw = gt_targets.get("obj_scale")
        if isinstance(obj_scale_raw, torch.Tensor):
            obj_scale = obj_scale_raw.to(device=device, dtype=dtype)
            if obj_scale.dim() == 1:
                obj_scale = obj_scale.unsqueeze(0)
            if obj_scale.shape[0] == 1 and batch_size > 1:
                obj_scale = obj_scale.expand(batch_size, -1)
            if obj_scale.shape[:2] != (batch_size, seq_len):
                obj_scale = None

        return obj_rot_mats, obj_trans_world, obj_scale

    def _build_object_condition(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict],
        feats: Dict[str, torch.Tensor],
        context: Dict,
        *,
        training: bool,
        joint_training: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = int(context["batch_size"])
        seq_len = int(context["seq_len"])
        device = context["device"]
        dtype = context["dtype"]
        tokenizer = self.obs_token_diffusion.tokenizer

        has_object_mask = context.get("has_object_mask")
        if isinstance(has_object_mask, torch.Tensor):
            sample_has_object = has_object_mask.to(device=device, dtype=torch.bool).any(dim=1)
        else:
            sample_has_object = torch.ones(batch_size, device=device, dtype=torch.bool)

        human_pose_for_vq, human_trans_for_vq = self._resolve_vq_human_state(
            data_dict,
            gt_targets,
            feats,
            use_gt=bool(training and gt_targets is not None),
        )
        obj_points = tokenizer._prepare_obj_points(data_dict.get("obj_points_canonical"), batch_size, device, dtype)
        latent_len = tokenizer.latent_length(seq_len)
        cond = torch.zeros(batch_size, latent_len, self.code_dim, device=device, dtype=dtype)
        null_cond = cond.clone()
        direct_indices = torch.zeros(batch_size, latent_len, tokenizer.num_codebooks, device=device, dtype=torch.long)
        obs_indices = torch.zeros_like(direct_indices)

        # Resolve object transforms for world-space tokenization
        obj_rot_mats, obj_trans_world, obj_scale = self._resolve_obj_transforms(
            gt_targets, batch_size, seq_len, device, dtype,
        )

        if training:
            mode = self._select_condition_mode(batch_size, device)
            mode = torch.where(sample_has_object, mode, torch.full_like(mode, 2))
            need_direct = bool((mode == 0).any().item())
            need_obs = bool((mode == 1).any().item())
        else:
            mode = torch.where(
                sample_has_object,
                torch.ones(batch_size, device=device, dtype=torch.long),
                torch.full((batch_size,), 2, device=device, dtype=torch.long),
            )
            need_direct = False
            need_obs = bool(sample_has_object.any().item())

        # For joint training, store auxiliary outputs for combined loss
        joint_vq_aux = {}
        joint_obs_aux = {}

        if joint_training and training:
            # --- Joint training path: differentiable throughout ---
            # Tokenize with gradients (straight-through VQ)
            direct_tokenized = self.obs_token_diffusion.tokenize_from_resolved(
                obj_points, human_pose_for_vq, human_trans_for_vq,
                obj_rot_mats=obj_rot_mats,
                obj_trans_world=obj_trans_world,
                obj_scale=obj_scale,
                differentiable=True,
            )
            direct_indices = direct_tokenized["indices"]
            z_q_st = direct_tokenized["z_q"]  # straight-through, HAS gradients
            z_q_detached = direct_tokenized["z_q_detached"]
            joint_vq_aux = {
                "codebook_loss": direct_tokenized["codebook_loss"],
                "commit_loss": direct_tokenized["commit_loss"],
                "perplexity": direct_tokenized["perplexity"],
            }

            if need_direct:
                cond[mode == 0] = z_q_st.to(dtype=dtype)[mode == 0]

            if need_obs:
                # Teacher-forced obs path: differentiable soft embeddings
                obs_cond_seq = self.obs_token_diffusion.build_condition_sequence(
                    human_pose_for_vq, human_trans_for_vq, feats["obj_imu"],
                )
                obs_cond_seq = obs_cond_seq * sample_has_object.to(dtype=dtype).view(batch_size, 1, 1)
                tf_result = self.obs_token_diffusion.teacher_forced_forward(
                    direct_indices, obs_cond_seq, sample_has_object,
                )
                cond[mode == 1] = tf_result["soft_z_q"].to(dtype=dtype)[mode == 1]
                obs_indices = direct_indices  # GT indices for reference
                joint_obs_aux = {
                    "logits": tf_result["logits"],
                    "target": tf_result["target"],
                    "t": tf_result["t"],
                    "sample_has_object": sample_has_object,
                }
        else:
            # --- Original non-joint path ---
            if need_direct:
                direct_tokenized = self.obs_token_diffusion.tokenize_from_resolved(
                    obj_points, human_pose_for_vq, human_trans_for_vq,
                    obj_rot_mats=obj_rot_mats,
                    obj_trans_world=obj_trans_world,
                    obj_scale=obj_scale,
                )
                direct_indices = direct_tokenized["indices"]
                direct_cond = direct_tokenized["z_q_detached"].to(device=device, dtype=dtype)
                cond[mode == 0] = direct_cond[mode == 0]

            if need_obs:
                obs_indices = self.obs_token_diffusion.sample_from_resolved(
                    human_pose_for_vq,
                    human_trans_for_vq,
                    feats["obj_imu"],
                    has_object=sample_has_object,
                    steps=None,
                )
                obs_cond = tokenizer.lookup_codes(obs_indices).to(device=device, dtype=dtype)
                cond[mode == 1] = obs_cond[mode == 1]

        cond[mode == 2] = null_cond[mode == 2]
        cond_info = {
            "mode": mode,
            "cond": cond,
            "sample_has_object": sample_has_object,
            "direct_indices": direct_indices,
            "obs_indices": obs_indices,
            "latent_len": torch.tensor(latent_len, device=device, dtype=torch.long),
            "joint_vq_aux": joint_vq_aux,
            "joint_obs_aux": joint_obs_aux,
        }
        return cond, cond_info

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
        batch_size, seq_len, feat_dim = observed_seq.shape
        device = observed_seq.device
        dtype = observed_seq.dtype

        if cond_seq.dim() != 3 or cond_seq.shape[0] != batch_size or cond_seq.shape[2] != self.code_dim:
            raise ValueError(f"cond_seq shape mismatch, got {tuple(cond_seq.shape)}")

        window = self.window_size
        unknown_idx = torch.nonzero(self.unknown_dim_mask.to(device=device), as_tuple=False).flatten()
        unknown_dim = int(unknown_idx.numel())

        history = torch.zeros(batch_size, 0, feat_dim, device=device, dtype=dtype)
        if isinstance(warmup_seq, torch.Tensor):
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
                    pad_src = hist[:, :1] if hist_len > 0 else current.unsqueeze(1)
                    hist = torch.cat([pad_src.expand(batch_size, pad_count, feat_dim), hist], dim=1)
                x_input = torch.cat([hist, current.unsqueeze(1)], dim=1)
            else:
                x_input = current.unsqueeze(1)

            inpaint_mask = torch.ones_like(x_input, dtype=torch.bool)
            if unknown_dim > 0:
                inpaint_mask[:, -1, unknown_idx] = False

            if inference_type == "sample_inpaint":
                x_out = self.dit.sample_inpaint(
                    x_input=x_input,
                    inpaint_mask=inpaint_mask,
                    cond=cond_seq,
                    x_start=x_input,
                    steps=steps,
                    sampler=sampler,
                    eta=eta,
                )
            else:
                x_out = self.dit.sample_inpaint_x0(
                    x_input=x_input,
                    inpaint_mask=inpaint_mask,
                    cond=cond_seq,
                    steps=steps,
                )

            current_pred = x_out[:, -1]
            if unknown_dim > 0:
                last_unknown = current_pred[:, unknown_idx]
                current[:, unknown_idx] = last_unknown
            history = torch.cat([history, current.unsqueeze(1)], dim=1)
        return history

    def forward(self, data_dict: Dict, hp_out: Optional[Dict] = None, gt_targets: Optional[Dict] = None, joint_training: bool = False):
        if gt_targets is None:
            raise ValueError("InteractionVQModule forward requires gt_targets for training")

        x_clean, context = self._build_clean_window(data_dict, hp_out, gt_targets)
        feats = self._build_features(
            data_dict,
            hp_out,
            gt_targets,
            batch_size=context["batch_size"],
            seq_len=context["seq_len"],
            device=context["device"],
            dtype=context["dtype"],
        )
        cond_seq, cond_info = self._build_object_condition(
            data_dict,
            gt_targets,
            feats,
            context,
            training=bool(self.training),
            joint_training=joint_training,
        )

        x0_pred, aux = self.dit(
            cond=cond_seq,
            x_start=x_clean,
            add_noise=bool(self.training and self.use_diffusion_noise),
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
            "cond_mode": cond_info["mode"],
        }
        return outputs

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
        cond_seq, cond_info = self._build_object_condition(
            data_dict,
            gt_targets,
            feats,
            context,
            training=False,
        )

        steps = self.inference_steps if sample_steps is None else int(sample_steps)
        if steps is None:
            steps = self.dit.timesteps
        sampler_name = self.inference_sampler if sampler is None else str(sampler).lower()
        eta_val = self.inference_eta if eta is None else float(eta)
        eta_val = max(eta_val, 0.0)

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
            steps=int(steps),
            inference_type=self.inference_type,
            sampler=sampler_name,
            eta=eta_val,
            warmup_seq=warmup_seq,
        )
        outputs = self._decode_outputs(pred_seq, context)
        outputs["object_prior_aux"] = cond_info
        outputs["diffusion_aux"] = {
            "x0_pred": pred_seq,
            "prediction_type": self.dit.prediction_type,
            "observed_dim_mask": self.observed_dim_mask,
            "unknown_dim_mask": self.unknown_dim_mask,
            "cond_mode": cond_info["mode"],
            "sampler": sampler_name,
            "eta": eta_val,
            "sample_steps": int(steps),
            "warmup_len": warmup_len,
        }
        return outputs
