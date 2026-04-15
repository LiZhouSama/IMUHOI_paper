"""
Cross-attention DiT backbone that supports different sequence lengths for x and cond.

This module mirrors the public API of `ConditionalDiT`, but the condition sequence is
allowed to have a different temporal length from the target sequence.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import (
    FinalLayer,
    PositionalEncoding,
    SinusoidalTimeEmbedding,
    _extract,
    _modulate,
    cosine_beta_schedule,
)


class CrossDiTBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and AdaLN-Zero modulation."""

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

    def forward(self, x: torch.Tensor, cond_ctx: Optional[torch.Tensor], global_cond: torch.Tensor) -> torch.Tensor:
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

        if cond_ctx is not None and cond_ctx.numel() > 0:
            q = _modulate(self.norm2(x), shift_cross, scale_cross)
            kv = self.cond_norm(cond_ctx)
            h, _ = self.cross_attn(q, kv, kv, need_weights=False)
            x = x + gate_cross.unsqueeze(1) * h

        h = _modulate(self.norm3(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class ConditionalCrossDiT(nn.Module):
    """Conditional DiT backbone with cross-attention conditioning."""

    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        *,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        max_cond_len: int = 256,
        timesteps: int = 1000,
        use_time_embed: bool = True,
        prediction_type: str = "eps",
    ):
        super().__init__()
        self.target_dim = int(target_dim)
        self.cond_dim = int(cond_dim)
        self.timesteps = int(timesteps)
        self.use_time_embed = bool(use_time_embed)
        self.d_model = int(d_model)
        self.prediction_type = str(prediction_type).lower()
        if self.prediction_type not in {"eps", "x0"}:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        betas = cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.float(), persistent=False)
        self.register_buffer("alphas", alphas.float(), persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float(), persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float(), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float(), persistent=False)

        self.input_proj = nn.Linear(self.target_dim, self.d_model)
        self.cond_proj = nn.Linear(self.cond_dim, self.d_model) if self.cond_dim > 0 else None
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=max_seq_len)
        self.cond_pos_encoding = PositionalEncoding(self.d_model, max_len=max_cond_len)

        if self.use_time_embed:
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(self.d_model),
                nn.Linear(self.d_model, self.d_model * 4),
                nn.SiLU(),
                nn.Linear(self.d_model * 4, self.d_model),
            )
        else:
            self.time_embed = None

        mlp_ratio = float(dim_feedforward) / float(max(self.d_model, 1))
        self.blocks = nn.ModuleList(
            [CrossDiTBlock(self.d_model, nhead, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(max(num_layers, 1))]
        )
        self.final_layer = FinalLayer(self.d_model, self.target_dim)

    def _normalize_cond(self, cond: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if self.cond_dim == 0:
            if cond is None:
                return torch.zeros(ref.shape[0], 0, 0, device=ref.device, dtype=ref.dtype)
            if cond.dim() != 3:
                raise ValueError(f"cond must be [B,L,0], got {tuple(cond.shape)}")
            if cond.shape[0] != ref.shape[0] or cond.shape[2] != 0:
                raise ValueError(f"cond shape mismatch for cond_dim=0, got {tuple(cond.shape)}, ref={tuple(ref.shape)}")
            return cond.to(device=ref.device, dtype=ref.dtype)

        if cond is None:
            raise ValueError("cond is required when cond_dim > 0")
        if cond.dim() != 3:
            raise ValueError(f"cond must be [B,L,C], got {tuple(cond.shape)}")
        if cond.shape[0] != ref.shape[0] or cond.shape[2] != self.cond_dim:
            raise ValueError(
                f"cond shape mismatch: expected batch={ref.shape[0]}, cond_dim={self.cond_dim}, got {tuple(cond.shape)}"
            )
        return cond.to(device=ref.device, dtype=ref.dtype)

    def _encode(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_t)
        h = self.pos_encoding(h)

        cond_ctx = None
        if self.cond_proj is not None:
            cond_ctx = self.cond_proj(cond)
            cond_ctx = self.cond_pos_encoding(cond_ctx)

        if self.use_time_embed and self.time_embed is not None:
            t_cond = self.time_embed(t)
        else:
            t_cond = torch.zeros(x_t.shape[0], self.d_model, device=x_t.device, dtype=x_t.dtype)

        if cond_ctx is not None and cond_ctx.numel() > 0:
            t_cond = t_cond + cond_ctx.mean(dim=1)

        for blk in self.blocks:
            h = blk(h, cond_ctx, t_cond)
        return self.final_layer(h, t_cond)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        return _extract(self.sqrt_alphas_cumprod, t, x0) * x0 + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_omab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        return (x_t - sqrt_omab * eps) / sqrt_ab.clamp_min(1e-8)

    def predict_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_omab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        return (x_t - sqrt_ab * x0) / sqrt_omab.clamp_min(1e-8)

    def predict(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond_use = self._normalize_cond(cond, x_t)
        model_out = self._encode(x_t, cond_use, t)
        if self.prediction_type == "eps":
            eps_pred = model_out
            x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
        else:
            x0_pred = model_out
            eps_pred = self.predict_eps_from_x0(x_t, t, x0_pred)
        return x0_pred, eps_pred, model_out

    def forward(
        self,
        cond: Optional[torch.Tensor] = None,
        x_start: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        add_noise: bool = True,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if x_start is None and cond is None:
            raise ValueError("Either x_start or cond must be provided")

        if x_start is None:
            assert cond is not None
            if cond.dim() != 3:
                raise ValueError(f"cond must be [B,L,C], got {tuple(cond.shape)}")
            batch = cond.shape[0]
            x_start = torch.zeros(batch, 1, self.target_dim, device=cond.device, dtype=cond.dtype)

        batch = x_start.shape[0]
        device = x_start.device

        if add_noise:
            if t is None:
                t = torch.randint(0, self.timesteps, (batch,), device=device, dtype=torch.long)
            if noise is None:
                noise = torch.randn_like(x_start)
            x_t = self.q_sample(x_start, t, noise=noise)
        else:
            if t is None:
                t = torch.zeros(batch, device=device, dtype=torch.long)
            if noise is None:
                noise = torch.zeros_like(x_start)
            x_t = x_start

        x0_pred, eps_pred, model_out = self.predict(x_t=x_t, t=t, cond=cond)
        aux = {
            "t": t,
            "x_t": x_t,
            "noise": noise,
            "eps_pred": eps_pred,
            "x0_target": x_start,
            "x0_pred": x0_pred,
            "model_out": model_out,
            "prediction_type": self.prediction_type,
        }
        return x0_pred, aux

    def _build_sampling_timesteps(self, steps: int) -> torch.Tensor:
        steps = int(steps)
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if steps >= self.timesteps:
            return torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long)

        t = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.float64)
        t = torch.round(t).to(torch.long)
        t = torch.unique_consecutive(t)
        if t[-1].item() != 0:
            t = torch.cat([t, torch.zeros(1, dtype=torch.long)], dim=0)
        return t

    def _ddim_update(
        self,
        x_t: torch.Tensor,
        x0_pred: torch.Tensor,
        eps_pred: torch.Tensor,
        t_cur: int,
        t_next: int,
        *,
        sampler: str,
        eta: float,
    ) -> torch.Tensor:
        if t_next < 0:
            return x0_pred

        alpha_t = self.alphas_cumprod[t_cur].to(device=x_t.device, dtype=x_t.dtype)
        alpha_next = self.alphas_cumprod[t_next].to(device=x_t.device, dtype=x_t.dtype)
        step_eta = 1.0 if sampler == "ddpm" else float(max(eta, 0.0))
        sigma = step_eta * torch.sqrt(
            ((1.0 - alpha_next) / (1.0 - alpha_t).clamp_min(1e-8))
            * (1.0 - (alpha_t / alpha_next).clamp(max=1.0))
        )
        sigma = sigma.clamp_min(0.0)
        coeff_eps = torch.sqrt((1.0 - alpha_next - sigma * sigma).clamp_min(0.0))
        mean = torch.sqrt(alpha_next) * x0_pred + coeff_eps * eps_pred
        if sigma.item() == 0.0:
            return mean
        return mean + sigma * torch.randn_like(x_t)

    def _sample_step(
        self,
        x_t: torch.Tensor,
        cond: Optional[torch.Tensor],
        t_cur: int,
        t_next: int,
        *,
        sampler: str,
        eta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = x_t.shape[0]
        t_batch = torch.full((batch,), t_cur, device=x_t.device, dtype=torch.long)
        x0_pred, eps_pred, _ = self.predict(x_t=x_t, t=t_batch, cond=cond)
        x_next = self._ddim_update(
            x_t,
            x0_pred,
            eps_pred,
            t_cur=t_cur,
            t_next=t_next,
            sampler=sampler,
            eta=eta,
        )
        return x_next, x0_pred, eps_pred

    def sample(
        self,
        cond: Optional[torch.Tensor] = None,
        x_start: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        sampler: str = "ddim",
        eta: float = 0.0,
    ) -> torch.Tensor:
        sampler = str(sampler).lower()
        if sampler not in {"ddim", "ddpm"}:
            raise ValueError(f"sampler must be 'ddim' or 'ddpm', got {sampler}")
        if cond is None and x_start is None:
            raise ValueError("sample() requires cond or x_start to infer shape")

        ref = x_start if x_start is not None else cond
        assert ref is not None
        if ref.dim() != 3:
            raise ValueError(f"ref tensor must be [B,T,C], got {tuple(ref.shape)}")

        batch = ref.shape[0]
        seq_len = x_start.shape[1] if x_start is not None else ref.shape[1]
        device = ref.device
        dtype = ref.dtype
        cond_use = self._normalize_cond(cond, x_start if x_start is not None else torch.zeros(batch, seq_len, self.target_dim, device=device, dtype=dtype))

        total_steps = self.timesteps if steps is None else int(steps)
        schedule = self._build_sampling_timesteps(total_steps)

        if x_start is None:
            x_t = torch.randn(batch, seq_len, self.target_dim, device=device, dtype=dtype)
        else:
            x_t = x_start.to(device=device, dtype=dtype)

        for idx, t_cur in enumerate(schedule.tolist()):
            t_next = schedule[idx + 1].item() if idx + 1 < len(schedule) else -1
            x_t, _, _ = self._sample_step(x_t, cond_use, t_cur=t_cur, t_next=t_next, sampler=sampler, eta=eta)
        return x_t

    def sample_inpaint_x0(
        self,
        x_input: torch.Tensor,
        inpaint_mask: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        if x_input.dim() != 3:
            raise ValueError(f"x_input must be [B,T,D], got {tuple(x_input.shape)}")
        if inpaint_mask.shape != x_input.shape:
            raise ValueError(f"inpaint_mask shape mismatch: {tuple(inpaint_mask.shape)} vs {tuple(x_input.shape)}")

        device = x_input.device
        dtype = x_input.dtype
        mask_bool = inpaint_mask.to(device=device, dtype=torch.bool)
        cond_use = self._normalize_cond(cond, x_input)
        total_steps = self.timesteps if steps is None else int(steps)
        schedule = self._build_sampling_timesteps(total_steps)

        batch = x_input.shape[0]
        x_t = x_input.to(device=device, dtype=dtype)
        for t_cur in schedule.tolist():
            t_batch = torch.full((batch,), int(t_cur), device=device, dtype=torch.long)
            x0_pred, _, _ = self.predict(x_t=x_t, t=t_batch, cond=cond_use)
            x_t = torch.where(mask_bool, x_input, x0_pred)
        return x_t

    def sample_inpaint(
        self,
        x_input: torch.Tensor,
        inpaint_mask: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        x_start: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        sampler: str = "ddim",
        eta: float = 0.0,
    ) -> torch.Tensor:
        if x_input.dim() != 3:
            raise ValueError(f"x_input must be [B,T,D], got {tuple(x_input.shape)}")
        if inpaint_mask.shape != x_input.shape:
            raise ValueError(f"inpaint_mask shape mismatch: {tuple(inpaint_mask.shape)} vs {tuple(x_input.shape)}")

        batch = x_input.shape[0]
        device = x_input.device
        dtype = x_input.dtype
        cond_use = self._normalize_cond(cond, x_input)
        mask_bool = inpaint_mask.to(device=device, dtype=torch.bool)

        total_steps = self.timesteps if steps is None else int(steps)
        schedule = self._build_sampling_timesteps(total_steps)
        sampler = str(sampler).lower()
        if sampler not in {"ddim", "ddpm"}:
            raise ValueError(f"sampler must be 'ddim' or 'ddpm', got {sampler}")

        x_input_use = x_input.to(device=device, dtype=dtype)
        base_start = x_input_use if x_start is None else x_start.to(device=device, dtype=dtype)
        known_noise = torch.randn_like(x_input_use)

        t_init = schedule[0].view(1).expand(batch).to(device=device)
        x_t = self.q_sample(base_start, t_init, noise=torch.randn_like(base_start))
        x_known = self.q_sample(x_input_use, t_init, noise=known_noise)
        x_t = torch.where(mask_bool, x_known, x_t)

        for idx, t_cur in enumerate(schedule.tolist()):
            t_next = schedule[idx + 1].item() if idx + 1 < len(schedule) else -1
            x_next, _, _ = self._sample_step(x_t, cond_use, t_cur=t_cur, t_next=t_next, sampler=sampler, eta=eta)
            if t_next >= 0:
                t_next_batch = torch.full((batch,), t_next, device=device, dtype=torch.long)
                x_known = self.q_sample(x_input_use, t_next_batch, noise=known_noise)
                x_t = torch.where(mask_bool, x_known, x_next)
            else:
                x_t = torch.where(mask_bool, x_input_use, x_next)
        return x_t
