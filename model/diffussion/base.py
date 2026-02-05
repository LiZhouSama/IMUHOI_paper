"""
Shared DiT (Diffusion Transformer) backbone utilities.

The ConditionalDiT module provides a lightweight diffusion-style
Transformer encoder over time that can optionally inject Gaussian noise
during training while remaining drop-in for supervised use.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from Nichol & Dhariwal 2021."""
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard 1D sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, dim) embedding
        """
        half_dim = self.dim // 2
        device = t.device
        emb_factor = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ConditionalDiT(nn.Module):
    """
    Diffusion-style Transformer encoder conditioned on per-frame features.

    - Accepts a noisy target sequence `x_start` (or zeros) and conditioning
      sequence `cond` of the same temporal length.
    - Optional timestep embeddings allow training with Gaussian noise while
      remaining usable as a deterministic transformer when `add_noise=False`.
    """

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
        timesteps: int = 1000,
        use_time_embed: bool = True,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.use_time_embed = use_time_embed

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas.float(), persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float(), persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float(), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float(), persistent=False)
        # posterior coefficients for sampling
        alpha_bar = alphas_cumprod
        alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=alpha_bar.dtype), alpha_bar[:-1]])
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_variance = posterior_variance.clamp(min=1e-5)
        posterior_log_variance = torch.log(posterior_variance)
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(1.0 - betas) / (1.0 - alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance.float(), persistent=False)
        self.register_buffer("posterior_log_variance", posterior_log_variance.float(), persistent=False)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1.float(), persistent=False)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2.float(), persistent=False)

        self.input_proj = nn.Linear(target_dim, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model) if cond_dim > 0 else None
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        self.time_embed = SinusoidalTimeEmbedding(d_model) if use_time_embed else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, target_dim)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(a_bar_t) * x0 + sqrt(1-a_bar_t) * eps."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

    def forward(
        self,
        cond: torch.Tensor,
        x_start: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        add_noise: bool = False,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            cond: (B, T, cond_dim) conditioning tokens
            x_start: (B, T, target_dim) clean targets or seeds (defaults to zeros)
            t: (B,) timesteps; auto-sampled when add_noise is True
            add_noise: if True, apply forward diffusion to x_start
        Returns:
            x0_pred: (B, T, target_dim)
            aux: dict containing timestep and the noisy input used
        """
        B, T, _ = cond.shape
        device = cond.device
        if x_start is None:
            x_start = torch.zeros(B, T, self.target_dim, device=device, dtype=cond.dtype)

        if add_noise:
            if t is None:
                t = torch.randint(0, self.timesteps, (B,), device=device, dtype=torch.long)
            x_t = self.q_sample(x_start, t, noise=noise)
        else:
            if t is None:
                t = torch.zeros(B, device=device, dtype=torch.long)
            x_t = x_start

        h = self.input_proj(x_t)
        if self.cond_proj is not None:
            h = h + self.cond_proj(cond)
        h = self.pos_encoding(h)
        if self.use_time_embed and self.time_embed is not None:
            t_emb = self.time_embed(t)
            h = h + t_emb.unsqueeze(1)

        h = self.transformer(h)
        x0_pred = self.output_proj(h)
        return x0_pred, {"t": t, "x_t": x_t}

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        x_start: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        DDPM-style sampling loop (predicting x0) for inference.

        Args:
            cond: (B, T, cond_dim)
            x_start: optional initial noise; defaults to N(0, I)
            steps: override total diffusion steps (defaults to self.timesteps)
        Returns:
            x: (B, T, target_dim) sampled sequence
        """
        B, T, _ = cond.shape
        device = cond.device
        total_steps = steps or self.timesteps
        if x_start is None:
            x = torch.randn(B, T, self.target_dim, device=device, dtype=cond.dtype)
        else:
            x = x_start

        for t_idx in reversed(range(total_steps)):
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
            x0_pred, _ = self.forward(cond, x_start=x, t=t_batch, add_noise=False)

            coef1 = self.posterior_mean_coef1[t_idx]
            coef2 = self.posterior_mean_coef2[t_idx]
            mean = coef1 * x0_pred + coef2 * x
            if t_idx == 0:
                x = mean
            else:
                noise = torch.randn_like(x)
                var = self.posterior_variance[t_idx]
                x = mean + var.sqrt() * noise

        return x
