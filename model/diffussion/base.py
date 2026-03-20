"""
Shared diffusion transformer backbone with AdaLN-Zero blocks.

This module supports two prediction modes:
- `eps`: model predicts epsilon noise (classic DDPM/DiT)
- `x0`: model predicts clean sample directly (DiffusionPoser-like)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule from Nichol & Dhariwal (2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


def _extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Extract per-batch coefficients and reshape to [B, 1, 1]."""
    out = a.gather(0, t)
    return out.view(-1, 1, 1).to(device=x.device, dtype=x.dtype)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb_factor = math.log(10000.0) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb_factor)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding over sequence length."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero modulation."""

    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # AdaLN-Zero: initialize modulation projection to zero.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)

        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    """Final AdaLN-Zero modulation + projection to target space."""

    def __init__(self, d_model: int, target_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.proj = nn.Linear(d_model, target_dim)

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.proj(x)


class ConditionalDiT(nn.Module):
    """Conditional DiT backbone with epsilon/x0 prediction."""

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
        prediction_type: str = "eps",
    ):
        super().__init__()
        self.target_dim = target_dim
        self.cond_dim = cond_dim
        self.timesteps = timesteps
        self.use_time_embed = use_time_embed
        self.d_model = d_model
        self.prediction_type = str(prediction_type).lower()
        if self.prediction_type not in {"eps", "x0"}:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.float(), persistent=False)
        self.register_buffer("alphas", alphas.float(), persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float(), persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float(), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float(), persistent=False)

        self.input_proj = nn.Linear(target_dim, d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model) if cond_dim > 0 else None
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        if use_time_embed:
            self.time_embed = nn.Sequential(
                SinusoidalTimeEmbedding(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.SiLU(),
                nn.Linear(d_model * 4, d_model),
            )
        else:
            self.time_embed = None

        mlp_ratio = float(dim_feedforward) / float(d_model)
        self.blocks = nn.ModuleList([DiTBlock(d_model, nhead, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)])
        self.final_layer = FinalLayer(d_model, target_dim)

    def _normalize_cond(self, cond: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if self.cond_dim == 0:
            if cond is None:
                return torch.zeros(ref.shape[0], ref.shape[1], 0, device=ref.device, dtype=ref.dtype)
            if cond.dim() != 3:
                raise ValueError(f"cond must be [B,T,0], got {cond.shape}")
            if cond.shape[0] != ref.shape[0] or cond.shape[1] != ref.shape[1] or cond.shape[2] != 0:
                raise ValueError(f"cond shape mismatch for cond_dim=0, got {cond.shape}, ref={ref.shape}")
            return cond.to(device=ref.device, dtype=ref.dtype)

        if cond is None:
            raise ValueError("cond is required when cond_dim > 0")
        if cond.dim() != 3:
            raise ValueError(f"cond must be [B,T,C], got {cond.shape}")
        if cond.shape[0] != ref.shape[0] or cond.shape[1] != ref.shape[1] or cond.shape[2] != self.cond_dim:
            raise ValueError(f"cond shape mismatch: expected [B,T,{self.cond_dim}], got {cond.shape}")
        return cond.to(device=ref.device, dtype=ref.dtype)

    def _encode(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_t)
        cond_token = None
        if self.cond_proj is not None:
            cond_token = self.cond_proj(cond)
            h = h + cond_token
        h = self.pos_encoding(h)

        if self.use_time_embed and self.time_embed is not None:
            t_cond = self.time_embed(t)
        else:
            t_cond = torch.zeros(x_t.shape[0], self.d_model, device=x_t.device, dtype=x_t.dtype)

        # Inject global condition into AdaLN modulation (in addition to token-level fusion).
        if cond_token is not None:
            t_cond = t_cond + cond_token.mean(dim=1)

        for blk in self.blocks:
            h = blk(h, t_cond)
        return self.final_layer(h, t_cond)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        return _extract(self.sqrt_alphas_cumprod, t, x0) * x0 + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Recover x_0 from x_t and epsilon."""
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_omab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        return (x_t - sqrt_omab * eps) / sqrt_ab.clamp_min(1e-8)

    def predict_eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Recover epsilon from x_t and x_0."""
        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_omab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        return (x_t - sqrt_ab * x0) / sqrt_omab.clamp_min(1e-8)

    def predict(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict x0/eps from noisy sample.

        Returns:
            x0_pred, eps_pred, model_out
        """
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
        """
        Returns:
            x0_pred: reconstructed x0
            aux: includes t, x_t, noise, eps_pred
        """
        if x_start is None and cond is None:
            raise ValueError("Either x_start or cond must be provided")

        if x_start is None:
            assert cond is not None
            if cond.dim() != 3:
                raise ValueError(f"cond must be [B,T,C], got {cond.shape}")
            batch, seq_len = cond.shape[:2]
            x_start = torch.zeros(batch, seq_len, self.target_dim, device=cond.device, dtype=cond.dtype)

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
        """Build descending timestep map from full schedule to sub-sampled schedule."""
        steps = int(steps)
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")

        if steps >= self.timesteps:
            return torch.arange(self.timesteps - 1, -1, -1, dtype=torch.long)

        # Uniformly sub-sample over [0, timesteps-1], keep descending order.
        t = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.float64)
        t = torch.round(t).to(torch.long)
        t = torch.unique_consecutive(t)

        # Ensure final step includes 0.
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

        if sampler == "ddpm":
            step_eta = 1.0
        else:
            step_eta = float(max(eta, 0.0))

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
        """One reverse step from t_cur -> t_next."""
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
        """
        Reverse sampling loop.

        Args:
            cond: [B, T, cond_dim] or None when cond_dim==0
            x_start: optional initial state, default N(0, I)
            steps: number of reverse steps (sub-sampled from full schedule)
            sampler: 'ddim' (deterministic if eta=0) or 'ddpm'
            eta: DDIM stochasticity (ignored by 'ddpm')
        """
        sampler = str(sampler).lower()
        if sampler not in {"ddim", "ddpm"}:
            raise ValueError(f"sampler must be 'ddim' or 'ddpm', got {sampler}")

        if cond is None and x_start is None:
            raise ValueError("sample() requires cond or x_start to infer [B,T]")

        ref = x_start if x_start is not None else cond
        assert ref is not None
        if ref.dim() != 3:
            raise ValueError(f"ref tensor must be [B,T,C], got {ref.shape}")

        batch, seq_len = ref.shape[:2]
        device = ref.device
        dtype = ref.dtype
        cond_use = self._normalize_cond(cond, ref if x_start is None else x_start)

        total_steps = self.timesteps if steps is None else int(steps)
        schedule = self._build_sampling_timesteps(total_steps)

        if x_start is None:
            x_t = torch.randn(batch, seq_len, self.target_dim, device=device, dtype=dtype)
        else:
            x_t = x_start.to(device=device, dtype=dtype)

        for idx, t_cur in enumerate(schedule.tolist()):
            t_next = schedule[idx + 1].item() if idx + 1 < len(schedule) else -1
            x_t, _, _ = self._sample_step(
                x_t,
                cond_use,
                t_cur=t_cur,
                t_next=t_next,
                sampler=sampler,
                eta=eta,
            )

        return x_t

    def sample_inpaint_x0(
        self,
        x_input: torch.Tensor,
        inpaint_mask: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Iterative x0 inpainting used by autoregressive sliding-window inference.

        Algorithm:
            x_t <- x_input
            for t in reverse_schedule:
                x0_pred <- model(x_t, t, cond)
                x_t <- mask * x_input + (1-mask) * x0_pred
        """
        if x_input.dim() != 3:
            raise ValueError(f"x_input must be [B,T,D], got {x_input.shape}")
        if inpaint_mask.shape != x_input.shape:
            raise ValueError(f"inpaint_mask shape mismatch: {inpaint_mask.shape} vs {x_input.shape}")

        batch = x_input.shape[0]
        device = x_input.device
        dtype = x_input.dtype
        mask_bool = inpaint_mask.to(device=device, dtype=torch.bool)

        cond_use = self._normalize_cond(cond, x_input)
        total_steps = self.timesteps if steps is None else int(steps)
        schedule = self._build_sampling_timesteps(total_steps)

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
        """
        Reverse diffusion with hard inpainting constraints.

        Args:
            x_input: known clean values, shape [B,T,D]
            inpaint_mask: True for constrained dims, shape [B,T,D]
            cond: optional condition
            x_start: optional initial clean state for unknown dims
        """
        if x_input.dim() != 3:
            raise ValueError(f"x_input must be [B,T,D], got {x_input.shape}")
        if inpaint_mask.shape != x_input.shape:
            raise ValueError(f"inpaint_mask shape mismatch: {inpaint_mask.shape} vs {x_input.shape}")

        sampler = str(sampler).lower()
        if sampler not in {"ddim", "ddpm"}:
            raise ValueError(f"sampler must be 'ddim' or 'ddpm', got {sampler}")

        batch, seq_len, _ = x_input.shape
        device = x_input.device
        dtype = x_input.dtype

        cond_use = self._normalize_cond(cond, x_input)
        mask_bool = inpaint_mask.to(device=device, dtype=torch.bool)

        total_steps = self.timesteps if steps is None else int(steps)
        schedule = self._build_sampling_timesteps(total_steps)

        base_start = x_input if x_start is None else x_start.to(device=device, dtype=dtype)
        # Keep one fixed noise realization for known constraints during one reverse pass.
        # Re-sampling this per step makes the conditioning itself jitter across timesteps.
        known_noise = torch.randn_like(x_input)

        # Initialize at highest noise level, then enforce known region.
        t0 = int(schedule[0].item())
        t0_batch = torch.full((batch,), t0, device=device, dtype=torch.long)
        x_t = self.q_sample(base_start, t0_batch, noise=torch.randn_like(base_start))
        known_noisy = self.q_sample(x_input, t0_batch, noise=known_noise)
        x_t = torch.where(mask_bool, known_noisy, x_t)

        for idx, t_cur in enumerate(schedule.tolist()):
            t_next = schedule[idx + 1].item() if idx + 1 < len(schedule) else -1
            t_batch = torch.full((batch,), t_cur, device=device, dtype=torch.long)

            # Re-impose known constraints at current noise level.
            known_noisy_cur = self.q_sample(x_input, t_batch, noise=known_noise)
            x_t = torch.where(mask_bool, known_noisy_cur, x_t)

            x0_pred, _, _ = self.predict(x_t=x_t, t=t_batch, cond=cond_use)
            x0_guided = torch.where(mask_bool, x_input, x0_pred)

            if t_next < 0:
                x_t = x0_guided
                break

            eps_guided = self.predict_eps_from_x0(x_t, t_batch, x0_guided)
            x_next = self._ddim_update(
                x_t,
                x0_guided,
                eps_guided,
                t_cur=t_cur,
                t_next=t_next,
                sampler=sampler,
                eta=eta,
            )

            t_next_batch = torch.full((batch,), t_next, device=device, dtype=torch.long)
            known_noisy_next = self.q_sample(x_input, t_next_batch, noise=known_noise)
            x_t = torch.where(mask_bool, known_noisy_next, x_next)

        return x_t
