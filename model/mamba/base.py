"""Causal dual-scale Mamba-style building blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class CausalDepthwiseConv1d(nn.Module):
    """Depthwise Conv1d with left padding only."""

    def __init__(self, dim: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]
        pad = (self.kernel_size - 1) * self.dilation
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (pad, 0))
        return self.conv(x_t).transpose(1, 2)


class CausalMovingAverage(nn.Module):
    """Causal low-pass filter used by the slow branch."""

    def __init__(self, window: int):
        super().__init__()
        self.window = max(1, int(window))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.window <= 1:
            return x
        x_t = F.pad(x.transpose(1, 2), (self.window - 1, 0))
        return F.avg_pool1d(x_t, kernel_size=self.window, stride=1).transpose(1, 2)


class MambaBlock(nn.Module):
    """
    Pure PyTorch causal Mamba-style block.

    This is a diagonal selective state-space block with depthwise causal
    convolution and input-dependent scan parameters. It keeps the public shape
    and causality expected from a Mamba temporal backbone without requiring the
    external CUDA mamba_ssm package.
    """

    def __init__(
        self,
        dim: int,
        *,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        dilation: int = 1,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.conv = CausalDepthwiseConv1d(dim, kernel_size=conv_kernel, dilation=dilation)
        self.param_proj = nn.Linear(dim, dim * 3)
        self.skip = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        u, gate = self.in_proj(x_norm).chunk(2, dim=-1)
        u = F.silu(self.conv(u))
        a_raw, b_raw, c_raw = self.param_proj(u).chunk(3, dim=-1)
        a = torch.sigmoid(a_raw)
        b = torch.tanh(b_raw)
        c = torch.sigmoid(c_raw)

        state = torch.zeros_like(u[:, 0])
        outputs = []
        for t in range(u.shape[1]):
            state = a[:, t] * state + b[:, t] * u[:, t]
            outputs.append((c[:, t] * state + self.skip * u[:, t]).unsqueeze(1))
        y = torch.cat(outputs, dim=1)
        y = y * F.silu(gate)
        y = self.out_proj(y)
        return residual + self.dropout(y)


class TemporalMambaEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        conv_kernel: int,
        dropout: float,
        dilation: int = 1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    dim,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                    dilation=dilation,
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class InputStem(nn.Module):
    """Linear projection plus short causal convolution."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, conv_kernel: int = 3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.conv = CausalDepthwiseConv1d(hidden_dim, kernel_size=conv_kernel)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x + self.mlp(self.conv(x))
        return self.norm(x)


class CrossScaleFusion(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, use_fusion_block: bool = True):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.fusion_block = MambaBlock(dim, conv_kernel=3, dropout=dropout) if use_fusion_block else nn.Identity()
        self.norm = RMSNorm(dim)

    def forward(self, fast: torch.Tensor, slow: torch.Tensor) -> torch.Tensor:
        cat = torch.cat((fast, slow), dim=-1)
        gate = self.gate(cat)
        fused = gate * fast + (1.0 - gate) * slow
        fused = fused + self.mix(cat)
        fused = self.fusion_block(fused)
        return self.norm(fused)


class TemporalMemoryFusion(nn.Module):
    """Optional cross-window memory bridge."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        heads = max(1, int(num_heads))
        while dim % heads != 0 and heads > 1:
            heads -= 1
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = RMSNorm(dim)

    @staticmethod
    def _pack_memory(memory) -> torch.Tensor | None:
        if memory is None:
            return None
        if isinstance(memory, torch.Tensor):
            return memory
        if not isinstance(memory, dict):
            return None
        tensors = []
        for key in ("ctx", "torso", "lower", "upper", "root"):
            value = memory.get(key)
            if isinstance(value, torch.Tensor) and value.dim() == 3:
                tensors.append(value)
        if not tensors:
            return None
        return torch.cat(tensors, dim=1)

    def forward(self, x: torch.Tensor, memory=None) -> torch.Tensor:
        mem = self._pack_memory(memory)
        if mem is None or mem.numel() == 0:
            return x
        mem = mem.to(device=x.device, dtype=x.dtype)
        if mem.shape[0] != x.shape[0] or mem.shape[-1] != x.shape[-1]:
            return x
        attended, _ = self.attn(x, mem, mem, need_weights=False)
        gate = self.gate(torch.cat((x, attended), dim=-1))
        out = gate * x + (1.0 - gate) * attended + self.out(torch.cat((x, attended), dim=-1))
        return self.norm(out)


class PartDecoder(nn.Module):
    """Hierarchical part decoder with latent, pose, and optional velocity heads."""

    def __init__(self, input_dim: int, hidden_dim: int, pose_dim: int, vel_dim: int = 0, dropout: float = 0.1):
        super().__init__()
        self.latent = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.pose_head = nn.Linear(hidden_dim, pose_dim)
        self.vel_head = nn.Linear(hidden_dim, vel_dim) if vel_dim > 0 else None

    def forward(self, x: torch.Tensor):
        h = self.latent(x)
        pose = self.pose_head(h)
        vel = self.vel_head(h) if self.vel_head is not None else None
        return h, pose, vel

