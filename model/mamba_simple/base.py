"""Mamba backbone components for the RNN-compatible three-stage pipeline."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_value(cfg, name: str, default):
    simple_cfg = getattr(cfg, "mamba_simple", {}) if cfg is not None else {}
    if isinstance(simple_cfg, dict) and name in simple_cfg:
        return simple_cfg[name]
    if hasattr(simple_cfg, name):
        return getattr(simple_cfg, name)
    legacy_name = f"mamba_simple_{name}"
    if cfg is not None and hasattr(cfg, legacy_name):
        return getattr(cfg, legacy_name)
    return default


def mamba_kwargs_from_cfg(cfg) -> dict:
    return {
        "d_state": int(_cfg_value(cfg, "d_state", 16)),
        "d_conv": int(_cfg_value(cfg, "d_conv", 4)),
        "expand": int(_cfg_value(cfg, "expand", 2)),
        "ff_mult": float(_cfg_value(cfg, "ff_mult", 2.0)),
        "init_prefix_len": int(_cfg_value(cfg, "init_prefix_len", 4)),
        "use_film": bool(_cfg_value(cfg, "use_film", True)),
    }


def _load_mamba_cls():
    try:
        from mamba_ssm import Mamba
    except Exception as exc:
        raise ImportError(
            "model_arch='mamba_simple' requires the standard mamba_ssm package. "
            "Install it with: pip install mamba-ssm[causal-conv1d] --no-build-isolation"
        ) from exc
    return Mamba


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class GatedFFN(nn.Module):
    def __init__(self, dim: int, mult: float = 2.0, dropout: float = 0.1):
        super().__init__()
        inner = max(dim, int(dim * mult))
        self.up = nn.Linear(dim, inner * 2)
        self.down = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * value))


class MambaResidualBlock(nn.Module):
    """Pre-norm Mamba block with a small gated FFN stabilizer."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ff_mult: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        Mamba = _load_mamba_cls()
        self.norm_mamba = RMSNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm_ff = RMSNorm(dim)
        self.ffn = GatedFFN(dim, mult=ff_mult, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.mamba(self.norm_mamba(x)))
        x = x + self.dropout(self.ffn(self.norm_ff(x)))
        return x


class MambaEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ff_mult: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        depth = max(int(depth), 1)
        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock(
                    dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class InitConditioner(nn.Module):
    """Inject x_init as scan prefix tokens and persistent FiLM conditioning."""

    def __init__(
        self,
        init_dim: int,
        hidden_dim: int,
        *,
        prefix_len: int = 4,
        use_film: bool = True,
    ):
        super().__init__()
        self.prefix_len = max(int(prefix_len), 1)
        self.use_film = bool(use_film)
        self.prefix_net = nn.Sequential(
            nn.Linear(init_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * self.prefix_len),
        )
        self.film_net = (
            nn.Sequential(
                nn.Linear(init_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
            if self.use_film
            else None
        )

    def forward(self, x: torch.Tensor, x_init: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch_size = x.shape[0]
        prefix = self.prefix_net(x_init).view(batch_size, self.prefix_len, x.shape[-1])
        if self.film_net is not None:
            scale, shift = self.film_net(x_init).chunk(2, dim=-1)
            x = x * (1.0 + torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        return torch.cat((prefix, x), dim=1), self.prefix_len


class RNN(nn.Module):
    """RNN-compatible sequence head backed by Mamba."""

    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_rnn_layer=2,
        bidirectional=False,
        dropout=0.2,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ff_mult: float = 2.0,
        init_prefix_len: int = 4,
        use_film: bool = True,
    ):
        super().__init__()
        self.n_hidden = int(n_hidden)
        self.n_rnn_layer = int(n_rnn_layer)
        self.num_directions = 2 if bidirectional else 1
        self.bidirectional = bool(bidirectional)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.forward_encoder = MambaEncoder(
            n_hidden,
            self.n_rnn_layer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        if self.bidirectional:
            self.backward_encoder = MambaEncoder(
                n_hidden,
                self.n_rnn_layer,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            self.bidir_gate = nn.Sequential(
                nn.Linear(n_hidden * 3, n_hidden),
                nn.Sigmoid(),
            )
        else:
            self.backward_encoder = None
            self.bidir_gate = None
        self.linear2 = nn.Linear(n_hidden * self.num_directions, n_output)
        self._unused_init_prefix_len = init_prefix_len
        self._unused_use_film = use_film

    def _encode_hidden(self, h: torch.Tensor) -> torch.Tensor:
        fwd = self.forward_encoder(h)
        if not self.bidirectional:
            return fwd
        bwd = torch.flip(self.backward_encoder(torch.flip(h, dims=(1,))), dims=(1,))
        gate = self.bidir_gate(torch.cat((fwd, bwd, h), dim=-1))
        return torch.cat((fwd, gate * bwd), dim=-1)

    def forward(self, x, h=None):
        _ = h
        hidden = self.dropout(F.silu(self.linear1(x)))
        return self.linear2(self._encode_hidden(hidden))


class RNNWithInit(nn.Module):
    """RNNWithInit-compatible head using prefix-state and FiLM-conditioned Mamba."""

    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_init,
        n_rnn_layer,
        bidirectional=False,
        dropout=0.2,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ff_mult: float = 2.0,
        init_prefix_len: int = 4,
        use_film: bool = True,
    ):
        super().__init__()
        self.n_hidden = int(n_hidden)
        self.n_rnn_layer = int(n_rnn_layer)
        self.num_directions = 2 if bidirectional else 1
        self.bidirectional = bool(bidirectional)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.conditioner = InitConditioner(
            n_init,
            n_hidden,
            prefix_len=init_prefix_len,
            use_film=use_film,
        )
        self.forward_encoder = MambaEncoder(
            n_hidden,
            self.n_rnn_layer,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        if self.bidirectional:
            self.backward_encoder = MambaEncoder(
                n_hidden,
                self.n_rnn_layer,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            self.bidir_gate = nn.Sequential(
                nn.Linear(n_hidden * 3, n_hidden),
                nn.Sigmoid(),
            )
        else:
            self.backward_encoder = None
            self.bidir_gate = None
        self.linear2 = nn.Linear(n_hidden * self.num_directions, n_output)

    def forward(self, inputs, _=None):
        x, x_init = inputs
        hidden = self.dropout(F.silu(self.linear1(x)))
        conditioned, prefix_len = self.conditioner(hidden, x_init)
        fwd = self.forward_encoder(conditioned)[:, prefix_len:]
        if not self.bidirectional:
            return self.linear2(fwd)

        # Reverse only frame tokens; the same prefix still acts as the scan state.
        rev_hidden = torch.flip(hidden, dims=(1,))
        rev_conditioned, rev_prefix_len = self.conditioner(rev_hidden, x_init)
        bwd = self.backward_encoder(rev_conditioned)[:, rev_prefix_len:]
        bwd = torch.flip(bwd, dims=(1,))
        gate = self.bidir_gate(torch.cat((fwd, bwd, hidden), dim=-1))
        return self.linear2(torch.cat((fwd, gate * bwd), dim=-1))


class SubPoser(nn.Module):
    """Sub pose predictor with init-conditioned Mamba velocity and pose heads."""

    def __init__(self, n_input, v_output, p_output, n_hidden, num_layer, dropout, extra_dim=0, **mamba_kwargs):
        super().__init__()
        self.extra_dim = extra_dim
        self.v_output = v_output
        self.p_output = p_output
        self.rnn1 = RNNWithInit(
            n_input=n_input - extra_dim,
            n_output=v_output,
            n_hidden=n_hidden,
            n_init=v_output,
            n_rnn_layer=num_layer,
            bidirectional=False,
            dropout=dropout,
            **mamba_kwargs,
        )
        self.rnn2 = RNNWithInit(
            n_input=n_input + v_output,
            n_output=p_output,
            n_hidden=n_hidden,
            n_init=p_output,
            n_rnn_layer=num_layer,
            bidirectional=False,
            dropout=dropout,
            **mamba_kwargs,
        )

    def forward(self, x, v_init, p_init):
        x_v = x[..., :-self.extra_dim] if self.extra_dim else x
        v = self.rnn1((x_v, v_init))
        p = self.rnn2((torch.cat((x, v), dim=-1), p_init))
        return v, p


class PartMambaBoundary(nn.Module):
    """Body-part temporal encoder replacing the GRU boundary stack."""

    def __init__(
        self,
        group_input_dims: Dict[str, int],
        group_hidden_dim: int,
        boundary_hidden_dim: int,
        *,
        layers: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        ff_mult: float = 2.0,
        init_prefix_len: int = 4,
        use_film: bool = True,
    ):
        super().__init__()
        _ = init_prefix_len, use_film
        if not group_input_dims:
            raise ValueError("group_input_dims cannot be empty")
        self.group_names = list(group_input_dims.keys())
        self.group_norms = nn.ModuleDict({name: nn.LayerNorm(dim) for name, dim in group_input_dims.items()})
        self.group_proj = nn.ModuleDict({name: nn.Linear(dim, group_hidden_dim) for name, dim in group_input_dims.items()})
        self.group_embedding = nn.Parameter(torch.zeros(len(self.group_names), group_hidden_dim))
        nn.init.normal_(self.group_embedding, mean=0.0, std=0.02)

        self.shared_group_encoder = MambaEncoder(
            group_hidden_dim,
            layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        gate_hidden = max(group_hidden_dim // 2, 16)
        self.part_gate = nn.Sequential(
            nn.Linear(group_hidden_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(group_hidden_dim * 2, boundary_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(boundary_hidden_dim),
        )
        self.boundary_encoder = MambaEncoder(
            boundary_hidden_dim,
            layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self.boundary_head = nn.Linear(boundary_hidden_dim, 2)

    def forward(self, group_inputs: Dict[str, torch.Tensor]):
        encoded = []
        for idx, name in enumerate(self.group_names):
            if name not in group_inputs:
                raise KeyError(f"missing group input: {name}")
            x = self.group_norms[name](group_inputs[name])
            h = self.group_proj[name](x) + self.group_embedding[idx].view(1, 1, -1)
            encoded.append(h)

        tokens = torch.stack(encoded, dim=2)
        batch_size, seq_len, num_groups, hidden_dim = tokens.shape
        flat = tokens.permute(0, 2, 1, 3).reshape(batch_size * num_groups, seq_len, hidden_dim)
        flat = self.shared_group_encoder(flat)
        tokens = flat.reshape(batch_size, num_groups, seq_len, hidden_dim).permute(0, 2, 1, 3)

        gate_logits = self.part_gate(tokens)
        weights = torch.softmax(gate_logits, dim=2)
        weighted = (weights * tokens).sum(dim=2)
        pooled = tokens.mean(dim=2)
        fused = self.fuse_proj(torch.cat((weighted, pooled), dim=-1))
        features = self.boundary_encoder(fused)
        logits = self.boundary_head(features)
        return logits, torch.sigmoid(logits), features


__all__ = [
    "RMSNorm",
    "MambaEncoder",
    "RNN",
    "RNNWithInit",
    "SubPoser",
    "PartMambaBoundary",
    "mamba_kwargs_from_cfg",
]
