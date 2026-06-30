"""Small neural network building blocks shared by comparison baselines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchRNN(nn.Module):
    """Linear input projection + LSTM + linear output, using [B, T, F]."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.rnn = nn.LSTM(
            n_hidden,
            n_hidden,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h=None) -> tuple[torch.Tensor, tuple]:
        y = F.relu(self.linear1(self.dropout(x)))
        y, h = self.rnn(y, h)
        return self.linear2(y), h


class BatchRNNWithInit(nn.Module):
    """RNN with an initialization network, mirroring GlobalPose's RNNWithInit use."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        init_size: int,
        num_layers: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_net = nn.Sequential(
            nn.Linear(init_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * num_layers * hidden_size),
        )
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, init: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        if init is None:
            init = torch.zeros(batch_size, self.init_net[0].in_features, device=x.device, dtype=x.dtype)
        hc = self.init_net(init).view(batch_size, 2, self.num_layers, self.hidden_size)
        hc = hc.permute(1, 2, 0, 3).contiguous()
        y, _ = self.rnn(x, (hc[0], hc[1]))
        return self.linear2(y)

