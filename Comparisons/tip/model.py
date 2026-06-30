"""TIP-style causal Transformer/RNN baseline with object-position extension."""
from __future__ import annotations

import torch
from torch import nn


class TIPHOIModel(nn.Module):
    """Transformer Inertial Poser core with an extra object output.

    The human state remains [18*6 pose, root velocity, optional SBP constraints].
    Object IMU is appended to the IMU stream; object translation is appended as an
    extra prediction head.
    """

    def __init__(
        self,
        input_size_imu: int = 72,
        obj_imu_dim: int = 12,
        n_sbps: int = 5,
        rnn_hidden_size: int = 512,
        tf_hidden_size: int = 1024,
        tf_input_dim: int = 256,
        n_heads: int = 8,
        tf_layers: int = 4,
        dropout: float = 0.0,
        in_dropout: float = 0.0,
        past_state_dropout: float = 0.8,
        with_rnn: bool = True,
    ):
        super().__init__()
        self.input_size_imu = input_size_imu
        self.obj_imu_dim = obj_imu_dim
        self.n_sbps = n_sbps
        self.human_state_dim = 18 * 6 + 3 + n_sbps * 4
        self.output_dim = self.human_state_dim + 3
        self.with_rnn = with_rnn
        self.in_dropout = nn.Dropout(in_dropout)
        self.past_dropout = nn.Dropout(past_state_dropout)
        self.in_linear = nn.Linear(input_size_imu + obj_imu_dim + self.human_state_dim, tf_input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_input_dim,
            nhead=n_heads,
            dim_feedforward=tf_hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.tf_encode = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        if with_rnn:
            self.rnn = nn.RNN(
                input_size=tf_input_dim,
                hidden_size=rnn_hidden_size,
                num_layers=1,
                nonlinearity="tanh",
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            )
            self.linear = nn.Linear(rnn_hidden_size, self.output_dim)
        else:
            self.rnn = None
            self.linear = nn.Linear(tf_input_dim, self.output_dim)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)

    def forward(self, imu: torch.Tensor, prev_state: torch.Tensor, obj_imu: torch.Tensor) -> dict[str, torch.Tensor]:
        state = prev_state.clone().nan_to_num(0.0)
        # Keep TIP's no-teacher-forcing rule for historical root velocity.
        state[:, :, 18 * 6 : 18 * 6 + 3] = 0.0
        x = torch.cat([self.in_dropout(imu), obj_imu, self.past_dropout(state)], dim=-1)
        x = self.in_linear(x)
        x = self.tf_encode(x, mask=self._causal_mask(x.shape[1], x.device))
        if self.rnn is not None:
            h0 = torch.zeros(1, x.shape[0], self.rnn.hidden_size, device=x.device, dtype=x.dtype)
            x, _ = self.rnn(x, h0)
        y = self.linear(x)
        return {
            "state": y[:, :, : self.human_state_dim],
            "obj_trans": y[:, :, self.human_state_dim :],
        }

