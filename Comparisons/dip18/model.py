"""DIP18-style BiRNN baseline with an explicit object-position extension."""
from __future__ import annotations

import torch
import torch.nn as nn


class DIPBiRNNHead(nn.Module):
    """DIP BiRNN block: input FC 512, 2-layer BiLSTM 512, output FC 256."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_hidden_size: int = 512,
        rnn_hidden_size: int = 512,
        output_hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, input_hidden_size),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_hidden_size,
            rnn_hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output_hidden = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, output_hidden_size),
            nn.ReLU(),
        )
        self.output = nn.Linear(output_hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input_layer(x)
        y, _ = self.rnn(y)
        y = self.output_hidden(y)
        return self.output(y)


class DIP18HOIModel(nn.Module):
    """PyTorch port of the DIP BiRNN reconstruction surface.

    Human pose prediction keeps DIP's Gaussian-output training semantics.  The
    object branch is intentionally separate and predicts object position from the
    same augmented IMU stream.
    """

    def __init__(
        self,
        human_input_dim: int = 60,
        obj_imu_dim: int = 12,
        pose_dim: int = 135,
        hidden_size: int = 512,
        input_hidden_size: int = 512,
        output_hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.human_input_dim = human_input_dim
        self.obj_imu_dim = obj_imu_dim
        self.pose_dim = pose_dim
        input_dim = human_input_dim + obj_imu_dim
        self.temporal = DIPBiRNNHead(
            input_dim=input_dim,
            output_dim=pose_dim * 2,
            input_hidden_size=input_hidden_size,
            rnn_hidden_size=hidden_size,
            output_hidden_size=output_hidden_size,
            num_layers=num_layers,
        )
        self.obj_head = DIPBiRNNHead(
            input_dim=input_dim,
            output_dim=3,
            input_hidden_size=input_hidden_size,
            rnn_hidden_size=hidden_size,
            output_hidden_size=output_hidden_size,
            num_layers=num_layers,
        )

    def forward(self, imu: torch.Tensor, obj_imu: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([imu, obj_imu], dim=-1)
        pose_stats = self.temporal(x)
        pose_mu, pose_log_sigma = pose_stats.split(self.pose_dim, dim=-1)
        obj_trans = self.obj_head(x)
        return {
            "pose_mu": pose_mu,
            "pose_log_sigma": pose_log_sigma,
            "obj_trans": obj_trans,
        }

    @torch.no_grad()
    def reconstruct_chunks(
        self,
        imu: torch.Tensor,
        obj_imu: torch.Tensor,
        past_frames: int = 20,
        future_frames: int = 5,
    ) -> dict[str, torch.Tensor]:
        """Chunked BiRNN reconstruction for DIP-style streaming evaluation."""
        batch_size, seq_len = imu.shape[:2]
        pose_acc = []
        obj_acc = []
        for t in range(seq_len):
            start = max(0, t - past_frames)
            end = min(seq_len, t + future_frames + 1)
            out = self.forward(imu[:, start:end], obj_imu[:, start:end])
            center = t - start
            pose_acc.append(out["pose_mu"][:, center])
            obj_acc.append(out["obj_trans"][:, center])
        return {
            "pose_mu": torch.stack(pose_acc, dim=1).reshape(batch_size, seq_len, -1),
            "obj_trans": torch.stack(obj_acc, dim=1).reshape(batch_size, seq_len, -1),
        }
