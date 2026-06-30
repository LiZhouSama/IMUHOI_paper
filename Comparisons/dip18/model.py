"""DIP18-style BiRNN baseline with an explicit object-position extension."""
from __future__ import annotations

import torch
import torch.nn as nn

from Comparisons.common.modules import BatchRNN


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
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.human_input_dim = human_input_dim
        self.obj_imu_dim = obj_imu_dim
        self.pose_dim = pose_dim
        input_dim = human_input_dim + obj_imu_dim
        self.temporal = BatchRNN(
            n_input=input_dim,
            n_output=pose_dim * 2,
            n_hidden=hidden_size,
            n_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
        )
        self.obj_head = BatchRNN(
            n_input=input_dim,
            n_output=3,
            n_hidden=hidden_size // 2,
            n_layers=max(1, num_layers),
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, imu: torch.Tensor, obj_imu: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([imu, obj_imu], dim=-1)
        pose_stats, _ = self.temporal(x)
        pose_mu, pose_log_sigma = pose_stats.split(self.pose_dim, dim=-1)
        obj_trans, _ = self.obj_head(x)
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

