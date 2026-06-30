"""TransPose architecture with the original pose and translation branches."""
from __future__ import annotations

import torch
import torch.nn as nn

from Comparisons.common.modules import BatchRNN


class TransPoseHOIModel(nn.Module):
    """TransPose pose-s1/s2/s3 + tran-b1/b2 with an object position head."""

    n_leaf = 5
    n_full = 23
    n_reduced = 15

    def __init__(
        self,
        human_imu_dim: int = 72,
        obj_imu_dim: int = 12,
        dropout: float = 0.2,
    ):
        super().__init__()
        n_imu = human_imu_dim + obj_imu_dim
        self.pose_s1 = BatchRNN(n_imu, self.n_leaf * 3, 256, n_layers=2, bidirectional=True, dropout=dropout)
        self.pose_s2 = BatchRNN(
            self.n_leaf * 3 + n_imu,
            self.n_full * 3,
            64,
            n_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.pose_s3 = BatchRNN(
            self.n_full * 3 + n_imu,
            self.n_reduced * 6,
            128,
            n_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.tran_b1 = BatchRNN(
            self.n_leaf * 3 + n_imu,
            2,
            64,
            n_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.tran_b2 = BatchRNN(
            self.n_full * 3 + n_imu,
            3,
            256,
            n_layers=2,
            bidirectional=False,
            dropout=dropout,
        )
        self.obj_head = BatchRNN(
            self.n_full * 3 + n_imu,
            3,
            128,
            n_layers=2,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, imu: torch.Tensor, obj_imu: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([imu, obj_imu], dim=-1)
        leaf, _ = self.pose_s1(x)
        full, _ = self.pose_s2(torch.cat([leaf, x], dim=-1))
        pose, _ = self.pose_s3(torch.cat([full, x], dim=-1))
        contact_logits, _ = self.tran_b1(torch.cat([leaf, x], dim=-1))
        root_vel, _ = self.tran_b2(torch.cat([full, x], dim=-1))
        obj_trans, _ = self.obj_head(torch.cat([full, x], dim=-1))
        return {
            "leaf_pos": leaf,
            "full_pos": full,
            "pose": pose,
            "contact_logits": contact_logits,
            "root_vel": root_vel,
            "obj_trans": obj_trans,
        }

