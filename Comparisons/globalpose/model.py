"""GlobalPose-style PL/IK/VR network with object-position extension."""
from __future__ import annotations

import torch
import torch.nn as nn

from utils.rotation_conversions import rotation_6d_to_matrix
from Comparisons.common.modules import BatchRNN, BatchRNNWithInit


class GlobalPoseHOIModel(nn.Module):
    """Network-supervised GlobalPose baseline.

    The original paper's per-frame physics optimizer is not a learnable module.
    This class preserves the PL -> IK1 -> IK2 -> VR prediction chain and exposes
    the predicted VR/contact outputs used by the physics stage.  Physics-based
    online refinement can be added around these outputs without changing the
    training contract.
    """

    def __init__(
        self,
        human_input_dim: int = 84,
        obj_imu_dim: int = 12,
        hidden_size: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.human_input_dim = human_input_dim
        self.obj_imu_dim = obj_imu_dim
        self.plnet = BatchRNNWithInit(
            input_size=human_input_dim + obj_imu_dim,
            output_size=18,
            hidden_size=hidden_size,
            init_size=18,
            num_layers=3,
            dropout=dropout,
        )
        self.iknet1 = BatchRNN(
            n_input=63,
            n_output=72,
            n_hidden=hidden_size,
            n_layers=3,
            bidirectional=False,
            dropout=dropout,
            input_linear=False,
        )
        self.iknet2 = BatchRNN(
            n_input=117,
            n_output=90,
            n_hidden=hidden_size,
            n_layers=3,
            bidirectional=False,
            dropout=dropout,
            input_linear=False,
        )
        self.vrnet = BatchRNNWithInit(
            input_size=243 + obj_imu_dim,
            output_size=9,
            hidden_size=hidden_size,
            init_size=9,
            num_layers=3,
            dropout=dropout,
        )
        self.obj_head = BatchRNN(
            n_input=243 + obj_imu_dim,
            n_output=3,
            n_hidden=256,
            n_layers=2,
            bidirectional=True,
            dropout=0.2,
        )

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor, obj_imu: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward on already-adapted GlobalPose inputs.

        Args:
            x: [B, T, 84] GlobalPose PL-stage input.
            obj_imu: [B, T, 12] object IMU extension.
        """
        model_input = torch.cat([x, obj_imu], dim=-1)
        pl = self.plnet(model_input)

        r_rb = x[:, :, 36:81]
        g_r0 = x[:, :, 81:84]
        p_rb = pl[:, :, :15]
        g_r1 = self._normalize(pl[:, :, 15:18])

        ik1_in = torch.cat([r_rb, g_r1, p_rb], dim=-1)
        ik1, _ = self.iknet1(ik1_in)
        p_rj = ik1[:, :, :69]
        g_r2 = self._normalize(ik1[:, :, 69:72])

        ik2_in = torch.cat([r_rb, g_r2, p_rj], dim=-1)
        rrj_6d, _ = self.iknet2(ik2_in)
        rrj_mat = rotation_6d_to_matrix(rrj_6d.reshape(-1, 6)).reshape(x.shape[0], x.shape[1], 15 * 9)

        aw_rb = x[:, :, :36]
        vr_features = torch.cat([rrj_mat, p_rj, aw_rb, g_r2], dim=-1)
        vr_input = torch.cat([vr_features, obj_imu], dim=-1)
        vr = self.vrnet(vr_input)
        obj_trans, _ = self.obj_head(vr_input)

        return {
            "human": torch.cat([pl, ik1, rrj_6d, vr], dim=-1),
            "pl": pl,
            "ik1": ik1,
            "ik2": rrj_6d,
            "vr": vr,
            "obj_trans": obj_trans,
        }
