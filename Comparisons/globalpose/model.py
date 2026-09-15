"""GlobalPose-style PL/IK/VR network with an object-motion extension."""
from __future__ import annotations

import torch
import torch.nn as nn

from utils.rotation_conversions import rotation_6d_to_matrix
from Comparisons.common.geometry import (
    from_to_rotation_matrix,
    r6d_columns_to_rotation_matrix,
)
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
        # The released GlobalPose object extension concatenates 9 object IMU
        # channels to the 243-D VR state.  Keep 12 as the constructor default
        # for checkpoints produced by the older comparison wrapper.
        obj_imu_dim: int = 12,
        hidden_size: int = 512,
        dropout: float = 0.4,
        object_bidirectional: bool = False,
        object_with_init: bool = True,
        human_condition_on_object: bool = False,
    ):
        super().__init__()
        self.human_input_dim = human_input_dim
        self.obj_imu_dim = obj_imu_dim
        self.object_with_init = bool(object_with_init)
        # The reference ObjectVRNet is a causal RNNWithInit.  The plain
        # BatchRNN branch is retained only for strict legacy-checkpoint loading.
        self.object_bidirectional = (
            bool(object_bidirectional) if not self.object_with_init else False
        )
        self.human_condition_on_object = bool(human_condition_on_object)
        self.plnet = BatchRNNWithInit(
            input_size=(
                human_input_dim + obj_imu_dim
                if self.human_condition_on_object
                else human_input_dim
            ),
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
            input_size=(
                243 + obj_imu_dim if self.human_condition_on_object else 243
            ),
            output_size=9,
            hidden_size=hidden_size,
            init_size=9,
            num_layers=3,
            dropout=dropout,
        )
        if self.object_with_init:
            self.obj_head = BatchRNNWithInit(
                input_size=243 + obj_imu_dim,
                output_size=3,
                hidden_size=256,
                init_size=3,
                num_layers=2,
                dropout=0.2,
            )
        else:
            self.obj_head = BatchRNN(
                n_input=243 + obj_imu_dim,
                n_output=3,
                n_hidden=256,
                n_layers=2,
                bidirectional=self.object_bidirectional,
                dropout=0.2,
            )

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        x: torch.Tensor,
        obj_imu: torch.Tensor,
        apply_gravity_correction: bool = True,
        column_r6d: bool = True,
        pl_init: torch.Tensor | None = None,
        vr_init: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward on already-adapted GlobalPose inputs.

        Args:
            x: [B, T, 84] GlobalPose PL-stage input.
            obj_imu: [B, T, ``obj_imu_dim``] object IMU extension.
        """
        object_input = torch.cat([x, obj_imu], dim=-1)
        model_input = object_input if self.human_condition_on_object else x
        pl = self.plnet(model_input, init=pl_init)

        r_rb = x[:, :, 36:81].reshape(*x.shape[:2], 5, 3, 3)
        g_r0 = x[:, :, 81:84]
        p_rb = pl[:, :, :15]
        g_r1 = self._normalize(pl[:, :, 15:18])
        cascade_g_r1 = g_r1.detach() if apply_gravity_correction else g_r1
        if apply_gravity_correction:
            r_rb = from_to_rotation_matrix(g_r0, cascade_g_r1).unsqueeze(2).matmul(r_rb)

        ik1_in = torch.cat(
            [r_rb.flatten(2), cascade_g_r1, p_rb],
            dim=-1,
        )
        ik1, _ = self.iknet1(ik1_in)
        p_rj = ik1[:, :, :69]
        g_r2 = self._normalize(ik1[:, :, 69:72])
        cascade_g_r2 = g_r2.detach() if apply_gravity_correction else g_r2
        if apply_gravity_correction:
            r_rb = from_to_rotation_matrix(cascade_g_r1, cascade_g_r2).unsqueeze(2).matmul(r_rb)

        ik2_in = torch.cat(
            [
                r_rb.flatten(2),
                cascade_g_r2,
                p_rj,
            ],
            dim=-1,
        )
        rrj_6d, _ = self.iknet2(ik2_in)
        decoder = r6d_columns_to_rotation_matrix if column_r6d else rotation_6d_to_matrix
        rrj_mat = decoder(rrj_6d.reshape(-1, 6)).reshape(
            x.shape[0], x.shape[1], 15 * 9
        )

        root_correction = from_to_rotation_matrix(cascade_g_r2, g_r0)
        aw_rb = x[:, :, :36]
        if apply_gravity_correction:
            aw_rb = torch.einsum(
                "btsi,btij->btsj", aw_rb.reshape(*x.shape[:2], 12, 3), root_correction
            ).flatten(2)
        vr_features = torch.cat(
            [
                rrj_mat.detach() if apply_gravity_correction else rrj_mat,
                p_rj.detach() if apply_gravity_correction else p_rj,
                aw_rb,
                cascade_g_r2,
            ],
            dim=-1,
        )
        object_vr_input = torch.cat([vr_features, obj_imu], dim=-1)
        human_vr_input = (
            object_vr_input if self.human_condition_on_object else vr_features
        )
        vr = self.vrnet(human_vr_input, init=vr_init)
        if self.object_with_init:
            obj_init = torch.zeros(
                x.shape[0], 3, device=x.device, dtype=x.dtype
            )
            obj_trans = self.obj_head(object_vr_input, init=obj_init)
        else:
            obj_trans, _ = self.obj_head(object_vr_input)
        if not apply_gravity_correction:
            root_correction = torch.eye(
                3, device=x.device, dtype=x.dtype
            ).view(1, 1, 3, 3).expand(*x.shape[:2], 3, 3)

        return {
            "human": torch.cat([pl, ik1, rrj_6d, vr], dim=-1),
            "pl": pl,
            "ik1": ik1,
            "ik2": rrj_6d,
            "vr": vr,
            "root_correction": root_correction,
            "obj_trans": obj_trans,
        }
