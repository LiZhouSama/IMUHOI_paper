"""Four-head IMUHOI ObjectTrans ablation with a static-hold branch."""
from __future__ import annotations

import torch

from configs import FRAME_RATE
from .object_trans_IMUHOI import IMUHOIObjectTransModule


class IMUHOI4HeadsObjectTransModule(IMUHOIObjectTransModule):
    """Fuse left FK, right FK, IMU integration, and static hold."""

    GATING_OUTPUT_DIM = 4
    ARCH_NAME = "IMUHOI_4heads"

    @staticmethod
    def _build_gating_prior(contact_prob: torch.Tensor, eps: float = 1e-6):
        """Build priors for [left FK, right FK, IMU motion, static hold]."""
        p_left = contact_prob[..., 0:1].clamp(0.0, 1.0)
        p_right = contact_prob[..., 1:2].clamp(0.0, 1.0)
        p_move = contact_prob[..., 2:3].clamp(0.0, 1.0)

        # Hand labels are unconditional while the IMU branch should only own
        # moving frames without either hand anchoring the object.
        p_left_cond = (p_left / p_move.clamp_min(eps)).clamp(0.0, 1.0)
        p_right_cond = (p_right / p_move.clamp_min(eps)).clamp(0.0, 1.0)
        no_hand_cond = (1.0 - p_left_cond) * (1.0 - p_right_cond)
        prior = torch.cat(
            (
                p_left,
                p_right,
                p_move * no_hand_cond,
                1.0 - p_move,
            ),
            dim=-1,
        ).clamp_min(eps)
        return prior / prior.sum(dim=-1, keepdim=True).clamp_min(eps)

    def _fuse_positions(
        self,
        weights,
        left_fk,
        right_fk,
        obj_vel_input,
        obj_trans_init,
        prefix,
        prefix_len,
    ):
        fused_steps = []
        imu_steps = []
        static_steps = []
        dt = 1.0 / FRAME_RATE
        for frame in range(weights.shape[1]):
            if frame < prefix_len:
                fused_steps.append(prefix[:, frame])
                imu_steps.append(prefix[:, frame])
                static_steps.append(prefix[:, frame])
                continue
            previous = fused_steps[-1] if frame > 0 else obj_trans_init
            imu_position = previous + obj_vel_input[:, frame] * dt
            static_position = previous
            imu_steps.append(imu_position)
            static_steps.append(static_position)
            fused_steps.append(
                weights[:, frame, 0:1] * left_fk[:, frame]
                + weights[:, frame, 1:2] * right_fk[:, frame]
                + weights[:, frame, 2:3] * imu_position
                + weights[:, frame, 3:4] * static_position
            )
        return (
            torch.stack(fused_steps, dim=1),
            torch.stack(imu_steps, dim=1),
            {"pred_static_obj_trans": torch.stack(static_steps, dim=1)},
        )

    def _extra_forward_outputs(self, prior, auxiliary_positions):
        return {"gating_prior": prior, **auxiliary_positions}

    def _mask_gate_diagnostics(self, mask, gate_logits_with_prior, prior):
        return gate_logits_with_prior * mask, prior * mask

    @classmethod
    def empty_output(cls, batch_size: int, seq_len: int, device: torch.device):
        output = super().empty_output(batch_size, seq_len, device)
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_weights = torch.zeros(
            batch_size, seq_len, cls.GATING_OUTPUT_DIM, device=device
        )
        output["gating_prior"] = zeros_weights
        output["pred_static_obj_trans"] = zeros_pos
        return output


ObjectTransModule = IMUHOI4HeadsObjectTransModule

__all__ = ["IMUHOI4HeadsObjectTransModule", "ObjectTransModule"]
