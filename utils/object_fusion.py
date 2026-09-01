"""Shared contact-to-object-fusion mappings."""
from __future__ import annotations

import torch


def build_contact_fusion_weights(
    contact_prob: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Map [left contact, right contact, object motion] to four weights."""
    p_left = contact_prob[..., 0:1].clamp(0.0, 1.0)
    p_right = contact_prob[..., 1:2].clamp(0.0, 1.0)
    p_move = contact_prob[..., 2:3].clamp(0.0, 1.0)

    p_left_cond = (p_left / p_move.clamp_min(eps)).clamp(0.0, 1.0)
    p_right_cond = (p_right / p_move.clamp_min(eps)).clamp(0.0, 1.0)
    no_hand_cond = (1.0 - p_left_cond) * (1.0 - p_right_cond)

    scores = torch.cat(
        (
            p_left,
            p_right,
            p_move * no_hand_cond,
            1.0 - p_move,
        ),
        dim=-1,
    ).clamp_min(0.0)
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(eps)