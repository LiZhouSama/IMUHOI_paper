from __future__ import annotations

import torch


def compute_hoi_error_at_gt_contact(
    pred_obj_trans: torch.Tensor,
    gt_obj_trans: torch.Tensor,
    pred_hand_pos: torch.Tensor,
    gt_hand_pos: torch.Tensor,
    gt_lhand_contact: torch.Tensor,
    gt_rhand_contact: torch.Tensor,
) -> float:
    """Compute hand-object relative displacement error on GT contact frames."""
    if pred_obj_trans is None or gt_obj_trans is None:
        return float("nan")
    if pred_hand_pos is None or gt_hand_pos is None:
        return float("nan")
    if gt_lhand_contact is None or gt_rhand_contact is None:
        return float("nan")

    contact_masks = (gt_lhand_contact.bool(), gt_rhand_contact.bool())
    relative_errors = []
    for hand_index, contact_mask in enumerate(contact_masks):
        if contact_mask.any():
            pred_relative = pred_obj_trans - pred_hand_pos[:, hand_index, :]
            gt_relative = gt_obj_trans - gt_hand_pos[:, hand_index, :]
            relative_errors.append(
                torch.linalg.norm(pred_relative - gt_relative, dim=-1)[contact_mask]
            )

    if not relative_errors:
        return float("nan")
    return torch.cat(relative_errors, dim=0).mean().item() * 100.0