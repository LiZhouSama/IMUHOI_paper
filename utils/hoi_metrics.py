from __future__ import annotations

import torch


def compute_hoi_error_egoimu(
    pred_obj_trans: torch.Tensor,
    gt_obj_trans: torch.Tensor,
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    gt_lhand_contact: torch.Tensor,
    gt_rhand_contact: torch.Tensor,
) -> float:
    """Wrist-relative contact error plus root-relative error while unheld, cm.

    Each contacting hand contributes one observation; a bimanual frame thus
    contributes twice. A frame without hand contact contributes once at root.
    The evaluator subsequently averages these per-sequence means equally.
    """
    values = (pred_obj_trans, gt_obj_trans, pred_joints, gt_joints,
              gt_lhand_contact, gt_rhand_contact)
    if any(value is None for value in values):
        return float("nan")
    if pred_obj_trans.shape != gt_obj_trans.shape or pred_obj_trans.ndim != 2 or pred_obj_trans.shape[-1] != 3:
        raise ValueError("Object trajectories must both have shape [T,3]")
    frames = pred_obj_trans.shape[0]
    for joints in (pred_joints, gt_joints):
        if joints.ndim != 3 or joints.shape[0] != frames or joints.shape[1] < 22 or joints.shape[-1] != 3:
            raise ValueError("EgoIMU HOI requires [T,J,3] joints with wrists 20/21 and root 0")
    if gt_lhand_contact.shape != (frames,) or gt_rhand_contact.shape != (frames,):
        raise ValueError("Contact labels must have shape [T]")
    if frames == 0:
        return float("nan")
    left, right = gt_lhand_contact.bool(), gt_rhand_contact.bool()
    anchors = [20, 21, 0]
    object_error = pred_obj_trans - gt_obj_trans
    anchor_error = pred_joints[:, anchors] - gt_joints[:, anchors]
    observations = torch.linalg.vector_norm(object_error[:, None] - anchor_error, dim=-1)
    mask = torch.stack((left, right, ~(left | right)), dim=-1)
    return observations[mask].mean().item() * 100.0


def hoi_metric_metadata(config) -> dict:
    protocol = str(getattr(config, "eval_hoi_protocol", "contact_palm"))
    if protocol not in ("contact_palm", "egoimu"):
        raise ValueError(f"Unknown eval_hoi_protocol: {protocol}")
    return {"hoi_metric_protocol": protocol,
            "hoi_hand_anchor": "wrists_20_21" if protocol == "egoimu" else "palms_22_23",
            "hoi_noncontact_reference": "root_0" if protocol == "egoimu" else None,
            "hoi_aggregation": "mean_of_per_sequence_observation_means"}


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
