"""Evaluation metrics shared by all IMUHOI comparison baselines."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import torch

from Comparisons.common.geometry import local_pose_axis_angle_to_rotmat
from utils.rotation_conversions import matrix_to_axis_angle
from utils.human_pose import select_hand_anchor_positions


@dataclass
class MetricAccumulator:
    sums: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)

    def add(self, name: str, values: torch.Tensor) -> None:
        values = values.detach()
        finite = torch.isfinite(values)
        if not finite.any():
            return
        self.sums[name] = self.sums.get(name, 0.0) + float(values[finite].sum().cpu())
        self.counts[name] = self.counts.get(name, 0) + int(finite.sum().cpu())

    def merge(self, other: "MetricAccumulator") -> None:
        for name, value in other.sums.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
            self.counts[name] = self.counts.get(name, 0) + other.counts[name]

    def add_sequence_means(self, sequence: "MetricAccumulator") -> None:
        """Add one scalar per metric so sequences receive equal weight."""
        for name, value in sequence.means().items():
            self.add(name, torch.tensor(value))

    def means(self) -> Dict[str, float]:
        return {
            name: self.sums[name] / self.counts[name]
            for name in self.sums
            if self.counts.get(name, 0) > 0
        }


def so3_geodesic_degrees(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise geodesic rotation error in degrees."""
    relative = pred.transpose(-1, -2).matmul(target)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5)
    return torch.acos(cosine.clamp(-1.0, 1.0)) * (180.0 / math.pi)


def _second_difference_norm(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 3:
        return x.new_empty((*x.shape[:1], 0, *x.shape[2:-1]))
    acceleration = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
    return torch.linalg.norm(acceleration, dim=-1)


def comparison_metric_stats(
    prediction: Dict[str, torch.Tensor | str | bool],
    batch: Dict[str, torch.Tensor],
) -> MetricAccumulator:
    """Compute per-element metric statistics for complete sequences."""
    stats = MetricAccumulator()
    pred_local = prediction["local_rot"]
    pred_joints = prediction["joints"]
    pred_root = prediction["root_trans"]
    pred_obj = prediction["obj_trans"]
    assert isinstance(pred_local, torch.Tensor)
    assert isinstance(pred_joints, torch.Tensor)
    assert isinstance(pred_root, torch.Tensor)
    assert isinstance(pred_obj, torch.Tensor)

    gt_local = local_pose_axis_angle_to_rotmat(batch["pose"], num_joints=24)
    target_joints = prediction.get("target_joints")
    gt_joints = (
        target_joints
        if isinstance(target_joints, torch.Tensor)
        else batch["position_global"]
    )
    num_joints = min(22, pred_joints.shape[2], gt_joints.shape[2])

    geodesic = so3_geodesic_degrees(
        pred_local[:, :, 1:num_joints], gt_local[:, :, 1:num_joints]
    )
    stats.add("human_geodesic_err_deg", geodesic)

    pred_body_axis = matrix_to_axis_angle(
        pred_local[:, :, 1:num_joints].reshape(-1, 3, 3)
    ).reshape(*pred_local.shape[:2], num_joints - 1, 3)
    gt_body_axis = batch["pose"].reshape(*batch["pose"].shape[:2], -1, 3)[
        :, :, 1:num_joints
    ]
    stats.add(
        "human_ang_err_deg",
        (pred_body_axis - gt_body_axis).abs() * (180.0 / math.pi),
    )

    pred_eval = pred_joints[:, :, :num_joints]
    gt_eval = gt_joints[:, :, :num_joints]
    pred_relative = pred_eval - pred_eval[:, :, :1]
    gt_relative = gt_eval - gt_eval[:, :, :1]
    stats.add(
        "human_pos_err_cm",
        torch.linalg.norm(pred_relative - gt_relative, dim=-1) * 100.0,
    )
    stats.add(
        "human_jitter_mm_frame2",
        _second_difference_norm(pred_eval) * 1000.0,
    )
    stats.add(
        "root_trans_err_cm",
        torch.linalg.norm(pred_root - batch["trans"], dim=-1) * 100.0,
    )

    # A released human-only checkpoint can still produce an object tensor
    # because the comparison wrapper must keep a fixed output schema.  Do not
    # turn that freshly initialized tensor into a misleading object/HOI score.
    # The evaluator sets this flag from checkpoint contents; direct callers
    # retain the historical behavior when the flag is absent.
    if not bool(prediction.get("object_metrics_available", True)):
        return stats

    has_object = batch.get("has_object")
    if has_object is None:
        has_object = torch.ones(pred_obj.shape[0], device=pred_obj.device, dtype=torch.bool)
    else:
        has_object = has_object.bool()

    pred_hands = select_hand_anchor_positions(pred_joints)
    gt_hands = select_hand_anchor_positions(gt_joints)
    left_contact = batch.get("lhand_contact")
    right_contact = batch.get("rhand_contact")

    for index in range(pred_obj.shape[0]):
        if not bool(has_object[index]):
            continue
        gt_obj = batch["obj_trans"][index]
        stats.add(
            "obj_err_cm",
            torch.linalg.norm(pred_obj[index] - gt_obj, dim=-1) * 100.0,
        )
        if pred_obj.shape[1] >= 3:
            obj_jitter = (
                pred_obj[index, 2:]
                - 2.0 * pred_obj[index, 1:-1]
                + pred_obj[index, :-2]
            )
            stats.add(
                "obj_trans_jitter_mm_frame2",
                torch.linalg.norm(obj_jitter, dim=-1) * 1000.0,
            )

        if left_contact is None or right_contact is None:
            continue
        left_mask = left_contact[index].bool()
        right_mask = right_contact[index].bool()
        hoi_errors = []
        if left_mask.any():
            pred_relative_obj = pred_obj[index] - pred_hands[index, :, 0]
            gt_relative_obj = gt_obj - gt_hands[index, :, 0]
            hoi_errors.append(
                torch.linalg.norm(
                    pred_relative_obj[left_mask] - gt_relative_obj[left_mask], dim=-1
                )
                * 100.0
            )
        if right_mask.any():
            pred_relative_obj = pred_obj[index] - pred_hands[index, :, 1]
            gt_relative_obj = gt_obj - gt_hands[index, :, 1]
            hoi_errors.append(
                torch.linalg.norm(
                    pred_relative_obj[right_mask] - gt_relative_obj[right_mask], dim=-1
                )
                * 100.0
            )
        non_contact = ~left_mask & ~right_mask
        if non_contact.any():
            pred_relative_obj = pred_obj[index] - pred_joints[index, :, 0]
            gt_relative_obj = gt_obj - gt_joints[index, :, 0]
            hoi_errors.append(
                torch.linalg.norm(
                    pred_relative_obj[non_contact] - gt_relative_obj[non_contact],
                    dim=-1,
                )
                * 100.0
            )
        if hoi_errors:
            stats.add("hoi_err_cm", torch.cat(hoi_errors))
    return stats


def add_compatibility_aliases(
    metrics: Dict[str, float],
) -> Dict[str, float | None]:
    """Return a stable schema plus names used by ``eval_batch_online.py``."""
    result: Dict[str, float | None] = {
        name: metrics.get(name) for name in METRIC_UNITS
    }
    aliases = {
        "ang_err": "human_ang_err_deg",
        "pos_err": "human_pos_err_cm",
        "human_jitter": "human_jitter_mm_frame2",
        "obj_err": "obj_err_cm",
        "mpjpe": "human_pos_err_cm",
        "mpjre_angle": "human_ang_err_deg",
        "jitter": "human_jitter_mm_frame2",
        "obj_trans_err": "obj_err_cm",
        "hoi_err": "hoi_err_cm",
        "obj_trans_jitter": "obj_trans_jitter_mm_frame2",
    }
    for alias, source in aliases.items():
        result[alias] = result[source]
    return result


METRIC_UNITS = {
    "human_ang_err_deg": "deg",
    "human_geodesic_err_deg": "deg",
    "human_pos_err_cm": "cm",
    "human_jitter_mm_frame2": "mm/frame^2",
    "root_trans_err_cm": "cm",
    "obj_err_cm": "cm",
    "hoi_err_cm": "cm",
    "obj_trans_jitter_mm_frame2": "mm/frame^2",
}
