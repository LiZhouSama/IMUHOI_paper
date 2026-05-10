"""Scheduled input mixing helpers for RNN staged training."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

from configs import _REDUCED_POSE_NAMES
from utils.rotation_conversions import matrix_to_rotation_6d


def prediction_mix_probability(epoch: int, cfg) -> float:
    """Linear scheduled-sampling probability for using predicted inputs."""
    start = int(getattr(cfg, "input_mix_start_epoch", 0))
    end = int(getattr(cfg, "input_mix_end_epoch", 100))
    if end <= start:
        return 1.0 if int(epoch) >= end else 0.0
    prob = (float(epoch) - float(start)) / float(end - start)
    return max(0.0, min(1.0, prob))


def _sample_mask(
    batch_size: int,
    pred_prob: float,
    ref: torch.Tensor,
    trailing_dims: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is not None:
        base = mask.to(device=ref.device, dtype=torch.bool).view(batch_size, -1).any(dim=1)
        return base.view(batch_size, *([1] * trailing_dims))
    prob = max(0.0, min(1.0, float(pred_prob)))
    if prob <= 0.0:
        base = torch.zeros(batch_size, device=ref.device, dtype=torch.bool)
    elif prob >= 1.0:
        base = torch.ones(batch_size, device=ref.device, dtype=torch.bool)
    else:
        base = torch.rand(batch_size, device=ref.device) < prob
    return base.view(batch_size, *([1] * trailing_dims))


def sample_mix_tensor(
    gt: torch.Tensor,
    pred: torch.Tensor,
    pred_prob: float,
    mask: Optional[torch.Tensor] = None,
    return_mask: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Sample-level mix between GT and predicted tensors."""
    if not isinstance(gt, torch.Tensor) or not isinstance(pred, torch.Tensor):
        raise TypeError("sample_mix_tensor expects tensor gt and pred inputs")
    if gt.shape != pred.shape:
        raise ValueError(f"Cannot mix tensors with different shapes: gt={gt.shape}, pred={pred.shape}")
    batch_size = gt.shape[0]
    mix_mask = _sample_mask(batch_size, pred_prob, pred, pred.dim() - 1, mask=mask)
    mixed = torch.where(mix_mask, pred, gt.to(device=pred.device, dtype=pred.dtype))
    if return_mask:
        return mixed, mix_mask
    return mixed


def sample_mix_dict(
    gt: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    keys: Iterable[str],
    pred_prob: float,
) -> Dict[str, torch.Tensor]:
    """Mix matching tensor values from two dicts with one sample-level mask."""
    out: Dict[str, torch.Tensor] = dict(pred)
    mask = None
    for key in keys:
        gt_value = gt.get(key)
        pred_value = pred.get(key)
        if not isinstance(gt_value, torch.Tensor) or not isinstance(pred_value, torch.Tensor):
            continue
        if gt_value.shape != pred_value.shape:
            continue
        if mask is None:
            mixed, mask = sample_mix_tensor(gt_value, pred_value, pred_prob, return_mask=True)
        else:
            mixed = sample_mix_tensor(gt_value, pred_value, pred_prob, mask=mask)
        out[key] = mixed
    return out


def _to_bt(value, shape, device, dtype, default: float = 0.0):
    if not isinstance(value, torch.Tensor):
        return torch.full(shape, default, device=device, dtype=dtype)
    out = value.to(device=device)
    if out.dim() == len(shape) - 1:
        out = out.unsqueeze(0)
    if out.shape[0] == 1 and shape[0] > 1:
        out = out.expand(shape[0], *out.shape[1:])
    if tuple(out.shape[: len(shape)]) != tuple(shape):
        return torch.full(shape, default, device=device, dtype=dtype)
    return out.to(dtype=dtype)


def _pad_joints(value: torch.Tensor, num_joints: int, fill_identity: bool = False) -> torch.Tensor:
    if value.shape[2] == num_joints:
        return value
    if value.shape[2] > num_joints:
        return value[:, :, :num_joints]
    batch_size, seq_len = value.shape[:2]
    missing = num_joints - value.shape[2]
    if fill_identity:
        eye = torch.eye(3, device=value.device, dtype=value.dtype).view(1, 1, 1, 3, 3)
        pad = eye.expand(batch_size, seq_len, missing, -1, -1)
    else:
        pad = torch.zeros(batch_size, seq_len, missing, *value.shape[3:], device=value.device, dtype=value.dtype)
    return torch.cat((value, pad), dim=2)


def build_gt_human_pose_outputs(batch, device, dtype=None, num_joints: int = 24) -> Dict[str, torch.Tensor]:
    """Build an hp_out-shaped dict from batch GT for VC/OT scheduled sampling."""
    human_imu = batch["human_imu"].to(device)
    dtype = dtype or human_imu.dtype
    batch_size, seq_len = human_imu.shape[:2]

    trans = _to_bt(batch.get("trans"), (batch_size, seq_len, 3), device, dtype)
    root_vel = _to_bt(batch.get("root_vel"), (batch_size, seq_len, 3), device, dtype)

    position_global = batch.get("position_global")
    if isinstance(position_global, torch.Tensor):
        position_global = position_global.to(device=device, dtype=dtype)
        if position_global.dim() == 3:
            position_global = position_global.unsqueeze(0)
        if position_global.shape[0] == 1 and batch_size > 1:
            position_global = position_global.expand(batch_size, -1, -1, -1)
        if position_global.shape[:2] != (batch_size, seq_len) or position_global.shape[-1] != 3:
            position_global = None
    else:
        position_global = None
    if position_global is None:
        position_global = torch.zeros(batch_size, seq_len, num_joints, 3, device=device, dtype=dtype)
    position_global = _pad_joints(position_global, num_joints)
    joints_local = position_global - trans.unsqueeze(2)
    hand_pos = torch.stack((position_global[:, :, 20], position_global[:, :, 21]), dim=2)

    rotation_global = batch.get("rotation_global")
    if isinstance(rotation_global, torch.Tensor):
        rotation_global = rotation_global.to(device=device, dtype=dtype)
        if rotation_global.dim() == 4:
            rotation_global = rotation_global.unsqueeze(0)
        if rotation_global.shape[0] == 1 and batch_size > 1:
            rotation_global = rotation_global.expand(batch_size, -1, -1, -1, -1)
        if rotation_global.shape[:2] != (batch_size, seq_len) or rotation_global.shape[-2:] != (3, 3):
            rotation_global = None
    else:
        rotation_global = None
    if rotation_global is None:
        eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
        rotation_global = eye.expand(batch_size, seq_len, num_joints, -1, -1)
    rotation_global = _pad_joints(rotation_global, num_joints, fill_identity=True)
    full_pose_6d = matrix_to_rotation_6d(rotation_global.reshape(-1, 3, 3)).reshape(
        batch_size, seq_len, num_joints, 6
    )

    reduced = batch.get("ori_root_reduced")
    if isinstance(reduced, torch.Tensor):
        reduced = reduced.to(device=device, dtype=dtype)
        if reduced.dim() == 4:
            reduced = reduced.unsqueeze(0)
        if reduced.shape[0] == 1 and batch_size > 1:
            reduced = reduced.expand(batch_size, -1, -1, -1, -1)
        if reduced.shape[:3] == (batch_size, seq_len, len(_REDUCED_POSE_NAMES)) and reduced.shape[-2:] == (3, 3):
            p_pred = matrix_to_rotation_6d(reduced.reshape(-1, 3, 3)).reshape(
                batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6
            )
        else:
            p_pred = torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6, device=device, dtype=dtype)
    else:
        p_pred = torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6, device=device, dtype=dtype)

    return {
        "p_pred": p_pred,
        "pred_full_pose_6d": full_pose_6d,
        "pred_joints_local": joints_local,
        "pred_joints_global": position_global,
        "pred_hand_glb_pos": hand_pos,
        "root_vel_pred": root_vel,
        "root_trans_pred": trans,
    }


def build_gt_velocity_contact_outputs(batch, device, dtype=None) -> Dict[str, torch.Tensor]:
    """Build VC output-shaped GT tensors used by OT scheduled sampling."""
    human_imu = batch["human_imu"].to(device)
    dtype = dtype or human_imu.dtype
    batch_size, seq_len = human_imu.shape[:2]

    def _contact(name: str) -> torch.Tensor:
        value = batch.get(name)
        if isinstance(value, torch.Tensor):
            out = value.to(device=device)
            if out.dim() == 1:
                out = out.unsqueeze(0)
            if out.shape[0] == 1 and batch_size > 1:
                out = out.expand(batch_size, -1)
            if out.shape[:2] == (batch_size, seq_len):
                return out.to(dtype=dtype)
        return torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

    obj_vel = _to_bt(batch.get("obj_vel"), (batch_size, seq_len, 3), device, dtype)
    contact_prob = torch.stack(
        (_contact("lhand_contact"), _contact("rhand_contact"), _contact("obj_contact")),
        dim=-1,
    )
    return {
        "pred_hand_contact_prob": contact_prob,
        "pred_obj_vel": obj_vel,
    }
