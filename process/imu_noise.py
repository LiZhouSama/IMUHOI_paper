import math
from typing import Dict, Optional, Tuple

import torch
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix


# Default feature-space noise profile for a mid-tier Noitom-like IMU.
#
# This module is applied after acceleration smoothing and root-frame feature
# construction, not to raw accelerometer / gyroscope signals.  The values below
# are therefore effective feature-space perturbations rather than MEMS specs.
NOITOM_IMU_NOISE_CFG: Dict[str, float] = {
    "acc_bias_std": 0.04,                 # m/s^2 initial bias
    "acc_bias_rw_std": 0.005,             # m/s^2 * sqrt(s) random walk
    "acc_noise_std": 0.06,                # m/s^2 white-noise proxy after smoothing
    "acc_scale_std": 0.005,               # dimensionless scale factor error
    "acc_misalignment_std": math.radians(0.3),  # rad, static axis misalignment
    "acc_spike_prob": 0.0,                # enable only after measuring real glitches
    "acc_spike_std": 0.0,                 # m/s^2 spike magnitude
    "acc_clip": 0.0,                      # feature-space acceleration is not raw sensor range
    "acc_drift_per_s": 0.0,               # avoid duplicating random-walk drift

    "ori_bias_std": math.radians(1.0),    # rad, static attitude / calibration residual
    "ori_rw_std": math.radians(0.01),     # rad * sqrt(s) attitude random walk
    "ori_noise_std": math.radians(0.10),  # rad, fused-attitude white noise
    "ori_misalignment_std": 0.0,          # avoid double-counting static orientation error
    "ori_spike_prob": 0.0,                # enable only after measuring real glitches
    "ori_spike_std": 0.0,                 # rad attitude glitch magnitude

    "sample_drop_prob": 0.0,              # model packet loss separately when needed
    "sample_hold_prob": 0.0,              # model packet loss separately when needed
}


def merge_noise_cfg(custom_cfg: Optional[Dict[str, float]]) -> Dict[str, float]:
    cfg = dict(NOITOM_IMU_NOISE_CFG)
    if custom_cfg:
        cfg.update(custom_cfg)
    return cfg


def _normalize_acc(acc: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Ensure acc shape is [T, N, 3]."""
    if acc.dim() == 2:  # [T, 3]
        return acc.unsqueeze(1), True
    return acc, False


def _normalize_rot(rot6d: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Ensure rot6d shape is [T, N, 6]."""
    if rot6d.dim() == 2:  # [T, 6]
        return rot6d.unsqueeze(1), True
    return rot6d, False


def _apply_acc_noise(acc: torch.Tensor, cfg: Dict[str, float], fps: float) -> torch.Tensor:
    seq_len, num_sensors, _ = acc.shape
    device, dtype = acc.device, acc.dtype
    dt = 1.0 / float(fps)
    sqrt_dt = math.sqrt(dt)

    bias0 = torch.randn(num_sensors, 3, device=device, dtype=dtype) * cfg["acc_bias_std"]
    bias_walk = torch.randn(seq_len, num_sensors, 3, device=device, dtype=dtype) * cfg["acc_bias_rw_std"] * sqrt_dt
    bias = bias0.unsqueeze(0) + torch.cumsum(bias_walk, dim=0)

    acc_base = acc + bias

    if cfg.get("acc_drift_per_s", 0.0) > 0:
        t = torch.arange(seq_len, device=device, dtype=dtype).view(seq_len, 1, 1) * dt
        drift_dir = torch.randn(num_sensors, 3, device=device, dtype=dtype) * cfg["acc_drift_per_s"]
        acc_base = acc_base + t * drift_dir

    if cfg.get("acc_scale_std", 0.0) > 0:
        scale = 1.0 + torch.randn(num_sensors, 3, device=device, dtype=dtype) * cfg["acc_scale_std"]
        acc_base = acc_base * scale.view(1, num_sensors, 3)

    if cfg.get("acc_misalignment_std", 0.0) > 0:
        misalign = torch.randn(num_sensors, 3, device=device, dtype=dtype) * cfg["acc_misalignment_std"]
        misalign_R = axis_angle_to_matrix(misalign).to(dtype=dtype)
        acc_flat = acc_base.reshape(seq_len * num_sensors, 3)
        misalign_R_expanded = misalign_R.unsqueeze(0).repeat(seq_len, 1, 1, 1).reshape(seq_len * num_sensors, 3, 3)
        acc_base = torch.bmm(misalign_R_expanded, acc_flat.unsqueeze(-1)).reshape(seq_len, num_sensors, 3)

    acc_noisy = acc_base + torch.randn_like(acc_base) * cfg["acc_noise_std"]

    if cfg.get("acc_spike_prob", 0.0) > 0:
        spike_mask = (torch.rand(seq_len, num_sensors, device=device) < cfg["acc_spike_prob"]).unsqueeze(-1)
        acc_noisy = acc_noisy + spike_mask * (torch.randn_like(acc_noisy) * cfg["acc_spike_std"])

    clip = cfg.get("acc_clip", 0.0)
    if clip and clip > 0:
        acc_noisy = acc_noisy.clamp(-clip, clip)

    return acc_noisy


def _apply_rot_noise(rot6d: torch.Tensor, cfg: Dict[str, float], fps: float) -> torch.Tensor:
    seq_len, num_sensors, _ = rot6d.shape
    device, dtype = rot6d.device, rot6d.dtype
    dt = 1.0 / float(fps)
    sqrt_dt = math.sqrt(dt)

    rot_mat = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(seq_len, num_sensors, 3, 3)

    bias0 = torch.randn(num_sensors, 3, device=device, dtype=dtype) * cfg["ori_bias_std"]
    bias_walk = torch.randn(seq_len, num_sensors, 3, device=device, dtype=dtype) * cfg["ori_rw_std"] * sqrt_dt
    bias = bias0.unsqueeze(0) + torch.cumsum(bias_walk, dim=0)

    noise_white = torch.randn(seq_len, num_sensors, 3, device=device, dtype=dtype) * cfg["ori_noise_std"]
    aa_total = bias + noise_white

    if cfg.get("ori_spike_prob", 0.0) > 0:
        spike_mask = (torch.rand(seq_len, num_sensors, device=device) < cfg["ori_spike_prob"]).unsqueeze(-1)
        aa_total = aa_total + spike_mask * (torch.randn_like(aa_total) * cfg["ori_spike_std"])

    misalign_std = cfg.get("ori_misalignment_std", 0.0)
    if misalign_std > 0:
        misalign = torch.randn(num_sensors, 3, device=device, dtype=dtype) * misalign_std
        misalign_R = axis_angle_to_matrix(misalign).to(dtype=dtype)
        misalign_R_expanded = misalign_R.unsqueeze(0).expand(seq_len, -1, -1, -1)
    else:
        misalign_R_expanded = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(seq_len, num_sensors, 3, 3)

    noise_R = axis_angle_to_matrix(aa_total.reshape(-1, 3)).reshape(seq_len, num_sensors, 3, 3)
    rot_noisy = torch.matmul(rot_mat, torch.matmul(misalign_R_expanded, noise_R))
    rot6d_noisy = matrix_to_rotation_6d(rot_noisy.reshape(-1, 3, 3)).reshape(seq_len, num_sensors, 6)
    return rot6d_noisy


def _apply_drop_and_hold(
    acc: torch.Tensor,
    rot6d: torch.Tensor,
    cfg: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    drop_prob = cfg.get("sample_drop_prob", 0.0)
    hold_prob = cfg.get("sample_hold_prob", 0.0)
    if drop_prob <= 0 and hold_prob <= 0:
        return acc, rot6d

    seq_len, num_sensors = acc.shape[:2]
    acc_out = acc.clone()
    rot_out = rot6d.clone()

    for t in range(1, seq_len):
        drop_mask = torch.rand(num_sensors, device=acc.device) < drop_prob
        hold_mask = torch.rand(num_sensors, device=acc.device) < hold_prob
        mask = drop_mask | hold_mask
        if mask.any():
            acc_out[t, mask] = acc_out[t - 1, mask]
            rot_out[t, mask] = rot_out[t - 1, mask]

    return acc_out, rot_out


def apply_imu_noise(
    acc: torch.Tensor,
    rot6d: torch.Tensor,
    fps: float,
    noise_cfg: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Apply a realistic IMU noise model (bias, random walk, misalignment, spikes, dropouts).

    Args:
        acc: [..., 3] accelerations shaped [T, 3] or [T, N, 3].
        rot6d: [..., 6] orientations shaped [T, 6] or [T, N, 6].
        fps: frame rate.
        noise_cfg: optional overrides for NOITOM_IMU_NOISE_CFG.

    Returns:
        noisy_acc, noisy_rot6d, merged_noise_cfg
    """
    cfg = merge_noise_cfg(noise_cfg)

    acc_norm, acc_squeezed = _normalize_acc(acc)
    rot_norm, rot_squeezed = _normalize_rot(rot6d)

    acc_noisy = _apply_acc_noise(acc_norm, cfg, fps)
    rot_noisy = _apply_rot_noise(rot_norm, cfg, fps)
    acc_noisy, rot_noisy = _apply_drop_and_hold(acc_noisy, rot_noisy, cfg)

    if acc_squeezed:
        acc_noisy = acc_noisy.squeeze(1)
    if rot_squeezed:
        rot_noisy = rot_noisy.squeeze(1)

    return acc_noisy, rot_noisy, cfg
