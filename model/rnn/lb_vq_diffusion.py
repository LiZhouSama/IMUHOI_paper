"""Discrete mesh-aware prior and observation-conditioned diffusion for OT bone lengths.

This module is intentionally independent from :class:`ObjectTransModule`.  It
implements two separately trainable stages:

1. A privileged VQ bottleneck encodes predicted human/contact streams together
   with canonical object geometry and ground-truth object pose.  Its decoder
   reconstructs left/right hand-object length quantiles for the current window.
2. A conditional latent diffusion model predicts the frozen VQ latent sequence
   from observation-only streams (HP/VC predictions and object IMU).

Canonical object points are converted to a BPS distance descriptor once per
window.  The point cloud is never expanded to one copy per frame.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import _REDUCED_POSE_NAMES


DEFAULT_LB_QUANTILES = (0.1, 0.5, 0.9)


def _prior_cfg(cfg):
    return getattr(cfg, "lb_vq_prior", cfg)


def _cfg_value(cfg, name: str, default):
    block = _prior_cfg(cfg)
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _as_bt(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.dim() == 2:
        value = value.unsqueeze(0)
    if value.dim() < 3:
        raise ValueError(f"{name} must have [B,T,...] shape, got {tuple(value.shape)}")
    return value


def _check_bt(reference: torch.Tensor, value: torch.Tensor, name: str) -> torch.Tensor:
    value = _as_bt(value, name)
    if value.shape[:2] != reference.shape[:2]:
        raise ValueError(
            f"{name} batch/time shape {tuple(value.shape[:2])} does not match "
            f"{tuple(reference.shape[:2])}"
        )
    return value.to(device=reference.device, dtype=reference.dtype)


def _prepare_has_object(has_object, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    if has_object is None:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    if isinstance(has_object, torch.Tensor):
        mask = has_object.to(device=device, dtype=torch.bool)
    else:
        mask = torch.as_tensor(has_object, device=device, dtype=torch.bool)
    if mask.dim() == 0:
        mask = mask.view(1)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, *mask.shape[1:])
    if mask.shape[0] != batch_size:
        raise ValueError(f"has_object batch mismatch: expected {batch_size}, got {tuple(mask.shape)}")
    if mask.dim() == 1:
        return mask
    if mask.shape[1] not in {1, seq_len}:
        raise ValueError(f"has_object time mismatch: expected 1 or {seq_len}, got {tuple(mask.shape)}")
    return mask.reshape(batch_size, -1).any(dim=1)


def compute_lb_quantile_targets(
    lhand_lb: torch.Tensor,
    rhand_lb: torch.Tensor,
    lhand_contact: torch.Tensor,
    rhand_contact: torch.Tensor,
    *,
    has_object=None,
    quantiles: Sequence[float] = DEFAULT_LB_QUANTILES,
    min_contact_frames: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-window hand length quantiles using only GT contact frames.

    Returns:
        targets: ``[B,2,Q]`` in metres.
        valid_hands: ``[B,2]``; false hands must be excluded from reconstruction loss.
        contact_counts: ``[B,2]``.
    """
    if lhand_lb.dim() == 1:
        lhand_lb = lhand_lb.unsqueeze(0)
    if rhand_lb.dim() == 1:
        rhand_lb = rhand_lb.unsqueeze(0)
    if lhand_contact.dim() == 1:
        lhand_contact = lhand_contact.unsqueeze(0)
    if rhand_contact.dim() == 1:
        rhand_contact = rhand_contact.unsqueeze(0)
    if lhand_lb.shape != rhand_lb.shape or lhand_lb.shape != lhand_contact.shape or lhand_lb.shape != rhand_contact.shape:
        raise ValueError(
            "l_b/contact tensors must share [B,T] shape, got "
            f"{tuple(lhand_lb.shape)}, {tuple(rhand_lb.shape)}, "
            f"{tuple(lhand_contact.shape)}, {tuple(rhand_contact.shape)}"
        )

    batch_size, seq_len = lhand_lb.shape
    device = lhand_lb.device
    dtype = lhand_lb.dtype
    q_values = torch.as_tensor(tuple(float(q) for q in quantiles), device=device, dtype=dtype)
    if q_values.numel() == 0 or bool(((q_values < 0.0) | (q_values > 1.0)).any().item()):
        raise ValueError(f"quantiles must lie in [0,1], got {tuple(quantiles)}")

    object_valid = _prepare_has_object(has_object, batch_size, seq_len, device)
    lengths = torch.stack((lhand_lb, rhand_lb), dim=1)
    contacts = torch.stack((lhand_contact, rhand_contact), dim=1).bool()
    contacts = contacts & object_valid[:, None, None]
    counts = contacts.sum(dim=-1)
    valid_hands = counts >= int(max(1, min_contact_frames))
    targets = torch.zeros(batch_size, 2, q_values.numel(), device=device, dtype=dtype)

    # Each hand can have a different number of valid contact frames, so a small
    # Bx2 loop is clearer and cheaper than padding all samples to a dense tensor.
    for batch_idx in range(batch_size):
        for hand_idx in range(2):
            if not bool(valid_hands[batch_idx, hand_idx].item()):
                continue
            values = lengths[batch_idx, hand_idx][contacts[batch_idx, hand_idx]]
            values = values[torch.isfinite(values)]
            if values.numel() < int(max(1, min_contact_frames)):
                valid_hands[batch_idx, hand_idx] = False
                continue
            targets[batch_idx, hand_idx] = torch.quantile(values, q_values)

    return targets, valid_hands, counts


def masked_lb_quantile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_hands: torch.Tensor,
    *,
    log_space: bool = True,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.dim() != 3:
        raise ValueError(f"quantile shape mismatch: prediction={prediction.shape}, target={target.shape}")
    mask = valid_hands.to(device=prediction.device, dtype=torch.bool)
    if mask.shape != prediction.shape[:2]:
        raise ValueError(f"valid_hands must be {prediction.shape[:2]}, got {mask.shape}")
    if not bool(mask.any().item()):
        return prediction.sum() * 0.0
    pred_valid = prediction[mask]
    target_valid = target.to(device=prediction.device, dtype=prediction.dtype)[mask]
    if log_space:
        pred_valid = torch.log1p(pred_valid.clamp_min(0.0))
        target_valid = torch.log1p(target_valid.clamp_min(0.0))
    return F.smooth_l1_loss(pred_valid, target_valid)


def masked_lb_quantile_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_hands: torch.Tensor,
) -> torch.Tensor:
    mask = valid_hands.to(device=prediction.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        return prediction.sum() * 0.0
    return (prediction[mask] - target.to(device=prediction.device, dtype=prediction.dtype)[mask]).abs().mean()


def _resolve_hp_vc_streams(hp_out: Dict, vc_out: Dict) -> Tuple[torch.Tensor, ...]:
    if not isinstance(hp_out, dict) or not isinstance(vc_out, dict):
        raise ValueError("hp_out and vc_out must be dictionaries")
    pose = _as_bt(hp_out.get("p_pred"), "hp_out.p_pred")
    batch_size, seq_len = pose.shape[:2]

    root = hp_out.get("root_trans_pred")
    if not isinstance(root, torch.Tensor):
        root = torch.zeros(batch_size, seq_len, 3, device=pose.device, dtype=pose.dtype)
    root = _check_bt(pose, root, "hp_out.root_trans_pred")
    if root.shape[-1] != 3:
        raise ValueError(f"root_trans_pred must end in 3, got {root.shape}")

    hands = hp_out.get("pred_palm_glb_pos", hp_out.get("pred_hand_glb_pos"))
    hands = _check_bt(pose, hands, "hp_out.pred_palm_glb_pos")
    if hands.shape[2:] != (2, 3):
        raise ValueError(f"hand positions must be [B,T,2,3], got {hands.shape}")

    contact = _check_bt(pose, vc_out.get("pred_hand_contact_prob"), "vc_out.pred_hand_contact_prob")
    if contact.shape[-1] != 3:
        raise ValueError(f"contact probabilities must end in 3, got {contact.shape}")
    return pose, root, hands, contact


def build_lb_observation_sequence(hp_out: Dict, vc_out: Dict, obj_imu: torch.Tensor) -> torch.Tensor:
    """Build inference-available condition ``[pose, root, hands, contact, obj_imu]``.

    Translation-sensitive streams are represented relative to the predicted root
    and the first frame so the model cannot memorize dataset world coordinates.
    """
    pose, root, hands, contact = _resolve_hp_vc_streams(hp_out, vc_out)
    obj_imu = _check_bt(pose, obj_imu, "obj_imu")
    root_delta = root - root[:, :1]
    hands_root = hands - root.unsqueeze(2)
    return torch.cat(
        (pose, root_delta, hands_root.flatten(2), contact, obj_imu),
        dim=-1,
    )


def build_lb_privileged_sequence(
    hp_out: Dict,
    vc_out: Dict,
    obj_rot: torch.Tensor,
    obj_trans: torch.Tensor,
    obj_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build VQ encoder input with privileged object pose but no object IMU."""
    pose, root, hands, contact = _resolve_hp_vc_streams(hp_out, vc_out)
    obj_trans = _check_bt(pose, obj_trans, "obj_trans")
    if obj_trans.shape[-1] != 3:
        raise ValueError(f"obj_trans must end in 3, got {obj_trans.shape}")

    obj_rot = _check_bt(pose, obj_rot, "obj_rot")
    if obj_rot.shape[-1] == 6:
        obj_rot_6d = obj_rot
    elif obj_rot.shape[-2:] == (3, 3):
        obj_rot_6d = obj_rot[..., :2, :].reshape(*obj_rot.shape[:2], 6)
    else:
        raise ValueError(f"obj_rot must be [B,T,6] or [B,T,3,3], got {obj_rot.shape}")

    if not isinstance(obj_scale, torch.Tensor):
        obj_scale = torch.ones(*pose.shape[:2], 1, device=pose.device, dtype=pose.dtype)
    else:
        obj_scale = obj_scale.to(device=pose.device, dtype=pose.dtype)
        if obj_scale.dim() == 1:
            obj_scale = obj_scale.unsqueeze(0)
        if obj_scale.dim() == 2:
            obj_scale = obj_scale.unsqueeze(-1)
        obj_scale = _check_bt(pose, obj_scale, "obj_scale")
        if obj_scale.shape[-1] != 1:
            raise ValueError(f"obj_scale must be [B,T] or [B,T,1], got {obj_scale.shape}")

    root_delta = root - root[:, :1]
    hands_root = hands - root.unsqueeze(2)
    obj_root = obj_trans - root
    return torch.cat(
        (
            pose,
            root_delta,
            hands_root.flatten(2),
            contact,
            obj_rot_6d,
            obj_root,
            obj_scale,
        ),
        dim=-1,
    )


def _downsample_mean(sequence: torch.Tensor, factor: int) -> torch.Tensor:
    factor = int(max(1, factor))
    if factor == 1:
        return sequence
    batch_size, seq_len, dim = sequence.shape
    padded_len = int(math.ceil(seq_len / factor) * factor)
    if padded_len != seq_len:
        pad = sequence[:, -1:].expand(batch_size, padded_len - seq_len, dim)
        sequence = torch.cat((sequence, pad), dim=1)
    return sequence.reshape(batch_size, padded_len // factor, factor, dim).mean(dim=2)


class _PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError(f"sequence length {x.shape[1]} exceeds positional limit {self.pe.shape[1]}")
        return x + self.pe[:, : x.shape[1]].to(device=x.device, dtype=x.dtype)


class _BPSMeshEncoder(nn.Module):
    """Encode one canonical point cloud per sample through a fixed BPS basis."""

    def __init__(
        self,
        num_basis: int,
        radius: float,
        hidden_dim: int,
        output_dim: int,
        seed: int = 2027,
    ):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        directions = torch.randn(num_basis, 3, generator=generator)
        directions = F.normalize(directions, dim=-1)
        radial = torch.rand(num_basis, 1, generator=generator).pow(1.0 / 3.0)
        basis = directions * radial * float(radius)
        self.register_buffer("basis", basis, persistent=True)
        self.radius = float(max(radius, 1e-4))
        self.net = nn.Sequential(
            nn.LayerNorm(num_basis + 5),
            nn.Linear(num_basis + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if not isinstance(points, torch.Tensor) or points.dim() != 3 or points.shape[-1] != 3:
            raise ValueError(f"mesh points must be [B,N,3], got {getattr(points, 'shape', None)}")
        points = torch.nan_to_num(points)
        centered = points - points.mean(dim=1, keepdim=True)
        basis = self.basis.to(device=points.device, dtype=points.dtype).unsqueeze(0).expand(points.shape[0], -1, -1)
        bps_dist = torch.cdist(basis, centered).amin(dim=-1)
        bps_dist = torch.log1p(bps_dist / self.radius)

        extent = centered.amax(dim=1) - centered.amin(dim=1)
        point_radius = centered.norm(dim=-1)
        rms_radius = centered.square().sum(dim=-1).mean(dim=1, keepdim=True).sqrt()
        max_radius = point_radius.amax(dim=1, keepdim=True)
        geometry_stats = torch.cat((extent, rms_radius, max_radius), dim=-1)
        return self.net(torch.cat((bps_dist, geometry_stats), dim=-1))


class _TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        output_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        downsample_factor: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.downsample_factor = int(max(1, downsample_factor))
        max_latent_len = int(math.ceil(max_seq_len / self.downsample_factor)) + 2
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
        )
        self.position = _PositionalEncoding(model_dim, max_latent_len)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim))
        self.output_proj = nn.Linear(model_dim, output_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.dim() != 3:
            raise ValueError(f"temporal sequence must be [B,T,D], got {sequence.shape}")
        sequence = _downsample_mean(sequence, self.downsample_factor)
        hidden = self.position(self.input_proj(sequence))
        return self.output_proj(self.encoder(hidden))


class VectorQuantizer(nn.Module):
    """Straight-through vector quantizer with code usage diagnostics."""

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        commitment_beta: float = 0.25,
        soft_assignment_temperature: float = 0.1,
    ):
        super().__init__()
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.commitment_beta = float(commitment_beta)
        self.soft_assignment_temperature = float(max(soft_assignment_temperature, 1e-4))
        self.embedding = nn.Embedding(self.num_codes, self.code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_codes, 1.0 / self.num_codes)

    def _distances(self, z: torch.Tensor) -> torch.Tensor:
        flat = z.reshape(-1, self.code_dim)
        codebook = self.embedding.weight.to(device=z.device, dtype=z.dtype)
        return (
            flat.square().sum(dim=-1, keepdim=True)
            - 2.0 * flat @ codebook.t()
            + codebook.square().sum(dim=-1).unsqueeze(0)
        )

    def nearest(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = self._distances(z)
        indices = distances.argmin(dim=-1)
        codebook = self.embedding.weight.to(device=z.device, dtype=z.dtype)
        quantized = F.embedding(indices, codebook).reshape_as(z)
        min_distance = distances.gather(1, indices.unsqueeze(1)).reshape(*z.shape[:-1])
        return quantized, indices.reshape(*z.shape[:-1]), min_distance

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_q, indices, min_distance = self.nearest(z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach()) * self.commitment_beta
        z_st = z_e + (z_q - z_e).detach()

        flat_indices = indices.reshape(-1)
        # Keep entropy diagnostics in fp32. Under AMP, 1e-8 underflows in fp16;
        # then an unused code produces 0 * log(0) = NaN.
        hard_prob = F.one_hot(flat_indices, num_classes=self.num_codes).float().mean(dim=0)
        perplexity = torch.exp(-(hard_prob * torch.log(hard_prob.clamp_min(1e-8))).sum())
        active_fraction = (hard_prob > 0).float().mean()

        soft_prob = torch.softmax(
            -self._distances(z_e).float() / self.soft_assignment_temperature,
            dim=-1,
        ).mean(dim=0)
        usage_loss = ((soft_prob * self.num_codes) - 1.0).square().mean()
        return {
            "z_e": z_e,
            "z_q": z_q,
            "z_st": z_st,
            "code_indices": indices,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "usage_loss": usage_loss,
            "quantization_error": min_distance.mean(),
            "perplexity": perplexity,
            "active_code_fraction": active_fraction,
        }


def _inverse_softplus(value: float) -> float:
    value = float(max(value, 1e-6))
    return math.log(math.expm1(value))


class _LBQuantileDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        model_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        max_latent_len: int,
    ):
        super().__init__()
        self.memory_proj = nn.Linear(latent_dim, model_dim)
        self.position = _PositionalEncoding(model_dim, max_latent_len)
        self.hand_queries = nn.Parameter(torch.randn(2, model_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim))
        self.output = nn.Linear(model_dim, 3)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(
                torch.tensor(
                    (_inverse_softplus(0.25), _inverse_softplus(0.10), _inverse_softplus(0.10)),
                    dtype=self.output.bias.dtype,
                )
            )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        memory = self.position(self.memory_proj(latents))
        queries = self.hand_queries.to(device=latents.device, dtype=latents.dtype).unsqueeze(0).expand(latents.shape[0], -1, -1)
        raw = self.output(self.decoder(queries, memory))
        q10 = F.softplus(raw[..., 0:1])
        q50 = q10 + F.softplus(raw[..., 1:2])
        q90 = q50 + F.softplus(raw[..., 2:3])
        return torch.cat((q10, q50, q90), dim=-1)


class LBVQAutoEncoder(nn.Module):
    """Privileged temporal VQ bottleneck that reconstructs window l_b quantiles."""

    def __init__(self, cfg):
        super().__init__()
        pose_dim = len(_REDUCED_POSE_NAMES) * 6
        self.privileged_dim = pose_dim + 3 + 6 + 3 + 6 + 3 + 1
        self.temporal_downsample = int(_cfg_value(cfg, "temporal_downsample", 8))
        self.latent_dim = int(_cfg_value(cfg, "latent_dim", 32))
        self.num_codes = int(_cfg_value(cfg, "num_codes", 64))
        max_seq_len = int(_cfg_value(cfg, "max_seq_len", 256))
        mesh_dim = int(_cfg_value(cfg, "mesh_feature_dim", 64))
        model_dim = int(_cfg_value(cfg, "vq_model_dim", 128))
        nhead = int(_cfg_value(cfg, "vq_nhead", 4))
        encoder_layers = int(_cfg_value(cfg, "vq_encoder_layers", 3))
        decoder_layers = int(_cfg_value(cfg, "vq_decoder_layers", 2))
        dropout = float(_cfg_value(cfg, "vq_dropout", 0.1))
        max_latent_len = int(math.ceil(max_seq_len / self.temporal_downsample)) + 2

        self.mesh_encoder = _BPSMeshEncoder(
            num_basis=int(_cfg_value(cfg, "bps_num_basis", 128)),
            radius=float(_cfg_value(cfg, "bps_radius", 1.0)),
            hidden_dim=int(_cfg_value(cfg, "mesh_hidden_dim", 128)),
            output_dim=mesh_dim,
            seed=int(_cfg_value(cfg, "bps_seed", 2027)),
        )
        self.temporal_encoder = _TemporalEncoder(
            input_dim=self.privileged_dim + mesh_dim,
            model_dim=model_dim,
            output_dim=self.latent_dim,
            nhead=nhead,
            num_layers=encoder_layers,
            dropout=dropout,
            downsample_factor=self.temporal_downsample,
            max_seq_len=max_seq_len,
        )
        self.quantizer = VectorQuantizer(
            num_codes=self.num_codes,
            code_dim=self.latent_dim,
            commitment_beta=float(_cfg_value(cfg, "commitment_beta", 0.25)),
            soft_assignment_temperature=float(_cfg_value(cfg, "soft_assignment_temperature", 0.1)),
        )
        self.quantile_decoder = _LBQuantileDecoder(
            latent_dim=self.latent_dim,
            model_dim=model_dim,
            nhead=nhead,
            num_layers=decoder_layers,
            dropout=dropout,
            max_latent_len=max_latent_len,
        )

    def encode(self, privileged_sequence: torch.Tensor, mesh_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        if privileged_sequence.dim() != 3 or privileged_sequence.shape[-1] != self.privileged_dim:
            raise ValueError(
                f"privileged_sequence must be [B,T,{self.privileged_dim}], got {privileged_sequence.shape}"
            )
        mesh_feature = self.mesh_encoder(mesh_points)
        mesh_sequence = mesh_feature.unsqueeze(1).expand(-1, privileged_sequence.shape[1], -1)
        z_e = self.temporal_encoder(torch.cat((privileged_sequence, mesh_sequence), dim=-1))
        output = self.quantizer(z_e)
        output["mesh_feature"] = mesh_feature
        return output

    def decode_latents(self, latents: torch.Tensor, *, project_to_codebook: bool = False) -> Dict[str, torch.Tensor]:
        if latents.dim() != 3 or latents.shape[-1] != self.latent_dim:
            raise ValueError(f"latents must be [B,L,{self.latent_dim}], got {latents.shape}")
        # Decoding is also called outside an autocast context during validation
        # sampling and diagnostic baselines. Keep the decoder input compatible
        # with its fp32 parameters when the producer ran under AMP.
        decoder_dtype = self.quantile_decoder.memory_proj.weight.dtype
        latents = latents.to(dtype=decoder_dtype)
        code_indices = None
        decoded_latents = latents
        if project_to_codebook:
            decoded_latents, code_indices, _ = self.quantizer.nearest(latents)
        return {
            "lb_quantiles": self.quantile_decoder(decoded_latents),
            "decoded_latents": decoded_latents,
            "code_indices": code_indices,
        }

    def forward(self, privileged_sequence: torch.Tensor, mesh_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        output = self.encode(privileged_sequence, mesh_points)
        decoded = self.decode_latents(output["z_st"])
        # Preserve the encoder's actual VQ assignments.  A continuous decoder
        # call has no projected indices and must not overwrite them with None.
        if decoded.get("code_indices") is None:
            decoded.pop("code_indices", None)
        output.update(decoded)
        return output


class _TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequency = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale
        )
        angle = timesteps.float().unsqueeze(-1) * frequency.unsqueeze(0)
        embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class _LatentDenoiser(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        model_dim: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        max_latent_len: int,
        cond_dropout: float,
    ):
        super().__init__()
        self.cond_dropout = float(max(0.0, min(1.0, cond_dropout)))
        self.input_proj = nn.Linear(latent_dim, model_dim)
        self.cond_proj = nn.Linear(cond_dim, model_dim)
        self.time_embed = _TimestepEmbedding(model_dim)
        self.position = _PositionalEncoding(model_dim, max_latent_len)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(model_dim))
        self.output = nn.Linear(model_dim, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if condition.shape[:2] != x_t.shape[:2]:
            raise ValueError(f"condition/latent time mismatch: {condition.shape} vs {x_t.shape}")
        condition_use = condition
        if self.training and self.cond_dropout > 0.0:
            keep = (torch.rand(x_t.shape[0], 1, 1, device=x_t.device) >= self.cond_dropout).to(x_t.dtype)
            condition_use = condition_use * keep
        hidden = self.input_proj(x_t) + self.cond_proj(condition_use)
        hidden = hidden + self.time_embed(timesteps).to(dtype=hidden.dtype).unsqueeze(1)
        hidden = self.position(hidden)
        return self.output(self.encoder(hidden))


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999).float()


def _extract(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return values.gather(0, timesteps).view(-1, 1, 1).to(device=reference.device, dtype=reference.dtype)


class LBLatentDiffusion(nn.Module):
    """Predict a low-rate VQ latent sequence from observation-only HP/VC streams."""

    def __init__(self, cfg):
        super().__init__()
        pose_dim = len(_REDUCED_POSE_NAMES) * 6
        obj_imu_dim = int(getattr(cfg, "obj_imu_dim", 9))
        self.observation_dim = pose_dim + 3 + 6 + 3 + obj_imu_dim
        self.latent_dim = int(_cfg_value(cfg, "latent_dim", 32))
        self.temporal_downsample = int(_cfg_value(cfg, "temporal_downsample", 8))
        self.timesteps = int(_cfg_value(cfg, "diffusion_timesteps", 1000))
        self.prediction_type = str(_cfg_value(cfg, "diffusion_prediction_type", "x0")).lower()
        if self.prediction_type not in {"x0", "eps"}:
            raise ValueError(f"diffusion_prediction_type must be x0 or eps, got {self.prediction_type}")

        max_seq_len = int(_cfg_value(cfg, "max_seq_len", 256))
        cond_dim = int(_cfg_value(cfg, "diffusion_cond_dim", 128))
        model_dim = int(_cfg_value(cfg, "diffusion_model_dim", 128))
        nhead = int(_cfg_value(cfg, "diffusion_nhead", 4))
        layers = int(_cfg_value(cfg, "diffusion_layers", 4))
        dropout = float(_cfg_value(cfg, "diffusion_dropout", 0.1))
        max_latent_len = int(math.ceil(max_seq_len / self.temporal_downsample)) + 2

        self.observation_encoder = _TemporalEncoder(
            input_dim=self.observation_dim,
            model_dim=cond_dim,
            output_dim=cond_dim,
            nhead=nhead,
            num_layers=int(_cfg_value(cfg, "observation_encoder_layers", 2)),
            dropout=dropout,
            downsample_factor=self.temporal_downsample,
            max_seq_len=max_seq_len,
        )
        self.denoiser = _LatentDenoiser(
            latent_dim=self.latent_dim,
            cond_dim=cond_dim,
            model_dim=model_dim,
            nhead=nhead,
            num_layers=layers,
            dropout=dropout,
            max_latent_len=max_latent_len,
            cond_dropout=float(_cfg_value(cfg, "diffusion_cond_dropout", 0.05)),
        )

        betas = _cosine_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=True)
        self.register_buffer("alphas_cumprod", alpha_bar, persistent=True)
        self.register_buffer("sqrt_alphas_cumprod", alpha_bar.sqrt(), persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alpha_bar).sqrt(), persistent=False)

    def encode_condition(self, observation_sequence: torch.Tensor) -> torch.Tensor:
        if observation_sequence.dim() != 3 or observation_sequence.shape[-1] != self.observation_dim:
            raise ValueError(
                f"observation_sequence must be [B,T,{self.observation_dim}], got {observation_sequence.shape}"
            )
        return self.observation_encoder(observation_sequence)

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        return (
            _extract(self.sqrt_alphas_cumprod, timesteps, x0) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0) * noise
        )

    def _predict(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_output = self.denoiser(x_t, timesteps, condition)
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, x_t)
        sqrt_noise = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t)
        if self.prediction_type == "x0":
            x0_pred = model_output
            eps_pred = (x_t - sqrt_alpha * x0_pred) / sqrt_noise.clamp_min(1e-8)
        else:
            eps_pred = model_output
            x0_pred = (x_t - sqrt_noise * eps_pred) / sqrt_alpha.clamp_min(1e-8)
        return x0_pred, eps_pred, model_output

    def forward(
        self,
        observation_sequence: torch.Tensor,
        target_latents: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        condition = self.encode_condition(observation_sequence)
        if target_latents.shape != (condition.shape[0], condition.shape[1], self.latent_dim):
            raise ValueError(
                "target latent shape must match encoded condition: "
                f"expected {(condition.shape[0], condition.shape[1], self.latent_dim)}, "
                f"got {tuple(target_latents.shape)}"
            )
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.timesteps,
                (target_latents.shape[0],),
                device=target_latents.device,
                dtype=torch.long,
            )
        if noise is None:
            noise = torch.randn_like(target_latents)
        x_t = self.q_sample(target_latents, timesteps, noise)
        x0_pred, eps_pred, model_output = self._predict(x_t, timesteps, condition)
        return {
            "condition": condition,
            "timesteps": timesteps,
            "noise": noise,
            "x_t": x_t,
            "x0_target": target_latents,
            "x0_pred": x0_pred,
            "eps_pred": eps_pred,
            "model_output": model_output,
        }

    @torch.no_grad()
    def sample(
        self,
        observation_sequence: torch.Tensor,
        *,
        steps: int = 50,
        eta: float = 0.0,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        condition = self.encode_condition(observation_sequence)
        steps = int(max(1, min(steps, self.timesteps)))
        schedule = torch.linspace(self.timesteps - 1, 0, steps, device=condition.device)
        schedule = torch.unique_consecutive(schedule.round().long())
        if schedule[-1].item() != 0:
            schedule = torch.cat((schedule, schedule.new_zeros(1)))

        shape = (condition.shape[0], condition.shape[1], self.latent_dim)
        x_t = torch.randn(shape, device=condition.device, dtype=condition.dtype) if initial_noise is None else initial_noise
        x_t = x_t.to(device=condition.device, dtype=condition.dtype)

        for index, current in enumerate(schedule.tolist()):
            next_step = schedule[index + 1].item() if index + 1 < schedule.numel() else -1
            t_batch = torch.full((shape[0],), current, device=condition.device, dtype=torch.long)
            x0_pred, eps_pred, _ = self._predict(x_t, t_batch, condition)
            if next_step < 0:
                x_t = x0_pred
                break

            alpha_t = self.alphas_cumprod[current].to(device=x_t.device, dtype=x_t.dtype)
            alpha_next = self.alphas_cumprod[next_step].to(device=x_t.device, dtype=x_t.dtype)
            sigma = float(max(eta, 0.0)) * torch.sqrt(
                ((1.0 - alpha_next) / (1.0 - alpha_t).clamp_min(1e-8))
                * (1.0 - alpha_t / alpha_next).clamp_min(0.0)
            )
            eps_coeff = torch.sqrt((1.0 - alpha_next - sigma.square()).clamp_min(0.0))
            x_t = alpha_next.sqrt() * x0_pred + eps_coeff * eps_pred
            if float(sigma.item()) > 0.0:
                x_t = x_t + sigma * torch.randn_like(x_t)

        return {"latents": x_t, "condition": condition}


class FrozenLBPriorPredictor(nn.Module):
    """Frozen observation-only VQ/diffusion inference used to condition OT.

    Sampling is deterministic by default: DDIM uses ``eta=0`` and starts from
    an all-zero latent tensor, matching the validation metric used to select
    ``diffusion_best_sample.pt``.
    """

    def __init__(
        self,
        cfg,
        vq_checkpoint: Optional[str] = None,
        diffusion_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.vq_model = LBVQAutoEncoder(cfg)
        self.diffusion_model = LBLatentDiffusion(cfg)
        self.sample_steps = int(
            _cfg_value(cfg, "ot_sample_steps", _cfg_value(cfg, "diffusion_sample_steps", 50))
        )
        self.vq_checkpoint = vq_checkpoint or _cfg_value(cfg, "vq_checkpoint", None)
        self.diffusion_checkpoint = diffusion_checkpoint or _cfg_value(
            cfg, "diffusion_checkpoint", None
        )
        self._load_frozen_checkpoint(self.vq_model, self.vq_checkpoint, "l_b VQ")
        self._load_frozen_checkpoint(
            self.diffusion_model, self.diffusion_checkpoint, "l_b diffusion"
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @staticmethod
    def _load_frozen_checkpoint(model: nn.Module, path: Optional[str], name: str) -> None:
        if not path:
            raise ValueError(f"{name} checkpoint is required")
        path = os.path.normpath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get(
                "model_state_dict", checkpoint.get("module_state_dict", checkpoint)
            )
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"{name} checkpoint has no valid state_dict: {path}")
        if state_dict and all(str(key).startswith("module.") for key in state_dict):
            state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

    def train(self, mode: bool = True):
        # Parent pipeline/trainers may enter train mode; this frozen prior must
        # keep dropout disabled so one observation window has one stable prior.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(
        self,
        hp_out: Dict[str, torch.Tensor],
        vc_out: Dict[str, torch.Tensor],
        obj_imu: torch.Tensor,
    ) -> torch.Tensor:
        self.vq_model.eval()
        self.diffusion_model.eval()
        observation = build_lb_observation_sequence(hp_out, vc_out, obj_imu)
        latent_len = int(
            math.ceil(observation.shape[1] / self.diffusion_model.temporal_downsample)
        )
        initial_noise = torch.zeros(
            observation.shape[0],
            latent_len,
            self.diffusion_model.latent_dim,
            device=observation.device,
            dtype=observation.dtype,
        )
        sampled = self.diffusion_model.sample(
            observation,
            steps=self.sample_steps,
            eta=0.0,
            initial_noise=initial_noise,
        )["latents"]
        quantiles = self.vq_model.decode_latents(
            sampled, project_to_codebook=False
        )["lb_quantiles"]
        return quantiles.to(device=observation.device, dtype=observation.dtype).detach()


__all__ = [
    "DEFAULT_LB_QUANTILES",
    "LBVQAutoEncoder",
    "LBLatentDiffusion",
    "FrozenLBPriorPredictor",
    "VectorQuantizer",
    "build_lb_observation_sequence",
    "build_lb_privileged_sequence",
    "compute_lb_quantile_targets",
    "masked_lb_quantile_loss",
    "masked_lb_quantile_mae",
]
