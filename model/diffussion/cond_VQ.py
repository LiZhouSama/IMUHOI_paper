"""
VQ tokenizer for interaction conditioning.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d


class _SequenceEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, nhead: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(num_layers), 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.in_proj(x))


class _MultiCodebookQuantizer(nn.Module):
    def __init__(self, code_dim: int, num_codebooks: int, codebook_size: int):
        super().__init__()
        if code_dim % num_codebooks != 0:
            raise ValueError(f"code_dim={code_dim} must be divisible by num_codebooks={num_codebooks}")
        self.code_dim = int(code_dim)
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        self.sub_dim = self.code_dim // self.num_codebooks
        self.codebooks = nn.ModuleList(
            [nn.Embedding(self.codebook_size, self.sub_dim) for _ in range(self.num_codebooks)]
        )
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / max(self.codebook_size, 1), 1.0 / max(self.codebook_size, 1))

    def lookup_codes(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() != 3 or indices.shape[-1] != self.num_codebooks:
            raise ValueError(
                f"indices must be [B,L,{self.num_codebooks}], got {tuple(indices.shape)}"
            )
        parts = []
        for codebook_idx, emb in enumerate(self.codebooks):
            parts.append(emb(indices[..., codebook_idx]))
        return torch.cat(parts, dim=-1)

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_e.dim() != 3 or z_e.shape[-1] != self.code_dim:
            raise ValueError(f"z_e must be [B,L,{self.code_dim}], got {tuple(z_e.shape)}")

        batch_size, latent_len, _ = z_e.shape
        z_parts = z_e.view(batch_size, latent_len, self.num_codebooks, self.sub_dim)

        indices_parts = []
        z_q_parts = []
        codebook_loss = z_e.new_tensor(0.0)
        commit_loss = z_e.new_tensor(0.0)
        perplexity_terms = []

        for codebook_idx, emb in enumerate(self.codebooks):
            z_sub = z_parts[:, :, codebook_idx, :]
            flat = z_sub.reshape(-1, self.sub_dim)
            weight = emb.weight
            dist = (
                flat.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * flat @ weight.t()
                + weight.pow(2).sum(dim=1).unsqueeze(0)
            )
            indices = torch.argmin(dist, dim=1).view(batch_size, latent_len)
            z_q = emb(indices)

            indices_parts.append(indices)
            z_q_parts.append(z_q)
            codebook_loss = codebook_loss + F.mse_loss(z_q, z_sub.detach())
            commit_loss = commit_loss + F.mse_loss(z_sub, z_q.detach())

            encodings = F.one_hot(indices.reshape(-1), num_classes=self.codebook_size).float()
            avg_probs = encodings.mean(dim=0)
            perplexity_terms.append(torch.exp(-(avg_probs * torch.log(avg_probs.clamp_min(1e-8))).sum()))

        indices_all = torch.stack(indices_parts, dim=-1)
        z_q_all = torch.cat(z_q_parts, dim=-1)
        z_q_st = z_e + (z_q_all - z_e).detach()

        return {
            "indices": indices_all,
            "z_q": z_q_st,
            "z_q_detached": z_q_all,
            "codebook_loss": codebook_loss / float(self.num_codebooks),
            "commit_loss": commit_loss / float(self.num_codebooks),
            "perplexity": torch.stack(perplexity_terms).mean() if perplexity_terms else z_e.new_tensor(0.0),
        }


class CondVQModule(nn.Module):
    """VQ-VAE tokenizer for object-conditioned interaction representations."""

    def __init__(self, cfg):
        super().__init__()
        self.num_joints = int(getattr(cfg, "num_joints", 24))
        self.code_dim = int(getattr(cfg, "object_code_dim", 128))
        self.hidden_dim = int(getattr(cfg, "vq_hidden_dim", getattr(cfg, "prior_encoder_hidden_dim", 256)))
        self.num_codebooks = int(getattr(cfg, "vq_num_codebooks", 4))
        self.codebook_size = int(getattr(cfg, "num_object_codes", getattr(cfg, "vq_codebook_size", 128)))
        self.downsample_stride = int(max(1, getattr(cfg, "vq_downsample_stride", 2)))
        self.obj_points_count = int(max(1, getattr(cfg, "mesh_downsample_points", 256)))
        self.encoder_layers = int(getattr(cfg, "vq_encoder_layers", 2))
        self.decoder_layers = int(getattr(cfg, "vq_decoder_layers", 2))
        self.nhead = int(getattr(cfg, "vq_nhead", 8))
        self.dropout = float(getattr(cfg, "vq_dropout", 0.1))

        pose_dim = self.num_joints * 6
        trans_dim = 3
        point_dim = 3

        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.frame_fuse = nn.Sequential(
            nn.Linear(pose_dim + trans_dim + self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.encoder = _SequenceEncoder(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.encoder_layers,
            nhead=self.nhead,
            dropout=self.dropout,
        )
        self.latent_proj = nn.Linear(self.hidden_dim, self.code_dim)

        self.quantizer = _MultiCodebookQuantizer(
            code_dim=self.code_dim,
            num_codebooks=self.num_codebooks,
            codebook_size=self.codebook_size,
        )

        self.decoder_in = nn.Linear(self.code_dim, self.hidden_dim)
        self.decoder = _SequenceEncoder(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.decoder_layers,
            nhead=self.nhead,
            dropout=self.dropout,
        )
        self.pose_head = nn.Linear(self.hidden_dim, pose_dim)
        self.trans_head = nn.Linear(self.hidden_dim, trans_dim)
        self.point_head = nn.Sequential(
            nn.Linear(self.code_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.obj_points_count * 3),
        )

        self._decode_seq_len: Optional[int] = None
        self._decode_num_points: Optional[int] = None

    def _infer_runtime_context(
        self,
        data_dict: Optional[Dict] = None,
        gt_targets: Optional[Dict] = None,
    ) -> Tuple[torch.device, torch.dtype]:
        def _pick_dtype(value: torch.Tensor) -> torch.dtype:
            if torch.is_floating_point(value):
                return value.dtype
            param = next(self.parameters(), None)
            return param.dtype if param is not None else torch.float32

        if isinstance(data_dict, dict):
            for key in ("human_imu", "obj_imu", "masked_x", "obj_points_canonical"):
                value = data_dict.get(key)
                if isinstance(value, torch.Tensor):
                    return value.device, _pick_dtype(value)

        if isinstance(gt_targets, dict):
            for key in ("trans", "rotation_global", "pose"):
                value = gt_targets.get(key)
                if isinstance(value, torch.Tensor):
                    return value.device, _pick_dtype(value)

        param = next(self.parameters(), None)
        if param is not None:
            return param.device, param.dtype
        return torch.device("cpu"), torch.float32

    def latent_length(self, seq_len: int) -> int:
        seq_len = int(seq_len)
        return max((seq_len + self.downsample_stride - 1) // self.downsample_stride, 1)

    def _prepare_obj_points(self, obj_points: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not isinstance(obj_points, torch.Tensor):
            return torch.zeros(batch_size, self.obj_points_count, 3, device=device, dtype=dtype)
        points = obj_points.to(device=device, dtype=dtype)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if points.shape[0] == 1 and batch_size > 1:
            points = points.expand(batch_size, -1, -1)
        if points.shape[0] != batch_size or points.shape[-1] != 3:
            return torch.zeros(batch_size, self.obj_points_count, 3, device=device, dtype=dtype)

        num_points = points.shape[1]
        if num_points > self.obj_points_count:
            idx = torch.linspace(0, max(num_points - 1, 0), steps=self.obj_points_count, device=device).long()
            points = points[:, idx]
        elif num_points < self.obj_points_count and num_points > 0:
            pad = points[:, -1:, :].expand(batch_size, self.obj_points_count - num_points, 3)
            points = torch.cat([points, pad], dim=1)
        elif num_points == 0:
            points = torch.zeros(batch_size, self.obj_points_count, 3, device=device, dtype=dtype)
        return points

    def _resolve_pose6d_from_gt(
        self,
        gt_targets: Dict,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rot_global = gt_targets.get("rotation_global")
        if isinstance(rot_global, torch.Tensor):
            rot_global = rot_global.to(device=device, dtype=dtype)
            if rot_global.dim() == 4:
                rot_global = rot_global.unsqueeze(0)
            if rot_global.shape[0] == 1 and batch_size > 1:
                rot_global = rot_global.expand(batch_size, -1, -1, -1, -1)
            if rot_global.shape[0] == batch_size and rot_global.shape[1] == seq_len and rot_global.shape[-2:] == (3, 3):
                joints_now = int(rot_global.shape[2])
                if joints_now > self.num_joints:
                    rot_global = rot_global[:, :, : self.num_joints]
                elif joints_now < self.num_joints:
                    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
                    pad = eye.expand(batch_size, seq_len, self.num_joints - joints_now, -1, -1)
                    rot_global = torch.cat([rot_global, pad], dim=2)
                return matrix_to_rotation_6d(rot_global.reshape(-1, 3, 3)).reshape(batch_size, seq_len, self.num_joints, 6)

        pose_aa = gt_targets.get("pose")
        if isinstance(pose_aa, torch.Tensor):
            pose_aa = pose_aa.to(device=device, dtype=dtype)
            if pose_aa.dim() == 2:
                pose_aa = pose_aa.unsqueeze(0)
            if pose_aa.shape[0] == 1 and batch_size > 1:
                pose_aa = pose_aa.expand(batch_size, -1, -1)
            if pose_aa.shape[0] == batch_size and pose_aa.shape[1] == seq_len and pose_aa.shape[-1] >= 3:
                joints_now = min(int(pose_aa.shape[-1] // 3), self.num_joints)
                pose_aa = pose_aa[..., : joints_now * 3].reshape(batch_size, seq_len, joints_now, 3)
                rot = axis_angle_to_matrix(pose_aa.reshape(-1, 3)).reshape(batch_size, seq_len, joints_now, 3, 3)
                pose6d = matrix_to_rotation_6d(rot.reshape(-1, 3, 3)).reshape(batch_size, seq_len, joints_now, 6)
                if joints_now < self.num_joints:
                    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
                    pad = matrix_to_rotation_6d(eye.expand(batch_size, seq_len, self.num_joints - joints_now, -1, -1).reshape(-1, 3, 3))
                    pad = pad.reshape(batch_size, seq_len, self.num_joints - joints_now, 6)
                    pose6d = torch.cat([pose6d, pad], dim=2)
                return pose6d

        raise ValueError("CondVQModule requires gt_targets['rotation_global'] or gt_targets['pose']")

    def _resolve_trans(
        self,
        gt_targets: Dict,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        trans = gt_targets.get("trans")
        if not isinstance(trans, torch.Tensor):
            raise ValueError("CondVQModule requires gt_targets['trans']")
        trans = trans.to(device=device, dtype=dtype)
        if trans.dim() == 2:
            trans = trans.unsqueeze(0)
        if trans.shape[0] == 1 and batch_size > 1:
            trans = trans.expand(batch_size, -1, -1)
        if trans.shape[0] != batch_size or trans.shape[1] != seq_len or trans.shape[-1] != 3:
            raise ValueError(f"Invalid trans shape for CondVQModule: {tuple(trans.shape)}")
        return trans

    def _chunk_average(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = x.shape
        latent_len = self.latent_length(seq_len)
        total_len = latent_len * self.downsample_stride
        if total_len > seq_len:
            pad = x[:, -1:, :].expand(batch_size, total_len - seq_len, feat_dim)
            x = torch.cat([x, pad], dim=1)
        x = x.view(batch_size, latent_len, self.downsample_stride, feat_dim)
        return x.mean(dim=2)

    def _repeat_upsample(self, z_q: torch.Tensor, seq_len: int) -> torch.Tensor:
        up = z_q.repeat_interleave(self.downsample_stride, dim=1)
        if up.shape[1] < seq_len:
            pad = up[:, -1:, :].expand(up.shape[0], seq_len - up.shape[1], up.shape[2])
            up = torch.cat([up, pad], dim=1)
        return up[:, :seq_len]

    def encode_to_latent(self, obj_points: torch.Tensor, pose6d: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
        if pose6d.dim() != 4 or pose6d.shape[-1] != 6:
            raise ValueError(f"pose6d must be [B,T,J,6], got {tuple(pose6d.shape)}")
        if trans.dim() != 3 or trans.shape[-1] != 3:
            raise ValueError(f"trans must be [B,T,3], got {tuple(trans.shape)}")

        batch_size, seq_len = pose6d.shape[:2]
        device = pose6d.device
        dtype = pose6d.dtype
        obj_points = self._prepare_obj_points(obj_points, batch_size, device, dtype)
        self._decode_seq_len = int(seq_len)
        self._decode_num_points = int(obj_points.shape[1])

        point_feat = self.point_encoder(obj_points).amax(dim=1)
        point_feat = point_feat.unsqueeze(1).expand(batch_size, seq_len, self.hidden_dim)

        pose_flat = pose6d.reshape(batch_size, seq_len, -1)
        frame_feat = self.frame_fuse(torch.cat([pose_flat, trans, point_feat], dim=-1))
        frame_feat = self.encoder(frame_feat)
        latent = self._chunk_average(frame_feat)
        return self.latent_proj(latent)

    def quantize(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.quantizer(z_e)

    def lookup_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.lookup_codes(indices)

    def decode(self, z_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z_q.dim() != 3 or z_q.shape[-1] != self.code_dim:
            raise ValueError(f"z_q must be [B,L,{self.code_dim}], got {tuple(z_q.shape)}")
        if self._decode_seq_len is None:
            raise RuntimeError("decode() requires encode_to_latent() or prepare_targets() to be called first")

        seq_len = int(self._decode_seq_len)
        num_points = int(self._decode_num_points or self.obj_points_count)
        up = self._repeat_upsample(z_q, seq_len)
        dec_h = self.decoder(self.decoder_in(up))
        pose = self.pose_head(dec_h).view(z_q.shape[0], seq_len, self.num_joints, 6)
        trans = self.trans_head(dec_h)
        obj_points = self.point_head(z_q.mean(dim=1)).view(z_q.shape[0], self.obj_points_count, 3)
        if num_points < self.obj_points_count:
            obj_points = obj_points[:, :num_points]
        elif num_points > self.obj_points_count:
            pad = obj_points[:, -1:, :].expand(z_q.shape[0], num_points - self.obj_points_count, 3)
            obj_points = torch.cat([obj_points, pad], dim=1)
        return {
            "recon_pose6d": pose,
            "recon_trans": trans,
            "recon_obj_points": obj_points,
        }

    def prepare_targets(
        self,
        data_dict: Dict,
        gt_targets: Dict,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(gt_targets, dict):
            raise ValueError("CondVQModule forward requires gt_targets")

        ref = gt_targets.get("trans")
        if not isinstance(ref, torch.Tensor):
            raise ValueError("CondVQModule requires gt_targets['trans']")
        if ref.dim() == 2:
            ref = ref.unsqueeze(0)
        batch_size, seq_len = ref.shape[:2]
        device, dtype = self._infer_runtime_context(data_dict, gt_targets)

        pose6d = self._resolve_pose6d_from_gt(gt_targets, batch_size, seq_len, device, dtype)
        trans = self._resolve_trans(gt_targets, batch_size, seq_len, device, dtype)
        obj_points = self._prepare_obj_points(data_dict.get("obj_points_canonical"), batch_size, device, dtype)
        has_object = data_dict.get("has_object")
        if isinstance(has_object, torch.Tensor):
            has_object = has_object.to(device=device, dtype=torch.bool)
            if has_object.dim() == 0:
                has_object = has_object.view(1)
            if has_object.shape[0] == 1 and batch_size > 1:
                has_object = has_object.expand(batch_size)
        else:
            has_object = torch.ones(batch_size, device=device, dtype=torch.bool)

        self._decode_seq_len = int(seq_len)
        self._decode_num_points = int(obj_points.shape[1])
        return {
            "pose6d": pose6d,
            "trans": trans,
            "obj_points": obj_points,
            "has_object": has_object,
        }

    def tokenize_from_resolved(
        self,
        obj_points: torch.Tensor,
        pose6d: torch.Tensor,
        trans: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_e = self.encode_to_latent(obj_points, pose6d, trans)
        quant = self.quantize(z_e)
        return {
            "z_e": z_e,
            **quant,
        }

    def forward(self, data_dict: Dict, gt_targets: Optional[Dict] = None):
        targets = self.prepare_targets(data_dict, gt_targets or {})
        tokenized = self.tokenize_from_resolved(
            targets["obj_points"],
            targets["pose6d"],
            targets["trans"],
        )
        recon = self.decode(tokenized["z_q"])
        return {
            **tokenized,
            **recon,
            "targets": targets,
            "vq_aux": {
                "codebook_loss": tokenized["codebook_loss"],
                "commit_loss": tokenized["commit_loss"],
                "perplexity": tokenized["perplexity"],
            },
        }

    @torch.no_grad()
    def inference(self, data_dict: Dict, gt_targets: Optional[Dict] = None):
        return self.forward(data_dict, gt_targets=gt_targets)
