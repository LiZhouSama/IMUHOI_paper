"""
Dual-temporal hierarchical Mamba-style human pose module.

This is the Stage-1 human-only implementation. It keeps the RNN path's public
input/output keys while replacing the temporal backbone with causal fast/slow
Mamba-style encoders and a torso -> lower -> upper -> root hierarchy.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from configs import (
    FRAME_RATE,
    _REDUCED_POSE_NAMES,
    _SENSOR_NAMES,
    _SENSOR_VEL_NAMES,
)
from utils.human_pose import (
    SMPL_PARENTS,
    compute_root_velocity_from_trans,
    compute_smpl_joints_from_global,
    ensure_body_model_device,
    full_global_to_reduced_root_6d,
    global_to_local_rotmats,
    integrate_root_velocity,
    load_body_model,
    reduced_root_pose_to_full_global,
)
from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix

from .base import (
    CausalMovingAverage,
    CrossScaleFusion,
    InputStem,
    PartDecoder,
    TemporalMambaEncoder,
    TemporalMemoryFusion,
)


def _cfg_value(cfg, name: str, default):
    mamba_cfg = getattr(cfg, "mamba", {})
    if isinstance(mamba_cfg, dict) and name in mamba_cfg:
        return mamba_cfg[name]
    if hasattr(mamba_cfg, name):
        return getattr(mamba_cfg, name)
    legacy_name = f"mamba_{name}"
    if hasattr(cfg, legacy_name):
        return getattr(cfg, legacy_name)
    return default


class HumanPoseModule(nn.Module):
    """Human pose estimator with dual temporal Mamba-style encoder."""

    def __init__(self, cfg, device, no_trans: bool = False):
        super().__init__()
        self.device = device
        self.no_trans = bool(no_trans)

        self.num_human_imus = int(getattr(cfg, "num_human_imus", len(_SENSOR_NAMES)))
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.num_joints = int(getattr(cfg, "num_joints", 24))
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.sensor_names = list(_SENSOR_NAMES)
        self.vel_names = list(_SENSOR_VEL_NAMES)
        self.pose_names = list(_REDUCED_POSE_NAMES)
        self.hand_joint_indices = (20, 21)
        self.foot_joint_indices = (7, 8)
        self.smpl_parents = torch.tensor(SMPL_PARENTS, dtype=torch.long)
        self.contact_fusion_enabled = bool(_cfg_value(cfg, "contact_fusion_enabled", True))
        self.contact_fusion_min_prob = float(_cfg_value(cfg, "contact_fusion_min_prob", 0.5))
        self.contact_fusion_full_prob = float(_cfg_value(cfg, "contact_fusion_full_prob", 0.9))
        self.contact_fusion_max_speed = float(_cfg_value(cfg, "contact_fusion_max_speed", 5.0))
        self.prevent_floor_penetration = bool(_cfg_value(cfg, "prevent_floor_penetration", True))
        self.floor_height = float(_cfg_value(cfg, "floor_height", 0.0))

        hidden_dim = int(_cfg_value(cfg, "hidden_dim", getattr(cfg, "human_pose_hidden", 256)))
        part_hidden_dim = int(_cfg_value(cfg, "part_hidden_dim", hidden_dim))
        dropout = float(_cfg_value(cfg, "dropout", getattr(cfg, "human_pose_dropout", 0.1)))
        fast_layers = int(_cfg_value(cfg, "fast_layers", 3))
        slow_layers = int(_cfg_value(cfg, "slow_layers", 3))
        fast_kernel = int(_cfg_value(cfg, "fast_kernel", 3))
        slow_kernel = int(_cfg_value(cfg, "slow_kernel", 9))
        slow_dilation = int(_cfg_value(cfg, "slow_dilation", 2))
        slow_ma_window = int(_cfg_value(cfg, "slow_ma_window", 5))
        memory_heads = int(_cfg_value(cfg, "memory_heads", 4))
        self.memory_size = int(_cfg_value(cfg, "memory_size", 8))
        self.use_delta_features = bool(_cfg_value(cfg, "use_delta_features", False))

        input_dim = self.num_human_imus * self.imu_dim
        if self.use_delta_features:
            input_dim *= 2

        self.stem = InputStem(input_dim, hidden_dim, dropout=dropout, conv_kernel=3)
        self.fast_encoder = TemporalMambaEncoder(
            hidden_dim,
            fast_layers,
            conv_kernel=fast_kernel,
            dropout=dropout,
            dilation=1,
        )
        self.slow_filter = CausalMovingAverage(slow_ma_window)
        self.slow_encoder = TemporalMambaEncoder(
            hidden_dim,
            slow_layers,
            conv_kernel=slow_kernel,
            dropout=dropout,
            dilation=slow_dilation,
        )
        self.cross_scale_fusion = CrossScaleFusion(hidden_dim, dropout=dropout, use_fusion_block=True)
        self.memory_fusion = TemporalMemoryFusion(hidden_dim, num_heads=memory_heads, dropout=dropout)

        # HiPoser-style hierarchy. Prediction order is torso -> lower -> upper,
        # but p_pred/v_pred are assembled in the legacy RNN output order.
        torso_pose_dim = 4 * 6
        lower_pose_dim = 2 * 6
        upper_pose_dim = 4 * 6
        torso_vel_dim = 2 * 3
        lower_vel_dim = 4 * 3
        upper_vel_dim = 2 * 3

        self.torso_decoder = PartDecoder(hidden_dim, part_hidden_dim, torso_pose_dim, torso_vel_dim, dropout)
        self.lower_decoder = PartDecoder(hidden_dim + part_hidden_dim, part_hidden_dim, lower_pose_dim, lower_vel_dim, dropout)
        self.upper_decoder = PartDecoder(
            hidden_dim + part_hidden_dim * 2,
            part_hidden_dim,
            upper_pose_dim,
            upper_vel_dim,
            dropout,
        )
        self.root_decoder = nn.Sequential(
            nn.Linear(hidden_dim + part_hidden_dim * 3, part_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(part_hidden_dim, part_hidden_dim),
            nn.SiLU(),
        )
        self.root_vel_head = nn.Linear(part_hidden_dim, 3)
        self.contact_head = nn.Linear(part_hidden_dim, 2)

        body_model_path = getattr(cfg, "body_model_path", None)
        if body_model_path is None:
            raise ValueError("body_model_path is not set")
        try:
            self.body_model = load_body_model(body_model_path, num_betas=16)
            self.body_model_device = torch.device("cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load BodyModel: {exc}") from exc

    def _build_imu_features(self, human_imu: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = human_imu.shape[:2]
        flat = human_imu.reshape(batch_size, seq_len, -1)
        if not self.use_delta_features:
            return flat

        delta = torch.zeros_like(flat)
        if seq_len > 1:
            delta[:, 1:] = flat[:, 1:] - flat[:, :-1]
        return torch.cat((flat, delta), dim=-1)

    @staticmethod
    def _resolve_trans(
        value,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            return torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        trans = value.to(device=device, dtype=dtype)
        if trans.dim() == 2:
            trans = trans.unsqueeze(1).expand(batch_size, seq_len, 3)
        if trans.shape[0] == 1 and batch_size > 1:
            trans = trans.expand(batch_size, -1, -1)
        if trans.shape[0] != batch_size or trans.shape[1] != seq_len or trans.shape[-1] != 3:
            return torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        return trans

    @staticmethod
    def _resolve_trans_init(
        value,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            return torch.zeros(batch_size, 3, device=device, dtype=dtype)
        trans_init = value.to(device=device, dtype=dtype)
        if trans_init.dim() == 3 and trans_init.shape[1] == 1:
            trans_init = trans_init[:, 0]
        if trans_init.dim() == 1:
            trans_init = trans_init.unsqueeze(0)
        if trans_init.shape[0] == 1 and batch_size > 1:
            trans_init = trans_init.expand(batch_size, -1)
        if trans_init.shape[0] != batch_size or trans_init.shape[-1] != 3:
            return torch.zeros(batch_size, 3, device=device, dtype=dtype)
        return trans_init

    @staticmethod
    def _assemble_reduced_pose(
        torso_pose: torch.Tensor,
        lower_pose: torch.Tensor,
        upper_pose: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = torso_pose.shape[:2]
        pose = torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES), 6, device=torso_pose.device, dtype=torso_pose.dtype)
        pose[:, :, 0:2] = lower_pose.reshape(batch_size, seq_len, 2, 6)
        pose[:, :, 2:6] = torso_pose.reshape(batch_size, seq_len, 4, 6)
        pose[:, :, 6:10] = upper_pose.reshape(batch_size, seq_len, 4, 6)
        return pose

    def _compute_full_pose_and_joints(self, p_pred: torch.Tensor, orientation_mat: torch.Tensor):
        batch_size, seq_len = p_pred.shape[:2]
        bt = batch_size * seq_len
        reduced = p_pred.reshape(bt, len(_REDUCED_POSE_NAMES), 6)
        sensor_count = min(self.num_human_imus, orientation_mat.shape[2])
        orientation = orientation_mat[:, :, :sensor_count].reshape(
            bt,
            sensor_count,
            3,
            3,
        )
        full_flat = reduced_root_pose_to_full_global(
            reduced,
            orientation,
            num_joints=self.num_joints,
        )
        full = full_flat.reshape(batch_size, seq_len, self.num_joints, 3, 3)
        joints = compute_smpl_joints_from_global(full, self.body_model, num_joints=self.num_joints)
        full_6d = matrix_to_rotation_6d(full.reshape(-1, 3, 3)).reshape(batch_size, seq_len, self.num_joints, 6)
        return full, full_6d, joints

    def _global2local(self, global_rotmats: torch.Tensor, parents=None) -> torch.Tensor:
        if global_rotmats.dim() == 4:
            seq_len, num_joints = global_rotmats.shape[:2]
            parents = self.smpl_parents.tolist() if parents is None else parents
            local = global_to_local_rotmats(
                global_rotmats.reshape(seq_len, num_joints, 3, 3),
                parents=parents,
            )
            return local.reshape(seq_len, num_joints, 3, 3)
        if global_rotmats.dim() == 5:
            batch_size, seq_len, num_joints = global_rotmats.shape[:3]
            parents = self.smpl_parents.tolist() if parents is None else parents
            local = global_to_local_rotmats(
                global_rotmats.reshape(batch_size * seq_len, num_joints, 3, 3),
                parents=parents,
            )
            return local.reshape(batch_size, seq_len, num_joints, 3, 3)
        raise ValueError(f"global_rotmats must be [T,J,3,3] or [B,T,J,3,3], got {global_rotmats.shape}")

    def _contact_fusion_weight(self, contact_logits: torch.Tensor) -> torch.Tensor:
        contact_prob = torch.sigmoid(contact_logits)
        contact_conf = contact_prob.max(dim=-1).values
        denom = max(self.contact_fusion_full_prob - self.contact_fusion_min_prob, 1e-6)
        return ((contact_conf - self.contact_fusion_min_prob) / denom).clamp(0.0, 1.0).unsqueeze(-1)

    def _foot_lock_root_velocity(
        self,
        joints_local: torch.Tensor,
        contact_logits: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = contact_logits.shape[:2]
        device = contact_logits.device
        dtype = contact_logits.dtype
        foot_end = max(self.foot_joint_indices)
        if joints_local is None or joints_local.shape[2] <= foot_end:
            return torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)

        left_idx, right_idx = self.foot_joint_indices
        left_pos = joints_local[:, :, left_idx, :]
        right_pos = joints_local[:, :, right_idx, :]
        left_vel = torch.zeros_like(left_pos)
        right_vel = torch.zeros_like(right_pos)
        if seq_len > 1:
            left_vel[:, 1:] = (left_pos[:, :-1] - left_pos[:, 1:]) * self.fps
            left_vel[:, 0] = left_vel[:, 1]
            right_vel[:, 1:] = (right_pos[:, :-1] - right_pos[:, 1:]) * self.fps
            right_vel[:, 0] = right_vel[:, 1]

        contact_prob = torch.sigmoid(contact_logits)
        left_prob = contact_prob[..., 0:1]
        right_prob = contact_prob[..., 1:2]
        denom = left_prob + right_prob + 1e-6
        foot_lock_vel = (left_prob * left_vel + right_prob * right_vel) / denom

        if self.contact_fusion_max_speed > 0:
            speed = torch.linalg.norm(foot_lock_vel, dim=-1, keepdim=True)
            scale = (self.contact_fusion_max_speed / (speed + 1e-6)).clamp(max=1.0)
            foot_lock_vel = foot_lock_vel * scale
        return foot_lock_vel

    def _fuse_root_velocity_with_contact(
        self,
        root_vel_local: torch.Tensor,
        contact_logits: torch.Tensor,
        joints_local: torch.Tensor,
        root_rot: torch.Tensor,
    ):
        inertial_vel = torch.matmul(root_rot, root_vel_local.unsqueeze(-1)).squeeze(-1)
        foot_end = max(self.foot_joint_indices)
        if (
            (not self.contact_fusion_enabled)
            or joints_local is None
            or joints_local.shape[2] <= foot_end
        ):
            weight = torch.zeros_like(inertial_vel[..., :1])
            return inertial_vel, inertial_vel, weight

        foot_lock_vel = self._foot_lock_root_velocity(joints_local, contact_logits)
        weight = self._contact_fusion_weight(contact_logits)
        fused_vel = (1.0 - weight) * inertial_vel + weight * foot_lock_vel
        return fused_vel, foot_lock_vel, weight

    def _apply_floor_penetration(
        self,
        root_velocity: torch.Tensor,
        joints_local: torch.Tensor,
        trans_init: torch.Tensor,
    ) -> torch.Tensor:
        if (not self.prevent_floor_penetration) or joints_local is None:
            return root_velocity
        foot_end = max(self.foot_joint_indices)
        if joints_local.shape[2] <= foot_end:
            return root_velocity

        axis = 1
        root_trans = integrate_root_velocity(root_velocity, self.fps, trans_init=trans_init)
        foot_y = joints_local[:, :, list(self.foot_joint_indices), axis] + root_trans[:, :, None, axis]
        foot_min_y = foot_y.amin(dim=2)
        floor = torch.as_tensor(self.floor_height, device=root_velocity.device, dtype=root_velocity.dtype)
        needed_offset = torch.relu(floor - foot_min_y)
        offset = torch.cummax(needed_offset, dim=1).values
        delta_offset = torch.zeros_like(offset)
        if offset.shape[1] > 1:
            delta_offset[:, 1:] = offset[:, 1:] - offset[:, :-1]

        corrected = root_velocity.clone()
        corrected[..., axis] = corrected[..., axis] + delta_offset * self.fps
        return corrected

    def _make_memory(self, ctx: torch.Tensor, torso: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, root: torch.Tensor):
        if self.memory_size <= 0:
            return {}
        size = min(self.memory_size, ctx.shape[1])
        return {
            "ctx": ctx[:, -size:].detach(),
            "torso": torso[:, -size:].detach(),
            "lower": lower[:, -size:].detach(),
            "upper": upper[:, -size:].detach(),
            "root": root[:, -size:].detach(),
        }

    def forward(self, data_dict: Dict):
        human_imu = data_dict["human_imu"]
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,num_imu,imu_dim], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype
        ensure_body_model_device(self, device)

        imu_features = self._build_imu_features(human_imu)
        f0 = self.stem(imu_features)
        fast = self.fast_encoder(f0)
        slow = self.slow_encoder(self.slow_filter(f0))
        enc = self.cross_scale_fusion(fast, slow)
        memory = data_dict.get("pose_memory", data_dict.get("memory", None))
        ctx = self.memory_fusion(enc, memory=memory)

        torso_h, torso_pose, torso_vel = self.torso_decoder(ctx)
        lower_input = torch.cat((ctx, torso_h), dim=-1)
        lower_h, lower_pose, lower_vel = self.lower_decoder(lower_input)
        upper_input = torch.cat((ctx, torso_h, lower_h), dim=-1)
        upper_h, upper_pose, upper_vel = self.upper_decoder(upper_input)

        root_input = torch.cat((ctx, torso_h, lower_h, upper_h), dim=-1)
        root_h = self.root_decoder(root_input)
        root_vel_local_raw = self.root_vel_head(root_h)
        contact_pred = self.contact_head(root_h)

        p_pred_part = self._assemble_reduced_pose(torso_pose, lower_pose, upper_pose)
        p_pred = p_pred_part.reshape(batch_size, seq_len, -1)
        v_pred = torch.cat((lower_vel, torso_vel, upper_vel), dim=-1)

        orientation_6d = human_imu[..., -6:]
        orientation_mat = rotation_6d_to_matrix(orientation_6d.reshape(-1, 6)).reshape(
            batch_size,
            seq_len,
            human_imu.shape[2],
            3,
            3,
        )
        root_rot = orientation_mat[:, :, 0]

        full_rotmats, full_rot6d, joints_local = self._compute_full_pose_and_joints(p_pred_part, orientation_mat)
        joints_for_contact_fusion = joints_local
        if joints_local is None:
            joints_local = torch.zeros(batch_size, seq_len, self.num_joints, 3, device=device, dtype=dtype)

        if self.no_trans:
            root_trans_pred = self._resolve_trans(
                data_dict.get("trans_gt"),
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                dtype=dtype,
            )
            root_vel_pred = compute_root_velocity_from_trans(root_trans_pred, self.fps)
            root_vel_local_pred = torch.matmul(root_rot.transpose(-1, -2), root_vel_pred.unsqueeze(-1)).squeeze(-1)
            root_vel_inertial_pred = root_vel_pred
            root_vel_contact_pred = torch.zeros_like(root_vel_pred)
            contact_fusion_weight = torch.zeros(batch_size, seq_len, 1, device=device, dtype=dtype)
        else:
            root_vel_local_pred = root_vel_local_raw
            root_vel_pred, root_vel_contact_pred, contact_fusion_weight = self._fuse_root_velocity_with_contact(
                root_vel_local_pred,
                contact_pred,
                joints_for_contact_fusion,
                root_rot,
            )
            root_vel_inertial_pred = torch.matmul(root_rot, root_vel_local_pred.unsqueeze(-1)).squeeze(-1)
            trans_init = self._resolve_trans_init(
                data_dict.get("trans_init"),
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            root_vel_pred = self._apply_floor_penetration(root_vel_pred, joints_for_contact_fusion, trans_init)
            root_trans_pred = integrate_root_velocity(root_vel_pred, self.fps, trans_init=trans_init)

        pred_joints_global = joints_local + root_trans_pred.unsqueeze(2)
        pred_hand_glb_pos = torch.stack(
            (
                pred_joints_global[:, :, self.hand_joint_indices[0]],
                pred_joints_global[:, :, self.hand_joint_indices[1]],
            ),
            dim=2,
        )

        reduced_from_full = full_global_to_reduced_root_6d(full_rotmats)

        return {
            "v_pred": v_pred,
            "p_pred": p_pred,
            "p_pred_part": p_pred_part,
            "p_pred_from_full": reduced_from_full,
            "contact_pred": contact_pred,
            "b_pred": contact_pred,
            "b_prob_pred": torch.sigmoid(contact_pred),
            "root_vel_local_pred": root_vel_local_pred,
            "root_vel_pred": root_vel_pred,
            "root_vel_inertial_pred": root_vel_inertial_pred,
            "root_vel_contact_pred": root_vel_contact_pred,
            "contact_fusion_weight": contact_fusion_weight,
            "root_trans_pred": root_trans_pred,
            "pred_hand_glb_pos": pred_hand_glb_pos,
            "pred_joints_local": joints_local,
            "pred_joints_global": pred_joints_global,
            "pred_full_pose_rotmat": full_rotmats,
            "pred_full_pose_6d": full_rot6d,
            "R_pred_rotmat": full_rotmats,
            "R_pred_6d": full_rot6d,
            "memory": self._make_memory(ctx, torso_h, lower_h, upper_h, root_h),
        }

    @torch.no_grad()
    def inference(self, data_dict: Dict, **_):
        return self.forward(data_dict)

    @staticmethod
    def empty_output(
        batch_size: int,
        seq_len: int,
        device: torch.device,
        no_trans: bool = False,
        num_joints: int = 24,
    ):
        return {
            "v_pred": torch.zeros(batch_size, seq_len, 8 * 3, device=device),
            "p_pred": torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6, device=device),
            "p_pred_part": torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES), 6, device=device),
            "contact_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "b_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "b_prob_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "root_vel_local_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "root_vel_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "root_vel_inertial_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "root_vel_contact_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "contact_fusion_weight": torch.zeros(batch_size, seq_len, 1, device=device),
            "root_trans_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_hand_glb_pos": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "pred_joints_local": torch.zeros(batch_size, seq_len, num_joints, 3, device=device),
            "pred_joints_global": torch.zeros(batch_size, seq_len, num_joints, 3, device=device),
            "pred_full_pose_rotmat": torch.zeros(batch_size, seq_len, num_joints, 3, 3, device=device),
            "pred_full_pose_6d": torch.zeros(batch_size, seq_len, num_joints, 6, device=device),
            "R_pred_rotmat": torch.zeros(batch_size, seq_len, num_joints, 3, 3, device=device),
            "R_pred_6d": torch.zeros(batch_size, seq_len, num_joints, 6, device=device),
            "memory": {},
        }
