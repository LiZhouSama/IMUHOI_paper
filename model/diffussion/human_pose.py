"""
Stage-1 HumanPose diffusion module.

Training:
- Full-window denoising only: x -> z_t -> x0_pred over all frames.
- No autoregressive rollout, no teacher forcing, no training-time masks.

Inference:
- Autoregressive sliding-window inpainting.
- History frames are fixed.
- Current frame only keeps observed dimensions fixed.
- Unknown dimensions are iteratively refined with x0 projection.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix

from .base import ConditionalDiT
from configs import (
    FRAME_RATE,
    _IGNORED_INDICES,
    _REDUCED_INDICES,
    _SENSOR_NAMES,
    _SENSOR_ROT_INDICES,
)


class HumanPoseModule(nn.Module):
    """Diffusion Transformer for Stage-1 human pose reconstruction."""

    def __init__(self, cfg, device, no_trans: bool = False):
        super().__init__()
        self.device = device
        self.no_trans = bool(no_trans)

        self.num_joints = int(getattr(cfg, "num_joints", 24))
        self.num_human_imus = int(getattr(cfg, "num_human_imus", len(_SENSOR_NAMES)))
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))

        self.hand_joint_indices = (20, 21)
        self.foot_joint_indices = (7, 8)
        self.smpl_parents = torch.tensor(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            dtype=torch.long,
        )

        # Feature definition per frame:
        # x_frame = [R_all_joints_global, a_imu6, delta_p_xz, p_y, b_foot2]
        self.rot_dim = self.num_joints * 6
        self.acc_dim = self.num_human_imus * 3
        self.delta_p_dim = 0 if self.no_trans else 2
        self.py_dim = 0 if self.no_trans else 1
        self.contact_dim = 2

        start = 0
        self.rot_slice = slice(start, start + self.rot_dim)
        start += self.rot_dim
        self.acc_slice = slice(start, start + self.acc_dim)
        start += self.acc_dim
        self.delta_p_slice = slice(start, start + self.delta_p_dim)
        start += self.delta_p_dim
        self.py_slice = slice(start, start + self.py_dim)
        start += self.py_dim
        self.contact_slice = slice(start, start + self.contact_dim)
        start += self.contact_dim
        self.target_dim = start

        # Observed dims at inference: IMU-joint rotations + IMU accelerations.
        observed_dim_mask = torch.zeros(self.target_dim, dtype=torch.bool)
        for j in _SENSOR_ROT_INDICES:
            rot_j = slice(j * 6, (j + 1) * 6)
            observed_dim_mask[rot_j] = True
        observed_dim_mask[self.acc_slice] = True

        unknown_dim_mask = ~observed_dim_mask
        self.register_buffer("observed_dim_mask", observed_dim_mask, persistent=False)
        self.register_buffer("unknown_dim_mask", unknown_dim_mask, persistent=False)

        train_cfg = getattr(cfg, "train", {})
        test_cfg = getattr(cfg, "test", {})
        train_window = train_cfg.get("window") if isinstance(train_cfg, dict) else getattr(train_cfg, "window", None)
        test_window = test_cfg.get("window") if isinstance(test_cfg, dict) else getattr(test_cfg, "window", None)
        self.window_size = int(test_window if test_window is not None else train_window)
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")

        dit_cfg = getattr(cfg, "dit", {})

        def _dit_param(name: str, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        self.use_diffusion_noise = bool(_dit_param("dit_use_noise", True))
        self.inference_steps = _dit_param("dit_inference_steps", None)
        self.inference_steps = int(self.inference_steps) if self.inference_steps is not None else None
        self.enable_root_correction = bool((not self.no_trans) and _dit_param("dit_enable_root_correction", False))
        self.root_correction_contact_threshold = float(_dit_param("dit_root_correction_contact_threshold", 0.5))
        self.root_correction_contact_threshold = min(max(self.root_correction_contact_threshold, 0.0), 1.0)
        # Test-time option: seed autoregressive inference with GT warmup frames.
        self.test_use_gt_warmup = bool(_dit_param("dit_test_use_gt_warmup", True))

        prediction_type = str(_dit_param("dit_prediction_type", "x0")).lower()
        if prediction_type not in {"x0", "eps"}:
            prediction_type = "x0"

        self.dit = ConditionalDiT(
            target_dim=self.target_dim,
            cond_dim=0,
            d_model=_dit_param("dit_d_model", 256),
            nhead=_dit_param("dit_nhead", 8),
            num_layers=_dit_param("dit_num_layers", 6),
            dim_feedforward=_dit_param("dit_dim_feedforward", 1024),
            dropout=_dit_param("dit_dropout", 0.1),
            max_seq_len=_dit_param("dit_max_seq_len", max(self.window_size, 256)),
            timesteps=_dit_param("dit_timesteps", 1000),
            use_time_embed=_dit_param("dit_use_time_embed", True),
            prediction_type=prediction_type,
        )

        self.body_model = None
        self.body_model_device = None
        body_model_path = getattr(cfg, "body_model_path", None)
        if body_model_path is None:
            raise ValueError("body_model_path is not set")

        try:
            self.body_model = BodyModel(bm_fname=body_model_path, num_betas=16)
            self.body_model.eval()
            for param in self.body_model.parameters():
                param.requires_grad_(False)
            self.body_model_device = torch.device("cpu")
        except Exception as exc:
            raise RuntimeError(f"Failed to load BodyModel: {exc}") from exc

    @staticmethod
    def _to_bt(
        value,
        *,
        batch_size: int,
        seq_len: int,
        trailing_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        default: float = 0.0,
    ) -> torch.Tensor:
        shape = (batch_size, seq_len, *trailing_shape)
        if not isinstance(value, torch.Tensor):
            return torch.full(shape, float(default), device=device, dtype=dtype)

        out = value.to(device=device, dtype=dtype)
        expected_dim = 2 + len(trailing_shape)
        if out.dim() == expected_dim - 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, *out.shape[1:])

        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return torch.full(shape, float(default), device=device, dtype=dtype)

        if len(trailing_shape) > 0 and tuple(out.shape[2:]) != trailing_shape:
            return torch.full(shape, float(default), device=device, dtype=dtype)

        return out

    def _ensure_body_model_device(self, device: torch.device):
        if self.body_model is not None and self.body_model_device != device:
            self.body_model = self.body_model.to(device)
            self.body_model_device = device

    def _imu_global_rotation(self, human_imu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert IMU orientations from root-relative format to global format."""
        batch_size, seq_len = human_imu.shape[:2]

        imu_rot6d = human_imu[..., 3:9]
        imu_rotm = rotation_6d_to_matrix(imu_rot6d.reshape(-1, 6)).reshape(
            batch_size, seq_len, self.num_human_imus, 3, 3
        )

        root_rot = imu_rotm[:, :, :1]
        imu_rotm_global = imu_rotm.clone()
        if self.num_human_imus > 1:
            imu_rotm_global[:, :, 1:] = torch.matmul(root_rot, imu_rotm[:, :, 1:])

        imu_rot6d_global = matrix_to_rotation_6d(imu_rotm_global.reshape(-1, 3, 3)).reshape(
            batch_size, seq_len, self.num_human_imus, 6
        )
        return imu_rotm_global, imu_rot6d_global

    def _global2local(self, global_rotmats: torch.Tensor, parents) -> torch.Tensor:
        batch_size, num_joints = global_rotmats.shape[:2]
        local_rotmats = torch.zeros_like(global_rotmats)
        local_rotmats[:, 0] = global_rotmats[:, 0]

        for i in range(1, num_joints):
            parent_idx = parents[i]
            parent_rot = global_rotmats[:, parent_idx]
            local_rotmats[:, i] = torch.matmul(parent_rot.transpose(-1, -2), global_rotmats[:, i])

        eye = torch.eye(3, device=global_rotmats.device, dtype=global_rotmats.dtype).view(1, 1, 3, 3)
        local_rotmats[:, _IGNORED_INDICES] = eye.expand(batch_size, len(_IGNORED_INDICES), -1, -1)
        return local_rotmats

    def _compute_fk_joints_from_global(self, full_global_rotmats: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Differentiable FK via BodyModel:
        global R -> local pose -> BodyModel(Jtr)
        """
        if self.body_model is None:
            return None

        batch_size, seq_len = full_global_rotmats.shape[:2]
        bt = batch_size * seq_len

        full_global = full_global_rotmats.reshape(bt, self.num_joints, 3, 3)
        local_pose = self._global2local(full_global, self.smpl_parents.tolist())
        pose_aa = matrix_to_axis_angle(local_pose.reshape(-1, 3, 3)).reshape(bt, self.num_joints, 3)

        try:
            body_out = self.body_model(
                pose_body=pose_aa[:, 1:22].reshape(bt, 63),
                root_orient=pose_aa[:, 0].reshape(bt, 3),
            )
            joints = body_out.Jtr
            if joints.size(1) > self.num_joints:
                joints = joints[:, : self.num_joints]
            elif joints.size(1) < self.num_joints:
                pad = torch.zeros(
                    joints.size(0),
                    self.num_joints - joints.size(1),
                    3,
                    device=joints.device,
                    dtype=joints.dtype,
                )
                joints = torch.cat([joints, pad], dim=1)
            return joints.reshape(batch_size, seq_len, self.num_joints, 3)
        except Exception:
            return None

    def _resolve_trans_init(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        trans_init = data_dict.get("trans_init")
        if isinstance(trans_init, torch.Tensor):
            trans_init = trans_init.to(device=device, dtype=dtype)
            if trans_init.dim() == 3 and trans_init.size(1) == 1:
                trans_init = trans_init[:, 0]
            if trans_init.dim() == 1:
                trans_init = trans_init.unsqueeze(0)
            if trans_init.shape[0] == 1 and batch_size > 1:
                trans_init = trans_init.expand(batch_size, -1)
            if trans_init.shape[0] == batch_size and trans_init.shape[-1] == 3:
                return trans_init

        if isinstance(gt_targets, dict) and isinstance(gt_targets.get("trans"), torch.Tensor):
            trans = gt_targets["trans"].to(device=device, dtype=dtype)
            if trans.dim() == 2:
                trans = trans.unsqueeze(0)
            if trans.shape[0] == 1 and batch_size > 1:
                trans = trans.expand(batch_size, -1, -1)
            if trans.shape[0] == batch_size and trans.shape[1] > 0:
                return trans[:, 0]

        return torch.zeros(batch_size, 3, device=device, dtype=dtype)

    def _build_clean_window(self, data_dict: Dict, gt_targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """Build full clean window x from GT for training."""
        if not isinstance(gt_targets, dict):
            raise ValueError("gt_targets must be provided for full-window diffusion training")

        human_imu = data_dict["human_imu"]
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        self._ensure_body_model_device(device)

        imu_rotm_global, imu_rot6d_global = self._imu_global_rotation(human_imu)
        imu_acc_flat = human_imu[..., :3].reshape(batch_size, seq_len, -1)

        rot_global = gt_targets.get("rotation_global")
        if not isinstance(rot_global, torch.Tensor):
            raise ValueError("gt_targets['rotation_global'] is required")

        rot_global = rot_global.to(device=device, dtype=dtype)
        if rot_global.dim() == 4:
            rot_global = rot_global.unsqueeze(0)
        if rot_global.shape[0] == 1 and batch_size > 1:
            rot_global = rot_global.expand(batch_size, -1, -1, -1, -1)

        if rot_global.shape[0] != batch_size or rot_global.shape[1] != seq_len:
            raise ValueError(
                f"rotation_global shape mismatch, got {rot_global.shape}, expected [B={batch_size},T={seq_len},J,3,3]"
            )

        joints_now = rot_global.shape[2]
        if joints_now > self.num_joints:
            rot_global = rot_global[:, :, : self.num_joints]
        elif joints_now < self.num_joints:
            eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
            pad = eye.expand(batch_size, seq_len, self.num_joints - joints_now, -1, -1)
            rot_global = torch.cat([rot_global, pad], dim=2)

        rot_global_6d = matrix_to_rotation_6d(rot_global.reshape(-1, 3, 3)).reshape(batch_size, seq_len, self.num_joints, 6)

        # Replace IMU joints with observed IMU global rotations.
        rot_global_6d[:, :, _SENSOR_ROT_INDICES] = imu_rot6d_global
        rot_global = rotation_6d_to_matrix(rot_global_6d.reshape(-1, 6)).reshape(batch_size, seq_len, self.num_joints, 3, 3)

        trans = self._to_bt(
            gt_targets.get("trans"),
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=(3,),
            device=device,
            dtype=dtype,
            default=0.0,
        )

        if self.no_trans:
            delta_p = torch.zeros(batch_size, seq_len, 0, device=device, dtype=dtype)
            p_y = torch.zeros(batch_size, seq_len, 0, device=device, dtype=dtype)
        else:
            delta_p = torch.zeros(batch_size, seq_len, 2, device=device, dtype=dtype)
            if seq_len > 1:
                delta_p[:, 1:] = trans[:, 1:, [0, 2]] - trans[:, :-1, [0, 2]]
            p_y = trans[:, :, 1:2]

        lfoot_contact = self._to_bt(
            gt_targets.get("lfoot_contact"),
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=(),
            device=device,
            dtype=dtype,
            default=0.0,
        )
        rfoot_contact = self._to_bt(
            gt_targets.get("rfoot_contact"),
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=(),
            device=device,
            dtype=dtype,
            default=0.0,
        )
        b = torch.stack([lfoot_contact, rfoot_contact], dim=-1).clamp(0.0, 1.0)

        x_parts = [
            rot_global_6d.reshape(batch_size, seq_len, -1),
            imu_acc_flat,
        ]
        if not self.no_trans:
            x_parts.extend([delta_p, p_y])
        x_parts.append(b)
        x_clean = torch.cat(x_parts, dim=-1)

        trans_init = self._resolve_trans_init(
            data_dict,
            gt_targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        gt_joints_local = self._compute_fk_joints_from_global(rot_global)

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "trans_init": trans_init,
            "trans_gt": trans,
            "gt_joints_local": gt_joints_local,
            "human_imu": human_imu,
            "imu_rotm_global": imu_rotm_global,
        }
        return x_clean, context

    def _build_observed_sequence(self, data_dict: Dict, gt_targets: Optional[Dict]) -> Tuple[torch.Tensor, Dict]:
        """Build observed-only sequence for autoregressive inference."""
        human_imu = data_dict["human_imu"]
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        self._ensure_body_model_device(device)

        imu_rotm_global, imu_rot6d_global = self._imu_global_rotation(human_imu)
        imu_acc_flat = human_imu[..., :3].reshape(batch_size, seq_len, -1)

        observed = torch.zeros(batch_size, seq_len, self.target_dim, device=device, dtype=dtype)
        for local_idx, joint_idx in enumerate(_SENSOR_ROT_INDICES):
            observed[:, :, joint_idx * 6 : (joint_idx + 1) * 6] = imu_rot6d_global[:, :, local_idx]
        observed[:, :, self.acc_slice] = imu_acc_flat

        trans_init = self._resolve_trans_init(
            data_dict,
            gt_targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "trans_init": trans_init,
            "trans_gt": None,
            "gt_joints_local": None,
            "human_imu": human_imu,
            "imu_rotm_global": imu_rotm_global,
        }
        return observed, context

    def _compute_root_velocity_from_trans(self, trans: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(trans)
        if trans.size(1) > 1:
            vel[:, 1:] = (trans[:, 1:] - trans[:, :-1]) * self.fps
            vel[:, 0] = vel[:, 1]
        return vel

    def _apply_root_correction_step(
        self,
        *,
        prev_frame: torch.Tensor,
        curr_frame: torch.Tensor,
        prev_root_pos: torch.Tensor,
        curr_root_xz: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference-only root correction on current frame delta_p x/z."""
        if self.no_trans or (not self.enable_root_correction) or self.delta_p_dim < 2:
            return curr_frame, curr_root_xz

        batch_size = curr_frame.shape[0]
        device = curr_frame.device
        dtype = curr_frame.dtype

        curr_root_pos = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        curr_root_pos[:, 0] = curr_root_xz[:, 0]
        curr_root_pos[:, 2] = curr_root_xz[:, 1]
        curr_root_pos[:, 1] = curr_frame[:, self.py_slice].squeeze(-1)

        rot_prev = rotation_6d_to_matrix(prev_frame[:, self.rot_slice].reshape(-1, 6)).reshape(
            batch_size, self.num_joints, 3, 3
        )
        rot_curr = rotation_6d_to_matrix(curr_frame[:, self.rot_slice].reshape(-1, 6)).reshape(
            batch_size, self.num_joints, 3, 3
        )
        rot_pair = torch.stack([rot_prev, rot_curr], dim=1)

        joints_local_pair = self._compute_fk_joints_from_global(rot_pair)
        if joints_local_pair is None:
            return curr_frame, curr_root_xz

        foot_prev_local = joints_local_pair[:, 0, self.foot_joint_indices, :]  # [B,2,3]
        foot_curr_local = joints_local_pair[:, 1, self.foot_joint_indices, :]  # [B,2,3]
        foot_prev_world = foot_prev_local + prev_root_pos.unsqueeze(1)
        foot_curr_world = foot_curr_local + curr_root_pos.unsqueeze(1)
        foot_disp_xz = foot_curr_world[..., [0, 2]] - foot_prev_world[..., [0, 2]]  # [B,2,2]

        contact_prob = torch.sigmoid(curr_frame[:, self.contact_slice])  # [B,2]
        contact_mask = contact_prob >= self.root_correction_contact_threshold
        contact_count = contact_mask.sum(dim=1, keepdim=True)  # [B,1]
        valid = contact_count.squeeze(-1) > 0
        if not bool(valid.any()):
            return curr_frame, curr_root_xz

        avg_disp_xz = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        masked_disp = foot_disp_xz * contact_mask.unsqueeze(-1).to(dtype=dtype)
        avg_disp_xz[valid] = masked_disp[valid].sum(dim=1) / contact_count[valid].to(dtype=dtype)

        corrected = curr_frame.clone()
        corrected_delta_xz = corrected[:, self.delta_p_slice] - avg_disp_xz
        corrected[:, self.delta_p_slice] = corrected_delta_xz
        corrected_root_xz = prev_root_pos[:, [0, 2]] + corrected_delta_xz
        return corrected, corrected_root_xz

    def _decode_outputs(self, x_pred: torch.Tensor, context: Dict) -> Dict:
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        trans_init = context["trans_init"]

        rot_6d = x_pred[:, :, self.rot_slice].reshape(batch_size, seq_len, self.num_joints, 6)
        rot_mat = rotation_6d_to_matrix(rot_6d.reshape(-1, 6)).reshape(batch_size, seq_len, self.num_joints, 3, 3)

        acc_pred = x_pred[:, :, self.acc_slice].reshape(batch_size, seq_len, self.num_human_imus, 3)

        if self.no_trans:
            delta_p_pred = torch.zeros(batch_size, seq_len, 0, device=device, dtype=dtype)
            p_y_pred = torch.zeros(batch_size, seq_len, 0, device=device, dtype=dtype)
            root_trans_pred = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        else:
            delta_p_pred = x_pred[:, :, self.delta_p_slice]
            p_y_pred = x_pred[:, :, self.py_slice]

            xz = torch.cumsum(delta_p_pred, dim=1)
            xz = xz + trans_init[:, [0, 2]].unsqueeze(1)
            root_trans_pred = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
            root_trans_pred[:, :, 0] = xz[:, :, 0]
            root_trans_pred[:, :, 2] = xz[:, :, 1]
            root_trans_pred[:, :, 1] = p_y_pred.squeeze(-1)

        b_pred = x_pred[:, :, self.contact_slice]
        b_prob_pred = torch.sigmoid(b_pred)

        root_vel_pred = self._compute_root_velocity_from_trans(root_trans_pred)
        root_rot = rot_mat[:, :, 0]
        root_vel_local_pred = torch.matmul(root_rot.transpose(-1, -2), root_vel_pred.unsqueeze(-1)).squeeze(-1)

        reduced_global = rot_mat[:, :, _REDUCED_INDICES]
        reduced_root = torch.matmul(root_rot.unsqueeze(2).transpose(-1, -2), reduced_global)
        p_pred = matrix_to_rotation_6d(reduced_root.reshape(-1, 3, 3)).reshape(
            batch_size, seq_len, len(_REDUCED_INDICES), 6
        )

        pred_joints_local = self._compute_fk_joints_from_global(rot_mat)
        if pred_joints_local is None:
            pred_joints_local = torch.zeros(batch_size, seq_len, self.num_joints, 3, device=device, dtype=dtype)

        pred_joints_global = pred_joints_local + root_trans_pred.unsqueeze(2)
        pred_hand_glb_pos = torch.stack(
            [
                pred_joints_global[:, :, self.hand_joint_indices[0]],
                pred_joints_global[:, :, self.hand_joint_indices[1]],
            ],
            dim=2,
        )

        pred_imu_feat = torch.cat(
            [
                acc_pred.reshape(batch_size, seq_len, -1),
                rot_6d[:, :, _SENSOR_ROT_INDICES].reshape(batch_size, seq_len, -1),
            ],
            dim=-1,
        )

        out = {
            "x_pred": x_pred,
            "R_pred_6d": rot_6d,
            "R_pred_rotmat": rot_mat,
            "a_pred": acc_pred,
            "delta_p_pred": delta_p_pred,
            "p_y_pred": p_y_pred,
            "b_pred": b_pred,
            "b_prob_pred": b_prob_pred,
            "pred_imu_feat": pred_imu_feat,
            "v_pred": torch.zeros(batch_size, seq_len, 0, device=device, dtype=dtype),
            "p_pred": p_pred,
            "pred_full_pose_rotmat": rot_mat,
            "pred_full_pose_6d": rot_6d,
            "pred_joints_local": pred_joints_local,
            "pred_joints_global": pred_joints_global,
            "pred_hand_glb_pos": pred_hand_glb_pos,
            "root_vel_local_pred": root_vel_local_pred,
            "root_vel_pred": root_vel_pred,
            "root_trans_pred": root_trans_pred,
        }

        if context.get("gt_joints_local") is not None:
            out["gt_joints_local"] = context["gt_joints_local"]
        if context.get("trans_gt") is not None:
            out["gt_root_trans"] = context["trans_gt"]

        return out

    def forward(self, data_dict: Dict, gt_targets: Optional[Dict] = None):
        """Training forward: full-window denoising only."""
        if gt_targets is None:
            raise ValueError("HumanPoseModule forward requires gt_targets for training")

        x_clean, context = self._build_clean_window(data_dict, gt_targets)
        add_noise = bool(self.training and self.use_diffusion_noise)

        x0_pred, aux = self.dit(
            cond=None,
            x_start=x_clean,
            add_noise=add_noise,
        )

        outputs = self._decode_outputs(x0_pred, context)
        outputs["diffusion_aux"] = {
            **aux,
            "x0_target": x_clean,
            "x0_pred": x0_pred,
            "observed_dim_mask": self.observed_dim_mask,
            "unknown_dim_mask": self.unknown_dim_mask,
            "prediction_type": self.dit.prediction_type,
        }
        return outputs

    def _autoregressive_inference(
        self,
        observed_seq: torch.Tensor,
        *,
        trans_init: torch.Tensor,
        steps: int,
        warmup_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive sliding-window inference with inpainting."""
        batch_size, seq_len, feat_dim = observed_seq.shape
        device = observed_seq.device
        dtype = observed_seq.dtype

        window = self.window_size
        if window < 1:
            raise ValueError(f"window_size must be >= 1, got {window}")

        unknown_idx = torch.nonzero(self.unknown_dim_mask.to(device=device), as_tuple=False).flatten()
        unknown_dim = int(unknown_idx.numel())

        history = torch.zeros(batch_size, 0, feat_dim, device=device, dtype=dtype)
        if isinstance(warmup_seq, torch.Tensor):
            if warmup_seq.dim() != 3 or warmup_seq.shape[0] != batch_size or warmup_seq.shape[-1] != feat_dim:
                raise ValueError(
                    f"warmup_seq shape mismatch, got {warmup_seq.shape}, expected [B={batch_size},T_w,{feat_dim}]"
                )
            if warmup_seq.shape[1] >= seq_len:
                raise ValueError(
                    f"warmup_seq is too long for autoregressive rollout: T_w={warmup_seq.shape[1]} vs seq_len={seq_len}"
                )
            history = warmup_seq.to(device=device, dtype=dtype).clone()

        last_unknown = torch.zeros(batch_size, unknown_dim, device=device, dtype=dtype)
        prev_root_pos = trans_init.to(device=device, dtype=dtype)
        if prev_root_pos.dim() == 1:
            prev_root_pos = prev_root_pos.unsqueeze(0).expand(batch_size, -1)

        if history.size(1) > 0:
            if unknown_dim > 0:
                last_unknown = history[:, -1, unknown_idx]

            if not self.no_trans and self.delta_p_dim >= 2:
                warmup_delta_xz = history[:, :, self.delta_p_slice]
                warmup_xz = torch.cumsum(warmup_delta_xz, dim=1) + prev_root_pos[:, [0, 2]].unsqueeze(1)
                prev_root_pos = prev_root_pos.clone()
                prev_root_pos[:, 0] = warmup_xz[:, -1, 0]
                prev_root_pos[:, 2] = warmup_xz[:, -1, 1]
                if self.py_dim > 0:
                    prev_root_pos[:, 1] = history[:, -1, self.py_slice].squeeze(-1)

        for frame_idx in range(int(history.size(1)), seq_len):
            current = observed_seq[:, frame_idx].clone()
            if unknown_dim > 0:
                current[:, unknown_idx] = last_unknown

            if window > 1:
                hist = history[:, -(window - 1) :] if history.size(1) > 0 else history
                hist_len = int(hist.size(1))
                if hist_len < (window - 1):
                    pad_count = (window - 1) - hist_len
                    if hist_len > 0:
                        pad_src = hist[:, :1]
                    else:
                        pad_src = current.unsqueeze(1)
                    pad = pad_src.expand(batch_size, pad_count, feat_dim)
                    hist = torch.cat([pad, hist], dim=1)
                x_input = torch.cat([hist, current.unsqueeze(1)], dim=1)
            else:
                x_input = current.unsqueeze(1)

            inpaint_mask = torch.ones_like(x_input, dtype=torch.bool)
            if unknown_dim > 0:
                inpaint_mask[:, -1, unknown_idx] = False

            x_out = self.dit.sample_inpaint_x0(
                x_input=x_input,
                inpaint_mask=inpaint_mask,
                cond=None,
                steps=steps,
            )

            current_pred = x_out[:, -1]
            if unknown_dim > 0:
                last_unknown = current_pred[:, unknown_idx]
                current[:, unknown_idx] = last_unknown

            if not self.no_trans and self.delta_p_dim >= 2:
                curr_delta_xz = current[:, self.delta_p_slice]
                curr_root_xz = prev_root_pos[:, [0, 2]] + curr_delta_xz
                if self.enable_root_correction and frame_idx > 0 and history.size(1) > 0:
                    corrected_current, corrected_root_xz = self._apply_root_correction_step(
                        prev_frame=history[:, -1],
                        curr_frame=current,
                        prev_root_pos=prev_root_pos,
                        curr_root_xz=curr_root_xz,
                    )
                    current = corrected_current
                    curr_root_xz = corrected_root_xz
                    if unknown_dim > 0:
                        last_unknown = current[:, unknown_idx]

                curr_root_pos = prev_root_pos.clone()
                curr_root_pos[:, 0] = curr_root_xz[:, 0]
                curr_root_pos[:, 2] = curr_root_xz[:, 1]
                if self.py_dim > 0:
                    curr_root_pos[:, 1] = current[:, self.py_slice].squeeze(-1)
                prev_root_pos = curr_root_pos

            history = torch.cat([history, current.unsqueeze(1)], dim=1)

        return history

    @torch.no_grad()
    def inference(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict] = None,
        sample_steps: Optional[int] = None,
        sampler: Optional[str] = None,
        eta: Optional[float] = None,
    ):
        """Inference with autoregressive sliding-window inpainting."""
        del sampler, eta  # Inference uses x0 iterative inpainting only.

        observed_seq, context = self._build_observed_sequence(data_dict, gt_targets)

        steps = self.inference_steps if sample_steps is None else int(sample_steps)
        if steps is None:
            steps = self.dit.timesteps
        steps = int(steps)

        warmup_seq = None
        warmup_len = 0
        seq_len = int(observed_seq.shape[1])
        if (
            (not self.training)
            and self.test_use_gt_warmup
            and isinstance(gt_targets, dict)
            and isinstance(gt_targets.get("rotation_global"), torch.Tensor)
        ):
            gt_clean_seq, _ = self._build_clean_window(data_dict, gt_targets)
            max_warmup = max(seq_len, 1) - 1
            warmup_len = min(max(self.window_size - 1, 0), max_warmup)
            if warmup_len > 0:
                warmup_seq = gt_clean_seq[:, :warmup_len]

        pred_seq = self._autoregressive_inference(
            observed_seq,
            trans_init=context["trans_init"],
            steps=steps,
            warmup_seq=warmup_seq,
        )
        outputs = self._decode_outputs(pred_seq, context)
        outputs["diffusion_aux"] = {
            "inference_steps": steps,
            "window_size": self.window_size,
            "warmup_len": int(warmup_len),
            "warmup_mode": "gt_prefix" if warmup_len > 0 else "none",
            "observed_dim_mask": self.observed_dim_mask,
            "unknown_dim_mask": self.unknown_dim_mask,
            "prediction_type": self.dit.prediction_type,
            "inference_mode": "autoregressive_inpaint_x0",
            "root_correction_enabled": bool(self.enable_root_correction and (not self.no_trans)),
            "root_correction_contact_threshold": float(self.root_correction_contact_threshold),
        }
        return outputs

    @staticmethod
    def empty_output(
        batch_size: int,
        seq_len: int,
        device: torch.device,
        no_trans: bool = False,
        num_joints: int = 24,
    ):
        rot_dim = num_joints * 6
        return {
            "x_pred": torch.zeros(batch_size, seq_len, rot_dim + 18 + (0 if no_trans else 3) + 2, device=device),
            "R_pred_6d": torch.zeros(batch_size, seq_len, num_joints, 6, device=device),
            "R_pred_rotmat": torch.zeros(batch_size, seq_len, num_joints, 3, 3, device=device),
            "a_pred": torch.zeros(batch_size, seq_len, 6, 3, device=device),
            "delta_p_pred": torch.zeros(batch_size, seq_len, 0 if no_trans else 2, device=device),
            "p_y_pred": torch.zeros(batch_size, seq_len, 0 if no_trans else 1, device=device),
            "b_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "b_prob_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_imu_feat": torch.zeros(batch_size, seq_len, 6 * 9, device=device),
            "v_pred": torch.zeros(batch_size, seq_len, 0, device=device),
            "p_pred": torch.zeros(batch_size, seq_len, len(_REDUCED_INDICES), 6, device=device),
            "pred_full_pose_rotmat": torch.zeros(batch_size, seq_len, num_joints, 3, 3, device=device),
            "pred_full_pose_6d": torch.zeros(batch_size, seq_len, num_joints, 6, device=device),
            "pred_joints_local": torch.zeros(batch_size, seq_len, num_joints, 3, device=device),
            "pred_joints_global": torch.zeros(batch_size, seq_len, num_joints, 3, device=device),
            "pred_hand_glb_pos": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "root_vel_local_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "root_vel_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "root_trans_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "diffusion_aux": {},
        }
