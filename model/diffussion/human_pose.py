"""
DiffusionPoser-style HumanPose module.

Key design choices:
- State variable is a sequence window, not per-frame direct regression.
- IMU observations are part of the diffusion state and are enforced via hard inpainting masks.
- Denoiser predicts x0 directly (clean sample), without explicit height conditioning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from .base import ConditionalDiT
from configs import (
    FRAME_RATE,
    _SENSOR_NAMES,
    _SENSOR_VEL_NAMES,
    _REDUCED_POSE_NAMES,
    _REDUCED_INDICES,
    _IGNORED_INDICES,
    _SENSOR_ROT_INDICES,
)


class HumanPoseModule(nn.Module):
    """Diffusion Transformer for Stage-1 pose reconstruction with inpainting guidance."""

    def __init__(self, cfg, device, no_trans: bool = False):
        super().__init__()
        self.device = device
        self.no_trans = no_trans
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

        # Match legacy velocity ordering: [Root, LFoot, RFoot, Head, Root, Head, LHand, RHand].
        self.velocity_order = [0, 1, 2, 3, 0, 3, 4, 5]
        self.v_dim = len(self.velocity_order) * 3
        self.p_dim = len(_REDUCED_POSE_NAMES) * 6
        self.root_dim = 0 if no_trans else 3  # root velocity (local)
        self.motion_dim = self.v_dim + self.p_dim + self.root_dim

        self.imu_feat_dim = self.num_human_imus * self.imu_dim
        self.target_dim = self.imu_feat_dim + self.motion_dim

        dit_cfg = getattr(cfg, "dit", {})

        def _dit_param(name, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        if isinstance(dit_cfg, dict):
            for removed_key in ("dit_train_current_mode", "dit_inference_mode"):
                if removed_key in dit_cfg:
                    raise ValueError(
                        f"'{removed_key}' has been removed. "
                        "Stage-1 now uses fixed K-step rollout training and autoregressive inference only."
                    )
        for removed_key in ("dit_train_current_mode", "dit_inference_mode"):
            if hasattr(cfg, removed_key):
                raise ValueError(
                    f"'{removed_key}' has been removed. "
                    "Please delete it from config and use the fixed Stage-1 rollout pipeline."
                )

        train_cfg = getattr(cfg, "train", {})
        test_cfg = getattr(cfg, "test", {})
        train_window = train_cfg.get("window") if isinstance(train_cfg, dict) else getattr(train_cfg, "window", None)
        test_window = test_cfg.get("window") if isinstance(test_cfg, dict) else getattr(test_cfg, "window", None)
        self.window_size = int(test_window if test_window is not None else train_window)
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {self.window_size}")
        self.warmup_len = self.window_size - 1

        max_seq_len = _dit_param("dit_max_seq_len", self.window_size)
        self.use_diffusion_noise = bool(_dit_param("dit_use_noise", True))
        self.inference_steps = _dit_param("dit_inference_steps", None)
        if self.inference_steps is not None:
            self.inference_steps = int(self.inference_steps)
        self.inference_sampler = str(_dit_param("dit_inference_sampler", "ddim")).lower()
        self.inference_eta = float(_dit_param("dit_inference_eta", 0.0))

        self.enable_root_correction = bool(_dit_param("dit_enable_root_correction", True))
        self.root_contact_vel_threshold = float(_dit_param("dit_root_contact_vel_threshold", 0.15))
        self.train_rollout_k = 30

        prediction_type = str(_dit_param("dit_prediction_type", "x0")).lower()
        if prediction_type not in {"x0", "eps"}:
            prediction_type = "x0"

        self.dit = ConditionalDiT(
            target_dim=self.target_dim,
            cond_dim=0,  # no explicit condition branch; observations are injected via inpainting
            d_model=_dit_param("dit_d_model", 256),
            nhead=_dit_param("dit_nhead", 8),
            num_layers=_dit_param("dit_num_layers", 6),
            dim_feedforward=_dit_param("dit_dim_feedforward", 1024),
            dropout=_dit_param("dit_dropout", 0.1),
            max_seq_len=max_seq_len,
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
            print(f"加载Body Model失败: {exc}")
            exit()

    def _reduced_glb_6d_to_full_glb_mat(self, glb_reduced_pose, orientation):
        root_rotation = orientation[:, 0]
        reduced_rot = rotation_6d_to_matrix(glb_reduced_pose.reshape(-1, 6)).reshape(
            glb_reduced_pose.shape[0], len(_REDUCED_POSE_NAMES), 3, 3
        )
        reduced_rot_global = torch.matmul(root_rotation.unsqueeze(1), reduced_rot)
        orientation_global = orientation.clone()
        orientation_global[:, 1:] = torch.matmul(root_rotation.unsqueeze(1), orientation[:, 1:])
        dtype = glb_reduced_pose.dtype
        device = glb_reduced_pose.device
        full_pose = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(
            glb_reduced_pose.shape[0], self.num_joints, 1, 1
        )
        full_pose[:, _REDUCED_INDICES] = reduced_rot_global
        full_pose[:, _SENSOR_ROT_INDICES] = orientation_global
        ignored_parents = self.smpl_parents[_IGNORED_INDICES]
        full_pose[:, _IGNORED_INDICES] = full_pose[:, ignored_parents]
        return full_pose

    def _global2local(self, global_rotmats, parents):
        batch_size, num_joints, _, _ = global_rotmats.shape
        local_rotmats = torch.zeros_like(global_rotmats)
        local_rotmats[:, 0] = global_rotmats[:, 0]
        for i in range(1, num_joints):
            parent_idx = parents[i]
            R_parent = global_rotmats[:, parent_idx]
            R_parent_inv = R_parent.transpose(-1, -2)
            local_rotmats[:, i] = torch.matmul(R_parent_inv, global_rotmats[:, i])
        local_rotmats[:, _IGNORED_INDICES] = torch.eye(
            3, device=global_rotmats.device, dtype=global_rotmats.dtype
        ).view(1, 1, 3, 3).repeat(batch_size, len(_IGNORED_INDICES), 1, 1)
        return local_rotmats

    def _compute_fk_joints_batched(self, glb_p_out_tensor: torch.Tensor, orientation: torch.Tensor):
        if self.body_model is None:
            return None

        batch_size, seq_len = glb_p_out_tensor.shape[:2]
        device = glb_p_out_tensor.device
        BT = batch_size * seq_len

        glb_pose = glb_p_out_tensor.reshape(BT, len(_REDUCED_POSE_NAMES), 6)
        orientation = orientation[:, :, : len(_SENSOR_ROT_INDICES)].reshape(BT, len(_SENSOR_ROT_INDICES), 3, 3)
        full_glb = self._reduced_glb_6d_to_full_glb_mat(glb_pose, orientation)
        local_pose = self._global2local(full_glb, self.smpl_parents.tolist())
        pose_aa = matrix_to_axis_angle(local_pose.reshape(-1, 3, 3)).reshape(BT, self.num_joints, 3)

        try:
            with torch.no_grad():
                body_out = self.body_model(
                    pose_body=pose_aa[:, 1:22].reshape(BT, 63),
                    root_orient=pose_aa[:, 0].reshape(BT, 3),
                )
            joints = body_out.Jtr
            if joints.size(1) != self.num_joints:
                if joints.size(1) > self.num_joints:
                    joints = joints[:, : self.num_joints, :]
                else:
                    pad = torch.zeros(
                        joints.size(0),
                        self.num_joints - joints.size(1),
                        joints.size(2),
                        device=device,
                        dtype=joints.dtype,
                    )
                    joints = torch.cat((joints, pad), dim=1)
            return joints.reshape(batch_size, seq_len, self.num_joints, 3)
        except Exception as exc:
            print(f"FK计算失败: {exc}")
            return None

    def _compute_root_velocity_from_trans(self, trans: torch.Tensor):
        if trans is None:
            return None
        if trans.dim() == 2:
            trans = trans.unsqueeze(1)
        vel = torch.zeros_like(trans)
        if trans.size(1) > 1:
            vel[:, 1:] = (trans[:, 1:] - trans[:, :-1]) * self.fps
            vel[:, 0] = vel[:, 1]
        return vel

    @staticmethod
    def _ensure_bt(tensor, batch_size, seq_len, last_shape, device, dtype):
        full_shape = (batch_size, seq_len, *last_shape)
        if not isinstance(tensor, torch.Tensor):
            return torch.zeros(*full_shape, device=device, dtype=dtype)
        out = tensor.to(device=device, dtype=dtype)
        if out.dim() == len(last_shape) + 1:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, *out.shape[1:])
        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return torch.zeros(*full_shape, device=device, dtype=dtype)
        return out

    def _prepare_inputs(self, data_dict: dict):
        """Pack observed IMU + seed motion features."""
        human_imu = data_dict["human_imu"]
        v_init = data_dict["v_init"]
        p_init = data_dict["p_init"]
        trans_init = data_dict.get("trans_init")
        trans_gt = data_dict.get("trans_gt")

        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B, T, num_imu, imu_dim], got {human_imu.shape}")
        batch_size, seq_len, _, _ = human_imu.shape
        device = human_imu.device
        dtype = human_imu.dtype

        if self.body_model is not None and (self.body_model_device != device):
            self.body_model = self.body_model.to(device)
            self.body_model_device = device

        human_flat = human_imu.reshape(batch_size, seq_len, -1)

        v_init_ordered = v_init[:, self.velocity_order, :].reshape(batch_size, len(self.velocity_order), 3)
        v_seed = v_init_ordered.reshape(batch_size, 1, -1).expand(batch_size, seq_len, -1)
        p_seed = p_init.reshape(batch_size, 1, -1).expand(batch_size, seq_len, -1)

        root_seed = torch.zeros(batch_size, seq_len, self.root_dim, device=device, dtype=dtype)
        motion_seed_parts = [v_seed, p_seed]
        if not self.no_trans:
            motion_seed_parts.append(root_seed)
        motion_seed = torch.cat(motion_seed_parts, dim=-1)

        x_seed = torch.cat((human_flat, motion_seed), dim=-1)

        orientation_6d = human_imu[..., -6:]
        orientation_mat = rotation_6d_to_matrix(orientation_6d.reshape(-1, 6)).reshape(
            batch_size, seq_len, self.num_human_imus, 3, 3
        )

        if trans_init is None:
            trans_init_vec = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        else:
            trans_init_vec = trans_init.to(device=device, dtype=dtype)
            if trans_init_vec.dim() == 3 and trans_init_vec.size(1) == 1:
                trans_init_vec = trans_init_vec[:, 0]
            if trans_init_vec.dim() == 1:
                trans_init_vec = trans_init_vec.unsqueeze(0).expand(batch_size, -1)

        if isinstance(trans_gt, torch.Tensor):
            trans_gt = trans_gt.to(device=device, dtype=dtype)
            if trans_gt.dim() == 2:
                trans_gt = trans_gt.unsqueeze(0).expand(batch_size, seq_len, 3)

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "human_flat": human_flat,
            "human_imu": human_imu,
            "orientation_mat": orientation_mat,
            "motion_seed": motion_seed,
            "trans_init": trans_init_vec,
            "trans_gt": trans_gt,
        }
        return x_seed, context

    def _build_motion_target_from_gt(self, gt_targets: dict, context: dict):
        """Construct clean motion target from GT labels."""
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        orientation_mat = context["orientation_mat"]

        sensor_vel_root_gt = gt_targets.get("sensor_vel_root") if isinstance(gt_targets, dict) else None
        sensor_vel_root_gt = self._ensure_bt(
            sensor_vel_root_gt,
            batch_size,
            seq_len,
            (len(_SENSOR_VEL_NAMES), 3),
            device,
            dtype,
        )
        v_gt = sensor_vel_root_gt[:, :, self.velocity_order, :].reshape(batch_size, seq_len, -1)

        ori_root_reduced_gt = gt_targets.get("ori_root_reduced") if isinstance(gt_targets, dict) else None
        ori_root_reduced_gt = self._ensure_bt(
            ori_root_reduced_gt,
            batch_size,
            seq_len,
            (len(_REDUCED_POSE_NAMES), 3, 3),
            device,
            dtype,
        )

        p_gt_6d = matrix_to_rotation_6d(ori_root_reduced_gt.reshape(-1, 3, 3)).reshape(
            batch_size, seq_len, len(_REDUCED_POSE_NAMES), 6
        )
        p_gt_flat = p_gt_6d.reshape(batch_size, seq_len, -1)

        motion_parts = [v_gt, p_gt_flat]
        if not self.no_trans:
            root_vel_gt = gt_targets.get("root_vel") if isinstance(gt_targets, dict) else None
            if isinstance(root_vel_gt, torch.Tensor):
                root_vel_gt = root_vel_gt.to(device=device, dtype=dtype)
                if root_vel_gt.dim() == 2:
                    root_vel_gt = root_vel_gt.unsqueeze(0)
                if root_vel_gt.shape[0] == 1 and batch_size > 1:
                    root_vel_gt = root_vel_gt.expand(batch_size, -1, -1)
            else:
                root_vel_gt = None

            if root_vel_gt is None or root_vel_gt.shape[0] != batch_size:
                trans_full_gt = gt_targets.get("trans") if isinstance(gt_targets, dict) else None
                if isinstance(trans_full_gt, torch.Tensor):
                    trans_full_gt = trans_full_gt.to(device=device, dtype=dtype)
                root_vel_gt = self._compute_root_velocity_from_trans(trans_full_gt)
                if root_vel_gt is None:
                    root_vel_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)

            root_R = orientation_mat[:, :, 0]
            root_vel_local_gt = torch.matmul(root_R.transpose(-1, -2), root_vel_gt.unsqueeze(-1)).squeeze(-1)
            motion_parts.append(root_vel_local_gt)

        return torch.cat(motion_parts, dim=-1)

    def set_train_rollout_k(self, rollout_k: int):
        rollout_k = int(rollout_k)
        if rollout_k <= 0:
            raise ValueError(f"rollout_k must be positive, got {rollout_k}")
        self.train_rollout_k = rollout_k

    def _build_step_inpaint(
        self,
        history_feats: torch.Tensor,
        human_flat: torch.Tensor,
        frame_idx: int,
    ):
        """Build inpainting input/mask for one autoregressive frame."""
        x_input = history_feats.clone()
        inpaint_mask = torch.ones_like(x_input, dtype=torch.bool)
        idx = max(0, min(int(frame_idx), x_input.shape[1] - 1))
        prev_idx = max(idx - 1, 0)

        # Observed IMU at current frame is fixed.
        x_input[:, idx, : self.imu_feat_dim] = human_flat[:, idx]
        # Unknown motion is initialized from previous frame reconstruction.
        x_input[:, idx, self.imu_feat_dim :] = history_feats[:, prev_idx, self.imu_feat_dim :]
        # Only current-frame motion dims are allowed to change.
        inpaint_mask[:, idx, self.imu_feat_dim :] = False

        unknown_mask = ~inpaint_mask
        unknown_motion_mask = unknown_mask[..., self.imu_feat_dim :]
        return x_input, inpaint_mask, unknown_mask, unknown_motion_mask

    def _correct_root_translation(self, root_trans: torch.Tensor, joints_local: torch.Tensor):
        """Contact-aware root correction to reduce foot sliding drift."""
        if root_trans is None or joints_local is None or root_trans.size(1) < 2:
            return root_trans

        trans_corr = root_trans.clone()
        batch_size, seq_len = trans_corr.shape[:2]

        for t in range(1, seq_len):
            foot_prev = joints_local[:, t - 1, self.foot_joint_indices, :] + trans_corr[:, t - 1].unsqueeze(1)
            foot_curr = joints_local[:, t, self.foot_joint_indices, :] + trans_corr[:, t].unsqueeze(1)

            foot_vel = (foot_curr - foot_prev) * self.fps
            foot_speed = torch.norm(foot_vel, dim=-1)  # [B,2]
            contact_mask = foot_speed < self.root_contact_vel_threshold
            contact_weight = contact_mask.float()

            denom = contact_weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
            drift = ((foot_curr - foot_prev) * contact_weight.unsqueeze(-1)).sum(dim=1) / denom
            has_contact = (contact_weight.sum(dim=-1, keepdim=True) > 0).float()
            correction = drift * has_contact

            # Apply correction from current frame onward for continuity.
            trans_corr[:, t:] = trans_corr[:, t:] - correction.unsqueeze(1)

        return trans_corr

    def _decode_outputs(self, pred_feats: torch.Tensor, context: dict):
        """Decode window feature prediction into structured outputs."""
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        orientation_mat = context["orientation_mat"]
        trans_init = context["trans_init"]
        trans_gt = context["trans_gt"]

        imu_feat_pred = pred_feats[..., : self.imu_feat_dim]
        motion_pred = pred_feats[..., self.imu_feat_dim :]

        offset = 0
        v_pred = motion_pred[..., offset : offset + self.v_dim]
        offset += self.v_dim
        p_pred_flat = motion_pred[..., offset : offset + self.p_dim]
        offset += self.p_dim

        root_vel_local_pred = None
        if not self.no_trans:
            root_vel_local_pred = motion_pred[..., offset : offset + 3]

        v_pred = v_pred.view(batch_size, seq_len, len(self.velocity_order), 3)
        p_pred = p_pred_flat.view(batch_size, seq_len, len(_REDUCED_POSE_NAMES), 6)

        root_R = orientation_mat[:, :, 0]

        if self.no_trans:
            if trans_gt is None:
                trans_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
            root_trans_pred = trans_gt
            root_vel_pred = self._compute_root_velocity_from_trans(root_trans_pred)
            root_vel_local_pred = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        else:
            root_vel_local_pred = root_vel_local_pred.view(batch_size, seq_len, 3)
            root_vel_pred = torch.matmul(root_R, root_vel_local_pred.unsqueeze(-1)).squeeze(-1)
            root_trans_delta = torch.cumsum(root_vel_pred, dim=1) / self.fps
            root_trans_pred = root_trans_delta + trans_init.unsqueeze(1)

        joints_pos = None
        full_glb_rotmats = None
        full_glb_rot6d = None
        if self.body_model is not None:
            joints_pos = self._compute_fk_joints_batched(p_pred, orientation_mat.clone())
            try:
                BT = batch_size * seq_len
                glb_pose_flat = p_pred.reshape(BT, len(_REDUCED_POSE_NAMES), 6)
                orientation_flat = orientation_mat[:, :, : len(_SENSOR_ROT_INDICES)].reshape(
                    BT, len(_SENSOR_ROT_INDICES), 3, 3
                )
                full_glb_rotmats_flat = self._reduced_glb_6d_to_full_glb_mat(glb_pose_flat, orientation_flat)
                full_glb_rotmats = full_glb_rotmats_flat.reshape(batch_size, seq_len, self.num_joints, 3, 3)
                full_glb_rot6d = matrix_to_rotation_6d(full_glb_rotmats_flat.reshape(-1, 3, 3)).reshape(
                    batch_size, seq_len, self.num_joints, 6
                )
            except Exception as exc:
                print(f"Failed to compute full pose rotations: {exc}")
                full_glb_rotmats = None
                full_glb_rot6d = None

        if self.enable_root_correction and (not self.no_trans) and joints_pos is not None:
            root_trans_pred = self._correct_root_translation(root_trans_pred, joints_pos)
            root_vel_pred = self._compute_root_velocity_from_trans(root_trans_pred)
            if root_vel_pred is None:
                root_vel_pred = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
            root_vel_local_pred = torch.matmul(root_R.transpose(-1, -2), root_vel_pred.unsqueeze(-1)).squeeze(-1)

        if joints_pos is not None:
            lhand = joints_pos[:, :, self.hand_joint_indices[0], :] + root_trans_pred
            rhand = joints_pos[:, :, self.hand_joint_indices[1], :] + root_trans_pred
            pred_hand_glb_pos = torch.stack((lhand, rhand), dim=2)
        else:
            pred_hand_glb_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)

        results = {
            "pred_imu_feat": imu_feat_pred,
            "v_pred": v_pred.reshape(batch_size, seq_len, -1),
            "p_pred": p_pred,
            "pred_hand_glb_pos": pred_hand_glb_pos,
            "root_vel_pred": root_vel_pred,
            "root_trans_pred": root_trans_pred,
        }

        if not self.no_trans:
            results["root_vel_local_pred"] = root_vel_local_pred

        if joints_pos is not None:
            results["pred_joints_local"] = joints_pos
            results["pred_joints_global"] = joints_pos + root_trans_pred.unsqueeze(2)
        else:
            results["pred_joints_local"] = torch.zeros(
                batch_size, seq_len, self.num_joints, 3, device=device, dtype=dtype
            )
            results["pred_joints_global"] = torch.zeros_like(results["pred_joints_local"])

        if full_glb_rotmats is not None:
            results["pred_full_pose_rotmat"] = full_glb_rotmats
            results["pred_full_pose_6d"] = full_glb_rot6d
        else:
            results["pred_full_pose_rotmat"] = torch.zeros(
                batch_size, seq_len, self.num_joints, 3, 3, device=device, dtype=dtype
            )
            results["pred_full_pose_6d"] = torch.zeros(
                batch_size, seq_len, self.num_joints, 6, device=device, dtype=dtype
            )

        return results

    def forward(self, data_dict: dict, gt_targets: dict | None = None):
        """Training forward with K-step autoregressive rollout."""
        if self.training and gt_targets is None:
            raise ValueError("Training forward requires gt_targets")

        x_seed, context = self._prepare_inputs(data_dict)
        human_flat = context["human_flat"]

        if gt_targets is None:
            motion_target = context["motion_seed"]
        else:
            motion_target = self._build_motion_target_from_gt(gt_targets, context)

        x_target = torch.cat((human_flat, motion_target), dim=-1)

        seq_len = int(context["seq_len"])
        if seq_len < 2:
            raise ValueError("Sequence length must be >= 2 for K-step rollout training.")

        effective_k = min(int(self.train_rollout_k), seq_len - 1)
        if effective_k < 1:
            raise ValueError(
                f"Effective rollout_k must be >= 1, got {effective_k}. "
                f"(requested={self.train_rollout_k}, seq_len={seq_len})"
            )
        rollout_start = seq_len - effective_k

        recon_history = x_seed.clone()
        recon_history[..., : self.imu_feat_dim] = human_flat
        recon_history[:, :rollout_start, self.imu_feat_dim :] = motion_target[:, :rollout_start]
        recon_for_loss = recon_history.clone()

        batch_size = int(context["batch_size"])
        device = context["device"]
        motion_slice = slice(self.imu_feat_dim, None)
        rollout_frames = torch.arange(rollout_start, seq_len, device=device, dtype=torch.long)
        rollout_t = []

        last_t = None
        last_x_t = None
        last_x_input = None
        last_x0_pred = None
        last_eps_pred = None
        last_model_out = None

        for frame_idx in rollout_frames.tolist():
            x_input, inpaint_mask_step, _, _ = self._build_step_inpaint(
                recon_history,
                human_flat,
                frame_idx,
            )

            if self.use_diffusion_noise:
                t = torch.randint(
                    0,
                    self.dit.timesteps,
                    (batch_size,),
                    device=device,
                    dtype=torch.long,
                )
                x_t_noisy = self.dit.q_sample(x_target, t, noise=torch.randn_like(x_target))
                known_noisy = self.dit.q_sample(x_input, t, noise=torch.randn_like(x_input))
                x_t = torch.where(inpaint_mask_step, known_noisy, x_t_noisy)
            else:
                t = torch.zeros(batch_size, device=device, dtype=torch.long)
                x_t = x_input

            x0_pred, eps_pred, model_out = self.dit.predict(x_t=x_t, t=t, cond=None)
            x0_projected_step = torch.where(inpaint_mask_step, x_input, x0_pred)

            step_motion = x0_projected_step[:, frame_idx, motion_slice]
            # Detach while writing history to avoid cross-step BPTT.
            recon_history[:, frame_idx, motion_slice] = step_motion.detach()
            recon_history[:, frame_idx, : self.imu_feat_dim] = human_flat[:, frame_idx]
            # Keep non-detached step predictions for loss accumulation.
            recon_for_loss[:, frame_idx, motion_slice] = step_motion
            recon_for_loss[:, frame_idx, : self.imu_feat_dim] = human_flat[:, frame_idx]

            rollout_t.append(t)
            last_t = t
            last_x_t = x_t
            last_x_input = x_input
            last_x0_pred = x0_pred
            last_eps_pred = eps_pred
            last_model_out = model_out

        rollout_frame_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        rollout_frame_mask[:, rollout_start:] = True
        unknown_mask = torch.zeros_like(x_target, dtype=torch.bool)
        unknown_mask[:, rollout_start:, self.imu_feat_dim :] = True
        unknown_motion_mask = unknown_mask[..., self.imu_feat_dim :]
        inpaint_mask = ~unknown_mask

        if rollout_t:
            rollout_t_tensor = torch.stack(rollout_t, dim=1)
        else:
            rollout_t_tensor = torch.zeros(batch_size, 0, device=device, dtype=torch.long)

        results = self._decode_outputs(recon_for_loss, context)
        results["diffusion_aux"] = {
            "t": last_t,
            "x_t": last_x_t,
            "x0_target": x_target,
            "x0_pred": last_x0_pred,
            "x0_projected": recon_for_loss,
            "eps_pred": last_eps_pred,
            "model_out": last_model_out,
            "inpaint_mask": inpaint_mask,
            "unknown_mask": unknown_mask,
            "unknown_motion_mask": unknown_motion_mask,
            "x_input": last_x_input,
            "rollout_k": int(effective_k),
            "rollout_start_idx": int(rollout_start),
            "rollout_frame_mask": rollout_frame_mask,
            "rollout_frame_indices": rollout_frames,
            "rollout_t": rollout_t_tensor,
            "rollout_detach_history": True,
            "prediction_type": self.dit.prediction_type,
        }
        return results

    def _infer_autoregressive(
        self,
        x_seed: torch.Tensor,
        context: dict,
        gt_motion: torch.Tensor,
        *,
        steps: int,
        sampler_name: str,
        eta_val: float,
    ) -> torch.Tensor:
        """Autoregressive rollout with GT warmup history."""
        human_flat = context["human_flat"]
        batch_size, seq_len = x_seed.shape[:2]
        warmup_len = self.warmup_len

        if seq_len <= warmup_len:
            raise ValueError(
                f"Sequence length {seq_len} is too short for warmup_len={warmup_len}. "
                "Need T > window_size - 1."
            )

        recon = x_seed.clone()
        recon[..., : self.imu_feat_dim] = human_flat
        recon[:, :warmup_len, self.imu_feat_dim :] = gt_motion[:, :warmup_len]

        for frame_idx in range(warmup_len, seq_len):
            x_input, inpaint_mask, _, _ = self._build_step_inpaint(recon, human_flat, frame_idx)

            pred_full = self.dit.sample_inpaint(
                x_input=x_input,
                inpaint_mask=inpaint_mask,
                cond=None,
                x_start=x_input,
                steps=steps,
                sampler=sampler_name,
                eta=eta_val,
            )

            recon[:, frame_idx, self.imu_feat_dim :] = pred_full[:, frame_idx, self.imu_feat_dim :]
            recon[:, frame_idx, : self.imu_feat_dim] = human_flat[:, frame_idx]

        return recon

    @torch.no_grad()
    def inference(
        self,
        data_dict: dict,
        gt_targets: dict | None = None,
        sample_steps: int | None = None,
        sampler: str | None = None,
        eta: float | None = None,
    ):
        """Inference with GT-warmup autoregressive rollout."""
        if gt_targets is None:
            raise ValueError("Stage-1 inference requires gt_targets for GT warmup history.")
        if not isinstance(gt_targets.get("sensor_vel_root"), torch.Tensor):
            raise ValueError("gt_targets['sensor_vel_root'] is required for Stage-1 GT warmup.")
        if not isinstance(gt_targets.get("ori_root_reduced"), torch.Tensor):
            raise ValueError("gt_targets['ori_root_reduced'] is required for Stage-1 GT warmup.")
        if (not self.no_trans) and (
            not isinstance(gt_targets.get("root_vel"), torch.Tensor)
            and not isinstance(gt_targets.get("trans"), torch.Tensor)
        ):
            raise ValueError("gt_targets must provide 'root_vel' or 'trans' for Stage-1 GT warmup.")

        x_seed, context = self._prepare_inputs(data_dict)
        gt_motion = self._build_motion_target_from_gt(gt_targets, context)
        steps = sample_steps if sample_steps is not None else self.inference_steps
        if steps is None:
            steps = self.dit.timesteps
        steps = int(steps)

        sampler_name = self.inference_sampler if sampler is None else str(sampler).lower()
        eta_val = self.inference_eta if eta is None else float(eta)
        pred_full = self._infer_autoregressive(
            x_seed,
            context,
            gt_motion=gt_motion,
            steps=steps,
            sampler_name=sampler_name,
            eta_val=eta_val,
        )

        results = self._decode_outputs(pred_full, context)
        results["diffusion_aux"] = {
            "inference_steps": steps,
            "sampler": sampler_name,
            "eta": eta_val,
            "window_size": self.window_size,
            "warmup_len": self.warmup_len,
            "prediction_type": self.dit.prediction_type,
        }
        return results

    @staticmethod
    def empty_output(
        batch_size: int, seq_len: int, device: torch.device, no_trans: bool = False, num_joints: int = 24
    ):
        results = {
            "pred_imu_feat": torch.zeros(batch_size, seq_len, len(_SENSOR_NAMES) * 9, device=device),
            "v_pred": torch.zeros(batch_size, seq_len, len(_SENSOR_VEL_NAMES) * 3, device=device),
            "p_pred": torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES), 6, device=device),
            "pred_hand_glb_pos": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "root_vel_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "root_trans_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_joints_local": torch.zeros(batch_size, seq_len, num_joints, 3, device=device),
            "pred_joints_global": torch.zeros(batch_size, seq_len, num_joints, 3, device=device),
            "pred_full_pose_rotmat": torch.zeros(batch_size, seq_len, num_joints, 3, 3, device=device),
            "pred_full_pose_6d": torch.zeros(batch_size, seq_len, num_joints, 6, device=device),
            "diffusion_aux": {},
        }
        if not no_trans:
            results["root_vel_local_pred"] = torch.zeros(batch_size, seq_len, 3, device=device)
        return results
