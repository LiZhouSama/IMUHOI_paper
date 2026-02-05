"""
DiT-based HumanPoseModule.

The module preserves the public outputs of the RNN version while swapping
the recurrent blocks with a diffusion-style Transformer encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Diffusion Transformer replacement for the Stage-1 pose model."""

    def __init__(self, cfg, device, no_trans: bool = False):
        super().__init__()
        self.device = device
        self.no_trans = no_trans
        self.num_joints = getattr(cfg, "num_joints", 24)
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.hand_joint_indices = (20, 21)
        self.smpl_parents = torch.tensor(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            dtype=torch.long,
        )

        # Match RNN ordering: [Root, LFoot, RFoot, Head, Root, Head, LHand, RHand]
        self.velocity_order = [0, 1, 2, 3, 0, 3, 4, 5]
        self.v_dim = len(self.velocity_order) * 3
        self.p_dim = len(_REDUCED_POSE_NAMES) * 6
        self.root_dim = 0 if no_trans else 3  # root velocity (local)
        self.target_dim = self.v_dim + self.p_dim + self.root_dim

        # conditioning: IMU stream + initial velocity/pose + trans init
        cond_dim = self.num_human_imus * self.imu_dim + self.v_dim + self.p_dim + 3
        dit_cfg = getattr(cfg, "dit", {})

        def _dit_param(name, default):
            if isinstance(dit_cfg, dict) and name in dit_cfg:
                return dit_cfg[name]
            return getattr(cfg, name, default)

        max_seq_len = _dit_param("dit_max_seq_len", getattr(getattr(cfg, "train", {}), "window", 256))
        self.use_diffusion_noise = bool(_dit_param("dit_use_noise", False))
        self.dit_x_start_mode = str(_dit_param("dit_x_start_mode", "gt")).lower()
        self.inference_steps = _dit_param("dit_inference_steps", None)
        if self.inference_steps is not None:
            self.inference_steps = int(self.inference_steps)
        self.dit = ConditionalDiT(
            target_dim=self.target_dim,
            cond_dim=cond_dim,
            d_model=_dit_param("dit_d_model", 256),
            nhead=_dit_param("dit_nhead", 8),
            num_layers=_dit_param("dit_num_layers", 6),
            dim_feedforward=_dit_param("dit_dim_feedforward", 1024),
            dropout=_dit_param("dit_dropout", 0.1),
            max_seq_len=max_seq_len,
            timesteps=_dit_param("dit_timesteps", 1000),
            use_time_embed=_dit_param("dit_use_time_embed", True),
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

    def _prepare_inputs(self, data_dict: dict):
        """Shared input packing for both training and inference."""
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

        if trans_init is None:
            trans_init_vec = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        else:
            trans_init_vec = trans_init.to(device=device, dtype=dtype)
            if trans_init_vec.dim() == 3 and trans_init_vec.size(1) == 1:
                trans_init_vec = trans_init_vec[:, 0]
        trans_seed = trans_init_vec.unsqueeze(1).expand(batch_size, seq_len, 3)

        cond = torch.cat((human_flat, v_seed, p_seed, trans_seed), dim=-1)
        seed_parts = [v_seed, p_seed]
        if not self.no_trans:
            seed_parts.append(torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype))
        x_seed = torch.cat(seed_parts, dim=-1)

        orientation_6d = human_imu[..., -6:]
        orientation_mat = rotation_6d_to_matrix(orientation_6d.reshape(-1, 6)).reshape(
            batch_size, seq_len, self.num_human_imus, 3, 3
        )

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "human_imu": human_imu,
            "trans_init": trans_init_vec,
            "trans_gt": trans_gt,
            "orientation_mat": orientation_mat,
        }
        return cond, x_seed, context

    def _build_x_start_from_gt(self, gt_targets: dict, context: dict):
        """Construct diffusion x_start from ground truth targets."""
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        orientation_mat = context["orientation_mat"]

        sensor_vel_root_gt = gt_targets.get("sensor_vel_root") if isinstance(gt_targets, dict) else None
        if isinstance(sensor_vel_root_gt, torch.Tensor):
            sensor_vel_root_gt = sensor_vel_root_gt.to(device=device, dtype=dtype)
            if sensor_vel_root_gt.dim() == 3:
                sensor_vel_root_gt = sensor_vel_root_gt.unsqueeze(0)
            if sensor_vel_root_gt.shape[0] == 1 and batch_size > 1:
                sensor_vel_root_gt = sensor_vel_root_gt.expand(batch_size, -1, -1, -1)
        if sensor_vel_root_gt is None or sensor_vel_root_gt.shape[0] != batch_size:
            sensor_vel_root_gt = torch.zeros(batch_size, seq_len, len(_SENSOR_VEL_NAMES), 3, device=device, dtype=dtype)
        v_gt = sensor_vel_root_gt[:, :, self.velocity_order, :].reshape(batch_size, seq_len, -1)

        ori_root_reduced_gt = gt_targets.get("ori_root_reduced") if isinstance(gt_targets, dict) else None
        if isinstance(ori_root_reduced_gt, torch.Tensor):
            ori_root_reduced_gt = ori_root_reduced_gt.to(device=device, dtype=dtype)
            if ori_root_reduced_gt.dim() == 4:
                ori_root_reduced_gt = ori_root_reduced_gt.unsqueeze(0)
            if ori_root_reduced_gt.shape[0] == 1 and batch_size > 1:
                ori_root_reduced_gt = ori_root_reduced_gt.expand(batch_size, -1, -1, -1, -1)
        if ori_root_reduced_gt is None or ori_root_reduced_gt.shape[0] != batch_size:
            ori_root_reduced_gt = torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES), 3, 3, device=device, dtype=dtype)
        p_gt_6d = matrix_to_rotation_6d(ori_root_reduced_gt.reshape(-1, 3, 3)).reshape(
            batch_size, seq_len, len(_REDUCED_POSE_NAMES), 6
        )
        p_gt_flat = p_gt_6d.reshape(batch_size, seq_len, -1)

        root_vel_local_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        if not self.no_trans:
            root_vel_gt = gt_targets.get("root_vel") if isinstance(gt_targets, dict) else None
            if isinstance(root_vel_gt, torch.Tensor):
                root_vel_gt = root_vel_gt.to(device=device, dtype=dtype)
                if root_vel_gt.dim() == 2:
                    root_vel_gt = root_vel_gt.unsqueeze(0)
                if root_vel_gt.dim() == 3 and root_vel_gt.shape[0] == 1 and batch_size > 1:
                    root_vel_gt = root_vel_gt.expand(batch_size, -1, -1)
            if root_vel_gt is None or root_vel_gt.shape[0] != batch_size:
                trans_full_gt = gt_targets.get("trans") if isinstance(gt_targets, dict) else None
                if isinstance(trans_full_gt, torch.Tensor):
                    trans_full_gt = trans_full_gt.to(device=device, dtype=dtype)
                root_vel_gt = self._compute_root_velocity_from_trans(trans_full_gt)
                if root_vel_gt is None:
                    root_vel_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)

            root_R = orientation_mat[:, :, 0]
            root_vel_local_gt = torch.matmul(root_R.transpose(-1, -2), root_vel_gt.unsqueeze(-1)).squeeze(-1)

        x_gt_parts = [v_gt, p_gt_flat]
        if not self.no_trans:
            x_gt_parts.append(root_vel_local_gt)
        return torch.cat(x_gt_parts, dim=-1)

    def _decode_outputs(self, pred_feats: torch.Tensor, context: dict):
        """Decode transformer outputs into structured predictions."""
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        orientation_mat = context["orientation_mat"]
        trans_init = context["trans_init"]
        trans_gt = context["trans_gt"]

        offset = 0
        v_pred = pred_feats[..., offset : offset + self.v_dim]
        offset += self.v_dim
        p_pred_flat = pred_feats[..., offset : offset + self.p_dim]
        offset += self.p_dim
        root_vel_local_pred = None
        if not self.no_trans:
            root_vel_local_pred = pred_feats[..., offset : offset + 3]

        v_pred = v_pred.view(batch_size, seq_len, len(self.velocity_order), 3)
        p_pred = p_pred_flat.view(batch_size, seq_len, len(_REDUCED_POSE_NAMES), 6)

        root_R = orientation_mat[:, :, 0]

        if self.no_trans:
            if trans_gt is None:
                trans_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
            else:
                trans_gt = trans_gt.to(device=device, dtype=dtype)
                if trans_gt.dim() == 2:
                    trans_gt = trans_gt.unsqueeze(1).expand(batch_size, seq_len, 3)
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

        if joints_pos is not None:
            lhand = joints_pos[:, :, self.hand_joint_indices[0], :] + root_trans_pred
            rhand = joints_pos[:, :, self.hand_joint_indices[1], :] + root_trans_pred
            pred_hand_glb_pos = torch.stack((lhand, rhand), dim=2)
        else:
            pred_hand_glb_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)

        results = {
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
        """Training forward pass (diffusion noise + optional teacher forcing)."""
        cond, x_seed, context = self._prepare_inputs(data_dict)

        using_gt_xstart = self.training and self.dit_x_start_mode == "gt" and gt_targets is not None
        if self.dit_x_start_mode == "gt" and gt_targets is None and self.training:
            raise ValueError("Training forward requires gt_targets when dit_x_start_mode='gt'")

        x_start = self._build_x_start_from_gt(gt_targets, context) if using_gt_xstart else x_seed
        add_noise = self.training and (self.use_diffusion_noise or using_gt_xstart)
        pred_feats, aux = self.dit(cond, x_start=x_start, add_noise=add_noise)

        results = self._decode_outputs(pred_feats, context)
        results["diffusion_aux"] = aux
        return results

    @torch.no_grad()
    def inference(self, data_dict: dict, sample_steps: int | None = None):
        """Inference-only sampling path that never consumes gt_targets."""
        cond, x_seed, context = self._prepare_inputs(data_dict)
        steps = sample_steps if sample_steps is not None else self.inference_steps
        pred_feats = self.dit.sample(cond=cond, x_start=x_seed, steps=steps)
        results = self._decode_outputs(pred_feats, context)
        results["diffusion_aux"] = {"inference_steps": steps if steps is not None else self.dit.timesteps}
        return results

    @staticmethod
    def empty_output(
        batch_size: int, seq_len: int, device: torch.device, no_trans: bool = False, num_joints: int = 24
    ):
        results = {
            "v_pred": torch.zeros(batch_size, seq_len, len(_SENSOR_VEL_NAMES) * 3, device=device),
            "p_pred": torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6, device=device),
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
