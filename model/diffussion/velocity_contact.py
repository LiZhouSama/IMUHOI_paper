"""
DiT-based VelocityContactModule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d, axis_angle_to_matrix

from .base import ConditionalDiT
from configs import FRAME_RATE, _SENSOR_NAMES
from utils.utils import _central_diff, _smooth_acceleration


class VelocityContactModule(nn.Module):
    """Diffusion Transformer replacement for Stage-2 velocity/contact."""

    def __init__(self, cfg):
        super().__init__()
        self.num_joints = getattr(cfg, "num_joints", 24)
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.obj_imu_dim = getattr(cfg, "obj_imu_dim", self.imu_dim)
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))

        self.hand_vel_dim = 2 * 3
        self.obj_vel_dim = 3
        self.contact_dim = 3  # left/right/object move
        self.boundary_dim = 2
        self.target_dim = self.hand_vel_dim + self.obj_vel_dim + self.contact_dim + self.boundary_dim
        self.hand_joint_indices = (20, 21)

        cond_dim = self.num_human_imus * self.imu_dim + self.obj_imu_dim
        # optional pose-derived conditioning (hand pos + root vel + full pose 6D)
        self.pose_dim = self.num_joints * 6
        self.extra_cond_dim = 6 + 3 + self.pose_dim
        cond_dim = cond_dim + self.extra_cond_dim

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
        self.teacher_force_noise_std = float(_dit_param("teacher_force_noise_std", getattr(cfg, "teacher_force_noise_std", 0.01)))
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

        self.left_hand_sensor = _SENSOR_NAMES.index("LeftForeArm")
        self.right_hand_sensor = _SENSOR_NAMES.index("RightForeArm")

    def _denormalize_imu(self, human_imu: torch.Tensor):
        """Convert IMU data back to world frame."""
        batch_size, seq_len = human_imu.shape[:2]

        human_imu_acc = human_imu[:, :, :, :3]
        human_imu_ori = human_imu[:, :, :, 3:9]
        human_imu_ori_6d = human_imu_ori.reshape(-1, 6)
        human_imu_ori_mat = rotation_6d_to_matrix(human_imu_ori_6d).reshape(batch_size, seq_len, self.num_human_imus, 3, 3)

        R0T = human_imu_ori_mat[:, :, 0].transpose(-1, -2)
        acc_world = torch.matmul(human_imu_acc, R0T)
        acc0_world = acc_world[:, :, :1, :]
        acc_rest_mix = acc_world[:, :, 1:, :] + acc0_world
        human_imu_acc_denorm = torch.cat([acc0_world, acc_rest_mix], dim=2)

        human_imu_ori_denorm = torch.cat(
            [
                human_imu_ori_mat[:, :, :1],
                human_imu_ori_mat[:, :, :1].matmul(human_imu_ori_mat[:, :, 1:]),
            ],
            dim=2,
        )
        human_imu_ori_denorm_6d = matrix_to_rotation_6d(human_imu_ori_denorm)
        human_imu_denorm = torch.cat([human_imu_acc_denorm, human_imu_ori_denorm_6d], dim=-1)

        return human_imu_denorm

    def _prepare_inputs(
        self,
        data_dict: dict,
        hp_out: dict | None = None,
        gt_targets: dict | None = None,
        teacher_forcing: bool = False,
    ):
        """Shared conditioning + seed preparation for VC module."""
        human_imu = data_dict["human_imu"]
        obj_imu = data_dict.get("obj_imu")
        hand_vel_init = data_dict["hand_vel_glb_init"]
        obj_vel_init = data_dict["obj_vel_init"]
        contact_init = data_dict.get("contact_init")

        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B, T, num_imu, imu_dim], got {human_imu.shape}")
        batch_size, seq_len, _, _ = human_imu.shape
        device = human_imu.device
        dtype = human_imu.dtype

        def _prepare_gt_tensor(val, target_dim):
            if not isinstance(val, torch.Tensor):
                return None
            tensor = val.to(device=device, dtype=dtype)
            if tensor.dim() == len(target_dim) - 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[0] == 1 and batch_size > 1:
                tensor = tensor.expand(batch_size, *tensor.shape[1:])
            if tensor.shape[0] != batch_size:
                return None
            return tensor

        if obj_imu is None:
            obj_imu = torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        else:
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(batch_size, seq_len, -1)
            elif obj_imu.dim() != 3:
                raise ValueError(f"obj_imu must be [B, T, obj_dim], got {obj_imu.shape}")
        if obj_imu.shape[-1] != self.obj_imu_dim:
            raise ValueError(f"obj_imu feature dim mismatch: expected {self.obj_imu_dim}, got {obj_imu.shape[-1]}")

        if hand_vel_init.dim() != 3 or hand_vel_init.shape[1:] != (2, 3):
            raise ValueError(f"hand_vel_init must be [B,2,3], got {hand_vel_init.shape}")
        l_hand_vel_init = hand_vel_init[:, 0, :]
        r_hand_vel_init = hand_vel_init[:, 1, :]

        if obj_vel_init.dim() == 1:
            obj_vel_init_vec = obj_vel_init.unsqueeze(0).expand(batch_size, -1)
        elif obj_vel_init.dim() == 2:
            obj_vel_init_vec = obj_vel_init
        else:
            raise ValueError(f"obj_vel_init must be [B,3] or [3], got {obj_vel_init.shape}")

        human_imu_denorm = self._denormalize_imu(human_imu)
        pose_streams = {}
        use_tf_cond = teacher_forcing and self.training and gt_targets is not None
        if use_tf_cond:
            pos_gt = _prepare_gt_tensor(
                gt_targets.get("position_global") if isinstance(gt_targets, dict) else None, (batch_size, seq_len, self.num_joints, 3)
            )
            if pos_gt is not None and pos_gt.shape[1] == seq_len and pos_gt.shape[2] > max(self.hand_joint_indices):
                pose_streams["hand_pos"] = pos_gt[:, :, list(self.hand_joint_indices), :]
            root_vel_gt = _prepare_gt_tensor(
                gt_targets.get("root_vel") if isinstance(gt_targets, dict) else None, (batch_size, seq_len, 3)
            )
            if root_vel_gt is not None:
                pose_streams["root_vel"] = root_vel_gt
            pose_aa = _prepare_gt_tensor(
                gt_targets.get("pose") if isinstance(gt_targets, dict) else None, (batch_size, seq_len, self.num_joints * 3)
            )
            if pose_aa is not None:
                pose_aa = pose_aa.view(pose_aa.shape[0], pose_aa.shape[1], -1, 3)
                rotm = axis_angle_to_matrix(pose_aa.reshape(-1, 3)).reshape(pose_aa.shape[0], pose_aa.shape[1], -1, 3, 3)
                current_joints = rotm.shape[2]
                if current_joints < self.num_joints:
                    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
                    pad = eye.expand(pose_aa.shape[0], pose_aa.shape[1], self.num_joints - current_joints, 3, 3)
                    rotm = torch.cat((rotm, pad), dim=2)
                elif current_joints > self.num_joints:
                    rotm = rotm[:, :, : self.num_joints]
                pose_streams["pose_6d"] = matrix_to_rotation_6d(rotm.reshape(-1, 3, 3)).reshape(
                    pose_aa.shape[0], pose_aa.shape[1], self.num_joints, 6
                )
            noise_std = self.teacher_force_noise_std
            if noise_std > 0:
                if "hand_pos" in pose_streams:
                    pose_streams["hand_pos"] = pose_streams["hand_pos"] + torch.randn_like(pose_streams["hand_pos"]) * noise_std
                if "root_vel" in pose_streams:
                    pose_streams["root_vel"] = pose_streams["root_vel"] + torch.randn_like(pose_streams["root_vel"]) * noise_std
                if "pose_6d" in pose_streams:
                    pose_streams["pose_6d"] = pose_streams["pose_6d"] + torch.randn_like(pose_streams["pose_6d"]) * noise_std
        if hp_out is not None:
            pose_streams.setdefault("hand_pos", hp_out["pred_hand_glb_pos"])
            pose_streams.setdefault("root_vel", hp_out["root_vel_pred"])
            pose_streams.setdefault("pose_6d", hp_out["pred_full_pose_6d"])
        if pose_streams.get("hand_pos") is None:
            pose_streams["hand_pos"] = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)
        if pose_streams.get("root_vel") is None:
            pose_streams["root_vel"] = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        if pose_streams.get("pose_6d") is None:
            pose_streams["pose_6d"] = torch.zeros(batch_size, seq_len, self.num_joints, 6, device=device, dtype=dtype)
            print(f"Warning: pose_6d is not provided, using zeros. Shape: {pose_streams['pose_6d'].shape}")
        else:
            current_joints = pose_streams["pose_6d"].shape[2]
            if current_joints != self.num_joints:
                if current_joints > self.num_joints:
                    pose_streams["pose_6d"] = pose_streams["pose_6d"][:, :, : self.num_joints, :]
                else:
                    pad = torch.zeros(batch_size, seq_len, self.num_joints - current_joints, 6, device=device, dtype=dtype)
                    pose_streams["pose_6d"] = torch.cat((pose_streams["pose_6d"], pad), dim=2)

        hand_pos = pose_streams["hand_pos"]
        root_vel = pose_streams["root_vel"]
        pose_6d = pose_streams["pose_6d"]
        pose_feat = pose_6d.reshape(batch_size, seq_len, -1)

        cond_parts = [
            human_imu_denorm.reshape(batch_size, seq_len, -1),
            obj_imu,
            hand_pos.reshape(batch_size, seq_len, -1),
            root_vel,
            pose_feat,
        ]
        cond = torch.cat(cond_parts, dim=-1)

        hand_seed = torch.stack((l_hand_vel_init, r_hand_vel_init), dim=1).reshape(batch_size, 1, -1)
        hand_seed = hand_seed.expand(batch_size, seq_len, -1)
        obj_vel_seed = obj_vel_init_vec.unsqueeze(1).expand(batch_size, seq_len, -1)
        contact_seed = contact_init
        if contact_seed is None:
            contact_seed = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        elif contact_seed.dim() == 1:
            contact_seed = contact_seed.unsqueeze(0).expand(batch_size, -1)
        contact_seed = contact_seed[:, :3]
        contact_seed = contact_seed.unsqueeze(1).expand(batch_size, seq_len, -1)
        x_seed = torch.cat(
            [
                hand_seed,
                obj_vel_seed,
                contact_seed,
                torch.zeros(batch_size, seq_len, self.boundary_dim, device=device, dtype=dtype),
            ],
            dim=-1,
        )

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "cond": cond,
            "x_seed": x_seed,
        }
        return cond, x_seed, context

    def _build_x_start(self, gt_targets: dict, context: dict):
        """Construct diffusion x_start for training using ground truth signals."""
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]

        sensor_vel_glb_gt = gt_targets.get("sensor_vel_glb") if isinstance(gt_targets, dict) else None
        if isinstance(sensor_vel_glb_gt, torch.Tensor):
            sensor_vel_glb_gt = sensor_vel_glb_gt.to(device=device, dtype=dtype)
            if sensor_vel_glb_gt.dim() == 3:
                sensor_vel_glb_gt = sensor_vel_glb_gt.unsqueeze(0)
            if sensor_vel_glb_gt.shape[0] == 1 and batch_size > 1:
                sensor_vel_glb_gt = sensor_vel_glb_gt.expand(batch_size, -1, -1, -1)
        if sensor_vel_glb_gt is None or sensor_vel_glb_gt.shape[0] != batch_size:
            sensor_vel_glb_gt = torch.zeros(batch_size, seq_len, len(_SENSOR_NAMES), 3, device=device, dtype=dtype)
        hand_indices = [-2, -1]
        hand_vel_gt = sensor_vel_glb_gt[:, :, hand_indices, :].reshape(batch_size, seq_len, -1)

        obj_vel_gt = gt_targets.get("obj_vel") if isinstance(gt_targets, dict) else None
        if isinstance(obj_vel_gt, torch.Tensor):
            obj_vel_gt = obj_vel_gt.to(device=device, dtype=dtype)
            if obj_vel_gt.dim() == 2:
                obj_vel_gt = obj_vel_gt.unsqueeze(0)
            if obj_vel_gt.shape[0] == 1 and batch_size > 1:
                obj_vel_gt = obj_vel_gt.expand(batch_size, -1, -1)
        if obj_vel_gt is None or obj_vel_gt.shape[0] != batch_size:
            obj_vel_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)

        def _prepare_contact(t):
            if isinstance(t, torch.Tensor):
                t = t.to(device=device, dtype=dtype)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                if t.shape[0] == 1 and batch_size > 1:
                    t = t.expand(batch_size, -1)
                return t
            return torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

        lhand_contact = _prepare_contact(gt_targets.get("lhand_contact") if isinstance(gt_targets, dict) else None)
        rhand_contact = _prepare_contact(gt_targets.get("rhand_contact") if isinstance(gt_targets, dict) else None)
        obj_contact = _prepare_contact(gt_targets.get("obj_contact") if isinstance(gt_targets, dict) else None)
        contact_gt = torch.stack([lhand_contact, rhand_contact, obj_contact], dim=-1).clamp(0.0, 1.0)

        def _prepare_boundary(t):
            if isinstance(t, torch.Tensor):
                t = t.to(device=device, dtype=dtype)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                if t.shape[0] == 1 and batch_size > 1:
                    t = t.expand(batch_size, -1)
                return t
            return torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

        start_gt = _prepare_boundary(
            gt_targets.get("interaction_start_gauss", gt_targets.get("interaction_start") if isinstance(gt_targets, dict) else None)
            if isinstance(gt_targets, dict)
            else None
        )
        end_gt = _prepare_boundary(
            gt_targets.get("interaction_end_gauss", gt_targets.get("interaction_end") if isinstance(gt_targets, dict) else None)
            if isinstance(gt_targets, dict)
            else None
        )
        boundary_gt = torch.stack([start_gt, end_gt], dim=-1)

        return torch.cat([hand_vel_gt, obj_vel_gt, contact_gt, boundary_gt], dim=-1)

    def _decode_outputs(self, pred_feats: torch.Tensor, context: dict):
        """Decode transformer outputs into structured VC predictions."""
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]

        offset = 0
        hand_vel = pred_feats[..., offset : offset + self.hand_vel_dim]
        offset += self.hand_vel_dim
        obj_vel = pred_feats[..., offset : offset + self.obj_vel_dim]
        offset += self.obj_vel_dim
        contact_logits = pred_feats[..., offset : offset + self.contact_dim]
        offset += self.contact_dim
        boundary_logits = pred_feats[..., offset : offset + self.boundary_dim]

        hand_vel = hand_vel.view(batch_size, seq_len, 2, 3)
        obj_vel = obj_vel.view(batch_size, seq_len, 3)
        contact_prob = torch.sigmoid(contact_logits)
        boundary_prob = torch.sigmoid(boundary_logits)

        return {
            "pred_hand_glb_vel": hand_vel,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": contact_logits,
            "pred_hand_contact_prob": contact_prob,
            "pred_obj_move_prob": contact_prob[..., 2:3],
            "pred_hand_contact_prob_cond": contact_prob[..., :2],
            "pred_hand_contact_logits_cond": contact_logits[..., :2],
            "pred_interaction_boundary_logits": boundary_logits,
            "pred_interaction_boundary_prob": boundary_prob,
        }

    def forward(
        self,
        data_dict: dict,
        hp_out: dict = None,
        gt_targets: dict | None = None,
        teacher_forcing: bool = False,
    ):
        cond, x_seed, context = self._prepare_inputs(
            data_dict, hp_out=hp_out, gt_targets=gt_targets, teacher_forcing=teacher_forcing
        )

        using_gt_xstart = self.training and self.dit_x_start_mode == "gt" and gt_targets is not None
        if self.dit_x_start_mode == "gt" and gt_targets is None and self.training:
            raise ValueError("Training forward requires gt_targets when dit_x_start_mode='gt'")

        x_start = self._build_x_start(gt_targets, context) if using_gt_xstart else x_seed
        add_noise = self.training and (self.use_diffusion_noise or using_gt_xstart)
        pred_feats, aux = self.dit(cond, x_start=x_start, add_noise=add_noise)

        outputs = self._decode_outputs(pred_feats, context)
        outputs["diffusion_aux"] = aux
        return outputs

    @torch.no_grad()
    def inference(
        self,
        data_dict: dict,
        hp_out: dict = None,
        sample_steps: int | None = None,
    ):
        """纯推理接口，固定使用采样路径且不接收gt_targets。"""
        cond, x_seed, context = self._prepare_inputs(
            data_dict, hp_out=hp_out, gt_targets=None, teacher_forcing=False
        )
        steps = sample_steps if sample_steps is not None else self.inference_steps
        pred_feats = self.dit.sample(cond=cond, x_start=x_seed, steps=steps)
        outputs = self._decode_outputs(pred_feats, context)
        outputs["diffusion_aux"] = {"inference_steps": steps if steps is not None else self.dit.timesteps}
        return outputs

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        return {
            "pred_hand_glb_vel": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "pred_obj_vel": zeros_pos.clone(),
            "pred_hand_contact_logits": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_hand_contact_prob": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_move_prob": torch.zeros(batch_size, seq_len, 1, device=device),
            "pred_hand_contact_prob_cond": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_hand_contact_logits_cond": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_interaction_boundary_logits": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_interaction_boundary_prob": torch.zeros(batch_size, seq_len, 2, device=device),
            "diffusion_aux": {},
        }
