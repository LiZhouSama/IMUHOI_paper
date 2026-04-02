"""
Unified IMUHOI diffusion module (single DiT).

Feature definition per frame:
    x_frame = (
        R_{human+obj},
        a_{human+obj},
        delta_p_root_2d,
        p_root_y,
        p_hands,
        b_foot_contact,
        contact_prob_hand,
        v_bone_dir,
        l_bone_len,
        p_obj,
    )

Training:
- Full-window denoising only.

Inference:
- Autoregressive sliding-window inpainting.
- Observed dims are IMU-only:
  - human IMU-joint rotations + human IMU accelerations
  - object IMU rotation + object IMU acceleration
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


class IMUHOIMixModule(nn.Module):
    """Single DiT for joint human pose + interaction reconstruction."""

    def __init__(self, cfg, device, no_trans: bool = False):
        super().__init__()
        self.device = device
        self.no_trans = bool(no_trans)

        self.num_joints = int(getattr(cfg, "num_joints", 24))
        self.num_human_imus = int(getattr(cfg, "num_human_imus", len(_SENSOR_NAMES)))
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(getattr(cfg, "obj_imu_dim", self.imu_dim))
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))

        self.hand_joint_indices = (20, 21)
        self.foot_joint_indices = (7, 8)
        self.smpl_parents = torch.tensor(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            dtype=torch.long,
        )

        # x_frame = [R_{human+obj}, a_{human+obj}, delta_p_root_2d, p_root_y, p_hands,
        #            b, contact_prob, v_bone, l_bone, p_obj]
        self.human_rot_dim = self.num_joints * 6
        self.obj_rot_dim = 6
        self.rot_dim = self.human_rot_dim + self.obj_rot_dim

        self.human_acc_dim = self.num_human_imus * 3
        self.obj_acc_dim = 3
        self.acc_dim = self.human_acc_dim + self.obj_acc_dim

        self.delta_p_dim = 0 if self.no_trans else 2
        self.py_dim = 0 if self.no_trans else 1
        self.hands_dim = 6
        self.b_dim = 2

        self.contact_dim = 2
        self.bone_dir_dim = 6
        self.bone_len_dim = 2
        self.p_obj_dim = 3

        start = 0
        self.rot_slice = slice(start, start + self.rot_dim)
        start += self.rot_dim
        self.acc_slice = slice(start, start + self.acc_dim)
        start += self.acc_dim
        self.delta_p_slice = slice(start, start + self.delta_p_dim)
        start += self.delta_p_dim
        self.py_slice = slice(start, start + self.py_dim)
        start += self.py_dim
        self.hands_slice = slice(start, start + self.hands_dim)
        start += self.hands_dim
        self.b_slice = slice(start, start + self.b_dim)
        start += self.b_dim
        self.contact_slice = slice(start, start + self.contact_dim)
        start += self.contact_dim
        self.bone_dir_slice = slice(start, start + self.bone_dir_dim)
        start += self.bone_dir_dim
        self.bone_len_slice = slice(start, start + self.bone_len_dim)
        start += self.bone_len_dim
        self.p_obj_slice = slice(start, start + self.p_obj_dim)
        start += self.p_obj_dim
        self.target_dim = start
        # Backward-compatible aliases for older call sites.
        self.delta_p_obj_dim = self.p_obj_dim
        self.delta_p_obj_slice = self.p_obj_slice

        # Observed dims at inference (IMU-only):
        # - human IMU-joint rotations
        # - object rotation from object IMU
        # - human/object accelerations
        observed_dim_mask = torch.zeros(self.target_dim, dtype=torch.bool)
        for j in _SENSOR_ROT_INDICES:
            rot_j = slice(j * 6, (j + 1) * 6)
            observed_dim_mask[rot_j] = True
        observed_dim_mask[self.human_rot_dim : self.rot_dim] = True
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

        inference_type_raw = str(_dit_param("dit_inference_type", "sample_inpaint_x0")).lower()
        if inference_type_raw in {"sample_inpaint_x0", "inpaint_x0", "x0"}:
            self.inference_type = "sample_inpaint_x0"
        elif inference_type_raw in {"sample_inpaint", "inpaint", "diffusion"}:
            self.inference_type = "sample_inpaint"
        else:
            self.inference_type = "sample_inpaint_x0"

        self.inference_sampler = str(_dit_param("dit_inference_sampler", "ddim")).lower()
        if self.inference_sampler not in {"ddim", "ddpm"}:
            self.inference_sampler = "ddim"
        self.inference_eta = float(_dit_param("dit_inference_eta", 0.0))
        self.inference_eta = max(self.inference_eta, 0.0)

        self.enable_root_correction = bool((not self.no_trans) and _dit_param("dit_enable_root_correction", False))
        self.root_correction_contact_threshold = float(_dit_param("dit_root_correction_contact_threshold", 0.5))
        self.root_correction_contact_threshold = min(max(self.root_correction_contact_threshold, 0.0), 1.0)
        self.test_use_gt_warmup = bool(_dit_param("dit_test_use_gt_warmup", True))
        test_warmup_len = _dit_param("dit_test_warmup_len", None)
        self.test_warmup_len = None if test_warmup_len is None else int(test_warmup_len)
        if self.test_warmup_len is not None and self.test_warmup_len < 0:
            raise ValueError(f"dit_test_warmup_len must be >= 0 or None, got {self.test_warmup_len}")

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

    @staticmethod
    def _prepare_has_object_mask(
        has_object,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if has_object is None:
            return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

        if isinstance(has_object, torch.Tensor):
            mask = has_object.to(device=device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(has_object, device=device, dtype=torch.bool)

        if mask.dim() == 0:
            mask = mask.view(1)
        if mask.dim() == 1:
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size)
            if mask.shape[0] != batch_size:
                mask = mask[:1].expand(batch_size)
            mask = mask.view(batch_size, 1).expand(batch_size, seq_len)
        elif mask.dim() == 2:
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size, mask.shape[1])
            if mask.shape[0] != batch_size:
                mask = mask[:1].expand(batch_size, mask.shape[1])
            if mask.shape[1] == 1 and seq_len > 1:
                mask = mask.expand(batch_size, seq_len)
            elif mask.shape[1] != seq_len:
                mask = mask[:, :1].expand(batch_size, seq_len)
        else:
            mask = mask.reshape(batch_size, -1)
            if mask.shape[1] == 1:
                mask = mask.expand(batch_size, seq_len)
            else:
                mask = mask[:, :seq_len]
                if mask.shape[1] < seq_len:
                    pad = mask[:, -1:].expand(batch_size, seq_len - mask.shape[1])
                    mask = torch.cat([mask, pad], dim=1)

        return mask

    def _prepare_human_imu(self, human_imu: torch.Tensor) -> torch.Tensor:
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        out = human_imu
        if out.shape[2] > self.num_human_imus:
            out = out[:, :, : self.num_human_imus]
        elif out.shape[2] < self.num_human_imus:
            pad = torch.zeros(
                batch_size,
                seq_len,
                self.num_human_imus - out.shape[2],
                out.shape[3],
                device=device,
                dtype=dtype,
            )
            out = torch.cat([out, pad], dim=2)

        if out.shape[3] < 9:
            pad = torch.zeros(batch_size, seq_len, self.num_human_imus, 9 - out.shape[3], device=device, dtype=dtype)
            out = torch.cat([out, pad], dim=3)
        elif out.shape[3] > 9:
            out = out[..., :9]

        return out

    def _prepare_obj_imu(self, obj_imu, *, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not isinstance(obj_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, 9, device=device, dtype=dtype)

        out = obj_imu.to(device=device, dtype=dtype)
        if out.dim() == 4:
            out = out.reshape(batch_size, seq_len, -1)
        if out.dim() != 3:
            return torch.zeros(batch_size, seq_len, 9, device=device, dtype=dtype)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, -1, -1)
        if out.shape[0] != batch_size or out.shape[1] != seq_len:
            return torch.zeros(batch_size, seq_len, 9, device=device, dtype=dtype)

        if out.shape[-1] < 9:
            pad = torch.zeros(batch_size, seq_len, 9 - out.shape[-1], device=device, dtype=dtype)
            out = torch.cat([out, pad], dim=-1)
        elif out.shape[-1] > 9:
            out = out[..., :9]

        return out

    def _ensure_body_model_device(self, device: torch.device):
        if self.body_model is not None and self.body_model_device != device:
            self.body_model = self.body_model.to(device)
            self.body_model_device = device

    def _imu_global_rotation(self, human_imu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _resolve_obj_trans_init(
        self,
        data_dict: Dict,
        gt_targets: Optional[Dict],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        obj_trans_init = data_dict.get("obj_trans_init")
        if isinstance(obj_trans_init, torch.Tensor):
            obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
            if obj_trans_init.dim() == 3 and obj_trans_init.size(1) == 1:
                obj_trans_init = obj_trans_init[:, 0]
            if obj_trans_init.dim() == 1:
                obj_trans_init = obj_trans_init.unsqueeze(0)
            if obj_trans_init.shape[0] == 1 and batch_size > 1:
                obj_trans_init = obj_trans_init.expand(batch_size, -1)
            if obj_trans_init.shape[0] == batch_size and obj_trans_init.shape[-1] == 3:
                return obj_trans_init

        if isinstance(gt_targets, dict) and isinstance(gt_targets.get("obj_trans"), torch.Tensor):
            obj_trans = gt_targets["obj_trans"].to(device=device, dtype=dtype)
            if obj_trans.dim() == 2:
                obj_trans = obj_trans.unsqueeze(0)
            if obj_trans.shape[0] == 1 and batch_size > 1:
                obj_trans = obj_trans.expand(batch_size, -1, -1)
            if obj_trans.shape[0] == batch_size and obj_trans.shape[1] > 0:
                return obj_trans[:, 0]

        return torch.zeros(batch_size, 3, device=device, dtype=dtype)

    def _build_clean_window(self, data_dict: Dict, gt_targets: Dict) -> Tuple[torch.Tensor, Dict]:
        if not isinstance(gt_targets, dict):
            raise ValueError("gt_targets must be provided for full-window diffusion training")

        human_imu = self._prepare_human_imu(data_dict["human_imu"])
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        self._ensure_body_model_device(device)

        imu_rotm_global, imu_rot6d_global = self._imu_global_rotation(human_imu)
        human_acc = human_imu[..., :3].reshape(batch_size, seq_len, -1)

        obj_imu = self._prepare_obj_imu(
            data_dict.get("obj_imu"),
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
        obj_rot6d = obj_imu[..., 3:9]
        obj_acc = obj_imu[..., :3]

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
        rot_global_6d[:, :, _SENSOR_ROT_INDICES] = imu_rot6d_global
        rot_global = rotation_6d_to_matrix(rot_global_6d.reshape(-1, 6)).reshape(batch_size, seq_len, self.num_joints, 3, 3)

        rot_human_obj = torch.cat(
            [
                rot_global_6d.reshape(batch_size, seq_len, -1),
                obj_rot6d,
            ],
            dim=-1,
        )
        acc_human_obj = torch.cat([human_acc, obj_acc], dim=-1)

        trans = self._to_bt(
            gt_targets.get("trans"),
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=(3,),
            device=device,
            dtype=dtype,
            default=0.0,
        )
        gt_joints_local = self._compute_fk_joints_from_global(rot_global)

        # Hand global positions in x_frame: prefer GT joints, fallback to FK+trans, then zeros.
        p_hands = None
        position_global = gt_targets.get("position_global")
        if isinstance(position_global, torch.Tensor):
            position_global = position_global.to(device=device, dtype=dtype)
            if position_global.dim() == 3:
                position_global = position_global.unsqueeze(0)
            if position_global.shape[0] == 1 and batch_size > 1:
                position_global = position_global.expand(batch_size, -1, -1, -1)
            if (
                position_global.shape[0] == batch_size
                and position_global.shape[1] == seq_len
                and position_global.shape[-1] == 3
                and position_global.shape[2] > max(self.hand_joint_indices)
            ):
                p_hands = torch.stack(
                    [
                        position_global[:, :, self.hand_joint_indices[0]],
                        position_global[:, :, self.hand_joint_indices[1]],
                    ],
                    dim=2,
                )

        if not isinstance(p_hands, torch.Tensor):
            if isinstance(gt_joints_local, torch.Tensor) and gt_joints_local.shape[2] > max(self.hand_joint_indices):
                p_hands = torch.stack(
                    [
                        gt_joints_local[:, :, self.hand_joint_indices[0]],
                        gt_joints_local[:, :, self.hand_joint_indices[1]],
                    ],
                    dim=2,
                )
                p_hands = p_hands + trans.unsqueeze(2)
            else:
                p_hands = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)
        p_hands_flat = p_hands.reshape(batch_size, seq_len, -1)

        if self.no_trans:
            delta_p = torch.zeros(batch_size, seq_len, 0, device=device, dtype=dtype)
            p_y = torch.zeros(batch_size, seq_len, 0, device=device, dtype=dtype)
        else:
            delta_p = torch.zeros(batch_size, seq_len, 2, device=device, dtype=dtype)
            if seq_len > 1:
                delta_p[:, 1:] = trans[:, 1:, [0, 2]] - trans[:, :-1, [0, 2]]
            p_y = trans[:, :, 1:2]

        b = torch.stack(
            [
                self._to_bt(
                    gt_targets.get("lfoot_contact"),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rfoot_contact"),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)

        contact_prob = torch.stack(
            [
                self._to_bt(
                    gt_targets.get("lhand_contact"),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rhand_contact"),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)

        bone_dir = torch.cat(
            [
                self._to_bt(
                    gt_targets.get("lhand_obj_direction"),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(3,),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rhand_obj_direction"),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(3,),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
            ],
            dim=-1,
        )

        bone_len = torch.stack(
            [
                self._to_bt(
                    gt_targets.get("lhand_lb"),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rhand_lb"),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
            ],
            dim=-1,
        )

        obj_trans = self._to_bt(
            gt_targets.get("obj_trans"),
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=(3,),
            device=device,
            dtype=dtype,
            default=0.0,
        )
        p_obj = obj_trans

        has_object_mask = self._prepare_has_object_mask(
            data_dict.get("has_object") if isinstance(data_dict, dict) else None,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        # Keep object-related terms zero for no-object frames/samples.
        obj_mask = has_object_mask.to(device=device, dtype=dtype).unsqueeze(-1)
        contact_prob = contact_prob * obj_mask
        bone_dir = bone_dir * obj_mask
        bone_len = bone_len * obj_mask
        p_obj = p_obj * obj_mask

        x_parts = [rot_human_obj, acc_human_obj]
        if not self.no_trans:
            x_parts.extend([delta_p, p_y])
        x_parts.extend([p_hands_flat, b, contact_prob, bone_dir, bone_len, p_obj])
        x_clean = torch.cat(x_parts, dim=-1)

        trans_init = self._resolve_trans_init(
            data_dict,
            gt_targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        obj_trans_init = self._resolve_obj_trans_init(
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
            "obj_trans_init": obj_trans_init,
            "trans_gt": trans,
            "obj_trans_gt": obj_trans,
            "gt_joints_local": gt_joints_local,
            "human_imu": human_imu,
            "imu_rotm_global": imu_rotm_global,
            "has_object_mask": has_object_mask,
        }
        return x_clean, context

    def _build_observed_sequence(self, data_dict: Dict, gt_targets: Optional[Dict]) -> Tuple[torch.Tensor, Dict]:
        human_imu = self._prepare_human_imu(data_dict["human_imu"])
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        self._ensure_body_model_device(device)

        imu_rotm_global, imu_rot6d_global = self._imu_global_rotation(human_imu)
        human_acc = human_imu[..., :3].reshape(batch_size, seq_len, -1)

        obj_imu = self._prepare_obj_imu(
            data_dict.get("obj_imu"),
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
        obj_rot6d = obj_imu[..., 3:9]
        obj_acc = obj_imu[..., :3]

        observed = torch.zeros(batch_size, seq_len, self.target_dim, device=device, dtype=dtype)
        for local_idx, joint_idx in enumerate(_SENSOR_ROT_INDICES):
            observed[:, :, joint_idx * 6 : (joint_idx + 1) * 6] = imu_rot6d_global[:, :, local_idx]
        observed[:, :, self.human_rot_dim : self.rot_dim] = obj_rot6d
        observed[:, :, self.acc_slice] = torch.cat([human_acc, obj_acc], dim=-1)

        trans_init = self._resolve_trans_init(
            data_dict,
            gt_targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        obj_trans_init = self._resolve_obj_trans_init(
            data_dict,
            gt_targets,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        has_object_mask = self._prepare_has_object_mask(
            data_dict.get("has_object") if isinstance(data_dict, dict) else None,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )

        obj_trans_gt = None
        if isinstance(gt_targets, dict):
            obj_trans_gt = self._to_bt(
                gt_targets.get("obj_trans"),
                batch_size=batch_size,
                seq_len=seq_len,
                trailing_shape=(3,),
                device=device,
                dtype=dtype,
                default=0.0,
            )

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "trans_init": trans_init,
            "obj_trans_init": obj_trans_init,
            "trans_gt": None,
            "obj_trans_gt": obj_trans_gt,
            "gt_joints_local": None,
            "human_imu": human_imu,
            "imu_rotm_global": imu_rotm_global,
            "has_object_mask": has_object_mask,
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
        if self.no_trans or (not self.enable_root_correction) or self.delta_p_dim < 2:
            return curr_frame, curr_root_xz

        batch_size = curr_frame.shape[0]
        device = curr_frame.device
        dtype = curr_frame.dtype

        curr_root_pos = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        curr_root_pos[:, 0] = curr_root_xz[:, 0]
        curr_root_pos[:, 2] = curr_root_xz[:, 1]
        curr_root_pos[:, 1] = curr_frame[:, self.py_slice].squeeze(-1)

        rot_prev = rotation_6d_to_matrix(prev_frame[:, : self.human_rot_dim].reshape(-1, 6)).reshape(
            batch_size, self.num_joints, 3, 3
        )
        rot_curr = rotation_6d_to_matrix(curr_frame[:, : self.human_rot_dim].reshape(-1, 6)).reshape(
            batch_size, self.num_joints, 3, 3
        )
        rot_pair = torch.stack([rot_prev, rot_curr], dim=1)

        joints_local_pair = self._compute_fk_joints_from_global(rot_pair)
        if joints_local_pair is None:
            return curr_frame, curr_root_xz

        foot_prev_local = joints_local_pair[:, 0, self.foot_joint_indices, :]
        foot_curr_local = joints_local_pair[:, 1, self.foot_joint_indices, :]
        foot_prev_world = foot_prev_local + prev_root_pos.unsqueeze(1)
        foot_curr_world = foot_curr_local + curr_root_pos.unsqueeze(1)
        foot_disp_xz = foot_curr_world[..., [0, 2]] - foot_prev_world[..., [0, 2]]

        contact_prob = torch.sigmoid(curr_frame[:, self.b_slice])
        contact_mask = contact_prob >= self.root_correction_contact_threshold
        contact_count = contact_mask.sum(dim=1, keepdim=True)
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
        obj_trans_init = context["obj_trans_init"]
        has_object_mask = context.get("has_object_mask")

        rot_human_obj = x_pred[:, :, self.rot_slice]
        rot_human_6d = rot_human_obj[:, :, : self.human_rot_dim].reshape(batch_size, seq_len, self.num_joints, 6)
        obj_rot_6d = rot_human_obj[:, :, self.human_rot_dim : self.rot_dim]
        rot_human_mat = rotation_6d_to_matrix(rot_human_6d.reshape(-1, 6)).reshape(batch_size, seq_len, self.num_joints, 3, 3)

        acc_human_obj = x_pred[:, :, self.acc_slice]
        acc_human = acc_human_obj[:, :, : self.human_acc_dim].reshape(batch_size, seq_len, self.num_human_imus, 3)
        acc_obj = acc_human_obj[:, :, self.human_acc_dim :]

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

        b_pred = x_pred[:, :, self.b_slice]
        b_prob_pred = torch.sigmoid(b_pred)

        p_hands_pred = x_pred[:, :, self.hands_slice].reshape(batch_size, seq_len, 2, 3)
        contact_prob = x_pred[:, :, self.contact_slice]
        bone_dir = x_pred[:, :, self.bone_dir_slice].reshape(batch_size, seq_len, 2, 3)
        bone_len = x_pred[:, :, self.bone_len_slice].reshape(batch_size, seq_len, 2)
        p_obj = x_pred[:, :, self.p_obj_slice]

        if isinstance(has_object_mask, torch.Tensor):
            obj_mask = has_object_mask.to(device=device, dtype=dtype).unsqueeze(-1)
            contact_prob = contact_prob * obj_mask
            bone_dir = bone_dir * obj_mask.unsqueeze(-1)
            bone_len = bone_len * obj_mask
            p_obj = p_obj * obj_mask

        pred_obj_trans = p_obj
        if isinstance(has_object_mask, torch.Tensor):
            pred_obj_trans = pred_obj_trans * has_object_mask.to(device=device, dtype=dtype).unsqueeze(-1)

        pred_obj_vel = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        pred_obj_acc = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        if seq_len > 1:
            pred_obj_vel[:, 1:] = (pred_obj_trans[:, 1:] - pred_obj_trans[:, :-1]) * self.fps
            pred_obj_vel[:, 0] = pred_obj_vel[:, 1]
        if seq_len > 2:
            pred_obj_acc[:, 2:] = (
                pred_obj_trans[:, 2:] - 2.0 * pred_obj_trans[:, 1:-1] + pred_obj_trans[:, :-2]
            ) * (self.fps ** 2)

        root_vel_pred = self._compute_root_velocity_from_trans(root_trans_pred)
        root_rot = rot_human_mat[:, :, 0]
        root_vel_local_pred = torch.matmul(root_rot.transpose(-1, -2), root_vel_pred.unsqueeze(-1)).squeeze(-1)

        reduced_global = rot_human_mat[:, :, _REDUCED_INDICES]
        reduced_root = torch.matmul(root_rot.unsqueeze(2).transpose(-1, -2), reduced_global)
        p_pred = matrix_to_rotation_6d(reduced_root.reshape(-1, 3, 3)).reshape(
            batch_size, seq_len, len(_REDUCED_INDICES), 6
        )

        pred_joints_local = self._compute_fk_joints_from_global(rot_human_mat)
        if pred_joints_local is None:
            pred_joints_local = torch.zeros(batch_size, seq_len, self.num_joints, 3, device=device, dtype=dtype)

        pred_joints_global = pred_joints_local + root_trans_pred.unsqueeze(2)
        pred_hand_glb_pos_fk = torch.stack(
            [
                pred_joints_global[:, :, self.hand_joint_indices[0]],
                pred_joints_global[:, :, self.hand_joint_indices[1]],
            ],
            dim=2,
        )

        pred_imu_feat = torch.cat(
            [
                acc_human.reshape(batch_size, seq_len, -1),
                rot_human_6d[:, :, _SENSOR_ROT_INDICES].reshape(batch_size, seq_len, -1),
                acc_obj,
                obj_rot_6d,
            ],
            dim=-1,
        )

        out = {
            "x_pred": x_pred,
            "rot_human_obj_pred": rot_human_obj,
            "acc_human_obj_pred": acc_human_obj,
            "R_pred_6d": rot_human_6d,
            "R_pred_rotmat": rot_human_mat,
            "a_pred": acc_human,
            "delta_p_pred": delta_p_pred,
            "p_y_pred": p_y_pred,
            "b_pred": b_pred,
            "b_prob_pred": b_prob_pred,
            "p_hands_pred": p_hands_pred,
            "contact_prob_pred": contact_prob,
            "bone_dir_pred": bone_dir,
            "bone_len_pred": bone_len,
            "p_obj_pred": p_obj,
            "delta_p_obj_pred": p_obj,
            "pred_imu_feat": pred_imu_feat,
            "v_pred": torch.zeros(batch_size, seq_len, 0, device=device, dtype=dtype),
            "p_pred": p_pred,
            "pred_full_pose_rotmat": rot_human_mat,
            "pred_full_pose_6d": rot_human_6d,
            "pred_joints_local": pred_joints_local,
            "pred_joints_global": pred_joints_global,
            "pred_hand_glb_pos": pred_hand_glb_pos_fk,
            "pred_hand_glb_pos_fk": pred_hand_glb_pos_fk,
            "root_vel_local_pred": root_vel_local_pred,
            "root_vel_pred": root_vel_pred,
            "root_trans_pred": root_trans_pred,
            "pred_obj_trans": pred_obj_trans,
            "pred_obj_vel": pred_obj_vel,
            "pred_obj_vel_from_posdiff": pred_obj_vel,
            "pred_obj_acc_from_posdiff": pred_obj_acc,
            "pred_lhand_obj_direction": bone_dir[:, :, 0],
            "pred_rhand_obj_direction": bone_dir[:, :, 1],
            "pred_lhand_lb": bone_len[:, :, 0],
            "pred_rhand_lb": bone_len[:, :, 1],
            "obj_trans_init": obj_trans_init,
            "has_object": has_object_mask,
        }

        if context.get("gt_joints_local") is not None:
            out["gt_joints_local"] = context["gt_joints_local"]
        if context.get("trans_gt") is not None:
            out["gt_root_trans"] = context["trans_gt"]
        if context.get("obj_trans_gt") is not None:
            out["gt_obj_trans"] = context["obj_trans_gt"]

        return out

    def forward(self, data_dict: Dict, gt_targets: Optional[Dict] = None):
        """Training forward: full-window denoising only."""
        if gt_targets is None:
            raise ValueError("IMUHOIMixModule forward requires gt_targets for training")

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
        inference_type: str,
        sampler: str,
        eta: float,
        warmup_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

            if inference_type == "sample_inpaint":
                x_out = self.dit.sample_inpaint(
                    x_input=x_input,
                    inpaint_mask=inpaint_mask,
                    cond=None,
                    x_start=x_input,
                    steps=steps,
                    sampler=sampler,
                    eta=eta,
                )
            else:
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
        observed_seq, context = self._build_observed_sequence(data_dict, gt_targets)

        steps = self.inference_steps if sample_steps is None else int(sample_steps)
        if steps is None:
            steps = self.dit.timesteps
        steps = int(steps)

        sampler_name = self.inference_sampler if sampler is None else str(sampler).lower()
        eta_val = self.inference_eta if eta is None else float(eta)
        eta_val = max(eta_val, 0.0)
        if self.inference_type == "sample_inpaint" and sampler_name not in {"ddim", "ddpm"}:
            raise ValueError(f"sampler must be 'ddim' or 'ddpm', got {sampler_name}")

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
            max_window_warmup = max(self.window_size - 1, 0)
            cfg_warmup = max_window_warmup if self.test_warmup_len is None else self.test_warmup_len
            warmup_len = min(max(cfg_warmup, 0), max_window_warmup, max_warmup)
            if warmup_len > 0:
                warmup_seq = gt_clean_seq[:, :warmup_len]

        pred_seq = self._autoregressive_inference(
            observed_seq,
            trans_init=context["trans_init"],
            steps=steps,
            inference_type=self.inference_type,
            sampler=sampler_name,
            eta=eta_val,
            warmup_seq=warmup_seq,
        )
        outputs = self._decode_outputs(pred_seq, context)

        if self.inference_type == "sample_inpaint":
            aux_sampler = sampler_name
            aux_eta = eta_val
        else:
            aux_sampler = None
            aux_eta = None

        outputs["diffusion_aux"] = {
            "inference_steps": steps,
            "window_size": self.window_size,
            "warmup_len": int(warmup_len),
            "warmup_mode": "gt_prefix" if warmup_len > 0 else "none",
            "observed_dim_mask": self.observed_dim_mask,
            "unknown_dim_mask": self.unknown_dim_mask,
            "prediction_type": self.dit.prediction_type,
            "inference_mode": f"autoregressive_{self.inference_type}",
            "inference_type": self.inference_type,
            "sampler": aux_sampler,
            "eta": aux_eta,
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
        num_human_imus: int = 6,
    ):
        human_rot_dim = num_joints * 6
        rot_dim = human_rot_dim + 6
        acc_dim = num_human_imus * 3 + 3
        trans_dim = 0 if no_trans else 3
        target_dim = rot_dim + acc_dim + trans_dim + 6 + 2 + 2 + 6 + 2 + 3
        return {
            "x_pred": torch.zeros(batch_size, seq_len, target_dim, device=device),
            "rot_human_obj_pred": torch.zeros(batch_size, seq_len, rot_dim, device=device),
            "acc_human_obj_pred": torch.zeros(batch_size, seq_len, acc_dim, device=device),
            "R_pred_6d": torch.zeros(batch_size, seq_len, num_joints, 6, device=device),
            "R_pred_rotmat": torch.zeros(batch_size, seq_len, num_joints, 3, 3, device=device),
            "a_pred": torch.zeros(batch_size, seq_len, num_human_imus, 3, device=device),
            "delta_p_pred": torch.zeros(batch_size, seq_len, 0 if no_trans else 2, device=device),
            "p_y_pred": torch.zeros(batch_size, seq_len, 0 if no_trans else 1, device=device),
            "b_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "b_prob_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "p_hands_pred": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "contact_prob_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "bone_dir_pred": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "bone_len_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "p_obj_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "delta_p_obj_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_imu_feat": torch.zeros(batch_size, seq_len, num_human_imus * 9 + 9, device=device),
            "v_pred": torch.zeros(batch_size, seq_len, 0, device=device),
            "p_pred": torch.zeros(batch_size, seq_len, len(_REDUCED_INDICES), 6, device=device),
            "pred_full_pose_rotmat": torch.zeros(batch_size, seq_len, num_joints, 3, 3, device=device),
            "pred_full_pose_6d": torch.zeros(batch_size, seq_len, num_joints, 6, device=device),
            "pred_joints_local": torch.zeros(batch_size, seq_len, num_joints, 3, device=device),
            "pred_joints_global": torch.zeros(batch_size, seq_len, num_joints, 3, device=device),
            "pred_hand_glb_pos": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "pred_hand_glb_pos_fk": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "root_vel_local_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "root_vel_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "root_trans_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_trans": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_vel": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_vel_from_posdiff": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_acc_from_posdiff": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_lhand_obj_direction": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_rhand_obj_direction": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_lhand_lb": torch.zeros(batch_size, seq_len, device=device),
            "pred_rhand_lb": torch.zeros(batch_size, seq_len, device=device),
            "obj_trans_init": torch.zeros(batch_size, 3, device=device),
            "has_object": torch.ones(batch_size, seq_len, device=device, dtype=torch.bool),
            "diffusion_aux": {},
        }
