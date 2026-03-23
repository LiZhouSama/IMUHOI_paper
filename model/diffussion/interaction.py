"""
Stage-2 Interaction diffusion module.

Training:
- Full-window denoising only: x -> z_t -> x0_pred over all frames.
- No autoregressive rollout during training.

Inference:
- Autoregressive sliding-window inpainting.
- History frames are fixed.
- Current frame keeps observed dimensions fixed.
- Unknown dimensions are refined with configurable inpainting backend.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_rotation_6d

from .base import ConditionalDiT
from configs import FRAME_RATE, _SENSOR_NAMES


class InteractionModule(nn.Module):
    """Diffusion Transformer for interaction reconstruction."""

    def __init__(self, cfg):
        super().__init__()

        self.num_joints = int(getattr(cfg, "num_joints", 24))
        self.num_human_imus = int(getattr(cfg, "num_human_imus", len(_SENSOR_NAMES)))
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(getattr(cfg, "obj_imu_dim", self.imu_dim))
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))

        self.hand_joint_indices = (20, 21)

        # x_frame = [R_{human+obj}, a_{human+obj}, p_hands, contact_prob, v_bone, l_bone, delta_p_obj]
        self.human_rot_dim = self.num_joints * 6
        self.obj_rot_dim = 6
        self.rot_dim = self.human_rot_dim + self.obj_rot_dim

        self.human_acc_dim = self.num_human_imus * 3
        self.obj_acc_dim = 3
        self.acc_dim = self.human_acc_dim + self.obj_acc_dim

        self.hands_dim = 6
        self.contact_dim = 2
        self.bone_dir_dim = 6
        self.bone_len_dim = 2
        self.delta_p_obj_dim = 3

        start = 0
        self.rot_slice = slice(start, start + self.rot_dim)
        start += self.rot_dim
        self.acc_slice = slice(start, start + self.acc_dim)
        start += self.acc_dim
        self.hands_slice = slice(start, start + self.hands_dim)
        start += self.hands_dim
        self.contact_slice = slice(start, start + self.contact_dim)
        start += self.contact_dim
        self.bone_dir_slice = slice(start, start + self.bone_dir_dim)
        start += self.bone_dir_dim
        self.bone_len_slice = slice(start, start + self.bone_len_dim)
        start += self.bone_len_dim
        self.delta_p_obj_slice = slice(start, start + self.delta_p_obj_dim)
        start += self.delta_p_obj_dim
        self.target_dim = start

        # Observed dims during inference: R_{human+obj}, a_{human+obj}, p_hands.
        observed_dim_mask = torch.zeros(self.target_dim, dtype=torch.bool)
        observed_dim_mask[self.rot_slice] = True
        observed_dim_mask[self.acc_slice] = True
        observed_dim_mask[self.hands_slice] = True
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

    def _resolve_human_rot6d(
        self,
        hp_out: Optional[Dict],
        gt_targets: Optional[Dict],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rot = hp_out.get("pred_full_pose_6d") if isinstance(hp_out, dict) else None
        if isinstance(rot, torch.Tensor):
            rot = rot.to(device=device, dtype=dtype)
            if rot.dim() == 3 and rot.shape[-1] == self.human_rot_dim:
                rot = rot.view(batch_size, seq_len, self.num_joints, 6)
            if rot.dim() == 4:
                if rot.shape[0] == 1 and batch_size > 1:
                    rot = rot.expand(batch_size, -1, -1, -1)
                if rot.shape[0] == batch_size and rot.shape[1] == seq_len:
                    joints_now = rot.shape[2]
                    if joints_now > self.num_joints:
                        rot = rot[:, :, : self.num_joints]
                    elif joints_now < self.num_joints:
                        pad = torch.zeros(batch_size, seq_len, self.num_joints - joints_now, 6, device=device, dtype=dtype)
                        rot = torch.cat([rot, pad], dim=2)
                    return rot

        rot_global = gt_targets.get("rotation_global") if isinstance(gt_targets, dict) else None
        if isinstance(rot_global, torch.Tensor):
            rot_global = rot_global.to(device=device, dtype=dtype)
            if rot_global.dim() == 4:
                rot_global = rot_global.unsqueeze(0)
            if rot_global.shape[0] == 1 and batch_size > 1:
                rot_global = rot_global.expand(batch_size, -1, -1, -1, -1)
            if rot_global.shape[0] == batch_size and rot_global.shape[1] == seq_len and rot_global.shape[-2:] == (3, 3):
                joints_now = rot_global.shape[2]
                if joints_now > self.num_joints:
                    rot_global = rot_global[:, :, : self.num_joints]
                elif joints_now < self.num_joints:
                    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
                    pad = eye.expand(batch_size, seq_len, self.num_joints - joints_now, -1, -1)
                    rot_global = torch.cat([rot_global, pad], dim=2)
                rot6d = matrix_to_rotation_6d(rot_global.reshape(-1, 3, 3)).reshape(batch_size, seq_len, self.num_joints, 6)
                return rot6d

        return torch.zeros(batch_size, seq_len, self.num_joints, 6, device=device, dtype=dtype)

    def _resolve_hand_positions(
        self,
        hp_out: Optional[Dict],
        gt_targets: Optional[Dict],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        hand_pos = hp_out.get("pred_hand_glb_pos") if isinstance(hp_out, dict) else None
        if isinstance(hand_pos, torch.Tensor):
            hand_pos = hand_pos.to(device=device, dtype=dtype)
            if hand_pos.dim() == 3 and hand_pos.shape[-1] == 6:
                hand_pos = hand_pos.view(batch_size, seq_len, 2, 3)
            if hand_pos.dim() == 4:
                if hand_pos.shape[0] == 1 and batch_size > 1:
                    hand_pos = hand_pos.expand(batch_size, -1, -1, -1)
                if hand_pos.shape[0] == batch_size and hand_pos.shape[1] == seq_len and hand_pos.shape[2:] == (2, 3):
                    return hand_pos

        position_global = gt_targets.get("position_global") if isinstance(gt_targets, dict) else None
        if isinstance(position_global, torch.Tensor):
            position_global = position_global.to(device=device, dtype=dtype)
            if position_global.dim() == 3:
                position_global = position_global.unsqueeze(0)
            if position_global.shape[0] == 1 and batch_size > 1:
                position_global = position_global.expand(batch_size, -1, -1, -1)
            if position_global.shape[0] == batch_size and position_global.shape[1] == seq_len and position_global.shape[2] > 21:
                return torch.stack(
                    [position_global[:, :, self.hand_joint_indices[0]], position_global[:, :, self.hand_joint_indices[1]]],
                    dim=2,
                )

        return torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)

    def _build_features(
        self,
        data_dict: Dict,
        hp_out: Optional[Dict],
        gt_targets: Optional[Dict],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        human_imu = self._prepare_human_imu(data_dict["human_imu"]).to(device=device, dtype=dtype)
        obj_imu = self._prepare_obj_imu(data_dict.get("obj_imu"), batch_size=batch_size, seq_len=seq_len, device=device, dtype=dtype)

        human_rot6d = self._resolve_human_rot6d(
            hp_out,
            gt_targets,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
        obj_rot6d = obj_imu[..., 3:9]
        rot_human_obj = torch.cat([human_rot6d.reshape(batch_size, seq_len, -1), obj_rot6d], dim=-1)

        human_acc = human_imu[..., :3].reshape(batch_size, seq_len, -1)
        obj_acc = obj_imu[..., :3]
        acc_human_obj = torch.cat([human_acc, obj_acc], dim=-1)

        hands = self._resolve_hand_positions(
            hp_out,
            gt_targets,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )
        hands_flat = hands.reshape(batch_size, seq_len, -1)

        contact_prob = torch.stack(
            [
                self._to_bt(
                    gt_targets.get("lhand_contact") if isinstance(gt_targets, dict) else None,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rhand_contact") if isinstance(gt_targets, dict) else None,
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
                    gt_targets.get("lhand_obj_direction") if isinstance(gt_targets, dict) else None,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(3,),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rhand_obj_direction") if isinstance(gt_targets, dict) else None,
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
                    gt_targets.get("lhand_lb") if isinstance(gt_targets, dict) else None,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    trailing_shape=(),
                    device=device,
                    dtype=dtype,
                    default=0.0,
                ),
                self._to_bt(
                    gt_targets.get("rhand_lb") if isinstance(gt_targets, dict) else None,
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
            gt_targets.get("obj_trans") if isinstance(gt_targets, dict) else None,
            batch_size=batch_size,
            seq_len=seq_len,
            trailing_shape=(3,),
            device=device,
            dtype=dtype,
            default=0.0,
        )

        delta_p_obj = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        if seq_len > 1:
            delta_p_obj[:, 1:] = obj_trans[:, 1:] - obj_trans[:, :-1]

        return {
            "rot_human_obj": rot_human_obj,
            "acc_human_obj": acc_human_obj,
            "hands": hands,
            "hands_flat": hands_flat,
            "contact_prob": contact_prob,
            "bone_dir": bone_dir,
            "bone_len": bone_len,
            "obj_trans": obj_trans,
            "delta_p_obj": delta_p_obj,
        }

    def _build_clean_window(self, data_dict: Dict, hp_out: Optional[Dict], gt_targets: Dict) -> Tuple[torch.Tensor, Dict]:
        if not isinstance(gt_targets, dict):
            raise ValueError("gt_targets must be provided for full-window diffusion training")

        human_imu = data_dict["human_imu"]
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        feats = self._build_features(
            data_dict,
            hp_out,
            gt_targets,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )

        x_clean = torch.cat(
            [
                feats["rot_human_obj"],
                feats["acc_human_obj"],
                feats["hands_flat"],
                feats["contact_prob"],
                feats["bone_dir"],
                feats["bone_len"],
                feats["delta_p_obj"],
            ],
            dim=-1,
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

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "obj_trans_init": obj_trans_init,
            "obj_trans_gt": feats["obj_trans"],
            "has_object_mask": has_object_mask,
        }
        return x_clean, context

    def _build_observed_sequence(
        self,
        data_dict: Dict,
        hp_out: Optional[Dict],
        gt_targets: Optional[Dict],
    ) -> Tuple[torch.Tensor, Dict]:
        human_imu = data_dict["human_imu"]
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B,T,N,D], got {human_imu.shape}")

        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype

        feats = self._build_features(
            data_dict,
            hp_out,
            gt_targets,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            dtype=dtype,
        )

        observed = torch.zeros(batch_size, seq_len, self.target_dim, device=device, dtype=dtype)
        observed[:, :, self.rot_slice] = feats["rot_human_obj"]
        observed[:, :, self.acc_slice] = feats["acc_human_obj"]
        observed[:, :, self.hands_slice] = feats["hands_flat"]

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

        context = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "obj_trans_init": obj_trans_init,
            "obj_trans_gt": feats["obj_trans"] if isinstance(gt_targets, dict) else None,
            "has_object_mask": has_object_mask,
        }
        return observed, context

    def _decode_outputs(self, x_pred: torch.Tensor, context: Dict) -> Dict:
        batch_size = context["batch_size"]
        seq_len = context["seq_len"]
        device = context["device"]
        dtype = context["dtype"]
        obj_trans_init = context["obj_trans_init"]

        rot_human_obj = x_pred[:, :, self.rot_slice]
        acc_human_obj = x_pred[:, :, self.acc_slice]
        hands = x_pred[:, :, self.hands_slice].reshape(batch_size, seq_len, 2, 3)

        contact_prob = x_pred[:, :, self.contact_slice]
        bone_dir = x_pred[:, :, self.bone_dir_slice].reshape(batch_size, seq_len, 2, 3)
        bone_len = x_pred[:, :, self.bone_len_slice].reshape(batch_size, seq_len, 2)
        delta_p_obj = x_pred[:, :, self.delta_p_obj_slice]

        has_object_mask = context.get("has_object_mask")
        if isinstance(has_object_mask, torch.Tensor):
            mask = has_object_mask.to(device=device, dtype=dtype).unsqueeze(-1)
            contact_prob = contact_prob * mask
            bone_dir = bone_dir * mask.unsqueeze(-1)
            bone_len = bone_len * mask
            delta_p_obj = delta_p_obj * mask

        pred_obj_trans = torch.cumsum(delta_p_obj, dim=1) + obj_trans_init.unsqueeze(1)
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

        out = {
            "x_pred": x_pred,
            "rot_human_obj_pred": rot_human_obj,
            "acc_human_obj_pred": acc_human_obj,
            "p_hands_pred": hands,
            "contact_prob_pred": contact_prob,
            "bone_dir_pred": bone_dir,
            "bone_len_pred": bone_len,
            "delta_p_obj_pred": delta_p_obj,
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

        if context.get("obj_trans_gt") is not None:
            out["gt_obj_trans"] = context["obj_trans_gt"]

        return out

    def forward(self, data_dict: Dict, hp_out: Optional[Dict] = None, gt_targets: Optional[Dict] = None):
        """Training forward: full-window denoising only."""
        if gt_targets is None:
            raise ValueError("InteractionModule forward requires gt_targets for training")

        x_clean, context = self._build_clean_window(data_dict, hp_out, gt_targets)
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
        steps: int,
        inference_type: str,
        sampler: str,
        eta: float,
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
        if history.size(1) > 0 and unknown_dim > 0:
            last_unknown = history[:, -1, unknown_idx]

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

            history = torch.cat([history, current.unsqueeze(1)], dim=1)

        return history

    @torch.no_grad()
    def inference(
        self,
        data_dict: Dict,
        hp_out: Optional[Dict] = None,
        gt_targets: Optional[Dict] = None,
        sample_steps: Optional[int] = None,
        sampler: Optional[str] = None,
        eta: Optional[float] = None,
    ):
        """Inference with autoregressive sliding-window inpainting."""
        observed_seq, context = self._build_observed_sequence(data_dict, hp_out, gt_targets)

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
            and isinstance(gt_targets.get("obj_trans"), torch.Tensor)
        ):
            gt_clean_seq, _ = self._build_clean_window(data_dict, hp_out, gt_targets)
            max_warmup = max(seq_len, 1) - 1
            warmup_len = min(max(self.window_size - 1, 0), max_warmup)
            if warmup_len > 0:
                warmup_seq = gt_clean_seq[:, :warmup_len]

        pred_seq = self._autoregressive_inference(
            observed_seq,
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
        }
        return outputs

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device, num_joints: int = 24, num_human_imus: int = 6):
        human_rot_dim = num_joints * 6
        target_dim = human_rot_dim + 6 + num_human_imus * 3 + 3 + 6 + 2 + 6 + 2 + 3
        return {
            "x_pred": torch.zeros(batch_size, seq_len, target_dim, device=device),
            "rot_human_obj_pred": torch.zeros(batch_size, seq_len, human_rot_dim + 6, device=device),
            "acc_human_obj_pred": torch.zeros(batch_size, seq_len, num_human_imus * 3 + 3, device=device),
            "p_hands_pred": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "contact_prob_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "bone_dir_pred": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "bone_len_pred": torch.zeros(batch_size, seq_len, 2, device=device),
            "delta_p_obj_pred": torch.zeros(batch_size, seq_len, 3, device=device),
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
