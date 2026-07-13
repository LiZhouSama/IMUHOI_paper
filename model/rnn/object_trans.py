"""
ObjectTransModule: 预测物体位置
Stage 3 - 依赖Stage1和Stage2的输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from .base import RNN, RNNWithInit
from .online import (
    append_stream_data,
    infer_batch_seq,
    normalize_inference_mode,
    resolve_online_window,
    select_time_context,
    slice_time_dict,
    slice_time_value,
    take_latest_frame,
    TimeDictAccumulator,
)
from configs import FRAME_RATE, _REDUCED_POSE_NAMES, _SENSOR_NAMES


class ObjectTransModule(nn.Module):
    """
    物体位置预测模块
    基于手部位置、接触概率和物体IMU预测物体位置
    """
    ARCH_VERSION = 3
    _LEGACY_HEAD_PREFIXES = (
        "lhand_fk_head.",
        "rhand_fk_head.",
        "gating_head.",
        "mesh_prior_encoder.",
        "obs_encoder.",
    )

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(max(getattr(cfg, "obj_imu_dim", self.imu_dim), 9))
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        hidden_dim_multiplier = getattr(cfg, "hidden_dim_multiplier", 1)

        self.refine_pose_dim = len(_REDUCED_POSE_NAMES) * 6

        # The predictor keeps every non-redundant feature formerly consumed by
        # the two FK heads and the gate, but shares one temporal state across
        # both hands and the fusion decision.
        n_prediction_input = self.obj_imu_dim + 3 + 3 + 3 + 2 * (3 + self.imu_dim)
        predictor_hidden = int(getattr(cfg, "object_trans_hidden_dim", 128)) * hidden_dim_multiplier
        predictor_layers = int(getattr(cfg, "object_trans_num_layers", 2))
        predictor_dropout = float(getattr(cfg, "object_trans_dropout", 0.2))
        self.object_trans_head = RNNWithInit(
            n_input=n_prediction_input,
            n_output=12,  # left FK (4), right FK (4), four-way gate logits (4)
            n_hidden=predictor_hidden,
            n_init=11,  # left/right initial geometry + initial contact state
            n_rnn_layer=predictor_layers,
            bidirectional=False,
            dropout=predictor_dropout,
        )
        self.n_prediction_input = n_prediction_input
        self.feedback_feature_dim = 13
        feedback_hidden_dim = int(getattr(cfg, "object_trans_feedback_hidden_dim", 64))
        self.feedback_embed = nn.Sequential(
            nn.LayerNorm(self.feedback_feature_dim),
            nn.Linear(self.feedback_feature_dim, feedback_hidden_dim),
            nn.GELU(),
            nn.Linear(feedback_hidden_dim, n_prediction_input),
        )
        # Feedback starts as an exact no-op, then learns a stable residual
        # correction through the final projection.
        nn.init.zeros_(self.feedback_embed[-1].weight)
        nn.init.zeros_(self.feedback_embed[-1].bias)
        self.register_buffer(
            "object_trans_arch_version",
            torch.tensor(self.ARCH_VERSION, dtype=torch.int64),
            persistent=True,
        )

        n_refine_input = self.refine_pose_dim + 22
        self.refine_head = RNN(
            n_input=n_refine_input,
            n_output=self.refine_pose_dim + 3,
            n_hidden=128 * hidden_dim_multiplier,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        nn.init.zeros_(self.refine_head.linear2.weight)
        nn.init.zeros_(self.refine_head.linear2.bias)

    @staticmethod
    def _unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return x / norm

    @staticmethod
    def _softplus_positive(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-4

    def _state_feedback_enabled(self) -> bool:
        value = getattr(self.cfg, "object_trans_state_feedback", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "on", "yes", "fused", "enabled"}
        return bool(value)

    def online_mode(self) -> str:
        mode = str(getattr(self.cfg, "object_trans_online_mode", "window")).strip().lower()
        if mode not in {"window", "stateful"}:
            raise ValueError(f"object_trans_online_mode must be 'window' or 'stateful', got {mode!r}")
        return mode

    def validate_checkpoint_state_dict(self, state_dict) -> None:
        """Reject checkpoints from the removed three-head/prior architecture."""
        if not isinstance(state_dict, dict):
            raise ValueError("ObjectTrans checkpoint does not contain a state_dict.")
        normalized_state = {
            str(key[7:] if str(key).startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        keys = set(normalized_state)
        legacy_prefixes = [prefix for prefix in self._LEGACY_HEAD_PREFIXES if any(key.startswith(prefix) for key in keys)]
        if legacy_prefixes:
            raise ValueError(
                "ObjectTrans checkpoint uses the removed mesh/obs or three-head architecture "
                f"({', '.join(legacy_prefixes)}). Retrain ObjectTrans with architecture v{self.ARCH_VERSION}; "
                "partial loading would leave the shared predictor randomly initialized."
            )
        version = normalized_state.get("object_trans_arch_version")
        if not isinstance(version, torch.Tensor) or version.numel() != 1 or int(version.item()) != self.ARCH_VERSION:
            raise ValueError(
                f"ObjectTrans checkpoint is not marked as architecture v{self.ARCH_VERSION}; retrain it before loading."
            )
        if not any(key.startswith("object_trans_head.") for key in keys):
            raise ValueError(
                f"ObjectTrans checkpoint is missing the architecture-v{self.ARCH_VERSION} shared predictor."
            )
        expected_state = self.state_dict()
        mismatched_head_keys = [
            key
            for key, expected in expected_state.items()
            if key.startswith(("object_trans_head.", "feedback_embed."))
            and (key not in normalized_state or not isinstance(normalized_state[key], torch.Tensor) or normalized_state[key].shape != expected.shape)
        ]
        if mismatched_head_keys:
            raise ValueError(
                "ObjectTrans checkpoint predictor shape does not match this configuration; "
                f"use the checkpoint generated with the same ObjectTrans v{self.ARCH_VERSION} config."
            )

    def _build_prediction_inputs(
        self,
        obj_imu,
        obj_rot_delta,
        obj_vel,
        contact_prob,
        lhand_position,
        lhand_imu,
        rhand_position,
        rhand_imu,
    ):
        return torch.cat(
            (
                obj_imu,
                obj_rot_delta,
                obj_vel,
                contact_prob,
                lhand_position,
                lhand_imu,
                rhand_position,
                rhand_imu,
            ),
            dim=-1,
        )

    @staticmethod
    def _pad_or_trim_last_dim(x: torch.Tensor, out_dim: int) -> torch.Tensor:
        if x.shape[-1] == out_dim:
            return x
        if x.shape[-1] > out_dim:
            return x[..., :out_dim]
        pad_shape = (*x.shape[:-1], out_dim - x.shape[-1])
        return torch.cat((x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)), dim=-1)

    def _prepare_obj_imu(self, obj_imu: torch.Tensor, batch_size: int, seq_len: int, device, dtype) -> torch.Tensor:
        if not isinstance(obj_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        out = obj_imu.to(device=device, dtype=dtype)
        if out.dim() == 4:
            out = out.reshape(batch_size, seq_len, -1)
        if out.dim() == 2:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, -1, -1)
        if out.dim() != 3 or out.shape[:2] != (batch_size, seq_len):
            return torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        return self._pad_or_trim_last_dim(out, self.obj_imu_dim)

    def _prepare_human_imu(self, human_imu: torch.Tensor, batch_size: int, seq_len: int, device, dtype) -> torch.Tensor:
        if not isinstance(human_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, self.num_human_imus, self.imu_dim, device=device, dtype=dtype)
        out = human_imu.to(device=device, dtype=dtype)
        if out.dim() == 3 and out.shape[-1] == self.num_human_imus * self.imu_dim:
            out = out.view(batch_size, seq_len, self.num_human_imus, self.imu_dim)
        if out.dim() != 4 or out.shape[:2] != (batch_size, seq_len):
            return torch.zeros(batch_size, seq_len, self.num_human_imus, self.imu_dim, device=device, dtype=dtype)
        if out.shape[2] != self.num_human_imus or out.shape[3] != self.imu_dim:
            flat = out.reshape(batch_size, seq_len, -1)
            flat = self._pad_or_trim_last_dim(flat, self.num_human_imus * self.imu_dim)
            out = flat.view(batch_size, seq_len, self.num_human_imus, self.imu_dim)
        return out


    @staticmethod
    def _rotm_delta(obj_rotm: torch.Tensor, previous_obj_rotm: torch.Tensor = None) -> torch.Tensor:
        """Return causal axis-angle deltas, optionally continuing a streamed chunk."""
        batch_size, seq_len = obj_rotm.shape[:2]
        if seq_len == 0:
            return obj_rotm.new_zeros(batch_size, 0, 3)

        if isinstance(previous_obj_rotm, torch.Tensor):
            previous_obj_rotm = previous_obj_rotm.to(device=obj_rotm.device, dtype=obj_rotm.dtype)
            if previous_obj_rotm.shape != (batch_size, 3, 3):
                raise ValueError(
                    "previous_obj_rotm must have shape "
                    f"[{batch_size},3,3], got {tuple(previous_obj_rotm.shape)}"
                )
            previous = torch.cat((previous_obj_rotm.unsqueeze(1), obj_rotm[:, :-1]), dim=1)
            rel = torch.matmul(obj_rotm.transpose(-1, -2), previous)
            return matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(batch_size, seq_len, 3)

        if seq_len == 1:
            return obj_rotm.new_zeros(batch_size, 1, 3)
        rel = torch.matmul(obj_rotm[:, 1:].transpose(-1, -2), obj_rotm[:, :-1])
        aa = matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(batch_size, seq_len - 1, 3)
        return F.pad(aa, (0, 0, 1, 0))

    def _rot6d_delta(self, rot6d: torch.Tensor, previous_obj_rotm: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = rot6d.shape
        obj_rotm = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
        return self._rotm_delta(obj_rotm, previous_obj_rotm)

    def _compute_init_dir_len(self, hand_pos_0, obj_rotm_0, obj_pos_0):
        vec_world = obj_pos_0 - hand_pos_0
        lb0 = vec_world.norm(dim=-1, keepdim=True)
        unit_world = self._unit_vector(vec_world)
        obj_Rt = obj_rotm_0.transpose(-1, -2)
        oe0 = torch.bmm(obj_Rt, unit_world.unsqueeze(-1)).squeeze(-1)
        return oe0, lb0

    def _prepare_refine_pose(self, human_pose_input, batch_size, seq_len, device, dtype):
        if not isinstance(human_pose_input, torch.Tensor):
            return None
        pose = human_pose_input.to(device=device, dtype=dtype)
        if pose.dim() == 4:
            pose = pose.reshape(batch_size, seq_len, -1)
        if pose.dim() == 3 and pose.shape[:2] == (batch_size, seq_len) and pose.shape[-1] == self.refine_pose_dim:
            return pose
        return None

    def _prepare_refine_root(self, root_trans_input, batch_size, seq_len, device, dtype):
        if not isinstance(root_trans_input, torch.Tensor):
            return None
        root = root_trans_input.to(device=device, dtype=dtype)
        if root.dim() == 2:
            root = root.unsqueeze(1).expand(batch_size, seq_len, 3)
        if root.dim() == 3 and root.shape[:2] == (batch_size, seq_len) and root.shape[-1] == 3:
            return root
        return None

    def _compute_human_refinement(
        self,
        human_pose_input,
        root_trans_input,
        fused_pos,
        obj_vel_feature,
        weights,
        pred_hand_contact_prob,
        lhand_position,
        rhand_position,
        has_object_mask,
        enable_refine,
    ):
        if not enable_refine:
            return {}
        batch_size, seq_len = fused_pos.shape[:2]
        device = fused_pos.device
        dtype = fused_pos.dtype
        base_pose = self._prepare_refine_pose(human_pose_input, batch_size, seq_len, device, dtype)
        base_root = self._prepare_refine_root(root_trans_input, batch_size, seq_len, device, dtype)
        if base_pose is None or base_root is None:
            return {}

        refine_input = torch.cat(
            (
                fused_pos,
                obj_vel_feature,
                weights,
                pred_hand_contact_prob,
                lhand_position,
                rhand_position,
                base_root,
                base_pose,
            ),
            dim=-1,
        )
        residual = self.refine_head(refine_input)
        pose_delta = residual[:, :, : self.refine_pose_dim]
        root_trans_delta = residual[:, :, self.refine_pose_dim :]
        if has_object_mask is not None:
            mask = has_object_mask
            if mask.dim() > 1:
                mask = mask.view(batch_size)
            mask = mask.to(device=device, dtype=dtype).view(batch_size, 1, 1)
            pose_delta = pose_delta * mask
            root_trans_delta = root_trans_delta * mask

        return {
            "pose_delta": pose_delta,
            "root_trans_delta": root_trans_delta,
            "refined_pose": base_pose + pose_delta,
            "refined_root_trans": base_root + root_trans_delta,
        }

    @staticmethod
    def _prepare_known_sequence(value, batch_size, seq_len, feature_dim, device, dtype, mask=None):
        values = torch.zeros(batch_size, seq_len, feature_dim, device=device, dtype=dtype)
        valid = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        if not isinstance(value, torch.Tensor):
            return values, valid
        value = value.to(device=device, dtype=dtype)
        if value.dim() == 2:
            value = value.unsqueeze(1)
        if value.dim() != 3 or value.shape[-1] != feature_dim:
            return values, valid
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1, -1)
        if value.shape[0] != batch_size:
            return values, valid
        length = min(int(value.shape[1]), seq_len)
        if length <= 0:
            return values, valid
        values[:, :length] = value[:, :length]
        valid[:, :length] = True
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device=device, dtype=torch.bool)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size, -1)
            if mask.shape[0] == batch_size:
                valid[:, :length] &= mask[:, :length]
        return values, valid

    def _prepare_prediction_context(
        self,
        hand_positions,
        pred_hand_contact_prob,
        obj_trans_init,
        *,
        obj_imu=None,
        human_imu=None,
        obj_vel_input=None,
        contact_init=None,
        has_object_mask=None,
        human_pose_input=None,
        root_trans_input=None,
        prediction_state=None,
    ):
        if hand_positions is None:
            raise ValueError("hand_positions cannot be None")
        if hand_positions.dim() == 3:
            batch_size, seq_len, _ = hand_positions.shape
            hand_positions = hand_positions.view(batch_size, seq_len, 2, 3)
        elif hand_positions.dim() == 4:
            batch_size, seq_len = hand_positions.shape[:2]
        else:
            raise ValueError(f"Unexpected hand_positions shape {hand_positions.shape}")

        device = hand_positions.device
        dtype = hand_positions.dtype
        hand_positions = hand_positions.to(device=device, dtype=dtype)
        lhand_position = hand_positions[:, :, 0, :]
        rhand_position = hand_positions[:, :, 1, :]
        obj_imu = self._prepare_obj_imu(obj_imu, batch_size, seq_len, device, dtype)
        human_imu = self._prepare_human_imu(human_imu, batch_size, seq_len, device, dtype)
        if not isinstance(pred_hand_contact_prob, torch.Tensor):
            raise ValueError("pred_hand_contact_prob must be a tensor")
        pred_hand_contact_prob = pred_hand_contact_prob.to(device=device, dtype=dtype)
        if pred_hand_contact_prob.shape[:2] != (batch_size, seq_len) or pred_hand_contact_prob.shape[-1] < 3:
            raise ValueError(
                "pred_hand_contact_prob must have shape "
                f"[{batch_size},{seq_len},>=3], got {tuple(pred_hand_contact_prob.shape)}"
            )
        pred_hand_contact_prob = pred_hand_contact_prob[..., :3]

        if isinstance(obj_vel_input, torch.Tensor):
            obj_vel_input = obj_vel_input.to(device=device, dtype=dtype)
            if obj_vel_input.shape[:2] != (batch_size, seq_len) or obj_vel_input.shape[-1] != 3:
                obj_vel_input = None
        if obj_vel_input is None:
            obj_vel_input = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)

        obj_rot6d = obj_imu[:, :, 3:9]
        obj_rotm = rotation_6d_to_matrix(obj_rot6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
        previous_obj_rotm = prediction_state.get("prev_obj_rotm") if isinstance(prediction_state, dict) else None
        obj_rot_delta = self._rotm_delta(obj_rotm, previous_obj_rotm)
        l_idx = _SENSOR_NAMES.index("LeftForeArm")
        r_idx = _SENSOR_NAMES.index("RightForeArm")
        lhand_imu = human_imu[:, :, l_idx, :]
        rhand_imu = human_imu[:, :, r_idx, :]

        obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
        if obj_trans_init.dim() == 1:
            obj_trans_init = obj_trans_init.unsqueeze(0).expand(batch_size, -1)
        if obj_trans_init.shape != (batch_size, 3):
            raise ValueError(f"obj_trans_init must have shape [{batch_size},3], got {tuple(obj_trans_init.shape)}")
        l_oe0, l_lb0 = self._compute_init_dir_len(lhand_position[:, 0], obj_rotm[:, 0], obj_trans_init)
        r_oe0, r_lb0 = self._compute_init_dir_len(rhand_position[:, 0], obj_rotm[:, 0], obj_trans_init)
        if contact_init is None:
            contact_init_vec = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        else:
            contact_init_vec = contact_init.to(device=device, dtype=dtype)
            if contact_init_vec.dim() == 1:
                contact_init_vec = contact_init_vec.unsqueeze(0).expand(batch_size, -1)
            contact_init_vec = contact_init_vec[..., :3]
            if contact_init_vec.shape != (batch_size, 3):
                raise ValueError(f"contact_init must have shape [{batch_size},3], got {tuple(contact_init_vec.shape)}")

        prediction_input = self._build_prediction_inputs(
            obj_imu,
            obj_rot_delta,
            obj_vel_input,
            pred_hand_contact_prob,
            lhand_position,
            lhand_imu,
            rhand_position,
            rhand_imu,
        )
        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": device,
            "dtype": dtype,
            "hand_positions": hand_positions,
            "lhand_position": lhand_position,
            "rhand_position": rhand_position,
            "obj_imu": obj_imu,
            "obj_rotm": obj_rotm,
            "obj_vel_input": obj_vel_input,
            "pred_hand_contact_prob": pred_hand_contact_prob,
            "prediction_input": prediction_input,
            "prediction_init": torch.cat((l_oe0, l_lb0, r_oe0, r_lb0, contact_init_vec), dim=-1),
            "obj_trans_init": obj_trans_init,
            "l_oe0": l_oe0,
            "r_oe0": r_oe0,
            "l_lb0": l_lb0,
            "r_lb0": r_lb0,
            "has_object_mask": has_object_mask,
            "human_pose_input": human_pose_input,
            "root_trans_input": root_trans_input,
        }

    def _init_prediction_state(self, context):
        batch_size = context["batch_size"]
        obj_trans_init = context["obj_trans_init"]
        return {
            "rnn_state": self.object_trans_head.initial_state(context["prediction_init"]),
            "prev_obj_trans": obj_trans_init,
            "prev_prev_obj_trans": torch.zeros_like(obj_trans_init),
            "prev_obj_delta": torch.zeros_like(obj_trans_init),
            "prev_gating_weights": obj_trans_init.new_tensor((0.0, 0.0, 0.0, 1.0)).expand(batch_size, -1).clone(),
            "prev_obj_rotm": None,
            "frames_seen": 0,
        }

    @staticmethod
    def _object_frame_vector(obj_rotm, vector):
        return torch.bmm(obj_rotm.transpose(-1, -2), vector.unsqueeze(-1)).squeeze(-1)

    def _feedback_features(self, obj_rotm, lhand_position, rhand_position, state):
        prev_pos = state["prev_obj_trans"]
        return torch.cat(
            (
                self._object_frame_vector(obj_rotm, prev_pos - lhand_position),
                self._object_frame_vector(obj_rotm, prev_pos - rhand_position),
                self._object_frame_vector(obj_rotm, state["prev_obj_delta"]),
                state["prev_gating_weights"],
            ),
            dim=-1,
        )

    def _predict_step(
        self,
        context,
        time_index,
        state,
        known_obj_trans=None,
        known_mask=None,
        known_weights=None,
        known_weights_mask=None,
        prediction_out=None,
        next_rnn_state=None,
    ):
        obj_rotm = context["obj_rotm"][:, time_index]
        lhand_position = context["lhand_position"][:, time_index]
        rhand_position = context["rhand_position"][:, time_index]
        step_input = context["prediction_input"][:, time_index]
        if self._state_feedback_enabled():
            step_input = step_input + self.feedback_embed(
                self._feedback_features(obj_rotm, lhand_position, rhand_position, state)
            )

        if prediction_out is None:
            prediction_out, rnn_state = self.object_trans_head.step(step_input, state["rnn_state"])
        else:
            rnn_state = state["rnn_state"] if next_rnn_state is None else next_rnn_state
        l_fk_out, r_fk_out, gate_logits = prediction_out.split((4, 4, 4), dim=-1)
        l_dir = self._unit_vector(l_fk_out[:, :3])
        r_dir = self._unit_vector(r_fk_out[:, :3])
        l_len = self._softplus_positive(l_fk_out[:, 3])
        r_len = self._softplus_positive(r_fk_out[:, 3])
        l_dir_world = torch.bmm(obj_rotm, l_dir.unsqueeze(-1)).squeeze(-1)
        r_dir_world = torch.bmm(obj_rotm, r_dir.unsqueeze(-1)).squeeze(-1)
        l_pos_fk = lhand_position + l_dir_world * l_len.unsqueeze(-1)
        r_pos_fk = rhand_position + r_dir_world * r_len.unsqueeze(-1)

        weights = F.softmax(gate_logits, dim=-1)
        previous_pos = state["prev_obj_trans"]
        imu_integrated_pos = previous_pos + context["obj_vel_input"][:, time_index] / FRAME_RATE
        static_pos = previous_pos
        fused_model_pos = (
            weights[:, 0:1] * l_pos_fk
            + weights[:, 1:2] * r_pos_fk
            + weights[:, 2:3] * imu_integrated_pos
            + weights[:, 3:4] * static_pos
        )
        if known_mask is None:
            known_mask = torch.zeros(context["batch_size"], device=previous_pos.device, dtype=torch.bool)
        if isinstance(known_obj_trans, torch.Tensor):
            known_obj_trans = known_obj_trans.to(device=previous_pos.device, dtype=previous_pos.dtype)
            if not self.training:
                known_obj_trans = known_obj_trans.detach()
            fused_pos = torch.where(known_mask.unsqueeze(-1), known_obj_trans, fused_model_pos)
        else:
            fused_pos = fused_model_pos

        effective_weights = weights
        if isinstance(known_weights, torch.Tensor) and isinstance(known_weights_mask, torch.Tensor):
            known_weights = known_weights.to(device=weights.device, dtype=weights.dtype)
            feedback_mask = known_mask & known_weights_mask.to(device=weights.device, dtype=torch.bool)
            effective_weights = torch.where(feedback_mask.unsqueeze(-1), known_weights, weights)
        frames_seen = int(state.get("frames_seen", 0))
        velocity = (fused_pos - previous_pos) * FRAME_RATE if frames_seen > 0 else torch.zeros_like(fused_pos)
        acceleration = (
            (fused_pos - 2.0 * previous_pos + state["prev_prev_obj_trans"]) * (FRAME_RATE ** 2)
            if frames_seen > 1
            else torch.zeros_like(fused_pos)
        )
        next_state = {
            "rnn_state": rnn_state,
            "prev_obj_trans": fused_pos,
            "prev_prev_obj_trans": previous_pos,
            "prev_obj_delta": fused_pos - previous_pos,
            "prev_gating_weights": effective_weights,
            "prev_obj_rotm": obj_rotm,
            "frames_seen": frames_seen + 1,
        }
        return {
            "pred_obj_trans": fused_pos,
            # Published window prefixes are authoritative for all derived
            # quantities (not only position feedback).  Keep raw logits and
            # softmax separately for supervision/diagnostics.
            "gating_weights": effective_weights,
            "gating_weights_raw": weights,
            "pred_obj_vel_from_posdiff": velocity,
            "pred_obj_acc_from_posdiff": acceleration,
            "pred_lhand_obj_direction": l_dir,
            "pred_rhand_obj_direction": r_dir,
            "pred_lhand_lb": l_len,
            "pred_rhand_lb": r_len,
            "pred_lhand_obj_trans": l_pos_fk,
            "pred_rhand_obj_trans": r_pos_fk,
            "pred_imu_obj_trans": imu_integrated_pos,
            "pred_static_obj_trans": static_pos,
            "lhand_fk_out": l_fk_out,
            "rhand_fk_out": r_fk_out,
            "gate_logits": gate_logits,
        }, next_state

    def _scan_predictor(
        self,
        context,
        prediction_state=None,
        known_obj_trans_prefix=None,
        known_obj_trans_prefix_mask=None,
        known_gating_weights_prefix=None,
    ):
        state = prediction_state if isinstance(prediction_state, dict) else self._init_prediction_state(context)
        use_vectorized_fast_path = (
            not self._state_feedback_enabled()
            and not isinstance(prediction_state, dict)
            and not isinstance(known_obj_trans_prefix, torch.Tensor)
            and not isinstance(known_gating_weights_prefix, torch.Tensor)
        )
        if use_vectorized_fast_path:
            # The no-feedback training baseline has no recurrent dependence in
            # its inputs.  Run the LSTM as one sequence (the original shared
            # head behavior) and retain only the causal fusion scan below.
            prediction_sequence, final_rnn_state = self.object_trans_head.forward_with_state(
                context["prediction_input"],
                x_init=context["prediction_init"],
            )
            collected = {}
            for time_index in range(context["seq_len"]):
                frame_output, state = self._predict_step(
                    context,
                    time_index,
                    state,
                    prediction_out=prediction_sequence[:, time_index],
                )
                for key, value in frame_output.items():
                    collected.setdefault(key, []).append(value)
            state = dict(state)
            state["rnn_state"] = final_rnn_state
            return {key: torch.stack(values, dim=1) for key, values in collected.items()}, state

        known_pos, known_mask = self._prepare_known_sequence(
            known_obj_trans_prefix,
            context["batch_size"],
            context["seq_len"],
            3,
            context["device"],
            context["dtype"],
            known_obj_trans_prefix_mask,
        )
        known_weights, known_weights_mask = self._prepare_known_sequence(
            known_gating_weights_prefix,
            context["batch_size"],
            context["seq_len"],
            4,
            context["device"],
            context["dtype"],
            None,
        )
        collected = {}
        for time_index in range(context["seq_len"]):
            frame_output, state = self._predict_step(
                context,
                time_index,
                state,
                known_obj_trans=known_pos[:, time_index],
                known_mask=known_mask[:, time_index],
                known_weights=known_weights[:, time_index],
                known_weights_mask=known_weights_mask[:, time_index],
            )
            for key, value in frame_output.items():
                collected.setdefault(key, []).append(value)
        return {key: torch.stack(values, dim=1) for key, values in collected.items()}, state

    @staticmethod
    def _resolve_object_mask(has_object_mask, batch_size, device, dtype):
        if has_object_mask is None:
            return None
        if not isinstance(has_object_mask, torch.Tensor):
            has_object_mask = torch.as_tensor(has_object_mask, device=device)
        if has_object_mask.dim() == 0:
            has_object_mask = has_object_mask.view(1).expand(batch_size)
        elif has_object_mask.dim() > 1:
            has_object_mask = has_object_mask.reshape(batch_size, -1)[:, 0]
        elif has_object_mask.shape[0] == 1 and batch_size > 1:
            has_object_mask = has_object_mask.expand(batch_size)
        return has_object_mask.to(device=device, dtype=dtype).view(batch_size)

    def _build_results(self, context, prediction, enable_refine=True):
        batch_size = context["batch_size"]
        sample_mask = self._resolve_object_mask(
            context["has_object_mask"], batch_size, context["device"], context["dtype"]
        )
        results = {
            **prediction,
            "obj_vel_input": context["obj_vel_input"],
            "obj_vel_corrected": context["obj_vel_input"],
            "init_lhand_oe_ho": context["l_oe0"],
            "init_rhand_oe_ho": context["r_oe0"],
            "init_lhand_lb": context["l_lb0"].squeeze(-1),
            "init_rhand_lb": context["r_lb0"].squeeze(-1),
            "gating_contact_prob": context["pred_hand_contact_prob"],
            "gating_smoothing_applied": False,
        }
        if sample_mask is not None:
            mask = sample_mask.view(batch_size, 1, 1)
            init_mask = sample_mask.view(batch_size, 1)
            for key in (
                "pred_obj_trans",
                "gating_weights",
                "gating_weights_raw",
                "pred_obj_vel_from_posdiff",
                "pred_obj_acc_from_posdiff",
                "pred_lhand_obj_direction",
                "pred_rhand_obj_direction",
                "pred_lhand_obj_trans",
                "pred_rhand_obj_trans",
                "pred_imu_obj_trans",
                "pred_static_obj_trans",
            ):
                results[key] = results[key] * mask
            results["pred_lhand_lb"] = results["pred_lhand_lb"] * init_mask
            results["pred_rhand_lb"] = results["pred_rhand_lb"] * init_mask
            results["init_lhand_oe_ho"] = results["init_lhand_oe_ho"] * init_mask
            results["init_rhand_oe_ho"] = results["init_rhand_oe_ho"] * init_mask
            results["init_lhand_lb"] = results["init_lhand_lb"] * sample_mask
            results["init_rhand_lb"] = results["init_rhand_lb"] * sample_mask

        if enable_refine:
            results.update(
                self._compute_human_refinement(
                    context["human_pose_input"],
                    context["root_trans_input"],
                    results["pred_obj_trans"],
                    context["obj_vel_input"],
                    results["gating_weights"],
                    context["pred_hand_contact_prob"],
                    context["lhand_position"],
                    context["rhand_position"],
                    context["has_object_mask"],
                    enable_refine,
                )
            )
        return results

    def compute_human_refinement_from_prediction(
        self,
        prediction,
        hand_positions,
        pred_hand_contact_prob,
        *,
        obj_vel_input,
        human_pose_input=None,
        root_trans_input=None,
        has_object_mask=None,
        enable_refine=True,
    ):
        if not enable_refine:
            return {}
        if hand_positions.dim() == 3:
            hand_positions = hand_positions.view(*hand_positions.shape[:2], 2, 3)
        return self._compute_human_refinement(
            human_pose_input,
            root_trans_input,
            prediction["pred_obj_trans"],
            obj_vel_input,
            prediction["gating_weights"],
            pred_hand_contact_prob,
            hand_positions[:, :, 0],
            hand_positions[:, :, 1],
            has_object_mask,
            enable_refine,
        )

    def forward(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        obj_imu: torch.Tensor = None,
        human_imu: torch.Tensor = None,
        obj_vel_input: torch.Tensor = None,
        contact_init: torch.Tensor = None,
        has_object_mask: torch.Tensor = None,
        human_pose_input: torch.Tensor = None,
        root_trans_input: torch.Tensor = None,
        enable_refine: bool = True,
        known_obj_trans_prefix: torch.Tensor = None,
        known_obj_trans_prefix_mask: torch.Tensor = None,
        known_gating_weights_prefix: torch.Tensor = None,
        prediction_state: dict = None,
        return_prediction_state: bool = False,
        compute_refine: bool = True,
    ):
        """Causal OT scan; optionally continue from an explicit prediction state."""
        context = self._prepare_prediction_context(
            hand_positions,
            pred_hand_contact_prob,
            obj_trans_init,
            obj_imu=obj_imu,
            human_imu=human_imu,
            obj_vel_input=obj_vel_input,
            contact_init=contact_init,
            has_object_mask=has_object_mask,
            human_pose_input=human_pose_input,
            root_trans_input=root_trans_input,
            prediction_state=prediction_state,
        )
        prediction, next_state = self._scan_predictor(
            context,
            prediction_state=prediction_state,
            known_obj_trans_prefix=known_obj_trans_prefix,
            known_obj_trans_prefix_mask=known_obj_trans_prefix_mask,
            known_gating_weights_prefix=known_gating_weights_prefix,
        )
        results = self._build_results(context, prediction, enable_refine=enable_refine and compute_refine)
        if return_prediction_state:
            return results, next_state
        return results

    @staticmethod
    def _slice_optional_time(value, start: int, end: int, batch_size: int, seq_len: int):
        return slice_time_value(value, start, end, batch_size, seq_len)

    def _inference_online_sequence(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        online_window: int,
        **kwargs,
    ):
        if hand_positions.dim() == 3:
            batch_size, seq_len, _ = hand_positions.shape
        elif hand_positions.dim() == 4:
            batch_size, seq_len = hand_positions.shape[:2]
        else:
            raise ValueError(f"Unexpected hand_positions shape {hand_positions.shape}")

        if seq_len <= online_window:
            return self.forward(
                hand_positions,
                pred_hand_contact_prob,
                obj_trans_init,
                **kwargs,
            )

        def _slice_kwargs(start: int, end: int):
            return {
                key: self._slice_optional_time(value, start, end, batch_size, seq_len)
                for key, value in kwargs.items()
                if key not in {"known_obj_trans_prefix", "known_gating_weights_prefix"}
            }

        warmup_len = int(online_window)
        warmup_out = self.forward(
            hand_positions[:, :warmup_len],
            pred_hand_contact_prob[:, :warmup_len],
            obj_trans_init,
            **_slice_kwargs(0, warmup_len),
        )
        history = warmup_out
        history_acc = TimeDictAccumulator(history, seq_len)
        history = history_acc.current()

        for end in range(warmup_len + 1, seq_len + 1):
            start = end - warmup_len
            prefix_context = select_time_context(history, start, end - 1)
            prefix = prefix_context.get("pred_obj_trans")
            prefix_weights = prefix_context.get("gating_weights")
            if isinstance(prefix, torch.Tensor) and prefix.shape[1] > 0:
                step_obj_trans_init = prefix[:, 0]
            else:
                step_obj_trans_init = obj_trans_init
            window_out = self.forward(
                hand_positions[:, start:end],
                pred_hand_contact_prob[:, start:end],
                step_obj_trans_init,
                known_obj_trans_prefix=prefix,
                known_gating_weights_prefix=prefix_weights,
                **_slice_kwargs(start, end),
            )
            latest = take_latest_frame(window_out, batch_size, end - start)
            history = history_acc.append(latest)

        return history_acc.current()

    def inference(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        inference_mode: str = "offline",
        online_window: int = None,
        online_state: dict = None,
        return_online_state: bool = False,
        **kwargs,
    ):
        mode = normalize_inference_mode(inference_mode)
        if mode == "offline":
            output = self.forward(
                hand_positions,
                pred_hand_contact_prob,
                obj_trans_init,
                **kwargs,
            )
            if return_online_state:
                return output, online_state or {}
            return output

        if self.online_mode() == "stateful":
            prediction_state = online_state.get("prediction_state") if isinstance(online_state, dict) else None
            output, prediction_state = self.forward(
                hand_positions,
                pred_hand_contact_prob,
                obj_trans_init,
                prediction_state=prediction_state,
                return_prediction_state=True,
                **kwargs,
            )
            state = {"prediction_state": prediction_state}
            if return_online_state:
                return output, state
            return output

        window = resolve_online_window(self.cfg, online_window)
        stream_dict = {
            "hand_positions": hand_positions,
            "pred_hand_contact_prob": pred_hand_contact_prob,
            **{key: value for key, value in kwargs.items() if key != "known_obj_trans_prefix"},
        }
        if isinstance(online_state, dict) and isinstance(online_state.get("data_dict"), dict):
            run_data, previous_len = append_stream_data(online_state["data_dict"], stream_dict, sequence_key="hand_positions")
            hand_positions = run_data["hand_positions"]
            pred_hand_contact_prob = run_data["pred_hand_contact_prob"]
            for key in stream_dict.keys():
                if key not in {"hand_positions", "pred_hand_contact_prob"}:
                    kwargs[key] = run_data.get(key)
        else:
            previous_len = 0
            run_data = stream_dict

        output = self._inference_online_sequence(
            hand_positions,
            pred_hand_contact_prob,
            obj_trans_init,
            window,
            **kwargs,
        )
        state = {"data_dict": run_data, "outputs": output}
        if return_online_state:
            if previous_len > 0:
                batch_size, seq_len = infer_batch_seq(run_data, key="hand_positions")
                output = slice_time_dict(output, previous_len, seq_len, batch_size, seq_len)
            return output, state
        return output

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        """返回空输出"""
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_dir = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_scalar = torch.zeros(batch_size, seq_len, device=device)
        zeros_weights = torch.zeros(batch_size, seq_len, 4, device=device)
        zeros_fk = torch.zeros(batch_size, seq_len, 4, device=device)
        return {
            "pred_obj_trans": zeros_pos,
            "gating_weights": zeros_weights,
            "gating_weights_raw": zeros_weights,
            "pred_obj_vel_from_posdiff": zeros_pos,
            "pred_obj_acc_from_posdiff": zeros_pos,
            "obj_vel_input": zeros_pos,
            "obj_vel_corrected": zeros_pos,
            "pred_lhand_obj_direction": zeros_dir,
            "pred_rhand_obj_direction": zeros_dir,
            "pred_lhand_lb": zeros_scalar,
            "pred_rhand_lb": zeros_scalar,
            "pred_lhand_obj_trans": zeros_pos,
            "pred_rhand_obj_trans": zeros_pos,
            "pred_imu_obj_trans": zeros_pos,
            "pred_static_obj_trans": zeros_pos,
            "lhand_fk_out": zeros_fk,
            "rhand_fk_out": zeros_fk,
            "gate_logits": zeros_weights,
            "gating_contact_prob": torch.zeros(batch_size, seq_len, 3, device=device),
            "init_lhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_rhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_lhand_lb": torch.zeros(batch_size, device=device),
            "init_rhand_lb": torch.zeros(batch_size, device=device),
            "gating_smoothing_applied": False,
        }
