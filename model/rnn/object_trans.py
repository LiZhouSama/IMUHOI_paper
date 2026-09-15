"""
ObjectTransModule: 预测物体位置
Stage 3 - 依赖Stage1和Stage2的输出
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from .base import RNN, RNNWithInit
from .bimanual_grasp import BimanualGrasp
from .grasp_state import GraspLatent
from .online import (
    append_stream_data,
    infer_batch_seq,
    normalize_inference_mode,
    NON_TEMPORAL_KEYS,
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
    # v6: the per-window l_b prior quantiles are broadcast to every frame of
    # the predictor input (init-state-only injection was too weak to constrain
    # per-frame lengths), and hands without usable statistics fall back to the
    # init geometry instead of rejecting the prior tensor.
    ARCH_VERSION = 6
    LENGTH_EPS = 1e-4
    LB_QUANTILE_DIM = 6  # two hands x (q10, q50, q90)
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

        self.bigrasp_enabled = bool(getattr(cfg, "object_trans_bigrasp", False))
        if self.bigrasp_enabled:
            if str(getattr(cfg, "object_trans_arch", "current")) != "current":
                raise ValueError("bigrasp is supported only by object_trans_arch=current")
            self.bigrasp = BimanualGrasp(hidden_dim=int(getattr(cfg, "bigrasp_hidden_dim", 96)))
            self.SEGMENT_STATE_OUTPUT_KEYS = ("bigrasp_vector", "bigrasp_active")

        self.refine_pose_dim = len(_REDUCED_POSE_NAMES) * 6

        # The predictor keeps every non-redundant feature formerly consumed by
        # the two FK heads and the gate, but shares one temporal state across
        # both hands and the fusion decision.  The l_b prior quantiles are a
        # per-window constant broadcast to every frame so the LSTM cannot wash
        # them out the way an init-state-only injection would.
        n_prediction_input = (
            self.obj_imu_dim + 3 + 3 + 3 + 2 * (3 + self.imu_dim) + self.LB_QUANTILE_DIM
        )
        predictor_hidden = int(getattr(cfg, "object_trans_hidden_dim", 128)) * hidden_dim_multiplier
        predictor_layers = int(getattr(cfg, "object_trans_num_layers", 2))
        predictor_dropout = float(getattr(cfg, "object_trans_dropout", 0.2))
        self.object_trans_head = RNNWithInit(
            n_input=n_prediction_input,
            n_output=12,  # left FK (4), right FK (4), four-way gate logits (4)
            n_hidden=predictor_hidden,
            # left/right initial geometry + initial contact state + the frozen
            # diffusion output [left/right, q10/q50/q90].
            n_init=17,
            n_rnn_layer=predictor_layers,
            bidirectional=False,
            dropout=predictor_dropout,
        )
        self.object_trans_length_init = float(
            max(getattr(cfg, "object_trans_length_init", 0.4), self.LENGTH_EPS * 2.0)
        )
        length_bias = math.log(
            math.expm1(self.object_trans_length_init - self.LENGTH_EPS)
        )
        with torch.no_grad():
            self.object_trans_head.linear2.bias[3] = length_bias
            self.object_trans_head.linear2.bias[7] = length_bias
        self.n_prediction_input = n_prediction_input
        # Event-anchored grasp latent (object_trans_grasp_latent): the bone is
        # a segment state anchored at contact onsets and propagated by learned
        # increments, so the published object enters the grasp geometry only
        # at onsets (model/rnn/grasp_state.py).
        self.grasp_latent_enabled = bool(getattr(cfg, "object_trans_grasp_latent", False))
        if self.grasp_latent_enabled:
            if not self._state_feedback_enabled():
                raise ValueError("object_trans_grasp_latent requires object_trans_state_feedback")
            if self.bigrasp_enabled or bool(getattr(cfg, "object_trans_state_trust", False)):
                raise ValueError("object_trans_grasp_latent cannot be combined with bigrasp or state trust")
            self.grasp_latent = GraspLatent(
                n_prediction_input=n_prediction_input,
                hidden_dim=predictor_hidden,
                n_layers=predictor_layers,
                dropout=predictor_dropout,
                lb_quantile_dim=self.LB_QUANTILE_DIM,
                length_init=self.object_trans_length_init,
                on_threshold=float(getattr(cfg, "object_trans_grasp_on_threshold", 0.5)),
                off_threshold=float(getattr(cfg, "object_trans_grasp_off_threshold", 0.25)),
                age_norm=float(getattr(cfg, "object_trans_grasp_age_norm", 60.0)),
                increment_scale=float(getattr(cfg, "object_trans_grasp_increment_scale", 0.1)),
                fusion_hidden=int(getattr(cfg, "object_trans_grasp_fusion_hidden", 32)),
                fusion_bias=float(getattr(cfg, "object_trans_grasp_fusion_bias", 3.0)),
                modes=bool(getattr(cfg, "object_trans_grasp_modes", False)),
                increment_mode=str(getattr(cfg, "object_trans_grasp_increment_mode", "integrate")),
                residual_bound=float(getattr(cfg, "object_trans_grasp_residual_bound", 0.10)),
            )
            self.SEGMENT_STATE_OUTPUT_KEYS = tuple(getattr(self, "SEGMENT_STATE_OUTPUT_KEYS", ())) + GraspLatent.STATE_KEYS
        self.feedback_feature_dim = GraspLatent.FEEDBACK_FEATURE_DIM if self.grasp_latent_enabled else 13
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
        # State trust (object_trans_state_trust): the published object feeds
        # back into this predictor (init geometry + feedback features), so a
        # drifted state corrupts the grasp offset on long streams.  Instead of
        # blindly distrusting the state, the module (a) runs a state-free
        # prior FK head, (b) measures the consistency between that prior and
        # the current state (innovation, bimanual agreement, gap/rest ages,
        # IMU rest flag), (c) predicts a supervised per-frame confidence c_t
        # from those signals, and (d) scales the state-derived inputs by c_t.
        self.state_trust_enabled = bool(getattr(cfg, "object_trans_state_trust", False))
        if self.state_trust_enabled:
            if not self._state_feedback_enabled():
                raise ValueError("object_trans_state_trust requires object_trans_state_feedback")
            self.prior_head = RNNWithInit(
                n_input=n_prediction_input,
                n_output=8,  # left dir(3)+len(1), right dir(3)+len(1); no state, no init geometry
                n_hidden=predictor_hidden,
                n_init=3 + self.LB_QUANTILE_DIM,  # initial contact state + l_b prior quantiles only
                n_rnn_layer=predictor_layers,
                bidirectional=False,
                dropout=predictor_dropout,
            )
            with torch.no_grad():
                self.prior_head.linear2.bias[3] = length_bias
                self.prior_head.linear2.bias[7] = length_bias
            self.consistency_feature_dim = 15
            consistency_hidden = int(getattr(cfg, "object_trans_consistency_hidden_dim", 64))
            self.consistency_embed = nn.Sequential(
                nn.LayerNorm(self.consistency_feature_dim),
                nn.Linear(self.consistency_feature_dim, consistency_hidden),
                nn.GELU(),
                nn.Linear(consistency_hidden, n_prediction_input),
            )
            nn.init.zeros_(self.consistency_embed[-1].weight)
            nn.init.zeros_(self.consistency_embed[-1].bias)
            self.confidence_head = nn.Sequential(
                nn.LayerNorm(self.consistency_feature_dim),
                nn.Linear(self.consistency_feature_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            # Ages are counted inside the current window (online windows restart
            # every frame), so normalise by the window length.
            self.state_trust_age_norm = float(getattr(cfg, "object_trans_state_trust_age_norm", 60.0))
            self.state_trust_contact_threshold = float(
                getattr(cfg, "object_trans_state_trust_contact_threshold", 0.5)
            )
            # Diagnostics: override the learned confidence with a constant at
            # inference (None = use the head).  1.0 disables the gating and the
            # init-geometry mixing while keeping the prior head and the
            # consistency features as predictor inputs.
            force = getattr(cfg, "object_trans_state_trust_force_confidence", None)
            self.state_trust_force_confidence = None if force is None else float(force)
        self.register_buffer(
            "object_trans_arch_version",
            torch.tensor(self.ARCH_VERSION, dtype=torch.int64),
            persistent=True,
        )
        # Object-static hard gate.  On the OMOMO test split 38% of the object
        # error growth happens on frames where the object is not moving (false
        # contacts dragging it, and the IMU branch leaking through a soft static
        # weight).  The object IMU detects "static" with 0.995 precision from a
        # short causal window, so those frames can be pinned to the previous
        # position instead of being softly fused.
        #   none  : original soft fusion
        #   imu   : rotation change and acceleration over the last k frames
        #   pmove : VC object-motion probability below a threshold
        #   both  : either signal
        self.static_gate_mode = str(
            getattr(cfg, "object_trans_static_gate", "none") or "none"
        ).strip().lower()
        if self.static_gate_mode not in {"none", "imu", "pmove", "both"}:
            raise ValueError(
                "object_trans_static_gate must be none, imu, pmove or both; "
                f"got {self.static_gate_mode!r}"
            )
        self.static_gate_pmove_threshold = float(
            getattr(cfg, "object_trans_static_gate_pmove_threshold", 0.5)
        )
        self.static_gate_imu_window = max(
            int(getattr(cfg, "object_trans_static_gate_imu_window", 5)), 1
        )
        self.static_gate_imu_angle_deg = float(
            getattr(cfg, "object_trans_static_gate_imu_angle_deg", 0.6)
        )
        self.static_gate_imu_acc = float(
            getattr(cfg, "object_trans_static_gate_imu_acc", 0.35)
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
        return F.softplus(x) + ObjectTransModule.LENGTH_EPS

    def _compute_static_gate(self, obj_imu, obj_rotm, pred_hand_contact_prob):
        """Causal per-frame object-static flag [B, T] (bool), or None if disabled."""
        mode = getattr(self, "static_gate_mode", "none")
        if mode == "none":
            return None
        batch_size, seq_len = obj_rotm.shape[:2]
        device = obj_rotm.device
        flags = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        if mode in {"imu", "both"}:
            flags = flags | self._imu_rest_flags(obj_imu, obj_rotm)
        if mode in {"pmove", "both"}:
            flags = flags | (
                pred_hand_contact_prob[..., 2] < self.static_gate_pmove_threshold
            )
        return flags

    def _imu_rest_flags(self, obj_imu, obj_rotm):
        """Causal object-at-rest flag [B, T] from the object IMU alone.

        Rotation change over the last k frames below ``static_gate_imu_angle_deg``
        and max |acc| over the same causal window below ``static_gate_imu_acc``
        (0.995+ precision on OMOMO/BEHAVE, ~0.55 recall).
        """
        batch_size, seq_len = obj_rotm.shape[:2]
        device = obj_rotm.device
        k = self.static_gate_imu_window
        index = torch.arange(seq_len, device=device)
        ref = (index - k).clamp_min(0)
        rel = obj_rotm.transpose(-1, -2) @ obj_rotm[:, ref]
        cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
        angle_deg = torch.rad2deg(torch.acos(cos))
        acc_norm = obj_imu[..., :3].norm(dim=-1)
        # max |acc| over the causal window [t-k, t]
        padded = torch.cat((acc_norm[:, :1].expand(-1, k), acc_norm), dim=1)
        acc_max = padded.unfold(1, k + 1, 1).max(dim=-1).values
        return (angle_deg < self.static_gate_imu_angle_deg) & (acc_max < self.static_gate_imu_acc)

    def _apply_static_gate(self, weights, context, time_index):
        """Pin the fusion to the static branch on gated frames."""
        gate = context.get("static_gate")
        if gate is None:
            return weights
        static_onehot = weights.new_tensor((0.0, 0.0, 0.0, 1.0)).expand_as(weights)
        return torch.where(gate[:, time_index].unsqueeze(-1), static_onehot, weights)

    def _state_feedback_enabled(self) -> bool:
        value = getattr(self.cfg, "object_trans_state_feedback", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "on", "yes", "fused", "enabled"}
        return bool(value)

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
        lb_prior_feature,
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
                lb_prior_feature,
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

        # The refine head is an output-side correction (see the online paths,
        # which never feed refined poses back upstream).  Detach its dynamic
        # inputs so the heavily weighted human-refinement losses cannot pull
        # the shared OT predictor and the upstream VC away from their own
        # objectives; refine gradients stay inside refine_head.
        refine_input = torch.cat(
            (
                fused_pos.detach(),
                obj_vel_feature.detach(),
                weights.detach(),
                pred_hand_contact_prob.detach(),
                lhand_position.detach(),
                rhand_position.detach(),
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
    def _prepare_feedback_tensor(
        value,
        batch_size,
        seq_len,
        feature_dim,
        device,
        dtype,
        name,
    ):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)
        if value.dim() == 2:
            value = value.unsqueeze(0)
        if value.dim() != 3 or value.shape[-1] != feature_dim:
            raise ValueError(
                f"{name} must have shape [B,T,{feature_dim}], got {tuple(value.shape)}"
            )
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1, -1)
        if value.shape != (batch_size, seq_len, feature_dim):
            raise ValueError(
                f"{name} must have shape [{batch_size},{seq_len},{feature_dim}], "
                f"got {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _prepare_feedback_mask(value, batch_size, seq_len, device):
        if value is None:
            return None
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device)
        value = value.to(device=device, dtype=torch.bool)
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if value.dim() != 2:
            raise ValueError(
                f"feedback_injection_mask must have shape [B,T], got {tuple(value.shape)}"
            )
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        if value.shape != (batch_size, seq_len):
            raise ValueError(
                "feedback_injection_mask must have shape "
                f"[{batch_size},{seq_len}], got {tuple(value.shape)}"
            )
        return value

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
        lb_prior_quantiles=None,
        feedback_control=None,
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
        obj_rot_delta = self._rotm_delta(obj_rotm, None)
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
        raw_lb_prior_quantiles = lb_prior_quantiles
        lb_prior_quantiles = self._prepare_lb_prior_quantiles(
            lb_prior_quantiles,
            l_lb0,
            r_lb0,
            batch_size,
            device,
            dtype,
        )
        if contact_init is None:
            contact_init_vec = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        else:
            contact_init_vec = contact_init.to(device=device, dtype=dtype)
            if contact_init_vec.dim() == 1:
                contact_init_vec = contact_init_vec.unsqueeze(0).expand(batch_size, -1)
            contact_init_vec = contact_init_vec[..., :3]
            if contact_init_vec.shape != (batch_size, 3):
                raise ValueError(f"contact_init must have shape [{batch_size},3], got {tuple(contact_init_vec.shape)}")

        lb_prior_feature = (
            lb_prior_quantiles.flatten(1).unsqueeze(1).expand(batch_size, seq_len, self.LB_QUANTILE_DIM)
        )
        prediction_input = self._build_prediction_inputs(
            obj_imu,
            obj_rot_delta,
            obj_vel_input,
            pred_hand_contact_prob,
            lhand_position,
            lhand_imu,
            rhand_position,
            rhand_imu,
            lb_prior_feature,
        )

        static_gate = self._compute_static_gate(obj_imu, obj_rotm, pred_hand_contact_prob)

        bigrasp = None
        if self.bigrasp_enabled:
            if not isinstance(root_trans_input, torch.Tensor) or root_trans_input.shape != (batch_size, seq_len, 3):
                raise ValueError("bigrasp requires root_trans_input with shape [B,T,3]")
            bigrasp = self.bigrasp(hand_positions, root_trans_input.to(hand_positions), human_imu, obj_imu, pred_hand_contact_prob)

        l_oe0_init, l_lb0_init, r_oe0_init, r_lb0_init = l_oe0, l_lb0, r_oe0, r_lb0
        state_trust = None
        if self.state_trust_enabled:
            prior_lb = self._state_free_lb_prior(raw_lb_prior_quantiles, batch_size, device, dtype)
            prior_input = torch.cat(
                (
                    prediction_input[..., : -self.LB_QUANTILE_DIM],
                    prior_lb.flatten(1).unsqueeze(1).expand(batch_size, seq_len, self.LB_QUANTILE_DIM),
                ),
                dim=-1,
            )
            state_trust = self._prepare_state_trust(
                prior_input,
                prior_lb,
                obj_imu,
                obj_rotm,
                lhand_position,
                rhand_position,
                pred_hand_contact_prob,
                contact_init_vec,
            )
            # Confidence at the window start decides how much of the init
            # geometry comes from the (possibly drifted) object state and how
            # much from the state-free prior.
            z0 = self._consistency_features_at(state_trust, 0, obj_trans_init, obj_rotm[:, 0])
            c0 = torch.sigmoid(self.confidence_head(z0))  # [B,1]
            if self.state_trust_force_confidence is not None:
                c0 = torch.full_like(c0, self.state_trust_force_confidence)
            l_oe0_init = c0 * l_oe0 + (1.0 - c0) * state_trust["prior_l_dir"][:, 0]
            r_oe0_init = c0 * r_oe0 + (1.0 - c0) * state_trust["prior_r_dir"][:, 0]
            l_lb0_init = c0 * l_lb0 + (1.0 - c0) * state_trust["prior_l_len"][:, 0].unsqueeze(-1)
            r_lb0_init = c0 * r_lb0 + (1.0 - c0) * state_trust["prior_r_len"][:, 0].unsqueeze(-1)
            state_trust["init_confidence"] = c0.squeeze(-1)

        grasp = None
        if self.grasp_latent_enabled:
            prior_lb = self._state_free_lb_prior(raw_lb_prior_quantiles, batch_size, device, dtype)
            prior_input = torch.cat(
                (
                    prediction_input[..., : -self.LB_QUANTILE_DIM],
                    prior_lb.flatten(1).unsqueeze(1).expand(batch_size, seq_len, self.LB_QUANTILE_DIM),
                ),
                dim=-1,
            )
            prior_init = torch.cat((contact_init_vec, prior_lb.flatten(1)), dim=-1)
            prior_bone, prior_pos = self.grasp_latent.prior(prior_input, prior_init, obj_rotm, hand_positions)
            rest_flag = self._imu_rest_flags(obj_imu, obj_rotm)
            rest_age, gap_age = self.grasp_latent.causal_ages(rest_flag, pred_hand_contact_prob)
            # Window-init geometry (object frame) is the observed bone at t=0.
            observed0 = torch.stack((l_oe0 * l_lb0, r_oe0 * r_lb0), dim=1)
            latent0, active0, age0, alpha0 = self.grasp_latent.initial_state(
                observed0, prior_bone[:, 0], rest_flag[:, 0], rest_age[:, 0], gap_age[:, 0], pred_hand_contact_prob[:, 0]
            )
            grasp = {
                "prior_bone": prior_bone,
                "prior_pos": prior_pos,
                "rest_flag": rest_flag,
                "rest_age": rest_age,
                "gap_age": gap_age,
                "latent0": latent0,
                "active0": active0,
                "age0": age0,
                "alpha0": alpha0,
            }
            # The predictor's initial state reads the anchored latent instead
            # of the raw init geometry (identical while alpha ~ 1).
            directions0, lengths0 = GraspLatent.split_latent(latent0)
            l_oe0_init, r_oe0_init = directions0[:, 0], directions0[:, 1]
            l_lb0_init, r_lb0_init = lengths0[:, 0:1], lengths0[:, 1:2]

        feedback_mode = "normal"
        feedback_prev_obj_trans = None
        feedback_prev_prev_obj_trans = None
        feedback_prev_obj_delta = None
        feedback_perturbation = None
        feedback_injection_mask = None
        if isinstance(feedback_control, dict):
            feedback_mode = str(feedback_control.get("mode", "normal")).strip().lower()
            if feedback_mode not in {"normal", "gt_prev", "perturb"}:
                raise ValueError(
                    "feedback_control.mode must be one of normal, gt_prev, perturb"
                )
            feedback_prev_obj_trans = self._prepare_feedback_tensor(
                feedback_control.get("prev_obj_trans"),
                batch_size,
                seq_len,
                3,
                device,
                dtype,
                "feedback_prev_obj_trans",
            )
            feedback_prev_prev_obj_trans = self._prepare_feedback_tensor(
                feedback_control.get("prev_prev_obj_trans"),
                batch_size,
                seq_len,
                3,
                device,
                dtype,
                "feedback_prev_prev_obj_trans",
            )
            feedback_prev_obj_delta = self._prepare_feedback_tensor(
                feedback_control.get("prev_obj_delta"),
                batch_size,
                seq_len,
                3,
                device,
                dtype,
                "feedback_prev_obj_delta",
            )
            feedback_perturbation = self._prepare_feedback_tensor(
                feedback_control.get("perturbation"),
                batch_size,
                seq_len,
                3,
                device,
                dtype,
                "feedback_perturbation",
            )
            feedback_injection_mask = self._prepare_feedback_mask(
                feedback_control.get("injection_mask"),
                batch_size,
                seq_len,
                device,
            )
            if feedback_mode == "gt_prev" and any(
                value is None
                for value in (
                    feedback_prev_obj_trans,
                    feedback_prev_prev_obj_trans,
                    feedback_prev_obj_delta,
                )
            ):
                raise ValueError(
                    "gt_prev feedback requires previous position, previous-previous "
                    "position, and previous delta tensors"
                )
            if feedback_mode == "perturb" and (
                feedback_perturbation is None or feedback_injection_mask is None
            ):
                raise ValueError(
                    "perturb feedback requires perturbation and injection_mask tensors"
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
            "static_gate": static_gate,
            "prediction_init": torch.cat(
                (
                    l_oe0_init,
                    l_lb0_init,
                    r_oe0_init,
                    r_lb0_init,
                    contact_init_vec,
                    lb_prior_quantiles.flatten(1),
                ),
                dim=-1,
            ),
            "state_trust": state_trust,
            "grasp": grasp,
            "contact_init": contact_init_vec,
            "bigrasp": bigrasp,
            "obj_trans_init": obj_trans_init,
            "l_oe0": l_oe0,
            "r_oe0": r_oe0,
            "l_lb0": l_lb0,
            "r_lb0": r_lb0,
            "lb_prior_quantiles": lb_prior_quantiles,
            "has_object_mask": has_object_mask,
            "human_pose_input": human_pose_input,
            "root_trans_input": root_trans_input,
            "feedback_mode": feedback_mode,
            "feedback_prev_obj_trans": feedback_prev_obj_trans,
            "feedback_prev_prev_obj_trans": feedback_prev_prev_obj_trans,
            "feedback_prev_obj_delta": feedback_prev_obj_delta,
            "feedback_perturbation": feedback_perturbation,
            "feedback_injection_mask": feedback_injection_mask,
        }

    def _prepare_lb_prior_quantiles(
        self,
        value,
        l_lb0,
        r_lb0,
        batch_size,
        device,
        dtype,
    ):
        centers = torch.cat((l_lb0, r_lb0), dim=-1).clamp_min(0.0)
        neutral = centers.unsqueeze(-1).expand(-1, -1, 3)
        if value is None:
            # Neutral fallback for direct module calls, cold starts, and
            # prior ablations.
            return neutral
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)
        if value.shape != (batch_size, 2, 3):
            raise ValueError(
                f"lb_prior_quantiles must have shape [{batch_size},2,3], got {tuple(value.shape)}"
            )
        # Hands without usable statistics are marked non-finite (NaN) by the
        # online/GT estimators; fall back to the init geometry per hand.
        finite = torch.isfinite(value).all(dim=-1, keepdim=True)
        value = torch.where(finite, value.clamp_min(0.0), neutral)
        # The estimators are ordered by construction; cummax also makes
        # externally supplied priors safe without a hard length interval.
        return torch.cummax(value, dim=-1).values

    @staticmethod
    @torch.no_grad()
    def compute_online_lb_quantiles(
        pred_obj_trans,
        hand_positions,
        contact_prob,
        *,
        contact_threshold: float = 0.5,
        min_contact_frames: int = 10,
        quantiles=(0.1, 0.5, 0.9),
        max_history_frames: int = 2048,
        tail_frames: int = 0,
    ):
        """Estimate per-hand l_b quantiles from the causal prediction history.

        ``tail_frames`` > 0 restricts the statistics to the most recent frames
        (a trailing window), so that on long streams the prior cannot carry
        errors accumulated over the whole history; ``max_history_frames`` only
        bounds the cost by uniform striding and keeps early segments.

        Frames whose predicted contact probability exceeds ``contact_threshold``
        contribute ``||pred_obj_trans - hand||``.  Hands with fewer than
        ``min_contact_frames`` usable frames are returned as NaN, which
        ``_prepare_lb_prior_quantiles`` maps back to the init geometry.
        Returns ``None`` when no hand has usable statistics.
        """
        if not isinstance(pred_obj_trans, torch.Tensor) or not isinstance(contact_prob, torch.Tensor):
            return None
        if not isinstance(hand_positions, torch.Tensor):
            return None
        if hand_positions.dim() == 3 and hand_positions.shape[-1] == 6:
            hand_positions = hand_positions.reshape(*hand_positions.shape[:2], 2, 3)
        if (
            pred_obj_trans.dim() != 3
            or hand_positions.dim() != 4
            or contact_prob.dim() != 3
            or pred_obj_trans.shape[-1] != 3
            or hand_positions.shape[2:] != (2, 3)
            or contact_prob.shape[-1] < 2
            or pred_obj_trans.shape[:2] != hand_positions.shape[:2]
            or pred_obj_trans.shape[:2] != contact_prob.shape[:2]
            or pred_obj_trans.shape[1] == 0
        ):
            return None

        lengths = (pred_obj_trans.unsqueeze(2) - hand_positions).norm(dim=-1)  # [B,T,2]
        contact = contact_prob[..., :2] > float(contact_threshold)
        seq_len = lengths.shape[1]
        if int(tail_frames) > 0 and seq_len > int(tail_frames):
            lengths = lengths[:, -int(tail_frames):]
            contact = contact[:, -int(tail_frames):]
            seq_len = lengths.shape[1]
        if seq_len > int(max_history_frames):
            # Uniform stride keeps early contact segments represented while
            # bounding the per-step quantile cost on long streams.
            stride = -(-seq_len // int(max_history_frames))
            lengths = lengths[:, ::stride]
            contact = contact[:, ::stride]

        batch_size = lengths.shape[0]
        q_values = torch.as_tensor(tuple(float(q) for q in quantiles), device=lengths.device, dtype=lengths.dtype)
        output = torch.full(
            (batch_size, 2, q_values.numel()),
            float("nan"),
            device=lengths.device,
            dtype=lengths.dtype,
        )
        min_count = int(max(1, min_contact_frames))
        for batch_idx in range(batch_size):
            for hand_idx in range(2):
                values = lengths[batch_idx, :, hand_idx][contact[batch_idx, :, hand_idx]]
                values = values[torch.isfinite(values)]
                if values.numel() >= min_count:
                    output[batch_idx, hand_idx] = torch.quantile(values, q_values)
        if not bool(torch.isfinite(output).any().item()):
            return None
        return output

    def _state_free_lb_prior(self, value, batch_size, device, dtype):
        """l_b prior for the state-free head: statistics when given, else a constant.

        ``_prepare_lb_prior_quantiles`` falls back to the window's init geometry
        (hand -> *estimated* object length), which would leak the object state
        into the prior head; here missing/non-finite hands fall back to the
        length-init constant instead.
        """
        neutral = torch.full((batch_size, 2, 3), self.object_trans_length_init, device=device, dtype=dtype)
        if value is None:
            return neutral
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype)
        if value.shape != (batch_size, 2, 3):
            return neutral
        finite = torch.isfinite(value).all(dim=-1, keepdim=True)
        value = torch.where(finite, value.clamp_min(0.0), neutral)
        return torch.cummax(value, dim=-1).values

    def _prepare_state_trust(
        self,
        prediction_input,
        lb_prior_quantiles,
        obj_imu,
        obj_rotm,
        lhand_position,
        rhand_position,
        pred_hand_contact_prob,
        contact_init_vec,
    ):
        """State-free prior FK, IMU rest flag and causal ages for the window."""
        batch_size, seq_len = obj_rotm.shape[:2]
        prior_init = torch.cat((contact_init_vec, lb_prior_quantiles.flatten(1)), dim=-1)
        prior_out, _ = self.prior_head.forward_with_state(prediction_input, x_init=prior_init)
        prior_out = prior_out.to(obj_rotm.dtype)
        l_dir = self._unit_vector(prior_out[..., 0:3])
        r_dir = self._unit_vector(prior_out[..., 4:7])
        l_len = self._softplus_positive(prior_out[..., 3])
        r_len = self._softplus_positive(prior_out[..., 7])
        l_pos = lhand_position + torch.einsum("btij,btj->bti", obj_rotm, l_dir) * l_len.unsqueeze(-1)
        r_pos = rhand_position + torch.einsum("btij,btj->bti", obj_rotm, r_dir) * r_len.unsqueeze(-1)
        rest = self._imu_rest_flags(obj_imu, obj_rotm)  # [B,T] bool
        contact_any = pred_hand_contact_prob[..., :2].max(dim=-1).values > self.state_trust_contact_threshold
        rest_age = torch.zeros(batch_size, seq_len, device=obj_rotm.device, dtype=obj_rotm.dtype)
        gap_age = torch.zeros_like(rest_age)
        rest_run = torch.zeros(batch_size, device=obj_rotm.device, dtype=obj_rotm.dtype)
        gap_run = torch.zeros_like(rest_run)
        for t in range(seq_len):
            rest_run = torch.where(rest[:, t], rest_run + 1.0, torch.zeros_like(rest_run))
            gap_run = torch.where(contact_any[:, t], torch.zeros_like(gap_run), gap_run + 1.0)
            rest_age[:, t] = rest_run
            gap_age[:, t] = gap_run
        return {
            "prior_l_pos": l_pos,
            "prior_r_pos": r_pos,
            "prior_l_dir": l_dir,
            "prior_r_dir": r_dir,
            "prior_l_len": l_len,
            "prior_r_len": r_len,
            "rest_flag": rest,
            "rest_age": rest_age,
            "gap_age": gap_age,
        }

    def _consistency_features_at(self, state_trust, time_index, prev_pos, obj_rotm):
        """15-dim consistency signal between the state and the state-free prior.

        [innov_l(3), innov_r(3), |innov_l|, |innov_r|, bimanual(3), |bimanual|,
         gap_age, rest_age, rest_flag], innovations expressed in the object frame.
        """
        l_pos = state_trust["prior_l_pos"][:, time_index]
        r_pos = state_trust["prior_r_pos"][:, time_index]
        innov_l = self._object_frame_vector(obj_rotm, l_pos - prev_pos)
        innov_r = self._object_frame_vector(obj_rotm, r_pos - prev_pos)
        bimanual = self._object_frame_vector(obj_rotm, l_pos - r_pos)
        norm = self.state_trust_age_norm
        return torch.cat(
            (
                innov_l,
                innov_r,
                innov_l.norm(dim=-1, keepdim=True),
                innov_r.norm(dim=-1, keepdim=True),
                bimanual,
                bimanual.norm(dim=-1, keepdim=True),
                (state_trust["gap_age"][:, time_index] / norm).clamp(max=2.0).unsqueeze(-1),
                (state_trust["rest_age"][:, time_index] / norm).clamp(max=2.0).unsqueeze(-1),
                state_trust["rest_flag"][:, time_index].to(prev_pos.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _grasp_prediction_init(self, context, latent0):
        """Predictor init vector rebuilt from a (possibly published) grasp latent."""
        directions0, lengths0 = GraspLatent.split_latent(latent0)
        return torch.cat(
            (
                directions0[:, 0],
                lengths0[:, 0:1],
                directions0[:, 1],
                lengths0[:, 1:2],
                context["contact_init"],
                context["lb_prior_quantiles"].flatten(1),
            ),
            dim=-1,
        )

    def _attach_known_grasp_state(self, context, known_segment_state_prefix, known_obj_trans_prefix_mask):
        """Teacher-force the grasp latent from the published window prefix.

        Frames covered by the prefix reuse the published latent / active flags /
        ages, and a valid prefix at t=0 also replaces the window-init anchoring
        (which would otherwise re-read the published object every window).
        """
        if not isinstance(known_segment_state_prefix, dict):
            return
        known = {}
        for key in GraspLatent.STATE_KEYS:
            values, valid = self._prepare_known_sequence(
                known_segment_state_prefix.get(key), context["batch_size"], context["seq_len"],
                GraspLatent.STATE_DIMS[key], context["device"], context["dtype"], known_obj_trans_prefix_mask,
            )
            known[key] = values
            known["mask"] = valid if "mask" not in known else known["mask"] & valid
        context["grasp_known"] = known
        valid0 = known["mask"][:, 0]
        if not bool(valid0.any()):
            return
        grasp = context["grasp"]
        latent0 = torch.where(valid0[:, None, None], known["grasp_latent"][:, 0].view(-1, 2, 3), grasp["latent0"])
        active0 = torch.where(valid0[:, None], known["grasp_active"][:, 0] > 0.5, grasp["active0"])
        age0 = torch.where(valid0[:, None], known["grasp_age"][:, 0], grasp["age0"])
        grasp.update({"latent0": latent0, "active0": active0, "age0": age0})
        context["prediction_init"] = self._grasp_prediction_init(context, latent0)

    def _init_scan_state(self, context):
        batch_size = context["batch_size"]
        obj_trans_init = context["obj_trans_init"]
        state = {
            "rnn_state": self.object_trans_head.initial_state(context["prediction_init"]),
            "prev_obj_trans": obj_trans_init,
            "prev_prev_obj_trans": torch.zeros_like(obj_trans_init),
            "prev_obj_delta": torch.zeros_like(obj_trans_init),
            "prev_gating_weights": obj_trans_init.new_tensor((0.0, 0.0, 0.0, 1.0)).expand(batch_size, -1).clone(),
            "prev_obj_rotm": None,
            "lb_prior_quantiles": context["lb_prior_quantiles"],
            "frames_seen": 0,
        }
        if self.bigrasp_enabled:
            state["bigrasp_vector"] = torch.zeros_like(obj_trans_init)
            state["bigrasp_active"] = torch.zeros(batch_size, device=obj_trans_init.device, dtype=torch.bool)
        if self.grasp_latent_enabled:
            grasp = context["grasp"]
            state["grasp_latent"] = grasp["latent0"]
            state["grasp_active"] = grasp["active0"]
            state["grasp_age"] = grasp["age0"]
        return state

    @staticmethod
    def _object_frame_vector(obj_rotm, vector):
        return torch.bmm(obj_rotm.transpose(-1, -2), vector.unsqueeze(-1)).squeeze(-1)

    def _feedback_features(self, obj_rotm, lhand_position, rhand_position, state):
        if self.grasp_latent_enabled:
            # Latent-based feedback: no published-object geometry per frame.
            return self.grasp_latent.feedback_features(
                state["grasp_latent"],
                state["grasp_active"],
                state["grasp_age"],
                self._object_frame_vector(obj_rotm, state["prev_obj_delta"]),
                state["prev_gating_weights"],
            )
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

    def _decode_prediction_output(self, prediction_out, contact_prob=None):
        del contact_prob
        l_fk_out, r_fk_out, gate_logits = prediction_out.split((4, 4, 4), dim=-1)
        return l_fk_out, r_fk_out, F.softmax(gate_logits, dim=-1), gate_logits

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
        step_state = state
        feedback_mode = context.get("feedback_mode", "normal")
        if feedback_mode == "gt_prev":
            step_state = dict(state)
            step_state["prev_obj_trans"] = context["feedback_prev_obj_trans"][:, time_index]
            step_state["prev_prev_obj_trans"] = context["feedback_prev_prev_obj_trans"][:, time_index]
            step_state["prev_obj_delta"] = context["feedback_prev_obj_delta"][:, time_index]
        elif feedback_mode == "perturb":
            injection_mask = context["feedback_injection_mask"][:, time_index]
            if bool(injection_mask.any().item()):
                step_state = dict(state)
                step_state["prev_obj_trans"] = torch.where(
                    injection_mask.unsqueeze(-1),
                    state["prev_obj_trans"]
                    + context["feedback_perturbation"][:, time_index],
                    state["prev_obj_trans"],
                )

        obj_rotm = context["obj_rotm"][:, time_index]
        lhand_position = context["lhand_position"][:, time_index]
        rhand_position = context["rhand_position"][:, time_index]
        step_input = context["prediction_input"][:, time_index]
        state_confidence = None
        state_confidence_logit = None
        if self._state_feedback_enabled():
            feedback = self.feedback_embed(
                self._feedback_features(obj_rotm, lhand_position, rhand_position, step_state)
            )
            state_trust = context.get("state_trust")
            if state_trust is not None:
                z = self._consistency_features_at(
                    state_trust, time_index, step_state["prev_obj_trans"], obj_rotm
                )
                state_confidence_logit = self.confidence_head(z).squeeze(-1)
                state_confidence = torch.sigmoid(state_confidence_logit)
                if self.state_trust_force_confidence is not None:
                    state_confidence = torch.full_like(state_confidence, self.state_trust_force_confidence)
                feedback = state_confidence.unsqueeze(-1).to(feedback.dtype) * feedback
                step_input = step_input + self.consistency_embed(z).to(step_input.dtype)
            step_input = step_input + feedback

        if prediction_out is None:
            prediction_out, rnn_state = self.object_trans_head.step(step_input, state["rnn_state"])
        else:
            rnn_state = state["rnn_state"] if next_rnn_state is None else next_rnn_state
        l_fk_out, r_fk_out, weights, gate_logits = self._decode_prediction_output(
            prediction_out,
            context["pred_hand_contact_prob"][:, time_index],
        )
        weights = self._apply_static_gate(weights, context, time_index)
        grasp_output = {}
        if self.grasp_latent_enabled:
            grasp_ctx = context["grasp"]
            hidden = rnn_state[0][-1].to(obj_rotm.dtype)  # top-layer LSTM state after this frame
            state_pos = step_state["prev_obj_trans"]
            observed = torch.stack(
                (
                    self._object_frame_vector(obj_rotm, state_pos - lhand_position),
                    self._object_frame_vector(obj_rotm, state_pos - rhand_position),
                ),
                dim=1,
            )
            update = self.grasp_latent.step(
                hidden, step_state["grasp_latent"], step_state["grasp_active"], step_state["grasp_age"],
                context["pred_hand_contact_prob"][:, time_index], observed, grasp_ctx["prior_bone"][:, time_index],
                grasp_ctx["rest_flag"][:, time_index], grasp_ctx["rest_age"][:, time_index], grasp_ctx["gap_age"][:, time_index],
            )
            grasp_latent, grasp_active, grasp_age = update["latent"], update["active"], update["age"]
            grasp_bone = update["bone"]
            known = context.get("grasp_known")
            if known is not None:
                # Published latents are authoritative on prefix frames.
                valid = known["mask"][:, time_index]
                known_latent = known["grasp_latent"][:, time_index].view(-1, 2, 3)
                # residual mode: bone = anchor + residual, so shift the bone with the anchor
                grasp_bone = torch.where(valid[:, None, None], known_latent + (grasp_bone - grasp_latent), grasp_bone)
                grasp_latent = torch.where(valid[:, None, None], known_latent, grasp_latent)
                grasp_active = torch.where(valid[:, None], known["grasp_active"][:, time_index] > 0.5, grasp_active)
                grasp_age = torch.where(valid[:, None], known["grasp_age"][:, time_index], grasp_age)
            directions, lengths = GraspLatent.split_latent(grasp_bone)
            l_dir, r_dir = directions[:, 0], directions[:, 1]
            l_len, r_len = lengths[:, 0], lengths[:, 1]
            grasp_output = {
                "grasp_latent": grasp_latent.flatten(1),
                "grasp_active": grasp_active.to(grasp_latent.dtype),
                "grasp_age": grasp_age,
                "grasp_alpha": update["alpha"],
                "grasp_onset": update["onset"].to(grasp_latent.dtype),
                "grasp_delta": update["delta"].flatten(1),
            }
            if update["rigid"] is not None:
                grasp_output["grasp_rigid_prob"] = update["rigid"]
                grasp_output["grasp_rigid_logit"] = update["rigid_logit"]
        else:
            l_dir = self._unit_vector(l_fk_out[:, :3])
            r_dir = self._unit_vector(r_fk_out[:, :3])
            l_len = self._softplus_positive(l_fk_out[:, 3])
            r_len = self._softplus_positive(r_fk_out[:, 3])
        l_dir_world = torch.bmm(obj_rotm, l_dir.unsqueeze(-1)).squeeze(-1)
        r_dir_world = torch.bmm(obj_rotm, r_dir.unsqueeze(-1)).squeeze(-1)
        l_pos_fk = lhand_position + l_dir_world * l_len.unsqueeze(-1)
        r_pos_fk = rhand_position + r_dir_world * r_len.unsqueeze(-1)

        previous_pos = step_state["prev_obj_trans"]
        imu_integrated_pos = previous_pos + context["obj_vel_input"][:, time_index] / FRAME_RATE
        static_pos = previous_pos
        bigrasp_output = {}
        if self.bigrasp_enabled:
            encoded = context["bigrasp"]
            hidden = encoded["hidden"][:, time_index]
            proposal = encoded["proposal"][:, time_index]
            vector, active, rate = self.bigrasp.update(
                hidden, proposal, step_state["bigrasp_vector"], step_state["bigrasp_active"],
                context["pred_hand_contact_prob"][:, time_index],
            )
            world_vector = torch.bmm(obj_rotm, vector.unsqueeze(-1)).squeeze(-1)
            pair_pos = (lhand_position + rhand_position) * 0.5 + world_vector
            candidates = torch.stack((l_pos_fk, r_pos_fk, imu_integrated_pos, static_pos, pair_pos), dim=1)
            pair_weights, quality_logits = self.bigrasp.quality(
                hidden, candidates, previous_pos, obj_rotm, weights,
                context["pred_hand_contact_prob"][:, time_index], active,
            )
            if context.get("static_gate") is not None:
                static_weights = pair_weights.new_tensor((0.0, 0.0, 0.0, 1.0, 0.0))
                pair_weights = torch.where(context["static_gate"][:, time_index, None], static_weights, pair_weights)
            effective_hands, weights = self.bigrasp.publication_candidates(candidates, pair_weights)
            l_pos_fk, r_pos_fk = effective_hands.unbind(dim=1)
            effective_bones = torch.stack((l_pos_fk - lhand_position, r_pos_fk - rhand_position), dim=1)
            effective_lengths = effective_bones.norm(dim=-1)
            local_directions = torch.einsum("bji,bhj->bhi", obj_rotm, effective_bones) / effective_lengths.clamp_min(1e-8).unsqueeze(-1)
            l_dir, r_dir = local_directions.unbind(dim=1)
            l_len, r_len = effective_lengths.unbind(dim=1)
            bigrasp_output = {
                "bigrasp_vector": vector,
                "bigrasp_active": active.to(vector.dtype).unsqueeze(-1),
                "bigrasp_update_rate": rate,
                "bigrasp_proposal_world": torch.bmm(obj_rotm, proposal.unsqueeze(-1)).squeeze(-1),
                "bigrasp_vector_world": world_vector,
                "bigrasp_candidates": candidates,
                "bigrasp_weights": pair_weights,
                "bigrasp_quality_logits": quality_logits,
                "pred_bimanual_obj_trans": pair_pos,
            }
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
        frames_seen = int(step_state.get("frames_seen", 0))
        velocity = (fused_pos - previous_pos) * FRAME_RATE if frames_seen > 0 else torch.zeros_like(fused_pos)
        acceleration = (
            (fused_pos - 2.0 * previous_pos + step_state["prev_prev_obj_trans"]) * (FRAME_RATE ** 2)
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
        frame_output = {
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
        }
        if gate_logits is not None:
            frame_output["gate_logits"] = gate_logits
        if self.bigrasp_enabled:
            next_state["bigrasp_vector"] = vector
            next_state["bigrasp_active"] = active
            known = context.get("bigrasp_known")
            if known is not None:
                valid = known["mask"][:, time_index]
                next_state["bigrasp_vector"] = torch.where(valid[:, None], known["bigrasp_vector"][:, time_index], vector)
                next_state["bigrasp_active"] = torch.where(valid, known["bigrasp_active"][:, time_index, 0] > 0.5, active)
                bigrasp_output["bigrasp_vector"] = next_state["bigrasp_vector"]
                bigrasp_output["bigrasp_active"] = next_state["bigrasp_active"].to(vector.dtype).unsqueeze(-1)
            frame_output.update(bigrasp_output)
        if self.grasp_latent_enabled:
            next_state["grasp_latent"] = grasp_latent
            next_state["grasp_active"] = grasp_active
            next_state["grasp_age"] = grasp_age
            frame_output.update(grasp_output)
        if state_confidence is not None:
            frame_output["state_confidence"] = state_confidence
            frame_output["state_confidence_logit"] = state_confidence_logit
            # The state this frame was conditioned on (fused position of the
            # previous frame, or the window init): the confidence target in
            # training compares it with the GT object of the previous frame.
            frame_output["ot_state_prev_obj_trans"] = step_state["prev_obj_trans"]
        return frame_output, next_state

    def _scan_predictor(
        self,
        context,
        known_obj_trans_prefix=None,
        known_obj_trans_prefix_mask=None,
        known_gating_weights_prefix=None,
    ):
        state = self._init_scan_state(context)
        use_vectorized_fast_path = (
            not self._state_feedback_enabled()
            and not isinstance(known_obj_trans_prefix, torch.Tensor)
            and not isinstance(known_gating_weights_prefix, torch.Tensor)
            and context.get("bigrasp_known") is None
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
            "lb_prior_quantiles": context["lb_prior_quantiles"],
            "gating_contact_prob": context["pred_hand_contact_prob"],
            "gating_smoothing_applied": False,
            # The hands this module consumed (stage-1 palms or the GT/pred mix
            # during training); the metric-aligned HOI loss needs them.
            "ot_hand_positions": context["hand_positions"],
        }
        state_trust = context.get("state_trust")
        if state_trust is not None:
            results.update(
                {
                    "prior_lhand_obj_trans": state_trust["prior_l_pos"],
                    "prior_rhand_obj_trans": state_trust["prior_r_pos"],
                    "prior_lhand_obj_direction": state_trust["prior_l_dir"],
                    "prior_rhand_obj_direction": state_trust["prior_r_dir"],
                    "prior_lhand_lb": state_trust["prior_l_len"],
                    "prior_rhand_lb": state_trust["prior_r_len"],
                    "state_trust_rest_flag": state_trust["rest_flag"],
                    "state_trust_rest_age": state_trust["rest_age"],
                    "state_trust_gap_age": state_trust["gap_age"],
                    "state_trust_init_confidence": state_trust["init_confidence"],
                }
            )
        grasp = context.get("grasp")
        if grasp is not None:
            results.update(
                {
                    "prior_lhand_obj_trans": grasp["prior_pos"][:, :, 0],
                    "prior_rhand_obj_trans": grasp["prior_pos"][:, :, 1],
                    "grasp_prior_bone": grasp["prior_bone"],
                    "grasp_rest_flag": grasp["rest_flag"],
                    "grasp_rest_age": grasp["rest_age"],
                    "grasp_gap_age": grasp["gap_age"],
                    "grasp_init_alpha": grasp["alpha0"],
                }
            )
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
                "prior_lhand_obj_trans",
                "prior_rhand_obj_trans",
            ):
                if key in results:
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
        lb_prior_quantiles: torch.Tensor = None,
        enable_refine: bool = True,
        known_obj_trans_prefix: torch.Tensor = None,
        known_obj_trans_prefix_mask: torch.Tensor = None,
        known_gating_weights_prefix: torch.Tensor = None,
        compute_refine: bool = True,
        feedback_control=None,
        known_segment_state_prefix=None,
    ):
        """Run a causal OT scan from the supplied window initializers."""
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
            lb_prior_quantiles=lb_prior_quantiles,
            feedback_control=feedback_control,
        )
        if self.bigrasp_enabled and isinstance(known_segment_state_prefix, dict):
            known = {}
            for key, feature_dim in (("bigrasp_vector", 3), ("bigrasp_active", 1)):
                values, valid = self._prepare_known_sequence(
                    known_segment_state_prefix.get(key), context["batch_size"], context["seq_len"], feature_dim,
                    context["device"], context["dtype"], known_obj_trans_prefix_mask,
                )
                known[key] = values
                known["mask"] = valid if "mask" not in known else known["mask"] & valid
            context["bigrasp_known"] = known
        if self.grasp_latent_enabled:
            self._attach_known_grasp_state(context, known_segment_state_prefix, known_obj_trans_prefix_mask)
        prediction, _ = self._scan_predictor(
            context,
            known_obj_trans_prefix=known_obj_trans_prefix,
            known_obj_trans_prefix_mask=known_obj_trans_prefix_mask,
            known_gating_weights_prefix=known_gating_weights_prefix,
        )
        results = self._build_results(context, prediction, enable_refine=enable_refine and compute_refine)
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
                key: (
                    value
                    if key in NON_TEMPORAL_KEYS
                    else self._slice_optional_time(value, start, end, batch_size, seq_len)
                )
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
            segment_kwargs = {}
            if self.bigrasp_enabled:
                segment_kwargs["known_segment_state_prefix"] = {key: prefix_context.get(key) for key in self.SEGMENT_STATE_OUTPUT_KEYS}
            if self.grasp_latent_enabled:
                segment_kwargs["known_segment_state_prefix"] = {key: prefix_context.get(key) for key in GraspLatent.STATE_KEYS}
            window_out = self.forward(
                hand_positions[:, start:end],
                pred_hand_contact_prob[:, start:end],
                step_obj_trans_init,
                known_obj_trans_prefix=prefix,
                known_gating_weights_prefix=prefix_weights,
                **segment_kwargs,
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
            # The l_b prior describes the caller's *current* statistics;
            # append_stream_data would otherwise freeze the first chunk's
            # value (correct for init vectors, wrong for a live prior).
            if isinstance(stream_dict.get("lb_prior_quantiles"), torch.Tensor):
                kwargs["lb_prior_quantiles"] = stream_dict["lb_prior_quantiles"]
                run_data["lb_prior_quantiles"] = stream_dict["lb_prior_quantiles"]
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
            "lb_prior_quantiles": torch.zeros(batch_size, 2, 3, device=device),
            "init_lhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_rhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_lhand_lb": torch.zeros(batch_size, device=device),
            "init_rhand_lb": torch.zeros(batch_size, device=device),
            "gating_smoothing_applied": False,
        }
