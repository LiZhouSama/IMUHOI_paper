"""Controlled VelocityContact ablations built on the EgoIMU IMUHOI backbone."""
from __future__ import annotations

import torch
import torch.nn as nn

from configs import FRAME_RATE, _SENSOR_NAMES
from .base import RNNWithInit
from .velocity_contact import (
    VelocityContactModule as CurrentVelocityContactModule,
    recover_contact_state_from_endpoints,
)


IMUHOI_VC_ABLATION_ARCHES = (
    "imuhoi_cond",
    "imuhoi_cond_bi",
    "imuhoi_cond_geom",
    "imuhoi_cond_boundary",
    "imuhoi_cond_boundary_endpoint",
)


class IMUHOIConditionalVelocityContactModule(nn.Module):
    """Incremental contact ablations with the original IMUHOI velocity heads.

    Every variant keeps the same velocity backbone:

    * all human IMUs -> one six-channel hand-velocity LSTM;
    * object IMU -> an independent three-channel object-velocity LSTM.

    The variants change only the contact path.  The base 36-D contact evidence
    is identical to the original IMUHOI contact input (hand/object IMUs plus
    predicted hand/object velocity).  Hand contact is predicted conditionally
    on object movement, so the final probabilities are
    ``P(hand contact) = P(object moving) * P(hand contact | object moving)``.
    """

    _denormalize_imu = CurrentVelocityContactModule._denormalize_imu
    _prepare_pose_streams = CurrentVelocityContactModule._prepare_pose_streams
    _causal_velocity = CurrentVelocityContactModule._causal_velocity
    _causal_acceleration = CurrentVelocityContactModule._causal_acceleration
    _prepare_init_vector = CurrentVelocityContactModule._prepare_init_vector
    _integrate_rough_object_translation = (
        CurrentVelocityContactModule._integrate_rough_object_translation
    )
    _cosine_similarity = staticmethod(CurrentVelocityContactModule._cosine_similarity)
    _build_contact_geom_features = (
        CurrentVelocityContactModule._build_contact_geom_features
    )
    _encode_body_groups = CurrentVelocityContactModule._encode_body_groups
    decode_contact_segments = CurrentVelocityContactModule.decode_contact_segments
    _apply_segment_contact = CurrentVelocityContactModule._apply_segment_contact
    _slice_hp_out = CurrentVelocityContactModule._slice_hp_out
    _inference_online_sequence = CurrentVelocityContactModule._inference_online_sequence
    inference = CurrentVelocityContactModule.inference

    def __init__(self, cfg, arch: str):
        super().__init__()
        if arch not in IMUHOI_VC_ABLATION_ARCHES:
            raise ValueError(f"Unknown IMUHOI VC ablation architecture: {arch!r}")

        self.cfg = cfg
        self.arch = arch
        self.num_human_imus = int(
            getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        )
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(getattr(cfg, "obj_imu_dim", 9))
        hidden_dim = int(getattr(cfg, "velocity_hidden_dim", 128))
        num_layers = int(getattr(cfg, "velocity_num_layers", 2))
        dropout = float(getattr(cfg, "velocity_dropout", 0.2))
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.detach_contact_geom_obj_vel = bool(
            getattr(cfg, "velocity_contact_geom_detach_obj_vel", True)
        )

        self.use_contact_geom_features = arch == "imuhoi_cond_geom"
        self.use_boundary_features = arch in {
            "imuhoi_cond_boundary",
            "imuhoi_cond_boundary_endpoint",
        }
        self.use_contact_endpoints = arch == "imuhoi_cond_boundary_endpoint"
        self.obj_move_bidirectional = arch == "imuhoi_cond_bi"
        self.requires_hp = self.use_contact_geom_features or self.use_boundary_features

        self.hand_vel_net = RNNWithInit(
            n_input=self.num_human_imus * self.imu_dim,
            n_output=6,
            n_hidden=hidden_dim,
            n_init=6,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )
        self.obj_vel_net = RNNWithInit(
            n_input=self.obj_imu_dim,
            n_output=3,
            n_hidden=hidden_dim,
            n_init=3,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        self.base_contact_input_dim = (
            2 * self.imu_dim + self.obj_imu_dim + 6 + 3
        )
        contact_hidden = max(hidden_dim // 2, 32)

        self.obj_move_input_dim = 2 * self.imu_dim + self.obj_imu_dim
        # Match the current VC's object-motion head: it sees only the two hand
        # IMUs and object IMU.  The conditional hand head keeps the original
        # IMUHOI 36-D evidence, including predicted velocities.
        self.obj_move_net = RNNWithInit(
            n_input=self.obj_move_input_dim,
            n_output=1,
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=self.obj_move_bidirectional,
            dropout=dropout,
        )

        self.contact_geom_dim = 2 * (3 + 1 + 3 + 1)
        hand_contact_input_dim = self.base_contact_input_dim

        if self.use_boundary_features:
            self._init_boundary_encoder(cfg, hidden_dim)
            hand_contact_input_dim += self.boundary_hidden_dim
        if self.use_contact_geom_features:
            hand_contact_input_dim += self.contact_geom_dim

        self.hand_contact_input_dim = hand_contact_input_dim
        self.hand_contact_net = RNNWithInit(
            n_input=hand_contact_input_dim,
            n_output=2,
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )

        if self.use_contact_endpoints:
            self.contact_endpoint_channels = 6
            self.contact_endpoint_input_dim = hand_contact_input_dim
            self.contact_endpoint_input_norm = nn.LayerNorm(
                self.contact_endpoint_input_dim
            )
            self.contact_endpoint_rnn = nn.GRU(
                input_size=self.contact_endpoint_input_dim,
                hidden_size=contact_hidden,
                num_layers=1,
                batch_first=True,
            )
            self.contact_endpoint_head = nn.Linear(
                contact_hidden,
                self.contact_endpoint_channels,
            )

        self.left_hand_sensor = _SENSOR_NAMES.index("LeftForeArm")
        self.right_hand_sensor = _SENSOR_NAMES.index("RightForeArm")

    def _init_boundary_encoder(self, cfg, hidden_dim):
        self.body_group_indices = {
            "left_arm": [13, 16, 18, 20, 22],
            "right_arm": [14, 17, 19, 21, 23],
            "torso": [0, 3, 6, 9, 12, 15],
            "left_leg": [1, 4, 7, 10],
            "right_leg": [2, 5, 8, 11],
        }
        group_hidden = int(
            getattr(cfg, "boundary_group_hidden", max(hidden_dim // 2, 32))
        )
        boundary_hidden = int(getattr(cfg, "boundary_hidden_dim", hidden_dim))
        self.boundary_hidden_dim = boundary_hidden
        # Keep this encoder causal in experiments 4/5.  Experiment 2 changes
        # only obj_move_net directionality, not the body-feature encoder.
        self.boundary_bidirectional = False
        self.group_encoders = nn.ModuleDict()
        self.group_norms = nn.ModuleDict()
        for name, indices in self.body_group_indices.items():
            group_input_dim = len(indices) * (6 + 3 + 3) + 3
            self.group_norms[name] = nn.LayerNorm(group_input_dim)
            self.group_encoders[name] = nn.GRU(
                input_size=group_input_dim,
                hidden_size=group_hidden,
                num_layers=1,
                batch_first=True,
            )
        boundary_input_dim = group_hidden * len(self.body_group_indices)
        self.boundary_input_norm = nn.LayerNorm(boundary_input_dim)
        self.boundary_rnn = nn.GRU(
            input_size=boundary_input_dim,
            hidden_size=boundary_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.boundary_fusion = nn.Identity()
        self.boundary_hidden_norm = nn.LayerNorm(boundary_hidden)

    @staticmethod
    def _prepare_init(value, batch_size, dim, device, dtype, name):
        if not isinstance(value, torch.Tensor):
            return torch.zeros(batch_size, dim, device=device, dtype=dtype)
        out = value.to(device=device, dtype=dtype)
        if out.dim() == 1:
            out = out.unsqueeze(0).expand(batch_size, -1)
        if out.dim() == 3:
            out = out.reshape(batch_size, -1)
        if out.shape != (batch_size, dim):
            raise ValueError(
                f"{name} must resolve to [B,{dim}], got {tuple(out.shape)}"
            )
        return out

    def validate_checkpoint_state_dict(self, state_dict):
        """Reject checkpoints from a different ablation topology."""
        if not isinstance(state_dict, dict):
            raise ValueError("VC ablation checkpoint must contain a state_dict.")
        normalized = {
            str(key[7:] if str(key).startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        expected = self.state_dict()
        missing = [key for key in expected if key not in normalized]
        mismatched = [
            key
            for key, value in expected.items()
            if key in normalized
            and getattr(normalized[key], "shape", None) != value.shape
        ]
        unexpected_topology = [
            key
            for key in normalized
            if key not in expected
            and key.startswith(
                (
                    "obj_move_net.",
                    "hand_contact_net.",
                    "group_encoders.",
                    "boundary_",
                    "contact_endpoint_",
                )
            )
        ]
        if missing or mismatched or unexpected_topology:
            raise ValueError(
                f"Checkpoint is incompatible with VC ablation {self.arch!r}: "
                f"missing={missing[:3]}, mismatched={mismatched[:3]}, "
                f"unexpected={unexpected_topology[:3]}"
            )

    def forward(
        self,
        data_dict: dict,
        hp_out: dict = None,
        velocity_only: bool = False,
    ):
        human_imu = data_dict["human_imu"]
        obj_imu = data_dict.get("obj_imu")
        if human_imu.dim() != 4:
            raise ValueError(
                "human_imu must be [B,T,num_imu,imu_dim], "
                f"got {tuple(human_imu.shape)}"
            )
        batch_size, seq_len = human_imu.shape[:2]
        device, dtype = human_imu.device, human_imu.dtype
        if human_imu.shape[2:] != (self.num_human_imus, self.imu_dim):
            raise ValueError(
                f"expected human_imu sensor shape "
                f"{(self.num_human_imus, self.imu_dim)}, "
                f"got {tuple(human_imu.shape[2:])}"
            )

        if obj_imu is None:
            obj_imu = torch.zeros(
                batch_size,
                seq_len,
                self.obj_imu_dim,
                device=device,
                dtype=dtype,
            )
        else:
            obj_imu = obj_imu.to(device=device, dtype=dtype)
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(batch_size, seq_len, -1)
            if obj_imu.shape != (batch_size, seq_len, self.obj_imu_dim):
                raise ValueError(
                    f"obj_imu must be [B,T,{self.obj_imu_dim}], "
                    f"got {tuple(obj_imu.shape)}"
                )

        hand_vel_init = self._prepare_init(
            data_dict.get("hand_vel_glb_init"),
            batch_size,
            6,
            device,
            dtype,
            "hand_vel_glb_init",
        )
        obj_vel_init = self._prepare_init(
            data_dict.get("obj_vel_init"),
            batch_size,
            3,
            device,
            dtype,
            "obj_vel_init",
        )
        contact_init = self._prepare_init_vector(
            data_dict.get("contact_init"),
            batch_size,
            3,
            device,
            dtype,
        )

        human_imu_denorm = self._denormalize_imu(human_imu)
        human_imu_flat = human_imu_denorm.reshape(batch_size, seq_len, -1)
        hand_vel_flat = self.hand_vel_net((human_imu_flat, hand_vel_init))
        hand_vel = hand_vel_flat.reshape(batch_size, seq_len, 2, 3)
        obj_vel = self.obj_vel_net((obj_imu, obj_vel_init))
        if velocity_only:
            return {
                "pred_hand_glb_vel": hand_vel,
                "pred_hand_glb_vel_flat": hand_vel_flat,
                "pred_obj_vel": obj_vel,
            }

        hand_imu_feat = human_imu_denorm[
            :, :, [self.left_hand_sensor, self.right_hand_sensor], :
        ].reshape(batch_size, seq_len, -1)
        contact_input_imu = torch.cat((hand_imu_feat, obj_imu), dim=-1)
        base_contact_input = torch.cat(
            (hand_imu_feat, obj_imu, hand_vel_flat, obj_vel),
            dim=-1,
        )

        hand_contact_input = base_contact_input
        pose_streams = None
        if self.requires_hp:
            pose_streams = self._prepare_pose_streams(
                hp_out or data_dict.get("hp_out"),
                device,
                dtype,
                batch_size,
                seq_len,
            )
        if self.use_boundary_features:
            boundary_out = self._encode_body_groups(pose_streams)
            hand_contact_input = torch.cat(
                (hand_contact_input, boundary_out),
                dim=-1,
            )

        obj_trans_rough = None
        if self.use_contact_geom_features:
            obj_trans_init = self._prepare_init_vector(
                data_dict.get("obj_trans_init"),
                batch_size,
                3,
                device,
                dtype,
            )
            geom_feat, obj_trans_rough = self._build_contact_geom_features(
                pose_streams,
                obj_vel,
                obj_trans_init,
            )
            hand_contact_input = torch.cat(
                (hand_contact_input, geom_feat),
                dim=-1,
            )

        obj_move_logits = self.obj_move_net((contact_input_imu, contact_init))
        obj_move_prob = torch.sigmoid(obj_move_logits)
        hand_contact_logits = self.hand_contact_net(
            (hand_contact_input, contact_init)
        )
        hand_contact_prob_cond = torch.sigmoid(hand_contact_logits)
        hand_contact_prob = obj_move_prob * hand_contact_prob_cond

        out = {
            "pred_hand_glb_vel": hand_vel,
            "pred_hand_glb_vel_flat": hand_vel_flat,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": torch.cat(
                (hand_contact_logits, obj_move_logits),
                dim=-1,
            ),
            "pred_hand_contact_prob": torch.cat(
                (hand_contact_prob, obj_move_prob),
                dim=-1,
            ),
            "pred_obj_move_prob": obj_move_prob,
            "pred_hand_contact_prob_cond": hand_contact_prob_cond,
            "pred_hand_contact_logits_cond": hand_contact_logits,
        }
        if obj_trans_rough is not None:
            out["pred_obj_trans_rough"] = obj_trans_rough

        if self.use_contact_endpoints:
            endpoint_input = self.contact_endpoint_input_norm(
                hand_contact_input
            )
            endpoint_hidden, _ = self.contact_endpoint_rnn(endpoint_input)
            endpoint_logits = self.contact_endpoint_head(endpoint_hidden)
            endpoint_prob = torch.sigmoid(endpoint_logits)
            out.update(
                {
                    "pred_contact_endpoint_logits": endpoint_logits,
                    "pred_contact_endpoint_prob": endpoint_prob,
                    "pred_contact_from_endpoints_prob": (
                        recover_contact_state_from_endpoints(
                            endpoint_prob,
                            contact_init,
                        )
                    ),
                }
            )
        return out

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        return {
            "pred_hand_glb_vel": torch.zeros(
                batch_size, seq_len, 2, 3, device=device
            ),
            "pred_hand_glb_vel_flat": torch.zeros(
                batch_size, seq_len, 6, device=device
            ),
            "pred_obj_vel": zeros_pos.clone(),
            "pred_hand_contact_logits": zeros_pos.clone(),
            "pred_hand_contact_prob": zeros_pos.clone(),
            "pred_obj_move_prob": torch.zeros(
                batch_size, seq_len, 1, device=device
            ),
            "pred_hand_contact_prob_cond": torch.zeros(
                batch_size, seq_len, 2, device=device
            ),
            "pred_hand_contact_logits_cond": torch.zeros(
                batch_size, seq_len, 2, device=device
            ),
        }


__all__ = [
    "IMUHOI_VC_ABLATION_ARCHES",
    "IMUHOIConditionalVelocityContactModule",
]
