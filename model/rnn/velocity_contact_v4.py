"""VC v4: a supervised, heading-invariant human-pose prior for contact.

v4 starts from v3 (conditional contact factorization plus a bidirectional
object-move head) and turns the human-pose stream from an unconstrained side
input into an explicitly supervised interaction prior:

* body-group features are heading-normalized, per-modality normalized and
  causally smoothed before the group encoders.  v3 concatenated world-frame
  pose 6D, ``fps``-scaled velocity and ``fps**2``-scaled acceleration and
  applied one LayerNorm over the whole vector, so the acceleration channels
  dominated the normalization statistics and the compass direction of the
  motion had to be memorized.
* the boundary encoder is bidirectional, matching the temporal access v3
  already granted the object-move head.  Reach evidence precedes an onset and
  retract evidence follows an offset, so this encoder benefits most from it.
* the boundary hidden state also conditions the object-move head.  With
  ``P(contact) = P(object moving) * P(contact | object moving)``, a prior that
  only reaches the conditional factor is capped by ``P(object moving)`` exactly
  at the onsets it is supposed to improve.
* two pose-only auxiliary heads read that hidden state: a contact-segment
  endpoint heatmap and a ``k``-frame anticipation head.  Both see nothing but
  the boundary features, so they force reach/retract motion to be encoded in
  the pose stream, and their accuracy directly measures how much transition
  information the pose stream carries.
* hand-contact evidence gains HP-derived hand velocity/acceleration.

Missing or malformed ``hp_out`` raises instead of being silently zero-filled:
a zeroed pose stream still trains and converges, and would quietly answer this
experiment's central question the wrong way.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from .base import RNNWithInit
from .velocity_contact import recover_contact_state_from_endpoints
from .velocity_contact_ablation import IMUHOIConditionalVelocityContactModule


# hp_out tensors the pose prior needs, with their expected trailing shape.
REQUIRED_HP_OUT_KEYS = (
    ("pred_full_pose_6d", (24, 6)),
    ("pred_joints_local", (24, 3)),
    ("root_vel_pred", (3,)),
    ("root_trans_pred", (3,)),
)
# Either key satisfies the hand-anchor requirement; the palm anchor is preferred.
HAND_POSITION_HP_OUT_KEYS = ("pred_palm_glb_pos", "pred_hand_glb_pos")

DEFAULT_ANTICIPATION_HORIZON = 15


def causal_moving_average(value: torch.Tensor, window: int) -> torch.Tensor:
    """Backward-looking box filter over the time axis of a ``[B, T, ...]`` tensor.

    Causal on purpose: the sliding-window online path publishes the last frame
    of every window, so a symmetric filter would leak future frames that do not
    exist yet at that point in the stream.
    """
    window = int(window)
    if window <= 1 or int(value.shape[1]) <= 1:
        return value
    batch_size, seq_len = value.shape[:2]
    flat = value.reshape(batch_size, seq_len, -1).transpose(1, 2)
    padded = F.pad(flat, (window - 1, 0), mode="replicate")
    smoothed = F.avg_pool1d(padded, kernel_size=window, stride=1)
    return smoothed.transpose(1, 2).reshape(value.shape)


def heading_rotation_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Closest rotation about the world +Y axis for each input rotation.

    ``theta = atan2(R[0,2] - R[2,0], R[0,0] + R[2,2])`` projects ``R`` onto the
    yaw subgroup, which stays well defined while the body leans or lies down.
    Reading a single body axis and flattening it instead degenerates exactly in
    those poses.
    """
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must be [..., 3, 3], got {tuple(rotation.shape)}")
    angle = torch.atan2(
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 0, 0] + rotation[..., 2, 2],
    )
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zero = torch.zeros_like(cos)
    one = torch.ones_like(cos)
    return torch.stack(
        (
            torch.stack((cos, zero, sin), dim=-1),
            torch.stack((zero, one, zero), dim=-1),
            torch.stack((-sin, zero, cos), dim=-1),
        ),
        dim=-2,
    )


class V4VelocityContactModule(IMUHOIConditionalVelocityContactModule):
    """v3 plus a heading-invariant, explicitly supervised pose prior."""

    ARCH_VERSION = 42
    requires_hp = True
    # v4's endpoint head reads boundary features only, so an endpoint-only
    # fine-tune stage would train 6 output channels on a frozen encoder and
    # give the pose prior — the whole point of the head — nothing.  Staged
    # training therefore runs velocity then joint, with no endpoint tail.
    SUPPORTS_ENDPOINT_FINETUNE_STAGE = False

    def __init__(self, cfg):
        super().__init__(cfg, arch="imuhoi_cond_boundary")
        self.arch = "v4"
        self.requires_hp = True
        # v4 replaces hand-object geometry with HP-derived hand dynamics.
        self.use_contact_geom_features = False
        self.obj_move_bidirectional = True
        self.use_contact_endpoints = True

        hidden_dim = int(getattr(cfg, "velocity_hidden_dim", 128))
        contact_hidden = max(hidden_dim // 2, 32)
        dropout = float(getattr(cfg, "velocity_dropout", 0.2))
        boundary_hidden = int(self.boundary_hidden_dim)

        # Per hand: velocity(3) + acceleration(3).
        self.hand_dyn_dim = 2 * (3 + 3)

        self.obj_move_input_dim = (
            2 * self.imu_dim + self.obj_imu_dim + boundary_hidden
        )
        del self.obj_move_net
        self.obj_move_net = RNNWithInit(
            n_input=self.obj_move_input_dim,
            n_output=1,
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=True,
            dropout=dropout,
        )

        self.hand_contact_input_dim = (
            self.base_contact_input_dim + boundary_hidden + self.hand_dyn_dim
        )
        del self.hand_contact_net
        self.hand_contact_net = RNNWithInit(
            n_input=self.hand_contact_input_dim,
            n_output=2,
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )

        # Auxiliary heads read boundary features only.  Denying them the object
        # IMU is what makes them supervise the pose prior instead of bypassing
        # it, and what makes their accuracy a readable probe of the prior.
        self.contact_endpoint_channels = 6
        self.contact_endpoint_input_dim = boundary_hidden
        self.contact_endpoint_input_norm = nn.LayerNorm(boundary_hidden)
        self.contact_endpoint_rnn = nn.GRU(
            input_size=boundary_hidden,
            hidden_size=contact_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.contact_endpoint_head = nn.Linear(
            contact_hidden,
            self.contact_endpoint_channels,
        )

        self.anticipation_horizon = max(
            1,
            int(
                getattr(
                    cfg,
                    "v4_anticipation_horizon",
                    DEFAULT_ANTICIPATION_HORIZON,
                )
            ),
        )
        self.contact_anticipation_channels = 6
        self.contact_anticipation_norm = nn.LayerNorm(boundary_hidden)
        # A shallow probe on purpose: if this predicts upcoming transitions, the
        # boundary features themselves encode them.
        self.contact_anticipation_head = nn.Sequential(
            nn.Linear(boundary_hidden, contact_hidden),
            nn.ReLU(),
            nn.Linear(contact_hidden, self.contact_anticipation_channels),
        )

        # One-shot content check; see _validate_hp_out.
        self._pose_stream_content_checked = False

        self.register_buffer(
            "_v4_arch_version",
            torch.tensor(self.ARCH_VERSION, dtype=torch.int64),
        )
        self.register_buffer(
            "_v4_anticipation_horizon",
            torch.tensor(self.anticipation_horizon, dtype=torch.int64),
        )

    def _init_boundary_encoder(self, cfg, hidden_dim):
        """Heading-invariant, per-modality-normalized, bidirectional encoder."""
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
        self.boundary_bidirectional = bool(
            getattr(cfg, "v4_boundary_bidirectional", True)
        )
        self.boundary_use_joint_acc = bool(
            getattr(cfg, "v4_boundary_use_joint_acc", True)
        )
        self.boundary_smooth_window = max(
            1, int(getattr(cfg, "v4_boundary_smooth_window", 5))
        )
        self.heading_normalize = bool(getattr(cfg, "v4_heading_normalize", True))

        self.group_encoders = nn.ModuleDict()
        self.group_pose_norms = nn.ModuleDict()
        self.group_vel_norms = nn.ModuleDict()
        self.group_acc_norms = nn.ModuleDict()
        for name, indices in self.body_group_indices.items():
            pose_dim = len(indices) * 6
            motion_dim = len(indices) * 3
            self.group_pose_norms[name] = nn.LayerNorm(pose_dim)
            self.group_vel_norms[name] = nn.LayerNorm(motion_dim)
            group_input_dim = pose_dim + motion_dim + 3
            if self.boundary_use_joint_acc:
                self.group_acc_norms[name] = nn.LayerNorm(motion_dim)
                group_input_dim += motion_dim
            self.group_encoders[name] = nn.GRU(
                input_size=group_input_dim,
                hidden_size=group_hidden,
                num_layers=1,
                batch_first=True,
            )
        self.root_vel_norm = nn.LayerNorm(3)

        boundary_input_dim = group_hidden * len(self.body_group_indices)
        self.boundary_input_norm = nn.LayerNorm(boundary_input_dim)
        self.boundary_rnn = nn.GRU(
            input_size=boundary_input_dim,
            hidden_size=boundary_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=self.boundary_bidirectional,
        )
        self.boundary_fusion = (
            nn.Linear(boundary_hidden * 2, boundary_hidden)
            if self.boundary_bidirectional
            else nn.Identity()
        )
        self.boundary_hidden_norm = nn.LayerNorm(boundary_hidden)

    @staticmethod
    def _rotate_vectors(value: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Express world-frame row vectors ``[B, T, ..., 3]`` in the yaw frame."""
        batch_size, seq_len = value.shape[:2]
        flat = value.reshape(batch_size, seq_len, -1, 3)
        rotated = torch.matmul(flat, yaw)
        return rotated.reshape(value.shape)

    def _heading_frame(self, pose_6d: torch.Tensor) -> torch.Tensor:
        """Yaw rotation of the pelvis, as a row-vector world->heading operator."""
        batch_size, seq_len = pose_6d.shape[:2]
        root_rot = rotation_6d_to_matrix(
            pose_6d[:, :, 0, :].reshape(-1, 6).float()
        )
        yaw = heading_rotation_matrix(root_rot).reshape(batch_size, seq_len, 3, 3)
        return yaw.to(dtype=pose_6d.dtype)

    def _validate_hp_out(self, hp_out, batch_size: int, seq_len: int):
        if not isinstance(hp_out, dict):
            raise ValueError(
                "VC v4 requires HumanPose outputs (hp_out) but received "
                f"{type(hp_out).__name__}. Zero-filling the pose stream would "
                "silently disable the interaction prior this architecture is "
                "built to test."
            )
        for key, trailing in REQUIRED_HP_OUT_KEYS:
            value = hp_out.get(key)
            expected = (batch_size, seq_len, *trailing)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"VC v4 requires hp_out['{key}'] as a tensor with shape "
                    f"{expected}; got {type(value).__name__}."
                )
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"VC v4 expects hp_out['{key}'] with shape {expected}, "
                    f"got {tuple(value.shape)}."
                )
        # HumanPose emits correctly shaped all-zero pose tensors when its body
        # model is unavailable.  That trains and converges while answering this
        # experiment's question with an empty prior, so reject it once per
        # process rather than paying a device sync on every step.
        if not self._pose_stream_content_checked:
            self._pose_stream_content_checked = True
            for key in ("pred_full_pose_6d", "pred_joints_local"):
                if not bool(hp_out[key].any()):
                    raise ValueError(
                        f"VC v4 received an all-zero hp_out['{key}']. The "
                        "HumanPose stage produced no usable pose; fix it "
                        "instead of training the interaction prior on zeros."
                    )
        expected_hand = (batch_size, seq_len, 2, 3)
        for key in HAND_POSITION_HP_OUT_KEYS:
            value = hp_out.get(key)
            if isinstance(value, torch.Tensor) and tuple(value.shape) == expected_hand:
                return value
        raise ValueError(
            "VC v4 requires one of "
            f"{HAND_POSITION_HP_OUT_KEYS} in hp_out with shape {expected_hand}."
        )

    def _prepare_pose_streams(self, hp_out, device, dtype, batch_size, seq_len):
        hand_pos = self._validate_hp_out(hp_out, batch_size, seq_len)

        def _cast(key):
            return hp_out[key].to(device=device, dtype=dtype)

        pose_6d = _cast("pred_full_pose_6d")
        joint_pos = _cast("pred_joints_local")
        root_vel = _cast("root_vel_pred")
        root_trans = _cast("root_trans_pred")
        hand_pos = hand_pos.to(device=device, dtype=dtype)

        # Smooth in the world frame, then rotate: the heading changes within a
        # window, so filtering already-rotated signals would mix frames.
        window = self.boundary_smooth_window
        joint_vel = causal_moving_average(self._causal_velocity(joint_pos), window)
        joint_acc = causal_moving_average(
            self._causal_acceleration(joint_pos), window
        )
        hand_vel = causal_moving_average(self._causal_velocity(hand_pos), window)
        hand_acc = causal_moving_average(
            self._causal_acceleration(hand_pos), window
        )

        if self.heading_normalize:
            yaw = self._heading_frame(pose_6d)
            rotation = rotation_6d_to_matrix(pose_6d.reshape(-1, 6)).reshape(
                batch_size, seq_len, 24, 3, 3
            )
            rotation = torch.matmul(
                yaw.transpose(-1, -2).unsqueeze(2),
                rotation,
            )
            pose_6d = matrix_to_rotation_6d(rotation.reshape(-1, 3, 3)).reshape(
                batch_size, seq_len, 24, 6
            )
            joint_vel = self._rotate_vectors(joint_vel, yaw)
            joint_acc = self._rotate_vectors(joint_acc, yaw)
            hand_vel = self._rotate_vectors(hand_vel, yaw)
            hand_acc = self._rotate_vectors(hand_acc, yaw)
            root_vel = self._rotate_vectors(root_vel, yaw)

        return {
            "pose_6d": pose_6d,
            "joint_vel": joint_vel,
            "joint_acc": joint_acc,
            "root_vel": root_vel,
            "root_trans": root_trans,
            "hand_pos": hand_pos,
            "hand_vel": hand_vel,
            "hand_acc": hand_acc,
        }

    def _encode_body_groups(self, pose_streams):
        pose_6d = pose_streams["pose_6d"]
        joint_vel = pose_streams["joint_vel"]
        joint_acc = pose_streams["joint_acc"]
        root_vel = self.root_vel_norm(pose_streams["root_vel"])
        batch_size, seq_len = pose_6d.shape[:2]

        group_features = []
        for name, encoder in self.group_encoders.items():
            indices = self.body_group_indices[name]
            parts = [
                self.group_pose_norms[name](
                    pose_6d[:, :, indices, :].reshape(batch_size, seq_len, -1)
                ),
                self.group_vel_norms[name](
                    joint_vel[:, :, indices, :].reshape(batch_size, seq_len, -1)
                ),
            ]
            if self.boundary_use_joint_acc:
                parts.append(
                    self.group_acc_norms[name](
                        joint_acc[:, :, indices, :].reshape(batch_size, seq_len, -1)
                    )
                )
            parts.append(root_vel)
            group_out, _ = encoder(torch.cat(parts, dim=-1))
            group_features.append(group_out)

        boundary_input = self.boundary_input_norm(
            torch.cat(group_features, dim=-1)
        )
        boundary_out, _ = self.boundary_rnn(boundary_input)
        return self.boundary_hidden_norm(self.boundary_fusion(boundary_out))

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
                batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype
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
            data_dict.get("contact_init"), batch_size, 3, device, dtype
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

        pose_streams = self._prepare_pose_streams(
            hp_out if hp_out is not None else data_dict.get("hp_out"),
            device,
            dtype,
            batch_size,
            seq_len,
        )
        boundary_out = self._encode_body_groups(pose_streams)
        hand_dyn_feat = torch.cat(
            (
                pose_streams["hand_vel"].reshape(batch_size, seq_len, -1),
                pose_streams["hand_acc"].reshape(batch_size, seq_len, -1),
            ),
            dim=-1,
        )

        obj_move_input = torch.cat((hand_imu_feat, obj_imu, boundary_out), dim=-1)
        obj_move_logits = self.obj_move_net((obj_move_input, contact_init))
        obj_move_prob = torch.sigmoid(obj_move_logits)

        # The IMU and predicted-velocity channels stay in the world frame, as in
        # v3, so v4 remains comparable to it there; only the pose-derived
        # channels are heading-normalized.  Normalizing the IMU evidence too is
        # a separate, larger change.
        hand_contact_input = torch.cat(
            (
                hand_imu_feat,
                obj_imu,
                hand_vel_flat,
                obj_vel,
                boundary_out,
                hand_dyn_feat,
            ),
            dim=-1,
        )
        hand_contact_logits = self.hand_contact_net(
            (hand_contact_input, contact_init)
        )
        hand_contact_prob_cond = torch.sigmoid(hand_contact_logits)
        hand_contact_prob = obj_move_prob * hand_contact_prob_cond

        endpoint_hidden, _ = self.contact_endpoint_rnn(
            self.contact_endpoint_input_norm(boundary_out)
        )
        endpoint_logits = self.contact_endpoint_head(endpoint_hidden)
        endpoint_prob = torch.sigmoid(endpoint_logits)
        anticipation_logits = self.contact_anticipation_head(
            self.contact_anticipation_norm(boundary_out)
        )

        return {
            "pred_hand_glb_vel": hand_vel,
            "pred_hand_glb_vel_flat": hand_vel_flat,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": torch.cat(
                (hand_contact_logits, obj_move_logits), dim=-1
            ),
            "pred_hand_contact_prob": torch.cat(
                (hand_contact_prob, obj_move_prob), dim=-1
            ),
            "pred_obj_move_prob": obj_move_prob,
            "pred_hand_contact_prob_cond": hand_contact_prob_cond,
            "pred_hand_contact_logits_cond": hand_contact_logits,
            # The downstream-consumed marginal needs its own BCE: conditional
            # supervision alone cannot calibrate the product.
            "pred_hand_contact_prob_marginal": hand_contact_prob,
            "pred_contact_endpoint_logits": endpoint_logits,
            "pred_contact_endpoint_prob": endpoint_prob,
            "pred_contact_from_endpoints_prob": recover_contact_state_from_endpoints(
                endpoint_prob,
                contact_init,
            ),
            "pred_contact_anticipation_logits": anticipation_logits,
            "pred_contact_anticipation_prob": torch.sigmoid(anticipation_logits),
        }

    def validate_checkpoint_state_dict(self, state_dict):
        """Reject v3/geometry-era v4 checkpoints and horizon mismatches."""
        if not isinstance(state_dict, dict):
            raise ValueError("VC v4 checkpoint must contain a state_dict.")
        normalized = {
            str(key[7:] if str(key).startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        version = normalized.get("_v4_arch_version")
        if (
            not isinstance(version, torch.Tensor)
            or version.numel() != 1
            or int(version.item()) != self.ARCH_VERSION
        ):
            raise ValueError(
                "Checkpoint is not marked as VC v4 supervised pose prior "
                "(heading-normalized bidirectional boundary + endpoint and "
                "anticipation heads); retrain or select the matching "
                "--velocity_contact_arch."
            )
        horizon = normalized.get("_v4_anticipation_horizon")
        if (
            isinstance(horizon, torch.Tensor)
            and horizon.numel() == 1
            and int(horizon.item()) != self.anticipation_horizon
        ):
            raise ValueError(
                "Checkpoint anticipation horizon "
                f"{int(horizon.item())} does not match the configured "
                f"v4_anticipation_horizon={self.anticipation_horizon}; the "
                "head would be fine-tuned against a different target."
            )
        IMUHOIConditionalVelocityContactModule.validate_checkpoint_state_dict(
            self,
            normalized,
        )

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        output = IMUHOIConditionalVelocityContactModule.empty_output(
            batch_size, seq_len, device
        )
        output.update(
            {
                "pred_hand_contact_prob_marginal": torch.zeros(
                    batch_size, seq_len, 2, device=device
                ),
                "pred_contact_endpoint_logits": torch.zeros(
                    batch_size, seq_len, 6, device=device
                ),
                "pred_contact_endpoint_prob": torch.zeros(
                    batch_size, seq_len, 6, device=device
                ),
                "pred_contact_from_endpoints_prob": torch.zeros(
                    batch_size, seq_len, 3, device=device
                ),
                "pred_contact_anticipation_logits": torch.zeros(
                    batch_size, seq_len, 6, device=device
                ),
                "pred_contact_anticipation_prob": torch.zeros(
                    batch_size, seq_len, 6, device=device
                ),
            }
        )
        return output


VelocityContactModule = V4VelocityContactModule

__all__ = [
    "DEFAULT_ANTICIPATION_HORIZON",
    "V4VelocityContactModule",
    "VelocityContactModule",
    "causal_moving_average",
    "heading_rotation_matrix",
]
