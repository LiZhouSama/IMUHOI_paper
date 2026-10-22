"""VC v5: v4's supervised pose prior plus rotation-invariant coupling evidence.

v4 gives the hand-contact net world-frame IMU channels, predicted velocities
and pose-prior features.  None of them measures directly what a grasp is
physically: the forearm and the object moving as one rigid body, and, for a
two-handed hold, the two hands keeping a fixed span.  Measured on the BEHAVE
test sequences (2026-09-30, grouped 5-fold CV over the online
v2_armrot_biclose dump, small GBDT): such features alone reach per-sequence
hand F1 0.803 against VC-v4's 0.784, and stacked on VC-v4's own probability
0.842, so they carry information the pose prior does not.

Added evidence, all rotation-invariant scalars (they cannot encode the
sequence's absolute heading, which a world-frame channel can be memorized by):

* per hand: rotation rate of ``B = R_obj^T R_forearm`` (about zero for a rigid
  grip while both rotate), forearm rotation rate, ``|a_forearm - a_obj|``;
* shared: object rotation rate, ``|a_obj|``, palm span ``|p_L - p_R|`` and its
  rate, relative palm speed ``|v_L - v_R|``, forearm-forearm relative rotation
  rate.

Each signal is averaged over ``[t-5, t+5]`` and ``[t-20, t+10]`` (never past
the 10-frame online lookahead) and also given as ``log1p``.  The hand-contact
net reads all of them; the object-move net reads the shared object terms.

``v5_heading_normalize_imu`` additionally expresses the contact nets' IMU and
predicted-velocity channels in the pelvis heading frame (yaw of the root IMU),
as v4 already does for the pose streams.  The velocity nets are untouched.

Initializer robustness (training only, default off).  Every window starts from
initializers that training takes from GT but the online protocol takes from
published predictions: on BEHAVE the object-velocity init (3-frame difference
of the published trajectory) is 0.23-0.35 m/s off, as large as the object's
speed, and VC's published velocity error (0.23-0.30 m/s) is essentially that
init error carried through the window.  ``v5_init_noise_prob`` perturbs a
fraction of the training windows' initializers so the nets learn to weigh them
against the IMU evidence: Gaussian noise of ``v5_obj_vel_init_noise_std`` /
``v5_hand_vel_init_noise_std`` (m/s per axis) on the velocity inits, and each
contact-init channel flipped with ``v5_contact_init_flip_prob``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from .base import RNNWithInit
from .velocity_contact import recover_contact_state_from_endpoints
from .velocity_contact_ablation import IMUHOIConditionalVelocityContactModule
from .velocity_contact_v4 import V4VelocityContactModule, heading_rotation_matrix


# (past, future) frames of the centred averages; future <= the online lookahead.
COUPLING_WINDOWS = ((5, 5), (20, 10))
PER_HAND_SIGNALS = 3
SHARED_OBJECT_SIGNALS = 2
SHARED_BIMANUAL_SIGNALS = 4


def rotation_angle(relative: torch.Tensor) -> torch.Tensor:
    """Angle (rad) of relative rotations ``[..., 3, 3]``."""
    cos = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    return torch.acos(cos.clamp(-1.0, 1.0))


def rotation_rate(rotation: torch.Tensor, fps: float) -> torch.Tensor:
    """Per-frame rotation speed (rad/s) of ``[B, T, 3, 3]``; frame 0 copies frame 1."""
    if rotation.shape[1] < 2:
        return rotation.new_zeros(rotation.shape[:2])
    step = rotation_angle(rotation[:, :-1].transpose(-1, -2) @ rotation[:, 1:]) * fps
    return torch.cat((step[:, :1], step), dim=1)


def centred_average(value: torch.Tensor, past: int, future: int) -> torch.Tensor:
    """Average ``[B, T, C]`` over ``[t-past, t+future]`` with replicated edges."""
    flat = value.transpose(1, 2)
    padded = F.pad(flat, (int(past), int(future)), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=int(past) + int(future) + 1, stride=1).transpose(1, 2)


def multiscale(value: torch.Tensor) -> torch.Tensor:
    """Centred averages over COUPLING_WINDOWS, raw and log1p-compressed."""
    parts = [centred_average(value, past, future) for past, future in COUPLING_WINDOWS]
    parts = parts + [torch.log1p(part.clamp_min(0.0)) for part in parts]
    return torch.cat(parts, dim=-1)


class V5VelocityContactModule(V4VelocityContactModule):
    """v4 plus rotation-invariant forearm/object and bimanual coupling features."""

    ARCH_VERSION = 52

    def __init__(self, cfg):
        super().__init__(cfg)
        self.arch = "v5"
        self.heading_normalize_imu = bool(getattr(cfg, "v5_heading_normalize_imu", False))
        scales = 2 * len(COUPLING_WINDOWS)
        self.object_coupling_dim = SHARED_OBJECT_SIGNALS * scales
        self.hand_coupling_dim = (
            2 * PER_HAND_SIGNALS + SHARED_OBJECT_SIGNALS + SHARED_BIMANUAL_SIGNALS
        ) * scales

        hidden_dim = int(getattr(cfg, "velocity_hidden_dim", 128))
        contact_hidden = max(hidden_dim // 2, 32)
        dropout = float(getattr(cfg, "velocity_dropout", 0.2))

        self.obj_move_input_dim = self.obj_move_input_dim + self.object_coupling_dim
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
        self.hand_contact_input_dim = self.hand_contact_input_dim + self.hand_coupling_dim
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
        self.init_noise_prob = float(getattr(cfg, "v5_init_noise_prob", 0.0) or 0.0)
        self.obj_vel_init_noise_std = float(getattr(cfg, "v5_obj_vel_init_noise_std", 0.0) or 0.0)
        self.hand_vel_init_noise_std = float(getattr(cfg, "v5_hand_vel_init_noise_std", 0.0) or 0.0)
        self.contact_init_flip_prob = float(getattr(cfg, "v5_contact_init_flip_prob", 0.0) or 0.0)
        self.register_buffer(
            "_v5_arch_version",
            torch.tensor(self.ARCH_VERSION, dtype=torch.int64),
        )
        self.register_buffer(
            "_v5_heading_normalize_imu",
            torch.tensor(int(self.heading_normalize_imu), dtype=torch.int64),
        )

    def _perturb_inits(self, hand_vel_init, obj_vel_init, contact_init):
        """Training-only initializer corruption on a random subset of windows."""
        if not self.training or self.init_noise_prob <= 0.0:
            return hand_vel_init, obj_vel_init, contact_init
        batch_size = obj_vel_init.shape[0]
        chosen = torch.rand(batch_size, 1, device=obj_vel_init.device) < self.init_noise_prob
        if self.obj_vel_init_noise_std > 0.0:
            noise = torch.randn_like(obj_vel_init) * self.obj_vel_init_noise_std
            obj_vel_init = obj_vel_init + noise * chosen.to(noise.dtype)
        if self.hand_vel_init_noise_std > 0.0:
            noise = torch.randn_like(hand_vel_init) * self.hand_vel_init_noise_std
            hand_vel_init = hand_vel_init + noise * chosen.to(noise.dtype)
        if self.contact_init_flip_prob > 0.0:
            flip = (torch.rand_like(contact_init) < self.contact_init_flip_prob) & chosen
            contact_init = torch.where(flip, 1.0 - contact_init, contact_init)
        return hand_vel_init, obj_vel_init, contact_init

    def _coupling_features(self, human_imu_denorm, obj_imu, hand_pos):
        """Rotation-invariant coupling signals; returns (object, hand) feature sets."""
        batch_size, seq_len = human_imu_denorm.shape[:2]
        fps = float(self.fps)
        # Fixed evidence: acos() has an infinite derivative at zero rotation, so
        # nothing (e.g. a jointly fine-tuned HumanPose) may backpropagate here.
        with torch.autocast(device_type=human_imu_denorm.device.type, enabled=False):
            imu = human_imu_denorm.detach().float()
            obj = obj_imu.detach().float()
            hand_pos = hand_pos.detach()
            sensors = [self.left_hand_sensor, self.right_hand_sensor]
            forearm_rot = rotation_6d_to_matrix(imu[:, :, sensors, 3:9].reshape(-1, 6)).reshape(
                batch_size, seq_len, 2, 3, 3
            )
            forearm_acc = imu[:, :, sensors, :3]
            obj_rot = rotation_6d_to_matrix(obj[..., 3:9].reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
            obj_acc = obj[..., :3]

            obj_rate = rotation_rate(obj_rot, fps)
            object_signals = torch.stack((obj_rate, obj_acc.norm(dim=-1)), dim=-1)

            per_hand = []
            for hand in range(2):
                relative = obj_rot.transpose(-1, -2) @ forearm_rot[:, :, hand]
                per_hand.append(
                    torch.stack(
                        (
                            rotation_rate(relative, fps),
                            rotation_rate(forearm_rot[:, :, hand], fps),
                            (forearm_acc[:, :, hand] - obj_acc).norm(dim=-1),
                        ),
                        dim=-1,
                    )
                )

            palms = hand_pos.float()
            span = (palms[:, :, 0] - palms[:, :, 1]).norm(dim=-1)
            span_rate = torch.zeros_like(span)
            palm_rel_speed = torch.zeros_like(span)
            if seq_len > 1:
                span_rate[:, 1:] = (span[:, 1:] - span[:, :-1]).abs() * fps
                span_rate[:, 0] = span_rate[:, 1]
                relative_disp = (palms[:, 1:, 0] - palms[:, :-1, 0]) - (palms[:, 1:, 1] - palms[:, :-1, 1])
                palm_rel_speed[:, 1:] = relative_disp.norm(dim=-1) * fps
                palm_rel_speed[:, 0] = palm_rel_speed[:, 1]
            forearm_pair = forearm_rot[:, :, 0].transpose(-1, -2) @ forearm_rot[:, :, 1]
            bimanual_signals = torch.stack(
                (span, span_rate, palm_rel_speed, rotation_rate(forearm_pair, fps)),
                dim=-1,
            )

            object_features = multiscale(object_signals)
            hand_features = multiscale(
                torch.cat(per_hand + [object_signals, bimanual_signals], dim=-1)
            )
        return object_features, hand_features

    def _heading_normalize_contact_inputs(self, human_imu_denorm, obj_imu, hand_vel_flat, obj_vel):
        """Express the contact nets' IMU/velocity channels in the root IMU's heading frame."""
        batch_size, seq_len = human_imu_denorm.shape[:2]
        dtype = human_imu_denorm.dtype
        with torch.autocast(device_type=human_imu_denorm.device.type, enabled=False):
            imu = human_imu_denorm.float()
            root_rot = rotation_6d_to_matrix(imu[:, :, 0, 3:9].reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
            yaw = heading_rotation_matrix(root_rot)  # row-vector world -> heading operator
            sensors = [self.left_hand_sensor, self.right_hand_sensor]
            hand_acc = imu[:, :, sensors, :3] @ yaw  # [B,T,2,3] rows times [B,T,3,3]
            hand_rot = rotation_6d_to_matrix(imu[:, :, sensors, 3:9].reshape(-1, 6)).reshape(
                batch_size, seq_len, 2, 3, 3
            )
            hand_rot = yaw.transpose(-1, -2).unsqueeze(2) @ hand_rot
            hand_imu_feat = torch.cat(
                (hand_acc, matrix_to_rotation_6d(hand_rot.reshape(-1, 3, 3)).reshape(batch_size, seq_len, 2, 6)),
                dim=-1,
            ).reshape(batch_size, seq_len, -1)
            obj = obj_imu.float()
            obj_rot = rotation_6d_to_matrix(obj[..., 3:9].reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
            obj_rot = yaw.transpose(-1, -2) @ obj_rot
            obj_feat = torch.cat(
                (
                    (obj[..., :3].unsqueeze(-2) @ yaw).squeeze(-2),
                    matrix_to_rotation_6d(obj_rot.reshape(-1, 3, 3)).reshape(batch_size, seq_len, 6),
                ),
                dim=-1,
            )
            hand_vel = (hand_vel_flat.float().reshape(batch_size, seq_len, 2, 3) @ yaw).reshape(
                batch_size, seq_len, 6
            )
            obj_vel_h = (obj_vel.float().unsqueeze(-2) @ yaw).squeeze(-2)
        return (
            hand_imu_feat.to(dtype),
            obj_feat.to(dtype),
            hand_vel.to(dtype),
            obj_vel_h.to(dtype),
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
            obj_imu = torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
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
            data_dict.get("hand_vel_glb_init"), batch_size, 6, device, dtype, "hand_vel_glb_init"
        )
        obj_vel_init = self._prepare_init(
            data_dict.get("obj_vel_init"), batch_size, 3, device, dtype, "obj_vel_init"
        )
        contact_init = self._prepare_init_vector(
            data_dict.get("contact_init"), batch_size, 3, device, dtype
        )
        hand_vel_init, obj_vel_init, contact_init = self._perturb_inits(
            hand_vel_init, obj_vel_init, contact_init
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
        object_coupling, hand_coupling = self._coupling_features(
            human_imu_denorm, obj_imu, pose_streams["hand_pos"]
        )
        object_coupling = object_coupling.to(dtype)
        hand_coupling = hand_coupling.to(dtype)

        if self.heading_normalize_imu:
            hand_imu_feat, obj_feat, hand_vel_feat, obj_vel_feat = self._heading_normalize_contact_inputs(
                human_imu_denorm, obj_imu, hand_vel_flat, obj_vel
            )
        else:
            hand_imu_feat = human_imu_denorm[
                :, :, [self.left_hand_sensor, self.right_hand_sensor], :
            ].reshape(batch_size, seq_len, -1)
            obj_feat, hand_vel_feat, obj_vel_feat = obj_imu, hand_vel_flat, obj_vel

        obj_move_input = torch.cat((hand_imu_feat, obj_feat, boundary_out, object_coupling), dim=-1)
        obj_move_logits = self.obj_move_net((obj_move_input, contact_init))
        obj_move_prob = torch.sigmoid(obj_move_logits)

        hand_contact_input = torch.cat(
            (
                hand_imu_feat,
                obj_feat,
                hand_vel_feat,
                obj_vel_feat,
                boundary_out,
                hand_dyn_feat,
                hand_coupling,
            ),
            dim=-1,
        )
        hand_contact_logits = self.hand_contact_net((hand_contact_input, contact_init))
        hand_contact_prob_cond = torch.sigmoid(hand_contact_logits)
        hand_contact_prob = obj_move_prob * hand_contact_prob_cond

        endpoint_hidden, _ = self.contact_endpoint_rnn(self.contact_endpoint_input_norm(boundary_out))
        endpoint_logits = self.contact_endpoint_head(endpoint_hidden)
        endpoint_prob = torch.sigmoid(endpoint_logits)
        anticipation_logits = self.contact_anticipation_head(
            self.contact_anticipation_norm(boundary_out)
        )

        return {
            "pred_hand_glb_vel": hand_vel,
            "pred_hand_glb_vel_flat": hand_vel_flat,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": torch.cat((hand_contact_logits, obj_move_logits), dim=-1),
            "pred_hand_contact_prob": torch.cat((hand_contact_prob, obj_move_prob), dim=-1),
            "pred_obj_move_prob": obj_move_prob,
            "pred_hand_contact_prob_cond": hand_contact_prob_cond,
            "pred_hand_contact_logits_cond": hand_contact_logits,
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
        """Reject v4 (and differently configured v5) checkpoints."""
        if not isinstance(state_dict, dict):
            raise ValueError("VC v5 checkpoint must contain a state_dict.")
        normalized = {
            str(key[7:] if str(key).startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        version = normalized.get("_v5_arch_version")
        if (
            not isinstance(version, torch.Tensor)
            or version.numel() != 1
            or int(version.item()) != self.ARCH_VERSION
        ):
            raise ValueError(
                "Checkpoint is not marked as VC v5 (v4 pose prior + coupling "
                "features); retrain or select the matching --velocity_contact_arch."
            )
        heading = normalized.get("_v5_heading_normalize_imu")
        if (
            isinstance(heading, torch.Tensor)
            and heading.numel() == 1
            and bool(int(heading.item())) != self.heading_normalize_imu
        ):
            raise ValueError(
                "Checkpoint v5_heading_normalize_imu="
                f"{bool(int(heading.item()))} does not match the configured "
                f"{self.heading_normalize_imu}."
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
                f"v4_anticipation_horizon={self.anticipation_horizon}."
            )
        IMUHOIConditionalVelocityContactModule.validate_checkpoint_state_dict(self, normalized)


VelocityContactModule = V5VelocityContactModule

__all__ = ["V5VelocityContactModule", "VelocityContactModule"]
