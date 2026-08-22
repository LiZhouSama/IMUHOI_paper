"""
VelocityContactModule: predict hand/object velocity and contact cues.
"""
import math

import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

from .base import RNNWithInit
from .online import (
    append_stream_data,
    infer_batch_seq,
    normalize_inference_mode,
    resolve_online_window,
    slice_time_dict,
    take_latest_frame,
    TimeDictAccumulator,
    update_data_inits_from_history,
)
from configs import FRAME_RATE, _SENSOR_NAMES
from utils.human_pose import select_hand_anchor_positions
from utils.utils import _central_diff, _smooth_acceleration


VELOCITY_CONTACT_ARCH_CHOICES = (
    "current",
    "legacy",
    "v2",
    "v3",
    "v4",
    "imuhoi",
    "imuhoi_cond",
    "imuhoi_cond_bi",
    "imuhoi_cond_geom",
    "imuhoi_cond_boundary",
    "imuhoi_cond_boundary_endpoint",
)


def normalize_velocity_contact_arch(arch):
    """Normalize and validate the RNN velocity-contact architecture name."""
    normalized = str(arch or "current").strip().lower()
    if normalized not in VELOCITY_CONTACT_ARCH_CHOICES:
        choices = ", ".join(VELOCITY_CONTACT_ARCH_CHOICES)
        raise ValueError(f"velocity_contact_arch must be one of: {choices}; got {arch!r}")
    return normalized


def build_velocity_contact_module(cfg, velocity_contact_arch=None):
    """Build the configured RNN velocity-contact implementation."""
    arch = normalize_velocity_contact_arch(
        velocity_contact_arch or getattr(cfg, "velocity_contact_arch", "current")
    )
    if arch == "legacy":
        from .velocity_contact_legacy import LegacyVelocityContactModule

        return LegacyVelocityContactModule(cfg)
    if arch == "v2":
        from .velocity_contact_v2 import V2VelocityContactModule

        return V2VelocityContactModule(cfg)
    if arch == "v3":
        from .velocity_contact_v3 import V3VelocityContactModule

        return V3VelocityContactModule(cfg)
    if arch == "v4":
        from .velocity_contact_v4 import V4VelocityContactModule

        return V4VelocityContactModule(cfg)
    if arch == "imuhoi":
        from .velocity_contact_IMUHOI import IMUHOIVelocityContactModule

        return IMUHOIVelocityContactModule(cfg)
    if arch.startswith("imuhoi_cond"):
        from .velocity_contact_ablation import (
            IMUHOIConditionalVelocityContactModule,
        )

        return IMUHOIConditionalVelocityContactModule(cfg, arch=arch)
    return VelocityContactModule(cfg)


# Contact-segment endpoint channels are ordered as
# (lhand_start, lhand_end, rhand_start, rhand_end, obj_start, obj_end);
# rasterized/interior channels follow (lhand, rhand, obj).
CONTACT_SEGMENT_CHANNEL_NAMES = ("lhand", "rhand", "obj")


def find_heatmap_peaks(prob, threshold=0.3, radius=5):
    """Find local maxima of a 1D endpoint heatmap with greedy NMS.

    Args:
        prob: [T] probability tensor.
        threshold: minimum peak height.
        radius: NMS radius in frames; lower peaks within it are suppressed.
    Returns:
        list of (index, value) sorted by index.
    """
    length = int(prob.shape[0])
    if length == 0:
        return []
    left = torch.cat((prob.new_full((1,), -1.0), prob[:-1]))
    right = torch.cat((prob[1:], prob.new_full((1,), -1.0)))
    is_peak = (prob >= left) & (prob > right) & (prob >= threshold)
    candidates = torch.nonzero(is_peak).flatten().tolist()
    candidates.sort(key=lambda index: float(prob[index]), reverse=True)
    kept = []
    for index in candidates:
        if any(abs(index - kept_index) <= radius for kept_index, _ in kept):
            continue
        kept.append((index, float(prob[index])))
    kept.sort(key=lambda item: item[0])
    return kept


def recover_contact_state_from_endpoints(
    endpoint_prob: torch.Tensor,
    initial_contact: torch.Tensor,
) -> torch.Tensor:
    """Differentiably recover dense contact state from start/end probabilities.

    Endpoint channels are ordered as ``(start, end)`` for each contact
    channel. ``initial_contact`` is the state at frame 0. A start at frame
    ``t`` activates frame ``t``; an end at frame ``t`` deactivates frame
    ``t + 1``, because the endpoint label marks the last active frame.
    """
    if endpoint_prob.dim() != 3 or endpoint_prob.shape[-1] % 2 != 0:
        raise ValueError(
            "endpoint_prob must be [B, T, 2*C], "
            f"got {tuple(endpoint_prob.shape)}"
        )
    batch_size, seq_len, twice_channels = endpoint_prob.shape
    num_channels = twice_channels // 2
    initial_contact = torch.as_tensor(
        initial_contact,
        device=endpoint_prob.device,
        dtype=endpoint_prob.dtype,
    )
    if initial_contact.dim() == 1:
        initial_contact = initial_contact.unsqueeze(0)
    if initial_contact.shape[0] == 1 and batch_size > 1:
        initial_contact = initial_contact.expand(batch_size, -1)
    if initial_contact.shape != (batch_size, num_channels):
        raise ValueError(
            "initial_contact must be [B, C] matching endpoint channels, "
            f"got {tuple(initial_contact.shape)} for B={batch_size}, C={num_channels}"
        )
    if seq_len == 0:
        return endpoint_prob.new_zeros(batch_size, 0, num_channels)

    endpoint_pairs = endpoint_prob.reshape(batch_size, seq_len, num_channels, 2)
    state = initial_contact.clamp(0.0, 1.0)
    states = [state]
    for frame in range(1, seq_len):
        start_prob = endpoint_pairs[:, frame, :, 0]
        previous_end_prob = endpoint_pairs[:, frame - 1, :, 1]
        state = (
            state * (1.0 - previous_end_prob)
            + (1.0 - state) * start_prob
        ).clamp(0.0, 1.0)
        states.append(state)
    return torch.stack(states, dim=1)


def _pair_endpoint_peaks(
    start_peaks,
    end_peaks,
    seq_len,
    allow_open_segments=True,
    initial_active=False,
    initial_confidence=None,
):
    """Greedily pair start peaks with the next end peak in time order.

    An end peak with no active start opens a segment from the previous
    segment's end + 1 (the stream entered the window mid-contact); a trailing
    start with no end opens a segment running to the last frame (contact
    still ongoing). ``initial_active`` explicitly represents contact already
    active at frame 0, including the endpoint-free case where the same segment
    spans the whole window. Open sides carry a None endpoint confidence.

    Returns:
        list of (start, end, start_confidence_or_None,
        end_confidence_or_None, initial_confidence_or_None).
    """
    events = [(index, 0, value) for index, value in start_peaks]
    events += [(index, 1, value) for index, value in end_peaks]
    events.sort(key=lambda item: (item[0], item[1]))

    segments = []
    active = (0, None, initial_confidence) if initial_active else None
    for index, kind, value in events:
        if kind == 0:
            if active is None:
                active = (index, value, None)
            elif active[0] == 0 and active[1] is None and index == 0:
                # Frame 0 can be both active and a genuine in-window start.
                active = (0, value, active[2])
        else:
            if active is not None:
                segments.append((active[0], index, active[1], value, active[2]))
                active = None
            elif allow_open_segments:
                open_start = segments[-1][1] + 1 if segments else 0
                if open_start <= index:
                    segments.append((open_start, index, None, value, None))
    if active is not None and allow_open_segments:
        segments.append((active[0], seq_len - 1, active[1], None, active[2]))
    return segments


def _segment_confidence(
    start_confidence,
    end_confidence,
    interior_confidence,
    initial_confidence=None,
    eps=1e-6,
):
    """Geometric mean of the available confidence terms."""
    terms = [
        term
        for term in (
            start_confidence,
            end_confidence,
            interior_confidence,
            initial_confidence,
        )
        if term is not None
    ]
    if not terms:
        return 0.0
    log_sum = sum(math.log(max(float(term), eps)) for term in terms)
    return float(math.exp(log_sum / len(terms)))


def decode_contact_segments(
    endpoint_prob,
    interior_prob=None,
    peak_threshold=0.3,
    segment_threshold=0.25,
    peak_radius=5,
    allow_open_segments=True,
    initial_contact=None,
    initial_contact_threshold=0.5,
):
    """Decode start/end endpoint heatmaps into contact segments with confidence.

    Args:
        endpoint_prob: [B, T, 6] heatmaps ordered as
            (lhand_start, lhand_end, rhand_start, rhand_end, obj_start, obj_end).
        interior_prob: optional [B, T, 3] per-frame contact probabilities
            (lhand, rhand, obj) used as interior actionness in the confidence.
        peak_threshold: minimum endpoint peak height.
        segment_threshold: minimum segment confidence (geometric mean of
            available initial state, endpoint peaks and mean interior probability).
        peak_radius: NMS radius for peak extraction, in frames.
        allow_open_segments: emit segments whose start/end lies outside the
            observed range (needed for streaming/ongoing contact).
        initial_contact: optional [B, 3] frame-0 contact state/probability.
            An active channel seeds an open segment at frame 0, so a segment
            spanning the whole window can be decoded without endpoint peaks.
        initial_contact_threshold: scalar or [3] per-channel thresholds for
            treating initial_contact as active.
    Returns:
        segments: per-batch list of dicts with keys
            channel, channel_name, start, end, confidence,
            start_confidence, end_confidence (open sides are None).
        raster: [B, T, 3] per-frame probabilities rasterized from segments;
            frames inside a segment take its confidence, overlaps take max.
    """
    if endpoint_prob.dim() != 3 or endpoint_prob.shape[-1] % 2 != 0:
        raise ValueError(f"endpoint_prob must be [B, T, 2*C], got {tuple(endpoint_prob.shape)}")
    batch_size, seq_len, twice_channels = endpoint_prob.shape
    num_channels = twice_channels // 2
    initial_thresholds = torch.as_tensor(
        initial_contact_threshold,
        device=endpoint_prob.device,
        dtype=endpoint_prob.dtype,
    )
    if initial_thresholds.dim() == 0:
        initial_thresholds = initial_thresholds.expand(num_channels)
    if initial_thresholds.shape != (num_channels,):
        raise ValueError(
            "initial_contact_threshold must be scalar or [C], "
            f"got {tuple(initial_thresholds.shape)} for C={num_channels}"
        )
    if initial_contact is not None:
        initial_contact = torch.as_tensor(
            initial_contact,
            device=endpoint_prob.device,
            dtype=endpoint_prob.dtype,
        )
        if initial_contact.dim() == 1:
            initial_contact = initial_contact.unsqueeze(0)
        if initial_contact.shape[0] == 1 and batch_size > 1:
            initial_contact = initial_contact.expand(batch_size, -1)
        if initial_contact.shape != (batch_size, num_channels):
            raise ValueError(
                "initial_contact must be [B, C] matching endpoint channels, "
                f"got {tuple(initial_contact.shape)} for B={batch_size}, C={num_channels}"
            )
        initial_contact = initial_contact.clamp(0.0, 1.0)
    raster = endpoint_prob.new_zeros(batch_size, seq_len, num_channels)
    segments = []
    for b in range(batch_size):
        batch_segments = []
        for channel in range(num_channels):
            start_peaks = find_heatmap_peaks(
                endpoint_prob[b, :, 2 * channel], threshold=peak_threshold, radius=peak_radius
            )
            end_peaks = find_heatmap_peaks(
                endpoint_prob[b, :, 2 * channel + 1], threshold=peak_threshold, radius=peak_radius
            )
            initial_conf = (
                float(initial_contact[b, channel])
                if initial_contact is not None
                else None
            )
            paired = _pair_endpoint_peaks(
                start_peaks,
                end_peaks,
                seq_len,
                allow_open_segments,
                initial_active=(
                    initial_conf is not None
                    and initial_conf >= float(initial_thresholds[channel])
                ),
                initial_confidence=initial_conf,
            )
            for seg_start, seg_end, start_conf, end_conf, init_conf in paired:
                interior_conf = None
                is_fully_open_initial_segment = (
                    init_conf is not None
                    and start_conf is None
                    and end_conf is None
                    and seg_start == 0
                    and seg_end == seq_len - 1
                )
                if interior_prob is not None and not is_fully_open_initial_segment:
                    interior_conf = float(interior_prob[b, seg_start:seg_end + 1, channel].mean())
                confidence = _segment_confidence(
                    start_conf,
                    end_conf,
                    interior_conf,
                    initial_confidence=init_conf,
                )
                if confidence < segment_threshold:
                    continue
                batch_segments.append({
                    "channel": channel,
                    "channel_name": (
                        CONTACT_SEGMENT_CHANNEL_NAMES[channel]
                        if channel < len(CONTACT_SEGMENT_CHANNEL_NAMES)
                        else str(channel)
                    ),
                    "start": int(seg_start),
                    "end": int(seg_end),
                    "confidence": confidence,
                    "start_confidence": start_conf,
                    "end_confidence": end_conf,
                    "initial_confidence": init_conf,
                })
                raster[b, seg_start:seg_end + 1, channel] = raster[
                    b, seg_start:seg_end + 1, channel
                ].clamp(min=confidence)
        batch_segments.sort(key=lambda seg: (seg["start"], seg["channel"]))
        segments.append(batch_segments)
    return segments, raster


class VelocityContactModule(nn.Module):
    """
    Speed and contact prediction.
    Input: human_imu, obj_imu, hp_out (optional human-pose outputs).
    Output: hand/object velocity, per-frame contact and contact endpoints.
    """

    _REMOVED_MERGED_CONTACT_PREFIXES = ("contact_rnn.", "obj_move_head.", "hand_contact_head.")

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.obj_imu_dim = getattr(cfg, "obj_imu_dim", self.imu_dim)
        if self.obj_imu_dim != self.imu_dim:
            raise ValueError("Shared velocity predictor requires obj_imu_dim to match imu_dim.")
        hidden_dim = getattr(cfg, "velocity_hidden_dim", 128)
        num_layers = getattr(cfg, "velocity_num_layers", 2)
        dropout = getattr(cfg, "velocity_dropout", 0.2)
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.use_contact_geom_features = bool(getattr(cfg, "velocity_contact_use_geom_features", True))
        self.detach_contact_geom_obj_vel = bool(getattr(cfg, "velocity_contact_geom_detach_obj_vel", True))
        self.boundary_bidirectional = bool(getattr(cfg, "boundary_bidirectional", True))
        self.adapt_checkpoint_hand_contact_input = bool(
            getattr(cfg, "velocity_contact_adapt_checkpoint_input", False)
        )

        # Shared body-part interaction encoder used by contact and endpoint heads.
        self.body_group_indices = {
            "left_arm": [13, 16, 18, 20, 22],
            "right_arm": [14, 17, 19, 21, 23],
            "torso": [0, 3, 6, 9, 12, 15],
            "left_leg": [1, 4, 7, 10],
            "right_leg": [2, 5, 8, 11],
        }
        group_hidden = getattr(cfg, "boundary_group_hidden", max(hidden_dim // 2, 32))
        boundary_hidden = getattr(cfg, "boundary_hidden_dim", hidden_dim)
        self.boundary_hidden_dim = boundary_hidden
        self.group_encoders = nn.ModuleDict()
        self.group_norms = nn.ModuleDict()
        for name, idx in self.body_group_indices.items():
            group_input_dim = len(idx) * (6 + 3 + 3) + 3  # pose6d + vel + acc + root vel
            self.group_norms[name] = nn.LayerNorm(group_input_dim)
            self.group_encoders[name] = nn.GRU(
                input_size=group_input_dim,
                hidden_size=group_hidden,
                num_layers=1,
                batch_first=True,
            )
        boundary_input_dim = group_hidden * len(self.body_group_indices)
        self.boundary_rnn = nn.GRU(
            input_size=boundary_input_dim,
            hidden_size=boundary_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=self.boundary_bidirectional,
        )
        boundary_rnn_output_dim = boundary_hidden * (2 if self.boundary_bidirectional else 1)
        self.boundary_fusion = (
            nn.Linear(boundary_rnn_output_dim, boundary_hidden)
            if boundary_rnn_output_dim != boundary_hidden
            else nn.Identity()
        )
        self.boundary_input_norm = nn.LayerNorm(boundary_input_dim)
        self.boundary_hidden_norm = nn.LayerNorm(boundary_hidden)

        # Velocity predictor (shared for hands/obj)
        self.vel_net = RNNWithInit(
            n_input=self.imu_dim,
            n_output=3,
            n_hidden=hidden_dim,
            n_init=3,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        # Object move probability
        contact_input_dim = 2 * self.imu_dim + self.obj_imu_dim
        contact_hidden = max(hidden_dim // 2, 32)
        self.obj_move_net = RNNWithInit(
            n_input=contact_input_dim,
            n_output=1,  # object moving or not
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )
        # Hand contact conditioned on object moving
        self.legacy_hand_contact_input_dim = contact_input_dim
        self.base_hand_contact_input_dim = 2 * (3 + 3) + 3 + self.obj_imu_dim + boundary_hidden
        # Per hand: rel_pos(3), distance(1), rel_vel(3), velocity cosine(1).
        self.contact_geom_dim = 2 * (3 + 1 + 3 + 1)
        self.geom_hand_contact_input_dim = self.base_hand_contact_input_dim + (
            self.contact_geom_dim if self.use_contact_geom_features else 0
        )
        hand_contact_input_dim = self.geom_hand_contact_input_dim
        self.hand_contact_input_mode = "geom" if self.use_contact_geom_features else "current"
        self.hand_contact_net = RNNWithInit(
            n_input=hand_contact_input_dim,
            n_output=2,  # left/right conditional prob
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )

        # Contact segment endpoint heatmaps, 6 channels:
        # (lhand_start, lhand_end, rhand_start, rhand_end, obj_start, obj_end).
        # Fed by the same current-mode features as the geom hand-contact path,
        # independent of any checkpoint-compat hand-contact input mode.
        self.contact_endpoint_channels = 6
        self.contact_endpoint_input_dim = self.base_hand_contact_input_dim + (
            self.contact_geom_dim if self.use_contact_geom_features else 0
        )
        self.contact_endpoint_input_norm = nn.LayerNorm(self.contact_endpoint_input_dim)
        self.contact_endpoint_rnn = nn.GRU(
            input_size=self.contact_endpoint_input_dim,
            hidden_size=contact_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.contact_endpoint_head = nn.Linear(contact_hidden, self.contact_endpoint_channels)

        self.left_hand_sensor = _SENSOR_NAMES.index("LeftForeArm")
        self.right_hand_sensor = _SENSOR_NAMES.index("RightForeArm")

    def should_adapt_checkpoint_hand_contact_input(self) -> bool:
        return bool(self.adapt_checkpoint_hand_contact_input)

    def enable_legacy_hand_contact_input(self, input_dim: int = None):
        """Use a checkpoint-compatible hand-contact input path."""
        input_dim = int(input_dim or self.legacy_hand_contact_input_dim)
        if input_dim == self.legacy_hand_contact_input_dim:
            self.hand_contact_input_mode = "legacy"
        elif input_dim == self.base_hand_contact_input_dim:
            self.hand_contact_input_mode = "current"
        elif input_dim == self.geom_hand_contact_input_dim:
            self.hand_contact_input_mode = "geom"
        else:
            self.hand_contact_input_mode = "legacy"
        current_dim = int(self.hand_contact_net.linear1.in_features)
        if current_dim == input_dim:
            return self

        hidden_dim = int(self.hand_contact_net.n_hidden)
        num_layers = int(self.hand_contact_net.n_rnn_layer)
        dropout = float(getattr(self.hand_contact_net.dropout, "p", 0.0))
        device = self.hand_contact_net.linear1.weight.device
        dtype = self.hand_contact_net.linear1.weight.dtype
        self.hand_contact_net = RNNWithInit(
            n_input=input_dim,
            n_output=2,
            n_hidden=hidden_dim,
            n_init=3,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        ).to(device=device, dtype=dtype)
        return self

    def validate_checkpoint_state_dict(self, state_dict) -> None:
        """Reject checkpoints from the abandoned merged-contact experiment."""
        if not isinstance(state_dict, dict):
            raise ValueError("VelocityContact checkpoint does not contain a state_dict.")
        keys = {
            str(key[7:] if str(key).startswith("module.") else key)
            for key in state_dict
        }
        removed_prefixes = [
            prefix for prefix in self._REMOVED_MERGED_CONTACT_PREFIXES
            if any(key.startswith(prefix) for key in keys)
        ]
        if removed_prefixes:
            raise ValueError(
                "VelocityContact checkpoint uses the abandoned merged-contact architecture "
                f"({', '.join(removed_prefixes)}). Use a checkpoint with separate obj_move_net "
                "and hand_contact_net."
            )
        # A direction mismatch would otherwise be dropped silently by
        # non-strict loading, leaving boundary_rnn randomly initialized while
        # its features still feed hand_contact_net.  The legacy architecture
        # shares this validator but has no boundary branch at all — it ignores
        # boundary weights, so the check must not touch boundary attributes.
        module_bidirectional = getattr(self, "boundary_bidirectional", None)
        has_boundary_weights = any(key.startswith("boundary_rnn.") for key in keys)
        if has_boundary_weights and module_bidirectional is not None:
            checkpoint_bidirectional = any(
                key.startswith("boundary_rnn.") and "_reverse" in key for key in keys
            )
            if checkpoint_bidirectional != bool(module_bidirectional):
                raise ValueError(
                    "VelocityContact checkpoint boundary_rnn direction (bidirectional="
                    f"{checkpoint_bidirectional}) does not match this configuration "
                    f"(boundary_bidirectional={bool(module_bidirectional)}); "
                    "retrain VelocityContact with the current boundary_bidirectional setting."
                )

    def _denormalize_imu(self, human_imu: torch.Tensor):
        """Convert IMU data back to world frame."""
        batch_size, seq_len = human_imu.shape[:2]

        human_imu_acc = human_imu[:, :, :, :3]
        human_imu_ori = human_imu[:, :, :, 3:9]
        human_imu_ori_6d = human_imu_ori.reshape(-1, 6)
        human_imu_ori_mat = rotation_6d_to_matrix(human_imu_ori_6d).reshape(batch_size, seq_len, 6, 3, 3)

        R0T = human_imu_ori_mat[:, :, 0].transpose(-1, -2)
        acc_world = torch.matmul(human_imu_acc, R0T)
        acc0_world = acc_world[:, :, :1, :]
        acc_rest_mix = acc_world[:, :, 1:, :] + acc0_world
        human_imu_acc_denorm = torch.cat([acc0_world, acc_rest_mix], dim=2)

        human_imu_ori_denorm = torch.cat([
            human_imu_ori_mat[:, :, :1],
            human_imu_ori_mat[:, :, :1].matmul(human_imu_ori_mat[:, :, 1:])
        ], dim=2)
        human_imu_ori_denorm_6d = matrix_to_rotation_6d(human_imu_ori_denorm)
        human_imu_denorm = torch.cat([human_imu_acc_denorm, human_imu_ori_denorm_6d], dim=-1)

        return human_imu_denorm

    def _prepare_pose_streams(self, hp_out, device, dtype, batch_size, seq_len):
        pose_6d = None
        joint_pos = None
        root_vel = None
        root_trans = None
        hand_pos = None
        if isinstance(hp_out, dict):
            pose_6d = hp_out.get("pred_full_pose_6d")
            joint_pos = hp_out.get("pred_joints_local")
            root_vel = hp_out.get("root_vel_pred")
            root_trans = hp_out.get("root_trans_pred")
            hand_pos = hp_out.get("pred_palm_glb_pos")
            if hand_pos is None:
                hand_pos = hp_out.get("pred_hand_glb_pos")

        if pose_6d is None:
            pose_6d = torch.zeros(batch_size, seq_len, 24, 6, device=device, dtype=dtype)
        else:
            pose_6d = pose_6d.to(device=device, dtype=dtype)
        if joint_pos is None:
            joint_pos = torch.zeros(batch_size, seq_len, 24, 3, device=device, dtype=dtype)
        else:
            joint_pos = joint_pos.to(device=device, dtype=dtype)
        if root_vel is None:
            root_vel = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        else:
            root_vel = root_vel.to(device=device, dtype=dtype)
            if root_vel.dim() == 2 and root_vel.shape == (batch_size, 3):
                root_vel = root_vel.unsqueeze(1).expand(batch_size, seq_len, 3)
            if root_vel.dim() != 3 or root_vel.shape[:2] != (batch_size, seq_len) or root_vel.shape[-1] != 3:
                root_vel = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        if root_trans is None:
            root_trans = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        else:
            root_trans = root_trans.to(device=device, dtype=dtype)
            if root_trans.dim() == 2 and root_trans.shape == (batch_size, 3):
                root_trans = root_trans.unsqueeze(1).expand(batch_size, seq_len, 3)
            if root_trans.dim() != 3 or root_trans.shape[:2] != (batch_size, seq_len) or root_trans.shape[-1] != 3:
                root_trans = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        if hand_pos is None:
            if joint_pos.shape[2] > 21:
                hand_pos = select_hand_anchor_positions(joint_pos) + root_trans.unsqueeze(2)
            else:
                hand_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)
        else:
            hand_pos = hand_pos.to(device=device, dtype=dtype)
            if hand_pos.dim() == 3 and hand_pos.shape[:2] == (batch_size, seq_len) and hand_pos.shape[-1] == 6:
                hand_pos = hand_pos.reshape(batch_size, seq_len, 2, 3)
            if hand_pos.dim() != 4 or hand_pos.shape[:3] != (batch_size, seq_len, 2) or hand_pos.shape[-1] != 3:
                hand_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)

        joint_vel = self._causal_velocity(joint_pos)
        joint_acc = self._causal_acceleration(joint_pos)
        hand_vel = self._causal_velocity(hand_pos)
        hand_acc = self._causal_acceleration(hand_pos)


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

    def _causal_velocity(self, value: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(value)
        if value.size(1) > 1:
            vel[:, 1:] = (value[:, 1:] - value[:, :-1]) * self.fps
        return vel

    def _causal_acceleration(self, value: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros_like(value)
        if value.size(1) > 2:
            acc[:, 2:] = (value[:, 2:] - 2.0 * value[:, 1:-1] + value[:, :-2]) * (self.fps ** 2)
        return acc

    def _prepare_init_vector(self, value, batch_size, dim, device, dtype):
        if isinstance(value, torch.Tensor):
            out = value.to(device=device, dtype=dtype)
            if out.dim() == 1:
                out = out.unsqueeze(0).expand(batch_size, -1)
            elif out.shape[0] == 1 and batch_size > 1:
                out = out.expand(batch_size, *out.shape[1:])
            if out.dim() == 2 and out.shape == (batch_size, dim):
                return out
        return torch.zeros(batch_size, dim, device=device, dtype=dtype)

    def _integrate_rough_object_translation(self, obj_vel, obj_trans_init):
        dt = 1.0 / self.fps
        disp = torch.cumsum(obj_vel * dt, dim=1)
        if disp.shape[1] > 0:
            zero = torch.zeros(disp.shape[0], 1, 3, device=disp.device, dtype=disp.dtype)
            disp = torch.cat((zero, disp[:, :-1]), dim=1)
        return obj_trans_init.unsqueeze(1) + disp

    @staticmethod
    def _cosine_similarity(u, v, eps=1e-6):
        denom = u.norm(dim=-1, keepdim=True).clamp_min(eps) * v.norm(dim=-1, keepdim=True).clamp_min(eps)
        return (u * v).sum(dim=-1, keepdim=True) / denom

    def _build_contact_geom_features(self, pose_streams, obj_vel, obj_trans_init):
        geom_obj_vel = obj_vel.detach() if self.detach_contact_geom_obj_vel else obj_vel
        obj_trans_rough = self._integrate_rough_object_translation(geom_obj_vel, obj_trans_init)
        hand_pos = pose_streams["hand_pos"]
        hand_vel = pose_streams["hand_vel"]

        obj_vel_hands = geom_obj_vel.unsqueeze(2).expand(-1, -1, 2, -1)
        rel_pos = hand_pos - obj_trans_rough.unsqueeze(2)
        rel_dist = rel_pos.norm(dim=-1, keepdim=True)
        rel_vel = hand_vel - obj_vel_hands
        vel_cos = self._cosine_similarity(hand_vel, obj_vel_hands)

        batch_size, seq_len = obj_vel.shape[:2]
        geom_feat = torch.cat((rel_pos, rel_dist, rel_vel, vel_cos), dim=-1).reshape(batch_size, seq_len, -1)
        return geom_feat, obj_trans_rough

    def _encode_body_groups(self, pose_streams):
        boundary_features = []
        pose_6d = pose_streams["pose_6d"]
        joint_vel = pose_streams["joint_vel"]
        joint_acc = pose_streams["joint_acc"]
        root_vel = pose_streams["root_vel"]
        batch_size, seq_len = root_vel.shape[:2]

        for name, encoder in self.group_encoders.items():
            idx = self.body_group_indices[name]
            pose_feat = pose_6d[:, :, idx, :].reshape(batch_size, seq_len, -1)
            vel_feat = joint_vel[:, :, idx, :].reshape(batch_size, seq_len, -1)
            acc_feat = joint_acc[:, :, idx, :].reshape(batch_size, seq_len, -1)
            group_input = torch.cat((pose_feat, vel_feat, acc_feat, root_vel), dim=-1)
            group_input = self.group_norms[name](group_input)
            group_out, _ = encoder(group_input)
            boundary_features.append(group_out)

        boundary_input = torch.cat(boundary_features, dim=-1)
        boundary_input = self.boundary_input_norm(boundary_input)
        boundary_out, _ = self.boundary_rnn(boundary_input)
        boundary_out = self.boundary_fusion(boundary_out)
        boundary_out_norm = self.boundary_hidden_norm(boundary_out)
        return boundary_out_norm

    def forward(
        self,
        data_dict: dict,
        hp_out: dict = None,
        velocity_only: bool = False,
    ):
        """
        Forward pass.
        Args:
            data_dict: contains human_imu, obj_imu, hand_vel_glb_init, obj_vel_init, contact_init.
            hp_out: optional human pose outputs (pose/vel/acc) for contact prediction.
            velocity_only: return after the shared hand/object velocity path.
        """
        human_imu = data_dict['human_imu']
        obj_imu = data_dict.get('obj_imu')
        hand_vel_init = data_dict['hand_vel_glb_init']
        obj_vel_init = data_dict['obj_vel_init']
        contact_init = data_dict.get('contact_init')
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B, T, num_imu, imu_dim], got {human_imu.shape}")
        batch_size, seq_len, _, _ = human_imu.shape
        device = human_imu.device
        dtype = human_imu.dtype

        # Object IMU
        if obj_imu is None:
            obj_imu = torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        else:
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(batch_size, seq_len, -1)
            elif obj_imu.dim() != 3:
                raise ValueError(f"obj_imu must be [B, T, obj_dim], got {obj_imu.shape}")
        if obj_imu.shape[-1] != self.imu_dim:
            raise ValueError(f"Shared velocity net expects obj_imu feature dim {self.imu_dim}, got {obj_imu.shape[-1]}")

        # Init vectors
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
        obj_trans_init_vec = self._prepare_init_vector(data_dict.get("obj_trans_init"), batch_size, 3, device, dtype)

        # Denormalize IMU
        human_imu_denorm = self._denormalize_imu(human_imu)

        # Hand velocities
        l_hand_imu = human_imu_denorm[:, :, self.left_hand_sensor, :]
        r_hand_imu = human_imu_denorm[:, :, self.right_hand_sensor, :]
        l_hand_vel = self.vel_net((l_hand_imu, l_hand_vel_init))
        r_hand_vel = self.vel_net((r_hand_imu, r_hand_vel_init))
        hand_vel = torch.stack((l_hand_vel, r_hand_vel), dim=2)

        # Object velocity
        obj_vel = self.vel_net((obj_imu, obj_vel_init_vec))
        if velocity_only:
            return {
                "pred_hand_glb_vel": hand_vel,
                "pred_obj_vel": obj_vel,
            }

        # Contact input (IMU-based)
        hand_imu_feat = human_imu_denorm[:, :, [self.left_hand_sensor, self.right_hand_sensor], :].reshape(
            batch_size, seq_len, -1
        )
        contact_input_imu = torch.cat((hand_imu_feat, obj_imu), dim=-1)

        # Contact init
        if contact_init is None:
            contact_init_vec = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        else:
            if contact_init.dim() == 1:
                contact_init_vec = contact_init.unsqueeze(0).expand(batch_size, -1)
            else:
                contact_init_vec = contact_init
            if contact_init_vec.shape[-1] > 3:
                contact_init_vec = contact_init_vec[..., :3]
            if contact_init_vec.shape[-1] != 3:
                raise ValueError(f"contact_init must have last dim 3, got {contact_init_vec.shape}")

        # Object move prob
        obj_move_logits = self.obj_move_net((contact_input_imu, contact_init_vec))
        obj_move_prob = torch.sigmoid(obj_move_logits)

        # Shared interaction features from HPE outputs
        pose_streams = self._prepare_pose_streams(hp_out or data_dict.get("hp_out"), device, dtype, batch_size, seq_len)
        boundary_out = self._encode_body_groups(pose_streams)

        # Shared per-frame features for hand contact and endpoint detection
        hand_dyn_feat = torch.cat(
            (
                pose_streams["hand_vel"].reshape(batch_size, seq_len, -1),
                pose_streams["hand_acc"].reshape(batch_size, seq_len, -1),
            ),
            dim=-1,
        )
        root_vel_feat = pose_streams["root_vel"]
        base_contact_input = torch.cat((hand_dyn_feat, root_vel_feat, obj_imu, boundary_out), dim=-1)
        geom_feat = None
        obj_trans_rough = None
        if self.use_contact_geom_features:
            geom_feat, obj_trans_rough = self._build_contact_geom_features(
                pose_streams, obj_vel, obj_trans_init_vec
            )

        # Hand contact using HPE-derived velocity/acc plus boundary hidden states
        if getattr(self, "hand_contact_input_mode", "current") == "legacy":
            hand_contact_input = contact_input_imu
        else:
            expected_contact_dim = int(self.hand_contact_net.linear1.in_features)
            if expected_contact_dim == self.geom_hand_contact_input_dim and geom_feat is not None:
                hand_contact_input = torch.cat((base_contact_input, geom_feat), dim=-1)
            else:
                hand_contact_input = base_contact_input
        expected_contact_dim = int(self.hand_contact_net.linear1.in_features)
        if hand_contact_input.shape[-1] != expected_contact_dim:
            raise ValueError(
                f"hand_contact_input dim {hand_contact_input.shape[-1]} does not match "
                f"hand_contact_net input dim {expected_contact_dim}"
            )

        hand_contact_logits = self.hand_contact_net((hand_contact_input, contact_init_vec))
        hand_contact_prob_cond = torch.sigmoid(hand_contact_logits)

        # Unconditional hand contact = object move prob * conditional
        hand_contact_prob = obj_move_prob * hand_contact_prob_cond

        # Contact segment endpoint heatmaps
        endpoint_input = base_contact_input
        if geom_feat is not None:
            endpoint_input = torch.cat((endpoint_input, geom_feat), dim=-1)
        endpoint_input = self.contact_endpoint_input_norm(endpoint_input)
        endpoint_hidden, _ = self.contact_endpoint_rnn(endpoint_input)
        contact_endpoint_logits = self.contact_endpoint_head(endpoint_hidden)
        contact_endpoint_prob = torch.sigmoid(contact_endpoint_logits)
        contact_from_endpoints_prob = recover_contact_state_from_endpoints(
            contact_endpoint_prob,
            contact_init_vec,
        )

        # Concatenated outputs for compatibility
        contact_logits = torch.cat((hand_contact_logits, obj_move_logits), dim=-1)
        contact_prob = torch.cat((hand_contact_prob, obj_move_prob), dim=-1)

        out = {
            "pred_hand_glb_vel": hand_vel,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": contact_logits,
            "pred_hand_contact_prob": contact_prob,
            "pred_obj_move_prob": obj_move_prob,
            "pred_hand_contact_prob_cond": hand_contact_prob_cond,
            "pred_hand_contact_logits_cond": hand_contact_logits,
            "pred_contact_endpoint_logits": contact_endpoint_logits,
            "pred_contact_endpoint_prob": contact_endpoint_prob,
            "pred_contact_from_endpoints_prob": contact_from_endpoints_prob,
        }
        if obj_trans_rough is not None:
            out["pred_obj_trans_rough"] = obj_trans_rough
        return out

    @torch.no_grad()
    def decode_contact_segments(self, endpoint_prob, interior_prob=None, **overrides):
        """Decode endpoint heatmaps into segments using cfg thresholds."""
        initial_thresholds = getattr(self.cfg, "contact_endpoint_init_thresholds", None)
        if initial_thresholds is None:
            initial_thresholds = getattr(
                self.cfg,
                "contact_endpoint_init_threshold",
                (
                    float(getattr(self.cfg, "hand_contact_threshold", 0.15)),
                    float(getattr(self.cfg, "hand_contact_threshold", 0.15)),
                    float(getattr(self.cfg, "object_contact_threshold", 0.5)),
                ),
            )
        kwargs = dict(
            peak_threshold=float(getattr(self.cfg, "contact_endpoint_peak_threshold", 0.3)),
            segment_threshold=float(getattr(self.cfg, "contact_endpoint_segment_threshold", 0.25)),
            peak_radius=int(getattr(self.cfg, "contact_endpoint_peak_radius", 5)),
            initial_contact_threshold=initial_thresholds,
        )
        kwargs.update(overrides)
        return decode_contact_segments(endpoint_prob, interior_prob, **kwargs)

    def _apply_segment_contact(self, output, contact_init=None):
        """Replace per-frame contact prob with rasterized decoded segments.

        The per-frame prediction is kept under ``pred_hand_contact_prob_frame``.
        ``contact_init`` is the three-channel contact state at frame 0.
        Segment frame indices refer to the sequence passed in (in online mode:
        the full accumulated stream).
        """
        endpoint_prob = output.get("pred_contact_endpoint_prob")
        if endpoint_prob is None:
            return output
        frame_prob = output.get("pred_hand_contact_prob")
        segments, raster = self.decode_contact_segments(
            endpoint_prob,
            frame_prob,
            initial_contact=contact_init,
        )
        output = dict(output)
        output["pred_contact_segments"] = segments
        if frame_prob is not None:
            output["pred_hand_contact_prob_frame"] = frame_prob
            raster = raster.to(dtype=frame_prob.dtype)
        output["pred_hand_contact_prob"] = raster
        return output

    def _slice_hp_out(self, hp_out, start: int, end: int, batch_size: int, seq_len: int):
        if not isinstance(hp_out, dict):
            return hp_out
        return slice_time_dict(hp_out, start, end, batch_size, seq_len)

    def _inference_online_sequence(self, data_dict: dict, hp_out: dict = None, online_window: int = 120):
        batch_size, seq_len = infer_batch_seq(data_dict)
        if seq_len <= online_window:
            return self.forward(data_dict, hp_out=hp_out)

        warmup_len = int(online_window)
        warmup_data = slice_time_dict(data_dict, 0, warmup_len, batch_size, seq_len)
        warmup_hp = self._slice_hp_out(hp_out, 0, warmup_len, batch_size, seq_len)
        history = self.forward(warmup_data, hp_out=warmup_hp)
        history_acc = TimeDictAccumulator(history, seq_len)
        history = history_acc.current()

        for end in range(warmup_len + 1, seq_len + 1):
            start = end - warmup_len
            window_data = slice_time_dict(data_dict, start, end, batch_size, seq_len)
            window_data = update_data_inits_from_history(
                window_data,
                history,
                index=start - 1,
                contact_index=start,
            )
            window_hp = self._slice_hp_out(hp_out, start, end, batch_size, seq_len)
            window_out = self.forward(window_data, hp_out=window_hp)
            latest = take_latest_frame(window_out, batch_size, end - start)
            history = history_acc.append(latest)

        return history_acc.current()

    def inference(
        self,
        data_dict: dict,
        hp_out: dict = None,
        inference_mode: str = "offline",
        online_window: int = None,
        online_state: dict = None,
        return_online_state: bool = False,
        **_,
    ):
        mode = normalize_inference_mode(inference_mode)
        contact_from_segments = bool(getattr(self.cfg, "contact_from_segments", False))
        if mode == "offline":
            output = self.forward(data_dict, hp_out=hp_out)
            if contact_from_segments:
                output = self._apply_segment_contact(
                    output,
                    contact_init=data_dict.get("contact_init"),
                )
            if return_online_state:
                return output, online_state or {}
            return output

        window = resolve_online_window(self.cfg, online_window)
        run_data, previous_len = append_stream_data(
            online_state.get("data_dict") if isinstance(online_state, dict) else None,
            data_dict,
        )
        output = self._inference_online_sequence(run_data, hp_out=hp_out, online_window=window)
        if contact_from_segments:
            output = self._apply_segment_contact(
                output,
                contact_init=run_data.get("contact_init"),
            )
        state = {"data_dict": run_data, "hp_out": hp_out, "outputs": output}
        if return_online_state:
            if previous_len > 0:
                batch_size, seq_len = infer_batch_seq(run_data)
                output = slice_time_dict(output, previous_len, seq_len, batch_size, seq_len)
            return output, state
        return output

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
            "pred_contact_endpoint_logits": torch.zeros(batch_size, seq_len, 6, device=device),
            "pred_contact_endpoint_prob": torch.zeros(batch_size, seq_len, 6, device=device),
            "pred_contact_from_endpoints_prob": torch.zeros(batch_size, seq_len, 3, device=device),
        }
