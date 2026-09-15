"""TransPose architecture with the original pose and translation branches."""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from Comparisons.common.modules import BatchRNN


class TransPoseHOIModel(nn.Module):
    """TransPose pose-s1/s2/s3 + tran-b1/b2 with an object-motion head."""

    n_leaf = 5
    n_full = 23
    n_reduced = 15

    def __init__(
        self,
        human_imu_dim: int = 72,
        obj_imu_dim: int = 12,
        dropout: float = 0.2,
        object_bidirectional: bool = False,
        object_condition_on_full: bool = False,
        human_condition_on_object: bool = False,
        fps: float = 30.0,
        transpose_reference: bool = False,
    ):
        super().__init__()
        object_input_dim = human_imu_dim + obj_imu_dim
        self.object_bidirectional = bool(object_bidirectional)
        self.object_condition_on_full = bool(object_condition_on_full)
        self.human_condition_on_object = bool(human_condition_on_object)
        human_model_input_dim = (
            object_input_dim if self.human_condition_on_object else human_imu_dim
        )
        self.fps = float(fps)
        self.velocity_scale = 3.0
        self.transpose_reference = bool(transpose_reference)
        self.pose_s1 = BatchRNN(
            human_model_input_dim,
            self.n_leaf * 3,
            256,
            n_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.pose_s2 = BatchRNN(
            self.n_leaf * 3 + human_model_input_dim,
            self.n_full * 3,
            64,
            n_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.pose_s3 = BatchRNN(
            self.n_full * 3 + human_model_input_dim,
            self.n_reduced * 6,
            128,
            n_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.tran_b1 = BatchRNN(
            self.n_leaf * 3 + human_model_input_dim,
            2,
            64,
            n_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.tran_b2 = BatchRNN(
            self.n_full * 3 + human_model_input_dim,
            3,
            256,
            n_layers=2,
            bidirectional=False,
            dropout=dropout,
        )
        self.obj_head = BatchRNN(
            self.n_full * 3 + object_input_dim
            if self.object_condition_on_full
            else object_input_dim,
            3,
            128,
            n_layers=2,
            bidirectional=self.object_bidirectional,
            dropout=dropout,
        )

    @staticmethod
    def _integrate_displacement(displacement: torch.Tensor) -> torch.Tensor:
        position = torch.zeros_like(displacement)
        if displacement.shape[1] > 1:
            position[:, 1:] = torch.cumsum(displacement[:, 1:], dim=1)
        return position

    def forward(self, imu: torch.Tensor, obj_imu: torch.Tensor) -> dict[str, torch.Tensor]:
        object_input = torch.cat([imu, obj_imu], dim=-1)
        human_input = object_input if self.human_condition_on_object else imu
        leaf, _ = self.pose_s1(human_input)
        full, _ = self.pose_s2(torch.cat([leaf, human_input], dim=-1))
        pose, _ = self.pose_s3(torch.cat([full, human_input], dim=-1))
        contact_logits, _ = self.tran_b1(torch.cat([leaf, human_input], dim=-1))
        root_vel, _ = self.tran_b2(torch.cat([full, human_input], dim=-1))
        obj_features = (
            torch.cat([full, object_input], dim=-1)
            if self.object_condition_on_full
            else object_input
        )
        obj_trans, _ = self.obj_head(obj_features)

        return self._assemble_outputs(
            imu,
            leaf,
            full,
            pose,
            contact_logits,
            root_vel,
            obj_trans,
        )

    def _assemble_outputs(
        self,
        imu: torch.Tensor,
        leaf: torch.Tensor,
        full: torch.Tensor,
        pose: torch.Tensor,
        contact_logits: torch.Tensor,
        root_vel: torch.Tensor,
        obj_trans: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build decoded branch outputs shared by offline and online inference."""

        root_rotation = imu[:, :, -9:].reshape(*imu.shape[:2], 3, 3)
        # The public TransPose target stores local velocity before the
        # ``vel_scale`` conversion.  The historical comparison branch exposed
        # the scaled value, so keep that behavior only for legacy checkpoints.
        root_velocity_local = (
            root_vel if self.transpose_reference else root_vel * self.velocity_scale
        )
        root_velocity_b2 = root_rotation.matmul(
            root_velocity_local.unsqueeze(-1)
        ).squeeze(-1)
        if self.transpose_reference:
            # RefCodes/TransPose converts the network's normalized local
            # velocity with ``vel_scale`` only when integrating translation.
            root_velocity_b2 = root_velocity_b2 * self.velocity_scale
        root_displacement_b2 = root_velocity_b2 / self.fps

        full_position = full.reshape(*full.shape[:2], self.n_full, 3)
        # full_position stores joints 1..23, so SMPL feet 10/11 map to 9/10.
        feet_root = full_position[:, :, [9, 10]]
        feet_world = torch.einsum(
            "btij,btsj->btsi", root_rotation, feet_root
        )
        foot_displacement = torch.zeros_like(feet_world)
        if feet_world.shape[1] > 1:
            foot_displacement[:, 1:] = feet_world[:, :-1] - feet_world[:, 1:]
        selected_foot = contact_logits.argmax(dim=-1, keepdim=True).unsqueeze(-1)
        root_displacement_b1 = torch.gather(
            foot_displacement, 2, selected_foot.expand(-1, -1, -1, 3)
        ).squeeze(2)
        root_displacement_b1[..., 1] += -0.018 * 60.0 / self.fps
        contact_weight = (
            (contact_logits.sigmoid().max(dim=-1).values - 0.5) / 0.4
        ).clamp(0.0, 1.0)
        root_displacement = torch.lerp(
            root_displacement_b2,
            root_displacement_b1,
            contact_weight.unsqueeze(-1),
        )
        if not self.transpose_reference:
            root_displacement[:, 0] = 0.0

        obj_displacement = obj_trans / self.fps
        obj_displacement = obj_displacement.clone()
        obj_displacement[:, 0] = 0.0
        return {
            "leaf_pos": leaf,
            "full_pos": full,
            "pose": pose,
            "contact_logits": contact_logits,
            "root_vel": root_vel,
            "root_velocity_local": root_velocity_local,
            "root_velocity_world": root_displacement * self.fps,
            "root_position_relative": self._integrate_displacement(
                root_displacement
            ),
            "obj_trans": obj_trans,
            "obj_position_relative": self._integrate_displacement(
                obj_displacement
            ),
        }

    @torch.no_grad()
    def reconstruct_chunks(
        self,
        imu: torch.Tensor,
        obj_imu: torch.Tensor,
        past_frames: int = 20,
        future_frames: int = 5,
        window_batch_size: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Run the published delayed-online window for bidirectional branches.

        S1/S2/S3 and B1 see at most ``past_frames + 1 + future_frames``
        observations per target frame. B2 and the protocol-v2 object head remain
        causal and are evaluated once over the aligned per-frame features.
        """
        if past_frames < 0 or future_frames < 0:
            raise ValueError("past_frames and future_frames must be non-negative")
        if window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")
        batch_size, seq_len = imu.shape[:2]
        if seq_len <= 0:
            raise ValueError("TransPose inference requires at least one frame")

        object_input = torch.cat([imu, obj_imu], dim=-1)
        human_input = (
            object_input if self.human_condition_on_object else imu
        )
        leaf_all = imu.new_empty(batch_size, seq_len, self.n_leaf * 3)
        full_all = imu.new_empty(batch_size, seq_len, self.n_full * 3)
        pose_all = imu.new_empty(batch_size, seq_len, self.n_reduced * 6)
        contact_all = imu.new_empty(batch_size, seq_len, 2)

        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for frame in range(seq_len):
            start = max(0, frame - past_frames)
            end = min(seq_len, frame + future_frames + 1)
            groups[(end - start, frame - start)].append((frame, start))

        frames_per_batch = max(1, window_batch_size // batch_size)
        for (window_len, center), entries in groups.items():
            for offset in range(0, len(entries), frames_per_batch):
                entry_chunk = entries[offset : offset + frames_per_batch]
                windows = torch.stack(
                    [
                        human_input[:, start : start + window_len]
                        for _, start in entry_chunk
                    ],
                    dim=1,
                )
                num_windows = windows.shape[1]
                windows = windows.reshape(
                    batch_size * num_windows, window_len, -1
                )
                leaf, _ = self.pose_s1(windows)
                full, _ = self.pose_s2(torch.cat([leaf, windows], dim=-1))
                pose, _ = self.pose_s3(torch.cat([full, windows], dim=-1))
                contact, _ = self.tran_b1(
                    torch.cat([leaf, windows], dim=-1)
                )

                frame_index = torch.as_tensor(
                    [frame for frame, _ in entry_chunk],
                    device=imu.device,
                    dtype=torch.long,
                )
                leaf_all.index_copy_(
                    1,
                    frame_index,
                    leaf[:, center].reshape(batch_size, num_windows, -1),
                )
                full_all.index_copy_(
                    1,
                    frame_index,
                    full[:, center].reshape(batch_size, num_windows, -1),
                )
                pose_all.index_copy_(
                    1,
                    frame_index,
                    pose[:, center].reshape(batch_size, num_windows, -1),
                )
                contact_all.index_copy_(
                    1,
                    frame_index,
                    contact[:, center].reshape(batch_size, num_windows, -1),
                )

        root_vel, _ = self.tran_b2(
            torch.cat([full_all, human_input], dim=-1)
        )
        obj_features = (
            torch.cat([full_all, object_input], dim=-1)
            if self.object_condition_on_full
            else object_input
        )
        obj_trans, _ = self.obj_head(obj_features)
        return self._assemble_outputs(
            imu,
            leaf_all,
            full_all,
            pose_all,
            contact_all,
            root_vel,
            obj_trans,
        )
