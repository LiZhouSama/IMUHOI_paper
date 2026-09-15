"""DIP18-style BiRNN baseline with an explicit object-motion extension."""
from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DIPBiRNNHead(nn.Module):
    """DIP BiRNN block: input FC 512, 2-layer BiLSTM 512, output FC 256."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_hidden_size: int = 512,
        rnn_hidden_size: int = 512,
        output_hidden_size: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, input_hidden_size),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(
            input_hidden_size,
            rnn_hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output_hidden = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, output_hidden_size),
            nn.ReLU(),
        )
        self.output = nn.Linear(output_hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input_layer(x)
        y, _ = self.rnn(y)
        y = self.output_hidden(y)
        return self.output(y)


class DIPReferenceRNN(nn.Module):
    """The RNN block used by the public OMOMO DIP adaptation.

    The reference implementation projects each frame with ``Linear -> ReLU``
    before a stacked LSTM and applies a final linear projection.  Keeping this
    block separate makes the published checkpoint state-dict names and tensor
    shapes load without an ad-hoc conversion layer.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_rnn_layer: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_hidden = int(n_hidden)
        self.n_rnn_layer = int(n_rnn_layer)
        self.num_directions = 2 if bidirectional else 1
        self.rnn = nn.LSTM(
            n_hidden,
            n_hidden,
            n_rnn_layer,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_rnn_layer > 1 else 0.0,
        )
        # These names/order match RefCodes/dip18/my_simple/model_obj.py.
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * self.num_directions, n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.linear1(x)))
        output, _ = self.rnn(x)
        return self.linear2(output)


class DIP18ReferenceHOIModel(nn.Module):
    """Published OMOMO/DynaIP-style DIP model with an object branch.

    The public comparison code uses all six normalized IMUs (72 features), a
    six-dimensional global recurrent feature, a bidirectional 22-joint human
    branch, and a causal object-velocity branch.  This is intentionally kept
    as a distinct variant so checkpoints produced by the earlier repository
    adapter remain loadable.
    """

    def __init__(self, fps: float = 30.0):
        super().__init__()
        self.dip_variant = "official"
        self.human_input_dim = 72
        self.obj_imu_dim = 12
        self.pose_dim = 22 * 9
        self.fps = float(fps)
        self.glb = DIPReferenceRNN(
            n_input=72,
            n_output=6,
            n_hidden=36,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=0.2,
        )
        self.human_branch = DIPReferenceRNN(
            n_input=72 + 6,
            n_output=self.pose_dim,
            n_hidden=512,
            n_rnn_layer=2,
            bidirectional=True,
            dropout=0.2,
        )
        self.obj_branch = DIPReferenceRNN(
            n_input=12 + 6,
            n_output=3,
            n_hidden=128,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )

    def forward(
        self, imu: torch.Tensor, obj_imu: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        global_feature = self.glb(imu)
        pose = self.human_branch(torch.cat((imu, global_feature), dim=-1))
        obj_trans = self.obj_branch(torch.cat((obj_imu, global_feature), dim=-1))
        obj_position_relative = torch.zeros_like(obj_trans)
        if obj_trans.shape[1] > 1:
            obj_position_relative[:, 1:] = torch.cumsum(
                obj_trans[:, 1:] / self.fps, dim=1
            )
        return {
            # ``pose_mu`` is the common comparison surface; the reference
            # model is deterministic, so sigma is only a compatibility alias.
            "pose": pose,
            "pose_mu": pose,
            "pose_sigma": torch.ones_like(pose),
            "obj_trans": obj_trans,
            "obj_position_relative": obj_position_relative,
            "global_feature": global_feature,
            "dip_reference": True,
        }

    @torch.no_grad()
    def reconstruct_chunks(
        self,
        imu: torch.Tensor,
        obj_imu: torch.Tensor,
        past_frames: int = 20,
        future_frames: int = 5,
        window_batch_size: int = 128,
    ) -> dict[str, torch.Tensor]:
        """Run the same delayed BiRNN chunk protocol as the public runner."""
        if past_frames < 0 or future_frames < 0:
            raise ValueError("past_frames and future_frames must be non-negative")
        if window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")
        batch_size, seq_len = imu.shape[:2]
        if seq_len <= 0:
            raise ValueError("DIP inference requires at least one frame")
        pose_all = imu.new_empty(batch_size, seq_len, self.pose_dim)
        obj_all = imu.new_empty(batch_size, seq_len, 3)
        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for frame in range(seq_len):
            start = max(0, frame - past_frames)
            end = min(seq_len, frame + future_frames + 1)
            groups[(end - start, frame - start)].append((frame, start))
        frames_per_batch = max(1, window_batch_size // batch_size)
        for (window_len, center), entries in groups.items():
            for offset in range(0, len(entries), frames_per_batch):
                chunk = entries[offset : offset + frames_per_batch]
                human_windows = torch.stack(
                    [imu[:, start : start + window_len] for _, start in chunk], dim=1
                )
                object_windows = torch.stack(
                    [obj_imu[:, start : start + window_len] for _, start in chunk], dim=1
                )
                num_windows = human_windows.shape[1]
                output = self.forward(
                    human_windows.reshape(batch_size * num_windows, window_len, -1),
                    object_windows.reshape(batch_size * num_windows, window_len, -1),
                )
                frame_index = torch.as_tensor(
                    [frame for frame, _ in chunk], device=imu.device, dtype=torch.long
                )
                pose_all.index_copy_(
                    1,
                    frame_index,
                    output["pose"][:, center].reshape(batch_size, num_windows, -1),
                )
                obj_all.index_copy_(
                    1,
                    frame_index,
                    output["obj_trans"][:, center].reshape(batch_size, num_windows, -1),
                )
        return {
            "pose": pose_all,
            "pose_mu": pose_all,
            "obj_trans": obj_all,
            "dip_reference": True,
        }


class DIP18HOIModel(nn.Module):
    """PyTorch port of the DIP BiRNN reconstruction surface.

    Human pose prediction keeps DIP's Gaussian-output training semantics.  The
    object branch is intentionally separate and predicts object velocity for v2
    checkpoints from the same augmented IMU stream.
    """

    def __init__(
        self,
        human_input_dim: int = 60,
        obj_imu_dim: int = 12,
        pose_dim: int = 135,
        hidden_size: int = 512,
        input_hidden_size: int = 512,
        output_hidden_size: int = 256,
        num_layers: int = 2,
        human_condition_on_object: bool = False,
    ):
        super().__init__()
        self.dip_variant = "legacy"
        self.human_input_dim = human_input_dim
        self.obj_imu_dim = obj_imu_dim
        self.pose_dim = pose_dim
        self.human_condition_on_object = bool(human_condition_on_object)
        self.fps = 30.0
        object_input_dim = human_input_dim + obj_imu_dim
        human_model_input_dim = (
            object_input_dim if self.human_condition_on_object else human_input_dim
        )
        self.temporal = DIPBiRNNHead(
            input_dim=human_model_input_dim,
            output_dim=pose_dim * 2,
            input_hidden_size=input_hidden_size,
            rnn_hidden_size=hidden_size,
            output_hidden_size=output_hidden_size,
            num_layers=num_layers,
        )
        self.obj_head = DIPBiRNNHead(
            input_dim=object_input_dim,
            output_dim=3,
            input_hidden_size=input_hidden_size,
            rnn_hidden_size=hidden_size,
            output_hidden_size=output_hidden_size,
            num_layers=num_layers,
        )

    def forward(self, imu: torch.Tensor, obj_imu: torch.Tensor) -> dict[str, torch.Tensor]:
        object_input = torch.cat([imu, obj_imu], dim=-1)
        human_input = object_input if self.human_condition_on_object else imu
        pose_stats = self.temporal(human_input)
        pose_mu, pose_sigma_raw = pose_stats.split(self.pose_dim, dim=-1)
        pose_sigma = F.softplus(pose_sigma_raw).clamp_min(1e-6)
        obj_trans = self.obj_head(object_input)
        obj_position_relative = torch.zeros_like(obj_trans)
        if obj_trans.shape[1] > 1:
            obj_position_relative[:, 1:] = torch.cumsum(
                obj_trans[:, 1:] / self.fps, dim=1
            )
        return {
            "pose_mu": pose_mu,
            "pose_sigma": pose_sigma,
            "pose_sigma_raw": pose_sigma_raw,
            "obj_trans": obj_trans,
            "obj_position_relative": obj_position_relative,
        }

    @torch.no_grad()
    def reconstruct_chunks(
        self,
        imu: torch.Tensor,
        obj_imu: torch.Tensor,
        past_frames: int = 20,
        future_frames: int = 5,
        window_batch_size: int = 128,
    ) -> dict[str, torch.Tensor]:
        """Chunked BiRNN reconstruction for DIP-style streaming evaluation."""
        if past_frames < 0 or future_frames < 0:
            raise ValueError("past_frames and future_frames must be non-negative")
        if window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")
        batch_size, seq_len = imu.shape[:2]
        if seq_len <= 0:
            raise ValueError("DIP inference requires at least one frame")
        pose_all = imu.new_empty(batch_size, seq_len, self.pose_dim)
        obj_all = imu.new_empty(batch_size, seq_len, 3)

        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for frame in range(seq_len):
            start = max(0, frame - past_frames)
            end = min(seq_len, frame + future_frames + 1)
            groups[(end - start, frame - start)].append((frame, start))

        frames_per_batch = max(1, window_batch_size // batch_size)
        for (window_len, center), entries in groups.items():
            for offset in range(0, len(entries), frames_per_batch):
                entry_chunk = entries[offset : offset + frames_per_batch]
                human_windows = torch.stack(
                    [imu[:, start : start + window_len] for _, start in entry_chunk],
                    dim=1,
                )
                object_windows = torch.stack(
                    [
                        obj_imu[:, start : start + window_len]
                        for _, start in entry_chunk
                    ],
                    dim=1,
                )
                num_windows = human_windows.shape[1]
                out = self.forward(
                    human_windows.reshape(
                        batch_size * num_windows, window_len, -1
                    ),
                    object_windows.reshape(
                        batch_size * num_windows, window_len, -1
                    ),
                )
                frame_index = torch.as_tensor(
                    [frame for frame, _ in entry_chunk],
                    device=imu.device,
                    dtype=torch.long,
                )
                pose_all.index_copy_(
                    1,
                    frame_index,
                    out["pose_mu"][:, center].reshape(
                        batch_size, num_windows, -1
                    ),
                )
                obj_all.index_copy_(
                    1,
                    frame_index,
                    out["obj_trans"][:, center].reshape(
                        batch_size, num_windows, -1
                    ),
                )
        return {
            "pose_mu": pose_all,
            "obj_trans": obj_all,
        }
