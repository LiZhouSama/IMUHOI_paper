"""TIP-style causal Transformer/RNN baseline with an object-motion extension."""
from __future__ import annotations

import torch
from torch import nn


class TIPHOIModel(nn.Module):
    """Transformer Inertial Poser core with an extra object output.

    The human state remains [18*6 pose, root velocity, optional SBP constraints].
    Object IMU is appended to the IMU stream; an object-motion head predicts
    velocity in protocol v2 (legacy checkpoints used absolute translation).
    """

    def __init__(
        self,
        input_size_imu: int = 90,
        obj_imu_dim: int = 12,
        n_sbps: int = 5,
        rnn_hidden_size: int = 512,
        tf_hidden_size: int = 1024,
        tf_input_dim: int = 256,
        n_heads: int = 16,
        tf_layers: int = 4,
        dropout: float = 0.0,
        transformer_dropout: float = 0.1,
        in_dropout: float = 0.0,
        past_state_dropout: float = 0.8,
        with_rnn: bool = True,
        object_state_feedback: bool = True,
    ):
        super().__init__()
        self.input_size_imu = input_size_imu
        self.obj_imu_dim = obj_imu_dim
        self.n_sbps = n_sbps
        self.human_state_dim = 18 * 6 + 3 + n_sbps * 4
        self.output_dim = self.human_state_dim + 3
        self.with_rnn = with_rnn
        self.n_heads = n_heads
        self.object_state_feedback = bool(object_state_feedback)
        self.in_dropout = nn.Dropout(in_dropout)
        self.past_dropout = nn.Dropout(past_state_dropout)
        history_dim = self.human_state_dim + (
            3 if self.object_state_feedback else 0
        )
        self.in_linear = nn.Linear(
            input_size_imu + obj_imu_dim + history_dim, tf_input_dim
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_input_dim,
            nhead=n_heads,
            dim_feedforward=tf_hidden_size,
            # TIP leaves TransformerEncoderLayer at PyTorch's 0.1 default;
            # ``dropout`` is the separate recurrent-layer CLI parameter.
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.tf_encode = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        if with_rnn:
            self.rnn = nn.RNN(
                input_size=tf_input_dim,
                hidden_size=rnn_hidden_size,
                num_layers=1,
                nonlinearity="tanh",
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            )
            self.linear = nn.Linear(rnn_hidden_size, self.output_dim)
        else:
            self.rnn = None
            self.linear = nn.Linear(tf_input_dim, self.output_dim)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1)

    def _official_head_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        return x.reshape(batch_size, seq_len, self.n_heads, -1).transpose(2, 3).reshape(batch_size, seq_len, dim)

    @staticmethod
    def _smooth_official_outputs(history: list[torch.Tensor]) -> torch.Tensor:
        """Apply the six-frame output filter used by TIP's real-time runner."""
        current = history[-1]
        if len(history) < 6:
            return current
        window = torch.stack(history[-6:], dim=1)
        weights = current.new_tensor([0.6**5, 0.6**4, 0.6**3, 0.6**2, 0.6, 1.0])
        return (window * weights.view(1, 6, 1)).sum(dim=1) / weights.sum()

    def _state_for_feedback(self, state: torch.Tensor) -> torch.Tensor:
        """Convert raw SBP logits/scaled offsets into TIP's feedback representation."""
        feedback = state.detach().clone()
        n_sbps = int(getattr(self, "n_sbps", 0))
        if n_sbps <= 0:
            return feedback
        constraints = feedback[:, 111 : 111 + n_sbps * 4]
        constraints[:, 0::4] = (constraints[:, 0::4] > 0.0).to(constraints.dtype)
        constraints[:, 1::4] /= 5.0
        constraints[:, 2::4] /= 5.0
        constraints[:, 3::4] /= 5.0
        return feedback

    def forward(
        self,
        imu: torch.Tensor,
        prev_state: torch.Tensor,
        obj_imu: torch.Tensor,
        prev_obj_velocity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state = prev_state.clone().nan_to_num(0.0)
        # Keep TIP's no-teacher-forcing rule for historical root velocity.
        state[:, :, 18 * 6 : 18 * 6 + 3] = 0.0
        history = state
        if self.object_state_feedback:
            if prev_obj_velocity is None:
                prev_obj_velocity = torch.zeros(
                    *state.shape[:2], 3, device=state.device, dtype=state.dtype
                )
            if prev_obj_velocity.shape != (*state.shape[:2], 3):
                raise ValueError(
                    "prev_obj_velocity must have shape "
                    f"{[*state.shape[:2], 3]}, got {list(prev_obj_velocity.shape)}"
                )
            history = torch.cat(
                [state, prev_obj_velocity.nan_to_num(0.0)], dim=-1
            )
        x = torch.cat(
            [self.in_dropout(imu), obj_imu, self.past_dropout(history)], dim=-1
        )
        x = self.in_linear(x)
        x = self._official_head_shuffle(x)
        x = self.tf_encode(x, mask=self._causal_mask(x.shape[1], x.device))
        if self.rnn is not None:
            h0 = torch.zeros(1, x.shape[0], self.rnn.hidden_size, device=x.device, dtype=x.dtype)
            x, _ = self.rnn(x, h0)
        y = self.linear(x)
        return {
            "state": y[:, :, : self.human_state_dim],
            "obj_trans": y[:, :, self.human_state_dim :],
        }

    def infer_autoregressive(
        self,
        imu: torch.Tensor,
        obj_imu: torch.Tensor,
        window_size: int = 40,
        initial_state: torch.Tensor | None = None,
        emit_initial_state: bool = False,
        smooth_output: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run TIP causally using only the initial and previously predicted states.

        With ``emit_initial_state=True``, frame 0 is the supplied initial state and
        IMU/state at frame ``t`` predicts frame ``t + 1``, matching TIP training.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        batch_size, seq_len = imu.shape[:2]
        if seq_len <= 0:
            raise ValueError("TIP inference requires at least one frame")
        if initial_state is None:
            initial_state = torch.zeros(
                batch_size,
                self.human_state_dim,
                device=imu.device,
                dtype=imu.dtype,
            )
        if initial_state.shape != (batch_size, self.human_state_dim):
            raise ValueError(
                "initial_state must have shape "
                f"[{batch_size}, {self.human_state_dim}], got {list(initial_state.shape)}"
            )

        previous = torch.zeros(
            batch_size,
            seq_len,
            self.human_state_dim,
            device=imu.device,
            dtype=imu.dtype,
        )
        previous_object = torch.zeros(
            batch_size,
            seq_len,
            3,
            device=imu.device,
            dtype=imu.dtype,
        )
        previous[:, 0] = initial_state
        states = []
        objects = []
        raw_output_history: list[torch.Tensor] = []
        first_output = 0
        if emit_initial_state:
            states.append(initial_state)
            objects.append(torch.zeros(batch_size, 3, device=imu.device, dtype=imu.dtype))
            first_output = 1

        for frame in range(first_output, seq_len):
            observation_end = frame if emit_initial_state else frame + 1
            start = max(0, observation_end - window_size)
            if bool(getattr(self, "object_state_feedback", False)):
                output = self(
                    imu[:, start:observation_end],
                    previous[:, start:observation_end],
                    obj_imu[:, start:observation_end],
                    previous_object[:, start:observation_end],
                )
            else:
                output = self(
                    imu[:, start:observation_end],
                    previous[:, start:observation_end],
                    obj_imu[:, start:observation_end],
            )
            raw_state = output["state"][:, -1]
            raw_object = output["obj_trans"][:, -1]
            raw_output_history.append(torch.cat([raw_state, raw_object], dim=-1))
            filtered_output = (
                self._smooth_official_outputs(raw_output_history)
                if smooth_output
                else raw_output_history[-1]
            )
            state = filtered_output[:, : self.human_state_dim]
            obj_trans = filtered_output[:, self.human_state_dim :]
            states.append(state)
            objects.append(obj_trans)
            feedback_index = frame if emit_initial_state else frame + 1
            if feedback_index < seq_len:
                previous[:, feedback_index] = self._state_for_feedback(state)
                previous_object[:, feedback_index] = obj_trans.detach()

        return {
            "state": torch.stack(states, dim=1),
            "obj_trans": torch.stack(objects, dim=1),
            "includes_initial_state": emit_initial_state,
        }


class TIPReferenceHOIModel(nn.Module):
    """Exact public TIP object checkpoint architecture.

    The released ``tip_obj_*`` files are raw ``TF_RNN_Past_State`` state dicts:
    six human sensors plus one object sensor (63 channels), followed by a
    114-dimensional state (18 pose 2-axis values, root velocity, object
    velocity).  This class intentionally keeps the original parameter names so
    those files load strictly and remain useful as an inference reference.
    """

    def __init__(
        self,
        input_size_imu: int = 63,
        state_dim: int = 114,
        rnn_hid_size: int = 512,
        tf_hid_size: int = 1024,
        tf_in_dim: int = 256,
        n_heads: int = 16,
        tf_layers: int = 4,
        transformer_dropout: float = 0.1,
        past_state_dropout: float = 0.8,
    ):
        super().__init__()
        self.tip_variant = "official"
        self.tip_reference = True
        self.input_size_imu = int(input_size_imu)
        self.state_dim = int(state_dim)
        self.human_state_dim = self.state_dim
        self.output_dim = self.state_dim
        self.obj_imu_dim = 9
        self.n_sbps = 0
        self.n_heads = int(n_heads)
        self.in_dropout = nn.Dropout(0.0)
        self.past_dropout = nn.Dropout(float(past_state_dropout))
        self.in_linear = nn.Linear(self.input_size_imu + self.state_dim, tf_in_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=tf_in_dim,
            nhead=n_heads,
            dim_feedforward=tf_hid_size,
            dropout=transformer_dropout,
            # The reference passes [T, B, C] to this module.
            batch_first=False,
        )
        self.tf_encode = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.rnn = nn.RNN(
            input_size=tf_in_dim,
            hidden_size=rnn_hid_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.linear = nn.Linear(rnn_hid_size, self.state_dim)
        self.object_state_feedback = False

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )

    def forward(self, imu: torch.Tensor, state_history: torch.Tensor) -> dict[str, torch.Tensor]:
        if imu.ndim != 3 or state_history.ndim != 3:
            raise ValueError("TIP reference expects [B,T,C] IMU and state tensors")
        if imu.shape[:2] != state_history.shape[:2]:
            raise ValueError("TIP reference IMU/state history lengths must match")
        state = state_history.clone().nan_to_num(0.0)
        # Official code excludes root velocity from recurrent conditioning.
        state[:, :, 108:111] = 0.0
        # The released implementation instantiates ``nn.Dropout`` inside
        # ``forward``.  Consequently its 0.8 history dropout remains active
        # during ``eval()``; preserve that observable checkpoint behavior.
        x = torch.cat(
            [
                torch.nn.functional.dropout(imu, p=0.0, training=True),
                torch.nn.functional.dropout(
                    state, p=self.past_dropout.p, training=True
                ),
            ],
            dim=-1,
        )
        x = self.in_linear(x)
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len, self.n_heads, dim // self.n_heads)
        x = x.transpose(2, 3).reshape(batch_size, seq_len, dim)
        x = x.transpose(0, 1)
        x = self.tf_encode(x, mask=self._causal_mask(seq_len, x.device))
        x = x.transpose(0, 1)
        hidden = torch.zeros(
            1, batch_size, self.rnn.hidden_size, device=x.device, dtype=x.dtype
        )
        x, _ = self.rnn(x, hidden)
        state_out = self.linear(x)
        return {
            "state": state_out,
            "obj_trans": state_out[:, :, -3:],
            "tip_reference": torch.tensor(True, device=x.device),
        }

    @staticmethod
    def _identity_state(
        batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        state = torch.zeros(batch_size, 114, device=device, dtype=dtype)
        # State targets in the released OMOMO adapter are raw row-major 6D.
        identity6 = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device, dtype=dtype
        )
        state[:, :108] = identity6.repeat(18)
        return state

    @staticmethod
    def _smooth(history: list[torch.Tensor]) -> torch.Tensor:
        current = history[-1]
        if len(history) < 6:
            return current
        values = torch.stack(history[-6:], dim=1)
        weights = current.new_tensor([0.6**5, 0.6**4, 0.6**3, 0.6**2, 0.6, 1.0])
        return (values * weights.view(1, 6, 1)).sum(dim=1) / weights.sum()

    @torch.no_grad()
    def infer_autoregressive(
        self,
        imu: torch.Tensor,
        window_size: int = 40,
        initial_state: torch.Tensor | None = None,
        measured_root_6d: torch.Tensor | None = None,
        smooth_output: bool = True,
        emit_initial_state: bool = True,
    ) -> dict[str, torch.Tensor]:
        if imu.ndim != 3:
            raise ValueError("TIP reference inference expects IMU [B,T,63]")
        batch_size, seq_len = imu.shape[:2]
        if seq_len < 1:
            raise ValueError("TIP reference inference requires at least one frame")
        if initial_state is None:
            initial_state = self._identity_state(batch_size, imu.device, imu.dtype)
        if initial_state.shape != (batch_size, self.state_dim):
            raise ValueError(f"initial_state must have shape {(batch_size, self.state_dim)}")
        history = torch.zeros(
            batch_size, seq_len, self.state_dim, device=imu.device, dtype=imu.dtype
        )
        history[:, 0] = initial_state
        outputs: list[torch.Tensor] = []
        raw_history: list[torch.Tensor] = []
        if emit_initial_state:
            outputs.append(initial_state)
        start_frame = 1 if emit_initial_state else 0
        for frame in range(start_frame, seq_len):
            end = frame if emit_initial_state else frame + 1
            start = max(0, end - int(window_size))
            pred = self(imu[:, start:end], history[:, start:end])["state"][:, -1]
            raw_history.append(pred)
            filtered = self._smooth(raw_history) if smooth_output else pred
            outputs.append(filtered)
            feedback_idx = frame
            if feedback_idx < seq_len:
                history[:, feedback_idx] = filtered
        state = torch.stack(outputs, dim=1)
        return {
            "state": state,
            "obj_trans": state[:, :, -3:],
            "includes_initial_state": bool(emit_initial_state),
            "tip_reference": torch.tensor(True, device=imu.device),
        }
