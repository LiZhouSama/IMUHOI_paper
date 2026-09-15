import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Comparisons.tip.model import TIPHOIModel
from Comparisons.common.losses import TIPLoss


class _RecordingTIP(TIPHOIModel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.human_state_dim = 4
        self.seen_previous = []

    def forward(self, imu, prev_state, obj_imu):
        del imu, obj_imu
        self.seen_previous.append(prev_state.detach().clone())
        state = prev_state + 1.0
        return {
            "state": state,
            "obj_trans": state[..., :3],
        }


class _SBPTIP(_RecordingTIP):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.n_sbps = 1
        self.human_state_dim = 115
        self.seen_previous = []

    def forward(self, imu, prev_state, obj_imu):
        del imu, obj_imu
        self.seen_previous.append(prev_state.detach().clone())
        state = prev_state.clone()
        state[..., 111:115] = torch.tensor(
            [-2.0, 10.0, -5.0, 2.5], device=state.device, dtype=state.dtype
        )
        return {"state": state, "obj_trans": state[..., :3]}


class _ObjectFeedbackTIP(_RecordingTIP):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.human_state_dim = 4
        self.object_state_feedback = True
        self.seen_object = []

    def forward(self, imu, prev_state, obj_imu, prev_obj_velocity=None):
        del imu, obj_imu
        self.seen_object.append(prev_obj_velocity.detach().clone())
        next_velocity = prev_obj_velocity + 1.0
        return {"state": prev_state, "obj_trans": next_velocity}


def test_tip_autoregressive_uses_predictions_as_history():
    model = _RecordingTIP()
    imu = torch.zeros(1, 4, 2)
    obj_imu = torch.zeros(1, 4, 2)
    output = model.infer_autoregressive(imu, obj_imu, window_size=3)

    assert torch.allclose(output["state"][0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(model.seen_previous[0], torch.zeros(1, 1, 4))
    assert torch.allclose(
        model.seen_previous[2][0, :, 0], torch.tensor([0.0, 1.0, 2.0])
    )
    assert torch.allclose(
        model.seen_previous[3][0, :, 0], torch.tensor([1.0, 2.0, 3.0])
    )


def test_tip_next_state_initialization_and_sbp_feedback_scaling():
    model = _SBPTIP()
    imu = torch.zeros(1, 3, 2)
    obj_imu = torch.zeros(1, 3, 2)
    initial = torch.zeros(1, 115)
    initial[:, 0] = 7.0
    output = model.infer_autoregressive(
        imu,
        obj_imu,
        initial_state=initial,
        emit_initial_state=True,
        smooth_output=False,
    )

    assert output["state"].shape == (1, 3, 115)
    assert output["state"][0, 0, 0] == 7.0
    assert output["includes_initial_state"] is True
    feedback = model.seen_previous[1][0, 1, 111:115]
    assert torch.allclose(feedback, torch.tensor([0.0, 2.0, -1.0, 0.5]))


def test_tip_official_six_frame_output_filter():
    history = [torch.full((1, 4), float(index)) for index in range(1, 7)]
    smoothed = TIPHOIModel._smooth_official_outputs(history)
    weights = torch.tensor([0.6**5, 0.6**4, 0.6**3, 0.6**2, 0.6, 1.0])
    expected = (torch.arange(1.0, 7.0) * weights).sum() / weights.sum()
    assert torch.allclose(smoothed, torch.full((1, 4), expected))

    model = _ObjectFeedbackTIP()
    output = model.infer_autoregressive(
        torch.zeros(1, 6, 2),
        torch.zeros(1, 6, 2),
        smooth_output=True,
    )
    assert torch.allclose(output["obj_trans"][0, -1], torch.full((3,), expected))


def test_tip_autoregressive_feeds_back_predicted_object_velocity():
    model = _ObjectFeedbackTIP()
    output = model.infer_autoregressive(
        torch.zeros(1, 4, 2),
        torch.zeros(1, 4, 2),
        window_size=3,
        smooth_output=False,
    )
    assert torch.allclose(
        output["obj_trans"][0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0])
    )
    assert torch.allclose(model.seen_object[0], torch.zeros(1, 1, 3))
    assert torch.allclose(
        model.seen_object[2][0, :, 0], torch.tensor([0.0, 1.0, 2.0])
    )


def test_tip_loss_excludes_external_initial_state():
    loss_fn = TIPLoss(n_sbps=0, jerk_weight=0.0, obj_weight=1.0)
    target = torch.zeros(1, 3, 111)
    output = {
        "state": target.clone(),
        "obj_trans": torch.zeros(1, 3, 3),
        "includes_initial_state": True,
    }
    # The external initial frame can be arbitrary without changing validation loss.
    output["state"][:, 0] = 1000.0
    output["obj_trans"][:, 0] = 1000.0
    adapted = {
        "state_target": target,
        "sbp_valid": torch.zeros(1, 3, dtype=torch.bool),
        "obj_vel_target": torch.zeros(1, 3, 3),
        "has_object": torch.ones(1, dtype=torch.bool),
    }
    assert torch.allclose(loss_fn(output, adapted)["loss"], torch.tensor(0.0))
