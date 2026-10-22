"""object_trans_nocontact_lock_vertical: the IMU branch keeps the height, at inference only."""
from types import SimpleNamespace

import torch

from configs import FRAME_RATE
from model.rnn.object_trans_v2 import V2ObjectTransModule


def _cfg(lock):
    return SimpleNamespace(
        imu_dim=9,
        obj_imu_dim=9,
        num_human_imus=6,
        hidden_dim_multiplier=1,
        object_trans_hidden_dim=16,
        object_trans_num_layers=2,
        object_trans_dropout=0.0,
        object_trans_state_feedback=False,
        enable_ot_refine=False,
        object_trans_nocontact_lock_vertical=lock,
        train={"window": 4},
        test={"window": 4},
    )


def _run(model, seq_len=4):
    obj_imu = torch.zeros(1, seq_len, 9)
    obj_imu[..., 3:9] = torch.eye(3)[:, :2].reshape(6)
    # Object moving, no hand contact: the fixed v2 fusion puts all weight on the IMU branch.
    contact = torch.tensor([0.0, 0.0, 1.0]).expand(1, seq_len, 3)
    obj_vel = torch.tensor([0.3, -0.6, 0.9]).expand(1, seq_len, 3)
    return model(
        torch.zeros(1, seq_len, 2, 3),
        contact,
        torch.tensor([[0.1, 0.5, -0.2]]),
        obj_imu=obj_imu,
        human_imu=torch.zeros(1, seq_len, 6, 9),
        obj_vel_input=obj_vel,
        enable_refine=False,
    )


def test_lock_zeroes_the_vertical_imu_displacement_at_inference():
    torch.manual_seed(0)
    model = V2ObjectTransModule(_cfg(True)).eval()
    with torch.no_grad():
        out = _run(model)
    step = out["pred_imu_obj_trans"] - out["pred_static_obj_trans"]
    assert torch.allclose(out["gating_weights"][..., 2], torch.ones(1, 4))
    assert torch.equal(step[..., 1], torch.zeros(1, 4))
    assert torch.allclose(step[..., 0], torch.full((1, 4), 0.3 / FRAME_RATE))
    assert torch.allclose(step[..., 2], torch.full((1, 4), 0.9 / FRAME_RATE))
    assert torch.allclose(out["pred_obj_trans"][..., 1], torch.full((1, 4), 0.5))


def test_lock_is_inference_only_and_off_by_default():
    torch.manual_seed(0)
    locked_train = V2ObjectTransModule(_cfg(True)).train()
    unlocked = V2ObjectTransModule(_cfg(False)).eval()
    for model in (locked_train, unlocked):
        with torch.no_grad():
            out = _run(model)
        step = out["pred_imu_obj_trans"] - out["pred_static_obj_trans"]
        assert torch.allclose(step[..., 1], torch.full((1, 4), -0.6 / FRAME_RATE))
    # Checkpoints stay loadable: the axis mask is not part of the state dict.
    assert "_horizontal_axes" not in unlocked.state_dict()
