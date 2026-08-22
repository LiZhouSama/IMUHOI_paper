from types import SimpleNamespace

import pytest
import torch
from pytorch3d.transforms import matrix_to_rotation_6d

import eval_batch_online
from model.rnn.velocity_contact import (
    build_velocity_contact_module,
    normalize_velocity_contact_arch,
)
from model.rnn.velocity_contact_v2 import V2VelocityContactModule
from train.rnn.loss import V2VelocityContactLoss


def _cfg():
    return SimpleNamespace(
        velocity_contact_arch="v2",
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        velocity_hidden_dim=16,
        velocity_num_layers=1,
        velocity_dropout=0.0,
        contact_from_segments=False,
        online_window=3,
        test=SimpleNamespace(window=3),
    )


def _inputs(batch_size=2, seq_len=5):
    rot6d = matrix_to_rotation_6d(torch.eye(3)).reshape(1, 1, 1, 6)
    human = torch.cat(
        (
            torch.randn(batch_size, seq_len, 6, 3),
            rot6d.expand(batch_size, seq_len, 6, 6),
        ),
        dim=-1,
    )
    obj = torch.cat(
        (
            torch.randn(batch_size, seq_len, 3),
            rot6d[:, :, 0].expand(batch_size, seq_len, 6),
        ),
        dim=-1,
    )
    return {
        "human_imu": human,
        "obj_imu": obj,
        "hand_vel_glb_init": torch.zeros(batch_size, 2, 3),
        "obj_vel_init": torch.zeros(batch_size, 3),
        "contact_init": torch.zeros(batch_size, 3),
    }


def test_v2_architecture_and_forward_contract():
    assert normalize_velocity_contact_arch("V2") == "v2"
    model = build_velocity_contact_module(_cfg()).eval()
    assert isinstance(model, V2VelocityContactModule)
    assert model.requires_hp is False
    assert model.obj_move_net.num_directions == 1
    assert model.hand_contact_net.num_directions == 1
    assert model.obj_move_net.linear1.in_features == 9
    assert model.hand_contact_net.linear1.in_features == 10

    with torch.no_grad():
        output = model(_inputs(), hp_out={"ignored": torch.randn(2, 5, 99)})
    assert output["pred_hand_glb_vel"].shape == (2, 5, 2, 3)
    assert output["pred_obj_vel"].shape == (2, 5, 3)
    assert output["pred_hand_contact_prob"].shape == (2, 5, 3)
    torch.testing.assert_close(
        output["pred_hand_contact_prob"][..., :2],
        output["pred_hand_contact_prob_direct"],
    )
    torch.testing.assert_close(
        output["pred_hand_contact_prob"][..., 2:3],
        output["pred_obj_move_prob"],
    )
    assert "pred_hand_contact_prob_cond" not in output
    assert output["pred_hand_contact_prob"].min() >= 0.0
    assert output["pred_hand_contact_prob"].max() <= 1.0


def test_v2_is_causal_when_future_imu_changes():
    model = build_velocity_contact_module(_cfg()).eval()
    original = _inputs(batch_size=1, seq_len=6)
    changed = {key: value.clone() for key, value in original.items()}
    changed["human_imu"][:, 4:, :, :3] += 100.0
    changed["obj_imu"][:, 4:, :3] -= 100.0
    with torch.no_grad():
        first = model(original)
        second = model(changed)
    for key in (
        "pred_hand_glb_vel",
        "pred_obj_vel",
        "pred_obj_move_prob",
        "pred_hand_contact_prob",
    ):
        torch.testing.assert_close(first[key][:, :4], second[key][:, :4])


def test_v2_direct_hand_loss_is_supervised_outside_object_contact():
    loss_fn = V2VelocityContactLoss(
        weights={
            "v2_hand_contact_pos_weight": 1.0,
            "v2_hand_contact_neg_weight": 1.0,
            "v2_hand_contact_hard_neg_weight": 1.0,
        }
    )
    batch = {
        "human_imu": torch.zeros(1, 2, 6, 9),
        "sensor_vel_glb": torch.zeros(1, 2, 6, 3),
        "obj_vel": torch.zeros(1, 2, 3),
        "lhand_contact": torch.tensor([[1.0, 0.0]]),
        "rhand_contact": torch.zeros(1, 2),
        "obj_contact": torch.zeros(1, 2),
    }
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    prediction = {
        "pred_hand_contact_logits_direct": logits,
        "pred_obj_move_logits": torch.zeros(1, 2, 1),
        "pred_hand_glb_vel": torch.zeros(1, 2, 2, 3),
        "pred_obj_vel": torch.zeros(1, 2, 3),
    }
    total, losses, _ = loss_fn(
        prediction,
        batch,
        torch.device("cpu"),
        active_loss_keys={"hand_contact"},
    )
    assert losses["hand_contact"].item() > 0.0
    total.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0.0


def test_v2_checkpoint_and_eval_parser_contract():
    model = build_velocity_contact_module(_cfg())
    model.validate_checkpoint_state_dict(model.state_dict())
    with pytest.raises(ValueError, match="not marked"):
        model.validate_checkpoint_state_dict({"vel_net.linear1.weight": torch.ones(1)})
    args = eval_batch_online._make_parser().parse_args(
        ["--velocity_contact_arch", "V2"]
    )
    assert args.velocity_contact_arch == "v2"
