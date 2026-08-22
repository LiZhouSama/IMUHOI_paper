from types import SimpleNamespace

import pytest
import torch
from pytorch3d.transforms import matrix_to_rotation_6d

import eval_batch_online
import vis_IMUHOI
from model.rnn.velocity_contact import (
    build_velocity_contact_module,
    normalize_velocity_contact_arch,
)
from model.rnn.velocity_contact_v3 import V3VelocityContactModule
from train.rnn.loss import VelocityContactLoss


def _cfg():
    return SimpleNamespace(
        velocity_contact_arch="v3",
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        velocity_hidden_dim=16,
        velocity_num_layers=1,
        velocity_dropout=0.0,
        boundary_group_hidden=8,
        boundary_hidden_dim=8,
        velocity_contact_geom_detach_obj_vel=True,
        contact_from_segments=False,
        online_window=4,
        test=SimpleNamespace(window=4),
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
    data = {
        "human_imu": human,
        "obj_imu": obj,
        "hand_vel_glb_init": torch.zeros(batch_size, 2, 3),
        "obj_vel_init": torch.zeros(batch_size, 3),
        "obj_trans_init": torch.zeros(batch_size, 3),
        "contact_init": torch.zeros(batch_size, 3),
    }
    hp_out = {
        "pred_full_pose_6d": rot6d.expand(batch_size, seq_len, 24, 6),
        "pred_joints_local": torch.randn(batch_size, seq_len, 24, 3),
        "pred_palm_glb_pos": torch.randn(batch_size, seq_len, 2, 3),
        "root_vel_pred": torch.randn(batch_size, seq_len, 3),
        "root_trans_pred": torch.randn(batch_size, seq_len, 3),
    }
    return data, hp_out


def test_v3_is_boundary_plus_bidirectional_obj_move_only():
    assert normalize_velocity_contact_arch("V3") == "v3"
    model = build_velocity_contact_module(_cfg()).eval()
    assert isinstance(model, V3VelocityContactModule)
    assert model.requires_hp is True
    assert model.use_contact_geom_features is False
    assert model.use_boundary_features is True
    assert model.use_contact_endpoints is False
    assert model.obj_move_bidirectional is True
    assert model.obj_move_net.num_directions == 2
    assert model.hand_contact_net.num_directions == 1
    assert model.boundary_rnn.bidirectional is False
    assert model.hand_contact_input_dim == 36 + 8
    assert not hasattr(model, "interaction_net")
    assert not hasattr(model, "interaction_history_net")
    assert not hasattr(model, "hand_contact_head")

    data, hp_out = _inputs()
    with torch.no_grad():
        output = model(data, hp_out=hp_out)
    assert output["pred_hand_glb_vel"].shape == (2, 5, 2, 3)
    assert output["pred_obj_vel"].shape == (2, 5, 3)
    assert output["pred_hand_contact_prob"].shape == (2, 5, 3)
    torch.testing.assert_close(
        output["pred_hand_contact_prob"][..., :2],
        output["pred_obj_move_prob"]
        * output["pred_hand_contact_prob_cond"],
    )
    torch.testing.assert_close(
        output["pred_hand_contact_prob_marginal"],
        output["pred_hand_contact_prob"][..., :2],
    )
    assert "pred_contact_future_valid_mask" not in output


def test_v3_trains_conditional_and_marginal_contact_paths():
    model = build_velocity_contact_module(_cfg()).train()
    data, hp_out = _inputs(batch_size=1, seq_len=4)
    prediction = model(data, hp_out=hp_out)
    batch = {
        **data,
        "sensor_vel_glb": torch.zeros(1, 4, 6, 3),
        "obj_vel": torch.zeros(1, 4, 3),
        "lhand_contact": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        "rhand_contact": torch.tensor([[0.0, 1.0, 0.0, 1.0]]),
        "obj_contact": torch.ones(1, 4),
    }
    total, losses, _ = VelocityContactLoss()(
        prediction,
        batch,
        torch.device("cpu"),
        active_loss_keys={
            "obj_move",
            "hand_contact_cond",
            "hand_contact_marginal",
        },
    )
    assert losses["obj_move"].item() > 0.0
    assert losses["hand_contact_cond"].item() > 0.0
    assert losses["hand_contact_marginal"].item() > 0.0
    total.backward()
    assert model.obj_move_net.linear2.weight.grad is not None
    assert model.obj_move_net.linear2.weight.grad.abs().sum() > 0.0
    assert model.hand_contact_net.linear2.weight.grad is not None
    assert model.hand_contact_net.linear2.weight.grad.abs().sum() > 0.0


def test_v3_checkpoint_marker_and_parser_contract():
    model = build_velocity_contact_module(_cfg())
    state = model.state_dict()
    model.validate_checkpoint_state_dict(state)
    old_v3_state = dict(state)
    old_v3_state["_v3_arch_version"] = torch.tensor(31)
    with pytest.raises(ValueError, match="not marked"):
        model.validate_checkpoint_state_dict(old_v3_state)
    args = eval_batch_online._make_parser().parse_args(
        ["--velocity_contact_arch", "V3"]
    )
    assert args.velocity_contact_arch == "v3"
    vis_args = vis_IMUHOI._make_parser().parse_args(
        ["--velocity_contact_arch", "V3"]
    )
    assert vis_args.velocity_contact_arch == "v3"
