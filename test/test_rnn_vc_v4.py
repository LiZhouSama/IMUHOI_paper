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
from model.rnn.velocity_contact_v4 import V4VelocityContactModule


def _cfg():
    return SimpleNamespace(
        velocity_contact_arch="v4",
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


def test_v4_is_v3_plus_geometry():
    assert normalize_velocity_contact_arch("V4") == "v4"
    model = build_velocity_contact_module(_cfg()).eval()
    assert isinstance(model, V4VelocityContactModule)
    assert model.requires_hp is True
    assert model.use_contact_geom_features is True
    assert model.use_boundary_features is True
    assert model.obj_move_bidirectional is True
    assert model.obj_move_net.num_directions == 2
    assert model.hand_contact_net.num_directions == 1
    assert model.boundary_rnn.bidirectional is False
    assert model.contact_geom_dim == 16
    assert model.hand_contact_input_dim == 36 + 8 + 16

    data, hp_out = _inputs()
    with torch.no_grad():
        output = model(data, hp_out=hp_out)
    assert output["pred_hand_glb_vel"].shape == (2, 5, 2, 3)
    assert output["pred_obj_vel"].shape == (2, 5, 3)
    assert output["pred_hand_contact_prob"].shape == (2, 5, 3)
    assert output["pred_obj_trans_rough"].shape == (2, 5, 3)
    torch.testing.assert_close(
        output["pred_hand_contact_prob"][..., :2],
        output["pred_obj_move_prob"] * output["pred_hand_contact_prob_cond"],
    )


def test_v4_checkpoint_marker_and_train_eval_vis_parser_contract():
    model = build_velocity_contact_module(_cfg())
    state = model.state_dict()
    model.validate_checkpoint_state_dict(state)
    old_state = dict(state)
    old_state["_v4_arch_version"] = torch.tensor(40)
    with pytest.raises(ValueError, match="not marked"):
        model.validate_checkpoint_state_dict(old_state)

    eval_args = eval_batch_online._make_parser().parse_args(
        ["--velocity_contact_arch", "V4"]
    )
    assert eval_args.velocity_contact_arch == "v4"
    vis_args = vis_IMUHOI._make_parser().parse_args(
        ["--velocity_contact_arch", "V4"]
    )
    assert vis_args.velocity_contact_arch == "v4"
