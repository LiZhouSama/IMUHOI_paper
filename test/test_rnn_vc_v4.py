from types import SimpleNamespace

import pytest
import torch
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    random_rotations,
    rotation_6d_to_matrix,
)

import eval_batch_online
import vis_IMUHOI
from model.rnn.velocity_contact import (
    build_velocity_contact_module,
    normalize_velocity_contact_arch,
)
from model.rnn.velocity_contact_v4 import (
    V4VelocityContactModule,
    causal_moving_average,
    heading_rotation_matrix,
)
from train.rnn.loss.velocity_contact_loss import VelocityContactLoss
from train.rnn.scheduled_inputs import build_gt_human_pose_outputs


def _cfg(**overrides):
    cfg = SimpleNamespace(
        velocity_contact_arch="v4",
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        velocity_hidden_dim=16,
        velocity_num_layers=1,
        velocity_dropout=0.0,
        boundary_group_hidden=8,
        boundary_hidden_dim=8,
        v4_anticipation_horizon=4,
        contact_from_segments=False,
        online_window=4,
        test=SimpleNamespace(window=4),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _inputs(batch_size=2, seq_len=6):
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
    rotations = random_rotations(batch_size * seq_len * 24)
    hp_out = {
        "pred_full_pose_6d": matrix_to_rotation_6d(rotations).reshape(
            batch_size, seq_len, 24, 6
        ),
        "pred_joints_local": torch.randn(batch_size, seq_len, 24, 3),
        "pred_palm_glb_pos": torch.randn(batch_size, seq_len, 2, 3),
        "root_vel_pred": torch.randn(batch_size, seq_len, 3),
        "root_trans_pred": torch.randn(batch_size, seq_len, 3),
    }
    return data, hp_out


def _yaw_matrix(angle):
    cos = torch.cos(torch.tensor(angle))
    sin = torch.sin(torch.tensor(angle))
    return torch.tensor(
        [
            [cos, 0.0, sin],
            [0.0, 1.0, 0.0],
            [-sin, 0.0, cos],
        ]
    )


def test_v4_topology_feeds_the_pose_prior_into_both_contact_heads():
    assert normalize_velocity_contact_arch("V4") == "v4"
    model = build_velocity_contact_module(_cfg()).eval()
    assert isinstance(model, V4VelocityContactModule)
    assert model.requires_hp is True
    assert model.use_contact_geom_features is False
    assert model.use_boundary_features is True

    assert model.boundary_rnn.bidirectional is True
    assert model.obj_move_net.num_directions == 2
    assert model.hand_contact_net.num_directions == 1
    # obj_move now sees the boundary state, so the pose prior is not capped by
    # an IMU-only P(object moving) at onsets.
    assert model.obj_move_input_dim == 27 + 8
    assert model.obj_move_net.linear1.in_features == 27 + 8
    # base evidence (36) + boundary state (8) + hand vel/acc (12), no geometry.
    assert model.hand_contact_input_dim == 36 + 8 + 12
    assert model.hand_contact_net.linear1.in_features == 36 + 8 + 12

    # Both auxiliary heads read boundary features only.
    assert model.contact_endpoint_rnn.input_size == 8
    assert model.contact_anticipation_head[0].in_features == 8

    data, hp_out = _inputs()
    with torch.no_grad():
        output = model(data, hp_out=hp_out)
    assert output["pred_hand_glb_vel"].shape == (2, 6, 2, 3)
    assert output["pred_obj_vel"].shape == (2, 6, 3)
    assert output["pred_hand_contact_prob"].shape == (2, 6, 3)
    assert output["pred_contact_endpoint_prob"].shape == (2, 6, 6)
    assert output["pred_contact_from_endpoints_prob"].shape == (2, 6, 3)
    assert output["pred_contact_anticipation_prob"].shape == (2, 6, 6)
    assert "pred_obj_trans_rough" not in output
    torch.testing.assert_close(
        output["pred_hand_contact_prob"][..., :2],
        output["pred_obj_move_prob"] * output["pred_hand_contact_prob_cond"],
    )
    torch.testing.assert_close(
        output["pred_hand_contact_prob_marginal"],
        output["pred_hand_contact_prob"][..., :2],
    )


def test_v4_velocity_only_needs_no_pose_but_contact_raises_without_it():
    model = build_velocity_contact_module(_cfg()).eval()
    data, hp_out = _inputs()

    with torch.no_grad():
        velocity_out = model(data, hp_out=None, velocity_only=True)
    assert velocity_out["pred_obj_vel"].shape == (2, 6, 3)
    assert "pred_hand_contact_prob" not in velocity_out

    with pytest.raises(ValueError, match="requires HumanPose outputs"):
        model(data, hp_out=None)

    missing_key = {k: v for k, v in hp_out.items() if k != "root_vel_pred"}
    with pytest.raises(ValueError, match="root_vel_pred"):
        model(data, hp_out=missing_key)

    wrong_shape = dict(hp_out)
    wrong_shape["pred_joints_local"] = torch.randn(2, 6, 22, 3)
    with pytest.raises(ValueError, match="pred_joints_local"):
        model(data, hp_out=wrong_shape)

    no_hands = {k: v for k, v in hp_out.items() if k != "pred_palm_glb_pos"}
    with pytest.raises(ValueError, match="pred_palm_glb_pos"):
        model(data, hp_out=no_hands)

    # Correctly shaped but empty: HumanPose emits this when its body model is
    # missing, and it would otherwise train a silently disabled prior.
    zeroed = build_velocity_contact_module(_cfg()).eval()
    empty_pose = dict(hp_out)
    empty_pose["pred_joints_local"] = torch.zeros(2, 6, 24, 3)
    with pytest.raises(ValueError, match="all-zero"):
        zeroed(data, hp_out=empty_pose)


def test_v4_pose_prior_is_heading_invariant():
    model = build_velocity_contact_module(_cfg()).eval()
    data, hp_out = _inputs()
    yaw = _yaw_matrix(0.7)

    pose_6d = hp_out["pred_full_pose_6d"]
    rotated = dict(hp_out)
    rotated["pred_full_pose_6d"] = matrix_to_rotation_6d(
        torch.matmul(yaw, rotation_6d_to_matrix(pose_6d.reshape(-1, 6)))
    ).reshape(pose_6d.shape)
    for key in ("pred_joints_local", "pred_palm_glb_pos", "root_vel_pred", "root_trans_pred"):
        rotated[key] = torch.matmul(hp_out[key], yaw.transpose(0, 1))

    with torch.no_grad():
        base = model(data, hp_out=hp_out)
        turned = model(data, hp_out=rotated)

    # Heading is removed from the pose prior, so the pose-only heads are
    # unchanged by turning the whole scene about the vertical axis.
    torch.testing.assert_close(
        base["pred_contact_endpoint_logits"],
        turned["pred_contact_endpoint_logits"],
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        base["pred_contact_anticipation_logits"],
        turned["pred_contact_anticipation_logits"],
        atol=1e-4,
        rtol=1e-4,
    )

    plain = build_velocity_contact_module(_cfg(v4_heading_normalize=False)).eval()
    with torch.no_grad():
        plain_base = plain(data, hp_out=hp_out)
        plain_turned = plain(data, hp_out=rotated)
    assert not torch.allclose(
        plain_base["pred_contact_endpoint_logits"],
        plain_turned["pred_contact_endpoint_logits"],
        atol=1e-4,
    )


def test_v4_boundary_options_are_ablatable():
    causal = build_velocity_contact_module(
        _cfg(v4_boundary_bidirectional=False)
    ).eval()
    assert causal.boundary_rnn.bidirectional is False
    assert isinstance(causal.boundary_fusion, torch.nn.Identity)

    no_acc = build_velocity_contact_module(
        _cfg(v4_boundary_use_joint_acc=False)
    ).eval()
    assert len(no_acc.group_acc_norms) == 0
    # left_arm: 5 joints -> pose 30 + vel 15 + root vel 3, no acceleration.
    assert no_acc.group_encoders["left_arm"].input_size == 30 + 15 + 3

    data, hp_out = _inputs()
    with torch.no_grad():
        assert no_acc(data, hp_out=hp_out)["pred_hand_contact_prob"].shape == (2, 6, 3)


def test_causal_moving_average_and_heading_helpers():
    value = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    smoothed = causal_moving_average(value, 3)
    # Left edge replicates, so early frames stay on the signal instead of
    # decaying toward zero.
    torch.testing.assert_close(smoothed[0, 0, 0], torch.tensor(0.0))
    torch.testing.assert_close(smoothed[0, 2, 0], torch.tensor(1.0))

    changed = value.clone()
    changed[0, -1, 0] = 100.0
    changed_smoothed = causal_moving_average(changed, 3)
    torch.testing.assert_close(smoothed[:, :-1], changed_smoothed[:, :-1])

    rotations = random_rotations(8)
    yaw = heading_rotation_matrix(rotations)
    torch.testing.assert_close(
        torch.matmul(yaw, yaw.transpose(-1, -2)),
        torch.eye(3).expand(8, 3, 3),
        atol=1e-5,
        rtol=1e-5,
    )
    # A pure yaw is recovered exactly.
    pure = _yaw_matrix(0.35).unsqueeze(0)
    torch.testing.assert_close(heading_rotation_matrix(pure), pure, atol=1e-6, rtol=1e-6)


def test_v4_anticipation_target_pools_only_future_frames():
    loss_fn = VelocityContactLoss(weights={}, anticipation_horizon=4)
    endpoint = torch.zeros(1, 12, 6)
    endpoint[0, 10, 0] = 1.0
    target, valid = loss_fn._future_transition_target(endpoint, 4)

    # max over (t, t + 4] hits frame 10 for t in [6, 9] only.
    assert target[0, :, 0].nonzero().flatten().tolist() == [6, 7, 8, 9]
    # Frames whose horizon runs past the window end carry no usable label.
    assert valid[0].nonzero().flatten().tolist() == list(range(0, 8))


def test_v4_anticipation_loss_is_active_and_masked():
    loss_fn = VelocityContactLoss(
        weights={"contact_anticipation": 1.0},
        anticipation_horizon=4,
    )
    batch_size, seq_len = 2, 12
    endpoint = torch.zeros(batch_size, seq_len, 6)
    endpoint[:, 6, 0] = 1.0
    logits = torch.zeros(batch_size, seq_len, 6, requires_grad=True)
    batch = {
        "human_imu": torch.randn(batch_size, seq_len, 6, 9),
        "contact_endpoint": endpoint,
        "has_object": torch.ones(batch_size, dtype=torch.bool),
    }

    total, losses, weighted = loss_fn(
        {"pred_contact_anticipation_logits": logits},
        batch,
        torch.device("cpu"),
        active_loss_keys={"contact_anticipation"},
    )
    assert losses["contact_anticipation"] > 0.0
    total.backward()
    # Only frames inside a complete horizon may contribute gradient.
    assert torch.count_nonzero(logits.grad[:, seq_len - 4:]) == 0

    batch["has_object"] = torch.zeros(batch_size, dtype=torch.bool)
    _, no_object_losses, _ = loss_fn(
        {"pred_contact_anticipation_logits": logits.detach()},
        batch,
        torch.device("cpu"),
        active_loss_keys={"contact_anticipation"},
    )
    assert float(no_object_losses["contact_anticipation"]) == 0.0


def test_v4_trains_end_to_end_on_scheduled_gt_pose_outputs():
    """The scheduled-sampling GT hp_out must satisfy v4's strict pose contract."""
    torch.manual_seed(0)
    batch_size, seq_len = 2, 16
    model = build_velocity_contact_module(_cfg()).train()
    data, _ = _inputs(batch_size, seq_len)

    endpoint = torch.zeros(batch_size, seq_len, 6)
    endpoint[:, 5, 4] = 1.0
    endpoint[:, 11, 5] = 1.0
    obj_contact = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    obj_contact[:, 5:12] = True
    batch = {
        "human_imu": data["human_imu"],
        "obj_imu": data["obj_imu"],
        "trans": torch.randn(batch_size, seq_len, 3),
        "root_vel": torch.randn(batch_size, seq_len, 3),
        "position_global": torch.randn(batch_size, seq_len, 24, 3),
        "rotation_global": random_rotations(batch_size * seq_len * 24).reshape(
            batch_size, seq_len, 24, 3, 3
        ),
        "obj_vel": torch.randn(batch_size, seq_len, 3),
        "sensor_vel_glb": torch.randn(batch_size, seq_len, 6, 3),
        "lhand_contact": obj_contact.clone(),
        "rhand_contact": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "obj_contact": obj_contact,
        "contact_endpoint": endpoint,
        "contact_endpoint_gauss": endpoint,
        "has_object": torch.ones(batch_size, dtype=torch.bool),
    }

    hp_out = build_gt_human_pose_outputs(batch, torch.device("cpu"))
    output = model(data, hp_out=hp_out)
    loss_fn = VelocityContactLoss(
        weights={key: 1.0 for key in VelocityContactLoss.LOSS_KEYS},
        anticipation_horizon=4,
    )
    total, losses, _ = loss_fn(output, batch, torch.device("cpu"))

    assert torch.isfinite(total)
    for key in ("obj_move", "hand_contact_cond", "hand_contact_marginal",
                "contact_endpoint", "contact_anticipation"):
        assert losses[key] > 0.0, key
    total.backward()

    # The auxiliary heads exist to shape the body encoder, so gradient must
    # reach it through them and not stop at the heads.
    boundary_grad = model.boundary_rnn.weight_ih_l0.grad
    assert boundary_grad is not None and torch.count_nonzero(boundary_grad) > 0


def test_v4_staged_training_drops_the_endpoint_only_tail():
    """v4's endpoint head reads frozen boundary features in that stage."""
    from train.rnn.train_velocity_contact import (
        _staged_lr_lambda,
        _vc_endpoint_finetune_stage,
        _vc_parameter_group,
    )

    v4 = build_velocity_contact_module(_cfg())
    endpoint_arch = build_velocity_contact_module(
        _cfg(velocity_contact_arch="imuhoi_cond_boundary_endpoint")
    )
    assert _vc_endpoint_finetune_stage(v4) is False
    assert _vc_endpoint_finetune_stage(endpoint_arch) is True

    groups = {name: _vc_parameter_group(name) for name, _ in v4.named_parameters()}
    assert groups["hand_vel_net.linear1.weight"] == "velocity"
    assert groups["obj_vel_net.linear1.weight"] == "velocity"
    assert groups["boundary_rnn.weight_ih_l0"] == "contact"
    assert groups["group_vel_norms.left_arm.weight"] == "contact"
    assert groups["root_vel_norm.weight"] == "contact"
    assert groups["obj_move_net.linear1.weight"] == "contact"
    # The anticipation head belongs with the encoder it supervises.
    assert groups["contact_anticipation_head.0.weight"] == "contact"
    assert groups["contact_endpoint_head.weight"] == "endpoint"

    stage_cfg = SimpleNamespace(
        vc_velocity_stage_epochs=10,
        vc_joint_stage_epochs=10,
        vc_group_lr_gamma=1.0,
    )
    without_tail = {
        group: _staged_lr_lambda(group, stage_cfg, endpoint_stage=False)
        for group in ("velocity", "contact", "endpoint")
    }
    with_tail = {
        group: _staged_lr_lambda(group, stage_cfg, endpoint_stage=True)
        for group in ("velocity", "contact", "endpoint")
    }
    # Velocity stage is identical either way.
    assert without_tail["velocity"](0) > 0.0
    assert without_tail["contact"](0) == 0.0
    # Past velocity + joint, the three-stage schedule parks the contact group at
    # zero; without a tail the joint stage must keep it learning.
    assert with_tail["contact"](25) == 0.0
    assert without_tail["contact"](25) > 0.0
    assert without_tail["endpoint"](25) > 0.0


def test_v4_checkpoint_marker_and_train_eval_vis_parser_contract():
    model = build_velocity_contact_module(_cfg())
    state = model.state_dict()
    model.validate_checkpoint_state_dict(state)

    old_state = dict(state)
    old_state["_v4_arch_version"] = torch.tensor(41)
    with pytest.raises(ValueError, match="not marked"):
        model.validate_checkpoint_state_dict(old_state)

    other_horizon = dict(state)
    other_horizon["_v4_anticipation_horizon"] = torch.tensor(30)
    with pytest.raises(ValueError, match="anticipation horizon"):
        model.validate_checkpoint_state_dict(other_horizon)

    eval_args = eval_batch_online._make_parser().parse_args(
        ["--velocity_contact_arch", "V4"]
    )
    assert eval_args.velocity_contact_arch == "v4"
    vis_args = vis_IMUHOI._make_parser().parse_args(
        ["--velocity_contact_arch", "V4"]
    )
    assert vis_args.velocity_contact_arch == "v4"
