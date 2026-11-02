from types import SimpleNamespace

import pytest
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    random_rotations,
    rotation_6d_to_matrix,
)

from model.rnn.velocity_contact import build_velocity_contact_module, normalize_velocity_contact_arch
from model.rnn.velocity_contact_v4 import V4VelocityContactModule
from model.rnn.velocity_contact_v5 import (
    V5VelocityContactModule,
    centred_average,
    rotation_rate,
)
from train.rnn.loss.velocity_contact_loss import VelocityContactLoss


def _cfg(**overrides):
    cfg = SimpleNamespace(
        velocity_contact_arch="v5",
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


def _rotation_walk(count, seq_len, step=0.1):
    """Random start orientations followed by small per-frame rotations, like real IMU streams.

    Frame-to-frame rotations of independent random orientations sit near 180 degrees,
    where acos() loses the precision the invariance checks need.
    """
    rotations = [random_rotations(count)]
    for _ in range(seq_len - 1):
        increment = axis_angle_to_matrix(torch.randn(count, 3) * step)
        rotations.append(rotations[-1] @ increment)
    return torch.stack(rotations, dim=1)


def _world_inputs(batch_size=2, seq_len=12):
    """World-frame human IMU (as after _denormalize_imu), object IMU and palms."""
    torch.manual_seed(0)
    human_rot = _rotation_walk(batch_size * 6, seq_len).reshape(batch_size, 6, seq_len, 3, 3).transpose(1, 2)
    human = torch.cat((torch.randn(batch_size, seq_len, 6, 3), matrix_to_rotation_6d(human_rot)), dim=-1)
    obj_rot = _rotation_walk(batch_size, seq_len)
    obj = torch.cat((torch.randn(batch_size, seq_len, 3), matrix_to_rotation_6d(obj_rot)), dim=-1)
    palms = torch.randn(batch_size, seq_len, 2, 3)
    return human, obj, palms


def _model_inputs(batch_size=2, seq_len=12):
    rot6d = matrix_to_rotation_6d(random_rotations(batch_size * seq_len * 6)).reshape(batch_size, seq_len, 6, 6)
    human = torch.cat((torch.randn(batch_size, seq_len, 6, 3), rot6d), dim=-1)
    obj = torch.cat(
        (torch.randn(batch_size, seq_len, 3), matrix_to_rotation_6d(random_rotations(batch_size * seq_len)).reshape(batch_size, seq_len, 6)),
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
        "pred_full_pose_6d": matrix_to_rotation_6d(random_rotations(batch_size * seq_len * 24)).reshape(
            batch_size, seq_len, 24, 6
        ),
        "pred_joints_local": torch.randn(batch_size, seq_len, 24, 3),
        "pred_palm_glb_pos": torch.randn(batch_size, seq_len, 2, 3),
        "root_vel_pred": torch.randn(batch_size, seq_len, 3),
        "root_trans_pred": torch.randn(batch_size, seq_len, 3),
    }
    return data, hp_out


def _yaw(angle):
    cos, sin = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    return torch.tensor([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def _rotate_world(human, obj, palms, rotation):
    """Apply one world rotation to every world-frame quantity (column-vector convention)."""
    batch_size, seq_len = human.shape[:2]
    hrot = rotation @ rotation_6d_to_matrix(human[..., 3:9])
    orot = rotation @ rotation_6d_to_matrix(obj[..., 3:9])
    human = torch.cat((human[..., :3] @ rotation.T, matrix_to_rotation_6d(hrot)), dim=-1)
    obj = torch.cat((obj[..., :3] @ rotation.T, matrix_to_rotation_6d(orot)), dim=-1)
    return human, obj, palms @ rotation.T


def test_v5_builds_and_widens_only_the_contact_nets():
    assert normalize_velocity_contact_arch("V5") == "v5"
    model = build_velocity_contact_module(_cfg()).eval()
    assert isinstance(model, V5VelocityContactModule)
    reference = V4VelocityContactModule(_cfg(velocity_contact_arch="v4"))
    # 2 hands x 3 + 2 object + 4 bimanual signals, 2 windows x {raw, log1p}.
    assert model.hand_coupling_dim == 12 * 4
    assert model.object_coupling_dim == 2 * 4
    assert model.hand_contact_net.linear1.in_features == reference.hand_contact_input_dim + 48
    assert model.obj_move_net.linear1.in_features == reference.obj_move_input_dim + 8
    # Velocity nets and the pose-prior branch are v4's, parameter for parameter.
    for name in ("hand_vel_net", "obj_vel_net", "boundary_rnn", "contact_endpoint_rnn"):
        ours = sum(p.numel() for p in getattr(model, name).parameters())
        theirs = sum(p.numel() for p in getattr(reference, name).parameters())
        assert ours == theirs


@pytest.mark.parametrize("heading", [False, True])
def test_v5_forward_shapes_loss_and_gradients(heading):
    model = build_velocity_contact_module(_cfg(v5_heading_normalize_imu=heading)).train()
    data, hp_out = _model_inputs()
    out = model(data, hp_out=hp_out)
    assert out["pred_hand_contact_prob"].shape == (2, 12, 3)
    assert torch.isfinite(out["pred_hand_contact_prob"]).all()
    batch = {
        "human_imu": data["human_imu"],
        "obj_vel": torch.randn(2, 12, 3),
        "sensor_vel_glb": torch.randn(2, 12, 6, 3),
        "lhand_contact": torch.rand(2, 12) > 0.5,
        "rhand_contact": torch.rand(2, 12) > 0.5,
        "obj_contact": torch.ones(2, 12, dtype=torch.bool),
        "contact_endpoint_gauss": torch.rand(2, 12, 6),
        "contact_endpoint": (torch.rand(2, 12, 6) > 0.9).float(),
    }
    total, _, _ = VelocityContactLoss(anticipation_horizon=4)(out, batch, torch.device("cpu"))
    total.backward()
    grad = model.hand_contact_net.linear1.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    # The coupling columns are the last ones of the hand-contact input.
    assert grad[:, -48:].abs().sum() > 0


def test_coupling_features_are_invariant_to_a_world_rotation():
    model = build_velocity_contact_module(_cfg()).eval()
    human, obj, palms = _world_inputs()
    object_a, hand_a = model._coupling_features(human, obj, palms)
    rotation = random_rotations(1)[0]
    object_b, hand_b = model._coupling_features(*_rotate_world(human, obj, palms, rotation))
    assert torch.allclose(object_a, object_b, atol=1e-3)
    assert torch.allclose(hand_a, hand_b, atol=1e-3)


def test_heading_normalized_contact_inputs_are_invariant_to_a_world_yaw():
    model = build_velocity_contact_module(_cfg(v5_heading_normalize_imu=True)).eval()
    human, obj, palms = _world_inputs()
    hand_vel, obj_vel = torch.randn(2, 12, 6), torch.randn(2, 12, 3)
    before = model._heading_normalize_contact_inputs(human, obj, hand_vel, obj_vel)
    yaw = _yaw(0.9)
    human_r, obj_r, _ = _rotate_world(human, obj, palms, yaw)
    after = model._heading_normalize_contact_inputs(
        human_r, obj_r, (hand_vel.reshape(2, 12, 2, 3) @ yaw.T).reshape(2, 12, 6), obj_vel @ yaw.T
    )
    for a, b in zip(before, after):
        assert torch.allclose(a, b, atol=1e-4)


def test_rotation_rate_and_centred_average():
    angles = torch.linspace(0.0, 1.1, 12)
    rotation = torch.stack([_yaw(float(a)) for a in angles]).unsqueeze(0)
    rate = rotation_rate(rotation, fps=30.0)
    assert torch.allclose(rate, torch.full_like(rate, 0.1 * 30.0), atol=1e-3)
    value = torch.arange(10.0).reshape(1, 10, 1)
    smoothed = centred_average(value, 2, 2)
    assert torch.allclose(smoothed[0, 2:8, 0], value[0, 2:8, 0])
    assert smoothed[0, 0, 0] == pytest.approx((0 + 0 + 0 + 1 + 2) / 5)


def test_v5_checkpoint_guard():
    model = build_velocity_contact_module(_cfg())
    model.validate_checkpoint_state_dict(model.state_dict())
    with pytest.raises(ValueError):
        model.validate_checkpoint_state_dict(V4VelocityContactModule(_cfg(velocity_contact_arch="v4")).state_dict())
    with pytest.raises(ValueError):
        model.validate_checkpoint_state_dict(
            build_velocity_contact_module(_cfg(v5_heading_normalize_imu=True)).state_dict()
        )


def test_init_perturbation_is_training_only_and_hits_a_window_subset():
    model = build_velocity_contact_module(
        _cfg(v5_init_noise_prob=0.5, v5_obj_vel_init_noise_std=0.3, v5_hand_vel_init_noise_std=0.3,
             v5_contact_init_flip_prob=1.0)
    )
    hand = torch.zeros(64, 6)
    obj = torch.zeros(64, 3)
    contact = torch.zeros(64, 3)
    model.eval()
    h, o, c = model._perturb_inits(hand, obj, contact)
    assert torch.equal(h, hand) and torch.equal(o, obj) and torch.equal(c, contact)
    model.train()
    torch.manual_seed(0)
    h, o, c = model._perturb_inits(hand, obj, contact)
    touched = o.abs().sum(-1) > 0
    assert 0 < int(touched.sum()) < 64
    # the same windows get their hand init and contact init corrupted
    assert torch.equal(touched, h.abs().sum(-1) > 0)
    assert torch.equal(touched, c.sum(-1) == 3)


def _loss_batch(data, obj_moving=None):
    batch_size, seq_len = data["human_imu"].shape[:2]
    torch.manual_seed(1)
    return {
        "human_imu": data["human_imu"],
        "obj_vel": torch.randn(batch_size, seq_len, 3),
        "sensor_vel_glb": torch.randn(batch_size, seq_len, 6, 3),
        "lhand_contact": torch.rand(batch_size, seq_len) > 0.5,
        "rhand_contact": torch.rand(batch_size, seq_len) > 0.5,
        "obj_contact": torch.rand(batch_size, seq_len) > 0.3 if obj_moving is None else obj_moving,
        "contact_endpoint_gauss": torch.rand(batch_size, seq_len, 6),
        "contact_endpoint": (torch.rand(batch_size, seq_len, 6) > 0.9).float(),
    }


def test_no_pose_prior_ablation_reads_no_hp_output():
    full = build_velocity_contact_module(_cfg())
    model = build_velocity_contact_module(_cfg(vc_ablation_no_pose_prior=True)).train()
    assert not model.requires_hp
    for name in ("group_encoders", "boundary_rnn", "contact_endpoint_head", "contact_anticipation_head"):
        assert not hasattr(model, name)
    # IMU evidence only: base 36 + IMU-only coupling (2 hands x 3 + 2 object + 1 bimanual) x 4.
    assert model.hand_contact_net.linear1.in_features == 36 + 9 * 4
    assert model.obj_move_net.linear1.in_features == 27 + 2 * 4
    assert full.hand_contact_net.linear1.in_features == 36 + 8 + 12 + 12 * 4
    data, _ = _model_inputs()
    out = model(data, hp_out=None)
    assert out["pred_hand_contact_prob"].shape == (2, 12, 3)
    assert "pred_contact_endpoint_logits" not in out and "pred_contact_anticipation_logits" not in out
    total, losses, _ = VelocityContactLoss(anticipation_horizon=4)(out, _loss_batch(data), torch.device("cpu"))
    assert losses["hand_contact_cond"] > 0 and losses["contact_endpoint"] == 0
    total.backward()
    assert model.hand_contact_net.linear1.weight.grad is not None


def test_no_pose_prior_coupling_keeps_the_imu_signals():
    full = build_velocity_contact_module(_cfg()).eval()
    ablated = build_velocity_contact_module(_cfg(vc_ablation_no_pose_prior=True)).eval()
    human, obj, palms = _world_inputs()
    object_a, hand_a = full._coupling_features(human, obj, palms)
    object_b, hand_b = ablated._coupling_features(human, obj, None)
    assert torch.equal(object_a, object_b)
    # Per scale block the full model has 12 signals; the palm ones are block offsets 8-10.
    keep = [block * 12 + k for block in range(4) for k in range(12) if k not in (8, 9, 10)]
    assert torch.equal(hand_a[..., keep], hand_b)


def test_direct_contact_ablation_skips_the_factorization():
    model = build_velocity_contact_module(_cfg(vc_ablation_direct_contact=True)).train()
    full = build_velocity_contact_module(_cfg())
    assert {k: v.shape for k, v in model.state_dict().items() if k != "_v5_ablation_code"} == {
        k: v.shape for k, v in full.state_dict().items()
    }
    data, hp_out = _model_inputs()
    out = model(data, hp_out=hp_out)
    assert "pred_hand_contact_prob_cond" not in out and "pred_hand_contact_prob_marginal" not in out
    assert torch.allclose(out["pred_hand_contact_prob"][..., :2], torch.sigmoid(out["pred_hand_contact_logits_direct"]))
    # Supervised on every frame, including those where the object rests.
    batch = _loss_batch(data, obj_moving=torch.zeros(2, 12, dtype=torch.bool))
    total, losses, _ = VelocityContactLoss(anticipation_horizon=4)(out, batch, torch.device("cpu"))
    assert losses["hand_contact_direct"] > 0
    assert losses["hand_contact_cond"] == 0 and losses["hand_contact_marginal"] == 0
    total.backward()
    assert model.hand_contact_net.linear1.weight.grad.abs().sum() > 0
    # The factorized model never produces the direct term.
    _, full_losses, _ = VelocityContactLoss(anticipation_horizon=4)(full(data, hp_out=hp_out), batch, torch.device("cpu"))
    assert full_losses["hand_contact_direct"] == 0


def test_drop_body_groups_ablation_shrinks_the_prior():
    model = build_velocity_contact_module(_cfg(vc_ablation_drop_body_groups=["left_leg", "right_leg"]))
    assert sorted(model.group_encoders) == ["left_arm", "right_arm", "torso"]
    assert model.dropped_body_groups == ("left_leg", "right_leg")
    assert model.boundary_rnn.input_size == 3 * 8
    data, hp_out = _model_inputs()
    assert model(data, hp_out=hp_out)["pred_hand_contact_prob"].shape == (2, 12, 3)
    with pytest.raises(ValueError):
        build_velocity_contact_module(_cfg(vc_ablation_drop_body_groups=["tail"]))
    with pytest.raises(ValueError):
        build_velocity_contact_module(
            _cfg(vc_ablation_no_pose_prior=True, vc_ablation_drop_body_groups=["torso"])
        )


def test_ablation_checkpoint_guard():
    configs = {
        "full": _cfg(),
        "no_prior": _cfg(vc_ablation_no_pose_prior=True),
        "direct": _cfg(vc_ablation_direct_contact=True),
        "no_legs": _cfg(vc_ablation_drop_body_groups=["left_leg", "right_leg"]),
        "no_torso": _cfg(vc_ablation_drop_body_groups=["torso"]),
    }
    models = {name: build_velocity_contact_module(cfg) for name, cfg in configs.items()}
    assert "_v5_ablation_code" not in models["full"].state_dict()
    for name, model in models.items():
        model.validate_checkpoint_state_dict(model.state_dict())
        for other, source in models.items():
            if other != name:
                with pytest.raises(ValueError):
                    model.validate_checkpoint_state_dict(source.state_dict())


def test_no_pose_prior_ablation_keeps_staged_training():
    from train.rnn.train_velocity_contact import _vc_staged_training

    cfg = _cfg(vc_ablation_no_pose_prior=True, vc_staged_training=True)
    assert _vc_staged_training(build_velocity_contact_module(cfg), cfg)


def _bundle_cfg(**overrides):
    return _cfg(vc_ablation_no_pose_prior=True, vc_ablation_no_coupling=True, vc_ablation_causal_obj_move=True,
                **overrides)


def test_hp_prior_bundle_ablation_leaves_imuhoi_evidence_and_the_factorization():
    model = build_velocity_contact_module(_bundle_cfg()).train()
    assert model.ablation_code == 1 | (1 << 7) | (1 << 8)
    assert not model.requires_hp and not model.obj_move_net.rnn.bidirectional
    # IMU-HOI's 36-D contact evidence for the hand head, hand + object IMUs for the motion head.
    assert model.hand_contact_net.linear1.in_features == 36
    assert model.obj_move_net.linear1.in_features == 27
    data, _ = _model_inputs()
    out = model(data, hp_out=None)
    assert torch.allclose(
        out["pred_hand_contact_prob"][..., :2], out["pred_obj_move_prob"] * out["pred_hand_contact_prob_cond"]
    )
    total, losses, _ = VelocityContactLoss(anticipation_horizon=4)(out, _loss_batch(data), torch.device("cpu"))
    assert losses["hand_contact_cond"] > 0 and losses["hand_contact_marginal"] > 0
    total.backward()


def test_causal_object_head_does_not_read_the_future():
    model = build_velocity_contact_module(_bundle_cfg()).eval()
    data, _ = _model_inputs()
    future = {k: v.clone() for k, v in data.items()}
    future["obj_imu"][:, 8:] += 5.0
    with torch.no_grad():
        a, b = model(data)["pred_obj_move_prob"], model(future)["pred_obj_move_prob"]
    assert torch.allclose(a[:, :8], b[:, :8]) and not torch.allclose(a[:, 8:], b[:, 8:])


def test_bundle_checkpoint_guard():
    bundle = build_velocity_contact_module(_bundle_cfg())
    prior_only = build_velocity_contact_module(_cfg(vc_ablation_no_pose_prior=True))
    bundle.validate_checkpoint_state_dict(bundle.state_dict())
    for source, target in ((bundle, prior_only), (prior_only, bundle)):
        with pytest.raises(ValueError):
            target.validate_checkpoint_state_dict(source.state_dict())
