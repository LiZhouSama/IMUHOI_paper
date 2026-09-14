"""ObjectTrans v3 = v2 fusion + forearm/object geometry + learned soft latch + static gate."""
from types import SimpleNamespace

import pytest
import torch

import eval_batch_online
import eval_IMUHOI
import model as model_entry
import train.rnn.train_object_trans as train_object_trans
import vis_IMUHOI
from model.rnn import V3ObjectTransModule, build_object_trans_module
from model.rnn.object_trans_v2 import V2ObjectTransModule
from train.rnn.loss import ObjectTransLoss
from utils.object_fusion import build_contact_fusion_weights
from utils.rotation_conversions import matrix_to_rotation_6d


def _cfg(**overrides):
    base = dict(
        object_trans_arch="v3",
        imu_dim=9,
        obj_imu_dim=9,
        num_human_imus=6,
        hidden_dim_multiplier=1,
        object_trans_hidden_dim=16,
        object_trans_num_layers=2,
        object_trans_dropout=0.0,
        object_trans_state_feedback=False,
        object_trans_feedback_hidden_dim=12,
        object_trans_v3_gate_floor=0.05,
        object_trans_v3_gate_init_bias=1.0,
        object_trans_v3_contact_on_threshold=0.5,
        object_trans_v3_contact_off_threshold=0.25,
        object_trans_static_gate="none",
        enable_ot_refine=False,
        train={"window": 3},
        test={"window": 3},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _identity_6d():
    return torch.eye(3)[:2].reshape(6)


def _inputs(seq_len=5, forearm_rotations=None):
    """Zero human IMU keeps every sensor at the identity fallback."""
    obj_imu = torch.zeros(1, seq_len, 9)
    obj_imu[..., 3:9] = _identity_6d()
    human_imu = torch.zeros(1, seq_len, 6, 9)
    human_imu[..., 3:9] = _identity_6d()
    if forearm_rotations is not None:
        human_imu[:, :, 4, 3:9] = matrix_to_rotation_6d(forearm_rotations)
        human_imu[:, :, 5, 3:9] = matrix_to_rotation_6d(forearm_rotations)
    return {
        "hand_positions": torch.zeros(1, seq_len, 2, 3),
        # left hand latches at frame 1 and stays in contact afterwards.
        "contact": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.9, 0.0, 0.9],
                    [0.9, 0.0, 0.9],
                    [0.9, 0.0, 0.9],
                    [0.9, 0.0, 0.9],
                ]
            ]
        )[:, :seq_len],
        "obj_trans_init": torch.zeros(1, 3),
        "obj_imu": obj_imu,
        "human_imu": human_imu,
        "obj_vel": torch.zeros(1, seq_len, 3),
    }


def _forward(model, data, **kwargs):
    return model(
        data["hand_positions"],
        data["contact"],
        data["obj_trans_init"],
        obj_imu=data["obj_imu"],
        human_imu=data["human_imu"],
        obj_vel_input=data["obj_vel"],
        enable_refine=False,
        **kwargs,
    )


def _z_rotations(angles):
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    zero = torch.zeros_like(cos)
    one = torch.ones_like(cos)
    return torch.stack(
        (
            torch.stack((cos, -sin, zero), dim=-1),
            torch.stack((sin, cos, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )


def test_v3_head_shape_and_version():
    model = V3ObjectTransModule(_cfg())

    assert model.object_trans_head.linear2.out_features == 2 * V3ObjectTransModule.HAND_OUTPUT_DIM
    assert int(model.object_trans_arch_version.item()) == 3
    # The fusion weights stay v2's fixed contact function; only the bone latch is learned.
    assert model.gate_floor == pytest.approx(0.05)


def test_v3_prediction_input_carries_the_relative_rotation_block():
    model = V3ObjectTransModule(_cfg())
    baseline = V2ObjectTransModule(_cfg())

    assert (
        model.n_prediction_input
        == baseline.n_prediction_input + V3ObjectTransModule.REL_ROT_FEATURE_DIM
    )
    assert model.object_trans_head.linear1.in_features == model.n_prediction_input
    assert model.feedback_embed[-1].out_features == model.n_prediction_input
    assert model.onset_embed[-1].out_features == model.n_prediction_input


def test_v3_forward_keeps_v2_fusion_contract():
    torch.manual_seed(3)
    model = V3ObjectTransModule(_cfg()).eval()
    data = _inputs()

    with torch.no_grad():
        output = _forward(model, data)
        online = model.inference(
            data["hand_positions"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            enable_refine=False,
            inference_mode="online",
            online_window=3,
        )

    assert output["gating_weights"].shape == (1, 5, 4)
    assert torch.allclose(output["gating_weights"].sum(dim=-1), torch.ones(1, 5))
    assert torch.allclose(
        output["gating_weights"], build_contact_fusion_weights(data["contact"])
    )
    assert "gate_logits" not in output
    assert output["gating_is_fixed"] is True
    assert output["object_trans_v3"] is True
    assert online["pred_obj_trans"].shape == (1, 5, 3)


def test_v3_emits_the_direction_and_length_the_object_trans_loss_reads():
    torch.manual_seed(4)
    model = V3ObjectTransModule(_cfg()).eval()
    data = _inputs()

    with torch.no_grad():
        output = _forward(model, data)

    for hand in ("lhand", "rhand"):
        vector = output[f"pred_{hand}_obj_vector"]
        assert vector.shape == (1, 5, 3)
        assert torch.allclose(
            output[f"pred_{hand}_lb"], vector.norm(dim=-1), atol=1e-6
        )
        assert torch.allclose(
            output[f"pred_{hand}_obj_direction"] * output[f"pred_{hand}_lb"].unsqueeze(-1),
            vector,
            atol=1e-6,
        )

    total, losses, _ = ObjectTransLoss()(
        output,
        {
            "human_imu": data["human_imu"],
            "obj_trans": torch.zeros(1, 5, 3),
            "lhand_contact": torch.ones(1, 5),
            "rhand_contact": torch.zeros(1, 5),
            "lhand_obj_direction": torch.zeros(1, 5, 3),
            "rhand_obj_direction": torch.zeros(1, 5, 3),
            "lhand_lb": torch.zeros(1, 5),
            "rhand_lb": torch.zeros(1, 5),
        },
        torch.device("cpu"),
    )
    assert torch.isfinite(total)
    assert torch.equal(losses["gate_weak"], torch.zeros(()))


def test_v3_gate_is_bounded_and_resets_the_bone_at_contact_onset():
    torch.manual_seed(5)
    model = V3ObjectTransModule(_cfg()).eval()
    data = _inputs()

    with torch.no_grad():
        output = _forward(model, data)

    gate = output["pred_lhand_bone_gate"]
    assert gate.shape == (1, 5)
    assert bool((gate >= model.gate_floor - 1e-6).all()) and bool((gate <= 1.0 + 1e-6).all())
    # Frame 1 is the left-hand onset: alpha is forced to 1 and the bone equals the proposal.
    assert gate[0, 1] == pytest.approx(1.0)
    assert torch.allclose(output["pred_lhand_obj_vector"][:, 1], output["pred_lhand_bone_proposal"][:, 1], atol=1e-6)
    # Afterwards the bone is the gated moving average of the proposals.
    bone = output["pred_lhand_obj_vector"]
    proposal = output["pred_lhand_bone_proposal"]
    for t in range(2, 5):
        expected = bone[:, t - 1] + gate[0, t] * (proposal[:, t] - bone[:, t - 1])
        assert torch.allclose(bone[:, t], expected, atol=1e-6)
    # The right hand never touches: no onset, so it just tracks softly from the init bone.
    assert bool((output["pred_rhand_bone_gate"][0] < 1.0).all())


def test_v3_gate_floor_one_is_plain_per_frame_regression():
    torch.manual_seed(6)
    model = V3ObjectTransModule(_cfg(object_trans_v3_gate_floor=1.0)).eval()
    data = _inputs()

    with torch.no_grad():
        output = _forward(model, data)

    assert torch.allclose(output["pred_lhand_obj_vector"], output["pred_lhand_bone_proposal"], atol=1e-6)
    assert torch.allclose(output["pred_rhand_obj_vector"], output["pred_rhand_bone_proposal"], atol=1e-6)


def test_v3_saturated_gate_is_the_pure_segment_latch():
    torch.manual_seed(7)
    model = V3ObjectTransModule(
        _cfg(object_trans_v3_gate_floor=0.0, object_trans_v3_gate_init_bias=-40.0)
    ).eval()
    data = _inputs()

    with torch.no_grad():
        output = _forward(model, data)

    bone = output["pred_lhand_obj_vector"]
    # Latched at the onset proposal and never moved again inside the segment.
    for t in range(2, 5):
        assert torch.allclose(bone[:, t], bone[:, 1], atol=1e-6)
    assert not torch.allclose(bone[:, 0], bone[:, 1], atol=1e-6)


def test_static_gate_pins_the_fusion_to_the_previous_position():
    torch.manual_seed(8)
    data = _inputs()
    # p_move = 0 on frames 0, 3, 4 -> object declared static there.
    data["contact"] = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.0, 0.9], [0.9, 0.0, 0.9], [0.9, 0.0, 0.0], [0.9, 0.0, 0.0]]]
    )
    data["hand_positions"] = torch.randn(1, 5, 2, 3)
    for arch in ("v2", "v3"):
        model = build_object_trans_module(
            _cfg(object_trans_static_gate="pmove", object_trans_static_gate_pmove_threshold=0.5),
            object_trans_arch=arch,
        ).eval()
        with torch.no_grad():
            output = _forward(model, data)
        weights = output["gating_weights"]
        static = torch.tensor([0.0, 0.0, 0.0, 1.0])
        for t in (0, 3, 4):
            assert torch.allclose(weights[0, t], static), (arch, t)
            assert torch.allclose(output["pred_obj_trans"][:, t], output["pred_static_obj_trans"][:, t]), (arch, t)
        # Frames with object motion keep the soft fusion.
        assert torch.allclose(weights[0, 1], build_contact_fusion_weights(data["contact"])[0, 1])
        assert torch.allclose(output["pred_obj_trans"][:, 3], output["pred_obj_trans"][:, 2])


def test_static_gate_imu_rule_detects_a_still_object():
    torch.manual_seed(9)
    data = _inputs()
    data["hand_positions"] = torch.randn(1, 5, 2, 3)
    model = build_object_trans_module(_cfg(object_trans_static_gate="imu"), object_trans_arch="v2").eval()
    with torch.no_grad():
        still = _forward(model, data)
    # Identity rotation and zero acceleration everywhere: every frame is static.
    assert torch.allclose(still["gating_weights"][..., 3], torch.ones(1, 5))
    assert torch.allclose(still["pred_obj_trans"], torch.zeros(1, 5, 3))
    # A rotating object is not static.
    moving = _inputs()
    moving["hand_positions"] = data["hand_positions"]
    moving["obj_imu"][:, :, 3:9] = matrix_to_rotation_6d(_z_rotations(torch.tensor([0.0, 0.3, 0.6, 0.9, 1.2])))
    with torch.no_grad():
        out = _forward(model, moving)
    assert bool((out["gating_weights"][0, 1:, 3] < 1.0).all())
    # The gate is off by default, so v2 checkpoints keep their behaviour.
    assert V2ObjectTransModule(_cfg()).static_gate_mode == "none"


def test_v3_always_scans_step_by_step_even_without_state_feedback(monkeypatch):
    """v2 vectorises the window when feedback is off; v3 cannot, so it must not."""
    model = V3ObjectTransModule(_cfg(object_trans_state_feedback=False)).eval()
    data = _inputs()
    step_calls = 0
    original_step = model.object_trans_head.step

    def track_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(model.object_trans_head, "step", track_step)

    with torch.no_grad():
        output = _forward(model, data)

    assert output["pred_obj_trans"].shape == (1, 5, 3)
    assert step_calls == 5


def test_v3_gradients_reach_direction_length_and_gate():
    torch.manual_seed(10)
    model = V3ObjectTransModule(_cfg()).train()
    angles = torch.tensor([0.0, 0.0, 0.3, 0.6, 0.9])
    data = _inputs(forearm_rotations=_z_rotations(angles))
    data["hand_positions"] = torch.randn(1, 5, 2, 3)

    output = _forward(model, data)
    (output["pred_obj_trans"] - torch.randn(1, 5, 3)).square().mean().backward()

    grad = model.object_trans_head.linear2.weight.grad
    assert grad is not None
    dim = V3ObjectTransModule.HAND_OUTPUT_DIM
    # Left hand (in contact): direction, length and gate channels all learn.
    assert grad[0:3].abs().sum() > 0
    assert grad[3].abs().sum() > 0
    assert grad[4].abs().sum() > 0
    assert grad.shape[0] == 2 * dim


def test_v3_empty_output_matches_the_forward_keys():
    torch.manual_seed(11)
    model = V3ObjectTransModule(_cfg()).eval()
    data = _inputs()

    with torch.no_grad():
        output = _forward(model, data)
    empty = V3ObjectTransModule.empty_output(1, 5, torch.device("cpu"))

    for key, value in empty.items():
        assert key in output, key
        if isinstance(value, torch.Tensor):
            assert output[key].shape == value.shape, key
    for key in V3ObjectTransModule.SEGMENT_STATE_OUTPUT_KEYS:
        assert key in output and key in empty


def test_v3_is_registered_everywhere_the_arch_is_selected():
    module = build_object_trans_module(_cfg(), object_trans_arch="v3")
    assert isinstance(module, V3ObjectTransModule)

    root = model_entry.ObjectTransModule(_cfg(), object_trans_arch="v3")
    assert isinstance(root, V3ObjectTransModule)

    for parser_module in (eval_batch_online, vis_IMUHOI):
        parsed = parser_module._make_parser().parse_args(
            ["--object_trans_arch", "v3"]
        )
        assert parsed.object_trans_arch == "v3"
    assert "v3" in eval_IMUHOI.OBJECT_TRANS_ARCH_CHOICES
    assert "v3" in train_object_trans.OBJECT_TRANS_ARCH_CHOICES


def test_v3_rejects_the_removed_per_frame_decode():
    model = V3ObjectTransModule(_cfg())
    with pytest.raises(NotImplementedError):
        model._decode_prediction_output(torch.zeros(1, 10), torch.zeros(1, 3))


def _window_with_published_prefix(model, data, full, start, end, carry):
    """Mimic one online window: frames [start, end) with frames [start, end-1) known."""
    known_len = end - start - 1
    sliced = {
        "hand_positions": data["hand_positions"][:, start:end],
        "contact": data["contact"][:, start:end],
        "obj_trans_init": full["pred_obj_trans"][:, start],
        "obj_imu": data["obj_imu"][:, start:end],
        "human_imu": data["human_imu"][:, start:end],
        "obj_vel": data["obj_vel"][:, start:end],
    }
    segment_prefix = {
        key: full[key][:, start : start + known_len]
        for key in V3ObjectTransModule.SEGMENT_STATE_OUTPUT_KEYS
    }
    model.online_carry_segment_state = carry
    return _forward(
        model,
        sliced,
        known_obj_trans_prefix=full["pred_obj_trans"][:, start : start + known_len],
        known_gating_weights_prefix=full["gating_weights"][:, start : start + known_len],
        known_segment_state_prefix=segment_prefix,
    )


def test_v3_online_window_carries_the_latch_state():
    """A sliding window that starts inside a segment must continue that segment.

    With the head's proposal held constant (zero output weights, biases only)
    the bone is a deterministic moving average, so a window that receives the
    published latch state reproduces the whole-sequence bone at its target
    frame exactly, while a re-initialised window starts from the geometric
    init and does not.
    """
    torch.manual_seed(12)
    model = V3ObjectTransModule(_cfg()).eval()
    with torch.no_grad():
        model.object_trans_head.linear2.weight.zero_()
        model.object_trans_head.linear2.bias[0:3] = torch.tensor([1.0, 0.5, -0.2])
        model.object_trans_head.linear2.bias[5:8] = torch.tensor([-0.4, 1.0, 0.3])
    angles = torch.tensor([[0.0, 0.0, 0.4, 0.9, 1.3]])
    data = _inputs(forearm_rotations=_z_rotations(angles))
    data["hand_positions"] = torch.randn(1, 5, 2, 3)
    with torch.no_grad():
        full = _forward(model, data)
        with_carry = _window_with_published_prefix(model, data, full, 2, 5, carry=True)
        without_carry = _window_with_published_prefix(model, data, full, 2, 5, carry=False)

    target = full["pred_lhand_obj_vector"][:, 4]
    assert torch.allclose(with_carry["pred_lhand_obj_vector"][:, 2], target, atol=1e-5)
    assert torch.allclose(with_carry["segment_rel_rot6d"][:, 2], full["segment_rel_rot6d"][:, 4], atol=1e-6)
    assert torch.allclose(with_carry["pred_lhand_obj_trans"][:, 2], full["pred_lhand_obj_trans"][:, 4], atol=1e-5)
    assert not torch.allclose(without_carry["pred_lhand_obj_vector"][:, 2], target, atol=1e-4)
    # Training / plain offline calls never pass a prefix and are unaffected.
    with torch.no_grad():
        again = _forward(model, data)
    assert torch.allclose(again["pred_obj_trans"], full["pred_obj_trans"])
