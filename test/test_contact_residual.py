"""CPU-only contracts for label-isolated, causal published-object residuals."""
from copy import deepcopy

import pytest
import torch

from model.rnn.contact_residual import (
    CONTACT_RESIDUAL_FEATURE_DIM,
    CONTACT_RESIDUAL_FEATURE_SLICES,
    CONTACT_RESIDUAL_TEMPORAL_SLICE,
    CausalContactResidual,
    PointwiseContactResidual,
    apply_contact_residual,
    build_contact_residual_features,
)
from utils.object_fusion import build_contact_fusion_weights
from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix


_MODELS = (CausalContactResidual, PointwiseContactResidual)


def _z_rotation(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), dim=-1).reshape(*angle.shape, 3, 3)


def _x_rotation(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((o, z, z, z, c, -s, z, s, c), dim=-1).reshape(*angle.shape, 3, 3)


def _record(seq_len=13, *, extras=True, published=True):
    generator = torch.Generator().manual_seed(617)

    def noise(*shape):
        return torch.randn(*shape, generator=generator)

    t = torch.arange(seq_len, dtype=torch.float32)
    root = (noise(seq_len, 3) * 0.005).cumsum(0)
    palms = root[:, None] + torch.tensor([[-0.3, 0.6, 0.2], [0.3, 0.6, 0.2]])
    palms = palms + noise(seq_len, 2, 3) * 0.01
    raw_obj = root + torch.tensor([0.1, 0.8, 0.6]) + noise(seq_len, 3) * 0.015
    obj_rotation = _z_rotation(-0.35 + 0.011 * t) @ _x_rotation(-0.1 + 0.005 * t)
    root_rotation = _z_rotation(0.4 + 0.007 * t) @ _x_rotation(0.2 + 0.006 * t)
    forearm_root = torch.stack(
        (_z_rotation(0.3 - 0.012 * t), _x_rotation(-0.25 + 0.01 * t)), dim=1
    )
    human_imu = torch.zeros(seq_len, 6, 9)
    human_imu[..., :3] = noise(seq_len, 6, 3) * 1.5
    human_imu[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    human_imu[:, 0, 3:9] = matrix_to_rotation_6d(root_rotation)
    human_imu[:, (4, 5), 3:9] = matrix_to_rotation_6d(forearm_root)
    record = {
        "seq_name": "synthetic_observed_online_dump",
        "publish_len": seq_len,
        "gt_human_imu": human_imu,
        "gt_obj_imu": torch.cat((noise(seq_len, 3), matrix_to_rotation_6d(obj_rotation)), dim=-1),
        "pred_obj_trans": raw_obj,
        "pred_palm_glb_pos": palms,
        "stage1_pred_palm_glb_pos": palms + torch.tensor([0.03, -0.02, 0.01]),
        "pred_hand_glb_pos": palms + 0.02,
        "root_trans_pred": root,
        "stage1_root_trans_pred": root - 0.02,
        "pred_hand_contact_prob": torch.sigmoid(noise(seq_len, 3)),
        "gt_obj_trans": noise(seq_len, 3),
        "gt_obj_rot": matrix_to_rotation_6d(torch.eye(3).expand(seq_len, 3, 3)),
        "gt_trans": noise(seq_len, 3),
        "gt_palm_pos": noise(seq_len, 2, 3),
        "gt_contact": noise(seq_len, 3) > 0.0,
        "gt_lhand_contact": noise(seq_len) > 0.0,
        "gt_rhand_contact": noise(seq_len) > 0.0,
        "gt_obj_contact": noise(seq_len) > 0.0,
        "gt_position_global": noise(seq_len, 24, 3),
        "gt_obj_vel": noise(seq_len, 3),
        # These are dump fields, but outside the permitted feature whitelist.
        "ot_hand_positions": palms + 0.07,
        "segment_contact_active": noise(seq_len, 2) > 0.0,
        "segment_rel_rot6d": noise(seq_len, 12),
    }
    if published:
        record["pred_obj_trans_published"] = raw_obj + torch.tensor([0.04, -0.01, 0.02])
    if extras:
        record["gating_weights"] = torch.softmax(noise(seq_len, 4), dim=-1)
        record["pred_imu_obj_trans"] = raw_obj + 0.02
        record["pred_static_obj_trans"] = raw_obj - 0.02
        for index, hand in enumerate(("lhand", "rhand")):
            candidate = raw_obj + noise(seq_len, 3) * 0.02
            bone = torch.einsum("tji,tj->ti", obj_rotation, candidate - palms[:, index])
            record[f"pred_{hand}_obj_trans"] = candidate
            record[f"pred_{hand}_obj_vector"] = bone
            record[f"pred_{hand}_obj_direction"] = torch.nn.functional.normalize(bone, dim=-1)
            record[f"pred_{hand}_lb"] = bone.norm(dim=-1)
            record[f"pred_{hand}_bone_proposal"] = bone + noise(seq_len, 3) * 0.01
            record[f"pred_{hand}_bone_gate"] = torch.sigmoid(noise(seq_len))
    return record


def _block(result, name):
    return result["features"][:, CONTACT_RESIDUAL_FEATURE_SLICES[name]]


def _slice_record(record, start, end):
    seq_len = record["pred_obj_trans"].shape[0]
    return {
        key: value[start:end] if isinstance(value, torch.Tensor) and value.shape[0] == seq_len else value
        for key, value in record.items()
    }


def _nonzero_head(model):
    generator = torch.Generator().manual_seed(981)
    with torch.no_grad():
        model.head.weight.copy_(torch.randn(model.head.weight.shape, generator=generator) * 0.15)
        model.head.bias.copy_(torch.tensor([0.02, -0.04, 0.01]))
    return model


@pytest.mark.parametrize("extras", [False, True])
@pytest.mark.parametrize("seq_len", [0, 1, 13])
def test_feature_schema_dtype_and_finite_values(extras, seq_len):
    record = _record(seq_len, extras=extras)
    for key, value in record.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            record[key] = value.double().requires_grad_()
    result = build_contact_residual_features(record)
    shapes = {
        "features": (seq_len, 99), "obj_rotation": (seq_len, 3, 3),
        "contact_prob": (seq_len, 3), "base_obj": (seq_len, 3), "pred_palm": (seq_len, 2, 3),
    }
    assert CONTACT_RESIDUAL_FEATURE_DIM == 99
    assert CONTACT_RESIDUAL_TEMPORAL_SLICE == slice(72, 96)
    assert list(CONTACT_RESIDUAL_FEATURE_SLICES.values())[-1].stop == 99
    for key, shape in shapes.items():
        assert result[key].shape == shape
        assert result[key].dtype == torch.float32
        assert result[key].device.type == "cpu"
        assert not result[key].requires_grad
        assert torch.isfinite(result[key]).all()
    assert (result["features"].abs() <= 5.0).all()


@pytest.mark.parametrize("model_class", _MODELS)
def test_zero_init_is_exact_published_identity_without_mutation(model_class):
    record = _record()
    before = deepcopy(record)
    result = build_contact_residual_features(record)
    model = model_class(CONTACT_RESIDUAL_FEATURE_DIM)
    delta, state = model(result["features"][None])
    corrected = apply_contact_residual(result["base_obj"], result["obj_rotation"], delta)
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(corrected[0], record["pred_obj_trans_published"])
    assert not torch.equal(corrected[0], record["pred_obj_trans"])
    assert torch.equal(result["pred_palm"], record["pred_palm_glb_pos"])
    assert not torch.equal(result["pred_palm"], record["stage1_pred_palm_glb_pos"])
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    if model_class is CausalContactResidual:
        assert state.shape == (1, 1, 64)
    else:
        assert state is None
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(record[key], value)


def test_published_priority_raw_difference_and_fallback():
    record = _record()
    result = build_contact_residual_features(record)
    expected = torch.einsum(
        "tji,tj->ti", result["obj_rotation"], record["pred_obj_trans"] - record["pred_obj_trans_published"]
    )
    torch.testing.assert_close(_block(result, "raw_published_offset"), expected)
    assert torch.equal(result["base_obj"], record["pred_obj_trans_published"])
    raw_only = deepcopy(record)
    del raw_only["pred_obj_trans_published"]
    fallback = build_contact_residual_features(raw_only)
    assert torch.equal(fallback["base_obj"], record["pred_obj_trans"])
    assert torch.count_nonzero(_block(fallback, "raw_published_offset")) == 0
    del record["pred_obj_trans"]
    published_only = build_contact_residual_features(record)
    assert torch.count_nonzero(_block(published_only, "raw_published_offset")) == 0


@pytest.mark.parametrize("model_class", _MODELS)
def test_future_observations_and_predictions_cannot_change_prefix(model_class):
    record = _record(83)
    changed = deepcopy(record)
    prefix = 41
    generator = torch.Generator().manual_seed(99)
    for value in changed.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            value[prefix:] = torch.randn(value[prefix:].shape, generator=generator)
    original_features = build_contact_residual_features(record)["features"]
    changed_features = build_contact_residual_features(changed)["features"]
    assert torch.equal(original_features[:prefix], changed_features[:prefix])
    assert not torch.equal(original_features[prefix:], changed_features[prefix:])
    torch.manual_seed(34)
    model = _nonzero_head(model_class(99)).eval()
    with torch.no_grad():
        original_output, _ = model(original_features[None])
        changed_output, _ = model(changed_features[None])
    assert torch.count_nonzero(original_output) > 0
    torch.testing.assert_close(original_output[:, :prefix], changed_output[:, :prefix], rtol=0, atol=0)


@pytest.mark.parametrize("num_layers,dropout", [(1, 0.0), (2, 0.0), (2, 0.3)])
def test_gru_chunk_and_whole_outputs_and_states_match(num_layers, dropout):
    torch.manual_seed(4)
    model = _nonzero_head(CausalContactResidual(99, hidden_dim=16, num_layers=num_layers, dropout=dropout))
    if dropout:
        model.eval()
    features = build_contact_residual_features(_record(137))["features"]
    features = torch.stack((features, features + 0.01))
    with torch.no_grad():
        whole, final_state = model(features)
        state, outputs, reset_outputs = None, [], []
        for start, end in ((0, 60), (60, 61), (61, 120), (120, 137)):
            delta, state = model(features[:, start:end], state)
            outputs.append(delta)
            reset_outputs.append(model(features[:, start:end])[0])
    torch.testing.assert_close(torch.cat(outputs, dim=1), whole, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(state, final_state, rtol=1e-5, atol=1e-6)
    # Nonzero heads ensure this test would catch an accidental 60-frame reset.
    assert not torch.allclose(torch.cat(reset_outputs, dim=1), whole, rtol=1e-5, atol=1e-6)


def test_streaming_feature_chunks_need_only_one_previous_frame():
    record = _record(131)
    whole = build_contact_residual_features(record)["features"]
    chunks = []
    for start, end in ((0, 60), (60, 61), (61, 120), (120, 131)):
        context = int(start > 0)
        part = build_contact_residual_features(_slice_record(record, start - context, end))["features"]
        chunks.append(part[context:])
    torch.testing.assert_close(torch.cat(chunks), whole, rtol=0, atol=0)
    assert torch.count_nonzero(whole[0, CONTACT_RESIDUAL_TEMPORAL_SLICE]) == 0


@pytest.mark.parametrize("model_class", _MODELS)
@pytest.mark.parametrize("max_correction", [0.0, 0.075, 0.20])
def test_l2_bound_not_just_per_axis(model_class, max_correction):
    model = model_class(99, max_correction=max_correction)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([1000.0, -1000.0, 1000.0]))
    features = torch.randn(2, 9, 99) * 100
    delta, _ = model(features)
    lengths = torch.linalg.vector_norm(delta, dim=-1)
    assert torch.isfinite(delta).all()
    assert (lengths <= max_correction + 2e-7).all()
    torch.testing.assert_close(lengths, torch.full_like(lengths, max_correction))
    # The full budget remains available along one axis, unlike tanh / sqrt(3).
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([1000.0, 0.0, 0.0]))
    delta, _ = model(features)
    torch.testing.assert_close(delta[..., 0], torch.full_like(lengths, max_correction))


@pytest.mark.parametrize("extras", [False, True])
def test_common_world_translation_leaves_features_unchanged(extras):
    record = _record(extras=extras)
    translated = deepcopy(record)
    offset = torch.tensor([8.0, -3.0, 5.0])
    world_positions = (
        "pred_obj_trans", "pred_obj_trans_published", "pred_palm_glb_pos",
        "stage1_pred_palm_glb_pos", "pred_hand_glb_pos", "root_trans_pred",
        "stage1_root_trans_pred", "pred_lhand_obj_trans", "pred_rhand_obj_trans",
        "pred_imu_obj_trans", "pred_static_obj_trans", "ot_hand_positions",
        "gt_obj_trans", "gt_palm_pos", "gt_trans", "gt_position_global",
    )
    for key in world_positions:
        if key in translated:
            translated[key] = translated[key] + offset
    original = build_contact_residual_features(record)
    moved = build_contact_residual_features(translated)
    torch.testing.assert_close(moved["features"], original["features"], rtol=1e-5, atol=5e-5)
    torch.testing.assert_close(moved["base_obj"], original["base_obj"] + offset)
    torch.testing.assert_close(moved["pred_palm"], original["pred_palm"] + offset)


@pytest.mark.parametrize("extras", [False, True])
def test_mutating_all_gt_labels_and_unapproved_fields_changes_nothing(extras):
    record = _record(extras=extras)
    original = build_contact_residual_features(record)
    changed = deepcopy(record)
    for key in list(changed):
        if key.startswith("gt_") and key not in ("gt_obj_imu", "gt_human_imu"):
            changed[key] = torch.tensor([float("nan")])  # Also poison the label shapes.
    for key in ("ot_hand_positions", "segment_contact_active", "segment_rel_rot6d", "publish_len", "seq_name"):
        changed[key] = object()
    new = build_contact_residual_features(changed)
    for key in original:
        assert torch.equal(original[key], new[key])

    class ObservationWhitelist(dict):
        def get(self, key, default=None):
            assert key in ("gt_obj_imu", "gt_human_imu", "root_trans_pred", "gating_weights") or key.startswith(("pred_", "stage1_"))
            return super().get(key, default)

        def __getitem__(self, key):
            self.get(key)  # Fail even on attempted reads of unused GT labels.
            return super().__getitem__(key)

    safe = build_contact_residual_features(ObservationWhitelist(changed))
    assert torch.equal(safe["features"], original["features"])


def test_noisy_object_imu_not_gt_object_rotation_is_the_source():
    record = _record()
    original = build_contact_residual_features(record)
    original_rotation = rotation_6d_to_matrix(record["gt_obj_imu"][:, 3:9])
    noisy_rotation = _z_rotation(torch.full((13,), 0.43)) @ original_rotation
    record["gt_obj_imu"][:, 3:9] = matrix_to_rotation_6d(noisy_rotation)
    record["gt_obj_rot"][:] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    noisy = build_contact_residual_features(record)
    torch.testing.assert_close(noisy["obj_rotation"], noisy_rotation, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(noisy["obj_rotation"], rotation_6d_to_matrix(record["gt_obj_rot"]))
    assert not torch.allclose(noisy["features"], original["features"])
    assert torch.equal(noisy["base_obj"], original["base_obj"])


def test_forearm_orientations_and_accelerations_are_restored_from_root_relative():
    record = _record()
    result = build_contact_residual_features(record)
    human_imu = record["gt_human_imu"]
    root_rotation = rotation_6d_to_matrix(human_imu[:, 0, 3:9])
    forearm_root = rotation_6d_to_matrix(human_imu[:, (4, 5), 3:9])
    obj_inverse = result["obj_rotation"].transpose(-1, -2)
    expected = matrix_to_rotation_6d(obj_inverse[:, None] @ root_rotation[:, None] @ forearm_root).flatten(1)
    torch.testing.assert_close(_block(result, "forearm_relative_rotation"), expected, rtol=1e-5, atol=1e-6)
    incorrect = matrix_to_rotation_6d(obj_inverse[:, None] @ forearm_root).flatten(1)
    assert not torch.allclose(expected, incorrect)
    forearm_acc_world = torch.einsum(
        "tij,thj->thi", root_rotation, human_imu[:, (4, 5), :3] + human_imu[:, 0, None, :3]
    )
    expected_acc = torch.einsum(
        "tij,thj->thi", obj_inverse, forearm_acc_world - record["gt_obj_imu"][:, None, :3]
    ).flatten(1) / 10.0
    torch.testing.assert_close(_block(result, "forearm_relative_acc"), expected_acc)


def test_missing_optional_fields_have_explicit_fixed_width_defaults():
    record = _record(extras=False)
    result = build_contact_residual_features(record)
    for name in ("candidate_offsets", "candidate_disagreement", "proposal_offsets"):
        assert torch.count_nonzero(_block(result, name)) == 0
    assert torch.equal(_block(result, "bone_alpha"), torch.ones(13, 2))
    torch.testing.assert_close(_block(result, "bones"), _block(result, "object_from_palms"))
    torch.testing.assert_close(_block(result, "gating_weights"), build_contact_fusion_weights(result["contact_prob"]))
    record["gating_weights"] = torch.zeros(13, 4)
    zero_weights = build_contact_residual_features(record)
    assert torch.equal(zero_weights["features"], result["features"])
    record["pred_lhand_obj_direction"] = torch.tensor([1.0, 0.0, 0.0]).expand(13, 3)
    record["pred_lhand_lb"] = torch.full((13, 1), 0.3)
    record["pred_lhand_bone_gate"] = torch.full((13, 1), 0.4)
    legacy = build_contact_residual_features(record)
    torch.testing.assert_close(_block(legacy, "bones")[:, :3], torch.tensor([0.3, 0.0, 0.0]).expand(13, 3))
    torch.testing.assert_close(_block(legacy, "bone_alpha")[:, 0], torch.full((13,), 0.4))
    record["pred_lhand_obj_vector"] = torch.tensor([0.0, 0.2, 0.0]).expand(13, 3)
    vector_priority = build_contact_residual_features(record)
    torch.testing.assert_close(_block(vector_priority, "bones")[:, :3], record["pred_lhand_obj_vector"])


@pytest.mark.parametrize("invalid", [
    [0.0] * 6, [1.0, 0.0, 0.0, 2.0, 0.0, 0.0], [float("nan")] * 6,
])
def test_invalid_6d_uses_identity_never_gt(invalid):
    record = _record(2)
    record["gt_obj_imu"][:, 3:9] = torch.tensor(invalid)
    record["gt_human_imu"][:, :, 3:9] = torch.tensor(invalid)
    record["gt_obj_rot"] = matrix_to_rotation_6d(_z_rotation(torch.ones(2)))
    result = build_contact_residual_features(record)
    assert torch.equal(result["obj_rotation"], torch.eye(3).expand(2, 3, 3))
    assert torch.isfinite(result["features"]).all()


def test_requires_final_palms_observations_and_temporal_alignment():
    record = _record()
    missing_palm = deepcopy(record)
    del missing_palm["pred_palm_glb_pos"]
    with pytest.raises(KeyError, match="pred_palm_glb_pos"):
        build_contact_residual_features(missing_palm)  # Stage1 and GT palms must not hide this error.
    for key in ("gt_obj_imu", "gt_human_imu", "pred_hand_contact_prob"):
        missing = deepcopy(record)
        del missing[key]
        with pytest.raises(KeyError, match=key):
            build_contact_residual_features(missing)
    short = deepcopy(record)
    short["gt_obj_imu"] = short["gt_obj_imu"][:-1]
    with pytest.raises(ValueError, match="gt_obj_imu.*shape"):
        build_contact_residual_features(short)
    invalid_acc = deepcopy(record)
    invalid_acc["gt_obj_imu"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="accelerations"):
        build_contact_residual_features(invalid_acc)
    del record["root_trans_pred"]
    stage1_root = build_contact_residual_features(record)
    expected = torch.einsum("tji,tj->ti", stage1_root["obj_rotation"], stage1_root["base_obj"] - record["stage1_root_trans_pred"])
    torch.testing.assert_close(_block(stage1_root, "object_from_root"), expected)


def test_apply_is_only_rotation_and_gain_and_is_differentiable():
    rotation = _z_rotation(torch.tensor([torch.pi / 2]))
    base = torch.tensor([[2.0, 3.0, -1.0]])
    delta = torch.tensor([[0.1, 0.0, 0.0]], requires_grad=True)
    corrected = apply_contact_residual(base, rotation, delta, gain=0.5)
    torch.testing.assert_close(corrected, torch.tensor([[2.0, 3.05, -1.0]]))
    assert torch.equal(base, torch.tensor([[2.0, 3.0, -1.0]]))
    assert torch.equal(apply_contact_residual(base, rotation, delta, gain=0.0), base)
    corrected.sum().backward()
    torch.testing.assert_close(delta.grad, torch.tensor([[0.5, -0.5, 0.5]]))


@pytest.mark.parametrize("model_class", _MODELS)
def test_zero_init_head_can_learn_without_backpropagating_into_frozen_inputs(model_class):
    torch.manual_seed(23)
    record = _record()
    record["pred_obj_trans_published"].requires_grad_()
    record["pred_palm_glb_pos"].requires_grad_()
    features = build_contact_residual_features(record)
    model = model_class(99)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    target = features["base_obj"] + torch.tensor([0.03, 0.01, -0.02])
    for step in range(2):
        optimizer.zero_grad()
        delta, _ = model(features["features"][None])
        world = apply_contact_residual(features["base_obj"], features["obj_rotation"], delta)
        (world - target[None]).square().mean().backward()
        assert torch.isfinite(model.head.weight.grad).all()
        assert model.head.weight.grad.norm() > 0
        assert record["pred_obj_trans_published"].grad is None
        assert record["pred_palm_glb_pos"].grad is None
        if step and model_class is CausalContactResidual:
            assert model.gru.weight_ih_l0.grad.norm() > 0
        optimizer.step()
    assert torch.count_nonzero(model(features["features"][None])[0]) > 0


def test_pointwise_has_no_memory_and_empty_gru_chunks_preserve_state():
    features = build_contact_residual_features(_record())["features"][None]
    pointwise = _nonzero_head(PointwiseContactResidual(99))
    whole, state = pointwise(features, torch.ones(1, 1, 5))
    assert state is None
    permutation = torch.tensor([5, 2, 10, 1, 6, 11, 0, 4, 9, 7, 3, 12, 8])
    permuted, _ = pointwise(features[:, permutation])
    torch.testing.assert_close(permuted, whole[:, permutation])
    assert sum(parameter.numel() for parameter in pointwise.parameters()) == 300
    gru = CausalContactResidual(99)
    assert sum(parameter.numel() for parameter in gru.parameters()) == 31875
    empty, state = gru(torch.empty(2, 0, 99))
    assert empty.shape == (2, 0, 3)
    assert state.shape == (1, 2, 64)
    previous = torch.ones_like(state)
    empty, state = gru(torch.empty(2, 0, 99), previous)
    assert state is previous
    assert pointwise(torch.empty(2, 0, 99))[0].shape == (2, 0, 3)


@pytest.mark.parametrize("model_class", _MODELS)
@pytest.mark.parametrize("kwargs", [
    {"feature_dim": 0}, {"feature_dim": 1.5}, {"max_correction": -0.1}, {"max_correction": float("nan")},
])
def test_invalid_capacity_is_rejected(model_class, kwargs):
    with pytest.raises(ValueError):
        model_class(**{"feature_dim": 99, **kwargs})


def test_invalid_gru_state_or_features_are_rejected():
    with pytest.raises(ValueError, match="dropout"):
        CausalContactResidual(99, dropout=1.0)
    with pytest.raises(ValueError, match="hidden_dim"):
        CausalContactResidual(99, hidden_dim=0)
    with pytest.raises(ValueError, match="num_layers"):
        CausalContactResidual(99, num_layers=0)
    gru = CausalContactResidual(99)
    with pytest.raises(ValueError, match="features"):
        gru(torch.zeros(2, 99))
    with pytest.raises(ValueError, match="state"):
        gru(torch.zeros(1, 2, 99), torch.zeros(1, 2, 64))
