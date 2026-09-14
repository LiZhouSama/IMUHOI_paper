import json

import numpy as np
import pytest
import torch

from scripts.analyze_bimanual_hoi import (
    analyze_record,
    causal_hand_constraint,
    correct_hand_difference,
    hoi_score,
    json_safe,
    summarize,
)


def make_record(hand_errors, contacts):
    hand_errors = torch.as_tensor(hand_errors, dtype=torch.float64)
    frames = len(hand_errors)
    identity_6d = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(frames, 1)
    gt_hands = torch.tensor([[0.2, 0., 0.], [-0.2, 0., 0.]], dtype=torch.float64).repeat(frames, 1, 1)
    return {
        "publish_len": frames,
        "pred_obj_trans_published": torch.zeros(frames, 3, dtype=torch.float64),
        "gt_obj_trans": torch.zeros(frames, 3, dtype=torch.float64),
        "pred_palm_glb_pos": gt_hands + hand_errors,
        "gt_palm_pos_eval": gt_hands,
        "gt_lhand_contact": torch.tensor(contacts)[:, 0],
        "gt_rhand_contact": torch.tensor(contacts)[:, 1],
        "pred_hand_contact_prob": torch.ones(frames, 3),
        "gt_obj_rot": identity_6d,
        "gt_obj_imu": torch.cat((torch.zeros(frames, 3), identity_6d), dim=1),
    }


def test_common_error_is_not_recoverable_by_hand_difference():
    record = make_record([[[0.1, 0., 0.], [0.1, 0., 0.]]], [[1, 1]])
    result = analyze_record(record)["scenarios"]
    assert result["baseline"]["hoi_cm"] == pytest.approx(10.0)
    assert result["oracle_diff_100"]["hoi_cm"] == pytest.approx(10.0)
    assert result["oracle_object_only_both"]["hoi_cm"] == pytest.approx(0.0)


def test_differential_error_can_be_removed_but_single_contact_is_unchanged():
    record = make_record([[[0.1, 0., 0.], [-0.1, 0., 0.]]] * 2, [[1, 1], [1, 0]])
    result = analyze_record(record)
    assert result["bimanual_metric_share"] == pytest.approx(2.0 / 3.0)
    assert result["scenarios"]["baseline"]["hoi_cm"] == pytest.approx(10.0)
    assert result["scenarios"]["oracle_diff_100"]["hoi_cm"] == pytest.approx(10.0 / 3.0)
    assert result["scenarios"]["oracle_diff_50"]["hoi_cm"] == pytest.approx(20.0 / 3.0)


def test_correction_preserves_midpoint_and_unselected_frames():
    generator = np.random.default_rng(7)
    hands = generator.normal(size=(9, 2, 3))
    target = generator.normal(size=(9, 3))
    selected = np.arange(9) % 2 == 0
    corrected = correct_hand_difference(hands, target, selected)
    np.testing.assert_allclose(corrected.mean(axis=1), hands.mean(axis=1))
    np.testing.assert_allclose(corrected[selected, 0] - corrected[selected, 1], target[selected])
    np.testing.assert_array_equal(corrected[~selected], hands[~selected])


def test_causal_constraint_never_uses_future_frames():
    generator = np.random.default_rng(9)
    hands = generator.normal(size=(12, 2, 3))
    rotations = np.broadcast_to(np.eye(3), (12, 3, 3))
    active = np.ones(12, dtype=bool)
    prefix = causal_hand_constraint(hands[:6], rotations[:6], active[:6])
    full = causal_hand_constraint(hands, rotations, active)
    np.testing.assert_allclose(prefix, full[:6])


def test_rigid_co_rotation_needs_no_correction():
    rotations = np.array([np.eye(3), [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    hands = np.einsum("tij,hj->thi", rotations, np.array([[0.2, 0., 0.], [-0.2, 0., 0.]]))
    corrected = causal_hand_constraint(hands, rotations, np.ones(2, dtype=bool))
    np.testing.assert_allclose(corrected, hands, atol=1e-12)


def test_aggregation_is_sequence_equal_weight_and_ignores_undefined_hoi():
    short = analyze_record(make_record([[[0.1, 0., 0.], [0.1, 0., 0.]]], [[1, 1]]))
    long = analyze_record(make_record([[[0.2, 0., 0.], [0.2, 0., 0.]]] * 10, [[1, 1]] * 10))
    empty = analyze_record(make_record([[[0., 0., 0.], [0., 0., 0.]]], [[0, 0]]))
    result = summarize([short, long, empty])
    assert result["scenarios"]["baseline"]["hoi_cm"] == pytest.approx(15.0)
    assert result["valid_hoi_sequences"] == 2


def test_exact_difference_correction_cannot_increase_paired_hoi():
    generator = np.random.default_rng(1)
    hands = generator.normal(size=(100, 2, 3))
    gt_hands = generator.normal(size=(100, 2, 3))
    obj = generator.normal(size=(100, 3))
    gt_obj = generator.normal(size=(100, 3))
    contact = np.ones((100, 2), dtype=bool)
    corrected = correct_hand_difference(hands, gt_hands[:, 0] - gt_hands[:, 1], contact.all(axis=1))
    assert hoi_score(obj, corrected, gt_obj, gt_hands, contact) <= hoi_score(obj, hands, gt_obj, gt_hands, contact)


def test_full_report_is_json_serializable():
    result = analyze_record(make_record([[[0., 0., 0.], [0., 0., 0.]]], [[1, 1]]))
    payload = {"summary": summarize([result]), "sequences": [result]}
    restored = json.loads(json.dumps(json_safe(payload), allow_nan=False))
    assert restored["sequences"][0]["segment_lengths"] == [1]