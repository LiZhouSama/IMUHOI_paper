"""Online l_b statistics: the trailing-window option must only see recent frames."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rnn.object_trans import ObjectTransModule  # noqa: E402


def _stream(T=500, switch=400, d_early=0.2, d_late=0.5):
    obj = torch.zeros(1, T, 3)
    hands = torch.zeros(1, T, 2, 3)
    hands[:, :switch, :, 0] = d_early
    hands[:, switch:, :, 0] = d_late
    prob = torch.ones(1, T, 3)
    return obj, hands, prob


def test_full_history_is_dominated_by_the_early_segment():
    obj, hands, prob = _stream()
    q = ObjectTransModule.compute_online_lb_quantiles(obj, hands, prob, contact_threshold=0.25, min_contact_frames=10)
    assert torch.allclose(q[0, :, 1], torch.full((2,), 0.2), atol=1e-6)


def test_tail_window_sees_only_recent_frames():
    obj, hands, prob = _stream()
    q = ObjectTransModule.compute_online_lb_quantiles(
        obj, hands, prob, contact_threshold=0.25, min_contact_frames=10, tail_frames=100
    )
    assert torch.allclose(q[0, :, 1], torch.full((2,), 0.5), atol=1e-6)


def test_tail_window_longer_than_history_is_a_no_op():
    obj, hands, prob = _stream()
    full = ObjectTransModule.compute_online_lb_quantiles(obj, hands, prob, contact_threshold=0.25, min_contact_frames=10)
    tail = ObjectTransModule.compute_online_lb_quantiles(
        obj, hands, prob, contact_threshold=0.25, min_contact_frames=10, tail_frames=10_000
    )
    assert torch.equal(full, tail)


def test_tail_window_without_enough_contact_has_no_statistics():
    obj, hands, prob = _stream()
    prob[:, 450:, :2] = 0.0  # last 50 frames are not in contact
    q = ObjectTransModule.compute_online_lb_quantiles(
        obj, hands, prob, contact_threshold=0.25, min_contact_frames=10, tail_frames=50
    )
    # No usable frames in the window: the estimator reports "no statistics"
    # (None, or NaN per hand), which ObjectTrans maps to the init-geometry fallback.
    assert q is None or torch.isnan(q).all()
