"""Causality, reset and FK-mass invariants for diagnostic candidate filters."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('v2_replay', Path(__file__).resolve().parents[1] / 'scripts/diag/v2_smoothing_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_noop_and_fk_mass():
    rng = np.random.RandomState(42)
    b = rng.randn(30, 2, 3)
    p = rng.rand(30, 3)
    w = rng.rand(30, 4)
    w /= w.sum(-1, keepdims=True)
    out, weights = replay.filter_candidates(b, p, w)
    np.testing.assert_allclose(out, b)
    np.testing.assert_allclose(weights, w)
    _, weights = replay.filter_candidates(b, p, w, alpha=.25, split_alpha=.25)
    np.testing.assert_allclose(weights[:, :2].sum(-1), w[:, :2].sum(-1))
    np.testing.assert_allclose(weights[:, 2:], w[:, 2:])


def test_world_filter_causal_and_reset():
    rng = np.random.RandomState(4)
    x = rng.randn(30, 2, 3)
    p = np.ones((30, 3))
    y = replay.filter_world_candidates(x, p, .35, .05)
    changed = x.copy()
    changed[20:] += 10
    np.testing.assert_allclose(replay.filter_world_candidates(changed, p, .35, .05)[:20], y[:20])
    p[12:17, :2] = 0
    y = replay.filter_world_candidates(x, p, .35, .05)
    np.testing.assert_allclose(y[12:18], x[12:18])


def test_alpha_beta_tracks_constant_velocity_without_steady_lag():
    x = np.zeros((300, 2, 3))
    x[:, :, 0] = np.arange(300)[:, None] / 30
    p = np.ones((300, 3))
    y = replay.filter_world_candidates(x, p, .35, .05)
    np.testing.assert_allclose(y[-20:], x[-20:], atol=1e-6)
    ema = replay.filter_world_candidates(x, p, .25)
    assert np.mean(np.abs(ema[-20:, :, 0]-x[-20:, :, 0])) > .09
