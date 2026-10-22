"""HumanPose output smoothing and the published-trajectory VC velocity initialiser."""
import types

import torch

from model.rnn.human_pose import HumanPoseModule
from model.rnn.online import update_data_inits_from_history


def _stub(half_width, polyorder):
    stub = types.SimpleNamespace(
        smoothing_half_width=half_width,
        smoothing_polyorder=polyorder,
        _savgol_center_kernel=HumanPoseModule._savgol_center_kernel,
    )
    stub._smoothing_matrix = lambda seq_len, device: HumanPoseModule._smoothing_matrix(stub, seq_len, device)
    stub._temporal_smooth = lambda x: HumanPoseModule._temporal_smooth(stub, x)
    return stub


def _smoother(half_width, polyorder=0):
    stub = _stub(half_width, polyorder)
    return lambda x: HumanPoseModule._temporal_smooth(stub, x)


def test_smoothing_disabled_is_identity():
    x = torch.randn(2, 20, 5)
    assert _smoother(0)(x) is x


def test_smoothing_matches_centred_average_with_shrinking_edges():
    torch.manual_seed(0)
    x = torch.randn(2, 30, 3, 4)
    y = _smoother(4)(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    for t in (0, 1, 3, 4, 15, 26, 28, 29):
        h = min(4, t, 29 - t)
        torch.testing.assert_close(y[:, t], x[:, t - h:t + h + 1].mean(dim=1), rtol=1e-5, atol=1e-5)


def test_smoothing_has_no_lag_and_removes_white_noise():
    torch.manual_seed(0)
    t = torch.arange(200, dtype=torch.float32)
    ramp = (0.01 * t).view(1, -1, 1)
    torch.testing.assert_close(_smoother(4)(ramp), ramp, rtol=1e-5, atol=1e-5)  # linear motion is preserved exactly
    noisy = ramp + 0.01 * torch.randn_like(ramp)
    jitter = lambda v: (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]).abs().mean()  # noqa: E731
    assert jitter(_smoother(4)(noisy)) < 0.25 * jitter(noisy)


def test_savgol_kernel_matches_scipy_and_preserves_quadratic_motion():
    from scipy.signal import savgol_coeffs

    kernel = HumanPoseModule._savgol_center_kernel(7, 2)
    torch.testing.assert_close(kernel, torch.tensor(savgol_coeffs(15, 2), dtype=torch.float32), rtol=1e-4, atol=1e-5)
    t = torch.arange(80, dtype=torch.float32)
    quad = (0.002 * t ** 2 - 0.05 * t).view(1, -1, 1)
    torch.testing.assert_close(_smoother(7, polyorder=2)(quad), quad, rtol=1e-4, atol=1e-4)  # accelerating motion kept
    torch.manual_seed(0)
    noisy = quad + 0.01 * torch.randn_like(quad)
    jitter = lambda v: (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]).abs().mean()  # noqa: E731
    assert jitter(_smoother(7, polyorder=2)(noisy)[:, 8:-8]) < 0.3 * jitter(noisy[:, 8:-8])


def test_smoothed_rotations_stay_valid():
    from pytorch3d.transforms import rotation_6d_to_matrix

    torch.manual_seed(0)
    stub = _stub(4, 2)
    rot6d = torch.randn(2, 30, 3, 6)
    out = HumanPoseModule._smooth_rotation_6d(stub, rot6d)
    mats = rotation_6d_to_matrix(out.reshape(-1, 6))
    torch.testing.assert_close(mats @ mats.transpose(-1, -2), torch.eye(3).expand_as(mats), rtol=1e-4, atol=1e-4)
    # already-valid, constant rotations are a fixed point
    const = out[:, :1].expand(-1, 30, -1, -1).contiguous()
    torch.testing.assert_close(HumanPoseModule._smooth_rotation_6d(stub, const), const, rtol=1e-4, atol=1e-4)


def test_smoothing_is_full_precision_under_autocast():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    x = torch.randn(4, 60, 12, device="cuda")
    reference = _smoother(7, polyorder=2)(x)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inside = _smoother(7, polyorder=2)(x)
    assert inside.dtype == x.dtype
    torch.testing.assert_close(inside, reference, rtol=1e-6, atol=1e-6)


def test_published_frame_is_causal_beyond_the_half_width():
    """Frames more than half_width ahead of the published frame cannot change it (lookahead budget)."""
    torch.manual_seed(0)
    x = torch.randn(1, 70, 6)
    y = x.clone()
    y[:, 64:] += 10.0  # published frame is local index 59, half-width 4 -> support [55, 63]
    torch.testing.assert_close(_smoother(4)(x)[:, 59], _smoother(4)(y)[:, 59])


def _history():
    trans = torch.zeros(2, 10, 3)
    trans[:, :, 0] = torch.arange(10, dtype=torch.float32).view(1, -1) * 0.02  # 0.6 m/s along x at 30 fps
    return {"pred_obj_vel": torch.full((2, 10, 3), 7.0), "pred_obj_trans": trans}


def test_obj_vel_init_defaults_to_vc_output():
    out = update_data_inits_from_history({}, _history(), index=8)
    torch.testing.assert_close(out["obj_vel_init"], torch.full((2, 3), 7.0))


def test_obj_vel_init_from_published_trajectory():
    out = update_data_inits_from_history({}, _history(), index=8, obj_vel_diff_frames=3, fps=30.0)
    expected = torch.zeros(2, 3)
    expected[:, 0] = 0.6
    torch.testing.assert_close(out["obj_vel_init"], expected, rtol=1e-5, atol=1e-6)


def test_obj_vel_init_clamps_the_span_and_falls_back():
    history = _history()
    out = update_data_inits_from_history({}, history, index=1, obj_vel_diff_frames=5, fps=30.0)
    assert abs(float(out["obj_vel_init"][0, 0]) - 0.6) < 1e-5  # span clamped to the one available frame
    out = update_data_inits_from_history({}, history, index=0, obj_vel_diff_frames=5, fps=30.0)
    torch.testing.assert_close(out["obj_vel_init"], torch.full((2, 3), 7.0))  # no earlier frame -> VC output
    history.pop("pred_obj_trans")
    out = update_data_inits_from_history({}, history, index=8, obj_vel_diff_frames=3, fps=30.0)
    torch.testing.assert_close(out["obj_vel_init"], torch.full((2, 3), 7.0))
