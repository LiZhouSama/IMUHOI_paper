"""Tests for the online contact-segment l_b statistics prior (OT arch v6)."""
from types import SimpleNamespace
import os
import sys
import types

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pytorch3d.transforms  # noqa: F401
except Exception:
    pytorch3d_mod = types.ModuleType("pytorch3d")
    transforms_mod = types.ModuleType("pytorch3d.transforms")

    def _rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(3, device=x.device, dtype=x.dtype)
        shape = (*x.shape[:-1], 3, 3)
        return eye.view(*([1] * (len(shape) - 2)), 3, 3).expand(shape).clone()

    def _matrix_to_rotation_6d(m: torch.Tensor) -> torch.Tensor:
        return m[..., :2].reshape(*m.shape[:-2], 6)

    transforms_mod.rotation_6d_to_matrix = _rotation_6d_to_matrix
    transforms_mod.matrix_to_rotation_6d = _matrix_to_rotation_6d
    transforms_mod.matrix_to_axis_angle = lambda m: torch.zeros(*m.shape[:-2], 3, device=m.device, dtype=m.dtype)
    pytorch3d_mod.transforms = transforms_mod
    sys.modules["pytorch3d"] = pytorch3d_mod
    sys.modules["pytorch3d.transforms"] = transforms_mod

from configs import _REDUCED_POSE_NAMES
from model.rnn.object_trans import ObjectTransModule
from train.rnn.loss.object_trans_loss import ObjectTransLoss
from train.rnn.scheduled_inputs import sample_gt_lb_prior_stats


def _small_cfg():
    return SimpleNamespace(
        model_arch="rnn",
        test={"window": 3},
        train={"window": 3},
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        hidden_dim_multiplier=1,
    )


def test_compute_online_lb_quantiles_measures_contact_frames_per_hand():
    seq_len = 12
    pred_obj_trans = torch.zeros(1, seq_len, 3)
    pred_obj_trans[..., 0] = 0.5  # object 0.5m from both hands along x
    hand_positions = torch.zeros(1, seq_len, 2, 3)
    contact_prob = torch.zeros(1, seq_len, 3)
    contact_prob[:, :, 0] = 0.9  # left hand always in contact
    contact_prob[:, :, 1] = 0.1  # right hand below threshold

    out = ObjectTransModule.compute_online_lb_quantiles(
        pred_obj_trans,
        hand_positions,
        contact_prob,
        contact_threshold=0.5,
        min_contact_frames=10,
    )

    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 3)
    assert torch.allclose(out[0, 0], torch.full((3,), 0.5))
    assert torch.isnan(out[0, 1]).all()


def test_compute_online_lb_quantiles_returns_none_without_enough_frames():
    out = ObjectTransModule.compute_online_lb_quantiles(
        torch.zeros(1, 4, 3),
        torch.zeros(1, 4, 2, 3),
        torch.ones(1, 4, 3),
        min_contact_frames=10,
    )
    assert out is None


def test_nan_prior_hands_fall_back_to_init_geometry():
    torch.manual_seed(5)
    model = ObjectTransModule(_small_cfg()).eval()
    batch_size, seq_len = 1, 3
    data = {
        "hand_pos": torch.zeros(batch_size, seq_len, 2, 3),
        "contact": torch.ones(batch_size, seq_len, 3),
        "obj_trans_init": torch.tensor([[1.0, 0.0, 0.0]]),
        "obj_imu": torch.zeros(batch_size, seq_len, 9),
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "obj_vel": torch.zeros(batch_size, seq_len, 3),
    }
    prior = torch.tensor([[[0.2, 0.3, 0.4], [float("nan")] * 3]])

    with torch.no_grad():
        out = model(
            data["hand_pos"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            lb_prior_quantiles=prior,
            enable_refine=False,
        )

    quantiles = out["lb_prior_quantiles"]
    assert torch.isfinite(quantiles).all()
    assert torch.allclose(quantiles[0, 0], torch.tensor([0.2, 0.3, 0.4]))
    # NaN right hand falls back to ||obj_trans_init - rhand_pos[0]|| = 1.0
    assert torch.allclose(quantiles[0, 1], torch.ones(3))


def test_stateful_continuation_reuses_stored_lb_quantiles():
    """Continuation calls without a fresh prior must reuse the quantiles from
    the state, not re-derive a drifting neutral from the new chunk's init."""
    torch.manual_seed(7)
    model = ObjectTransModule(_small_cfg()).eval()
    batch_size, seq_len = 1, 3

    def _chunk(obj_trans_init):
        return dict(
            hand_positions=torch.zeros(batch_size, seq_len, 2, 3),
            pred_hand_contact_prob=torch.ones(batch_size, seq_len, 3),
            obj_trans_init=obj_trans_init,
            obj_imu=torch.zeros(batch_size, seq_len, 9),
            human_imu=torch.zeros(batch_size, seq_len, 6, 9),
            obj_vel_input=torch.zeros(batch_size, seq_len, 3),
        )

    with torch.no_grad():
        first_kwargs = _chunk(torch.tensor([[1.0, 0.0, 0.0]]))
        first, state = model(
            first_kwargs.pop("hand_positions"),
            first_kwargs.pop("pred_hand_contact_prob"),
            first_kwargs.pop("obj_trans_init"),
            enable_refine=False,
            return_prediction_state=True,
            **first_kwargs,
        )
        second_kwargs = _chunk(torch.tensor([[2.0, 0.0, 0.0]]))
        second = model(
            second_kwargs.pop("hand_positions"),
            second_kwargs.pop("pred_hand_contact_prob"),
            second_kwargs.pop("obj_trans_init"),
            enable_refine=False,
            prediction_state=state,
            **second_kwargs,
        )

    # First chunk's neutral: ||(1,0,0) - 0|| = 1 for both hands.
    assert torch.allclose(first["lb_prior_quantiles"], torch.ones(1, 2, 3))
    # Continuation keeps it even though the new chunk's init would give 2.
    assert torch.allclose(second["lb_prior_quantiles"], torch.ones(1, 2, 3))


def test_sample_gt_lb_prior_stats_marks_missing_hands_as_nan():
    torch.manual_seed(11)
    batch_size, seq_len = 3, 30
    batch = {
        "lhand_lb": torch.full((batch_size, seq_len), 0.4),
        "rhand_lb": torch.full((batch_size, seq_len), 0.6),
        "lhand_contact": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "rhand_contact": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }

    out = sample_gt_lb_prior_stats(
        batch,
        torch.device("cpu"),
        torch.float32,
        min_contact_frames=5,
        noise_std=0.0,
        drop_prob=0.0,
    )

    assert isinstance(out, torch.Tensor)
    assert out.shape == (batch_size, 2, 3)
    # Right hand never touches: NaN for every sample.
    assert torch.isnan(out[:, 1]).all()
    # Left hand: samples whose random prefix is long enough carry the GT value.
    valid_left = torch.isfinite(out[:, 0]).all(dim=-1)
    if bool(valid_left.any()):
        assert torch.allclose(out[valid_left, 0], torch.full_like(out[valid_left, 0], 0.4))


def test_lb_consistency_loss_is_sliding_safe():
    """A constant offset (wrong length, correct temporal change) must not be
    penalized by the consistency term, while a wrong temporal change is."""
    batch_size, seq_len = 1, 5
    obj_trans_gt = torch.zeros(batch_size, seq_len, 3)
    # GT length slides from 0.4 to 0.8 while in contact.
    obj_trans_gt[..., 0] = torch.linspace(0.4, 0.8, seq_len)
    batch = {
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
        "obj_trans": obj_trans_gt,
        "position_global": torch.zeros(batch_size, seq_len, 24, 3),
        "lhand_contact": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "rhand_contact": torch.zeros(batch_size, seq_len, dtype=torch.bool),
        "obj_contact": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }
    gt_lb = obj_trans_gt.norm(dim=-1)

    loss_fn = ObjectTransLoss(weights={"lhand_lb_cons": 1.0})

    # Same per-frame delta as GT, constant offset: cons loss must be zero.
    offset_pred = {"pred_lhand_lb": gt_lb + 0.5, "pred_rhand_lb": gt_lb}
    _, losses_offset, _ = loss_fn(offset_pred, batch, torch.device("cpu"))
    assert torch.allclose(losses_offset["lhand_lb_cons"], torch.tensor(0.0), atol=1e-8)
    assert losses_offset["lhand_lb"].item() > 0

    # Flat prediction while GT slides: cons loss must fire.
    flat_pred = {"pred_lhand_lb": torch.full_like(gt_lb, 0.6), "pred_rhand_lb": gt_lb}
    _, losses_flat, _ = loss_fn(flat_pred, batch, torch.device("cpu"))
    assert losses_flat["lhand_lb_cons"].item() > 0


def test_gate_weak_target_prefers_gt_contact_when_available():
    batch_size, seq_len = 1, 2
    # Non-uniform logits so the two candidate targets give different losses.
    gate_logits = torch.tensor([[[2.0, 0.0, 0.0, -1.0]] * seq_len])
    base_batch = {
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
        "obj_trans": torch.zeros(batch_size, seq_len, 3),
    }
    pred = {
        "pred_obj_trans": torch.zeros(batch_size, seq_len, 3),
        "gate_logits": gate_logits,
        # Deliberately wrong predicted contact: object static, no hands.
        "gating_contact_prob": torch.zeros(batch_size, seq_len, 3),
    }
    loss_fn = ObjectTransLoss(
        weights={"obj_trans": 0.0, "gate_weak": 1.0, "gate_weak_target_smoothing": 0.0}
    )

    # Without GT contact the target follows the (wrong) predicted contact:
    # static branch (index 3) is preferred.
    _, losses_pred, _ = loss_fn(pred, base_batch, torch.device("cpu"))
    expected_static = -torch.log_softmax(gate_logits, dim=-1)[..., 3].mean()
    assert torch.allclose(losses_pred["gate_weak"], expected_static)

    # With GT contact (left hand grasps a moving object) the target must
    # come from GT: left-FK branch (index 0).
    gt_batch = dict(base_batch)
    gt_batch["lhand_contact"] = torch.ones(batch_size, seq_len, dtype=torch.bool)
    gt_batch["rhand_contact"] = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    gt_batch["obj_contact"] = torch.ones(batch_size, seq_len, dtype=torch.bool)
    _, losses_gt, _ = loss_fn(pred, gt_batch, torch.device("cpu"))
    expected_left = -torch.log_softmax(gate_logits, dim=-1)[..., 0].mean()
    assert torch.allclose(losses_gt["gate_weak"], expected_left)


def test_model_selection_total_excludes_refine_terms():
    batch_size, seq_len = 1, 2
    num_reduced = len(_REDUCED_POSE_NAMES)
    batch = {
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
        "obj_trans": torch.zeros(batch_size, seq_len, 3),
        "trans": torch.ones(batch_size, seq_len, 3),
    }
    pred = {
        "pred_obj_trans": torch.zeros(batch_size, seq_len, 3),
        "refined_root_trans": torch.zeros(batch_size, seq_len, 3),
    }
    loss_fn = ObjectTransLoss(weights={"refine_root_trans": 10.0})
    total, test_losses = loss_fn.compute_test_loss(pred, batch, torch.device("cpu"))

    assert test_losses["refine_root_trans"].item() > 0
    assert "refine_root_trans" not in loss_fn.MODEL_SELECTION_KEYS
    assert total.item() < test_losses["refine_root_trans"].item()
