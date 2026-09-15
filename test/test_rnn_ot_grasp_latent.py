"""Event-anchored grasp latent for ObjectTrans (object_trans_grasp_latent)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rnn.object_trans import ObjectTransModule  # noqa: E402
from model.rnn.grasp_state import GraspLatent  # noqa: E402
from train.rnn.loss.object_trans_loss import ObjectTransLoss  # noqa: E402
from utils.utils import load_config  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/IMUHOI_train_rnn.yaml")


def _module(grasp=True, feedback=True, modes=False, **extra):
    cfg = load_config(CONFIG)
    cfg.object_trans_state_feedback = feedback
    cfg.object_trans_grasp_latent = grasp
    cfg.object_trans_grasp_modes = modes
    for key, value in extra.items():
        setattr(cfg, key, value)
    torch.manual_seed(0)
    return ObjectTransModule(cfg).eval(), cfg


def _inputs(B=3, T=16, seed=0, probs=None):
    g = torch.Generator().manual_seed(seed)
    hands = torch.randn(B, T, 2, 3, generator=g) * 0.3
    prob = torch.rand(B, T, 3, generator=g) if probs is None else probs
    init = torch.randn(B, 3, generator=g) * 0.3
    obj_imu = torch.cat((torch.randn(B, T, 3, generator=g), torch.tensor([1.0, 0, 0, 0, 1, 0]).expand(B, T, 6)), -1)
    human_imu = torch.randn(B, T, 6, 9, generator=g)
    return dict(hand_positions=hands, pred_hand_contact_prob=prob, obj_trans_init=init, obj_imu=obj_imu,
                human_imu=human_imu, contact_init=torch.zeros(B, 3), has_object_mask=torch.ones(B, dtype=torch.bool),
                enable_refine=False)


def _segment_probs(B=3, T=16, hold=(0, 8)):
    """Left hand in contact on frames [hold[0], hold[1]), right hand never, object moving."""
    probs = torch.full((B, T, 3), 0.05)
    probs[:, hold[0]:hold[1], 0] = 0.9
    probs[..., 2] = 0.9
    return probs


def test_flag_off_leaves_the_baseline_untouched():
    m, _ = _module(grasp=False)
    assert m.feedback_feature_dim == 13
    assert not hasattr(m, "grasp_latent")
    with torch.no_grad():
        out = m(**_inputs())
    assert not any(key.startswith("grasp") for key in out)
    assert "prior_lhand_obj_trans" not in out


def test_requires_feedback_and_excludes_state_trust():
    with pytest.raises(ValueError):
        _module(grasp=True, feedback=False)
    with pytest.raises(ValueError):
        _module(grasp=True, object_trans_state_trust=True)


def test_outputs_and_hysteresis_release_to_prior():
    m, _ = _module()
    probs = _segment_probs()
    inp = _inputs(probs=probs)
    with torch.no_grad():
        out = m(**inp)
    B, T = 3, 16
    for key, shape in (("grasp_latent", (B, T, 6)), ("grasp_active", (B, T, 2)), ("grasp_age", (B, T, 2)),
                       ("grasp_alpha", (B, T, 2)), ("grasp_onset", (B, T, 2)), ("grasp_delta", (B, T, 6)),
                       ("prior_lhand_obj_trans", (B, T, 3)), ("prior_rhand_obj_trans", (B, T, 3)),
                       ("grasp_prior_bone", (B, T, 2, 3)), ("grasp_init_alpha", (B, 2))):
        assert tuple(out[key].shape) == shape, key
    active = out["grasp_active"] > 0.5
    assert active[:, :8, 0].all() and not active[:, 8:, 0].any()   # left: held, then released
    assert not active[..., 1].any()                                  # right: never above the on threshold
    assert torch.equal(out["grasp_age"][:, :8, 0], torch.arange(1, 9, dtype=out["grasp_age"].dtype).expand(3, -1))
    # zero-initialised increments: a hard latch inside the segment
    latent = out["grasp_latent"].view(B, T, 2, 3)
    assert torch.allclose(latent[:, 1:8, 0], latent[:, :1, 0].expand(-1, 7, -1), atol=1e-6)
    # a free hand follows the state-free prior
    assert torch.allclose(latent[:, 8:, 0], out["grasp_prior_bone"][:, 8:, 0], atol=1e-6)
    assert torch.allclose(latent[..., 1, :], out["grasp_prior_bone"][..., 1, :], atol=1e-6)
    assert m.SEGMENT_STATE_OUTPUT_KEYS == GraspLatent.STATE_KEYS


def test_window_init_anchoring_mixes_observed_geometry_with_the_prior():
    m, _ = _module(object_trans_grasp_fusion_bias=3.0)
    probs = _segment_probs()
    inp = _inputs(probs=probs)
    with torch.no_grad():
        out = m(**inp)
    alpha0 = out["grasp_init_alpha"]
    assert torch.allclose(alpha0[:, 0], torch.full((3,), torch.sigmoid(torch.tensor(3.0)).item()), atol=1e-5)
    assert torch.all(alpha0[:, 1] == 0)
    observed0 = out["init_lhand_oe_ho"] * out["init_lhand_lb"].unsqueeze(-1)
    expected = alpha0[:, :1] * observed0 + (1 - alpha0[:, :1]) * out["grasp_prior_bone"][:, 0, 0]
    assert torch.allclose(out["grasp_latent"][:, 0, :3], expected, atol=1e-5)


def test_object_state_enters_the_bone_only_at_onset():
    m, _ = _module()
    probs = _segment_probs(hold=(0, 16))
    inp = _inputs(probs=probs)
    B, T = 3, 16
    injection = torch.zeros(B, T, dtype=torch.bool)
    injection[:, 6] = True
    perturbation = torch.zeros(B, T, 3)
    perturbation[:, 6] = torch.tensor([0.3, -0.2, 0.25])
    with torch.no_grad():
        clean = m(**inp)
        perturbed = m(**inp, feedback_control={"mode": "perturb", "perturbation": perturbation, "injection_mask": injection})
    # the published object moves (static / IMU branches read the perturbed state)
    assert not torch.allclose(clean["pred_obj_trans"][:, 6:], perturbed["pred_obj_trans"][:, 6:])
    # ... but the held hand's bone is anchored at the onset and does not
    # re-read the object state frame by frame
    assert torch.allclose(clean["pred_lhand_lb"], perturbed["pred_lhand_lb"], atol=1e-6)
    assert torch.allclose(clean["pred_lhand_obj_direction"], perturbed["pred_lhand_obj_direction"], atol=1e-6)


def test_prior_head_is_independent_of_the_object_state():
    m, _ = _module()
    inp = _inputs()
    shifted = dict(inp)
    shifted["obj_trans_init"] = inp["obj_trans_init"] + 0.5
    with torch.no_grad():
        a = m(**inp)
        b = m(**shifted)
    assert torch.allclose(a["prior_lhand_obj_trans"], b["prior_lhand_obj_trans"], atol=1e-6)
    assert torch.allclose(a["grasp_prior_bone"], b["grasp_prior_bone"], atol=1e-6)
    assert not torch.allclose(a["pred_lhand_obj_trans"], b["pred_lhand_obj_trans"])


def test_online_windows_continue_the_segment_state():
    m, _ = _module()
    T = 40
    probs = torch.full((2, T, 3), 0.05)
    probs[:, 5:30, 0] = 0.9
    probs[:, 20:36, 1] = 0.9
    probs[..., 2] = 0.9
    inp = _inputs(B=2, T=T, probs=probs)
    with torch.no_grad():
        offline = m(**inp)
        online = m.inference(**inp, inference_mode="online", online_window=12)
    # active flags are a deterministic function of the probabilities given
    # the carried state, so the published online flags must equal offline
    assert torch.equal(online["grasp_active"] > 0.5, offline["grasp_active"] > 0.5)
    # exactly one onset per hand in both protocols (no re-anchoring at window starts)
    assert torch.equal((online["grasp_onset"] > 0.5).sum(dim=1), torch.tensor([[1, 1], [1, 1]]))
    assert torch.equal(online["grasp_age"], offline["grasp_age"])


def test_modes_scale_the_increment_and_grasp_mode_loss_is_supervised():
    m, cfg = _module(modes=True)
    inp = _inputs()
    with torch.no_grad():
        out = m(**inp)
    assert tuple(out["grasp_rigid_prob"].shape) == (3, 16, 2)
    assert torch.allclose(out["grasp_rigid_prob"], torch.full((3, 16, 2), 0.5))  # zero-initialised mode head
    B, T = 3, 16
    g = torch.Generator().manual_seed(1)
    batch = {
        "human_imu": inp["human_imu"], "obj_trans": torch.randn(B, T, 3, generator=g), "obj_vel": torch.zeros(B, T, 3),
        "position_global": torch.randn(B, T, 24, 3, generator=g), "lhand_obj_direction": torch.randn(B, T, 3, generator=g),
        "rhand_obj_direction": torch.randn(B, T, 3, generator=g), "lhand_contact": torch.ones(B, T, dtype=torch.bool),
        "rhand_contact": torch.ones(B, T, dtype=torch.bool), "obj_contact": torch.ones(B, T, dtype=torch.bool),
        "has_object": torch.ones(B, dtype=torch.bool),
    }
    default = ObjectTransLoss({})
    _, losses, weighted = default.compute_loss(out, batch, torch.device("cpu"))
    assert float(losses["grasp_mode"]) > 0 and float(weighted["grasp_mode"]) == 0.0  # zero default weight
    on = ObjectTransLoss({"grasp_mode": 1.0})
    _, _, weighted_on = on.compute_loss(out, batch, torch.device("cpu"))
    assert abs(float(weighted_on["grasp_mode"]) - float(losses["grasp_mode"])) < 1e-6
    # prior FK losses are available for the state-free posture prior
    assert float(losses["prior_hoi_l"]) > 0 and float(losses["prior_hoi_r"]) > 0


def test_residual_mode_holds_the_anchor_and_bounds_the_bone():
    """residual mode: the published latent is the onset anchor (constant inside a segment) and the
    FK bone deviates from it by at most residual_bound, so nothing integrates over long streams."""
    m, _ = _module(object_trans_grasp_increment_mode="residual", object_trans_grasp_residual_bound=0.05)
    # give the increment head non-zero weights so the residual is exercised
    with torch.no_grad():
        m.grasp_latent.increment.weight.normal_(0, 0.5)
        m.grasp_latent.increment.bias.normal_(0, 0.5)
    T = 16
    inp = _inputs(T=T, probs=_segment_probs(T=T, hold=(0, T)))  # left hand in contact all window, right free
    with torch.no_grad():
        out = m(**inp)
    latent = out["grasp_latent"].view(3, T, 2, 3)
    assert (out["grasp_active"][:, :, 0] > 0.5).all() and (out["grasp_active"][:, :, 1] < 0.5).all()
    assert torch.allclose(latent[:, 1:, 0], latent[:, :1, 0].expand(-1, T - 1, -1), atol=1e-6), "anchor must not drift"
    bone_l = out["pred_lhand_obj_direction"] * out["pred_lhand_lb"].unsqueeze(-1)
    deviation = (bone_l - latent[:, :, 0]).norm(dim=-1)
    assert deviation.max() <= 0.05 + 1e-5
    assert deviation.max() > 1e-4, "residual head should move the bone"
    # the free hand follows the prior exactly (no residual on inactive hands)
    bone_r = out["pred_rhand_obj_direction"] * out["pred_rhand_lb"].unsqueeze(-1)
    assert torch.allclose(bone_r, latent[:, :, 1], atol=1e-5)
