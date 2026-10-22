"""Bimanual closure (object_trans_bimanual_closure): five-way fusion with a state-free hand-pair candidate."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rnn.object_trans import ObjectTransModule  # noqa: E402
from train.rnn.loss.object_trans_loss import ObjectTransLoss  # noqa: E402
from utils.utils import load_config  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/IMUHOI_train_rnn.yaml")


def _module(closure=True, feedback=True, static_gate="none"):
    cfg = load_config(CONFIG)
    cfg.object_trans_state_feedback = feedback
    cfg.object_trans_bimanual_closure = closure
    cfg.object_trans_static_gate = static_gate
    torch.manual_seed(0)
    return ObjectTransModule(cfg).eval(), cfg


def _inputs(B=3, T=16, seed=0, contact=None):
    g = torch.Generator().manual_seed(seed)
    hands = torch.randn(B, T, 2, 3, generator=g) * 0.3
    prob = torch.rand(B, T, 3, generator=g) if contact is None else contact
    init = torch.randn(B, 3, generator=g) * 0.3
    obj_imu = torch.cat((torch.randn(B, T, 3, generator=g), torch.tensor([1.0, 0, 0, 0, 1, 0]).expand(B, T, 6)), -1)
    human_imu = torch.cat((torch.randn(B, T, 6, 3, generator=g), torch.tensor([1.0, 0, 0, 0, 1, 0]).expand(B, T, 6, 6)), -1)
    return dict(hand_positions=hands, pred_hand_contact_prob=prob, obj_trans_init=init, obj_imu=obj_imu,
                human_imu=human_imu, contact_init=torch.zeros(B, 3), has_object_mask=torch.ones(B, dtype=torch.bool),
                root_trans_input=torch.randn(B, T, 3, generator=g), enable_refine=False)


def test_outputs_and_five_way_fusion_identity():
    m, _ = _module()
    inp = _inputs(contact=torch.ones(3, 16, 3))
    with torch.no_grad():
        out = m(**inp)
    B, T = 3, 16
    for key, shape in (("pred_bimanual_obj_trans", (B, T, 3)), ("bimanual_prior_c", (B, T, 3)), ("bimanual_weight", (B, T)),
                       ("gating_weights_5", (B, T, 5)), ("bimanual_logit", (B, T)), ("bimanual_consistency", (B, T, 9))):
        assert tuple(out[key].shape) == shape, key
    w5 = out["gating_weights_5"]
    assert torch.allclose(w5.sum(-1), torch.ones(B, T), atol=1e-5)
    # published four-way weights fold the bimanual weight into the hand branches
    assert torch.allclose(out["gating_weights"][..., 0], w5[..., 0] + 0.5 * w5[..., 4])
    assert torch.allclose(out["gating_weights"][..., 2:], w5[..., 2:4])
    fused = (w5[..., 0:1] * out["pred_lhand_obj_trans"] + w5[..., 1:2] * out["pred_rhand_obj_trans"]
             + w5[..., 2:3] * out["pred_imu_obj_trans"] + w5[..., 3:4] * out["pred_static_obj_trans"]
             + w5[..., 4:5] * out["pred_bimanual_obj_trans"])
    assert torch.allclose(fused, out["pred_obj_trans"], atol=1e-5)
    assert (out["bimanual_weight"] > 0).all()
    # candidate = hand midpoint + R c (identity rotation here)
    midpoint = inp["hand_positions"].mean(dim=2)
    assert torch.allclose(out["pred_bimanual_obj_trans"], midpoint + out["bimanual_prior_c"], atol=1e-5)


def test_candidate_masked_unless_both_hands_in_contact():
    m, _ = _module()
    contact = torch.ones(3, 16, 3)
    contact[:, :8, 0] = 0.05  # left hand free for the first half
    with torch.no_grad():
        out = m(**_inputs(contact=contact))
    assert (out["bimanual_weight"][:, :8] == 0).all()
    assert (out["bimanual_weight"][:, 8:] > 0).all()


def test_prior_is_independent_of_the_object_state():
    m, _ = _module()
    inp = _inputs(contact=torch.ones(3, 16, 3))
    shifted = dict(inp)
    shifted["obj_trans_init"] = inp["obj_trans_init"] + 0.7
    with torch.no_grad():
        a = m(**inp)
        b = m(**shifted)
    assert torch.allclose(a["bimanual_prior_c"], b["bimanual_prior_c"], atol=1e-6)
    assert torch.allclose(a["pred_bimanual_obj_trans"], b["pred_bimanual_obj_trans"], atol=1e-6)
    assert not torch.allclose(a["pred_lhand_obj_trans"], b["pred_lhand_obj_trans"])  # FK bones still read the state


def test_static_gate_pins_all_five_branches():
    m, _ = _module(static_gate="imu")
    inp = _inputs(contact=torch.ones(3, 16, 3))
    inp["obj_imu"][..., :3] = 0.0  # zero acceleration + constant rotation -> IMU rest
    with torch.no_grad():
        out = m(**inp)
    w5 = out["gating_weights_5"][:, 8:]  # after the causal window
    assert torch.allclose(w5[..., 3], torch.ones_like(w5[..., 3]))
    assert (w5[..., 4] == 0).all()


def test_baseline_has_no_closure_outputs():
    m, _ = _module(closure=False)
    with torch.no_grad():
        out = m(**_inputs())
    assert "pred_bimanual_obj_trans" not in out and "gating_weights_5" not in out


def test_losses_and_gradients():
    m, _ = _module()
    m.train()
    B, T = 3, 16
    inp = _inputs(contact=torch.ones(B, T, 3))
    out = m(**inp)
    g = torch.Generator().manual_seed(3)
    batch = {"human_imu": inp["human_imu"], "obj_trans": torch.randn(B, T, 3, generator=g), "obj_vel": torch.zeros(B, T, 3),
             "obj_rot": torch.tensor([1.0, 0, 0, 0, 1, 0]).expand(B, T, 6).clone(),
             "has_object": torch.ones(B, dtype=torch.bool), "lhand_contact": torch.ones(B, T, dtype=torch.bool),
             "rhand_contact": torch.ones(B, T, dtype=torch.bool), "obj_contact": torch.ones(B, T, dtype=torch.bool),
             "position_global": torch.randn(B, T, 24, 3, generator=g), "lhand_obj_direction": torch.randn(B, T, 3, generator=g),
             "rhand_obj_direction": torch.randn(B, T, 3, generator=g)}
    zero_default = ObjectTransLoss({"obj_trans": 1.0})
    _, losses, weighted = zero_default(out, batch, "cpu")
    assert float(losses["bimanual_hoi"]) > 0 and float(losses["bimanual_prior"]) > 0
    assert float(weighted["bimanual_hoi"]) == 0 and float(weighted["bimanual_prior"]) == 0  # zero unless configured
    total, _, _ = ObjectTransLoss({"obj_trans": 1.0, "hoi_fused_l": 10.0, "hoi_fused_r": 10.0,
                                   "bimanual_hoi": 5.0, "bimanual_prior": 1.0})(out, batch, "cpu")
    total.backward()
    for name in ("bimanual_closure.prior_head.weight", "bimanual_closure.gate_head.3.weight", "bimanual_closure.temporal.weight_ih_l0"):
        grad = dict(m.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0, name


def test_cannot_combine_with_bigrasp():
    cfg = load_config(CONFIG)
    cfg.object_trans_state_feedback = True
    cfg.object_trans_bimanual_closure = True
    cfg.object_trans_bigrasp = True
    with pytest.raises(ValueError):
        ObjectTransModule(cfg)
