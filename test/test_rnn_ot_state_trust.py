"""State trust for ObjectTrans: state-free prior, consistency features, supervised confidence, early stopping."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rnn.object_trans import ObjectTransModule  # noqa: E402
from train.rnn.loss.object_trans_loss import ObjectTransLoss  # noqa: E402
from utils.utils import load_config  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/IMUHOI_train_rnn.yaml")


def _module(state_trust=True, feedback=True):
    cfg = load_config(CONFIG)
    cfg.object_trans_state_feedback = feedback
    cfg.object_trans_state_trust = state_trust
    torch.manual_seed(0)
    return ObjectTransModule(cfg).eval(), cfg


def _inputs(B=3, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    hands = torch.randn(B, T, 2, 3, generator=g) * 0.3
    prob = torch.rand(B, T, 3, generator=g)
    init = torch.randn(B, 3, generator=g) * 0.3
    obj_imu = torch.cat((torch.randn(B, T, 3, generator=g), torch.tensor([1.0, 0, 0, 0, 1, 0]).expand(B, T, 6)), -1)
    human_imu = torch.randn(B, T, 6, 9, generator=g)
    return dict(hand_positions=hands, pred_hand_contact_prob=prob, obj_trans_init=init, obj_imu=obj_imu,
                human_imu=human_imu, contact_init=torch.zeros(B, 3), has_object_mask=torch.ones(B, dtype=torch.bool),
                enable_refine=False)


def test_requires_state_feedback():
    with pytest.raises(ValueError):
        _module(state_trust=True, feedback=False)


def test_outputs_and_confidence_range():
    m, _ = _module()
    with torch.no_grad():
        out = m(**_inputs())
    B, T = 3, 16
    for key, shape in (("prior_lhand_obj_trans", (B, T, 3)), ("prior_rhand_obj_trans", (B, T, 3)),
                       ("state_confidence", (B, T)), ("state_confidence_logit", (B, T)),
                       ("ot_state_prev_obj_trans", (B, T, 3)), ("state_trust_rest_age", (B, T)),
                       ("state_trust_gap_age", (B, T)), ("state_trust_init_confidence", (B,))):
        assert tuple(out[key].shape) == shape, key
    assert (out["state_confidence"] > 0).all() and (out["state_confidence"] < 1).all()
    # the state fed to frame 0 is the window init; afterwards the fused position of t-1
    inp = _inputs()
    assert torch.allclose(out["ot_state_prev_obj_trans"][:, 0], inp["obj_trans_init"])
    assert torch.allclose(out["ot_state_prev_obj_trans"][:, 1:], out["pred_obj_trans"][:, :-1])


def test_prior_head_is_independent_of_the_object_state():
    m, _ = _module()
    inp = _inputs()
    shifted = dict(inp)
    shifted["obj_trans_init"] = inp["obj_trans_init"] + 0.5
    with torch.no_grad():
        a = m(**inp)
        b = m(**shifted)
    assert torch.allclose(a["prior_lhand_obj_trans"], b["prior_lhand_obj_trans"], atol=1e-6)
    assert torch.allclose(a["prior_rhand_obj_trans"], b["prior_rhand_obj_trans"], atol=1e-6)
    # the main head does depend on the state (feedback + init geometry)
    assert not torch.allclose(a["pred_lhand_obj_trans"], b["pred_lhand_obj_trans"])


def test_ages_are_causal_counters():
    m, _ = _module()
    inp = _inputs()
    inp["pred_hand_contact_prob"] = torch.zeros(3, 16, 3)   # never in contact -> gap age counts up
    inp["obj_imu"][..., :3] = 0.0                             # no acceleration and constant rotation -> at rest
    with torch.no_grad():
        out = m(**inp)
    assert torch.equal(out["state_trust_gap_age"][0], torch.arange(1, 17, dtype=out["state_trust_gap_age"].dtype))
    assert torch.equal(out["state_trust_rest_age"][0], torch.arange(1, 17, dtype=out["state_trust_rest_age"].dtype))


def test_baseline_module_has_no_trust_outputs_and_same_keys():
    m, _ = _module(state_trust=False)
    with torch.no_grad():
        out = m(**_inputs())
    assert "state_confidence" not in out and "prior_lhand_obj_trans" not in out


def test_losses_default_to_zero_weight_and_confidence_target():
    m, _ = _module()
    inp = _inputs()
    with torch.no_grad():
        out = m(**inp)
    B, T = 3, 16
    batch = {
        "human_imu": inp["human_imu"], "obj_trans": torch.zeros(B, T, 3), "obj_vel": torch.zeros(B, T, 3),
        "lhand_contact": torch.ones(B, T, dtype=torch.bool), "rhand_contact": torch.ones(B, T, dtype=torch.bool),
        "position_global": torch.zeros(B, T, 24, 3), "lhand_lb": torch.full((B, T), 0.3), "rhand_lb": torch.full((B, T), 0.3),
        "lhand_obj_direction": torch.tensor([1.0, 0, 0]).expand(B, T, 3), "rhand_obj_direction": torch.tensor([1.0, 0, 0]).expand(B, T, 3),
    }
    _, losses, weighted = ObjectTransLoss(weights={}).compute_loss(out, batch, torch.device("cpu"))
    for key in ("prior_hoi_l", "prior_hoi_r", "state_conf"):
        assert key in losses and float(weighted[key]) == 0.0
    assert float(losses["state_conf"]) > 0.0
    _, _, weighted2 = ObjectTransLoss(weights={"state_conf": 1.0, "prior_hoi_l": 5.0}).compute_loss(out, batch, torch.device("cpu"))
    assert float(weighted2["state_conf"]) > 0.0 and float(weighted2["prior_hoi_l"]) > 0.0


def test_early_stopping_rule():
    from train.rnn.train_object_trans import Stage3JointTrainer
    stop = Stage3JointTrainer.should_stop_early
    assert not stop(epoch=50, last_improvement_epoch=0, patience_epochs=0, min_epochs=0)      # disabled
    assert not stop(epoch=30, last_improvement_epoch=0, patience_epochs=20, min_epochs=40)    # before min epochs
    assert stop(epoch=40, last_improvement_epoch=10, patience_epochs=30, min_epochs=40)
    assert not stop(epoch=40, last_improvement_epoch=20, patience_epochs=30, min_epochs=40)
    assert not stop(epoch=40, last_improvement_epoch=0, patience_epochs=30, min_epochs=40, has_test_loader=False)


def test_rest_state_perturbation_targets_rest_frames_and_persists():
    """The rest-phase corruption is injected on a GT-rest frame and the model's state carries it."""
    from easydict import EasyDict as edict
    from train.rnn.train_object_trans import Stage3JointTrainer
    m, cfg = _module()
    cfg.object_trans_rest_perturb_std = 0.2
    cfg.object_trans_rest_perturb_prob = 1.0
    cfg.object_trans_rest_perturb_vel = 0.02
    fake = edict(cfg=cfg, device=torch.device("cpu"))
    B, T = 2, 30
    obj_trans = torch.zeros(B, T, 3)
    obj_trans[:, 20:, 0] = torch.linspace(0, 0.5, 10)  # moving in the last 10 frames only
    batch = {"obj_trans": obj_trans}
    torch.manual_seed(1)
    fc = Stage3JointTrainer._build_rest_state_perturbation(fake, batch, {"has_object": torch.ones(B, dtype=torch.bool)}, training=True)
    assert fc is not None and fc["mode"] == "perturb"
    inject = fc["injection_mask"].nonzero()
    assert inject.shape[0] == B
    assert (inject[:, 1] >= 5).all() and (inject[:, 1] < 20).all()   # rest frames only, away from the edges
    assert Stage3JointTrainer._build_rest_state_perturbation(fake, batch, {}, training=False) is None
    inp = _inputs(B=B, T=T)
    with torch.no_grad():
        clean = m(**inp)
        pert = m(**inp, feedback_control=fc)
    t0 = int(inject[0, 1])
    d = (pert["ot_state_prev_obj_trans"][0] - clean["ot_state_prev_obj_trans"][0]).norm(dim=-1)
    assert torch.all(d[:t0] < 1e-6)          # identical before the injection frame
    assert d[t0] > 1e-3                      # the state seen at the injection frame carries the shift


def test_force_confidence_overrides_the_head_at_inference():
    m, cfg = _module()
    inp = _inputs()
    with torch.no_grad():
        learned = m(**inp)
    m.state_trust_force_confidence = 1.0
    with torch.no_grad():
        forced = m(**inp)
    assert torch.all(forced["state_confidence"] == 1.0)
    assert torch.all(forced["state_trust_init_confidence"] == 1.0)
    # the head still runs (frame 0 sees the same init state in both runs); later
    # logits differ because the fused track (hence the state) differs once the
    # gating changed.
    assert torch.allclose(forced["state_confidence_logit"][:, 0], learned["state_confidence_logit"][:, 0])
    assert not torch.allclose(forced["pred_obj_trans"], learned["pred_obj_trans"])
