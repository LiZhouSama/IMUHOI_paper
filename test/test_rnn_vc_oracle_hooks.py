"""Diagnostic VC oracle hooks used by eval_batch_online --oracle_vc_* / --eval_vc_contact_init."""
from types import SimpleNamespace

import numpy as np
import torch

import eval_batch_online
from model.rnn.imuhoi_model import IMUHOIModel


def test_apply_vc_oracle_replaces_only_finite_entries():
    pred = torch.full((1, 4, 3), 0.3)
    oracle = torch.full((1, 4, 3), float("nan"))
    oracle[0, 1:3, 0] = 1.0
    out = IMUHOIModel._apply_vc_oracle(
        {
            "pred_hand_contact_prob": pred,
            "pred_obj_move_prob": pred[..., 2:3],
            "pred_hand_contact_prob_marginal": pred[..., :2],
            "pred_obj_vel": torch.zeros(1, 4, 3),
        },
        {"oracle_vc_contact": oracle, "oracle_vc_obj_vel": torch.ones(1, 4, 3)},
    )
    prob = out["pred_hand_contact_prob"]
    assert torch.equal(prob[0, :, 0], torch.tensor([0.3, 1.0, 1.0, 0.3]))
    assert torch.equal(prob[..., 1:], pred[..., 1:])
    assert torch.equal(out["pred_hand_contact_prob_marginal"], prob[..., :2])
    assert torch.equal(out["pred_obj_move_prob"], prob[..., 2:3])
    assert torch.equal(out["pred_obj_vel"], torch.ones(1, 4, 3))
    # No oracle keys: the output object is returned untouched.
    plain = {"pred_hand_contact_prob": pred}
    assert IMUHOIModel._apply_vc_oracle(plain, {}) is plain


def test_build_vc_oracles_region_and_shift():
    labels = torch.zeros(1, 40, dtype=torch.bool)
    labels[0, 10:30] = True
    raw = {
        "lhand_contact": labels,
        "rhand_contact": torch.zeros_like(labels),
        "obj_contact": labels,
        "obj_vel": torch.zeros(1, 40, 3),
    }
    spec = {"contact": "hands", "region": "boundary", "boundary_frames": 3, "shift": 0, "obj_vel": True}
    oracles = eval_batch_online._build_vc_oracles(raw, spec, torch.device("cpu"), torch.float32)
    contact = oracles["oracle_vc_contact"][0]
    kept = torch.isfinite(contact[:, 0]).numpy()
    # Boundaries sit at 9.5 and 29.5: frames within 3 of them are kept.
    assert np.flatnonzero(kept).tolist() == [7, 8, 9, 10, 11, 12, 27, 28, 29, 30, 31, 32]
    assert torch.isnan(contact[:, 2]).all()  # the move channel is not overridden
    assert "oracle_vc_obj_vel" in oracles

    spec = {"contact": "hands", "region": "all", "shift": 5}
    shifted = eval_batch_online._build_vc_oracles(raw, spec, torch.device("cpu"), torch.float32)
    onsets = np.flatnonzero(np.diff(shifted["oracle_vc_contact"][0, :, 0].numpy()) > 0) + 1
    assert onsets.tolist() == [15]


def test_vc_window_contact_init_modes():
    model = IMUHOIModel.__new__(IMUHOIModel)
    init = torch.full((2, 3), 0.7)
    window = {"contact_init": init, "oracle_vc_contact_init": torch.zeros(2, 5, 3)}
    for mode, expected in ((None, init), ("gt", torch.zeros(2, 3)), ("neutral", torch.full((2, 3), 0.5))):
        model.cfg = SimpleNamespace(eval_vc_contact_init=mode)
        assert torch.equal(model._vc_window_contact_init(window), expected)
