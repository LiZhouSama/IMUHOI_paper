"""Arch-scoped ObjectTrans loss weights: one arch's recipe must not move another's."""
import os
import sys

import pytest
from easydict import EasyDict as edict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.rnn.loss import resolve_object_trans_loss_weights  # noqa: E402
from utils.utils import load_config  # noqa: E402

CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs/IMUHOI_train_rnn.yaml",
)


def _cfg(arch, per_arch=None, shared=None):
    return edict({
        "object_trans_arch": arch,
        "loss_weights": shared if shared is not None else {"obj_trans": 5, "hoi_error_l": 10},
        **({"object_trans_arch_loss_weights": per_arch} if per_arch is not None else {}),
    })


def test_selected_arch_block_overrides_shared_weights():
    per_arch = {"current": {"obj_trans": 1.0, "hoi_fused_l": 10.0}}
    weights = resolve_object_trans_loss_weights(_cfg("current", per_arch))
    assert weights["obj_trans"] == 1.0
    assert weights["hoi_fused_l"] == 10.0
    # untouched shared entries survive
    assert weights["hoi_error_l"] == 10


def test_other_archs_keep_the_shared_weights():
    per_arch = {"current": {"obj_trans": 1.0, "hoi_fused_l": 10.0, "hoi_fused_r": 10.0}}
    for arch in ("v3", "v2", "legacy"):
        weights = resolve_object_trans_loss_weights(_cfg(arch, per_arch))
        assert weights["obj_trans"] == 5
        # absent key means weight 0 in ObjectTransLoss, i.e. the term is off
        assert "hoi_fused_l" not in weights
        assert "hoi_fused_r" not in weights


def test_missing_block_is_a_no_op():
    shared = {"obj_trans": 5, "hoi_error_l": 10}
    assert resolve_object_trans_loss_weights(_cfg("current", shared=shared)) == shared
    assert resolve_object_trans_loss_weights(_cfg("current", {}, shared=shared)) == shared
    assert resolve_object_trans_loss_weights(_cfg("current", {"v3": {"obj_trans": 9}}, shared=shared)) == shared


def test_arch_name_is_case_insensitive_and_stripped():
    per_arch = {"current": {"obj_trans": 1.0}}
    assert resolve_object_trans_loss_weights(_cfg(" Current ", per_arch))["obj_trans"] == 1.0


def test_malformed_block_raises():
    with pytest.raises(ValueError):
        resolve_object_trans_loss_weights(_cfg("current", ["current"]))
    with pytest.raises(ValueError):
        resolve_object_trans_loss_weights(_cfg("current", {"current": 1.0}))


def test_shipped_config_gives_current_the_cur_hoi_recipe_and_leaves_v3_alone():
    cfg = load_config(CONFIG)
    assert cfg.object_trans_arch == "current"
    current = resolve_object_trans_loss_weights(cfg)
    assert current["obj_trans"] == 1.0
    assert current["hoi_fused_l"] == 10.0 and current["hoi_fused_r"] == 10.0

    cfg.object_trans_arch = "v3"
    v3 = resolve_object_trans_loss_weights(cfg)
    assert v3 == dict(cfg.loss_weights)
    assert v3["obj_trans"] == 5
    assert "hoi_fused_l" not in v3 and "hoi_fused_r" not in v3
