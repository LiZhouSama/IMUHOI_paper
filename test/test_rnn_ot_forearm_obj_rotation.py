"""Forearm-vs-object rotation input (object_trans_forearm_obj_rotation)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rnn.object_trans import ObjectTransModule  # noqa: E402
from utils.rotation_conversions import matrix_to_rotation_6d  # noqa: E402
from utils.utils import load_config  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/IMUHOI_train_rnn.yaml")
IDENTITY6 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _module(enabled=True, **extra):
    cfg = load_config(CONFIG)
    cfg.object_trans_state_feedback = True
    cfg.object_trans_forearm_obj_rotation = enabled
    for key, value in extra.items():
        setattr(cfg, key, value)
    torch.manual_seed(0)
    return ObjectTransModule(cfg).eval()


def _random_rot6d(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(*shape, 3, 3, generator=g)
    u, _, vh = torch.linalg.svd(q)
    r = u @ vh
    det = torch.linalg.det(r)
    r[..., 2, :] = r[..., 2, :] * det.unsqueeze(-1).sign()
    return matrix_to_rotation_6d(r), r


def _inputs(B=2, T=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    obj6, obj_rot = _random_rot6d(B, T, seed=seed + 1)
    human6, human_rot = _random_rot6d(B, T, 6, seed=seed + 2)
    obj_imu = torch.cat((torch.randn(B, T, 3, generator=g), obj6), dim=-1)
    human_imu = torch.cat((torch.randn(B, T, 6, 3, generator=g), human6), dim=-1)
    data = dict(hand_positions=torch.randn(B, T, 2, 3, generator=g) * 0.3,
                pred_hand_contact_prob=torch.rand(B, T, 3, generator=g),
                obj_trans_init=torch.randn(B, 3, generator=g) * 0.3,
                obj_imu=obj_imu, human_imu=human_imu, contact_init=torch.zeros(B, 3),
                has_object_mask=torch.ones(B, dtype=torch.bool),
                root_trans_input=torch.randn(B, T, 3, generator=g), enable_refine=False)
    return data, obj_rot, human_rot


def test_input_and_init_widths_follow_the_flag():
    off, on = _module(enabled=False), _module(enabled=True)
    assert on.n_prediction_input - off.n_prediction_input == 2 * ObjectTransModule.FOREARM_OBJ_FEATURE_DIM == 38
    assert on.object_trans_head.linear1.in_features == on.n_prediction_input
    assert on.object_trans_head.init_net[0].in_features == off.object_trans_head.init_net[0].in_features + 12
    data, _, _ = _inputs()
    with torch.no_grad():
        out_on, out_off = on(**data), off(**data)
    for o in (out_on, out_off):
        assert o["pred_obj_trans"].shape == (2, 12, 3)


def test_features_equal_forearm_in_the_object_frame():
    module = _module()
    data, obj_rot, human_rot = _inputs()
    feat, init = module._forearm_object_features(data["human_imu"], obj_rot)
    assert feat.shape == (2, 12, 38) and init.shape == (2, 12)
    world_arm = human_rot[:, :, 0] @ human_rot[:, :, 4]          # root world * root-relative forearm
    expected = obj_rot.transpose(-1, -2) @ world_arm             # R_obj^T R_forearm
    torch.testing.assert_close(feat[..., :6], matrix_to_rotation_6d(expected), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(init[:, :6], matrix_to_rotation_6d(expected)[:, 0], atol=1e-5, rtol=1e-5)
    # the rotation since the window start is identity at t=0, and its angle grows from 0
    torch.testing.assert_close(feat[:, 0, 12:18], IDENTITY6.expand(2, 6), atol=1e-5, rtol=1e-5)
    assert torch.allclose(feat[:, 0, 18], torch.zeros(2), atol=1e-5)
    assert (feat[..., 18] >= -1e-6).all() and (feat[..., 18] <= 3.1416).all()


def test_features_are_invariant_to_a_global_rotation_and_causal():
    module = _module()
    data, obj_rot, human_rot = _inputs()
    feat, _ = module._forearm_object_features(data["human_imu"], obj_rot)
    g6, g = _random_rot6d(seed=99)
    rotated_obj = g @ obj_rot                                     # rotate the scene: object and root follow
    rotated_human = data["human_imu"].clone()
    rotated_human[:, :, 0, 3:9] = matrix_to_rotation_6d(g @ human_rot[:, :, 0])
    feat_rot, _ = module._forearm_object_features(rotated_human, rotated_obj)
    torch.testing.assert_close(feat, feat_rot, atol=1e-5, rtol=1e-5)
    prefix, _ = module._forearm_object_features(data["human_imu"][:, :7], obj_rot[:, :7])
    torch.testing.assert_close(feat[:, :7], prefix, atol=1e-6, rtol=1e-6)


def test_object_rotation_changes_the_prediction():
    module = _module()
    data, obj_rot, _ = _inputs()
    other, _, _ = _inputs(seed=5)
    swapped = dict(data)
    swapped["obj_imu"] = torch.cat((data["obj_imu"][..., :3], other["obj_imu"][..., 3:9]), dim=-1)
    with torch.no_grad():
        assert not torch.allclose(module(**data)["pred_lhand_obj_direction"], module(**swapped)["pred_lhand_obj_direction"])


def test_checkpoint_from_the_other_width_is_rejected():
    off, on = _module(enabled=False), _module(enabled=True)
    with pytest.raises(ValueError):
        on.validate_checkpoint_state_dict(off.state_dict())
    with pytest.raises(ValueError):
        off.validate_checkpoint_state_dict(on.state_dict())


@pytest.mark.parametrize("extra", [{}, {"object_trans_bimanual_closure": True}, {"object_trans_grasp_latent": True}])
def test_runs_with_the_other_ot_options(extra):
    module = _module(**extra)
    data, _, _ = _inputs()
    with torch.no_grad():
        out = module(**data)
    assert torch.isfinite(out["pred_obj_trans"]).all()
