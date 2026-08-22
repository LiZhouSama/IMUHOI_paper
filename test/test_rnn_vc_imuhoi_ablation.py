from types import SimpleNamespace

import pytest
import torch
from pytorch3d.transforms import matrix_to_rotation_6d

from model.rnn.velocity_contact import build_velocity_contact_module


ARCHES = (
    "imuhoi_cond",
    "imuhoi_cond_bi",
    "imuhoi_cond_geom",
    "imuhoi_cond_boundary",
    "imuhoi_cond_boundary_endpoint",
)


def _cfg(arch):
    return SimpleNamespace(
        velocity_contact_arch=arch,
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        velocity_hidden_dim=16,
        velocity_num_layers=1,
        velocity_dropout=0.0,
        boundary_group_hidden=8,
        boundary_hidden_dim=8,
        velocity_contact_geom_detach_obj_vel=True,
        contact_from_segments=False,
    )


def _inputs(batch_size=2, seq_len=4):
    rot6d = matrix_to_rotation_6d(torch.eye(3)).reshape(1, 1, 1, 6)
    human = torch.cat(
        (
            torch.randn(batch_size, seq_len, 6, 3),
            rot6d.expand(batch_size, seq_len, 6, 6),
        ),
        dim=-1,
    )
    obj = torch.cat(
        (
            torch.randn(batch_size, seq_len, 3),
            rot6d[:, :, 0].expand(batch_size, seq_len, 6),
        ),
        dim=-1,
    )
    data = {
        "human_imu": human,
        "obj_imu": obj,
        "hand_vel_glb_init": torch.zeros(batch_size, 2, 3),
        "obj_vel_init": torch.zeros(batch_size, 3),
        "obj_trans_init": torch.zeros(batch_size, 3),
        "contact_init": torch.zeros(batch_size, 3),
    }
    hp_out = {
        "pred_full_pose_6d": rot6d.expand(batch_size, seq_len, 24, 6),
        "pred_joints_local": torch.randn(batch_size, seq_len, 24, 3),
        "pred_palm_glb_pos": torch.randn(batch_size, seq_len, 2, 3),
        "root_vel_pred": torch.randn(batch_size, seq_len, 3),
        "root_trans_pred": torch.randn(batch_size, seq_len, 3),
    }
    return data, hp_out


@pytest.mark.parametrize("arch", ARCHES)
def test_imuhoi_contact_ablation_forward_contract(arch):
    model = build_velocity_contact_module(_cfg(arch)).eval()
    data, hp_out = _inputs()
    with torch.no_grad():
        out = model(data, hp_out=hp_out)
    assert out["pred_hand_glb_vel"].shape == (2, 4, 2, 3)
    assert out["pred_obj_vel"].shape == (2, 4, 3)
    assert out["pred_hand_contact_prob"].shape == (2, 4, 3)
    torch.testing.assert_close(
        out["pred_hand_contact_prob"][..., :2],
        out["pred_obj_move_prob"] * out["pred_hand_contact_prob_cond"],
    )
    assert ("pred_contact_endpoint_prob" in out) == arch.endswith("endpoint")


def test_only_obj_move_ablation_is_bidirectional():
    causal = build_velocity_contact_module(_cfg("imuhoi_cond"))
    bidirectional = build_velocity_contact_module(_cfg("imuhoi_cond_bi"))
    assert causal.obj_move_net.num_directions == 1
    assert bidirectional.obj_move_net.num_directions == 2
    assert causal.hand_contact_net.num_directions == 1
    assert bidirectional.hand_contact_net.num_directions == 1


def test_ablation_features_are_incremental_and_isolated():
    base = build_velocity_contact_module(_cfg("imuhoi_cond"))
    geom = build_velocity_contact_module(_cfg("imuhoi_cond_geom"))
    boundary = build_velocity_contact_module(_cfg("imuhoi_cond_boundary"))
    endpoint = build_velocity_contact_module(
        _cfg("imuhoi_cond_boundary_endpoint")
    )

    assert base.hand_contact_input_dim == 36
    assert base.obj_move_net.linear1.in_features == 27
    assert geom.hand_contact_input_dim == 36 + 16
    assert boundary.hand_contact_input_dim == 36 + 8
    assert endpoint.hand_contact_input_dim == boundary.hand_contact_input_dim
    assert not hasattr(geom, "boundary_rnn")
    assert not hasattr(boundary, "contact_endpoint_head")
    assert hasattr(endpoint, "contact_endpoint_head")
