import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import eval_batch_online
import model.rnn.imuhoi_model as imuhio_model_module
from model.rnn.object_trans_mergeHeads_obsEncoder import (
    ObjectTransModule as MergeHeadsObsObjectTransModule,
)


def _cfg():
    return SimpleNamespace(
        imu_dim=9,
        obj_imu_dim=9,
        num_human_imus=6,
        hidden_dim_multiplier=1,
        interaction_code_dim=128,
        object_trans_hidden_dim=192,
        object_trans_num_layers=2,
        object_trans_dropout=0.0,
        object_trans_feedback_hidden_dim=16,
        object_trans_state_feedback=False,
        object_trans_online_mode="window",
        obs_encoder_hidden_dim=16,
        obs_encoder_layers=1,
        obs_encoder_dropout=0.0,
        enable_ot_lb_prior=True,
    )


def test_mergeheads_obs_uses_fixed_compact_code_and_hidden_size():
    model = MergeHeadsObsObjectTransModule(_cfg())

    assert model.interaction_code_dim == 32
    assert model.obs_encoder.out_proj.out_features == 32
    assert model.object_trans_head.n_hidden == 320
    assert model.object_trans_head.linear1.in_features == 42
    assert model.code_residual_proj.weight.shape == (42, 32)
    assert torch.equal(model.code_residual_gate, torch.zeros(42))


def test_mergeheads_obs_code_residual_is_exact_noop_at_initialization():
    torch.manual_seed(31)
    model = MergeHeadsObsObjectTransModule(_cfg()).eval()
    shape = (2, 3)
    obj_imu = torch.randn(*shape, 9)
    obj_rot_delta = torch.randn(*shape, 3)
    obj_vel = torch.randn(*shape, 3)
    contact = torch.randn(*shape, 3)
    lhand_position = torch.randn(*shape, 3)
    lhand_imu = torch.randn(*shape, 9)
    rhand_position = torch.randn(*shape, 3)
    rhand_imu = torch.randn(*shape, 9)
    code = torch.randn(*shape, 32)

    fused = model._build_prediction_inputs(
        obj_imu,
        obj_rot_delta,
        obj_vel,
        contact,
        lhand_position,
        lhand_imu,
        rhand_position,
        rhand_imu,
        code,
    )
    raw = torch.cat(
        (
            obj_imu,
            obj_rot_delta,
            obj_vel,
            contact,
            lhand_position,
            lhand_imu,
            rhand_position,
            rhand_imu,
        ),
        dim=-1,
    )

    assert torch.allclose(fused, model.raw_input_norm(raw))


def test_mergeheads_obs_forward_uses_compact_code_path():
    torch.manual_seed(37)
    model = MergeHeadsObsObjectTransModule(_cfg()).eval()
    batch_size, seq_len = 1, 3
    obj_imu = torch.zeros(batch_size, seq_len, 9)
    obj_imu[..., 3:9] = torch.eye(3)[:, :2].reshape(6)
    output = model(
        torch.randn(batch_size, seq_len, 2, 3),
        torch.sigmoid(torch.randn(batch_size, seq_len, 3)),
        torch.zeros(batch_size, 3),
        obj_imu=obj_imu,
        human_imu=torch.randn(batch_size, seq_len, 6, 9),
        obj_vel_input=torch.randn(batch_size, seq_len, 3),
        human_pose_input=torch.randn(
            batch_size,
            seq_len,
            model.refine_pose_dim,
        ),
        root_trans_input=torch.randn(batch_size, seq_len, 3),
        enable_refine=False,
    )

    assert output["interaction_code"].shape == (batch_size, seq_len, 32)
    assert output["pred_obj_trans"].shape == (batch_size, seq_len, 3)
    assert output["gate_logits"].shape == (batch_size, seq_len, 4)


def test_mergeheads_obs_rejects_pre_change_checkpoint_version():
    model = MergeHeadsObsObjectTransModule(_cfg())
    state_dict = model.state_dict()
    state_dict["object_trans_arch_version"] = torch.tensor(4)

    with pytest.raises(ValueError, match="architecture v5"):
        model.validate_checkpoint_state_dict(state_dict)


def test_imuhoi_selects_mergeheads_obs_without_external_vq_prior(monkeypatch):
    monkeypatch.setattr(
        imuhio_model_module,
        "VelocityContactModule",
        lambda cfg: nn.Identity(),
    )
    monkeypatch.setattr(
        imuhio_model_module,
        "HumanPoseModule",
        lambda cfg, device, no_trans=False: nn.Identity(),
    )

    def _unexpected_prior(_):
        raise AssertionError("mergeHeads_obs must not instantiate the external VQ prior")

    monkeypatch.setattr(
        imuhio_model_module,
        "FrozenLBPriorPredictor",
        _unexpected_prior,
    )
    model = imuhio_model_module.IMUHOIModel(
        _cfg(),
        torch.device("cpu"),
        object_trans_arch="mergeHeads_obs",
    )

    assert model.object_trans_arch == "mergeheads_obs"
    assert isinstance(model.object_trans_module, MergeHeadsObsObjectTransModule)
    assert model.lb_prior_predictor is None


def test_batch_eval_parser_accepts_mergeheads_obs():
    args = eval_batch_online._make_parser().parse_args(
        ["--object_trans_arch", "mergeHeads_obs"]
    )

    assert args.object_trans_arch == "mergeHeads_obs"


def test_batch_eval_parser_accepts_legacy_velocity_contact_arch():
    args = eval_batch_online._make_parser().parse_args(
        ["--velocity_contact_arch", "legacy"]
    )

    assert args.velocity_contact_arch == "legacy"
