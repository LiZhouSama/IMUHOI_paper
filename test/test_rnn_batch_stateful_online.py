"""Focused regression tests for batched stateful ObjectTrans online evaluation."""
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import eval_batch_online as batch_eval
from model.rnn.object_trans import ObjectTransModule


def test_batch_online_simulate_imu_noise_defaults_to_disabled():
    args = batch_eval._make_parser().parse_args([])

    assert args.simulate_imu_noise is False
    assert args.simulate_imu_noise_seed == 42
    assert args.eval_imu_noise is False


def test_batch_online_noise_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        batch_eval._make_parser().parse_args(["--simulate_imu_noise", "--eval_imu_noise"])


def _stateful_cfg():
    return SimpleNamespace(
        imu_dim=9,
        obj_imu_dim=9,
        num_human_imus=6,
        hidden_dim_multiplier=1,
        object_trans_hidden_dim=8,
        object_trans_num_layers=2,
        object_trans_dropout=0.0,
        object_trans_feedback_hidden_dim=6,
        object_trans_state_feedback=True,
        object_trans_online_mode="stateful",
    )


def test_ot_prediction_state_split_stack_preserves_reordered_lstm_batch():
    state = {
        "rnn_state": (torch.randn(2, 3, 5), torch.randn(2, 3, 5)),
        "prev_obj_trans": torch.randn(3, 3),
        "prev_prev_obj_trans": torch.randn(3, 3),
        "prev_obj_delta": torch.randn(3, 3),
        "prev_gating_weights": torch.randn(3, 4),
        "prev_obj_rotm": torch.randn(3, 3, 3),
        "frames_seen": 17,
    }

    rows = batch_eval._split_ot_prediction_state(state, batch_size=3)
    repacked = batch_eval._stack_ot_prediction_states([rows[2], rows[0]])

    assert torch.equal(repacked["rnn_state"][0], state["rnn_state"][0][:, [2, 0]])
    assert torch.equal(repacked["rnn_state"][1], state["rnn_state"][1][:, [2, 0]])
    assert torch.equal(repacked["prev_obj_trans"], state["prev_obj_trans"][[2, 0]])
    assert repacked["frames_seen"] == 17


class _DummyHumanPose(nn.Module):
    def forward(self, inputs):
        human_imu = inputs["human_imu"]
        base = human_imu[:, :, 0, :3]
        palm = torch.stack((base, base + 0.2), dim=2)
        return {
            "p_pred": torch.cat((base, base), dim=-1),
            "v_pred": base,
            "root_trans_pred": base * 0.1,
            "pred_hand_glb_pos": palm,
            "pred_palm_glb_pos": palm,
        }


class _DummyVelocityContact(nn.Module):
    def forward(self, inputs, hp_out=None):
        base = inputs["human_imu"][:, :, 1, :3]
        contact = torch.sigmoid(torch.cat((base[..., :2], base[..., :1]), dim=-1))
        return {
            "pred_hand_contact_prob": contact,
            "pred_obj_vel": base * 0.05,
            "pred_hand_glb_vel": torch.cat((base, base), dim=-1),
        }


class _StatefulPipeline:
    """Minimal pipeline exposing the methods used by the batched evaluator."""

    def __init__(self):
        self.human_pose_module = _DummyHumanPose()
        self.velocity_contact_module = _DummyVelocityContact()
        self.object_trans_module = ObjectTransModule(_stateful_cfg()).eval()
        torch.nn.init.normal_(self.object_trans_module.feedback_embed[-1].weight, std=0.03)

    def _build_hp_input_dict(self, data):
        return {
            "human_imu": data["human_imu"],
            "v_init": data["v_init"],
            "p_init": data["p_init"],
        }

    @staticmethod
    def _resolve_hoi_hand_positions(hp_out):
        return hp_out["pred_palm_glb_pos"]

    @staticmethod
    def _promote_refined_human_outputs(*_):
        return None

    @staticmethod
    def _compute_fk_output(*_):
        return None

    def forward(self, data, use_object_data=True, compute_fk=False, refine_human=False):
        hp_out = self.human_pose_module.forward(self._build_hp_input_dict(data))
        vc_out = self.velocity_contact_module.forward({**data, "hp_out": hp_out}, hp_out=hp_out)
        output = {**hp_out, **vc_out}
        if use_object_data:
            output.update(
                self.object_trans_module(
                    self._resolve_hoi_hand_positions(hp_out),
                    vc_out["pred_hand_contact_prob"],
                    data["obj_trans_init"],
                    obj_imu=data["obj_imu"],
                    human_imu=data["human_imu"],
                    obj_vel_input=vc_out["pred_obj_vel"],
                    contact_init=data.get("contact_init"),
                    has_object_mask=data.get("has_object"),
                    enable_refine=False,
                    compute_refine=False,
                )
            )
        output["has_object"] = data.get("has_object")
        return output


class _RecordingObjectTrans(ObjectTransModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.received_priors = []

    def forward(self, *args, **kwargs):
        self.received_priors.append(kwargs.get("lb_prior_quantiles"))
        return super().forward(*args, **kwargs)


class _PriorPipeline(_StatefulPipeline):
    def __init__(self, online_mode="window"):
        self.human_pose_module = _DummyHumanPose()
        self.velocity_contact_module = _DummyVelocityContact()
        cfg = _stateful_cfg()
        cfg.object_trans_online_mode = online_mode
        self.object_trans_module = _RecordingObjectTrans(cfg).eval()
        self.prior_calls = 0

    def _predict_lb_prior(self, hp_out, vc_out, obj_imu):
        self.prior_calls += 1
        batch_size = obj_imu.shape[0]
        return torch.full(
            (batch_size, 2, 3),
            float(self.prior_calls),
            device=obj_imu.device,
            dtype=obj_imu.dtype,
        )

    def forward(self, data, use_object_data=True, compute_fk=False, refine_human=False):
        hp_out = self.human_pose_module.forward(self._build_hp_input_dict(data))
        vc_out = self.velocity_contact_module.forward({**data, "hp_out": hp_out}, hp_out=hp_out)
        output = {**hp_out, **vc_out}
        if use_object_data:
            lb_prior_quantiles = self._predict_lb_prior(hp_out, vc_out, data["obj_imu"])
            output.update(
                self.object_trans_module(
                    self._resolve_hoi_hand_positions(hp_out),
                    vc_out["pred_hand_contact_prob"],
                    data["obj_trans_init"],
                    obj_imu=data["obj_imu"],
                    human_imu=data["human_imu"],
                    obj_vel_input=vc_out["pred_obj_vel"],
                    contact_init=data.get("contact_init"),
                    has_object_mask=data.get("has_object"),
                    lb_prior_quantiles=lb_prior_quantiles,
                    enable_refine=False,
                    compute_refine=False,
                )
            )
        output["has_object"] = data.get("has_object")
        return output


def _sample(seq_len):
    obj_imu = torch.randn(seq_len, 9)
    obj_imu[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return {
        "human_imu": torch.randn(seq_len, 6, 9),
        "obj_imu": obj_imu,
        "v_init": torch.zeros(1, 3),
        "p_init": torch.zeros(1, 6),
        "trans_init": torch.zeros(3),
        "hand_vel_glb_init": torch.zeros(2, 3),
        "obj_vel_init": torch.zeros(3),
        "obj_trans_init": torch.tensor([0.1, -0.2, 0.3]),
        "contact_init": torch.zeros(3),
        "has_object": True,
    }


def test_batched_stateful_ot_matches_full_causal_scan_for_reordered_active_batch(monkeypatch):
    torch.manual_seed(13)
    model = _StatefulPipeline()
    samples = [_sample(seq_len) for seq_len in (7, 6, 5)]
    monkeypatch.setattr(
        batch_eval,
        "build_model_input_dict",
        lambda raw, config, device, add_noise=False: raw,
    )

    outputs = batch_eval._run_batched_online_group(
        model,
        config=None,
        device=torch.device("cpu"),
        samples=samples,
        online_window=3,
        use_object_data=True,
        compute_fk=False,
        refine_human=False,
        desc="stateful-test",
    )

    for sample, output in zip(samples, outputs):
        seq_len = sample["human_imu"].shape[0]
        full_data = batch_eval._collate_sample_slices([sample], [0], [seq_len])
        full_output = model.forward(full_data, use_object_data=True, refine_human=False)
        assert output["pred_obj_trans"].shape[1] == seq_len
        assert torch.allclose(
            output["pred_obj_trans"],
            full_output["pred_obj_trans"],
            atol=1e-5,
            rtol=1e-5,
        )


def test_batched_window_ot_passes_prior_to_every_reset_window(monkeypatch):
    torch.manual_seed(19)
    model = _PriorPipeline(online_mode="window")
    monkeypatch.setattr(
        batch_eval,
        "build_model_input_dict",
        lambda raw, config, device, add_noise=False: raw,
    )

    batch_eval._run_batched_online_group(
        model,
        config=None,
        device=torch.device("cpu"),
        samples=[_sample(seq_len=5)],
        online_window=3,
        use_object_data=True,
        compute_fk=False,
        refine_human=False,
        desc="window-prior-test",
    )

    received = model.object_trans_module.received_priors
    assert len(received) == 3
    assert all(isinstance(prior, torch.Tensor) for prior in received)
    assert [float(prior[0, 0, 0]) for prior in received] == [1.0, 2.0, 3.0]


def test_stateful_warmup_replay_passes_prior_to_initial_state():
    torch.manual_seed(23)
    model = _PriorPipeline(online_mode="stateful")
    warmup_data = batch_eval._collate_sample_slices([_sample(seq_len=3)], [0], [3])

    states = batch_eval._initialize_stateful_ot_states(model, warmup_data, batch_size=1)

    assert len(states) == 1
    assert len(model.object_trans_module.received_priors) == 1
    prior = model.object_trans_module.received_priors[0]
    assert isinstance(prior, torch.Tensor)
    assert torch.equal(prior, torch.ones(1, 2, 3))
