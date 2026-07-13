"""Focused regression tests for batched stateful ObjectTrans online evaluation."""
from types import SimpleNamespace

import torch
import torch.nn as nn

import eval_batch_online as batch_eval
from model.rnn.object_trans import ObjectTransModule


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
