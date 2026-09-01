from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import eval_batch_online
import eval_IMUHOI
import model as model_entry
import vis_IMUHOI
import model.rnn.imuhoi_model as imuhio_model_module
import train.rnn.train_object_trans as train_object_trans
from model.rnn import build_object_trans_module
from model.rnn.object_trans_v2 import V2ObjectTransModule
from model.rnn.object_trans import ObjectTransModule
from train.rnn.loss.object_trans_loss import ObjectTransLoss
from utils.object_fusion import build_contact_fusion_weights


def _cfg():
    return SimpleNamespace(
        imu_dim=9,
        obj_imu_dim=9,
        num_human_imus=6,
        hidden_dim_multiplier=1,
        object_trans_hidden_dim=16,
        object_trans_num_layers=2,
        object_trans_dropout=0.0,
        object_trans_state_feedback=False,
        enable_ot_refine=False,
        train={"window": 3},
        test={"window": 3},
    )


def _inputs(seq_len=4):
    obj_imu = torch.zeros(1, seq_len, 9)
    obj_imu[..., 3:9] = torch.eye(3)[:, :2].reshape(6)
    return {
        "hand_positions": torch.zeros(1, seq_len, 2, 3),
        "contact": torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.8, 0.0, 0.8],
                    [0.4, 0.2, 0.6],
                    [0.0, 0.0, 0.0],
                    [0.2, 0.1, 0.3],
                ]
            ]
        )[:, :seq_len],
        "obj_trans_init": torch.zeros(1, 3),
        "obj_imu": obj_imu,
        "human_imu": torch.zeros(1, seq_len, 6, 9),
        "obj_vel": torch.zeros(1, seq_len, 3),
    }


def test_v2_uses_the_object_trans_loss_mapping_without_gate_parameters():
    model = V2ObjectTransModule(_cfg())
    contact = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.8, 0.0, 0.8], [0.4, 0.2, 0.6]]]
    )

    expected = ObjectTransLoss._build_gate_weak_target(contact)
    actual = build_contact_fusion_weights(contact)

    assert torch.allclose(actual, expected)
    assert model.object_trans_head.linear2.out_features == 8
    assert not any("gate" in name.lower() for name, _ in model.named_parameters())
    assert int(model.object_trans_arch_version.item()) == 2


def test_v2_forward_and_online_outputs_keep_four_weight_compatibility():
    torch.manual_seed(3)
    model = V2ObjectTransModule(_cfg()).eval()
    data = _inputs()

    with torch.no_grad():
        output = model(
            data["hand_positions"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            enable_refine=False,
        )
        online = model.inference(
            data["hand_positions"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            enable_refine=False,
            inference_mode="online",
            online_window=3,
        )

    assert output["gating_weights"].shape == (1, 4, 4)
    assert torch.allclose(output["gating_weights"].sum(dim=-1), torch.ones(1, 4))
    assert output["gating_weights"].shape == output["gating_weights_raw"].shape
    assert "gate_logits" not in output
    assert output["gating_is_fixed"] is True
    assert online["pred_obj_trans"].shape == (1, 4, 3)


def test_v2_no_feedback_vectorizes_the_window_rnn(monkeypatch):
    model = V2ObjectTransModule(_cfg()).eval()
    data = _inputs()
    forward_calls = 0
    original_forward = model.object_trans_head.forward_with_state

    def track_forward(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*args, **kwargs)

    def reject_step(*args, **kwargs):
        raise AssertionError("no-feedback fast path must not call the per-frame RNN step")

    monkeypatch.setattr(model.object_trans_head, "forward_with_state", track_forward)
    monkeypatch.setattr(model.object_trans_head, "step", reject_step)

    with torch.no_grad():
        output = model(
            data["hand_positions"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            enable_refine=False,
        )

    assert forward_calls == 1
    assert output["pred_obj_trans"].shape == (1, 4, 3)


def test_v2_preserves_feedback_and_published_prefix_replay():
    torch.manual_seed(13)
    cfg = _cfg()
    cfg.object_trans_state_feedback = True
    cfg.object_trans_feedback_hidden_dim = 12
    model = V2ObjectTransModule(cfg).eval()
    data = _inputs(seq_len=5)
    known_weights = build_contact_fusion_weights(data["contact"][:, :3])
    known_positions = torch.randn(1, 3, 3)

    with torch.no_grad():
        output = model(
            data["hand_positions"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            known_obj_trans_prefix=known_positions,
            known_gating_weights_prefix=known_weights,
            enable_refine=False,
        )

    assert torch.allclose(output["pred_obj_trans"][:, :3], known_positions)
    assert torch.allclose(output["gating_weights"][:, :3], known_weights)
    assert output["pred_obj_trans"].shape == (1, 5, 3)


def test_v2_is_selected_by_rnn_builders_and_combined_model(monkeypatch):
    cfg = _cfg()
    cfg.velocity_contact_arch = "current"

    assert isinstance(build_object_trans_module(cfg, "v2"), V2ObjectTransModule)
    assert isinstance(
        model_entry.ObjectTransModule(cfg, object_trans_arch="v2"),
        V2ObjectTransModule,
    )

    monkeypatch.setattr(
        imuhio_model_module,
        "HumanPoseModule",
        lambda cfg, device, no_trans=False: nn.Identity(),
    )
    monkeypatch.setattr(
        imuhio_model_module,
        "VelocityContactModule",
        lambda cfg: nn.Identity(),
    )
    combined = imuhio_model_module.IMUHOIModel(
        cfg,
        torch.device("cpu"),
        object_trans_arch="v2",
    )
    assert isinstance(combined.object_trans_module, V2ObjectTransModule)


def test_v2_checkpoint_marker_and_shape_are_isolated_from_current():
    cfg = _cfg()
    current = ObjectTransModule(cfg)
    v2 = V2ObjectTransModule(cfg)

    current.validate_checkpoint_state_dict(current.state_dict())
    v2.validate_checkpoint_state_dict(v2.state_dict())
    with pytest.raises(ValueError):
        v2.validate_checkpoint_state_dict(current.state_dict())
    with pytest.raises(ValueError):
        current.validate_checkpoint_state_dict(v2.state_dict())


def test_object_trans_entrypoint_parsers_accept_v2():
    batch_args = eval_batch_online._make_parser().parse_args(
        ["--object_trans_arch", "v2"]
    )
    vis_args = vis_IMUHOI._make_parser().parse_args(
        ["--object_trans_arch", "v2"]
    )

    assert batch_args.object_trans_arch == "v2"
    assert vis_args.object_trans_arch == "v2"
    assert "v2" in eval_IMUHOI.OBJECT_TRANS_ARCH_CHOICES
    assert eval_batch_online._make_parser().parse_args([]).object_trans_arch is None
    assert "v2" in train_object_trans.OBJECT_TRANS_ARCH_CHOICES


def test_root_loader_forwards_object_trans_arch(monkeypatch):
    calls = {}

    class FakeRnnModule:
        @staticmethod
        def load_model(
            config,
            device,
            no_trans=False,
            module_paths=None,
            velocity_contact_arch=None,
            object_trans_arch=None,
        ):
            calls.update(
                object_trans_arch=object_trans_arch,
                velocity_contact_arch=velocity_contact_arch,
            )
            return "loaded"

    monkeypatch.setattr(model_entry, "_arch_module", lambda arch: FakeRnnModule)
    result = model_entry.load_model(
        SimpleNamespace(model_arch="rnn"),
        "cpu",
        object_trans_arch="v2",
        velocity_contact_arch="v3",
    )

    assert result == "loaded"
    assert calls == {
        "object_trans_arch": "v2",
        "velocity_contact_arch": "v3",
    }


def test_v2_disables_trainable_gate_weak_loss():
    model = V2ObjectTransModule(_cfg()).eval()
    data = _inputs()
    with torch.no_grad():
        output = model(
            data["hand_positions"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            enable_refine=False,
        )

    batch = {
        "human_imu": data["human_imu"],
        "obj_trans": torch.zeros(1, 4, 3),
        "obj_vel": torch.zeros(1, 4, 3),
        "has_object": torch.ones(1, dtype=torch.bool),
        "lhand_contact": data["contact"][..., 0].bool(),
        "rhand_contact": data["contact"][..., 1].bool(),
        "obj_contact": data["contact"][..., 2].bool(),
    }
    loss_fn = ObjectTransLoss(weights={"gate_weak": 1.0})
    _, raw_losses, _ = loss_fn(output, batch, torch.device("cpu"))

    assert raw_losses["gate_weak"].item() == 0.0