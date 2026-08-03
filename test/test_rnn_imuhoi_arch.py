from types import SimpleNamespace

import eval_batch_online
import pytest
import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_rotation_6d

from model.rnn.object_trans_IMUHOI import IMUHOIObjectTransModule
from model.rnn.object_trans_IMUHOI_4heads import (
    IMUHOI4HeadsObjectTransModule,
)
from model.rnn.velocity_contact import (
    build_velocity_contact_module,
    normalize_velocity_contact_arch,
)
from model.rnn.velocity_contact_IMUHOI import (
    IMUHOIVelocityContactModule,
)
from train.rnn.loss.imuhoi_loss import (
    IMUHOIObjectTransLoss,
    IMUHOIVelocityContactLoss,
)
from train.rnn.train_object_trans import (
    Stage3JointTrainer,
    _joint_trainable_modules,
)


def _cfg():
    return SimpleNamespace(
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        velocity_hidden_dim=128,
        velocity_num_layers=2,
        velocity_dropout=0.2,
        hidden_dim_multiplier=1,
        gating_prior_beta=5.0,
        gating_temperature=5.0,
        gating_smoothing_enabled=False,
        contact_from_segments=False,
        online_window=3,
        test=SimpleNamespace(window=3),
    )


def _imu(batch_size, seq_len):
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
    return human, obj


def test_imuhoi_vc_matches_original_parameterization_and_ignores_hp():
    cfg = _cfg()
    cfg.velocity_contact_arch = "IMUHOI"
    assert normalize_velocity_contact_arch("IMUHOI") == "imuhoi"
    model = build_velocity_contact_module(cfg)
    assert isinstance(model, IMUHOIVelocityContactModule)
    assert sum(parameter.numel() for parameter in model.parameters()) == 917260

    human, obj = _imu(2, 5)
    data = {
        "human_imu": human,
        "obj_imu": obj,
        "hand_vel_glb_init": torch.zeros(2, 2, 3),
        "obj_vel_init": torch.zeros(2, 3),
        "contact_init": torch.zeros(2, 3),
    }
    model.eval()
    with torch.no_grad():
        first = model(data, hp_out={"unused": torch.randn(2, 5, 3)})
        second = model(data, hp_out={"unused": torch.randn(2, 5, 99)})
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    assert first["pred_hand_contact_prob"].shape == (2, 5, 3)


def _run_ot_offline_and_online(model):
    human, obj = _imu(2, 6)
    hands = torch.randn(2, 6, 2, 3)
    contact = torch.sigmoid(torch.randn(2, 6, 3))
    kwargs = {
        "obj_imu": obj,
        "human_imu": human,
        "obj_vel_input": torch.randn(2, 6, 3),
        "contact_init": torch.zeros(2, 3),
        "has_object_mask": torch.ones(2, dtype=torch.bool),
    }
    with torch.no_grad():
        offline = model(hands, contact, torch.zeros(2, 3), **kwargs)
        online = model.inference(
            hands,
            contact,
            torch.zeros(2, 3),
            inference_mode="online",
            online_window=3,
            **kwargs,
        )
    return offline, online


def test_imuhoi_ot_restores_original_three_branch_contract():
    cfg = _cfg()
    model = IMUHOIObjectTransModule(cfg).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 915915
    assert not hasattr(model, "refine_head")
    assert model.gating_head.linear2.out_features == 3

    offline, online = _run_ot_offline_and_online(model)
    assert offline["gating_weights"].shape == (2, 6, 3)
    assert "pred_static_obj_trans" not in offline
    assert online["pred_obj_trans"].shape == (2, 6, 3)
    torch.testing.assert_close(
        offline["gating_weights"].sum(dim=-1),
        torch.ones(2, 6),
    )


def test_imuhoi_4heads_keeps_static_branch_and_online_contract():
    cfg = _cfg()
    model = IMUHOI4HeadsObjectTransModule(cfg).eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 915980
    assert not hasattr(model, "refine_head")
    assert model.gating_head.linear2.out_features == 4

    offline, online = _run_ot_offline_and_online(model)
    assert offline["gating_weights"].shape == (2, 6, 4)
    assert offline["pred_static_obj_trans"].shape == (2, 6, 3)
    assert online["pred_obj_trans"].shape == (2, 6, 3)
    torch.testing.assert_close(
        offline["pred_static_obj_trans"][:, 1:],
        offline["pred_obj_trans"][:, :-1],
    )
    torch.testing.assert_close(
        online["pred_static_obj_trans"][:, 1:],
        online["pred_obj_trans"][:, :-1],
    )
    torch.testing.assert_close(
        offline["gating_weights"].sum(dim=-1),
        torch.ones(2, 6),
    )


def test_imuhoi_checkpoint_validation_rejects_other_topologies():
    cfg = _cfg()
    vc = IMUHOIVelocityContactModule(cfg)
    ot = IMUHOIObjectTransModule(cfg)
    ot_4heads = IMUHOI4HeadsObjectTransModule(cfg)

    vc.validate_checkpoint_state_dict(vc.state_dict())
    ot.validate_checkpoint_state_dict(ot.state_dict())
    ot_4heads.validate_checkpoint_state_dict(ot_4heads.state_dict())

    with pytest.raises(ValueError, match="incompatible"):
        vc.validate_checkpoint_state_dict({"not_a_vc.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="incompatible"):
        ot.validate_checkpoint_state_dict({"refine_head.weight": torch.ones(1)})
    with pytest.raises(ValueError, match="incompatible"):
        ot.validate_checkpoint_state_dict(ot_4heads.state_dict())
    with pytest.raises(ValueError, match="incompatible"):
        ot_4heads.validate_checkpoint_state_dict(ot.state_dict())


def test_batch_online_parser_accepts_case_insensitive_imuhoi_arches():
    args = eval_batch_online._make_parser().parse_args(
        [
            "--velocity_contact_arch",
            "IMUHOI",
            "--object_trans_arch",
            "IMUHOI",
        ]
    )
    assert args.velocity_contact_arch == "imuhoi"
    assert args.object_trans_arch == "imuhoi"

    four_head_args = eval_batch_online._make_parser().parse_args(
        ["--object_trans_arch", "IMUHOI_4heads"]
    )
    assert four_head_args.object_trans_arch == "imuhoi_4heads"


def test_imuhoi_losses_and_joint_train_module_selection():
    assert _joint_trainable_modules("IMUHOI", True) == (True, False)
    assert _joint_trainable_modules("IMUHOI_4heads", True) == (True, False)
    assert _joint_trainable_modules("legacy", True) == (False, True)
    assert _joint_trainable_modules("IMUHOI", False) == (False, False)
    assert IMUHOIVelocityContactLoss.LOSS_KEYS == {
        "obj_vel",
        "hand_vel",
        "hand_contact",
    }
    assert "refine_pose" not in IMUHOIObjectTransLoss.LOSS_KEYS
    assert "gate_weak" in IMUHOIObjectTransLoss.LOSS_KEYS
    assert IMUHOIObjectTransLoss.DEFAULT_WEIGHTS["gate_weak"] == 0.0
    target = IMUHOIObjectTransLoss._build_gate_weak_target(
        torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    )
    torch.testing.assert_close(target.sum(dim=-1), torch.ones(1, 2))
    assert target[0, 0].argmax().item() == 3
    assert target[0, 1].argmax().item() == 2


def test_imuhoi_gate_weak_is_computed_but_disabled_by_default():
    cfg = _cfg()
    model = IMUHOI4HeadsObjectTransModule(cfg).eval()
    batch_size, seq_len = 1, 4
    human, obj = _imu(batch_size, seq_len)
    contacts = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    output = model(
        torch.randn(batch_size, seq_len, 2, 3),
        torch.stack((contacts, contacts, contacts), dim=-1),
        torch.zeros(batch_size, 3),
        obj_imu=obj,
        human_imu=human,
        obj_vel_input=torch.randn(batch_size, seq_len, 3),
        contact_init=torch.zeros(batch_size, 3),
        has_object_mask=torch.ones(batch_size, dtype=torch.bool),
    )
    batch = {
        "human_imu": human,
        "obj_imu": obj,
        "obj_trans": torch.randn(batch_size, seq_len, 3),
        "obj_vel": torch.randn(batch_size, seq_len, 3),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
        "lhand_contact": contacts.bool(),
        "rhand_contact": contacts.bool(),
        "obj_contact": contacts.bool(),
    }
    loss_fn = IMUHOIObjectTransLoss(
        weights=IMUHOIObjectTransLoss.DEFAULT_WEIGHTS
    )
    total, raw, weighted = loss_fn(output, batch, torch.device("cpu"))
    assert raw["gate_weak"].item() > 0.0
    assert weighted["gate_weak"].item() == 0.0
    total.backward()
    gate_grad = model.gating_head.linear2.weight.grad
    assert gate_grad is not None and gate_grad.abs().sum().item() > 0.0


def test_imuhoi_joint_trainer_unfreezes_hp_and_ot_but_not_vc():
    class TinyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)

    cfg = SimpleNamespace(
        device="cpu",
        object_trans_arch="imuhoi",
        use_multi_gpu=False,
        lr=1e-3,
        weight_decay=0.0,
        milestones=[],
        gamma=0.1,
        use_tensorboard=False,
        debug=True,
        no_trans=False,
        loss_weights={},
        imuhoi_loss_weights=IMUHOIObjectTransLoss.DEFAULT_WEIGHTS,
    )
    trainer = Stage3JointTrainer(
        cfg,
        vc_model=TinyModule(),
        hp_model=TinyModule(),
        ot_model=TinyModule(),
        train_loader=[],
        test_loader=None,
        joint_train=True,
    )
    assert all(parameter.requires_grad for parameter in trainer.hp_model.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in trainer.vc_model.parameters()
    )
    assert all(parameter.requires_grad for parameter in trainer.ot_model.parameters())
    assert {group["name"] for group in trainer.optimizer.param_groups} == {
        "human_pose",
        "object_trans",
    }
    assert trainer.hp_loss_fn.weights["contact"] == 0.0


def test_imuhoi_joint_forward_uses_predictions_without_scheduled_gt():
    class TinyHP(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(2.0))

        def forward(self, data):
            batch_size, seq_len = data["human_imu"].shape[:2]
            hand = self.offset.expand(batch_size, seq_len, 2, 3)
            return {
                "pred_hand_glb_pos": hand,
                "p_pred": self.offset.expand(batch_size, seq_len, 1),
                "root_trans_pred": self.offset.expand(
                    batch_size, seq_len, 3
                ),
            }

    class TinyVC(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, data, hp_out=None):
            batch_size, seq_len = data["human_imu"].shape[:2]
            return {
                "pred_hand_contact_prob": self.scale.expand(
                    batch_size, seq_len, 3
                ),
                "pred_obj_vel": self.scale.expand(
                    batch_size, seq_len, 3
                ),
            }

    class TinyOT(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.last_hands = None

        def forward(self, hands, contact, obj_trans_init, **kwargs):
            del contact, obj_trans_init, kwargs
            self.last_hands = hands
            return {"pred_obj_trans": hands[:, :, 0] * self.scale}

    cfg = SimpleNamespace(
        device="cpu",
        object_trans_arch="imuhoi",
        use_multi_gpu=False,
        lr=1e-3,
        weight_decay=0.0,
        milestones=[],
        gamma=0.1,
        use_tensorboard=False,
        debug=True,
        no_trans=False,
        loss_weights={},
        imuhoi_loss_weights=IMUHOIObjectTransLoss.DEFAULT_WEIGHTS,
    )
    trainer = Stage3JointTrainer(
        cfg,
        vc_model=TinyVC(),
        hp_model=TinyHP(),
        ot_model=TinyOT(),
        train_loader=[],
        test_loader=None,
        joint_train=True,
    )
    data = {
        "human_imu": torch.zeros(1, 3, 6, 9),
        "obj_imu": torch.zeros(1, 3, 9),
        "obj_trans_init": torch.zeros(1, 3),
        "contact_init": torch.zeros(1, 3),
        "has_object": torch.ones(1, dtype=torch.bool),
    }
    _, _, hp_out, _ = trainer.forward_all(
        data,
        # Deliberately lacks every GT scheduled-sampling field.
        batch={},
        epoch=0,
        training=True,
    )
    assert trainer.ot_model.last_hands is hp_out["pred_hand_glb_pos"]
    assert trainer.ot_model.last_hands.requires_grad
