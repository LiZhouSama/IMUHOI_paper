from types import SimpleNamespace
import os
import sys
import types

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pytorch3d.transforms  # noqa: F401
except Exception:
    pytorch3d_mod = types.ModuleType("pytorch3d")
    transforms_mod = types.ModuleType("pytorch3d.transforms")

    def _rotation_6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(3, device=x.device, dtype=x.dtype)
        shape = (*x.shape[:-1], 3, 3)
        return eye.view(*([1] * (len(shape) - 2)), 3, 3).expand(shape).clone()

    def _matrix_to_rotation_6d(m: torch.Tensor) -> torch.Tensor:
        return m[..., :2].reshape(*m.shape[:-2], 6)

    transforms_mod.rotation_6d_to_matrix = _rotation_6d_to_matrix
    transforms_mod.matrix_to_rotation_6d = _matrix_to_rotation_6d
    transforms_mod.matrix_to_axis_angle = lambda m: torch.zeros(*m.shape[:-2], 3, device=m.device, dtype=m.dtype)
    pytorch3d_mod.transforms = transforms_mod
    sys.modules["pytorch3d"] = pytorch3d_mod
    sys.modules["pytorch3d.transforms"] = transforms_mod

from configs import _REDUCED_POSE_NAMES
from model.rnn.imuhoi_model import IMUHOIModel
from model.rnn.object_trans import ObjectTransModule
from model.rnn.velocity_contact import VelocityContactModule
from train.rnn.loss.object_trans_loss import ObjectTransLoss
from train.rnn.loss.velocity_contact_loss import VelocityContactLoss
from train.rnn.scheduled_inputs import prediction_mix_probability, sample_mix_tensor
from train.rnn.train_object_trans import Stage3JointTrainer, _resolve_pretrained_ckpt
from train.rnn.train_velocity_contact import _resolve_hp_ckpt
from utils.utils import build_model_input_dict


def test_prediction_mix_probability_and_sample_mask_shape():
    cfg = SimpleNamespace(input_mix_start_epoch=0, input_mix_end_epoch=100)
    assert prediction_mix_probability(0, cfg) == 0.0
    assert prediction_mix_probability(100, cfg) == 1.0

    gt = torch.zeros(3, 4, 2, 3)
    pred = torch.ones_like(gt)
    mixed, mask = sample_mix_tensor(gt, pred, 0.0, return_mask=True)
    assert torch.equal(mixed, gt)
    assert mask.shape == (3, 1, 1, 1)

    mixed = sample_mix_tensor(gt, pred, 1.0)
    assert torch.equal(mixed, pred)

    other = sample_mix_tensor(torch.zeros(3, 4, 3), torch.ones(3, 4, 3), 0.5, mask=mask)
    assert other.shape == (3, 4, 3)


def test_object_trans_refinement_outputs_identity_at_init():
    cfg = SimpleNamespace(imu_dim=9, num_human_imus=6, hidden_dim_multiplier=1)
    model = ObjectTransModule(cfg)
    model.train()

    batch_size, seq_len = 2, 4
    hand_pos = torch.randn(batch_size, seq_len, 2, 3)
    contact = torch.sigmoid(torch.randn(batch_size, seq_len, 3))
    obj_trans_init = torch.randn(batch_size, 3)
    obj_imu = torch.randn(batch_size, seq_len, 9)
    human_imu = torch.randn(batch_size, seq_len, 6, 9)
    obj_vel = torch.randn(batch_size, seq_len, 3)
    pose = torch.randn(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6)
    root = torch.randn(batch_size, seq_len, 3)

    out = model(
        hand_pos,
        contact,
        obj_trans_init,
        obj_imu=obj_imu,
        human_imu=human_imu,
        obj_vel_input=obj_vel,
        human_pose_input=pose,
        root_trans_input=root,
        enable_refine=True,
    )

    assert out["pose_delta"].shape == pose.shape
    assert out["root_trans_delta"].shape == root.shape
    assert torch.allclose(out["pose_delta"], torch.zeros_like(pose))
    assert torch.allclose(out["root_trans_delta"], torch.zeros_like(root))
    assert torch.allclose(out["refined_pose"], pose)
    assert torch.allclose(out["refined_root_trans"], root)

    disabled = model(
        hand_pos,
        contact,
        obj_trans_init,
        obj_imu=obj_imu,
        human_imu=human_imu,
        obj_vel_input=obj_vel,
        human_pose_input=pose,
        root_trans_input=root,
        enable_refine=False,
    )
    assert "refined_pose" not in disabled
    assert "refined_root_trans" not in disabled


def test_imuhoi_model_refine_human_defaults_to_config_flag():
    model = object.__new__(IMUHOIModel)
    model.cfg = SimpleNamespace(enable_ot_refine=False)
    assert model._resolve_refine_human(None) is False
    assert model._resolve_refine_human(True) is True

    model.cfg = SimpleNamespace(enable_ot_refine=True)
    assert model._resolve_refine_human(None) is True
    assert model._resolve_refine_human(False) is False

    model.cfg = SimpleNamespace()
    assert model._resolve_refine_human(None) is False


def test_object_trans_loss_includes_refinement_terms():
    batch_size, seq_len = 2, 3
    num_reduced = len(_REDUCED_POSE_NAMES)
    eye = torch.eye(3).view(1, 1, 1, 3, 3).expand(batch_size, seq_len, num_reduced, 3, 3)
    batch = {
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
        "obj_trans": torch.zeros(batch_size, seq_len, 3),
        "trans": torch.ones(batch_size, seq_len, 3),
        "ori_root_reduced": eye,
    }
    pred = {
        "pred_obj_trans": torch.zeros(batch_size, seq_len, 3),
        "refined_pose": torch.zeros(batch_size, seq_len, num_reduced * 6),
        "refined_root_trans": torch.zeros(batch_size, seq_len, 3),
    }

    loss_fn = ObjectTransLoss(weights={"obj_trans": 0.0})
    total, losses, weighted = loss_fn(pred, batch, torch.device("cpu"))

    assert losses["refine_pose"].item() > 0
    assert losses["refine_root_trans"].item() > 0
    assert weighted["refine_pose"].item() > 0
    assert total.item() > 0


def test_object_trans_uses_one_shared_predictor_and_returns_raw_outputs():
    cfg = SimpleNamespace(imu_dim=9, obj_imu_dim=9, num_human_imus=6, hidden_dim_multiplier=1)
    model = ObjectTransModule(cfg).eval()
    batch_size, seq_len = 2, 3
    obj_imu = torch.zeros(batch_size, seq_len, 9)
    obj_imu[..., 3:9] = torch.eye(3)[:, :2].reshape(6)

    with torch.no_grad():
        out = model(
            torch.randn(batch_size, seq_len, 2, 3),
            torch.sigmoid(torch.randn(batch_size, seq_len, 3)),
            torch.randn(batch_size, 3),
            obj_imu=obj_imu,
            human_imu=torch.randn(batch_size, seq_len, 6, 9),
            obj_vel_input=torch.randn(batch_size, seq_len, 3),
        )

    assert hasattr(model, "object_trans_head")
    assert not hasattr(model, "lhand_fk_head")
    assert not hasattr(model, "rhand_fk_head")
    assert not hasattr(model, "gating_head")
    assert not hasattr(model, "mesh_prior_encoder")
    assert not hasattr(model, "obs_encoder")
    assert out["lhand_fk_out"].shape == (batch_size, seq_len, 4)
    assert out["rhand_fk_out"].shape == (batch_size, seq_len, 4)
    assert out["gate_logits"].shape == (batch_size, seq_len, 4)
    assert torch.allclose(torch.softmax(out["gate_logits"], dim=-1), out["gating_weights"])


def test_object_trans_rejects_legacy_checkpoint_architecture():
    model = ObjectTransModule(SimpleNamespace(imu_dim=9, obj_imu_dim=9, num_human_imus=6))
    model.validate_checkpoint_state_dict(model.state_dict())

    try:
        model.validate_checkpoint_state_dict({"lhand_fk_head.linear1.weight": torch.zeros(1)})
    except ValueError as exc:
        assert "removed mesh/obs or three-head architecture" in str(exc)
    else:
        raise AssertionError("legacy ObjectTrans checkpoint must be rejected")


def test_object_trans_loss_uses_gate_logits_when_available():
    batch_size, seq_len = 1, 2
    batch = {
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
        "obj_trans": torch.zeros(batch_size, seq_len, 3),
    }
    gate_logits = torch.tensor([[[4.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]])
    pred = {
        "pred_obj_trans": torch.zeros(batch_size, seq_len, 3),
        "gate_logits": gate_logits,
        "gating_weights": torch.tensor([[[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]),
        "gating_contact_prob": torch.tensor([[[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]),
    }

    loss_fn = ObjectTransLoss(
        weights={"obj_trans": 0.0, "gate_weak": 1.0, "gate_weak_target_smoothing": 0.0}
    )
    _, losses, _ = loss_fn(pred, batch, torch.device("cpu"))
    expected = -torch.log_softmax(gate_logits, dim=-1)[..., 0].mean()
    assert torch.allclose(losses["gate_weak"], expected)


def test_object_trans_test_loss_records_obj_trans():
    batch_size, seq_len = 2, 3
    batch = {
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
        "obj_trans": torch.ones(batch_size, seq_len, 3),
    }
    pred = {"pred_obj_trans": torch.zeros(batch_size, seq_len, 3)}

    loss_fn = ObjectTransLoss()
    total, test_losses = loss_fn.compute_test_loss(pred, batch, torch.device("cpu"))

    assert "obj_trans" in test_losses
    assert test_losses["obj_trans"].item() > 0
    assert total.item() >= test_losses["obj_trans"].item()


def test_velocity_contact_legacy_hand_contact_input_mode_forward():
    cfg = SimpleNamespace(
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        velocity_hidden_dim=16,
        velocity_num_layers=1,
        velocity_dropout=0.0,
        boundary_group_hidden=8,
        boundary_hidden_dim=8,
    )
    model = VelocityContactModule(cfg)
    model.enable_legacy_hand_contact_input(27)

    batch_size, seq_len = 2, 4
    eye6 = torch.eye(3)[:, :2].reshape(6)
    data = {
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "obj_imu": torch.zeros(batch_size, seq_len, 9),
        "hand_vel_glb_init": torch.zeros(batch_size, 2, 3),
        "obj_vel_init": torch.zeros(batch_size, 3),
        "contact_init": torch.zeros(batch_size, 3),
    }
    data["human_imu"][..., 3:9] = eye6
    data["obj_imu"][..., 3:9] = eye6

    out = model(data)
    assert model.hand_contact_input_mode == "legacy"
    assert model.hand_contact_net.linear1.in_features == 27
    assert out["pred_hand_contact_prob"].shape == (batch_size, seq_len, 3)
    assert out["pred_obj_vel"].shape == (batch_size, seq_len, 3)


def test_velocity_contact_hp_ckpt_resolves_from_config_pretrained_modules():
    cfg = SimpleNamespace(
        hp_ckpt=None,
        pretrained_modules=SimpleNamespace(human_pose="outputs/IMUHOI_rnn/human_pose_12302142/best.pt"),
    )

    assert _resolve_hp_ckpt(cfg) == "outputs/IMUHOI_rnn/human_pose_12302142/best.pt"
    assert _resolve_hp_ckpt(cfg, "custom_hp.pt") == "custom_hp.pt"


def test_object_trans_checkpoint_resolution_prefers_config_before_auto():
    cfg = SimpleNamespace(
        hp_ckpt=None,
        pretrained_modules=SimpleNamespace(
            human_pose="cfg_hp.pt",
            velocity_contact="cfg_vc.pt",
            object_trans="cfg_ot.pt",
        ),
    )

    assert _resolve_pretrained_ckpt(cfg, None, "human_pose", "hp_ckpt") == (
        "cfg_hp.pt",
        "config.pretrained_modules.human_pose",
    )
    assert _resolve_pretrained_ckpt(cfg, "cli_vc.pt", "velocity_contact", "vc_ckpt") == (
        "cli_vc.pt",
        "cli",
    )
    assert _resolve_pretrained_ckpt(SimpleNamespace(), None, "human_pose", "hp_ckpt") == (None, None)


def test_velocity_contact_focal_binary_mask_denominator_expands_to_loss_shape():
    pred = torch.full((2, 3, 2), 0.5)
    target = torch.zeros_like(pred)
    sample_mask = torch.tensor([[[1.0]], [[0.0]]])

    loss = VelocityContactLoss._focal_binary(pred, target, mask=sample_mask)
    expected = -(1 - 0.25) * (0.5 ** 2) * torch.log(torch.tensor(0.5))

    assert torch.allclose(loss, expected, atol=1e-6)


def test_build_model_input_dict_preserves_object_metadata_fields():
    cfg = SimpleNamespace(
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        mesh_downsample_points=5,
        imu_acc_noise_std=0.0,
        imu_rot_noise_std=0.0,
        obj_imu_acc_noise_std=0.0,
        obj_imu_rot_noise_std=0.0,
    )
    batch_size, seq_len = 2, 4
    batch = {
        "human_imu": torch.zeros(batch_size, seq_len, cfg.num_human_imus, cfg.imu_dim),
        "obj_imu": torch.zeros(batch_size, seq_len, cfg.obj_imu_dim),
        "obj_rot": torch.randn(batch_size, seq_len, 6),
        "obj_trans": torch.randn(batch_size, seq_len, 3),
        "obj_scale": torch.ones(batch_size, seq_len),
        "obj_points_canonical": torch.randn(batch_size, 5, 3),
    }

    out = build_model_input_dict(batch, cfg, torch.device("cpu"), add_noise=False)
    assert out["obj_rot_gt"].shape == (batch_size, seq_len, 6)
    assert out["obj_trans_gt"].shape == (batch_size, seq_len, 3)
    assert out["obj_scale_gt"].shape == (batch_size, seq_len)
    assert out["obj_points_canonical"].shape == (batch_size, 5, 3)


def test_stage3_joint_trains_vc_immediately():
    class TinyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)

    cfg = SimpleNamespace(
        device="cpu",
        use_multi_gpu=False,
        lr=1e-3,
        weight_decay=0.0,
        milestones=[],
        gamma=0.1,
        use_tensorboard=False,
        debug=True,
        no_trans=False,
        loss_weights={},
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

    assert all(not p.requires_grad for p in trainer.hp_model.parameters())
    assert all(p.requires_grad for p in trainer.vc_model.parameters())
    assert any(p.requires_grad for p in trainer.ot_model.parameters())
