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
from model.rnn.object_trans import ObjectTransModule
from train.rnn.loss.object_trans_loss import ObjectTransLoss
from train.rnn.scheduled_inputs import prediction_mix_probability, sample_mix_tensor
from train.rnn.train_object_trans import Stage3JointTrainer
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


def _small_prior_cfg(cond_mode_probs=None):
    return SimpleNamespace(
        imu_dim=9,
        obj_imu_dim=9,
        num_human_imus=6,
        hidden_dim_multiplier=1,
        interaction_code_dim=16,
        prior_encoder_hidden_dim=32,
        prior_encoder_layers=1,
        prior_encoder_dropout=0.0,
        dgcnn_k=4,
        dgcnn_chunk_size=8,
        cond_mode_probs=cond_mode_probs or [0.4, 0.4, 0.2],
    )


def _object_trans_prior_inputs(batch_size=2, seq_len=3, num_points=8, has_object=True):
    eye = torch.eye(3).view(1, 1, 3, 3).expand(batch_size, seq_len, 3, 3)
    obj_rot6d = eye[..., :2].reshape(batch_size, seq_len, 6).contiguous()
    return {
        "hand_pos": torch.randn(batch_size, seq_len, 2, 3),
        "contact": torch.sigmoid(torch.randn(batch_size, seq_len, 3)),
        "obj_trans_init": torch.randn(batch_size, 3),
        "obj_imu": torch.cat((torch.randn(batch_size, seq_len, 3), obj_rot6d), dim=-1),
        "human_imu": torch.randn(batch_size, seq_len, 6, 9),
        "obj_vel": torch.randn(batch_size, seq_len, 3),
        "pose": torch.randn(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6),
        "root": torch.randn(batch_size, seq_len, 3),
        "obj_points": torch.randn(batch_size, num_points, 3),
        "obj_rot": obj_rot6d,
        "obj_trans": torch.randn(batch_size, seq_len, 3),
        "obj_scale": torch.ones(batch_size, seq_len),
        "has_object": torch.full((batch_size,), bool(has_object), dtype=torch.bool),
    }


def test_object_trans_interaction_prior_mesh_train_outputs_frame_codes():
    torch.manual_seed(123)
    model = ObjectTransModule(_small_prior_cfg(cond_mode_probs=[1.0, 0.0, 0.0])).train()
    data = _object_trans_prior_inputs()

    out = model(
        data["hand_pos"],
        data["contact"],
        data["obj_trans_init"],
        obj_imu=data["obj_imu"],
        human_imu=data["human_imu"],
        obj_vel_input=data["obj_vel"],
        has_object_mask=data["has_object"],
        human_pose_input=data["pose"],
        root_trans_input=data["root"],
        obj_points_canonical=data["obj_points"],
        obj_rot_gt=data["obj_rot"],
        obj_trans_gt=data["obj_trans"],
        obj_scale_gt=data["obj_scale"],
    )

    aux = out["interaction_prior_aux"]
    assert out["interaction_code"].shape == (2, 3, 16)
    assert aux["mesh_code"].shape == (2, 3, 16)
    assert aux["obs_code"].shape == (2, 3, 16)
    assert bool((aux["mesh_valid_mask"] == 1).all())
    assert bool((aux["mode"] == 0).all())
    assert out["pred_obj_trans"].shape == (2, 3, 3)


def test_object_trans_interaction_prior_eval_uses_obs_without_mesh_gt():
    torch.manual_seed(124)
    model = ObjectTransModule(_small_prior_cfg(cond_mode_probs=[1.0, 0.0, 0.0])).eval()
    data = _object_trans_prior_inputs()

    with torch.no_grad():
        out = model(
            data["hand_pos"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            has_object_mask=data["has_object"],
            human_pose_input=data["pose"],
            root_trans_input=data["root"],
        )

    aux = out["interaction_prior_aux"]
    assert out["interaction_code"].shape == (2, 3, 16)
    assert bool((aux["mode"] == 1).all())
    assert torch.allclose(aux["mesh_code"], torch.zeros_like(aux["mesh_code"]))


def test_object_trans_interaction_prior_mesh_missing_falls_back_to_obs():
    torch.manual_seed(125)
    model = ObjectTransModule(_small_prior_cfg(cond_mode_probs=[1.0, 0.0, 0.0])).train()
    data = _object_trans_prior_inputs()

    out = model(
        data["hand_pos"],
        data["contact"],
        data["obj_trans_init"],
        obj_imu=data["obj_imu"],
        human_imu=data["human_imu"],
        obj_vel_input=data["obj_vel"],
        has_object_mask=data["has_object"],
        human_pose_input=data["pose"],
        root_trans_input=data["root"],
    )

    aux = out["interaction_prior_aux"]
    assert bool((aux["mesh_valid_mask"] == 0).all())
    assert bool((aux["mode"] == 1).all())


def test_object_trans_interaction_prior_no_object_forces_null_code():
    torch.manual_seed(126)
    model = ObjectTransModule(_small_prior_cfg(cond_mode_probs=[0.0, 1.0, 0.0])).train()
    data = _object_trans_prior_inputs(has_object=False)

    out = model(
        data["hand_pos"],
        data["contact"],
        data["obj_trans_init"],
        obj_imu=data["obj_imu"],
        human_imu=data["human_imu"],
        obj_vel_input=data["obj_vel"],
        has_object_mask=data["has_object"],
        human_pose_input=data["pose"],
        root_trans_input=data["root"],
        obj_points_canonical=data["obj_points"],
        obj_rot_gt=data["obj_rot"],
        obj_trans_gt=data["obj_trans"],
        obj_scale_gt=data["obj_scale"],
    )

    aux = out["interaction_prior_aux"]
    assert bool((aux["mode"] == 2).all())
    assert torch.allclose(out["interaction_code"], torch.zeros_like(out["interaction_code"]))


def test_object_trans_loss_interaction_code_align_term():
    batch_size, seq_len, code_dim = 2, 3, 4
    batch = {
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "has_object": torch.ones(batch_size, dtype=torch.bool),
        "obj_trans": torch.zeros(batch_size, seq_len, 3),
    }
    pred = {
        "pred_obj_trans": torch.zeros(batch_size, seq_len, 3),
        "interaction_prior_aux": {
            "obs_code": torch.zeros(batch_size, seq_len, code_dim),
            "mesh_code": torch.ones(batch_size, seq_len, code_dim),
            "mesh_valid_mask": torch.tensor([True, False]),
            "sample_has_object": torch.ones(batch_size, dtype=torch.bool),
        },
    }

    loss_fn = ObjectTransLoss(weights={"obj_trans": 0.0})
    _, losses, weighted = loss_fn(pred, batch, torch.device("cpu"))
    assert losses["interaction_code_align"].item() > 0
    assert weighted["interaction_code_align"].item() > 0

    pred_without_aux = {"pred_obj_trans": torch.zeros(batch_size, seq_len, 3)}
    _, losses_no_aux, _ = loss_fn(pred_without_aux, batch, torch.device("cpu"))
    assert losses_no_aux["interaction_code_align"].item() == 0


def test_build_model_input_dict_passes_object_prior_fields():
    cfg = SimpleNamespace(
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        mesh_downsample_points=5,
        imu_noise_std=0.0,
        obj_imu_noise_std=0.0,
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


def test_stage3_joint_unfreezes_vc_only_after_boundary():
    class TinyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)

    cfg = SimpleNamespace(
        device="cpu",
        joint_vc_unfreeze_epoch=2,
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
    assert all(not p.requires_grad for p in trainer.vc_model.parameters())
    assert any(p.requires_grad for p in trainer.ot_model.parameters())

    trainer._update_joint_state(1)
    assert all(not p.requires_grad for p in trainer.vc_model.parameters())

    trainer._update_joint_state(2)
    assert all(not p.requires_grad for p in trainer.hp_model.parameters())
    assert all(p.requires_grad for p in trainer.vc_model.parameters())
