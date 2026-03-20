import os
import sys
import types

import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Lightweight stubs for optional deps in minimal test envs.
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

from pytorch3d.transforms import matrix_to_rotation_6d

try:
    import human_body_prior.body_model.body_model  # noqa: F401
except Exception:
    hbp_mod = types.ModuleType("human_body_prior")
    body_model_pkg = types.ModuleType("human_body_prior.body_model")
    body_model_mod = types.ModuleType("human_body_prior.body_model.body_model")

    class _DummyBodyModel(torch.nn.Module):
        def __init__(self, *_, **__):
            super().__init__()

        def forward(self, pose_body=None, root_orient=None):
            batch = pose_body.shape[0] if isinstance(pose_body, torch.Tensor) else 1
            return types.SimpleNamespace(Jtr=torch.zeros(batch, 24, 3))

    body_model_mod.BodyModel = _DummyBodyModel
    hbp_mod.body_model = body_model_pkg
    body_model_pkg.body_model = body_model_mod
    sys.modules["human_body_prior"] = hbp_mod
    sys.modules["human_body_prior.body_model"] = body_model_pkg
    sys.modules["human_body_prior.body_model.body_model"] = body_model_mod

from train.diffussion.loss.human_pose_loss import HumanPoseLoss
from model.diffussion.human_pose import HumanPoseModule
import model.diffussion.human_pose as human_pose_mod


def _make_hp_cfg(enable_root_correction: bool):
    return edict(
        {
            "num_joints": 24,
            "num_human_imus": 6,
            "imu_dim": 9,
            "frame_rate": 30,
            "body_model_path": "dummy_smpl_path",
            "train": {"window": 4},
            "test": {"window": 4},
            "dit": {
                "dit_d_model": 32,
                "dit_nhead": 4,
                "dit_num_layers": 1,
                "dit_dim_feedforward": 64,
                "dit_dropout": 0.0,
                "dit_timesteps": 8,
                "dit_use_time_embed": True,
                "dit_use_noise": False,
                "dit_prediction_type": "x0",
                "dit_max_seq_len": 8,
                "dit_inference_steps": 2,
                "dit_enable_root_correction": enable_root_correction,
                "dit_root_correction_contact_threshold": 0.5,
            },
        }
    )


def _make_hp_model(cfg):
    class _DummyBodyModel(torch.nn.Module):
        def __init__(self, *_, **__):
            super().__init__()

        def forward(self, pose_body=None, root_orient=None):
            batch = pose_body.shape[0] if isinstance(pose_body, torch.Tensor) else 1
            return types.SimpleNamespace(Jtr=torch.zeros(batch, 24, 3))

    orig_body_model = human_pose_mod.BodyModel
    human_pose_mod.BodyModel = _DummyBodyModel
    try:
        model = HumanPoseModule(cfg, torch.device("cpu"), no_trans=False)
    finally:
        human_pose_mod.BodyModel = orig_body_model
    return model


def test_compute_test_loss_metrics_and_weighted_total():
    device = torch.device("cpu")
    bs, seq, joints = 1, 4, 24
    dtype = torch.float32

    test_metric_weights = {
        "mpjre_mse": 2.0,
        "root_trans_err_mm": 0.5,
        "jitter_local_fk_mm": 1.0,
        "root_jitter_mm": 1.5,
    }
    loss_fn = HumanPoseLoss(test_metric_weights=test_metric_weights, no_trans=False)

    rot_gt = torch.eye(3, dtype=dtype).view(1, 1, 1, 3, 3).expand(bs, seq, joints, 3, 3).clone()
    r_pred = torch.zeros(bs, seq, joints, 6, dtype=dtype)

    # root x trajectory: [0, 1, 4, 9]
    root_trans_pred = torch.zeros(bs, seq, 3, dtype=dtype)
    root_trans_pred[0, :, 0] = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=dtype)
    trans_gt = torch.zeros(bs, seq, 3, dtype=dtype)

    pred_joints_local = torch.zeros(bs, seq, joints, 3, dtype=dtype)
    pred_joints_local[0, :, :, 0] = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=dtype).view(seq, 1).expand(seq, joints)

    batch = {
        "human_imu": torch.zeros(bs, seq, 6, 9, dtype=dtype),
        "rotation_global": rot_gt,
        "trans": trans_gt,
    }
    pred_dict = {
        "R_pred_6d": r_pred,
        "root_trans_pred": root_trans_pred,
        "pred_joints_local": pred_joints_local,
    }

    total, metrics = loss_fn.compute_test_loss(pred_dict, batch, device)

    gt_6d = matrix_to_rotation_6d(rot_gt.reshape(-1, 3, 3)).reshape(bs, seq, joints, 6)
    mpjre_expected = F.mse_loss(r_pred, gt_6d)
    root_err_expected = torch.tensor((0.0 + 1.0 + 4.0 + 9.0) / 4.0 * 1000.0, dtype=dtype)
    jitter_expected = torch.tensor(2.0 * 1000.0, dtype=dtype)
    total_expected = (
        mpjre_expected * test_metric_weights["mpjre_mse"]
        + root_err_expected * test_metric_weights["root_trans_err_mm"]
        + jitter_expected * test_metric_weights["jitter_local_fk_mm"]
        + jitter_expected * test_metric_weights["root_jitter_mm"]
    )

    assert torch.isclose(metrics["mpjre_mse"], mpjre_expected, atol=1e-6)
    assert torch.isclose(metrics["root_trans_err_mm"], root_err_expected, atol=1e-6)
    assert torch.isclose(metrics["jitter_local_fk_mm"], jitter_expected, atol=1e-6)
    assert torch.isclose(metrics["root_jitter_mm"], jitter_expected, atol=1e-6)
    assert torch.isclose(total, total_expected, atol=1e-5)


def test_compute_test_loss_no_trans_zeroes_root_metrics():
    device = torch.device("cpu")
    bs, seq, joints = 1, 4, 24
    dtype = torch.float32

    loss_fn = HumanPoseLoss(
        test_metric_weights={
            "mpjre_mse": 1.0,
            "root_trans_err_mm": 10.0,
            "jitter_local_fk_mm": 1.0,
            "root_jitter_mm": 10.0,
        },
        no_trans=True,
    )

    rot_gt = torch.eye(3, dtype=dtype).view(1, 1, 1, 3, 3).expand(bs, seq, joints, 3, 3).clone()
    r_pred = torch.zeros(bs, seq, joints, 6, dtype=dtype)
    pred_joints_local = torch.zeros(bs, seq, joints, 3, dtype=dtype)
    pred_joints_local[0, :, :, 0] = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=dtype).view(seq, 1).expand(seq, joints)

    batch = {
        "human_imu": torch.zeros(bs, seq, 6, 9, dtype=dtype),
        "rotation_global": rot_gt,
        "trans": torch.zeros(bs, seq, 3, dtype=dtype),
    }
    pred_dict = {
        "R_pred_6d": r_pred,
        "root_trans_pred": torch.full((bs, seq, 3), 123.0, dtype=dtype),
        "pred_joints_local": pred_joints_local,
    }

    total, metrics = loss_fn.compute_test_loss(pred_dict, batch, device)

    gt_6d = matrix_to_rotation_6d(rot_gt.reshape(-1, 3, 3)).reshape(bs, seq, joints, 6)
    expected_total = F.mse_loss(r_pred, gt_6d) + torch.tensor(2.0 * 1000.0, dtype=dtype)

    assert torch.isclose(metrics["root_trans_err_mm"], torch.tensor(0.0, dtype=dtype))
    assert torch.isclose(metrics["root_jitter_mm"], torch.tensor(0.0, dtype=dtype))
    assert torch.isclose(total, expected_total, atol=1e-5)


def _prepare_deterministic_inference_model(enable_root_correction: bool):
    cfg = _make_hp_cfg(enable_root_correction=enable_root_correction)
    model = _make_hp_model(cfg)
    model.eval()

    def _fake_fk(_rot_pair):
        # Keep local joints unchanged between frames; world foot displacement then equals root displacement.
        return torch.zeros(1, 2, model.num_joints, 3, dtype=torch.float32)

    model._compute_fk_joints_from_global = _fake_fk  # type: ignore[method-assign]

    def _fake_sample_inpaint_x0(x_input, inpaint_mask, cond=None, steps=None):
        del inpaint_mask, cond, steps
        out = x_input.clone()
        out[:, -1, model.delta_p_slice] = torch.tensor([1.0, 2.0], dtype=out.dtype, device=out.device)
        out[:, -1, model.py_slice] = 0.0
        out[:, -1, model.contact_slice] = 10.0
        out[:, -1, model.rot_slice] = 0.0
        return out

    model.dit.sample_inpaint_x0 = _fake_sample_inpaint_x0
    return model


def test_root_correction_enabled_corrects_delta_xz_after_first_frame():
    model = _prepare_deterministic_inference_model(enable_root_correction=True)
    observed_seq = torch.zeros(1, 3, model.target_dim, dtype=torch.float32)
    trans_init = torch.zeros(1, 3, dtype=torch.float32)

    pred_seq = model._autoregressive_inference(observed_seq, trans_init=trans_init, steps=1)
    delta_seq = pred_seq[:, :, model.delta_p_slice]

    assert torch.allclose(delta_seq[:, 0], torch.tensor([[1.0, 2.0]], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(delta_seq[:, 1], torch.tensor([[0.0, 0.0]], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(delta_seq[:, 2], torch.tensor([[0.0, 0.0]], dtype=torch.float32), atol=1e-6)
    # non-root features (e.g. contact logits) remain unchanged
    assert torch.all(pred_seq[:, 1:, model.contact_slice] > 9.0)


def test_root_correction_disabled_keeps_predicted_delta_xz():
    model = _prepare_deterministic_inference_model(enable_root_correction=False)
    observed_seq = torch.zeros(1, 3, model.target_dim, dtype=torch.float32)
    trans_init = torch.zeros(1, 3, dtype=torch.float32)

    pred_seq = model._autoregressive_inference(observed_seq, trans_init=trans_init, steps=1)
    delta_seq = pred_seq[:, :, model.delta_p_slice]

    expected = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]], dtype=torch.float32)
    assert torch.allclose(delta_seq, expected, atol=1e-6)


def test_autoregressive_warmup_prefix_keeps_gt_history():
    model = _prepare_deterministic_inference_model(enable_root_correction=False)
    observed_seq = torch.zeros(1, 4, model.target_dim, dtype=torch.float32)
    trans_init = torch.zeros(1, 3, dtype=torch.float32)

    warmup_seq = torch.zeros(1, 2, model.target_dim, dtype=torch.float32)
    warmup_seq[:, 0, model.delta_p_slice] = torch.tensor([3.0, 4.0], dtype=torch.float32)
    warmup_seq[:, 1, model.delta_p_slice] = torch.tensor([5.0, 6.0], dtype=torch.float32)
    warmup_seq[:, :, model.contact_slice] = 7.0

    pred_seq = model._autoregressive_inference(
        observed_seq,
        trans_init=trans_init,
        steps=1,
        warmup_seq=warmup_seq,
    )

    assert torch.allclose(pred_seq[:, :2], warmup_seq, atol=1e-6)
    expected_rollout = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]], dtype=torch.float32)
    assert torch.allclose(pred_seq[:, 2:, model.delta_p_slice], expected_rollout, atol=1e-6)
