import os
import sys
import types

import torch
from easydict import EasyDict as edict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Provide lightweight stubs when optional runtime deps are unavailable in test env.
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

try:
    import human_body_prior.body_model.body_model  # noqa: F401
except Exception:
    hbp_mod = types.ModuleType("human_body_prior")
    body_model_pkg = types.ModuleType("human_body_prior.body_model")
    body_model_mod = types.ModuleType("human_body_prior.body_model.body_model")

    class _DummyBodyModel(torch.nn.Module):
        def __init__(self, *_, **__):
            super().__init__()

        def forward(self, *_, **__):
            return types.SimpleNamespace(Jtr=torch.zeros(1, 24, 3))

    body_model_mod.BodyModel = _DummyBodyModel
    hbp_mod.body_model = body_model_pkg
    body_model_pkg.body_model = body_model_mod
    sys.modules["human_body_prior"] = hbp_mod
    sys.modules["human_body_prior.body_model"] = body_model_pkg
    sys.modules["human_body_prior.body_model.body_model"] = body_model_mod

from pytorch3d.transforms import matrix_to_rotation_6d
from model.diffussion.interaction import InteractionModule
from model.diffussion.human_pose import HumanPoseModule
import model.diffussion.human_pose as human_pose_mod
from model.diffussion.imuhoi_model import IMUHOIModel
from train.diffussion.loss.human_pose_loss import HumanPoseLoss
from train.diffussion.loss.interaction_loss import InteractionLoss
from train.diffussion.train_human_pose import HumanPoseTrainer
from utils.utils import build_model_input_dict, load_checkpoint


def _make_cfg():
    return edict(
        {
            "num_joints": 24,
            "num_human_imus": 6,
            "imu_dim": 9,
            "obj_imu_dim": 9,
            "frame_rate": 30,
            "body_model_path": "dummy_smpl_path",
            "imu_noise_std": 0.5,
            "obj_imu_noise_std": 0.5,
            "train": {"window": 32},
            "test": {"window": 32},
            "dit": {
                "dit_d_model": 64,
                "dit_nhead": 4,
                "dit_num_layers": 2,
                "dit_dim_feedforward": 128,
                "dit_dropout": 0.0,
                "dit_timesteps": 50,
                "dit_use_time_embed": True,
                "dit_use_noise": True,
                "dit_max_seq_len": 64,
                "dit_inference_steps": 10,
                "dit_inference_sampler": "ddim",
                "dit_inference_eta": 0.0,
                "formulation": "residual",
            },
            "interaction": {"head_mode": "split"},
        }
    )


def test_build_model_input_dict_uses_sample_level_noise_mask():
    torch.manual_seed(123)
    cfg = _make_cfg()
    device = torch.device("cpu")
    bs, seq = 2, 6
    batch = {
        "human_imu": torch.zeros(bs, seq, cfg.num_human_imus, cfg.imu_dim),
        "obj_imu": torch.zeros(bs, seq, cfg.obj_imu_dim),
        "imu_noise_applied": torch.tensor([True, False]),
    }

    out = build_model_input_dict(batch, cfg, device, add_noise=True)
    assert torch.allclose(out["human_imu"][0], torch.zeros_like(out["human_imu"][0]))
    assert not torch.allclose(out["human_imu"][1], torch.zeros_like(out["human_imu"][1]))
    assert torch.allclose(out["obj_imu"][0], torch.zeros_like(out["obj_imu"][0]))
    assert not torch.allclose(out["obj_imu"][1], torch.zeros_like(out["obj_imu"][1]))


def test_build_model_input_dict_passes_object_metadata():
    cfg = _make_cfg()
    device = torch.device("cpu")
    bs, seq = 2, 4
    batch = {
        "human_imu": torch.zeros(bs, seq, cfg.num_human_imus, cfg.imu_dim),
        "obj_imu": torch.zeros(bs, seq, cfg.obj_imu_dim),
        "obj_name": ["mug", "bottle"],
        "seq_file": ["s1.pt", "s2.pt"],
        "window_start": torch.tensor([3, 7]),
        "window_end": torch.tensor([63, 67]),
    }
    out = build_model_input_dict(batch, cfg, device, add_noise=False)
    assert out["obj_name"] == ["mug", "bottle"]
    assert out["seq_file"] == ["s1.pt", "s2.pt"]
    assert torch.equal(out["window_start"], torch.tensor([3, 7]))
    assert torch.equal(out["window_end"], torch.tensor([63, 67]))


def _make_interaction_inputs(cfg, bs=2, seq=8, has_object=True):
    eye = torch.eye(3).view(1, 1, 3, 3).expand(bs, seq, 3, 3).clone()
    obj_rot6d = matrix_to_rotation_6d(eye.reshape(-1, 3, 3)).reshape(bs, seq, 6)
    has_object_tensor = torch.full((bs,), bool(has_object), dtype=torch.bool)

    data_dict = {
        "human_imu": torch.randn(bs, seq, cfg.num_human_imus, cfg.imu_dim),
        "obj_imu": torch.randn(bs, seq, cfg.obj_imu_dim),
        "obj_trans_init": torch.randn(bs, 3),
        "contact_init": torch.zeros(bs, 3),
        "has_object": has_object_tensor,
        "obj_name": [f"obj_{i}" for i in range(bs)],
        "seq_file": [f"seq_{i}.pt" for i in range(bs)],
        "window_start": torch.arange(bs, dtype=torch.long),
        "window_end": torch.arange(bs, dtype=torch.long) + seq,
    }
    hp_out = {
        "pred_hand_glb_pos": torch.randn(bs, seq, 2, 3),
        "root_vel_pred": torch.randn(bs, seq, 3),
        "pred_full_pose_6d": torch.randn(bs, seq, cfg.num_joints, 6),
    }
    gt_targets = {
        "obj_trans": torch.randn(bs, seq, 3),
        "obj_rot": obj_rot6d,
        "obj_scale": torch.ones(bs, seq),
        "lhand_contact": torch.randint(0, 2, (bs, seq), dtype=torch.float32),
        "rhand_contact": torch.randint(0, 2, (bs, seq), dtype=torch.float32),
        "obj_contact": torch.randint(0, 2, (bs, seq), dtype=torch.float32),
        "interaction_start_gauss": torch.zeros(bs, seq),
        "interaction_end_gauss": torch.zeros(bs, seq),
    }
    return data_dict, hp_out, gt_targets


def test_interaction_split_head_forward_keeps_output_contract():
    torch.manual_seed(0)
    cfg = _make_cfg()
    module = InteractionModule(cfg).train()
    bs, seq = 2, 8
    data_dict, hp_out, gt_targets = _make_interaction_inputs(cfg, bs=bs, seq=seq, has_object=True)

    out = module(data_dict, hp_out=hp_out, gt_targets=gt_targets)
    assert out["pred_obj_trans"].shape == (bs, seq, 3)
    assert out["contact_prob_pred"].shape == (bs, seq, 2)
    assert out["bone_dir_pred"].shape == (bs, seq, 2, 3)
    assert out["object_prior_aux"]["z_q_obs"].shape[0] == bs
    assert out["diffusion_aux"]["object_code_idx_pred"].shape == (bs,)
    assert module.dit.cond_dim == module.code_dim


def test_interaction_inference_obs_prior_outputs_code_index():
    torch.manual_seed(1)
    cfg = _make_cfg()
    module = InteractionModule(cfg).eval()
    bs, seq = 2, 6
    data_dict, hp_out, _ = _make_interaction_inputs(cfg, bs=bs, seq=seq, has_object=True)
    out = module.inference(data_dict, hp_out=hp_out, gt_targets=None, sample_steps=2)
    assert out["pred_obj_trans"].shape == (bs, seq, 3)
    assert out["diffusion_aux"]["object_code_idx_pred"].shape == (bs,)
    assert out["object_prior_aux"]["z_q_obs"].shape == (bs, module.code_dim)


def test_interaction_no_object_forces_null_condition():
    torch.manual_seed(2)
    cfg = _make_cfg()
    cfg.cond_mode_probs = [1.0, 0.0, 0.0]
    module = InteractionModule(cfg).train()
    bs, seq = 2, 6
    data_dict, hp_out, gt_targets = _make_interaction_inputs(cfg, bs=bs, seq=seq, has_object=False)
    out = module(data_dict, hp_out=hp_out, gt_targets=gt_targets)
    prior = out["object_prior_aux"]
    assert bool((prior["mode"] == 2).all())
    expected_null = module.null_object_code.expand(bs, -1).to(prior["cond"].dtype)
    assert torch.allclose(prior["cond"], expected_null, atol=1e-6)


def test_interaction_mesh_loader_failure_falls_back_to_obs():
    torch.manual_seed(3)
    cfg = _make_cfg()
    cfg.cond_mode_probs = [1.0, 0.0, 0.0]
    module = InteractionModule(cfg).train()
    bs, seq = 2, 6
    data_dict, hp_out, gt_targets = _make_interaction_inputs(cfg, bs=bs, seq=seq, has_object=True)

    def _broken_loader(*_args, **_kwargs):
        raise RuntimeError("mock mesh loader failure")

    module._get_load_object_geometry = lambda: _broken_loader
    out = module(data_dict, hp_out=hp_out, gt_targets=gt_targets)
    prior = out["object_prior_aux"]
    assert bool((prior["mesh_valid_mask"] == 0).all())
    # mesh mode should fallback to obs mode when teacher is unavailable
    assert bool((prior["mode"] == 1).all())


def test_interaction_metadata_mismatch_disables_mesh_teacher():
    torch.manual_seed(4)
    cfg = _make_cfg()
    cfg.cond_mode_probs = [1.0, 0.0, 0.0]
    module = InteractionModule(cfg).train()
    bs, seq = 2, 6
    data_dict, hp_out, gt_targets = _make_interaction_inputs(cfg, bs=bs, seq=seq, has_object=True)
    data_dict["obj_name"] = ["only_one_name"]  # mismatch with batch size

    out = module(data_dict, hp_out=hp_out, gt_targets=gt_targets)
    prior = out["object_prior_aux"]
    assert bool(prior["mesh_dp_mismatch"].all())
    assert bool((prior["mesh_valid_mask"] == 0).all())
    assert bool((prior["mode"] == 1).all())


def test_interaction_loss_prior_terms_with_and_without_aux():
    torch.manual_seed(5)
    loss_fn = InteractionLoss(
        weights={
            "Loss_simple": 1.0,
            "Loss_vel_obj": 0.0,
            "Loss_jitter_obj": 0.0,
            "Loss_align": 1.0,
            "Loss_code_cls": 1.0,
            "Loss_commit": 0.1,
        }
    )

    bs, seq, target_dim = 2, 6, 16
    batch = {
        "human_imu": torch.zeros(bs, seq, 6, 9),
        "has_object": torch.ones(bs, dtype=torch.bool),
        "obj_trans": torch.zeros(bs, seq, 3),
        "obj_vel": torch.zeros(bs, seq, 3),
    }
    pred_base = {
        "x_pred": torch.zeros(bs, seq, target_dim),
        "pred_obj_trans": torch.zeros(bs, seq, 3),
        "pred_obj_vel": torch.zeros(bs, seq, 3),
        "diffusion_aux": {"x0_target": torch.zeros(bs, seq, target_dim)},
    }
    pred_with_aux = {
        **pred_base,
        "object_prior_aux": {
            "z_e_obs": torch.randn(bs, 8),
            "z_e_mesh": torch.randn(bs, 8),
            "z_q_mesh": torch.randn(bs, 8),
            "code_idx_mesh": torch.tensor([1, 2], dtype=torch.long),
            "code_logits_obs": torch.randn(bs, 16),
            "mesh_valid_mask": torch.ones(bs, dtype=torch.bool),
            "vq_beta": torch.tensor(0.25),
        },
    }

    _, losses_with_aux, _ = loss_fn(pred_with_aux, batch, torch.device("cpu"))
    assert losses_with_aux["align"].item() > 0.0
    assert losses_with_aux["code_cls"].item() > 0.0
    assert losses_with_aux["commit"].item() > 0.0

    _, losses_no_aux, _ = loss_fn(pred_base, batch, torch.device("cpu"))
    assert losses_no_aux["align"].item() == 0.0
    assert losses_no_aux["code_cls"].item() == 0.0
    assert losses_no_aux["commit"].item() == 0.0


def test_load_checkpoint_strict_false_skips_shape_mismatch(tmp_path):
    model = torch.nn.Linear(4, 4)
    weight_before = model.weight.detach().clone()
    bad_weight = torch.randn(3, 4)
    good_bias = torch.randn(4)

    ckpt_path = tmp_path / "shape_mismatch.pt"
    torch.save(
        {
            "epoch": 5,
            "model_state_dict": {
                "weight": bad_weight,
                "bias": good_bias,
            },
        },
        ckpt_path,
    )

    epoch = load_checkpoint(model, str(ckpt_path), torch.device("cpu"), strict=False, use_ema=False)
    assert epoch == 5
    assert torch.allclose(model.bias, good_bias)
    assert torch.allclose(model.weight, weight_before)


def test_run_hp_uses_sampling_inference_in_all_stage2_paths():
    class _DummyHP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0
            self.inference_calls = []

        def forward(self, *_args, **_kwargs):
            self.forward_calls += 1
            return {"path": "forward"}

        def inference(self, _hp_input, gt_targets=None, sample_steps=None, sampler=None, eta=None):
            self.inference_calls.append(
                {
                    "gt_targets_is_none": gt_targets is None,
                    "sample_steps": sample_steps,
                    "sampler": sampler,
                    "eta": eta,
                    "grad_enabled": torch.is_grad_enabled(),
                }
            )
            return {"path": "inference"}

    model = IMUHOIModel.__new__(IMUHOIModel)
    torch.nn.Module.__init__(model)
    model.human_pose_module = _DummyHP()
    model.hp_train_sample_steps = 20
    model.hp_train_sampler = "ddim"
    model.hp_train_eta = 0.0

    hp_input = {"human_imu": torch.zeros(1, 4, 6, 9), "v_init": torch.zeros(1, 8, 3), "p_init": torch.zeros(1, 10, 6)}
    gt_targets = {"sensor_vel_root": torch.zeros(1, 4, 6, 3), "ori_root_reduced": torch.zeros(1, 4, 10, 3, 3), "trans": torch.zeros(1, 4, 3)}

    model.training = True
    out_freeze = IMUHOIModel._run_hp(
        model,
        hp_input,
        gt_targets,
        detach_hp=True,
        sample_steps=None,
        sampler=None,
        eta=None,
    )
    assert out_freeze["path"] == "inference"
    assert model.human_pose_module.forward_calls == 0
    assert model.human_pose_module.inference_calls[-1]["gt_targets_is_none"] is False
    assert model.human_pose_module.inference_calls[-1]["sample_steps"] == 20
    assert model.human_pose_module.inference_calls[-1]["sampler"] == "ddim"
    assert model.human_pose_module.inference_calls[-1]["eta"] == 0.0
    assert model.human_pose_module.inference_calls[-1]["grad_enabled"] is True

    out_joint = IMUHOIModel._run_hp(
        model,
        hp_input,
        gt_targets,
        detach_hp=False,
        sample_steps=None,
        sampler="ddpm",
        eta=0.5,
    )
    assert out_joint["path"] == "inference"
    assert model.human_pose_module.forward_calls == 0
    assert model.human_pose_module.inference_calls[-1]["sample_steps"] == 20
    assert model.human_pose_module.inference_calls[-1]["sampler"] == "ddpm"
    assert model.human_pose_module.inference_calls[-1]["eta"] == 0.5
    assert model.human_pose_module.inference_calls[-1]["grad_enabled"] is True

    model.training = False
    out_eval = IMUHOIModel._run_hp(
        model,
        hp_input,
        gt_targets,
        detach_hp=False,
        sample_steps=9,
        sampler=None,
        eta=None,
    )
    assert out_eval["path"] == "inference"
    assert model.human_pose_module.forward_calls == 0
    assert model.human_pose_module.inference_calls[-1]["sample_steps"] == 9
    assert model.human_pose_module.inference_calls[-1]["sampler"] is None
    assert model.human_pose_module.inference_calls[-1]["eta"] is None
    assert model.human_pose_module.inference_calls[-1]["grad_enabled"] is True


def _make_hp_cfg(window=5):
    cfg = _make_cfg()
    cfg.train = {"window": window}
    cfg.test = {"window": window}
    cfg.dit["dit_timesteps"] = 20
    cfg.dit["dit_inference_steps"] = 4
    cfg.dit["dit_max_seq_len"] = 64
    return cfg


def _make_hp_inputs(bs, seq, device):
    human_imu = torch.randn(bs, seq, 6, 9, device=device)
    v_init = torch.randn(bs, 6, 3, device=device)
    p_init = torch.randn(bs, 10, 6, device=device)
    trans_init = torch.randn(bs, 3, device=device)
    eye = torch.eye(3, device=device).view(1, 1, 1, 3, 3)
    gt_targets = {
        "sensor_vel_root": torch.randn(bs, seq, 6, 3, device=device),
        "ori_root_reduced": eye.expand(bs, seq, 10, 3, 3).clone(),
        "rotation_global": eye.expand(bs, seq, 24, 3, 3).clone(),
        "root_vel": torch.randn(bs, seq, 3, device=device),
        "trans": torch.randn(bs, seq, 3, device=device),
    }
    data_dict = {
        "human_imu": human_imu,
        "v_init": v_init,
        "p_init": p_init,
        "trans_init": trans_init,
    }
    return data_dict, gt_targets


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
    model.body_model = None
    return model


def test_human_pose_forward_uses_rollout_unknown_mask():
    torch.manual_seed(0)
    cfg = _make_hp_cfg(window=5)
    model = _make_hp_model(cfg)
    model.train()
    model.set_train_rollout_k(3)

    bs, seq = 2, 6
    data_dict, gt_targets = _make_hp_inputs(bs, seq, torch.device("cpu"))
    out = model(data_dict, gt_targets=gt_targets)
    aux = out["diffusion_aux"]

    assert aux["rollout_k"] == 3
    assert aux["rollout_start_idx"] == 3
    unknown_motion_mask = aux["unknown_motion_mask"]
    assert unknown_motion_mask[:, :3].sum().item() == 0
    assert bool(unknown_motion_mask[:, 3:].all())
    assert bool(aux["rollout_frame_mask"][:, :3].eq(False).all())
    assert bool(aux["rollout_frame_mask"][:, 3:].all())


def test_human_pose_rollout_uses_predicted_history_not_gt():
    torch.manual_seed(11)
    cfg = _make_hp_cfg(window=5)
    cfg.dit["dit_use_noise"] = False
    model = _make_hp_model(cfg)
    model.train()
    model.set_train_rollout_k(2)

    bs, seq = 1, 6
    data_dict, gt_targets = _make_hp_inputs(bs, seq, torch.device("cpu"))
    x_seed, context = model._prepare_inputs(data_dict)
    gt_motion = model._build_motion_target_from_gt(gt_targets, context)

    captured_inputs = []
    call_idx = {"value": 0}
    motion_slice = slice(model.imu_feat_dim, None)

    def _fake_predict(x_t, t, cond=None):
        del t, cond
        captured_inputs.append(x_t.detach().clone())
        x0_pred = x_t.clone()
        rollout_start = seq - 2
        frame_idx = rollout_start + call_idx["value"]
        fill_value = 100.0 + call_idx["value"]
        x0_pred[:, frame_idx, motion_slice] = fill_value
        call_idx["value"] += 1
        eps_pred = torch.zeros_like(x_t)
        return x0_pred, eps_pred, x0_pred

    model.dit.predict = _fake_predict
    _ = model(data_dict, gt_targets=gt_targets)

    assert len(captured_inputs) == 2
    second_step_input = captured_inputs[1][:, 5, motion_slice]
    assert torch.allclose(second_step_input, torch.full_like(second_step_input, 100.0))
    assert not torch.allclose(second_step_input, gt_motion[:, 4])


def test_human_pose_rollout_k_curriculum_schedule():
    assert HumanPoseTrainer.rollout_k_for_epoch(0, 100) == 3
    assert HumanPoseTrainer.rollout_k_for_epoch(24, 100) == 3
    assert HumanPoseTrainer.rollout_k_for_epoch(25, 100) == 5
    assert HumanPoseTrainer.rollout_k_for_epoch(50, 100) == 10
    assert HumanPoseTrainer.rollout_k_for_epoch(75, 100) == 30
    assert HumanPoseTrainer.rollout_k_for_epoch(99, 100) == 30


def test_human_pose_dynamic_micro_batch_size_schedule():
    assert HumanPoseTrainer.micro_batch_size_for_rollout(150, 3) == 150
    assert HumanPoseTrainer.micro_batch_size_for_rollout(150, 5) == 90
    assert HumanPoseTrainer.micro_batch_size_for_rollout(150, 10) == 45
    assert HumanPoseTrainer.micro_batch_size_for_rollout(150, 30) == 15
    assert HumanPoseTrainer.micro_batch_size_for_rollout(2, 30) == 1


def test_human_pose_loss_rollout_mask_ignores_warmup_error():
    bs, seq = 1, 6
    device = torch.device("cpu")
    human_imu = torch.zeros(bs, seq, 6, 9, device=device)
    eye = torch.eye(3, device=device).view(1, 1, 1, 3, 3)
    batch = {
        "human_imu": human_imu,
        "sensor_vel_root": torch.zeros(bs, seq, 6, 3, device=device),
        "ori_root_reduced": eye.expand(bs, seq, 10, 3, 3).clone(),
        "trans": torch.zeros(bs, seq, 3, device=device),
        "root_vel": torch.zeros(bs, seq, 3, device=device),
    }

    v_pred = torch.ones(bs, seq, 8 * 3, device=device)
    v_pred[:, :3] = 10.0
    pred_dict = {
        "v_pred": v_pred,
        "diffusion_aux": {
            "rollout_frame_mask": torch.tensor([[0, 0, 0, 1, 1, 1]], device=device, dtype=torch.bool),
            "rollout_start_idx": 3,
        },
    }

    loss_fn = HumanPoseLoss(no_trans=True)
    _, losses, _ = loss_fn(pred_dict, batch, device)
    assert torch.isclose(losses["simple_vel"], torch.tensor(1.0, device=device), atol=1e-6)


def test_human_pose_loss_drift_uses_rollout_start_reference():
    bs, seq = 1, 4
    device = torch.device("cpu")
    human_imu = torch.zeros(bs, seq, 6, 9, device=device)
    batch = {
        "human_imu": human_imu,
        "sensor_vel_root": torch.zeros(bs, seq, 6, 3, device=device),
        "trans": torch.zeros(bs, seq, 3, device=device),
        "root_vel": torch.zeros(bs, seq, 3, device=device),
    }
    root_trans_pred = torch.tensor(
        [[[100.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        device=device,
    )

    loss_fn = HumanPoseLoss(
        weights={
            "simple_vel": 0.0,
            "simple_pose": 0.0,
            "simple_root_vel_local": 0.0,
            "simple_root_vel": 0.0,
            "simple_root_trans": 0.0,
            "vel_smooth": 0.0,
            "fk_joint": 0.0,
            "drift": 1.0,
            "foot_slide": 0.0,
            "hand_pos": 0.0,
            "diffusion_x0": 0.0,
        },
        no_trans=False,
    )

    pred_rollout = {
        "root_trans_pred": root_trans_pred.clone(),
        "root_vel_pred": torch.zeros(bs, seq, 3, device=device),
        "root_vel_local_pred": torch.zeros(bs, seq, 3, device=device),
        "diffusion_aux": {
            "rollout_frame_mask": torch.tensor([[0, 0, 1, 1]], device=device, dtype=torch.bool),
            "rollout_start_idx": 2,
        },
    }
    _, losses_rollout, _ = loss_fn(pred_rollout, batch, device)
    assert losses_rollout["drift"].item() < 1e-6

    pred_nomask = {
        "root_trans_pred": root_trans_pred.clone(),
        "root_vel_pred": torch.zeros(bs, seq, 3, device=device),
        "root_vel_local_pred": torch.zeros(bs, seq, 3, device=device),
        "diffusion_aux": {},
    }
    _, losses_nomask, _ = loss_fn(pred_nomask, batch, device)
    assert losses_nomask["drift"].item() > 1000.0


def test_human_pose_inference_raises_when_seq_not_longer_than_warmup():
    torch.manual_seed(1)
    cfg = _make_hp_cfg(window=5)
    model = _make_hp_model(cfg)
    model.eval()

    bs, seq = 1, 4  # warmup_len = 4
    data_dict, gt_targets = _make_hp_inputs(bs, seq, torch.device("cpu"))
    try:
        model.inference(data_dict, gt_targets=gt_targets, sample_steps=2)
        raised = False
    except ValueError as exc:
        raised = "too short" in str(exc)
    assert raised


def test_human_pose_inference_reports_warmup_len():
    torch.manual_seed(2)
    cfg = _make_hp_cfg(window=5)
    model = _make_hp_model(cfg)
    model.eval()

    bs, seq = 1, 6
    data_dict, gt_targets = _make_hp_inputs(bs, seq, torch.device("cpu"))
    out = model.inference(data_dict, gt_targets=gt_targets, sample_steps=2)

    assert out["root_trans_pred"].shape[:2] == (bs, seq)
    assert out["diffusion_aux"]["warmup_len"] == 4
    assert out["diffusion_aux"]["window_size"] == 5


def test_human_pose_inference_uses_configured_warmup_len():
    torch.manual_seed(3)
    cfg = _make_hp_cfg(window=5)
    cfg.dit["dit_test_warmup_len"] = 2
    model = _make_hp_model(cfg)
    model.eval()

    bs, seq = 1, 6
    data_dict, gt_targets = _make_hp_inputs(bs, seq, torch.device("cpu"))
    out = model.inference(data_dict, gt_targets=gt_targets, sample_steps=2)

    assert out["root_trans_pred"].shape[:2] == (bs, seq)
    assert out["diffusion_aux"]["warmup_len"] == 2
    assert out["diffusion_aux"]["window_size"] == 5


def test_human_pose_rejects_removed_mode_keys():
    cfg = _make_hp_cfg(window=5)
    cfg.dit["dit_train_current_mode"] = "last"
    class _DummyBodyModel(torch.nn.Module):
        def __init__(self, *_, **__):
            super().__init__()

        def forward(self, *_, **__):
            return types.SimpleNamespace(Jtr=torch.zeros(1, 24, 3))

    orig_body_model = human_pose_mod.BodyModel
    human_pose_mod.BodyModel = _DummyBodyModel
    try:
        try:
            HumanPoseModule(cfg, torch.device("cpu"), no_trans=False)
            raised = False
        except ValueError as exc:
            raised = "removed" in str(exc)
    finally:
        human_pose_mod.BodyModel = orig_body_model
    assert raised
