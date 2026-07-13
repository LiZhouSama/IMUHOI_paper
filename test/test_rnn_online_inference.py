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
from model.rnn.online import concat_time_dicts, select_time_context, slice_time_dict
from train.rnn.train_utils import call_model_inference


def _small_cfg():
    return SimpleNamespace(
        model_arch="rnn",
        test={"window": 3},
        train={"window": 3},
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        hidden_dim_multiplier=1,
    )


def _feedback_cfg(online_mode="window"):
    cfg = _small_cfg()
    cfg.object_trans_state_feedback = True
    cfg.object_trans_online_mode = online_mode
    cfg.object_trans_hidden_dim = 16
    cfg.object_trans_num_layers = 2
    cfg.object_trans_dropout = 0.0
    cfg.object_trans_feedback_hidden_dim = 12
    cfg.enable_ot_refine = False
    return cfg


def _object_inputs(batch_size=1, seq_len=4):
    eye6 = torch.eye(3)[:, :2].reshape(6)
    return {
        "hand_pos": torch.zeros(batch_size, seq_len, 2, 3),
        "contact": torch.ones(batch_size, seq_len, 3),
        "obj_trans_init": torch.zeros(batch_size, 3),
        "obj_imu": torch.zeros(batch_size, seq_len, 9),
        "human_imu": torch.zeros(batch_size, seq_len, 6, 9),
        "obj_vel": torch.zeros(batch_size, seq_len, 3),
        "pose": torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6),
        "root": torch.zeros(batch_size, seq_len, 3),
    }


def test_object_trans_offline_inference_matches_forward():
    torch.manual_seed(7)
    model = ObjectTransModule(_small_cfg()).eval()
    data = _object_inputs(seq_len=3)

    with torch.no_grad():
        forward_out = model(
            data["hand_pos"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            human_pose_input=data["pose"],
            root_trans_input=data["root"],
        )
        inference_out = model.inference(
            data["hand_pos"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            human_pose_input=data["pose"],
            root_trans_input=data["root"],
            inference_mode="offline",
        )

    assert torch.allclose(inference_out["pred_obj_trans"], forward_out["pred_obj_trans"])
    assert torch.allclose(inference_out["gating_weights"], forward_out["gating_weights"])


def test_object_trans_online_shapes_for_short_equal_long_windows():
    torch.manual_seed(8)
    model = ObjectTransModule(_small_cfg()).eval()
    for seq_len in (2, 3, 5):
        data = _object_inputs(seq_len=seq_len)
        with torch.no_grad():
            out = model.inference(
                data["hand_pos"],
                data["contact"],
                data["obj_trans_init"],
                obj_imu=data["obj_imu"],
                human_imu=data["human_imu"],
                obj_vel_input=data["obj_vel"],
                human_pose_input=data["pose"],
                root_trans_input=data["root"],
                inference_mode="online",
                online_window=3,
            )
        assert out["pred_obj_trans"].shape == (1, seq_len, 3)
        assert out["gating_weights"].shape == (1, seq_len, 4)


def test_object_trans_forward_preserves_known_online_prefix():
    torch.manual_seed(9)
    model = ObjectTransModule(_small_cfg()).eval()
    data = _object_inputs(seq_len=4)
    known_prefix = torch.tensor([[[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0]]])

    with torch.no_grad():
        out = model(
            data["hand_pos"],
            data["contact"],
            data["obj_trans_init"],
            obj_imu=data["obj_imu"],
            human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"],
            human_pose_input=data["pose"],
            root_trans_input=data["root"],
            known_obj_trans_prefix=known_prefix,
        )

    assert torch.allclose(out["pred_obj_trans"][:, :3], known_prefix)


def test_object_trans_window_prefix_preserves_published_gate_weights_for_refine_context():
    torch.manual_seed(91)
    model = ObjectTransModule(_feedback_cfg()).eval()
    data = _object_inputs(seq_len=4)
    known_prefix = torch.randn(1, 3, 3)
    known_weights = torch.tensor(
        [[[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]]]
    )

    with torch.no_grad():
        out = model(
            data["hand_pos"], data["contact"], data["obj_trans_init"],
            obj_imu=data["obj_imu"], human_imu=data["human_imu"], obj_vel_input=data["obj_vel"],
            known_obj_trans_prefix=known_prefix,
            known_gating_weights_prefix=known_weights,
            enable_refine=False,
        )

    assert torch.allclose(out["pred_obj_trans"][:, :3], known_prefix)
    assert torch.allclose(out["gating_weights"][:, :3], known_weights)
    assert torch.allclose(torch.softmax(out["gate_logits"], dim=-1), out["gating_weights_raw"])


def test_object_trans_feedback_state_matches_full_causal_scan_and_is_causal():
    torch.manual_seed(10)
    model = ObjectTransModule(_feedback_cfg()).eval()
    # The zero-initialized residual is intentional for stable training, but a
    # nonzero projection makes this test exercise the feedback path itself.
    nn.init.normal_(model.feedback_embed[-1].weight, std=0.05)
    data = _object_inputs(batch_size=2, seq_len=7)
    data["obj_imu"] = torch.randn_like(data["obj_imu"])

    with torch.no_grad():
        full = model(
            data["hand_pos"], data["contact"], data["obj_trans_init"],
            obj_imu=data["obj_imu"], human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"], enable_refine=False,
        )
        first, state = model(
            data["hand_pos"][:, :3], data["contact"][:, :3], data["obj_trans_init"],
            obj_imu=data["obj_imu"][:, :3], human_imu=data["human_imu"][:, :3],
            obj_vel_input=data["obj_vel"][:, :3], enable_refine=False,
            return_prediction_state=True,
        )
        second, state = model(
            data["hand_pos"][:, 3:], data["contact"][:, 3:], data["obj_trans_init"],
            obj_imu=data["obj_imu"][:, 3:], human_imu=data["human_imu"][:, 3:],
            obj_vel_input=data["obj_vel"][:, 3:], enable_refine=False,
            prediction_state=state, return_prediction_state=True,
        )

        perturbed_hand_pos = data["hand_pos"].clone()
        perturbed_hand_pos[:, 5:] += 100.0
        perturbed = model(
            perturbed_hand_pos, data["contact"], data["obj_trans_init"],
            obj_imu=data["obj_imu"], human_imu=data["human_imu"],
            obj_vel_input=data["obj_vel"], enable_refine=False,
        )

    for key in (
        "pred_obj_trans", "gating_weights", "gate_logits", "lhand_fk_out",
        "rhand_fk_out", "pred_obj_vel_from_posdiff", "pred_obj_acc_from_posdiff",
    ):
        assert torch.allclose(torch.cat((first[key], second[key]), dim=1), full[key], atol=1e-6)
        assert torch.allclose(perturbed[key][:, :5], full[key][:, :5], atol=1e-6)
    assert state["frames_seen"] == 7
    assert state["prev_obj_rotm"].shape == (2, 3, 3)


def test_online_concat_keeps_non_temporal_init_vectors_out_of_history_time():
    history = {
        "init_lhand_oe_ho": torch.zeros(1, 3),
        "p_pred": torch.arange(4, dtype=torch.float32).view(1, 4, 1),
        "pred_obj_trans": torch.zeros(1, 4, 3),
    }
    latest = {
        "init_lhand_oe_ho": torch.ones(1, 3),
        "p_pred": torch.ones(1, 1, 1),
        "pred_obj_trans": torch.ones(1, 1, 3),
    }

    out = concat_time_dicts([history, latest])

    assert out["init_lhand_oe_ho"].shape == (1, 3)
    assert out["p_pred"].shape == (1, 5, 1)

    context = select_time_context(out, 2, 4)
    assert context["init_lhand_oe_ho"].shape == (1, 3)
    assert context["p_pred"].shape == (1, 2, 1)


class _FakeHumanPose(nn.Module):
    def forward(self, data_dict):
        human_imu = data_dict["human_imu"]
        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype
        frame = human_imu[:, :, 0, 0:1]
        pose_dim = len(_REDUCED_POSE_NAMES) * 6
        p_pred = frame.expand(batch_size, seq_len, pose_dim).clone()
        root = frame.expand(batch_size, seq_len, 3).clone()
        joints = frame.view(batch_size, seq_len, 1, 1).expand(batch_size, seq_len, 24, 3).clone()
        return {
            "v_pred": frame.expand(batch_size, seq_len, 12).clone(),
            "p_pred": p_pred,
            "root_vel_pred": torch.ones(batch_size, seq_len, 3, device=device, dtype=dtype),
            "root_vel_local_pred": torch.ones(batch_size, seq_len, 3, device=device, dtype=dtype),
            "root_trans_pred": root,
            "pred_hand_glb_pos": torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype),
            "pred_joints_local": joints,
            "pred_joints_global": joints + root.unsqueeze(2),
            "pred_full_pose_rotmat": torch.zeros(batch_size, seq_len, 24, 3, 3, device=device, dtype=dtype),
            "pred_full_pose_6d": torch.zeros(batch_size, seq_len, 24, 6, device=device, dtype=dtype),
        }


class _FakeVelocityContact(nn.Module):
    def forward(self, data_dict, hp_out=None):
        human_imu = data_dict["human_imu"]
        batch_size, seq_len = human_imu.shape[:2]
        device = human_imu.device
        dtype = human_imu.dtype
        assert hp_out is None or hp_out["p_pred"].shape[1] == seq_len
        return {
            "pred_hand_glb_vel": torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype),
            "pred_obj_vel": torch.ones(batch_size, seq_len, 3, device=device, dtype=dtype),
            "pred_hand_contact_logits": torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype),
            "pred_hand_contact_prob": torch.ones(batch_size, seq_len, 3, device=device, dtype=dtype),
            "pred_obj_move_prob": torch.ones(batch_size, seq_len, 1, device=device, dtype=dtype),
            "pred_hand_contact_prob_cond": torch.ones(batch_size, seq_len, 2, device=device, dtype=dtype),
            "pred_hand_contact_logits_cond": torch.zeros(batch_size, seq_len, 2, device=device, dtype=dtype),
            "pred_interaction_boundary_logits": torch.zeros(batch_size, seq_len, 2, device=device, dtype=dtype),
            "pred_interaction_boundary_prob": torch.zeros(batch_size, seq_len, 2, device=device, dtype=dtype),
        }


class _FakeObjectTrans(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix_lengths = []

    def forward(self, hand_positions, pred_hand_contact_prob, obj_trans_init, known_obj_trans_prefix=None, **_):
        batch_size, seq_len = pred_hand_contact_prob.shape[:2]
        device = pred_hand_contact_prob.device
        dtype = pred_hand_contact_prob.dtype
        out = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        prefix_len = 0
        if isinstance(known_obj_trans_prefix, torch.Tensor):
            prefix_len = min(known_obj_trans_prefix.shape[1], seq_len)
            out[:, :prefix_len] = known_obj_trans_prefix[:, :prefix_len]
        self.prefix_lengths.append(prefix_len)
        for t in range(prefix_len, seq_len):
            prev = out[:, t - 1] if t > 0 else obj_trans_init
            out[:, t] = prev + 1.0
        return {
            "pred_obj_trans": out,
            "gating_weights": torch.zeros(batch_size, seq_len, 4, device=device, dtype=dtype),
            "pred_obj_vel_from_posdiff": torch.zeros_like(out),
            "pred_obj_acc_from_posdiff": torch.zeros_like(out),
        }


def _fake_pipeline_model():
    model = object.__new__(IMUHOIModel)
    nn.Module.__init__(model)
    model.cfg = _small_cfg()
    model.device = torch.device("cpu")
    model.no_trans = False
    model.human_pose_module = _FakeHumanPose()
    model.velocity_contact_module = _FakeVelocityContact()
    model.object_trans_module = _FakeObjectTrans()
    return model


def _pipeline_data(seq_len=5):
    frame = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1, 1)
    human_imu = frame.expand(1, seq_len, 6, 9).clone()
    return {
        "human_imu": human_imu,
        "obj_imu": torch.zeros(1, seq_len, 9),
        "v_init": torch.zeros(1, 4, 3),
        "p_init": torch.zeros(1, len(_REDUCED_POSE_NAMES), 6),
        "trans_init": torch.zeros(1, 3),
        "trans_gt": torch.zeros(1, seq_len, 3),
        "obj_trans_init": torch.zeros(1, 3),
        "obj_trans_gt": torch.zeros(1, seq_len, 3),
        "obj_rot_gt": torch.zeros(1, seq_len, 6),
        "obj_scale_gt": torch.ones(1, seq_len),
        "obj_vel_init": torch.zeros(1, 3),
        "hand_vel_glb_init": torch.zeros(1, 2, 3),
        "contact_init": torch.zeros(1, 3),
        "has_object": torch.ones(1, dtype=torch.bool),
        "obj_points_canonical": torch.zeros(1, 4, 3),
    }


def test_imuhoi_offline_inference_matches_forward_with_stub_modules():
    model = _fake_pipeline_model()
    data = _pipeline_data(seq_len=3)
    forward_out = model(data, use_object_data=True, compute_fk=False, refine_human=False)
    inference_out = model.inference(data, use_object_data=True, compute_fk=False, refine_human=False, inference_mode="offline")
    assert torch.allclose(inference_out["p_pred"], forward_out["p_pred"])
    assert torch.allclose(inference_out["pred_obj_trans"], forward_out["pred_obj_trans"])


def test_imuhoi_online_uses_object_history_prefix_for_new_frames():
    model = _fake_pipeline_model()
    data = _pipeline_data(seq_len=5)
    out = model.inference(data, use_object_data=True, compute_fk=False, refine_human=False, inference_mode="online", online_window=3)
    assert out["pred_obj_trans"].shape == (1, 5, 3)
    assert torch.allclose(out["pred_obj_trans"][0, :, 0], torch.arange(1, 6, dtype=torch.float32))
    assert model.object_trans_module.prefix_lengths == [0, 2, 2]


def test_imuhoi_stateful_ot_persists_across_public_online_chunks():
    torch.manual_seed(11)
    model = _fake_pipeline_model()
    model.cfg = _feedback_cfg(online_mode="stateful")
    model.object_trans_module = ObjectTransModule(model.cfg)
    model.eval()
    data = _pipeline_data(seq_len=8)
    data["obj_imu"][..., 3:9] = torch.eye(3)[:, :2].reshape(6)

    with torch.no_grad():
        whole = model.inference(
            data,
            use_object_data=True,
            compute_fk=False,
            refine_human=False,
            inference_mode="online",
            online_window=3,
        )
        state = None
        chunks = []
        for start, end in ((0, 2), (2, 5), (5, 8)):
            chunk = slice_time_dict(data, start, end, 1, 8)
            out, state = model.inference(
                chunk,
                use_object_data=True,
                compute_fk=False,
                refine_human=False,
                inference_mode="online",
                online_window=3,
                online_state=state,
                return_online_state=True,
            )
            chunks.append(out)
        streamed = concat_time_dicts(chunks)

    for key in ("p_pred", "pred_hand_contact_prob", "pred_obj_trans", "gating_weights", "gate_logits"):
        assert torch.allclose(streamed[key], whole[key], atol=1e-6)
    assert state["input_window"]["human_imu"].shape[1] == 3
    assert state["output_window"]["p_pred"].shape[1] == 3
    assert state["ot_prediction_state"]["frames_seen"] == 8


def test_imuhoi_stateful_ot_stream_matches_single_call_with_refinement_enabled():
    torch.manual_seed(12)
    model = _fake_pipeline_model()
    model.cfg = _feedback_cfg(online_mode="stateful")
    model.cfg.enable_ot_refine = True
    model.object_trans_module = ObjectTransModule(model.cfg)
    model.eval()
    data = _pipeline_data(seq_len=6)
    data["obj_imu"][..., 3:9] = torch.eye(3)[:, :2].reshape(6)

    with torch.no_grad():
        whole = model.inference(
            data,
            use_object_data=True,
            compute_fk=False,
            refine_human=True,
            inference_mode="online",
            online_window=3,
        )
        first, state = model.inference(
            slice_time_dict(data, 0, 1, 1, 6),
            use_object_data=True,
            compute_fk=False,
            refine_human=True,
            inference_mode="online",
            online_window=3,
            return_online_state=True,
        )
        second, state = model.inference(
            slice_time_dict(data, 1, 6, 1, 6),
            use_object_data=True,
            compute_fk=False,
            refine_human=True,
            inference_mode="online",
            online_window=3,
            online_state=state,
            return_online_state=True,
        )
        streamed = concat_time_dicts((first, second))

    for key in ("p_pred", "stage1_p_pred", "pred_obj_trans", "refined_pose", "refined_root_trans"):
        assert torch.allclose(streamed[key], whole[key], atol=1e-6)


class _InferenceOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_mode = None

    def forward(self, data):
        return {"mode": "forward"}

    def inference(self, data, inference_mode="offline"):
        self.seen_mode = inference_mode
        return {"mode": inference_mode}


def test_training_helper_marks_offline_inference():
    model = _InferenceOnly()
    out = call_model_inference(model, {}, inference_mode="offline")
    assert out["mode"] == "offline"
    assert model.seen_mode == "offline"
