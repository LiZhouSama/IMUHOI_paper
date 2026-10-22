"""Online refine promotion: the refined skeleton is derived once after the loop.

Later windows read only the stage1_* history, so the refined pose's FK joints
and full-pose rotations are output-only.  Deriving them for the whole
published sequence after the loop must give what per-window derivation gave,
apart from the root velocity, which becomes the difference of published
roots.
"""
import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from test_rnn_online_inference import (  # noqa: E402
    _FakeHumanPose,
    _FakeObjectTrans,
    _fake_pipeline_model,
    _pipeline_data,
)
from model.rnn.imuhoi_model import IMUHOIModel  # noqa: E402

WINDOW = 3
LOOKAHEAD = 1
SEQ = 9
DERIVED = ("pred_joints_local", "pred_joints_global", "pred_hand_glb_pos", "pred_palm_glb_pos",
           "pred_full_pose_rotmat", "pred_full_pose_6d")


class _RefiningHumanPose(_FakeHumanPose):
    """Stub HP with the FK hooks the promotion uses, all per frame except
    a centred orientation smoothing (half width 1) that reaches into the
    lookahead."""

    body_model = None
    fps = 30.0

    def __init__(self):
        super().__init__()
        self.fk_calls = 0

    def fk_orientation_6d(self, human_imu):
        x = human_imu[..., -6:]
        padded = torch.cat((x[:, :1], x, x[:, -1:]), dim=1)
        smoothed = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3.0
        # The support shrinks symmetrically at the edges, like the SG filter.
        smoothed[:, 0], smoothed[:, -1] = x[:, 0], x[:, -1]
        return smoothed

    def _compute_root_velocity_from_trans(self, trans):
        vel = torch.zeros_like(trans)
        if trans.size(1) > 1:
            vel[:, 1:] = (trans[:, 1:] - trans[:, :-1]) * self.fps
            vel[:, 0] = vel[:, 1]
        return vel

    def _compute_fk_joints_batched(self, pose, orientation, body_model_type=None):
        self.fk_calls += 1
        batch_size, seq_len = pose.shape[:2]
        per_frame = pose[..., :3] + 2.0 * orientation[:, :, 0, 0, :] + orientation[:, :, 1, 1, :]
        joint_offset = torch.arange(24, dtype=pose.dtype).view(1, 1, 24, 1) * 0.01
        return per_frame.view(batch_size, seq_len, 1, 3) + joint_offset

    def _reduced_glb_6d_to_full_glb_mat(self, reduced, orientation):
        rows = reduced.shape[0]
        base = torch.eye(3, dtype=reduced.dtype).expand(rows, 24, 3, 3).clone()
        return base + reduced[:, :1, :1].view(rows, 1, 1, 1) + orientation[:, :1, :1, :1]


class _RefiningObjectTrans(_FakeObjectTrans):
    """Refined outputs depend on the frame's place in the window, as a causal
    refine RNN started at the window's first frame would; the object scan
    reads the hands so a refined value leaking upstream would change it."""

    def forward(self, hand_positions, pred_hand_contact_prob, obj_trans_init,
                known_obj_trans_prefix=None, human_pose_input=None, root_trans_input=None,
                enable_refine=True, **kwargs):
        out = super().forward(hand_positions, pred_hand_contact_prob, obj_trans_init,
                              known_obj_trans_prefix=known_obj_trans_prefix, **kwargs)
        seq_len = pred_hand_contact_prob.shape[1]
        prefix_len = self.prefix_lengths[-1]
        hands = hand_positions.reshape(*hand_positions.shape[:2], -1).sum(-1, keepdim=True)
        out["pred_obj_trans"][:, prefix_len:] += 0.001 * hands[:, prefix_len:]
        if enable_refine and human_pose_input is not None:
            place = torch.arange(seq_len, dtype=human_pose_input.dtype).view(1, seq_len, 1)
            out["refined_pose"] = human_pose_input + 0.1 * (place + 1.0)
            out["refined_root_trans"] = root_trans_input + 0.01 * (place + 1.0)
        return out


def _model():
    model = _fake_pipeline_model()
    model.human_pose_module = _RefiningHumanPose()
    model.object_trans_module = _RefiningObjectTrans()
    return model


def _data():
    rows = []
    for row in range(2):
        data = _pipeline_data(seq_len=SEQ)
        # Quadratic in time: a moving average leaves a linear signal
        # unchanged, which would hide what the smoothing's neighbours are.
        data["human_imu"] = 0.1 * (1.0 + row) * data["human_imu"] ** 2 + torch.linspace(0, 1, 9)
        rows.append(data)
    return {key: torch.cat((rows[0][key], rows[1][key]), dim=0) for key in rows[0]}


def _run(model):
    return model.inference(
        _data(), use_object_data=True, compute_fk=False, refine_human=True,
        inference_mode="online", online_window=WINDOW, online_lookahead=LOOKAHEAD,
    )


def _per_window_reference(monkeypatch):
    """The previous behaviour: derive inside every window, nothing after."""
    promote = IMUHOIModel._promote_refined_human_outputs
    derive = IMUHOIModel._derive_refined_human_outputs
    monkeypatch.setattr(
        IMUHOIModel, "_promote_refined_human_outputs",
        lambda self, results, ot_out, data_dict, derive=True: promote(self, results, ot_out, data_dict, True))
    monkeypatch.setattr(
        IMUHOIModel, "_derive_refined_human_outputs",
        lambda self, results, data_dict, chunk_rows=None:
        None if chunk_rows is not None else derive(self, results, data_dict, chunk_rows))
    model = _model()
    out = _run(model)
    monkeypatch.undo()
    return out, model.human_pose_module.fk_calls


def test_deferred_derivation_matches_per_window_promotion(monkeypatch):
    # Chunked derivation (one frame of the batch-2 sequence per FK call)
    # must not change anything either.
    monkeypatch.setattr(IMUHOIModel, "DERIVE_CHUNK_ROWS", 2)
    reference, reference_calls = _per_window_reference(monkeypatch)
    monkeypatch.setattr(IMUHOIModel, "DERIVE_CHUNK_ROWS", 2)
    model = _model()
    out = _run(model)

    publish_len = SEQ - LOOKAHEAD
    assert out["p_pred"].shape[1] == publish_len
    assert set(out) == set(reference)
    for key, ref in reference.items():
        if not isinstance(ref, torch.Tensor) or key in ("root_vel_pred", "root_vel_local_pred"):
            continue
        assert torch.allclose(out[key], ref, atol=1e-6), key
    for key in DERIVED + ("stage1_p_pred", "stage1_pred_joints_global"):
        assert key in out, key
    # Refined, not Stage-1, values are published.
    assert not torch.allclose(out["pred_joints_global"], out["stage1_pred_joints_global"])

    # One FK call per published frame became one per derivation chunk.
    warmup_calls = 1
    assert reference_calls == warmup_calls + (publish_len - WINDOW)
    assert model.human_pose_module.fk_calls == warmup_calls + publish_len


def test_root_velocity_is_the_difference_of_published_roots():
    out = _run(_model())
    root = out["root_trans_pred"]
    expected = torch.zeros_like(root)
    expected[:, 1:] = (root[:, 1:] - root[:, :-1]) * _RefiningHumanPose.fps
    expected[:, 0] = expected[:, 1]
    assert torch.allclose(out["root_vel_pred"], expected)


def test_fk_baseline_keeps_per_window_derivation(monkeypatch):
    """compute_fk reads refined palms inside the window, so it still derives there."""
    calls = []
    monkeypatch.setattr(IMUHOIModel, "_compute_fk_output",
                        lambda self, results, *a, **k: calls.append(results["pred_joints_global"]))
    model = _model()
    derive = IMUHOIModel._derive_refined_human_outputs
    post_loop = []
    monkeypatch.setattr(
        IMUHOIModel, "_derive_refined_human_outputs",
        lambda self, results, data_dict, chunk_rows=None:
        (post_loop.append(chunk_rows) if chunk_rows is not None else None)
        or derive(self, results, data_dict, chunk_rows))
    model.inference(
        _data(), use_object_data=True, compute_fk=True, refine_human=True,
        inference_mode="online", online_window=WINDOW, online_lookahead=LOOKAHEAD,
    )
    assert calls and post_loop == []
