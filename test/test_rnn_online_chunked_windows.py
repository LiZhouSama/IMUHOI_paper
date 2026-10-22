"""Chunked online inference must publish exactly what the per-frame loop does.

`online_chunk` runs the HumanPose and VelocityContact window forwards once for a
whole chunk of target frames, because every window in the chunk reads its
initializers from frames published before the chunk started.  Only the
published-frame finalization and the ObjectTrans scan stay sequential.

The stub modules here are deterministic functions of their inputs, so a chunked
run has to match the sequential run bit for bit: any mistake in the staged
history, the window initializers or the published prefix shows up immediately.
On real modules the same comparison only holds to cuDNN's batch-size noise,
which is why the logic is pinned here instead.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from types import SimpleNamespace

from configs import _REDUCED_POSE_NAMES
from model.rnn.imuhoi_model import IMUHOIModel

WINDOW = 4
LOOKAHEAD = 2
POSE_DIM = len(_REDUCED_POSE_NAMES) * 6


def _cfg():
    return SimpleNamespace(
        model_arch="rnn",
        test={"window": WINDOW},
        train={"window": WINDOW},
        num_human_imus=6,
        imu_dim=9,
        obj_imu_dim=9,
        hidden_dim_multiplier=1,
        enable_ot_refine=False,
        object_trans_lb_stats=False,
    )


class _StubHumanPose(nn.Module):
    """Window forward plus the serial published-frame finalization."""

    fps = 30.0

    def forward(self, data_dict):
        human_imu = data_dict["human_imu"]
        batch_size, seq_len = human_imu.shape[:2]
        # Depend on both the window inputs and the window initializers, so a
        # wrong initializer index cannot pass unnoticed.
        signal = human_imu.reshape(batch_size, seq_len, -1).mean(dim=-1, keepdim=True)
        signal = signal + data_dict["p_init"].reshape(batch_size, 1, -1).mean(dim=-1, keepdim=True)
        signal = signal + data_dict["trans_init"].reshape(batch_size, 1, -1).mean(dim=-1, keepdim=True)
        root = signal.expand(batch_size, seq_len, 3).contiguous()
        joints = signal.view(batch_size, seq_len, 1, 1).expand(batch_size, seq_len, 24, 3).contiguous()
        hands = signal.view(batch_size, seq_len, 1, 1).expand(batch_size, seq_len, 2, 3).contiguous()
        return {
            "v_pred": signal.expand(batch_size, seq_len, 12).contiguous(),
            "p_pred": signal.expand(batch_size, seq_len, POSE_DIM).contiguous(),
            "root_vel_pred": root.clone(),
            "root_vel_unadjusted_pred": root.clone(),
            "root_trans_pred": root,
            "pred_foot_pos": hands.clone(),
            "pred_hand_glb_pos": hands.clone(),
            "pred_palm_glb_pos": hands.clone(),
            "pred_joints_local": joints,
            "pred_joints_global": joints.clone(),
        }

    def finalize_online_frame(self, frame, history):
        """Integrate the published root from the frame published just before."""
        previous = history.get("root_trans_pred") if isinstance(history, dict) else None
        if not isinstance(previous, torch.Tensor):
            return frame
        updated = dict(frame)
        root = previous[:, -1:] + frame["root_vel_unadjusted_pred"] / self.fps
        updated["root_trans_pred"] = root
        updated["pred_joints_global"] = frame["pred_joints_local"] + root.unsqueeze(2)
        return updated


class _StubVelocityContact(nn.Module):
    """Reads the whole HumanPose context, prefix included."""

    def forward(self, data_dict, hp_out=None):
        human_imu = data_dict["human_imu"]
        batch_size, seq_len = human_imu.shape[:2]
        assert hp_out is not None and hp_out["p_pred"].shape[1] == seq_len
        # A cumulative sum over the context makes every published prefix frame
        # observable in the output.
        context = hp_out["p_pred"].mean(dim=-1, keepdim=True).cumsum(dim=1)
        context = context + data_dict["obj_vel_init"].reshape(batch_size, 1, -1).mean(dim=-1, keepdim=True)
        return {
            "pred_hand_glb_vel": context.view(batch_size, seq_len, 1, 1).expand(batch_size, seq_len, 2, 3).contiguous(),
            "pred_obj_vel": context.expand(batch_size, seq_len, 3).contiguous(),
            "pred_hand_contact_prob": torch.sigmoid(context).expand(batch_size, seq_len, 3).contiguous(),
            "pred_hand_contact_logits": context.expand(batch_size, seq_len, 3).contiguous(),
        }


class _StubObjectTrans(nn.Module):
    """Autoregressive on its own published prefix, like the real scan."""

    def forward(
        self,
        hand_positions,
        pred_hand_contact_prob,
        obj_trans_init,
        obj_vel_input=None,
        known_obj_trans_prefix=None,
        **_,
    ):
        batch_size, seq_len = pred_hand_contact_prob.shape[:2]
        out = torch.zeros(batch_size, seq_len, 3, dtype=obj_trans_init.dtype)
        prefix_len = 0
        if isinstance(known_obj_trans_prefix, torch.Tensor):
            prefix_len = min(int(known_obj_trans_prefix.shape[1]), seq_len)
            out[:, :prefix_len] = known_obj_trans_prefix[:, :prefix_len]
        step = pred_hand_contact_prob[..., :1] + obj_vel_input[..., :1]
        for t in range(prefix_len, seq_len):
            previous = out[:, t - 1] if t > 0 else obj_trans_init
            out[:, t] = previous + step[:, t]
        return {
            "pred_obj_trans": out,
            "gating_weights": torch.zeros(batch_size, seq_len, 4, dtype=out.dtype),
            "pred_obj_vel_from_posdiff": torch.zeros_like(out),
        }


def _model():
    model = object.__new__(IMUHOIModel)
    nn.Module.__init__(model)
    model.cfg = _cfg()
    model.device = torch.device("cpu")
    model.no_trans = False
    model.human_pose_module = _StubHumanPose()
    model.velocity_contact_module = _StubVelocityContact()
    model.object_trans_module = _StubObjectTrans()
    return model


def _data(seq_len):
    torch.manual_seed(0)
    return {
        "human_imu": torch.randn(1, seq_len, 6, 9),
        "obj_imu": torch.randn(1, seq_len, 9),
        "v_init": torch.zeros(1, 4, 3),
        "p_init": torch.zeros(1, len(_REDUCED_POSE_NAMES), 6),
        "trans_init": torch.zeros(1, 3),
        "trans_gt": torch.zeros(1, seq_len, 3),
        "obj_trans_init": torch.zeros(1, 3),
        "obj_vel_init": torch.zeros(1, 3),
        "hand_vel_glb_init": torch.zeros(1, 2, 3),
        "contact_init": torch.zeros(1, 3),
        "has_object": torch.ones(1, dtype=torch.bool),
    }


def _run(model, data, chunk):
    return model.inference(
        data,
        use_object_data=True,
        compute_fk=False,
        refine_human=False,
        inference_mode="online",
        online_window=WINDOW,
        online_lookahead=LOOKAHEAD,
        online_chunk=chunk,
    )


@pytest.mark.parametrize("chunk", [2, 3, WINDOW - 1, 64])
def test_chunked_online_matches_the_per_frame_loop(chunk):
    model = _model()
    data = _data(seq_len=14)
    reference = _run(model, data, 1)
    chunked = _run(model, data, chunk)
    assert set(chunked) == set(reference)
    for key, ref in reference.items():
        if not isinstance(ref, torch.Tensor):
            continue
        assert chunked[key].shape == ref.shape, key
        torch.testing.assert_close(chunked[key], ref, rtol=0, atol=0, msg=f"{key} differs")


def test_chunk_is_clamped_to_the_window_so_initializers_stay_published():
    """A chunk wider than the window would need frames it has not published."""
    model = _model()
    data = _data(seq_len=14)
    reference = _run(model, data, 1)
    # 64 is far wider than the window; the loop must clamp it rather than read
    # unpublished history.
    clamped = _run(model, data, 64)
    torch.testing.assert_close(clamped["pred_obj_trans"], reference["pred_obj_trans"], rtol=0, atol=0)


def test_chunked_run_calls_the_window_modules_once_per_chunk():
    model = _model()
    data = _data(seq_len=14)
    calls = {"hp": 0, "vc": 0}
    hp_forward = model.human_pose_module.forward
    vc_forward = model.velocity_contact_module.forward

    def counted_hp(*args, **kwargs):
        calls["hp"] += 1
        return hp_forward(*args, **kwargs)

    def counted_vc(*args, **kwargs):
        calls["vc"] += 1
        return vc_forward(*args, **kwargs)

    model.human_pose_module.forward = counted_hp
    model.velocity_contact_module.forward = counted_vc
    _run(model, data, 3)
    # 14 frames, lookahead 2 -> 12 published, window 4 warms up 4 of them, so 8
    # sequential targets become 3 chunk forwards, plus the warmup forward.
    assert calls["hp"] == 1 + 3, calls
    assert calls["vc"] == 1 + 3, calls


class _StopRequested(BaseException):
    """Stands in for vis_IMUHOI.SequenceCancelled without importing the viewer."""


@pytest.mark.parametrize("chunk", [1, 3])
def test_progress_callback_can_abort_the_online_loop(chunk):
    """The viewer interrupts a sequence by raising from the progress hook.

    The exception derives from BaseException so the viewer's broad
    `except Exception` handlers let it through; the online loop must not
    swallow it either, and must stop at the frame that raised.
    """
    model = _model()
    data = _data(seq_len=14)
    seen = []

    def stop_after_two(completed, total):
        seen.append(completed)
        if len(seen) >= 3:
            raise _StopRequested()

    with pytest.raises(_StopRequested):
        model.inference(
            data,
            use_object_data=True,
            compute_fk=False,
            refine_human=False,
            inference_mode="online",
            online_window=WINDOW,
            online_lookahead=LOOKAHEAD,
            online_chunk=chunk,
            online_progress_callback=stop_after_two,
        )
    # the warmup report, then one call per published frame until the raise
    assert len(seen) == 3, seen
    assert seen == sorted(seen)
