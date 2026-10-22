"""The batched known prefix must reproduce the per-frame scan.

Online sliding-window inference publishes one frame per window and hands the
next window its published object trajectory and gate weights as a teacher-forced
prefix.  `fast_known_prefix` replaces that prefix's Python loop with a single
batched LSTM call and `scan_end` stops the scan after the published frame; both
must leave the published frame unchanged, for every ObjectTrans branch that
declares support.
"""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from easydict import EasyDict

from model.rnn.object_trans import ObjectTransModule
from model.rnn.object_trans_mergeHeads_obsEncoder import (
    ObjectTransModule as MergeHeadsObjectTransModule,
)
from model.rnn.object_trans_v3 import V3ObjectTransModule

BATCH = 2
SEQ = 12
PREFIX = 8  # published frames, mirrors the online window prefix
NUM_IMUS = 6
IMU_DIM = 9


def _cfg(**overrides):
    cfg = EasyDict(
        {
            "object_trans_hidden_dim": 32,
            "object_trans_num_layers": 2,
            "object_trans_dropout": 0.0,
            "object_trans_state_feedback": True,
            "object_trans_feedback_hidden_dim": 16,
            "object_trans_length_init": 0.4,
            "object_trans_static_gate": "none",
            "object_trans_lb_stats": False,
            "enable_ot_refine": False,
            "velocity_hidden_dim": 32,
            "velocity_num_layers": 1,
            "hidden_dim_multiplier": 1,
        }
    )
    cfg.update(overrides)
    return cfg


def _inputs(seed=0):
    torch.manual_seed(seed)
    hand_positions = torch.randn(BATCH, SEQ, 2, 3) * 0.2
    contact = torch.rand(BATCH, SEQ, 3)
    obj_trans_init = torch.randn(BATCH, 3) * 0.1
    obj_imu = torch.randn(BATCH, SEQ, IMU_DIM)
    human_imu = torch.randn(BATCH, SEQ, NUM_IMUS, IMU_DIM)
    return {
        "hand_positions": hand_positions,
        "pred_hand_contact_prob": contact,
        "obj_trans_init": obj_trans_init,
        "obj_imu": obj_imu,
        "human_imu": human_imu,
        "obj_vel_input": torch.randn(BATCH, SEQ, 3) * 0.05,
        "contact_init": contact[:, 0],
        "human_pose_input": torch.randn(BATCH, SEQ, 60) * 0.1,
        "root_trans_input": torch.randn(BATCH, SEQ, 3) * 0.1,
    }


def _run(module, data, prefix_pos, prefix_weights, **hints):
    return module(
        data["hand_positions"],
        data["pred_hand_contact_prob"],
        prefix_pos[:, 0],
        obj_imu=data["obj_imu"],
        human_imu=data["human_imu"],
        obj_vel_input=data["obj_vel_input"],
        contact_init=data["contact_init"],
        human_pose_input=data["human_pose_input"],
        root_trans_input=data["root_trans_input"],
        enable_refine=False,
        known_obj_trans_prefix=prefix_pos,
        known_gating_weights_prefix=prefix_weights,
        **hints,
    )


def _published_prefix(module, data, segment_keys=()):
    """A published prefix produced by the module itself, as online inference does."""
    full = _run(
        module,
        data,
        data["obj_trans_init"].unsqueeze(1).expand(BATCH, 1, 3),
        torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(BATCH, 1, 4).contiguous(),
    )
    prefix_pos = full["pred_obj_trans"][:, :PREFIX].detach().clone()
    prefix_weights = full["gating_weights"][:, :PREFIX].detach().clone()
    segment = {key: full[key][:, :PREFIX].detach().clone() for key in segment_keys}
    return prefix_pos, prefix_weights, segment


def _compare(module, data, target_index, segment_keys=()):
    prefix_pos, prefix_weights, segment = _published_prefix(module, data, segment_keys)
    extra = {"known_segment_state_prefix": segment} if segment else {}
    reference = _run(module, data, prefix_pos, prefix_weights, **extra)

    # Guard against a silent fallback making this test vacuous.
    calls = []
    original = module._batched_known_prefix

    def counted(*args, **kwargs):
        calls.append(int(args[4]) if len(args) > 4 else kwargs["prefix_len"])
        return original(*args, **kwargs)

    module._batched_known_prefix = counted
    try:
        fast = _run(
            module,
            data,
            prefix_pos,
            prefix_weights,
            fast_known_prefix=True,
            scan_end=target_index + 1,
            **extra,
        )
    finally:
        module._batched_known_prefix = original
    assert calls == [PREFIX], f"batched prefix not used as expected: {calls}"
    for key, ref in reference.items():
        if not isinstance(ref, torch.Tensor) or ref.dim() < 2 or ref.shape[1] != SEQ:
            continue
        got = fast[key]
        assert got.shape == ref.shape, f"{key}: {got.shape} != {ref.shape}"
        torch.testing.assert_close(
            got[:, target_index], ref[:, target_index], rtol=1e-4, atol=1e-5,
            msg=lambda m, key=key: f"published frame differs for {key}: {m}",
        )


@pytest.mark.parametrize("target_index", [PREFIX, SEQ - 1])
def test_current_arch_fast_prefix_matches_sequential_scan(target_index):
    module = ObjectTransModule(_cfg()).eval()
    with torch.no_grad():
        _compare(module, _inputs(seed=1), target_index)


def test_v3_fast_prefix_matches_sequential_scan():
    cfg = _cfg(object_trans_v3_online_carry_segment_state=True)
    module = V3ObjectTransModule(cfg).eval()
    with torch.no_grad():
        _compare(
            module,
            _inputs(seed=2),
            PREFIX,
            segment_keys=V3ObjectTransModule.SEGMENT_STATE_OUTPUT_KEYS,
        )


def test_static_gate_still_allows_the_batched_prefix():
    """The hard static gate only rewrites weights the published prefix replaces."""
    cfg = _cfg(
        object_trans_static_gate="imu",
        object_trans_static_gate_imu_window=3,
        object_trans_static_gate_imu_angle_deg=90.0,
        object_trans_static_gate_imu_acc=1e6,
    )
    module = ObjectTransModule(cfg).eval()
    data = _inputs(seed=5)
    with torch.no_grad():
        # Thresholds above are wide enough that the gate fires; a gate that
        # never fires would make this test vacuous.
        gate = module._compute_static_gate(
            data["obj_imu"],
            torch.eye(3).view(1, 1, 3, 3).expand(BATCH, SEQ, 3, 3),
            data["pred_hand_contact_prob"],
        )
        assert gate is not None and bool(gate.any()), "static gate never fires"
        _compare(module, data, PREFIX)


def test_mergeheads_arch_fast_prefix_matches_sequential_scan():
    module = MergeHeadsObjectTransModule(_cfg()).eval()
    with torch.no_grad():
        _compare(module, _inputs(seed=3), PREFIX)


def test_bimanual_closure_fast_prefix_matches_sequential_scan():
    """The closure is state-free, so it keeps the batched prefix."""
    module = ObjectTransModule(_cfg(object_trans_bimanual_closure=True)).eval()
    _open_closure_gate(module)
    with torch.no_grad():
        _compare(module, _inputs(seed=6), PREFIX)


def _open_closure_gate(module):
    # A fresh closure gate starts near zero weight (bias -2, zero last layer);
    # push it open so the fifth candidate actually moves the fused position.
    with torch.no_grad():
        module.bimanual_closure.gate_head[-1].bias.fill_(2.0)
        module.bimanual_closure.min_contact = 0.0


def _v2(**overrides):
    from model.rnn.object_trans_v2 import V2ObjectTransModule

    return V2ObjectTransModule(_cfg(**overrides)).eval()


@pytest.mark.parametrize("target_index", [PREFIX, SEQ - 1])
@pytest.mark.parametrize("closure", [False, True], ids=["v2", "v2_biclose"])
def test_v2_fk_filter_fast_prefix_matches_sequential_scan(target_index, closure):
    """v2's alpha-beta FK filter is restored from its published state."""
    module = _v2(object_trans_bimanual_closure=closure, object_trans_forearm_obj_rotation=True)
    assert module.fk_filter_enabled
    if closure:
        _open_closure_gate(module)
    data = _inputs(seed=7)
    # Both hands firmly in contact: the filter latches on frame 0 and stays
    # latched across the prefix boundary.  With random contact it drops out
    # on the first scanned frame, and a wrong restored state would go unseen.
    data["pred_hand_contact_prob"][..., :2] = 0.8 + 0.2 * torch.rand(BATCH, SEQ, 2)
    with torch.no_grad():
        _, _, segment = _published_prefix(module, data, module.SEGMENT_STATE_OUTPUT_KEYS)
        assert bool((segment["v2_fk_filter_active"][:, -1] > 0.5).all())
        _compare(
            module,
            data,
            target_index,
            segment_keys=module.SEGMENT_STATE_OUTPUT_KEYS,
        )


def test_v2_without_fk_filter_fast_prefix_matches_sequential_scan():
    module = _v2(v2_fk_filter_alpha=1.0, object_trans_bimanual_closure=True)
    assert not module.fk_filter_enabled
    _open_closure_gate(module)
    with torch.no_grad():
        _compare(module, _inputs(seed=8), PREFIX)


def test_v2_fk_filter_without_published_state_falls_back():
    """Without the published filter state the prefix has to be scanned."""
    module = _v2()
    data = _inputs(seed=9)
    calls = []
    original = module._batched_known_prefix
    module._batched_known_prefix = lambda *a, **k: calls.append(1) or original(*a, **k)
    with torch.no_grad():
        prefix_pos, prefix_weights, _ = _published_prefix(module, data)
        reference = _run(module, data, prefix_pos, prefix_weights)
        fast = _run(module, data, prefix_pos, prefix_weights,
                    fast_known_prefix=True, scan_end=SEQ)
    assert calls == []
    torch.testing.assert_close(fast["pred_obj_trans"], reference["pred_obj_trans"])


def test_branches_that_keep_their_own_recurrence_opt_out():
    """Every ObjectTrans branch here restores its recurrence from published state."""
    from model.rnn.object_trans_v2 import V2ObjectTransModule

    assert ObjectTransModule.SUPPORTS_FAST_KNOWN_PREFIX is True
    assert V3ObjectTransModule.SUPPORTS_FAST_KNOWN_PREFIX is True
    assert MergeHeadsObjectTransModule.SUPPORTS_FAST_KNOWN_PREFIX is True
    assert V2ObjectTransModule.SUPPORTS_FAST_KNOWN_PREFIX is True


def test_grasp_latent_config_falls_back_to_the_sequential_scan():
    module = ObjectTransModule(
        _cfg(object_trans_grasp_latent=True, object_trans_grasp_latent_hidden_dim=16)
    ).eval()
    data = _inputs(seed=4)
    with torch.no_grad():
        prefix_pos, prefix_weights, _ = _published_prefix(module, data)
        assert module._fast_known_prefix_supported({"static_gate": None, "grasp_known": None,
                                                    "state_trust": None, "bigrasp_known": None,
                                                    "seq_len": SEQ, "feedback_mode": "normal"},
                                                   PREFIX) is False
        reference = _run(module, data, prefix_pos, prefix_weights)
        fast = _run(module, data, prefix_pos, prefix_weights,
                    fast_known_prefix=True, scan_end=SEQ)
    torch.testing.assert_close(fast["pred_obj_trans"], reference["pred_obj_trans"])
