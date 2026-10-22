"""Anchor-state feedback (object_trans_feedback_source=anchor).

The predictor's per-frame feedback reads a second recursion fused with the fixed
contact-driven weights of ObjectTrans v2, never the learned-gate publication.
The anchor is published, handed back as a known prefix, and the batched known
prefix must reproduce the sequential scan.
"""
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import FRAME_RATE
from model.rnn.object_trans import ObjectTransModule
from utils.object_fusion import build_contact_fusion_weights
sys.path.insert(0, str(PROJECT_ROOT / "test"))
from test_rnn_ot_known_prefix_fast_path import BATCH, PREFIX, SEQ, _cfg, _compare, _inputs, _run  # noqa: E402


def _module(**overrides):
    return ObjectTransModule(_cfg(object_trans_feedback_source="anchor", **overrides)).eval()


def test_requires_state_feedback_and_rejects_latent_feedback():
    with pytest.raises(ValueError):
        ObjectTransModule(_cfg(object_trans_feedback_source="anchor", object_trans_state_feedback=False))
    with pytest.raises(ValueError):
        ObjectTransModule(_cfg(object_trans_feedback_source="anchor", object_trans_grasp_latent=True))
    with pytest.raises(ValueError):
        ObjectTransModule(_cfg(object_trans_feedback_source="bogus"))


def test_anchor_is_the_fixed_contact_fusion_of_the_fk_candidates():
    module = _module()
    assert "pred_anchor_obj_trans" in module.SEGMENT_STATE_OUTPUT_KEYS
    data = _inputs(seed=3)
    with torch.no_grad():
        out = _run(
            module, data,
            data["obj_trans_init"].unsqueeze(1).expand(BATCH, 1, 3),
            torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(BATCH, 1, 4).contiguous(),
        )
    anchor = out["pred_anchor_obj_trans"]
    assert anchor.shape == (BATCH, SEQ, 3)
    weights = build_contact_fusion_weights(data["pred_hand_contact_prob"])
    prev = data["obj_trans_init"]
    for t in range(SEQ):
        expected = (
            weights[:, t, 0:1] * out["pred_lhand_obj_trans"][:, t]
            + weights[:, t, 1:2] * out["pred_rhand_obj_trans"][:, t]
            + weights[:, t, 2:3] * (prev + data["obj_vel_input"][:, t] / FRAME_RATE)
            + weights[:, t, 3:4] * prev
        )
        torch.testing.assert_close(anchor[:, t], expected, rtol=1e-4, atol=1e-6)
        prev = anchor[:, t]
    # The published output still comes from the learned gate, not the anchor.
    assert not torch.allclose(out["pred_obj_trans"], anchor)


def test_feedback_reads_the_anchor_not_the_published_state():
    """Perturbing the published prefix (with the anchor prefix fixed) must not change the next prediction."""
    module = _module()
    # The feedback projection is zero-initialised (an exact no-op); give it weight so the test can see it.
    torch.nn.init.normal_(module.feedback_embed[-1].weight, std=0.1)
    data = _inputs(seed=4)
    with torch.no_grad():
        full = _run(
            module, data,
            data["obj_trans_init"].unsqueeze(1).expand(BATCH, 1, 3),
            torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(BATCH, 1, 4).contiguous(),
        )
        prefix_pos = full["pred_obj_trans"][:, :PREFIX].clone()
        prefix_w = full["gating_weights"][:, :PREFIX].clone()
        segment = {"pred_anchor_obj_trans": full["pred_anchor_obj_trans"][:, :PREFIX].clone()}
        # Frame 0 is also the window init (initial bone geometry), so only the
        # later published frames are shifted: they reach the predictor through
        # the feedback alone.
        shifted_pos = prefix_pos.clone()
        shifted_pos[:, 1:] += 0.5
        ref = _run(module, data, prefix_pos, prefix_w, known_segment_state_prefix=segment)
        shifted = _run(module, data, shifted_pos, prefix_w, known_segment_state_prefix=segment)
    # Bone outputs at the first free frame depend on the feedback only through the anchor.
    for key in ("pred_lhand_obj_direction", "pred_lhand_lb", "pred_rhand_obj_direction", "pred_rhand_lb"):
        torch.testing.assert_close(shifted[key][:, PREFIX], ref[key][:, PREFIX], rtol=1e-4, atol=1e-5)
    # ... whereas the published-state feedback would have reacted to the shift.
    published = ObjectTransModule(_cfg()).eval()
    published.load_state_dict(module.state_dict())
    with torch.no_grad():
        ref_p = _run(published, data, prefix_pos, prefix_w)
        shifted_p = _run(published, data, shifted_pos, prefix_w)
    assert not torch.allclose(shifted_p["pred_lhand_lb"][:, PREFIX], ref_p["pred_lhand_lb"][:, PREFIX])


@pytest.mark.parametrize("target_index", [PREFIX, SEQ - 1])
def test_anchor_fast_prefix_matches_sequential_scan(target_index):
    module = _module()
    with torch.no_grad():
        _compare(module, _inputs(seed=5), target_index, segment_keys=("pred_anchor_obj_trans",))


def test_without_anchor_prefix_the_scan_stays_sequential():
    module = _module()
    data = _inputs(seed=6)
    calls = []
    original = module._batched_known_prefix
    module._batched_known_prefix = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    with torch.no_grad():
        full = _run(
            module, data,
            data["obj_trans_init"].unsqueeze(1).expand(BATCH, 1, 3),
            torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(BATCH, 1, 4).contiguous(),
        )
        _run(module, data, full["pred_obj_trans"][:, :PREFIX], full["gating_weights"][:, :PREFIX],
             fast_known_prefix=True, scan_end=PREFIX + 1)
    assert calls == []
