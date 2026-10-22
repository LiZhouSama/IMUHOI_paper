"""v2 fixed VC fusion combined with forearm rotation and pair closure."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'test'))
from test_rnn_ot_known_prefix_fast_path import _cfg, _inputs, _run  # noqa: E402
from model.rnn.object_trans_v2 import V2ObjectTransModule  # noqa: E402
from model.rnn.object_trans_v3 import V3ObjectTransModule  # noqa: E402
from eval_IMUHOI import _build_published_object_trans  # noqa: E402


def _module(filtered=True):
    cfg = _cfg(object_trans_forearm_obj_rotation=True, object_trans_bimanual_closure=True,
               v2_fk_filter_alpha=.35 if filtered else 1.0,
               v2_fk_filter_beta=.05 if filtered else 0.)
    return V2ObjectTransModule(cfg), cfg


def _data():
    data = _inputs(seed=14)
    data['pred_hand_contact_prob'][:, :6] = 1.0
    data['pred_hand_contact_prob'][:, 6:, 0] = .01
    return data


def test_v2_combined_five_way_is_normalized_and_masked():
    module, _ = _module(filtered=False)
    with torch.no_grad():
        out = module(**_data(), enable_refine=False)
    w5, w = out['gating_weights_5'], out['gating_weights']
    torch.testing.assert_close(w5.sum(-1), torch.ones_like(w5[..., 0]))
    assert (w5[:, :6, 4] > 0).all() and (w5[:, 6:, 4] == 0).all()
    torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]))
    raw_pair = out['pred_bimanual_obj_trans']
    assert raw_pair.shape == out['pred_obj_trans'].shape
    # Four-way effective candidates encode the five-way pair exactly.
    from model.rnn.bimanual_closure import BimanualClosure
    left = out['pred_lhand_obj_trans']; right = out['pred_rhand_obj_trans']
    raw_left = (left * w[:, :, :1] - .5*w5[:, :, 4:5]*raw_pair) / w5[:, :, :1].clamp_min(1e-8)
    raw_right = (right * w[:, :, 1:2] - .5*w5[:, :, 4:5]*raw_pair) / w5[:, :, 1:2].clamp_min(1e-8)
    l2, r2 = BimanualClosure.effective_candidates(raw_left, raw_right, raw_pair, w5)
    torch.testing.assert_close(l2[:, :6], left[:, :6], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(r2[:, :6], right[:, :6], atol=1e-5, rtol=1e-5)
    four = (w[..., 0:1]*left + w[..., 1:2]*right
            + w[..., 2:3]*out['pred_imu_obj_trans'] + w[..., 3:4]*out['pred_static_obj_trans'])
    torch.testing.assert_close(four, out['pred_obj_trans'], atol=1e-5, rtol=1e-5)


def test_effective_four_way_publication_keeps_pair_with_unequal_refinement():
    module, _ = _module(filtered=False)
    data = _data()
    with torch.no_grad():
        out = module(**data, enable_refine=False)
    row = 0
    hs = data['hand_positions'][row]
    refined = hs.clone()
    refined[:, 0, 0] += .03
    refined[:, 1, 1] -= .02
    published = _build_published_object_trans(
        out['pred_obj_trans'][row], hs, refined,
        out['pred_lhand_obj_trans'][row], out['pred_rhand_obj_trans'][row],
        out['pred_imu_obj_trans'][row], out['pred_static_obj_trans'][row], out['gating_weights'][row])
    previous = out['pred_static_obj_trans'][row, 0]
    expected = []
    w5 = out['gating_weights_5'][row]
    for t in range(len(w5)):
        # Reconstitute raw FK solely for an independent explicit five-way replay.
        w = out['gating_weights'][row, t]
        q = w5[t, 4]
        pair = out['pred_bimanual_obj_trans'][row, t]
        left = (w[0]*out['pred_lhand_obj_trans'][row, t] - q/2*pair)/w5[t, 0].clamp_min(1e-8)
        right = (w[1]*out['pred_rhand_obj_trans'][row, t] - q/2*pair)/w5[t, 1].clamp_min(1e-8)
        delta = refined[t]-hs[t]
        current = (w5[t, 0]*(left+delta[0]) + w5[t, 1]*(right+delta[1])
                   + w5[t, 4]*(pair+delta.mean(0))
                   + w5[t, 2]*(previous+out['pred_imu_obj_trans'][row, t]-out['pred_static_obj_trans'][row, t])
                   + w5[t, 3]*previous)
        expected.append(current)
        previous = current
    torch.testing.assert_close(published, torch.stack(expected), atol=1e-5, rtol=1e-5)


def test_filtered_combo_online_prefix_and_gradients():
    module, _ = _module(filtered=True)
    data = _data()
    out = module(**data, enable_refine=False)
    assert module.forearm_obj_rotation_enabled and module.fk_filter_enabled
    out['pred_obj_trans'].square().mean().backward()
    assert module.bimanual_closure.prior_head.weight.grad is not None
    assert module.bimanual_closure.gate_head[-1].weight.grad is not None
    assert torch.isfinite(module.bimanual_closure.prior_head.weight.grad).all()
    # After filtering the pair-inclusive candidates, the published position
    # must still be exactly the four-way effective-candidate recurrence.
    w = out['gating_weights']
    expected = (w[..., 0:1]*out['pred_lhand_obj_trans']
                + w[..., 1:2]*out['pred_rhand_obj_trans']
                + w[..., 2:3]*out['pred_imu_obj_trans']
                + w[..., 3:4]*out['pred_static_obj_trans'])
    torch.testing.assert_close(out['pred_obj_trans'], expected, atol=1e-5, rtol=1e-5)
    module.eval()
    with torch.no_grad():
        full = _run(module, data, data['obj_trans_init'].unsqueeze(1),
                    data['obj_trans_init'].new_tensor([0., 0., 0., 1.]).view(1, 1, 4).expand(data['obj_trans_init'].shape[0], 1, 4))
        prefix = 8
        extra = {k: full[k][:, :prefix] for k in module.SEGMENT_STATE_OUTPUT_KEYS}
        replay = _run(module, data, full['pred_obj_trans'][:, :prefix],
                      full['gating_weights'][:, :prefix], known_segment_state_prefix=extra)
        torch.testing.assert_close(replay['pred_obj_trans'][:, prefix], full['pred_obj_trans'][:, prefix], atol=1e-5, rtol=1e-4)


def test_v3_rejects_biclose_but_preserves_filter_inheritance_guard():
    _, cfg = _module()
    with pytest.raises(ValueError, match='not supported by v3'):
        V3ObjectTransModule(cfg)
