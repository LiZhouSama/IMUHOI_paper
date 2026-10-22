"""Cascade refine architecture (model/cascade): scan, VC-refine and publication invariants."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))
from test_rnn_ot_known_prefix_fast_path import _cfg, _inputs, _run  # noqa: E402
from eval_IMUHOI import _build_published_object_trans  # noqa: E402
from model.cascade.cascade_model import publication_chain  # noqa: E402
from model.cascade.ot_nofb import CascadeObjectTransModule  # noqa: E402
from model.cascade.vc_refine import VCRefineModule  # noqa: E402
from model.rnn.object_trans_v2 import V2ObjectTransModule  # noqa: E402
from train.cascade.train_vc_refine import corrupt_contact_logits  # noqa: E402


def _nofb_cfg(**overrides):
    cfg = _cfg(
        object_trans_state_feedback=False,
        object_trans_forearm_obj_rotation=True,
        object_trans_bimanual_closure=True,
        v2_fk_filter_alpha=.35,
        v2_fk_filter_beta=.05,
        enable_ot_refine=True,
    )
    cfg.update(overrides)
    return cfg


def _data(seed=3):
    data = _inputs(seed=seed)
    data["pred_hand_contact_prob"][:, 2:9] = 0.95  # bimanual stretch: the closure gate is active
    return data


def _pair(cfg):
    torch.manual_seed(0)
    cascade = CascadeObjectTransModule(cfg).eval()
    if cascade.bimanual_closure_enabled:
        # The gate's last layer starts at zero; give it weights so the logit
        # actually depends on the predictor hidden state under test.
        torch.nn.init.normal_(cascade.bimanual_closure.gate_head[-1].weight, std=0.5)
    stock = V2ObjectTransModule(cfg).eval()
    stock.load_state_dict(cascade.state_dict())
    return cascade, stock


def _init_prefix(data):
    batch = data["obj_trans_init"].shape[0]
    weights = data["obj_trans_init"].new_tensor([0., 0., 0., 1.]).view(1, 1, 4).expand(batch, 1, 4)
    return data["obj_trans_init"].unsqueeze(1), weights


def test_scan_matches_stepped_lstm_reference():
    """Each frame's closure gate must see that frame's hidden state, as when the LSTM is stepped."""
    cascade, stock = _pair(_nofb_cfg())
    data = _data()
    prefix_pos, prefix_weights = _init_prefix(data)
    with torch.no_grad():
        # A tensor prefix forces the stock scan onto its per-frame LSTM.step path.
        reference = _run(stock, data, prefix_pos, prefix_weights)
        ours = _run(cascade, data, prefix_pos, prefix_weights)
        # Training windows have no prefix.  A prefix whose mask is all False keeps
        # that semantics but still forces the stock scan to step the LSTM.
        unmasked = torch.zeros(prefix_pos.shape[:2], dtype=torch.bool)
        reference_offline = _run(stock, data, prefix_pos, prefix_weights, known_obj_trans_prefix_mask=unmasked)
        offline = cascade(data["hand_positions"], data["pred_hand_contact_prob"], data["obj_trans_init"],
                          obj_imu=data["obj_imu"], human_imu=data["human_imu"], obj_vel_input=data["obj_vel_input"],
                          contact_init=data["contact_init"], human_pose_input=data["human_pose_input"],
                          root_trans_input=data["root_trans_input"], enable_refine=False)
    for key in ("pred_obj_trans", "gating_weights", "bimanual_logit", "pred_lhand_obj_trans"):
        torch.testing.assert_close(ours[key], reference[key], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(offline[key], reference_offline[key], atol=1e-5, rtol=1e-4)


def test_stock_vectorized_nofb_path_hands_the_gate_the_initial_state():
    """Documents why the subclass exists: the stock no-feedback training path differs from stepping."""
    _, stock = _pair(_nofb_cfg())
    data = _data()
    prefix_pos, prefix_weights = _init_prefix(data)
    with torch.no_grad():
        stepped = _run(stock, data, prefix_pos, prefix_weights)
        vectorized = stock(data["hand_positions"], data["pred_hand_contact_prob"], data["obj_trans_init"],
                           obj_imu=data["obj_imu"], human_imu=data["human_imu"], obj_vel_input=data["obj_vel_input"],
                           contact_init=data["contact_init"], enable_refine=False,
                           human_pose_input=data["human_pose_input"], root_trans_input=data["root_trans_input"])
    assert not torch.allclose(vectorized["bimanual_logit"][:, 3:9], stepped["bimanual_logit"][:, 3:9])


def test_teacher_forced_prefix_skip_matches_sequential_teacher_forcing():
    cascade, _ = _pair(_nofb_cfg())
    data = _data()
    prefix_pos, prefix_weights = _init_prefix(data)
    with torch.no_grad():
        full = _run(cascade, data, prefix_pos, prefix_weights)
        prefix = 8
        segment = {key: full[key][:, :prefix] for key in cascade.SEGMENT_STATE_OUTPUT_KEYS}
        sequential = _run(cascade, data, full["pred_obj_trans"][:, :prefix], full["gating_weights"][:, :prefix],
                          known_segment_state_prefix=segment)
        skipped = _run(cascade, data, full["pred_obj_trans"][:, :prefix], full["gating_weights"][:, :prefix],
                       known_segment_state_prefix=segment, fast_known_prefix=True, scan_end=prefix + 1)
    for key in ("pred_obj_trans", "gating_weights", "pred_lhand_obj_trans", "v2_fk_filter_pos"):
        torch.testing.assert_close(skipped[key][:, prefix], sequential[key][:, prefix], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(skipped[key][:, prefix], full[key][:, prefix], atol=1e-5, rtol=1e-4)


def test_fk_candidates_are_the_unfiltered_per_hand_proposals():
    cfg = _nofb_cfg(object_trans_bimanual_closure=False, v2_fk_filter_alpha=1.0, v2_fk_filter_beta=0.0)
    cascade, _ = _pair(cfg)
    data = _data()
    prefix_pos, prefix_weights = _init_prefix(data)
    with torch.no_grad():
        out = _run(cascade, data, prefix_pos, prefix_weights)
    torch.testing.assert_close(out["cascade_fk_candidates"][:, 1:, 0], out["pred_lhand_obj_trans"][:, 1:], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out["cascade_fk_candidates"][:, 1:, 1], out["pred_rhand_obj_trans"][:, 1:], atol=1e-5, rtol=1e-4)


def test_vc_refine_starts_as_identity():
    cfg = _nofb_cfg()
    module = VCRefineModule(cfg).eval()
    batch, seq = 2, 12
    logits = torch.randn(batch, seq, 3) * 3
    out = module(logits, torch.randn(batch, seq, 3) * .1, torch.randn(batch, seq, 9), torch.randn(batch, seq, 2, 3),
                 torch.randn(batch, seq, 2, 3), torch.randn(batch, seq, 3), torch.randn(batch, 3),
                 bimanual_candidate=torch.randn(batch, seq, 3), static_gate=torch.zeros(batch, seq))
    move = torch.sigmoid(logits[..., 2:3])
    expected = torch.cat((move * torch.sigmoid(logits[..., :2]), move), dim=-1)
    torch.testing.assert_close(out["pred_hand_contact_prob"], expected)
    torch.testing.assert_close(out["pred_hand_contact_logits"], logits)


def test_publication_chain_matches_evaluator_replay():
    cascade, _ = _pair(_nofb_cfg())
    data = _data()
    with torch.no_grad():
        out = cascade(data["hand_positions"], data["pred_hand_contact_prob"], data["obj_trans_init"],
                      obj_imu=data["obj_imu"], human_imu=data["human_imu"], obj_vel_input=data["obj_vel_input"],
                      contact_init=data["contact_init"], human_pose_input=data["human_pose_input"],
                      root_trans_input=data["root_trans_input"], enable_refine=False)
    hands = data["hand_positions"]
    refined = hands + torch.randn_like(hands) * .02
    chain = publication_chain(out["pred_static_obj_trans"][:, 0], out["gating_weights"], out["pred_lhand_obj_trans"],
                              out["pred_rhand_obj_trans"], out["pred_imu_obj_trans"], out["pred_static_obj_trans"],
                              refined - hands)
    for row in range(hands.shape[0]):
        replay = _build_published_object_trans(
            out["pred_obj_trans"][row], hands[row], refined[row], out["pred_lhand_obj_trans"][row],
            out["pred_rhand_obj_trans"][row], out["pred_imu_obj_trans"][row], out["pred_static_obj_trans"][row],
            out["gating_weights"][row])
        torch.testing.assert_close(chain[row], replay, atol=1e-5, rtol=1e-4)
    # Without a hand correction the chain reproduces the fused trajectory itself.
    plain = publication_chain(out["pred_static_obj_trans"][:, 0], out["gating_weights"], out["pred_lhand_obj_trans"],
                              out["pred_rhand_obj_trans"], out["pred_imu_obj_trans"], out["pred_static_obj_trans"],
                              torch.zeros_like(hands))
    torch.testing.assert_close(plain, out["pred_obj_trans"], atol=1e-5, rtol=1e-4)


def test_corruption_writes_one_error_where_eligible():
    torch.manual_seed(5)
    batch, seq = 64, 60
    logits = torch.randn(batch, seq, 3)
    gt = torch.zeros(batch, seq, 3, dtype=torch.bool)
    gt[:, 20:40, 0] = True
    gt[:, 20:40, 2] = True
    out = corrupt_contact_logits(logits, gt, prob=1.0)
    changed = (out - logits).abs().sum(dim=(1, 2)) > 0
    # Only "missed segment" on the never-touching right hand has nowhere to go (1/6 of rows).
    assert changed.float().mean() > 0.7
    touched_hands = ((out[..., :2] - logits[..., :2]).abs().sum(dim=1) > 0).sum(dim=-1)
    assert (touched_hands <= 1).all()
    # Right-hand false segments over a static object must also raise the move logit there.
    raised_move = (out[..., 2] > logits[..., 2] + 1e-6) & ~gt[..., 2]
    assert raised_move.any()
