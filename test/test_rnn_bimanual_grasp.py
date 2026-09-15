import torch

from model.rnn.bimanual_grasp import BimanualGrasp


def inputs(frames=9):
    generator = torch.Generator().manual_seed(12)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return {
        "hands": torch.randn(2, frames, 2, 3, generator=generator),
        "root_trans": torch.randn(2, frames, 3, generator=generator),
        "human_imu": torch.cat((torch.randn(2, frames, 6, 3, generator=generator), identity.expand(2, frames, 6, 6)), dim=-1),
        "obj_imu": torch.cat((torch.randn(2, frames, 3, generator=generator), identity.expand(2, frames, 6)), dim=-1),
        "contact": torch.ones(2, frames, 3),
    }


def test_proposal_is_translation_invariant_and_causal():
    torch.manual_seed(0)
    module = BimanualGrasp(hidden_dim=16).eval()
    original = inputs()
    shifted = dict(original)
    shift = torch.tensor([0.3, 1.0, -0.5])
    shifted["hands"] = original["hands"] + shift
    shifted["root_trans"] = original["root_trans"] + shift
    with torch.no_grad():
        full = module(**original)
        translated = module(**shifted)
        prefix = module(**{key: value[:, :5] for key, value in original.items()})
    torch.testing.assert_close(full["proposal"], translated["proposal"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(full["proposal"][:, :5], prefix["proposal"], atol=1e-6, rtol=1e-6)


def test_memory_resets_on_onset_and_uses_hysteresis():
    module = BimanualGrasp(hidden_dim=16)
    hidden = torch.zeros(2, 16)
    proposal = torch.ones(2, 3)
    previous = torch.zeros(2, 3)
    active = torch.tensor([False, True])
    contact = torch.tensor([[0.2, 0.2, 1.0], [0.1, 0.1, 1.0]])
    vector, next_active, rate = module.update(hidden, proposal, previous, active, contact)
    torch.testing.assert_close(vector[0], proposal[0])
    assert next_active.all()
    assert 0.05 < float(rate[1]) < 1.0
    _, released, _ = module.update(hidden, proposal, vector, next_active, torch.zeros_like(contact))
    assert not released.any()


def test_single_hand_fallback_is_exact_and_weights_sum_to_one():
    module = BimanualGrasp(hidden_dim=16)
    base = torch.tensor([[0.6, 0.2, 0.1, 0.1], [0.2, 0.5, 0.2, 0.1]])
    candidates = torch.randn(2, 5, 3)
    weights, _ = module.quality(torch.zeros(2, 16), candidates, torch.zeros(2, 3),
                                 torch.eye(3).expand(2, 3, 3), base, torch.ones(2, 3), torch.tensor([False, True]))
    assert torch.equal(weights[0, :4], base[0])
    assert weights[0, 4] == 0
    assert weights[1, 4] > 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
    torch.testing.assert_close(module.feedback_weights(weights).sum(dim=-1), torch.ones(2))


def test_proposal_update_and_quality_receive_gradients():
    module = BimanualGrasp(hidden_dim=16)
    data = inputs()
    encoded = module(**data)
    hidden = encoded["hidden"][:, -1]
    proposal = encoded["proposal"][:, -1]
    vector, active, _ = module.update(hidden, proposal, torch.zeros_like(proposal), torch.ones(2, dtype=torch.bool), data["contact"][:, -1])
    candidates = torch.cat((torch.randn(2, 4, 3), vector.unsqueeze(1)), dim=1)
    weights, logits = module.quality(hidden, candidates, torch.zeros_like(proposal), torch.eye(3).expand(2, 3, 3),
                                     torch.full((2, 4), 0.25), data["contact"][:, -1], active)
    loss = proposal.square().mean() + vector.square().mean() + (weights.unsqueeze(-1) * candidates).sum(dim=1).square().mean() - logits[:, 4].mean()
    loss.backward()
    for head in (module.proposal_head, module.update_head[-1], module.quality_head[-1]):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0


def test_four_branch_publication_equals_five_candidates_after_hand_refinement():
    candidates = torch.randn(2, 5, 3)
    weights = torch.softmax(torch.randn(2, 5), dim=-1)
    hands, effective = BimanualGrasp.publication_candidates(candidates, weights)
    refined_delta = torch.randn(2, 2, 3)
    refined_candidates = candidates.clone()
    refined_candidates[:, :2] += refined_delta
    refined_candidates[:, 4] += refined_delta.mean(dim=1)
    expanded = torch.cat((hands + refined_delta, candidates[:, 2:4]), dim=1)
    expected = (refined_candidates * weights.unsqueeze(-1)).sum(dim=1)
    actual = (expanded * effective.unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(actual, expected)


def module_inputs(enabled=True):
    from easydict import EasyDict
    from model.rnn.object_trans import ObjectTransModule
    cfg = EasyDict(object_trans_arch="current", object_trans_bigrasp=enabled, bigrasp_hidden_dim=16,
                   object_trans_hidden_dim=24, object_trans_num_layers=1, object_trans_dropout=0.0,
                   object_trans_state_feedback=True, object_trans_static_gate="none")
    data = inputs(frames=7)
    arguments = dict(hand_positions=data["hands"], root_trans_input=data["root_trans"],
                     human_imu=data["human_imu"], obj_imu=data["obj_imu"],
                     pred_hand_contact_prob=data["contact"], obj_trans_init=torch.zeros(2, 3), enable_refine=False)
    return ObjectTransModule(cfg).eval(), arguments


def test_current_fusion_and_state_independent_proposal():
    module, arguments = module_inputs()
    original = module(**arguments)
    shifted = dict(arguments, obj_trans_init=torch.ones(2, 3), lb_prior_quantiles=torch.ones(2, 2, 3))
    changed = module(**shifted)
    torch.testing.assert_close(original["bigrasp_proposal_world"], changed["bigrasp_proposal_world"])
    direct = (original["bigrasp_candidates"] * original["bigrasp_weights"].unsqueeze(-1)).sum(dim=2)
    torch.testing.assert_close(original["pred_obj_trans"], direct)
    assert not torch.allclose(original["pred_obj_trans"], changed["pred_obj_trans"])


def test_published_grasp_prefix_is_authoritative():
    module, arguments = module_inputs()
    prefix = {"bigrasp_vector": torch.full((2, 5, 3), 0.8), "bigrasp_active": torch.ones(2, 5, 1)}
    result = module(**arguments, known_segment_state_prefix=prefix)
    torch.testing.assert_close(result["bigrasp_vector"][:, :5], prefix["bigrasp_vector"])
    assert result["bigrasp_vector"][:, 5].mean() > 0.2


def test_current_without_option_has_no_bimanual_outputs():
    module, arguments = module_inputs(enabled=False)
    result = module(**arguments)
    assert not any(key.startswith("bigrasp") for key in result)
    assert result["gating_weights"].shape == (2, 7, 4)


def test_grasp_target_uses_gt_midpoint_and_zero_default_weights():
    from train.rnn.loss.object_trans_loss import ObjectTransLoss
    module, arguments = module_inputs()
    result = module(**arguments)
    batch = {"human_imu": arguments["human_imu"], "obj_trans": torch.ones(2, 7, 3),
             "position_global": torch.zeros(2, 7, 24, 3), "lhand_contact": torch.ones(2, 7, dtype=torch.bool),
             "rhand_contact": torch.ones(2, 7, dtype=torch.bool)}
    result["bigrasp_vector_world"] = torch.ones(2, 7, 3)
    result["bigrasp_proposal_world"] = torch.ones(2, 7, 3)
    result["ot_hand_positions"] = torch.full((2, 7, 2, 3), 15.0)
    _, losses, weighted = ObjectTransLoss().compute_loss(result, batch, torch.device("cpu"))
    for name in ("bigrasp_common", "bigrasp_proposal", "bigrasp_delta"):
        assert losses[name] == 0
    assert weighted["bigrasp_quality"] == 0
    assert torch.isfinite(losses["bigrasp_quality"])
    batch["rhand_contact"][:] = False
    _, absent, _ = ObjectTransLoss().compute_loss(result, batch, torch.device("cpu"))
    assert all(absent[name] == 0 for name in ("bigrasp_common", "bigrasp_proposal", "bigrasp_delta", "bigrasp_quality"))


def test_auxiliary_losses_train_both_heads_without_training_quality_labels():
    from train.rnn.loss.object_trans_loss import ObjectTransLoss
    module, arguments = module_inputs()
    result = module(**arguments)
    result["bigrasp_candidates"].retain_grad()
    batch = {"human_imu": arguments["human_imu"], "obj_trans": torch.ones(2, 7, 3),
             "position_global": torch.zeros(2, 7, 24, 3), "lhand_contact": torch.ones(2, 7, dtype=torch.bool),
             "rhand_contact": torch.ones(2, 7, dtype=torch.bool)}
    _, losses, _ = ObjectTransLoss().compute_loss(result, batch, torch.device("cpu"))
    (losses["bigrasp_common"] + losses["bigrasp_proposal"] + losses["bigrasp_quality"]).backward()
    assert result["bigrasp_candidates"].grad is None
    assert module.bigrasp.proposal_head.weight.grad.abs().sum() > 0
    assert module.bigrasp.quality_head[-1].weight.grad.abs().sum() > 0


def test_overlapping_training_keeps_prediction_prefix_and_supervises_new_frames():
    from train.rnn.ot_history_training import forward_with_ot_history
    from train.rnn.loss.object_trans_loss import ObjectTransLoss
    for enabled in (False, True):
        module, arguments = module_inputs(enabled)
        hands = arguments.pop('hand_positions')
        contact = arguments.pop('pred_hand_contact_prob')
        obj_init = arguments.pop('obj_trans_init')
        result = forward_with_ot_history(module, hands, contact, obj_init, window=5, **arguments)
        assert result['pred_obj_trans'].shape == (2, 7, 3)
        assert not result['ot_loss_mask'][:, :5].any()
        assert result['ot_loss_mask'][:, 5:].all()
        batch = {'human_imu': arguments['human_imu'], 'obj_trans': result['pred_obj_trans'].detach().clone()}
        batch['obj_trans'][:, :5] += 100.0
        _, losses, _ = ObjectTransLoss().compute_loss(result, batch, torch.device('cpu'))
        assert losses['obj_trans'] == 0
        result['pred_obj_trans'][:, 5:].square().mean().backward()
        assert module.object_trans_head.linear2.weight.grad is not None