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