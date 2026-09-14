"""Tests for the published-HOI objective and immutable initial position."""
import importlib.util
from pathlib import Path

import torch

from utils.hoi_metrics import compute_hoi_error_at_gt_contact

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / 'train_hoi_contact_residual.py'
spec = importlib.util.spec_from_file_location('train_hoi_contact_residual', MODULE_PATH)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_objective_matches_official_sequence_weighted_hoi():
    generator = torch.Generator().manual_seed(3)
    pred = torch.randn(2, 5, 3, generator=generator)
    batch = {
        'valid': torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        'contact': torch.tensor([[[1, 0], [1, 1], [0, 1], [0, 0], [1, 1]],
                                 [[0, 1], [1, 1], [1, 0], [1, 1], [1, 1]]], dtype=torch.bool),
        'pred_palm': torch.randn(2, 5, 2, 3, generator=generator),
        'target_obj': torch.randn(2, 5, 3, generator=generator),
        'target_palm': torch.randn(2, 5, 2, 3, generator=generator),
    }
    hoi, obj, valid = training.sequence_losses(pred, batch)
    assert valid.all()
    for i, n in enumerate((5, 3)):
        expected = compute_hoi_error_at_gt_contact(
            pred[i, :n], batch['target_obj'][i, :n], batch['pred_palm'][i, :n],
            batch['target_palm'][i, :n], batch['contact'][i, :n, 0], batch['contact'][i, :n, 1])
        assert abs(float(hoi[i]) * 100 - expected) < 5e-5
        expected_obj = torch.linalg.vector_norm(pred[i, :n] - batch['target_obj'][i, :n], dim=-1).mean()
        torch.testing.assert_close(obj[i], expected_obj)


def test_no_contact_and_padding_do_not_produce_nan():
    prediction = torch.zeros(2, 4, 3)
    batch = dict(valid=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool),
                 contact=torch.zeros(2, 4, 2, dtype=torch.bool),
                 pred_palm=torch.zeros(2, 4, 2, 3), target_obj=prediction.clone(),
                 target_palm=torch.zeros(2, 4, 2, 3))
    hoi, obj, has_contact = training.sequence_losses(prediction, batch)
    assert not has_contact.any()
    assert torch.isfinite(hoi).all() and torch.isfinite(obj).all()


class ConstantResidual(torch.nn.Module):
    def forward(self, features):
        return torch.ones(*features.shape[:2], 3) * 0.1, None


def test_initial_position_and_zero_contact_are_preserved():
    batch = dict(features=torch.ones(2, 4, 2), base_obj=torch.randn(2, 4, 3),
                 contact_prob=torch.ones(2, 4, 3),
                 obj_rotation=torch.eye(3).expand(2, 4, 3, 3))
    batch['contact_prob'][1] = 0
    output, correction = training.corrected_object(ConstantResidual(), batch)
    torch.testing.assert_close(output[:, 0], batch['base_obj'][:, 0])
    torch.testing.assert_close(output[1], batch['base_obj'][1])
    torch.testing.assert_close(correction[0, 1:], torch.full((3, 3), 0.1))
    identity, _ = training.corrected_object(ConstantResidual(), batch, gain=0)
    torch.testing.assert_close(identity, batch['base_obj'])


def test_digest_hashes_bytes_without_representation_conversion():
    import hashlib
    assert training.digest(b'abc') == hashlib.sha256(b'abc').hexdigest()
