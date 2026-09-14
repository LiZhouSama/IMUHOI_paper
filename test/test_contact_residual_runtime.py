"""Runtime postprocessing must match full-sequence training and keep history."""
from copy import deepcopy

import pytest
import torch

from model.rnn.contact_residual import CausalContactResidual, PointwiseContactResidual, build_contact_residual_features
from model.rnn.contact_residual_runtime import ContactResidualPredictor
import importlib.util
from pathlib import Path

_fixture_spec = importlib.util.spec_from_file_location('contact_residual_fixtures', Path(__file__).with_name('test_contact_residual.py'))
_fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixtures)
_record = _fixtures._record


def checkpoint(architecture, feature_dim=99, gain=0.75):
    torch.manual_seed(812)
    model = (CausalContactResidual(feature_dim, hidden_dim=16) if architecture == 'gru'
             else PointwiseContactResidual(feature_dim))
    with torch.no_grad():
        model.head.weight.normal_(0, 0.02)
        model.head.bias.fill_(0.01)
    return dict(schema_version=1, settings=dict(architecture=architecture, feature_dim=feature_dim,
                hidden_dim=16, max_correction=0.2), model_state_dict=model.state_dict(),
                feature_mean=torch.zeros(feature_dim), feature_std=torch.ones(feature_dim), gain=gain)


def chunk(record, start, end):
    return {k: v[start:end] if isinstance(v, torch.Tensor) and v.ndim else v for k, v in record.items()}


@pytest.mark.parametrize('architecture', ['pointwise', 'gru'])
def test_runtime_matches_full_and_chunked_training_formula(architecture):
    record = _record(141)
    ckpt = checkpoint(architecture)
    prepared = build_contact_residual_features(record)
    expected_model = ContactResidualPredictor(ckpt).model
    with torch.no_grad():
        delta, _ = expected_model(prepared['features'][None])
    contact = prepared['contact_prob'][:, :2].clamp(0, 1)
    gate = 1 - (1 - contact[:, 0]) * (1 - contact[:, 1])
    gate[0] = 0
    delta = delta[0] * (gate * ckpt['gain'])[:, None]
    expected = prepared['base_obj'] + (prepared['obj_rotation'] @ delta[..., None]).squeeze(-1)
    full_runtime = ContactResidualPredictor(ckpt)
    full = full_runtime.predict(record)
    torch.testing.assert_close(full, expected)
    stream = ContactResidualPredictor(ckpt)
    streamed = torch.cat([stream.predict(chunk(record, a, b)) for a, b in ((0, 17), (17, 60), (60, 121), (121, 141))])
    torch.testing.assert_close(streamed, full, rtol=1e-5, atol=2e-7)
    assert stream.frames_seen == 141
    stream.reset()
    torch.testing.assert_close(stream.predict(record), full)


def test_runtime_does_not_keep_gt_targets_and_zero_gain_is_identity():
    record = _record(20)
    runtime = ContactResidualPredictor(checkpoint('gru', gain=0))
    prediction = runtime.predict(record)
    torch.testing.assert_close(prediction, record['pred_obj_trans_published'], rtol=0, atol=0)
    assert {k for k in runtime.previous_observation if k.startswith('gt_')} == {'gt_obj_imu', 'gt_human_imu'}
    altered = deepcopy(record)
    for key, value in altered.items():
        if key.startswith('gt_') and key not in ('gt_obj_imu', 'gt_human_imu') and isinstance(value, torch.Tensor):
            value.fill_(123)
    torch.testing.assert_close(ContactResidualPredictor(checkpoint('gru')).predict(record),
                               ContactResidualPredictor(checkpoint('gru')).predict(altered))


def test_runtime_requires_published_output():
    record = _record(10, published=False)
    with pytest.raises(ValueError, match='publisher'):
        ContactResidualPredictor(checkpoint('gru')).predict(record)

@pytest.mark.parametrize('ablation,active', [('no_publication', 96), ('no_history', 75), ('no_extra_imu', 72)])
def test_checkpointed_ablation_matches_masked_training(ablation, active):
    from model.rnn.contact_residual import build_contact_residual_feature_mask
    mask = build_contact_residual_feature_mask(99, ablation)
    assert int(mask.sum()) == active
    record = _record(21)
    ckpt = checkpoint('gru')
    ckpt['settings']['feature_ablation'] = ablation
    runtime = ContactResidualPredictor(ckpt)
    prepared = build_contact_residual_features(record)
    with torch.no_grad():
        delta, _ = runtime.model((prepared['features'] * mask)[None])
    contact = prepared['contact_prob'][:, :2].clamp(0, 1)
    gate = 1 - (1 - contact[:, 0]) * (1 - contact[:, 1])
    gate[0] = 0
    expected = prepared['base_obj'] + (prepared['obj_rotation'] @
        (delta[0] * (gate * ckpt['gain'])[:, None])[..., None]).squeeze(-1)
    torch.testing.assert_close(runtime.predict(record), expected)
