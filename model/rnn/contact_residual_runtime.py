"""Inference-only, chunk-continuous runtime for the learned HOI postprocessor.

This is AFTER the existing object publisher, not a replacement for OT feedback.
It never alters the human pose or feeds corrected positions upstream. Call reset
at a real sequence boundary; do not reset at sliding-window/chunk boundaries.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from .contact_residual import CausalContactResidual, PointwiseContactResidual, build_contact_residual_features, build_contact_residual_feature_mask


class ContactResidualPredictor:
    """Apply a locked checkpoint to observations, with no ground-truth targets."""

    def __init__(self, checkpoint, device='cpu'):
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, Mapping) or checkpoint.get('schema_version') != 1:
            raise ValueError('Expected a version-1 contact residual checkpoint.')
        self.device = torch.device(device)
        settings = checkpoint['settings']
        common = dict(feature_dim=settings['feature_dim'], max_correction=settings['max_correction'])
        if settings['architecture'] == 'pointwise':
            self.model = PointwiseContactResidual(**common)
        elif settings['architecture'] == 'gru':
            self.model = CausalContactResidual(**common, hidden_dim=settings['hidden_dim'], num_layers=1, dropout=0.0)
        else:
            raise ValueError('Unsupported residual architecture.')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device).eval()
        self.mean = checkpoint['feature_mean'].to(self.device)
        self.std = checkpoint['feature_std'].to(self.device)
        if self.mean.shape != (settings['feature_dim'],) or self.std.shape != self.mean.shape:
            raise ValueError('Checkpoint feature normalization has the wrong shape.')
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError('Invalid feature normalization.')
        self.feature_mask = build_contact_residual_feature_mask(settings['feature_dim'], settings.get('feature_ablation', 'full')).to(self.device)
        self.gain = float(checkpoint['gain'])
        if not 0 <= self.gain <= 1:
            raise ValueError('Locked gain must be between zero and one.')
        self.reset()

    def reset(self):
        self.state = None
        self.previous_observation = None
        self.frames_seen = 0

    @staticmethod
    def _observations(record):
        # Only two historical dump keys beginning with gt_ are measurements.
        # Ground-truth poses, object transforms and contact masks are excluded.
        return {k: v for k, v in record.items()
                if isinstance(v, torch.Tensor) and v.ndim > 0 and
                (k in ('gt_human_imu', 'gt_obj_imu', 'root_trans_pred', 'gating_weights')
                 or k.startswith('pred_') or k.startswith('stage1_'))}

    @torch.inference_mode()
    def predict(self, record):
        """Return final object positions [T,3]; input is a NEW contiguous chunk.

        Required published/object/palm/IMU fields match the feature builder.
        Do not pass overlapping chunks or the already-corrected output as the
        base. One last observation supplies exact backward-difference context.
        """
        if 'pred_obj_trans_published' not in record:
            raise ValueError('Apply the existing publisher before contact residual inference.')
        observations = self._observations(record)
        n = observations['pred_obj_trans_published'].shape[0]
        if n == 0:
            return observations['pred_obj_trans_published'].detach().to(self.device).clone()
        start = 0
        if self.previous_observation is not None:
            if observations.keys() != self.previous_observation.keys():
                raise ValueError('Observation fields must remain fixed within a streamed sequence.')
            observations = {k: torch.cat((self.previous_observation[k].to(v.device), v), dim=0)
                            for k, v in observations.items()}
            start = 1
        prepared = build_contact_residual_features(observations)
        features = prepared['features'][start:].to(self.device)
        features = ((features - self.mean) / self.std).clamp(-10, 10) * self.feature_mask
        delta, self.state = self.model(features.unsqueeze(0), self.state)
        delta = delta[0]
        contact = prepared['contact_prob'][start:, :2].to(self.device).clamp(0, 1)
        probability = 1 - (1 - contact[:, 0]) * (1 - contact[:, 1])
        if self.frames_seen == 0:
            probability[0] = 0
        delta = delta * (self.gain * probability).unsqueeze(-1)
        rotation = prepared['obj_rotation'][start:].to(self.device)
        output = prepared['base_obj'][start:].to(self.device) + (rotation @ delta.unsqueeze(-1)).squeeze(-1)
        self.previous_observation = {k: v[-1:].detach().clone() for k, v in observations.items()}
        self.frames_seen += n
        return output
