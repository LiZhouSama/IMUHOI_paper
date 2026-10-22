"""Explicit bimanual closure for ObjectTrans (``object_trans_bimanual_closure``).

On frames where both hands hold the object the two per-hand FK proposals
``h_L + R b_L`` and ``h_R + R b_R`` are two estimates of one point; the current
model lets a gate average them (they disagree by ~10-12 cm on OMOMO bimanual
frames).  This module adds a third, *state-free* estimate built in the
hand-pair frame:

    P_B = m + R c,   m = (h_L + h_R) / 2,   c = object centre relative to the
                                            hand midpoint, object frame.

``c`` (common-mode grasp offset) is far less ambiguous than a per-hand bone
(train-set spread 15 cm vs 24.5 cm) and is regressed from hand-pair geometry
and IMU cues only -- never from the published object -- so it cannot inherit
object drift on long streams.  The fusion becomes five-way; the extra gate
logit is predicted from the predictor state plus two closure signals: the
disagreement of the two FK proposals (hand inconsistency) and the difference
between the observed common-mode offset ``R^T (p_{t-1} - m)`` and the prior
(object-state inconsistency).  Nothing is latched: hard latches of the
bimanual vector were shown to hurt, so the prior is a per-frame regression
with a recurrent encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from configs import _SENSOR_NAMES
from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix


class BimanualClosure(nn.Module):
    FEATURE_DIM = 43
    CONSISTENCY_DIM = 9
    MASKED_LOGIT = -1.0e4

    def __init__(self, predictor_hidden, hidden_dim=96, min_contact=0.2, gate_bias=-2.0, gate_hidden=64):
        super().__init__()
        self.min_contact = float(min_contact)
        self.arm_indices = [_SENSOR_NAMES.index(name) for name in ("LeftForeArm", "RightForeArm")]
        self.encoder = nn.Sequential(nn.LayerNorm(self.FEATURE_DIM), nn.Linear(self.FEATURE_DIM, hidden_dim), nn.GELU())
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.prior_head = nn.Linear(hidden_dim, 3)
        # Extra fusion logit for the bimanual candidate: predictor state, the
        # closure encoder state and the two consistency signals.
        self.gate_head = nn.Sequential(
            nn.LayerNorm(predictor_hidden + hidden_dim + self.CONSISTENCY_DIM),
            nn.Linear(predictor_hidden + hidden_dim + self.CONSISTENCY_DIM, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(gate_bias))

    @staticmethod
    def _temporal_difference(values):
        return torch.cat((torch.zeros_like(values[:, :1]), values[:, 1:] - values[:, :-1]), dim=1)

    @staticmethod
    def _object_frame(obj_rotm, vectors):
        """[B,T,3,3] x [B,T,...,3] world -> object frame."""
        return torch.einsum("btji,bt...j->bt...i", obj_rotm, vectors)

    def features(self, hands, root_trans, human_imu, obj_imu, contact, obj_rotm):
        """State-free per-frame features [B,T,43]."""
        root_rotation = rotation_6d_to_matrix(human_imu[:, :, 0, 3:9])
        arm_imu = human_imu[:, :, self.arm_indices]
        arm_rotation = root_rotation.unsqueeze(2) @ rotation_6d_to_matrix(arm_imu[..., 3:9])
        relative_rotation = obj_rotm.transpose(-1, -2).unsqueeze(2) @ arm_rotation
        relative_6d = matrix_to_rotation_6d(relative_rotation).flatten(2)  # 12
        span = self._object_frame(obj_rotm, hands[:, :, 1] - hands[:, :, 0])  # 3
        hands_rel_root = self._object_frame(obj_rotm, hands - root_trans.unsqueeze(2)).flatten(2)  # 6
        obj_acc = self._object_frame(obj_rotm, obj_imu[..., :3])  # 3
        arm_acc_world = torch.einsum("btij,bthj->bthi", root_rotation, arm_imu[..., :3])
        arm_acc = self._object_frame(obj_rotm, arm_acc_world).flatten(2)  # 6
        return torch.cat(
            (
                span,
                span.norm(dim=-1, keepdim=True),
                self._temporal_difference(span),
                hands_rel_root,
                obj_imu[..., 3:9],  # object orientation (tilt) 6
                relative_6d,
                obj_acc,
                arm_acc,
                contact[..., :3],
            ),
            dim=-1,
        )

    def forward(self, hands, root_trans, human_imu, obj_imu, contact, obj_rotm):
        """Encoder state [B,T,H], common-mode prior c [B,T,3] (object frame) and candidate [B,T,3]."""
        features = self.features(hands, root_trans, human_imu, obj_imu, contact, obj_rotm)
        encoded = self.encoder(features)
        if encoded.is_cuda and encoded.dtype == torch.bfloat16:
            # PyTorch 1.13/cu117 has no fused CUDA GRU kernel for BF16 (same
            # workaround as model/rnn/base.py for the LSTM cores).
            with torch.cuda.amp.autocast(enabled=False):
                hidden, _ = self.temporal(encoded.float())
        else:
            hidden, _ = self.temporal(encoded)
        prior_c = self.prior_head(hidden).to(hands.dtype)
        midpoint = hands.mean(dim=2)
        candidate = midpoint + torch.einsum("btij,btj->bti", obj_rotm, prior_c)
        return {"hidden": hidden.to(hands.dtype), "prior_c": prior_c, "candidate": candidate, "midpoint": midpoint}

    def consistency(self, obj_rotm, l_pos_fk, r_pos_fk, prev_pos, midpoint, prior_c, contact):
        """[B,9]: closure residual (3+1), observed-vs-prior common mode (3+1), min contact."""
        closure = torch.einsum("bji,bj->bi", obj_rotm, l_pos_fk - r_pos_fk)
        observed_c = torch.einsum("bji,bj->bi", obj_rotm, prev_pos - midpoint)
        innovation = observed_c - prior_c
        support = contact[..., :2].amin(dim=-1, keepdim=True)
        return torch.cat(
            (closure, closure.norm(dim=-1, keepdim=True), innovation, innovation.norm(dim=-1, keepdim=True), support),
            dim=-1,
        )

    def logit(self, predictor_hidden, closure_hidden, consistency, contact):
        """Bimanual fusion logit [B]; masked out unless both hands are in contact."""
        z = torch.cat((predictor_hidden, closure_hidden, consistency), dim=-1)
        out = self.gate_head(z).squeeze(-1).to(consistency.dtype)
        support = contact[..., :2].amin(dim=-1)
        return torch.where(support >= self.min_contact, out, torch.full_like(out, self.MASKED_LOGIT))

    @staticmethod
    def effective_candidates(left_fk, right_fk, pair, weights5, eps=1e-8):
        """Fold the pair candidate into two FK positions for exact four-way replay.

        Assign half of the pair mass to each hand. Translating both positions
        by the respective refined-palm deltas then translates the pair term by
        their mean, as the full five-way publication would.
        """
        half_pair = 0.5 * weights5[..., 4:5]
        left_mass = weights5[..., 0:1] + half_pair
        right_mass = weights5[..., 1:2] + half_pair
        left = (weights5[..., 0:1] * left_fk + half_pair * pair) / left_mass.clamp_min(eps)
        right = (weights5[..., 1:2] * right_fk + half_pair * pair) / right_mass.clamp_min(eps)
        return torch.where(left_mass > eps, left, left_fk), torch.where(right_mass > eps, right, right_fk)

    @staticmethod
    def effective_weights(weights5):
        """Fold the bimanual weight into the two hand branches (four-way view for feedback/publication)."""
        return torch.cat((weights5[..., :2] + 0.5 * weights5[..., 4:5], weights5[..., 2:4]), dim=-1)
