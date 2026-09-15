"""State-independent bimanual proposals with separate memory and quality heads."""
from __future__ import annotations

import torch
from torch import nn

from configs import _SENSOR_NAMES
from utils.rotation_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix


class BimanualGrasp(nn.Module):
    FEATURE_DIM = 51

    def __init__(self, hidden_dim=96, gate_floor=0.05, contact_on=0.15, contact_off=0.075):
        super().__init__()
        self.gate_floor = float(gate_floor)
        self.contact_on = float(contact_on)
        self.contact_off = float(contact_off)
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("gate_floor must be in [0, 1)")
        if not 0.0 <= self.contact_off <= self.contact_on <= 1.0:
            raise ValueError("contact thresholds must satisfy 0 <= off <= on <= 1")
        self.arm_indices = [_SENSOR_NAMES.index(name) for name in ("LeftForeArm", "RightForeArm")]
        self.encoder = nn.Sequential(nn.LayerNorm(self.FEATURE_DIM), nn.Linear(self.FEATURE_DIM, hidden_dim), nn.GELU())
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.proposal_head = nn.Linear(hidden_dim, 3)
        self.update_head = nn.Sequential(nn.Linear(hidden_dim + 3, 32), nn.GELU(), nn.Linear(32, 1))
        self.quality_head = nn.Sequential(nn.LayerNorm(hidden_dim + 23), nn.Linear(hidden_dim + 23, 64), nn.GELU(), nn.Linear(64, 5))
        nn.init.constant_(self.update_head[-1].bias, -2.0)
        nn.init.zeros_(self.quality_head[-1].weight)
        nn.init.zeros_(self.quality_head[-1].bias)
        with torch.no_grad():
            self.quality_head[-1].bias[4] = -6.0

    @staticmethod
    def temporal_difference(values):
        return torch.cat((torch.zeros_like(values[:, :1]), values[:, 1:] - values[:, :-1]), dim=1)

    def features(self, hands, root_trans, human_imu, obj_imu, contact):
        obj_rotation = rotation_6d_to_matrix(obj_imu[..., 3:9])
        root_rotation = rotation_6d_to_matrix(human_imu[:, :, 0, 3:9])
        arm_imu = human_imu[:, :, self.arm_indices]
        arm_rotation = root_rotation.unsqueeze(2) @ rotation_6d_to_matrix(arm_imu[..., 3:9])
        relative_rotation = obj_rotation.transpose(-1, -2).unsqueeze(2) @ arm_rotation
        relative_6d = matrix_to_rotation_6d(relative_rotation).flatten(2)
        span = torch.einsum("btji,btj->bti", obj_rotation, hands[:, :, 1] - hands[:, :, 0])
        center = torch.einsum("btji,btj->bti", obj_rotation, hands.mean(dim=2) - root_trans)
        obj_acc = torch.einsum("btji,btj->bti", obj_rotation, obj_imu[..., :3])
        arm_acc_world = torch.einsum("btij,bthj->bthi", root_rotation, arm_imu[..., :3])
        arm_acc = torch.einsum("btji,bthj->bthi", obj_rotation, arm_acc_world).flatten(2)
        previous_rotation = torch.cat((obj_rotation[:, :1], obj_rotation[:, :-1]), dim=1)
        rotation_delta = matrix_to_rotation_6d(previous_rotation.transpose(-1, -2) @ obj_rotation)
        return torch.cat((span, self.temporal_difference(span), center, obj_acc, arm_acc,
                          relative_6d, self.temporal_difference(relative_6d), rotation_delta, contact[..., :3]), dim=-1)

    def forward(self, hands, root_trans, human_imu, obj_imu, contact):
        features = self.features(hands, root_trans, human_imu, obj_imu, contact)
        hidden, _ = self.temporal(self.encoder(features))
        return {"hidden": hidden, "proposal": self.proposal_head(hidden)}

    def update(self, hidden, proposal, previous, previous_active, contact):
        support = contact[..., :2].amin(dim=-1)
        active = torch.where(previous_active, support >= self.contact_off, support >= self.contact_on)
        update_input = torch.cat((hidden, proposal - previous), dim=-1)
        rate = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(self.update_head(update_input)).squeeze(-1)
        rate = torch.where(active & previous_active, rate, torch.ones_like(rate))
        vector = previous + rate.unsqueeze(-1) * (proposal - previous)
        return vector, active, rate

    def quality(self, hidden, candidates, previous, obj_rotation, base_weights, contact, active):
        innovation = torch.einsum("bji,bkj->bki", obj_rotation, candidates.detach() - previous.detach().unsqueeze(1))
        quality_input = torch.cat((hidden, innovation.flatten(1), innovation.norm(dim=-1), contact[..., :3]), dim=-1)
        residual_logits = self.quality_head(quality_input)
        logits = torch.cat((base_weights.clamp_min(1e-8).log(), torch.zeros_like(base_weights[:, :1])), dim=-1) + residual_logits
        support = contact[..., :2].amin(dim=-1).clamp(0.0, 1.0)
        logits = torch.cat((logits[:, :4], logits[:, 4:5] + support.clamp_min(1e-8).log().unsqueeze(-1)), dim=-1)
        baseline = torch.cat((base_weights, torch.zeros_like(base_weights[:, :1])), dim=-1)
        weights = torch.where(active.unsqueeze(-1), torch.softmax(logits, dim=-1), baseline)
        return weights, logits

    @staticmethod
    def feedback_weights(weights):
        return torch.cat((weights[..., :2] + weights[..., 4:5] * 0.5, weights[..., 2:4]), dim=-1)

    @classmethod
    def publication_candidates(cls, candidates, weights):
        """Expand the midpoint candidate equally into the two hand branches."""
        effective_weights = cls.feedback_weights(weights)
        share = 0.5 * weights[..., 4:5] / effective_weights[..., :2].clamp_min(1e-8)
        hands = candidates[..., :2, :] + share.unsqueeze(-1) * (candidates[..., 4:5, :] - candidates[..., :2, :])
        return hands, effective_weights