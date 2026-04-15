"""
Losses for VQ interaction stages.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .interaction_loss import InteractionLoss


class InteractionVQLoss:
    OBS_LOSS_KEYS = ("code_ce", "code_kl")
    OBS_TEST_KEYS = ("token_acc", "exact_match", "perplexity")

    _OBS_WEIGHT_ALIASES = {
        "code_ce": ("code_ce", "Loss_code_ce", "Loss_code_cls"),
        "code_kl": ("code_kl", "Loss_code_kl"),
    }

    def __init__(self, stage: str, weights=None, test_metric_weights=None):
        self.stage = str(stage).lower()
        self.weights = weights or {}
        self.test_metric_weights = test_metric_weights or {}

        if self.stage == "main_dit":
            base_weights = dict(self.weights)
            base_weights.setdefault("align", 0.0)
            base_weights.setdefault("Loss_align", 0.0)
            self.main_loss = InteractionLoss(weights=base_weights, test_metric_weights=self.test_metric_weights)
        elif self.stage != "obs_diffusion":
            raise ValueError(f"Unsupported stage for InteractionVQLoss: {stage}")

    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)

    def _obs_weight(self, key: str, default: float = 1.0) -> float:
        for name in self._OBS_WEIGHT_ALIASES.get(key, (key,)):
            if name in self.weights:
                return float(self.weights[name])
        return float(default)

    def _compute_obs_losses(self, pred_dict, device):
        logits = pred_dict.get("code_logits") if isinstance(pred_dict, dict) else None
        target = pred_dict.get("code_target") if isinstance(pred_dict, dict) else None
        q_posterior = pred_dict.get("q_posterior") if isinstance(pred_dict, dict) else None
        p_posterior = pred_dict.get("p_posterior") if isinstance(pred_dict, dict) else None
        sample_has_object = pred_dict.get("sample_has_object") if isinstance(pred_dict, dict) else None

        if not isinstance(logits, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("obs_diffusion loss expects code_logits and code_target in pred_dict")

        device = logits.device if device is None else device
        dtype = logits.dtype
        zero = logits.new_tensor(0.0)

        if isinstance(sample_has_object, torch.Tensor):
            valid = sample_has_object.to(device=device, dtype=torch.bool).view(-1, 1, 1)
        else:
            valid = torch.ones_like(target, dtype=torch.bool)
        valid_flat = valid.expand_as(target).reshape(-1)

        logits_flat = logits.reshape(-1, logits.shape[-1])
        target_flat = target.reshape(-1)

        losses = {key: zero.clone() for key in self.OBS_LOSS_KEYS}
        if valid_flat.any():
            losses["code_ce"] = F.cross_entropy(logits_flat[valid_flat], target_flat[valid_flat])

        if (
            isinstance(q_posterior, torch.Tensor)
            and isinstance(p_posterior, torch.Tensor)
            and q_posterior.shape == p_posterior.shape
        ):
            q_flat = q_posterior.reshape(-1, q_posterior.shape[-1]).to(device=device, dtype=dtype)
            p_flat = p_posterior.reshape(-1, p_posterior.shape[-1]).to(device=device, dtype=dtype)
            if valid_flat.any():
                kl = q_flat[valid_flat] * (
                    torch.log(q_flat[valid_flat].clamp_min(1e-8))
                    - torch.log(p_flat[valid_flat].clamp_min(1e-8))
                )
                losses["code_kl"] = kl.sum(dim=-1).mean()

        weighted = {}
        total = zero.clone()
        defaults = {"code_ce": 1.0, "code_kl": 1.0}
        for key in self.OBS_LOSS_KEYS:
            weighted[key] = losses[key] * self._obs_weight(key, defaults[key])
            total = total + weighted[key]
        return total, losses, weighted

    def compute_loss(self, pred_dict, batch, device):
        if self.stage == "main_dit":
            return self.main_loss.compute_loss(pred_dict, batch, device)
        return self._compute_obs_losses(pred_dict, device)

    def compute_test_loss(self, pred_dict, batch, device):
        if self.stage == "main_dit":
            return self.main_loss.compute_test_loss(pred_dict, batch, device)

        total, obs_losses, _ = self._compute_obs_losses(pred_dict, device)
        logits = pred_dict.get("code_logits") if isinstance(pred_dict, dict) else None
        target = pred_dict.get("code_target") if isinstance(pred_dict, dict) else None
        sample_has_object = pred_dict.get("sample_has_object") if isinstance(pred_dict, dict) else None
        if not isinstance(logits, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("obs_diffusion test loss expects code_logits and code_target in pred_dict")

        device = logits.device if device is None else device
        dtype = logits.dtype
        zero = logits.new_tensor(0.0)
        metrics = {key: zero.clone() for key in self.OBS_TEST_KEYS}

        pred = torch.argmax(logits, dim=-1)
        if isinstance(sample_has_object, torch.Tensor):
            valid = sample_has_object.to(device=device, dtype=torch.bool).view(-1, 1, 1)
        else:
            valid = torch.ones_like(target, dtype=torch.bool)
        valid_flat = valid.expand_as(target).reshape(-1)

        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        if valid_flat.any():
            metrics["token_acc"] = (pred_flat[valid_flat] == target_flat[valid_flat]).float().mean()

            pred_match = pred.eq(target)
            exact = pred_match.all(dim=-1).all(dim=-1)
            metrics["exact_match"] = exact[valid.squeeze(-1).squeeze(-1)].float().mean()

        probs = torch.softmax(logits, dim=-1)
        avg_probs = probs.mean(dim=(0, 1, 2))
        metrics["perplexity"] = torch.exp(-(avg_probs * torch.log(avg_probs.clamp_min(1e-8))).sum())

        return total, {
            "code_ce": obs_losses["code_ce"],
            "code_kl": obs_losses["code_kl"],
            **metrics,
        }
