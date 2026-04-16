"""
Loss for CondVQModule.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _chamfer_distance_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError(f"Chamfer inputs must be [B,N,3], got {tuple(pred.shape)} and {tuple(target.shape)}")
    dist = torch.cdist(pred, target, p=2)
    return dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean()


class CondVQLoss:
    LOSS_KEYS = ("pose_recon", "trans_recon", "point_recon", "codebook", "commit")
    TEST_LOSS_KEYS = ("pose_rmse", "trans_rmse_mm", "point_cd")

    _WEIGHT_ALIASES = {
        "pose_recon": ("pose_recon", "Loss_pose_recon", "simple_pose"),
        "trans_recon": ("trans_recon", "Loss_trans_recon", "simple_root_trans"),
        "point_recon": ("point_recon", "Loss_point_recon", "point_recon"),
        "codebook": ("codebook", "Loss_codebook", "Loss_code_cls"),
        "commit": ("commit", "Loss_commit", "commit"),
    }

    def __init__(self, weights=None, test_metric_weights=None, commit_beta: float = 0.25):
        self.weights = weights or {}
        self.test_metric_weights = test_metric_weights or {}
        self.commit_beta = float(commit_beta)

    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)

    def _weight(self, key: str, default: float = 1.0) -> float:
        for name in self._WEIGHT_ALIASES.get(key, (key,)):
            if name in self.weights:
                return float(self.weights[name])
        return float(default)

    def compute_loss(self, pred_dict, batch, device):
        targets = pred_dict.get("targets", {}) if isinstance(pred_dict, dict) else {}
        pose_target = targets.get("pose6d")
        trans_target = targets.get("trans")
        points_target = targets.get("obj_points_world_mid", targets.get("obj_points"))
        pose_pred = pred_dict.get("recon_pose6d") if isinstance(pred_dict, dict) else None
        trans_pred = pred_dict.get("recon_trans") if isinstance(pred_dict, dict) else None
        points_pred = pred_dict.get("recon_obj_points") if isinstance(pred_dict, dict) else None
        vq_aux = pred_dict.get("vq_aux", {}) if isinstance(pred_dict, dict) else {}

        ref = pose_pred if isinstance(pose_pred, torch.Tensor) else trans_pred
        if not isinstance(ref, torch.Tensor):
            raise ValueError("CondVQLoss expects recon_pose6d or recon_trans in pred_dict")
        dtype = ref.dtype
        zero = ref.new_tensor(0.0, device=device)

        losses = {key: zero.clone() for key in self.LOSS_KEYS}
        if isinstance(pose_pred, torch.Tensor) and isinstance(pose_target, torch.Tensor) and pose_pred.shape == pose_target.shape:
            losses["pose_recon"] = F.mse_loss(pose_pred.to(device=device, dtype=dtype), pose_target.to(device=device, dtype=dtype))
        if isinstance(trans_pred, torch.Tensor) and isinstance(trans_target, torch.Tensor) and trans_pred.shape == trans_target.shape:
            losses["trans_recon"] = F.mse_loss(trans_pred.to(device=device, dtype=dtype), trans_target.to(device=device, dtype=dtype))
        if isinstance(points_pred, torch.Tensor) and isinstance(points_target, torch.Tensor) and points_pred.shape == points_target.shape:
            losses["point_recon"] = _chamfer_distance_l2(
                points_pred.to(device=device, dtype=dtype),
                points_target.to(device=device, dtype=dtype),
            )

        codebook_loss = vq_aux.get("codebook_loss") if isinstance(vq_aux, dict) else None
        commit_loss = vq_aux.get("commit_loss") if isinstance(vq_aux, dict) else None
        if isinstance(codebook_loss, torch.Tensor):
            losses["codebook"] = codebook_loss.to(device=device, dtype=dtype)
        if isinstance(commit_loss, torch.Tensor):
            losses["commit"] = commit_loss.to(device=device, dtype=dtype)

        defaults = {
            "pose_recon": 1.0,
            "trans_recon": 1.0,
            "point_recon": 1.0,
            "codebook": 1.0,
            "commit": self.commit_beta,
        }
        total = zero.clone()
        weighted = {}
        for key in self.LOSS_KEYS:
            w = self._weight(key, defaults[key])
            weighted[key] = losses[key] * w
            total = total + weighted[key]
        return total, losses, weighted

    def compute_test_loss(self, pred_dict, batch, device):
        targets = pred_dict.get("targets", {}) if isinstance(pred_dict, dict) else {}
        pose_target = targets.get("pose6d")
        trans_target = targets.get("trans")
        points_target = targets.get("obj_points_world_mid", targets.get("obj_points"))
        pose_pred = pred_dict.get("recon_pose6d") if isinstance(pred_dict, dict) else None
        trans_pred = pred_dict.get("recon_trans") if isinstance(pred_dict, dict) else None
        points_pred = pred_dict.get("recon_obj_points") if isinstance(pred_dict, dict) else None

        ref = pose_pred if isinstance(pose_pred, torch.Tensor) else trans_pred
        if not isinstance(ref, torch.Tensor):
            raise ValueError("CondVQLoss expects recon_pose6d or recon_trans in pred_dict")
        dtype = ref.dtype
        zero = ref.new_tensor(0.0, device=device)
        metrics = {key: zero.clone() for key in self.TEST_LOSS_KEYS}

        if isinstance(pose_pred, torch.Tensor) and isinstance(pose_target, torch.Tensor) and pose_pred.shape == pose_target.shape:
            mse = F.mse_loss(pose_pred.to(device=device, dtype=dtype), pose_target.to(device=device, dtype=dtype))
            metrics["pose_rmse"] = torch.sqrt(torch.clamp(mse, min=0.0))
        if isinstance(trans_pred, torch.Tensor) and isinstance(trans_target, torch.Tensor) and trans_pred.shape == trans_target.shape:
            mse = F.mse_loss(trans_pred.to(device=device, dtype=dtype), trans_target.to(device=device, dtype=dtype))
            metrics["trans_rmse_mm"] = torch.sqrt(torch.clamp(mse, min=0.0)) * 1000.0
        if isinstance(points_pred, torch.Tensor) and isinstance(points_target, torch.Tensor) and points_pred.shape == points_target.shape:
            metrics["point_cd"] = _chamfer_distance_l2(
                points_pred.to(device=device, dtype=dtype),
                points_target.to(device=device, dtype=dtype),
            )

        defaults = {"pose_rmse": 1.0, "trans_rmse_mm": 1.0, "point_cd": 1.0}
        total = zero.clone()
        for key in self.TEST_LOSS_KEYS:
            total = total + metrics[key] * float(self.test_metric_weights.get(key, defaults[key]))
        return total, metrics
