"""Bone-vector temporal consistency loss (lhand_bone_cons / rhand_bone_cons)."""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import FRAME_RATE
from train.rnn.loss.object_trans_loss import ObjectTransLoss


def _case(seed=0, bs=2, seq=6):
    torch.manual_seed(seed)
    dir_gt = torch.nn.functional.normalize(torch.randn(bs, seq, 3), dim=-1)
    lb_gt = torch.rand(bs, seq) * 0.3 + 0.2
    return dir_gt, lb_gt


def test_bone_cons_is_zero_when_prediction_follows_gt_delta_and_off_by_default():
    loss_fn = ObjectTransLoss({"lhand_bone_cons": 1.0, "rhand_bone_cons": 1.0})
    assert "lhand_bone_cons" in loss_fn._ZERO_DEFAULT_WEIGHT_KEYS
    dir_gt, lb_gt = _case()
    vec_gt = dir_gt * lb_gt.unsqueeze(-1)
    bias = torch.tensor([0.05, -0.02, 0.01])
    vec_pred = vec_gt + bias  # constant offset: same frame-to-frame delta as GT
    pred_lb = vec_pred.norm(dim=-1)
    pred_dir = vec_pred / pred_lb.unsqueeze(-1)
    mask = torch.ones(2, 6, dtype=torch.bool)
    losses = loss_fn._bone_consistency_terms(pred_dir, pred_lb, dir_gt, lb_gt, mask)
    assert float(losses) < 1e-10
    noisy = vec_gt + 0.02 * torch.randn_like(vec_gt)
    n_lb = noisy.norm(dim=-1)
    assert float(loss_fn._bone_consistency_terms(noisy / n_lb.unsqueeze(-1), n_lb, dir_gt, lb_gt, mask)) > 1e-4


def test_acc_smooth_constant_velocity_and_mask_boundaries():
    direction = torch.zeros(1, 7, 3)
    direction[..., 0] = 1
    length = (torch.arange(7).float() * .01 + .3).unsqueeze(0).requires_grad_()
    mask = torch.ones(1, 7, dtype=torch.bool)
    fn = ObjectTransLoss._bone_acc_smooth
    assert fn(direction, length, mask).item() < 1e-8
    noisy = length.detach().clone(); noisy[:, 3] += .02; noisy.requires_grad_()
    value = fn(direction, noisy, mask)
    assert value.item() > 0
    value.backward(); assert torch.isfinite(noisy.grad).all() and noisy.grad.abs().sum() > 0
    mask[:, 3] = False
    assert fn(direction, noisy, mask).item() < 1e-8
    assert fn(direction[:, :2], length[:, :2], mask[:, :2]).item() == 0
    assert fn(direction, noisy, torch.zeros_like(mask)).item() == 0
    assert 'lhand_bone_acc_smooth' in ObjectTransLoss._ZERO_DEFAULT_WEIGHT_KEYS
    assert 'lhand_bone_acc_smooth' not in ObjectTransLoss.MODEL_SELECTION_KEYS
