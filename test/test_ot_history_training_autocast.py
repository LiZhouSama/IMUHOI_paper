"""Regression: history training must give gradients to every OT head under autocast.

``forward_with_ot_history`` runs a no-grad warm-up window before the training
window.  Inside one autocast context the warm-up populated autocast's weight
cast cache with copies that carry no autograd edge, so the training window
reused them and every Linear/LayerNorm parameter of ObjectTrans silently got
``grad is None`` (only the fp32 LSTM cores trained).  The helper now clears the
cache after the warm-up.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rnn.object_trans import ObjectTransModule  # noqa: E402
from train.rnn.loss.object_trans_loss import ObjectTransLoss  # noqa: E402
from train.rnn.ot_history_training import forward_with_ot_history  # noqa: E402
from utils.utils import load_config  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs/IMUHOI_train_rnn.yaml")


def _run(with_autocast):
    # torch 1.13's CPU autocast rejects bf16 LayerNorm inputs, so the autocast
    # case (the trainer's real configuration) runs on CUDA.
    device = "cuda" if with_autocast else "cpu"
    if with_autocast and not torch.cuda.is_available():
        pytest.skip("autocast case needs CUDA")
    cfg = load_config(CONFIG)
    cfg.object_trans_state_feedback = True
    torch.manual_seed(0)
    module = ObjectTransModule(cfg).train().to(device)
    B, T = 2, 90
    g = torch.Generator().manual_seed(1)
    hands = torch.randn(B, T, 2, 3, generator=g) * 0.3
    prob = torch.rand(B, T, 3, generator=g)
    init = torch.randn(B, 3, generator=g) * 0.3
    obj_imu = torch.cat((torch.randn(B, T, 3, generator=g), torch.tensor([1.0, 0, 0, 0, 1, 0]).expand(B, T, 6)), -1)
    human_imu = torch.randn(B, T, 6, 9, generator=g)
    kwargs = dict(obj_imu=obj_imu, human_imu=human_imu, contact_init=torch.zeros(B, 3),
                  has_object_mask=torch.ones(B, dtype=torch.bool), enable_refine=False)
    batch = {"human_imu": human_imu, "obj_trans": torch.randn(B, T, 3, generator=g), "obj_vel": torch.zeros(B, T, 3),
             "has_object": torch.ones(B, dtype=torch.bool), "lhand_contact": torch.ones(B, T, dtype=torch.bool),
             "rhand_contact": torch.ones(B, T, dtype=torch.bool), "obj_contact": torch.ones(B, T, dtype=torch.bool)}
    loss_fn = ObjectTransLoss({"obj_trans": 1.0, "hoi_fused_l": 10.0, "hoi_fused_r": 10.0})
    kwargs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    batch = {k: v.to(device) for k, v in batch.items()}
    hands, prob, init = hands.to(device), prob.to(device), init.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=with_autocast):
        out = forward_with_ot_history(module, hands, prob, init, 60, **kwargs)
        total, _, _ = loss_fn(out, batch, device)
    total.float().backward()
    return module, out


@pytest.mark.parametrize("with_autocast", [False, True])
def test_history_training_gives_gradients_to_all_predictor_layers(with_autocast):
    module, out = _run(with_autocast)
    for name in ("object_trans_head.linear1.weight", "object_trans_head.linear2.weight",
                 "object_trans_head.rnn.weight_ih_l0", "feedback_embed.3.weight"):
        grad = dict(module.named_parameters())[name].grad
        assert grad is not None, f"{name} received no gradient (autocast={with_autocast})"
        assert torch.isfinite(grad).all() and float(grad.abs().sum()) > 0, name
    assert int(out["ot_loss_mask"].sum()) == 2 * 30  # only the last 30 of 90 frames are supervised


def test_autocast_cache_pitfall_is_real():
    """Document the pitfall: a no-grad forward inside autocast poisons the weight cache (CUDA)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA autocast")
    layer = torch.nn.Linear(4, 4).cuda()
    x = torch.randn(3, 4, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            layer(x)
        poisoned = layer(x)
    # poisoned cache: the cached bf16 weight copy has no autograd edge, so the
    # output is detached from the parameter (a whole-model loss would only see
    # the parameters that bypass autocast, e.g. fp32 LSTM cores).
    assert not poisoned.requires_grad
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            layer(x)
        torch.clear_autocast_cache()
        layer(x).float().sum().backward()
    assert layer.weight.grad is not None and float(layer.weight.grad.abs().sum()) > 0
