import os
import sys
import importlib.util

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BASE_PATH = os.path.join(PROJECT_ROOT, "model", "diffussion", "base.py")
_spec = importlib.util.spec_from_file_location("dit_base", BASE_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_mod)
ConditionalDiT = _mod.ConditionalDiT


def _make_model():
    return ConditionalDiT(
        target_dim=8,
        cond_dim=16,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=32,
        timesteps=100,
        use_time_embed=True,
    )


def test_sampling_timestep_subsample_descending_and_zero():
    model = _make_model()
    t = model._build_sampling_timesteps(steps=13)
    assert t[0].item() == model.timesteps - 1
    assert t[-1].item() == 0
    assert bool(torch.all(t[:-1] >= t[1:]))


def test_ddim_eta_zero_is_deterministic():
    torch.manual_seed(42)
    model = _make_model().eval()
    cond = torch.randn(2, 10, 16)
    x_seed = torch.randn(2, 10, 8)

    out1 = model.sample(cond=cond, x_start=x_seed.clone(), steps=20, sampler="ddim", eta=0.0)
    out2 = model.sample(cond=cond, x_start=x_seed.clone(), steps=20, sampler="ddim", eta=0.0)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_forward_returns_eps_aux_and_x0_shape():
    torch.manual_seed(0)
    model = _make_model().train()
    cond = torch.randn(4, 12, 16)
    x0 = torch.randn(4, 12, 8)

    x0_pred, aux = model(cond=cond, x_start=x0, add_noise=True)

    assert x0_pred.shape == x0.shape
    assert aux["eps_pred"].shape == x0.shape
    assert aux["noise"].shape == x0.shape
    assert aux["x_t"].shape == x0.shape
    assert aux["t"].shape == (4,)


def test_predict_x0_from_eps_inverse_relation():
    torch.manual_seed(1)
    model = _make_model().eval()

    x0 = torch.randn(2, 6, 8)
    noise = torch.randn_like(x0)
    t = torch.randint(0, model.timesteps, (2,), dtype=torch.long)

    x_t = model.q_sample(x0, t, noise=noise)
    x0_recon = model.predict_x0_from_eps(x_t, t, noise)

    assert torch.allclose(x0, x0_recon, atol=1e-5)
