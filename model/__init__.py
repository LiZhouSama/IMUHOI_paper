"""
Unified model entrypoint with runtime architecture selection.

Use ``cfg.model_arch`` (or env ``IMUHOI_MODEL_ARCH``) to switch between
the legacy RNN stack and the new DiT (Diffusion Transformer) stack under
``model/diffussion``. All exported callables keep the same signatures as
the RNN versions so existing training scripts continue to work.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from . import rnn as _rnn
from . import diffussion as _dit

ARCH_CHOICES = ("rnn", "dit")
_ARCH_ENV = os.environ.get("IMUHOI_MODEL_ARCH", "").lower()


def _resolve_arch(cfg: Optional[Any]) -> str:
    """Pick architecture from cfg.model_arch or IMUHOI_MODEL_ARCH env."""
    arch = None
    if cfg is not None:
        arch = getattr(cfg, "model_arch", None)
    if not arch and _ARCH_ENV:
        arch = _ARCH_ENV
    arch = (arch or "rnn").lower()
    if arch not in ARCH_CHOICES:
        arch = "rnn"
    return arch


def HumanPoseModule(cfg, *args, **kwargs):
    """Factory for HumanPoseModule (RNN or DiT)."""
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _dit.HumanPoseModule(cfg, *args, **kwargs)
    return _rnn.HumanPoseModule(cfg, *args, **kwargs)


def VelocityContactModule(cfg, *args, **kwargs):
    """Factory for VelocityContactModule (RNN or DiT)."""
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _dit.VelocityContactModule(cfg, *args, **kwargs)
    return _rnn.VelocityContactModule(cfg, *args, **kwargs)


def ObjectTransModule(cfg, *args, **kwargs):
    """Factory for ObjectTransModule (RNN or DiT)."""
    arch = _resolve_arch(cfg)
    if arch == "dit":
        ot_variant = getattr(cfg, "ot_variant", None)
        dit_cfg = getattr(cfg, "dit", {})
        if isinstance(dit_cfg, dict) and dit_cfg.get("ot_variant"):
            ot_variant = dit_cfg.get("ot_variant")
        if ot_variant and str(ot_variant).lower() == "fk_gating" and hasattr(_dit, "ObjectTransModuleFK"):
            return _dit.ObjectTransModuleFK(cfg, *args, **kwargs)
        return _dit.ObjectTransModule(cfg, *args, **kwargs)
    return _rnn.ObjectTransModule(cfg, *args, **kwargs)


def IMUHOIModel(cfg, *args, **kwargs):
    """Factory for unified IMUHOIModel."""
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _dit.IMUHOIModel(cfg, *args, **kwargs)
    return _rnn.IMUHOIModel(cfg, *args, **kwargs)


def load_model(
    config,
    device,
    no_trans: bool = False,
    module_paths: Optional[dict] = None,
):
    """Load full IMUHOI model according to selected architecture."""
    arch = _resolve_arch(config)
    if arch == "dit":
        return _dit.load_model(config, device, no_trans=no_trans, module_paths=module_paths)
    return _rnn.load_model(config, device, no_trans=no_trans, module_paths=module_paths)


__all__ = [
    "ARCH_CHOICES",
    "HumanPoseModule",
    "VelocityContactModule",
    "ObjectTransModule",
    "IMUHOIModel",
    "load_model",
]
