"""
Unified model entrypoint with runtime architecture selection.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from . import diffussion as _dit
from . import rnn as _rnn

ARCH_CHOICES = ("rnn", "dit")
_ARCH_ENV = os.environ.get("IMUHOI_MODEL_ARCH", "").lower()


def _resolve_arch(cfg: Optional[Any]) -> str:
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
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _dit.HumanPoseModule(cfg, *args, **kwargs)
    return _rnn.HumanPoseModule(cfg, *args, **kwargs)


def CondVQModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _dit.CondVQModule(cfg, *args, **kwargs)
    raise RuntimeError("CondVQModule is only available for model_arch='dit'.")


def InteractionModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _dit.InteractionModule(cfg, *args, **kwargs)
    raise RuntimeError("InteractionModule is only available for model_arch='dit'.")


def InteractionVQModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _dit.InteractionVQModule(cfg, *args, **kwargs)
    raise RuntimeError("InteractionVQModule is only available for model_arch='dit'.")


def IMUHOIMixModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _dit.IMUHOIMixModule(cfg, *args, **kwargs)
    raise RuntimeError("IMUHOIMixModule is only available for model_arch='dit'.")


def VelocityContactModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        raise RuntimeError(
            "DiT path removed standalone VelocityContactModule. Use InteractionModule or IMUHOIModel instead."
        )
    return _rnn.VelocityContactModule(cfg, *args, **kwargs)


def ObjectTransModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        raise RuntimeError(
            "DiT path removed standalone ObjectTransModule. Use InteractionModule or IMUHOIModel instead."
        )
    return _rnn.ObjectTransModule(cfg, *args, **kwargs)


def IMUHOIModel(cfg, *args, **kwargs):
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
    arch = _resolve_arch(config)
    if arch == "dit":
        return _dit.load_model(config, device, no_trans=no_trans, module_paths=module_paths)
    return _rnn.load_model(config, device, no_trans=no_trans, module_paths=module_paths)


__all__ = [
    "ARCH_CHOICES",
    "CondVQModule",
    "HumanPoseModule",
    "InteractionModule",
    "InteractionVQModule",
    "IMUHOIMixModule",
    "VelocityContactModule",
    "ObjectTransModule",
    "IMUHOIModel",
    "load_model",
]
