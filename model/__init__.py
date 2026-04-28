"""
Unified model entrypoint with runtime architecture selection.
"""
from __future__ import annotations

import os
from typing import Any, Optional

ARCH_CHOICES = ("rnn", "dit", "mamba")
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


def _arch_module(arch: str):
    if arch == "dit":
        from . import diffussion as module
    elif arch == "mamba":
        from . import mamba as module
    else:
        from . import rnn as module
    return module


def HumanPoseModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    return _arch_module(arch).HumanPoseModule(cfg, *args, **kwargs)


def InteractionModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _arch_module(arch).InteractionModule(cfg, *args, **kwargs)
    if arch == "mamba":
        return _arch_module(arch).InteractionModule(cfg, *args, **kwargs)
    raise RuntimeError("InteractionModule is only available for model_arch='dit'.")


def IMUHOIMixModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _arch_module(arch).IMUHOIMixModule(cfg, *args, **kwargs)
    if arch == "mamba":
        raise RuntimeError("IMUHOIMixModule is not implemented for model_arch='mamba' yet.")
    raise RuntimeError("IMUHOIMixModule is only available for model_arch='dit'.")


def VelocityContactModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        raise RuntimeError(
            "DiT path removed standalone VelocityContactModule. Use InteractionModule or IMUHOIModel instead."
        )
    if arch == "mamba":
        raise RuntimeError("VelocityContactModule is not implemented for model_arch='mamba' yet.")
    return _arch_module(arch).VelocityContactModule(cfg, *args, **kwargs)


def ObjectTransModule(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        raise RuntimeError(
            "DiT path removed standalone ObjectTransModule. Use InteractionModule or IMUHOIModel instead."
        )
    if arch == "mamba":
        raise RuntimeError("ObjectTransModule is not implemented for model_arch='mamba' yet.")
    return _arch_module(arch).ObjectTransModule(cfg, *args, **kwargs)


def IMUHOIModel(cfg, *args, **kwargs):
    arch = _resolve_arch(cfg)
    if arch == "dit":
        return _arch_module(arch).IMUHOIModel(cfg, *args, **kwargs)
    if arch == "mamba":
        return _arch_module(arch).IMUHOIModel(cfg, *args, **kwargs)
    return _arch_module(arch).IMUHOIModel(cfg, *args, **kwargs)


def load_model(
    config,
    device,
    no_trans: bool = False,
    module_paths: Optional[dict] = None,
):
    arch = _resolve_arch(config)
    return _arch_module(arch).load_model(config, device, no_trans=no_trans, module_paths=module_paths)


__all__ = [
    "ARCH_CHOICES",
    "HumanPoseModule",
    "InteractionModule",
    "IMUHOIMixModule",
    "VelocityContactModule",
    "ObjectTransModule",
    "IMUHOIModel",
    "load_model",
]
