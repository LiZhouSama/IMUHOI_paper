"""Mamba-style IMUHOI modules."""
from __future__ import annotations

import os
from typing import Optional

from .human_pose import HumanPoseModule
from .interaction import InteractionModule
from .imuhoi_model import IMUHOIModel, _load_checkpoint


def _get_pretrained_modules(config, module_paths: Optional[dict] = None):
    if isinstance(module_paths, dict):
        return module_paths
    pretrained_modules = getattr(config, "pretrained_modules", None)
    if isinstance(pretrained_modules, dict):
        return pretrained_modules
    if pretrained_modules is not None:
        out = {}
        for key in ("human_pose", "interaction", "velocity_contact", "object_trans"):
            value = getattr(pretrained_modules, key, None)
            if value:
                out[key] = value
        return out
    return {}


def _interaction_enabled(config, module_paths: Optional[dict] = None) -> bool:
    modules = _get_pretrained_modules(config, module_paths)
    if any(modules.get(key) for key in ("interaction", "velocity_contact", "object_trans")):
        return True
    mamba_cfg = getattr(config, "mamba", {})
    interaction_cfg = mamba_cfg.get("interaction", {}) if isinstance(mamba_cfg, dict) else getattr(mamba_cfg, "interaction", {})
    for container in (interaction_cfg, mamba_cfg, getattr(config, "mamba_interaction", {})):
        if isinstance(container, dict) and "enabled" in container:
            return bool(container["enabled"])
        if hasattr(container, "enabled"):
            return bool(getattr(container, "enabled"))
    return bool(getattr(config, "mamba_enable_interaction", False))


def load_model(config, device, no_trans: bool = False, module_paths: Optional[dict] = None):
    """Load a Mamba human-only model or full human+interaction pipeline."""
    if _interaction_enabled(config, module_paths):
        return _load_pipeline(config, device, no_trans, module_paths)

    model = HumanPoseModule(config, device, no_trans=no_trans)
    modules = _get_pretrained_modules(config, module_paths)
    hp_path = modules.get("human_pose")
    if hp_path and os.path.exists(hp_path):
        _load_checkpoint(model, hp_path, device, strict=False)
    model = model.to(device)
    model.eval()
    return model


def _load_pipeline(config, device, no_trans: bool = False, module_paths: Optional[dict] = None):
    from .imuhoi_model import load_model as _load

    return _load(config, device, no_trans=no_trans, module_paths=module_paths)


__all__ = ["HumanPoseModule", "InteractionModule", "IMUHOIModel", "load_model"]
