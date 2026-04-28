"""Mamba-style IMUHOI modules."""
from __future__ import annotations

import os
from typing import Optional

from .human_pose import HumanPoseModule


def load_model(config, device, no_trans: bool = False, module_paths: Optional[dict] = None):
    """Load the human pose module for the Mamba path."""
    model = HumanPoseModule(config, device, no_trans=no_trans)
    hp_path = None
    if isinstance(module_paths, dict):
        hp_path = module_paths.get("human_pose")
    if not hp_path:
        pretrained_modules = getattr(config, "pretrained_modules", None)
        if isinstance(pretrained_modules, dict):
            hp_path = pretrained_modules.get("human_pose")
        elif pretrained_modules is not None:
            hp_path = getattr(pretrained_modules, "human_pose", None)
    if hp_path and os.path.exists(hp_path):
        from utils.utils import load_checkpoint

        load_checkpoint(model, hp_path, device, strict=False)
    model = model.to(device)
    model.eval()
    return model


__all__ = ["HumanPoseModule", "load_model"]
