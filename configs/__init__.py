"""
IMUHOI shared constants and config loader.

Constants are defined here directly to avoid runtime mismatch caused by
binding to a single YAML file at import time.
"""
from __future__ import annotations

import os

import torch
import yaml

# Stable project constants
FRAME_RATE = 30
_SENSOR_POS_INDICES = [0, 7, 8, 15, 20, 21]
_SENSOR_ROT_INDICES = [0, 4, 5, 15, 18, 19]
_VEL_SELECTION_INDICES = torch.tensor([0, 7, 8, 15, 20, 21], dtype=torch.long)
_REDUCED_INDICES = [1, 2, 3, 6, 9, 12, 13, 14, 16, 17]
_IGNORED_INDICES = [7, 8, 10, 11, 20, 21, 22, 23]
_SENSOR_NAMES = ["Root", "LeftLowerLeg", "RightLowerLeg", "Head", "LeftForeArm", "RightForeArm"]
_SENSOR_VEL_NAMES = ["Root", "LeftFoot", "RightFoot", "Head", "LeftHand", "RightHand"]
_REDUCED_POSE_NAMES = [
    "LeftHip",
    "RightHip",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "LeftCollar",
    "RightCollar",
    "LeftShoulder",
    "RightShoulder",
]


def _default_config_path() -> str:
    env_path = os.environ.get("IMUHOI_CONFIG_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, "IMUHOI_train.yaml"),
        os.path.join(base_dir, "IMUHOI_fineTuning.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No IMUHOI config YAML found under configs/")


def load_config(config_path: str = None):
    """Load YAML config as EasyDict."""
    path = config_path or _default_config_path()
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    from easydict import EasyDict as edict

    return edict(cfg)


__all__ = [
    "FRAME_RATE",
    "_SENSOR_POS_INDICES",
    "_SENSOR_ROT_INDICES",
    "_VEL_SELECTION_INDICES",
    "_REDUCED_INDICES",
    "_IGNORED_INDICES",
    "_SENSOR_NAMES",
    "_SENSOR_VEL_NAMES",
    "_REDUCED_POSE_NAMES",
    "load_config",
]
