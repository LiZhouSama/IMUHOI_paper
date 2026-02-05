"""
Diffusion Transformer (DiT) variants of the IMUHOI modules.

These modules mirror the public interfaces of the RNN stack so that
training scripts can switch architectures via ``cfg.model_arch`` or the
``IMUHOI_MODEL_ARCH`` environment variable without code changes.
"""
from .human_pose import HumanPoseModule
from .velocity_contact import VelocityContactModule
from .object_trans import ObjectTransModule
from .object_trans_fk import ObjectTransModuleFK
from .imuhoi_model import IMUHOIModel, load_model

__all__ = [
    "HumanPoseModule",
    "VelocityContactModule",
    "ObjectTransModule",
    "ObjectTransModuleFK",
    "IMUHOIModel",
    "load_model",
]
