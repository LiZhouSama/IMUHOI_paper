"""
IMUHOI loss modules for DiT path.
"""
from .cond_vq_loss import CondVQLoss
from .human_pose_loss import HumanPoseLoss
from .interaction_loss import InteractionLoss
from .interaction_vq_loss import InteractionVQLoss
from .imuhoi_mix_loss import IMUHOIMixLoss

__all__ = [
    "CondVQLoss",
    "HumanPoseLoss",
    "InteractionLoss",
    "InteractionVQLoss",
    "IMUHOIMixLoss",
]
