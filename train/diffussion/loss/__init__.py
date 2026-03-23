"""
IMUHOI loss modules for DiT path.
"""
from .human_pose_loss import HumanPoseLoss
from .interaction_loss import InteractionLoss
from .imuhoi_mix_loss import IMUHOIMixLoss

__all__ = [
    "HumanPoseLoss",
    "InteractionLoss",
    "IMUHOIMixLoss",
]
