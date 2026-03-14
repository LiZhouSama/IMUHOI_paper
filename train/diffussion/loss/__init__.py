"""
IMUHOI loss modules for DiT path.
"""
from .human_pose_loss import HumanPoseLoss
from .interaction_loss import InteractionLoss

__all__ = [
    "HumanPoseLoss",
    "InteractionLoss",
]
