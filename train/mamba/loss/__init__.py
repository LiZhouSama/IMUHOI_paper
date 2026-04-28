"""Loss modules for the Mamba path."""
from .human_pose_loss import HumanPoseLoss
from .interaction_loss import InteractionLoss

__all__ = ["HumanPoseLoss", "InteractionLoss"]
