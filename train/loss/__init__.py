"""
IMUHOI损失函数模块
"""
from .velocity_contact_loss import VelocityContactLoss
from .human_pose_loss import HumanPoseLoss
from .object_trans_loss import ObjectTransLoss

__all__ = [
    'VelocityContactLoss',
    'HumanPoseLoss',
    'ObjectTransLoss',
]

