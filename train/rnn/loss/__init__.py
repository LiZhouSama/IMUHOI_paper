"""
IMUHOI损失函数模块
"""
from .velocity_contact_loss import VelocityContactLoss, V2VelocityContactLoss
from .human_pose_loss import HumanPoseLoss
from .object_trans_loss import ObjectTransLoss
from .imuhoi_loss import IMUHOIVelocityContactLoss, IMUHOIObjectTransLoss

__all__ = [
    'VelocityContactLoss',
    'V2VelocityContactLoss',
    'HumanPoseLoss',
    'ObjectTransLoss',
    'IMUHOIVelocityContactLoss',
    'IMUHOIObjectTransLoss',
]
