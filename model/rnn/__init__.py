"""
IMUHOI模块化网络组件
"""
from .base import RNN, RNNWithInit, SubPoser
from .velocity_contact import (
    VELOCITY_CONTACT_ARCH_CHOICES,
    VelocityContactModule,
    build_velocity_contact_module,
    normalize_velocity_contact_arch,
)
from .velocity_contact_IMUHOI import IMUHOIVelocityContactModule
from .object_trans_IMUHOI import (
    IMUHOI_OBJECT_TRANS_ARCHES,
    IMUHOIObjectTransModule,
    is_imuhoi_object_trans_arch,
)
from .object_trans_IMUHOI_4heads import IMUHOI4HeadsObjectTransModule
from .velocity_contact_legacy import LegacyVelocityContactModule
from .velocity_contact_v2 import V2VelocityContactModule
from .velocity_contact_v3 import V3VelocityContactModule
from .velocity_contact_v4 import V4VelocityContactModule
from .human_pose import HumanPoseModule
from .object_trans import ObjectTransModule
from .object_trans_mergeHeads_obsEncoder import (
    ObjectTransModule as MergeHeadsObsObjectTransModule,
)
from .lb_vq_diffusion import LBVQAutoEncoder, LBLatentDiffusion
from .imuhoi_model import IMUHOIModel, load_model

__all__ = [
    'RNN',
    'RNNWithInit', 
    'SubPoser',
    'VelocityContactModule',
    'LegacyVelocityContactModule',
    'V2VelocityContactModule',
    'V3VelocityContactModule',
    'V4VelocityContactModule',
    'VELOCITY_CONTACT_ARCH_CHOICES',
    'build_velocity_contact_module',
    'normalize_velocity_contact_arch',
    'IMUHOIVelocityContactModule',
    'IMUHOIObjectTransModule',
    'IMUHOI4HeadsObjectTransModule',
    'IMUHOI_OBJECT_TRANS_ARCHES',
    'is_imuhoi_object_trans_arch',
    'HumanPoseModule',
    'ObjectTransModule',
    'MergeHeadsObsObjectTransModule',
    'LBVQAutoEncoder',
    'LBLatentDiffusion',
    'IMUHOIModel',
    'load_model',
]
