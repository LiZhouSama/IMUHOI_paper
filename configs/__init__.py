"""
IMUHOI配置模块 - 从IMUHOI_train.yaml加载所有配置
"""
import os
import yaml
import torch

# 加载配置文件
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'IMUHOI_train.yaml')

with open(_CONFIG_PATH, 'r') as f:
    _cfg = yaml.safe_load(f)

# 导出全局常量
FRAME_RATE = _cfg['FRAME_RATE']
_SENSOR_POS_INDICES = _cfg['SENSOR_POS_INDICES']
_SENSOR_ROT_INDICES = _cfg['SENSOR_ROT_INDICES']
_VEL_SELECTION_INDICES = torch.tensor(_cfg['VEL_SELECTION_INDICES'], dtype=torch.long)
_REDUCED_INDICES = _cfg['REDUCED_INDICES']
_IGNORED_INDICES = _cfg['IGNORED_INDICES']
_SENSOR_NAMES = _cfg['SENSOR_NAMES']
_SENSOR_VEL_NAMES = _cfg['SENSOR_VEL_NAMES']
_REDUCED_POSE_NAMES = _cfg['REDUCED_POSE_NAMES']

# 导出配置加载函数
def load_config(config_path: str = None):
    """加载配置文件"""
    path = config_path or _CONFIG_PATH
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    from easydict import EasyDict as edict
    return edict(cfg)

__all__ = [
    'FRAME_RATE',
    '_SENSOR_POS_INDICES',
    '_SENSOR_ROT_INDICES', 
    '_VEL_SELECTION_INDICES',
    '_REDUCED_INDICES',
    '_IGNORED_INDICES',
    '_SENSOR_NAMES',
    '_SENSOR_VEL_NAMES',
    '_REDUCED_POSE_NAMES',
    'load_config',
]

