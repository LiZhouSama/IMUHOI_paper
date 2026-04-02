"""
通用工具函数
"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from pytorch3d.transforms import matrix_to_rotation_6d

from configs import (
    _SENSOR_POS_INDICES,
    _SENSOR_VEL_NAMES,
    _REDUCED_POSE_NAMES,
)


# ============ 配置和设备 ============

def load_config(config_path: str) -> edict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return edict(cfg)


def setup_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device(cfg):
    """设置计算设备"""
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        print(f"可用GPU数量: {available_gpus}")
        
        if hasattr(cfg, 'gpus') and cfg.gpus:
            valid_gpus = [gpu for gpu in cfg.gpus if gpu < available_gpus]
            if len(valid_gpus) != len(cfg.gpus):
                print(f"警告: 配置的GPU {cfg.gpus} 中部分不可用，使用可用GPU: {valid_gpus}")
                cfg.gpus = valid_gpus
            cfg.use_multi_gpu = getattr(cfg, 'use_multi_gpu', True) and len(cfg.gpus) > 1
            cfg.device = f"cuda:{cfg.gpus[0]}"
            torch.cuda.set_device(cfg.gpus[0])
        else:
            cfg.gpus = [0]
            cfg.use_multi_gpu = False
            cfg.device = "cuda:0"
    else:
        print("CUDA不可用，使用CPU")
        cfg.device = "cpu"
        cfg.use_multi_gpu = False
    
    return cfg


# ============ 模型加载 ============

def load_smpl_model(smpl_path: str, device: torch.device):
    """加载SMPL模型"""
    from human_body_prior.body_model.body_model import BodyModel
    
    if not os.path.exists(smpl_path):
        raise FileNotFoundError(f"SMPL model not found at {smpl_path}")
    print(f"Loading SMPL model from: {smpl_path}")
    smpl_model = BodyModel(bm_fname=smpl_path, num_betas=16)
    return smpl_model.to(device)


def load_checkpoint(model, checkpoint_path, device, strict=True, use_ema=True):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if use_ema and checkpoint.get('ema_state_dict') is not None:
            state_dict = checkpoint['ema_state_dict']
        else:
            state_dict = checkpoint.get('module_state_dict', checkpoint.get('model_state_dict', checkpoint))
    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        model_state = model.state_dict()
        filtered_state = {}
        skipped_shape = []
        for key, value in state_dict.items():
            mapped_key = key
            if mapped_key not in model_state and mapped_key.startswith('module.') and mapped_key[7:] in model_state:
                mapped_key = mapped_key[7:]
            elif mapped_key not in model_state and f'module.{mapped_key}' in model_state:
                mapped_key = f'module.{mapped_key}'

            if mapped_key not in model_state:
                continue
            if model_state[mapped_key].shape != value.shape:
                skipped_shape.append((key, tuple(value.shape), tuple(model_state[mapped_key].shape)))
                continue
            filtered_state[mapped_key] = value

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        if skipped_shape:
            print(f"加载检查点时跳过 {len(skipped_shape)} 个shape不匹配参数")
        if missing:
            print(f"加载检查点缺失参数数量: {len(missing)}")
        if unexpected:
            print(f"加载检查点多余参数数量: {len(unexpected)}")
    print(f"加载检查点: {checkpoint_path}")
    return checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0


def save_checkpoint(model, optimizer, epoch, save_path, loss, additional_info=None):
    """保存检查点"""
    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'module_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss,
    }
    
    if additional_info:
        checkpoint_data.update(additional_info)
    
    torch.save(checkpoint_data, save_path)
    print(f"保存检查点: {save_path}")


def flatten_lstm_parameters(module):
    """递归调用所有LSTM模块的flatten_parameters()"""
    for child in module.children():
        if isinstance(child, torch.nn.LSTM):
            child.flatten_parameters()
        else:
            flatten_lstm_parameters(child)


# ============ 数据处理 ============
def central_diff(tensor: torch.Tensor, dt: float) -> torch.Tensor:
    """使用中心差分计算一阶导数，支持任意最后一维"""
    if tensor.shape[0] <= 1:
        return torch.zeros_like(tensor)
    vel = torch.zeros_like(tensor)
    vel[1:-1] = (tensor[2:] - tensor[:-2]) / (2.0 * dt)
    vel[0] = (tensor[1] - tensor[0]) / dt
    vel[-1] = (tensor[-1] - tensor[-2]) / dt
    return vel


def smooth_acceleration(position: torch.Tensor, fps: float, smooth_n: int = 4) -> torch.Tensor:
    """
    平滑二阶差分（TransPose风格）以计算加速度。
    position: [T, ..., 3]
    """
    if position.shape[0] <= 2:
        return torch.zeros_like(position)

    original_shape = position.shape
    pos_flat = position.reshape(position.shape[0], -1)

    mid = smooth_n // 2
    fps_squared = fps * fps

    acc = torch.stack([
        (pos_flat[i] + pos_flat[i + 2] - 2 * pos_flat[i + 1]) * fps_squared
        for i in range(0, pos_flat.shape[0] - 2)
    ])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))

    if mid != 0 and acc.shape[0] > smooth_n * 2:
        acc[smooth_n:-smooth_n] = torch.stack([
            (pos_flat[i] + pos_flat[i + smooth_n * 2] - 2 * pos_flat[i + smooth_n]) * fps_squared / (smooth_n ** 2)
            for i in range(0, pos_flat.shape[0] - smooth_n * 2)
        ])

    acc = acc.reshape(original_shape)
    return acc

# backward-compatible aliases
_central_diff = central_diff
_smooth_acceleration = smooth_acceleration

def build_model_input_dict(batch, cfg, device, add_noise=True):
    """
    构建模型输入字典
    
    Args:
        batch: 数据批次
        cfg: 配置对象
        device: 计算设备
        add_noise: 是否添加噪声
    
    Returns:
        dict: 模型输入字典
    """
    human_imu = batch['human_imu'].to(device)
    bs, seq = human_imu.shape[:2]
    dtype = human_imu.dtype

    skip_extra_noise = batch.get('imu_noise_applied', False)
    if isinstance(skip_extra_noise, torch.Tensor):
        skip_mask = skip_extra_noise.to(device=device, dtype=torch.bool).flatten()
    elif isinstance(skip_extra_noise, (list, tuple, np.ndarray)):
        skip_mask = torch.as_tensor(skip_extra_noise, device=device, dtype=torch.bool).flatten()
    else:
        skip_mask = torch.tensor([bool(skip_extra_noise)], device=device, dtype=torch.bool)
    if skip_mask.numel() == 1 and bs > 1:
        skip_mask = skip_mask.expand(bs)
    elif skip_mask.numel() != bs:
        skip_mask = skip_mask[:1].expand(bs)
    skip_mask_human = skip_mask.view(bs, *([1] * (human_imu.dim() - 1)))
    
    # 添加噪声
    imu_noise_std = getattr(cfg, 'imu_noise_std', 0.1)
    if add_noise and imu_noise_std > 0:
        human_noise = torch.randn_like(human_imu) * imu_noise_std
        human_imu = torch.where(skip_mask_human, human_imu, human_imu + human_noise)
    
    obj_imu = batch.get('obj_imu')
    if isinstance(obj_imu, torch.Tensor):
        obj_imu = obj_imu.to(device)
        obj_noise_std = getattr(cfg, 'obj_imu_noise_std', 0.1)
        if add_noise and obj_noise_std > 0:
            skip_mask_obj = skip_mask.view(bs, *([1] * (obj_imu.dim() - 1)))
            obj_noise = torch.randn_like(obj_imu) * obj_noise_std
            obj_imu = torch.where(skip_mask_obj, obj_imu, obj_imu + obj_noise)
    else:
        obj_feat_dim = getattr(cfg, 'obj_imu_dim', 9)
        obj_imu = torch.zeros(bs, seq, obj_feat_dim, device=device, dtype=dtype)
    
    def _get_tensor(key, tensor_dtype=dtype):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            value = value.to(device=device)
            if tensor_dtype is not None and value.dtype != tensor_dtype:
                value = value.to(dtype=tensor_dtype)
            return value
        return None
    
    def _ensure_bt(tensor, default_shape, fill_dtype):
        if tensor is None:
            return torch.zeros(*default_shape, device=device, dtype=fill_dtype)
        tensor = tensor.to(device=device)
        if tensor.dim() == len(default_shape) - 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and bs > 1:
            tensor = tensor.expand(bs, *tensor.shape[1:])
        if tensor.dtype != fill_dtype:
            tensor = tensor.to(dtype=fill_dtype)
        return tensor
    
    sensor_vel_root = _ensure_bt(_get_tensor('sensor_vel_root'), (bs, seq, len(_SENSOR_VEL_NAMES), 3), dtype)
    sensor_vel_glb = _ensure_bt(_get_tensor('sensor_vel_glb'), (bs, seq, len(_SENSOR_POS_INDICES), 3), dtype)
    obj_vel = _ensure_bt(_get_tensor('obj_vel'), (bs, seq, 3), dtype)
    trans = _ensure_bt(_get_tensor('trans'), (bs, seq, 3), dtype)
    obj_trans = _ensure_bt(_get_tensor('obj_trans'), (bs, seq, 3), dtype)
    
    v_init = sensor_vel_root[:, 0]
    hand_indices = [len(_SENSOR_POS_INDICES) - 2, len(_SENSOR_POS_INDICES) - 1]
    hand_vel_glb_init = sensor_vel_glb[:, 0, hand_indices, :]
    obj_vel_init = obj_vel[:, 0, :]
    trans_init = trans[:, 0, :]
    obj_trans_init = obj_trans[:, 0, :]
    
    # 初始姿态
    ori_root_reduced_val = _get_tensor('ori_root_reduced')
    if isinstance(ori_root_reduced_val, torch.Tensor):
        ori_root_reduced = ori_root_reduced_val
        if ori_root_reduced.dim() == 4:
            ori_root_reduced = ori_root_reduced.unsqueeze(0)
        if ori_root_reduced.shape[0] == 1 and bs > 1:
            ori_root_reduced = ori_root_reduced.expand(bs, *ori_root_reduced.shape[1:])
        if ori_root_reduced.shape[0] != bs:
            ori_root_reduced = ori_root_reduced.reshape(bs, seq, len(_REDUCED_POSE_NAMES), 3, 3)
        p_init = matrix_to_rotation_6d(
            ori_root_reduced[:, 0].reshape(bs * len(_REDUCED_POSE_NAMES), 3, 3)
        ).reshape(bs, len(_REDUCED_POSE_NAMES), 6)
    else:
        p_init = torch.zeros(bs, len(_REDUCED_POSE_NAMES), 6, device=device, dtype=dtype)
    
    # 接触状态
    lhand_contact = _ensure_bt(_get_tensor('lhand_contact', tensor_dtype=torch.bool), (bs, seq), torch.bool)
    rhand_contact = _ensure_bt(_get_tensor('rhand_contact', tensor_dtype=torch.bool), (bs, seq), torch.bool)
    obj_contact = _ensure_bt(_get_tensor('obj_contact', tensor_dtype=torch.bool), (bs, seq), torch.bool)
    
    contact_first = torch.stack([
        lhand_contact[:, 0].float().to(device=device, dtype=dtype),
        rhand_contact[:, 0].float().to(device=device, dtype=dtype),
        obj_contact[:, 0].float().to(device=device, dtype=dtype),
    ], dim=-1)
    contact_init = contact_first
    
    # has_object
    def _prepare_has_object(value):
        if value is None:
            return torch.ones(bs, dtype=torch.bool, device=device)
        if isinstance(value, torch.Tensor):
            value = value.to(device=device, dtype=torch.bool)
            if value.dim() == 0:
                value = value.view(1)
            if value.shape[0] == 1 and bs > 1:
                value = value.expand(bs)
            return value
        if isinstance(value, (bool, int)):
            return torch.tensor([bool(value)], dtype=torch.bool, device=device).expand(bs)
        value = torch.as_tensor(value, dtype=torch.bool, device=device)
        if value.dim() == 0:
            value = value.view(1)
        if value.shape[0] == 1 and bs > 1:
            value = value.expand(bs)
        return value
    
    has_object = _prepare_has_object(batch.get('has_object'))
    obj_points_count = int(getattr(cfg, "mesh_downsample_points", 256))
    obj_points_canonical_val = batch.get('obj_points_canonical')
    if isinstance(obj_points_canonical_val, torch.Tensor):
        obj_points_canonical = obj_points_canonical_val.to(device=device, dtype=dtype)
        if obj_points_canonical.dim() == 2:
            obj_points_canonical = obj_points_canonical.unsqueeze(0)
        if obj_points_canonical.shape[0] == 1 and bs > 1:
            obj_points_canonical = obj_points_canonical.expand(bs, -1, -1)
        if obj_points_canonical.shape[0] != bs or obj_points_canonical.shape[-1] != 3:
            obj_points_canonical = torch.zeros(bs, obj_points_count, 3, device=device, dtype=dtype)
    else:
        obj_points_canonical = torch.zeros(bs, obj_points_count, 3, device=device, dtype=dtype)
    
    return {
        'human_imu': human_imu,
        'obj_imu': obj_imu,
        'v_init': v_init,
        'p_init': p_init,
        'trans_init': trans_init,
        'trans_gt': trans,
        'obj_trans_init': obj_trans_init,
        'obj_vel_init': obj_vel_init,
        'hand_vel_glb_init': hand_vel_glb_init,
        'contact_init': contact_init,
        'has_object': has_object,
        # object metadata for mesh-prior teacher/cache path
        'obj_name': batch.get('obj_name'),
        'obj_points_canonical': obj_points_canonical,
        'seq_file': batch.get('seq_file'),
        'window_start': batch.get('window_start'),
        'window_end': batch.get('window_end'),
    }


# ============ 数学工具 ============

def tensor2numpy(tensor):
    """将PyTorch张量转换为NumPy数组"""
    return tensor.detach().cpu().numpy()


def _aa_to_R(a: torch.Tensor) -> torch.Tensor:
    """Axis-angle -> rotation matrix. a: [..., 3] -> [..., 3, 3]"""
    orig_shape = a.shape
    a = a.view(-1, 3)
    angle = torch.norm(a, dim=1, keepdim=True)
    axis = torch.where(angle > 1e-8, a / angle, torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype).expand_as(a))
    x, y, z = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c
    R = torch.stack([
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C
    ], dim=1).view(-1, 3, 3)
    return R.view(*orig_shape[:-1], 3, 3)


def _R_to_aa(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> axis-angle. R: [..., 3, 3] -> [..., 3]"""
    orig_shape = R.shape
    R = R.view(-1, 3, 3)
    trace = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp(-1.0, 3.0)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    eps = 1e-8
    sin_theta = torch.sin(theta)
    rx = (R[:, 2, 1] - R[:, 1, 2]) / (2.0 * torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta))
    ry = (R[:, 0, 2] - R[:, 2, 0]) / (2.0 * torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta))
    rz = (R[:, 1, 0] - R[:, 0, 1]) / (2.0 * torch.where(sin_theta.abs() < eps, torch.ones_like(sin_theta), sin_theta))
    axis = torch.stack([rx, ry, rz], dim=1)
    aa = axis * theta.unsqueeze(1)
    small = theta.abs() < eps
    if small.any():
        aa[small] = 0.0
    return aa.view(*orig_shape[:-2], 3)


def _R_to_r6d(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> 6D representation. R: [...,3,3] -> [...,6]"""
    orig_shape = R.shape
    R = R.view(-1, 3, 3)
    r6 = R[:, :, :2].transpose(1, 2).contiguous().view(-1, 6)
    return r6.view(*orig_shape[:-2], 6)


def normalize_vector(v):
    """归一化向量 [..., 3]"""
    batch = v.shape[:-1]
    v_mag = torch.sqrt(v.pow(2).sum(-1))
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).to(v.device)))
    v_mag = v_mag.view(*batch, 1)
    v = v / v_mag
    return v


def cross_product(u, v):
    """计算叉积 [..., 3]"""
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return torch.stack((i, j, k), dim=-1)


def global2local(global_rotmats, parents):
    """
    将全局旋转矩阵转换为局部旋转矩阵
    
    Args:
        global_rotmats: [batch_size, num_joints, 3, 3]
        parents: 父关节索引数组
    
    Returns:
        local_rotmats: [batch_size, num_joints, 3, 3]
    """
    batch_size, num_joints, _, _ = global_rotmats.shape
    local_rotmats = torch.zeros_like(global_rotmats)
    local_rotmats[:, 0] = global_rotmats[:, 0]
    
    for i in range(1, num_joints):
        parent_idx = parents[i]
        R_global_parent = global_rotmats[:, parent_idx]
        R_global_current = global_rotmats[:, i]
        R_global_parent_inv = R_global_parent.transpose(-1, -2)
        local_rotmats[:, i] = torch.matmul(R_global_parent_inv, R_global_current)
    
    return local_rotmats
