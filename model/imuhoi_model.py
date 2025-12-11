"""
IMUHOI完整模型 - 统一的前向推理接口
"""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from .velocity_contact import VelocityContactModule
from .human_pose import HumanPoseModule
from .object_trans import ObjectTransModule


class IMUHOIModel(nn.Module):
    """
    统一的IMUHOI模型，包含三个模块：
    - VelocityContactModule (Stage 1): 预测手和物体速度、接触概率
    - HumanPoseModule (Stage 2): 预测人体姿态
    - ObjectTransModule (Stage 3): 预测物体位置
    """
    
    def __init__(self, cfg, device, no_trans: bool = False):
        """
        Args:
            cfg: 配置对象
            device: 计算设备
            no_trans: 是否使用noTrans模式（不预测根位移）
        """
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.no_trans = no_trans
        
        # 创建三个模块
        self.velocity_contact_module = VelocityContactModule(cfg)
        self.human_pose_module = HumanPoseModule(cfg, device, no_trans=no_trans)
        self.object_trans_module = ObjectTransModule(cfg)
    
    def load_pretrained_modules(self, module_paths: Dict[str, str], strict: bool = True):
        """
        加载预训练模块权重
        
        Args:
            module_paths: 模块名称到权重路径的映射
                - velocity_contact: VelocityContactModule权重路径
                - human_pose: HumanPoseModule权重路径
                - object_trans: ObjectTransModule权重路径
            strict: 是否严格匹配state_dict
        """
        from utils.utils import load_checkpoint
        
        for name, path in module_paths.items():
            if path and os.path.exists(path):
                module = getattr(self, f"{name}_module", None)
                if module is not None:
                    load_checkpoint(module, path, self.device, strict=strict)
                    print(f"Loaded {name} from {path}")
            else:
                if path:
                    print(f"Warning: {name} checkpoint not found at {path}")
    
    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            data_dict: 输入数据字典，需包含：
                - human_imu: [B, T, N_sensors, feat_dim] 人体IMU数据
                - obj_imu: [B, T, obj_feat_dim] 物体IMU数据
                - v_init: [B, N_vel, 3] 初始速度
                - p_init: [B, N_pose, 6] 初始姿态
                - trans_init / trans_gt: 根位置初始值或GT
                - obj_trans_init: [B, 3] 物体初始位置
                - obj_vel_init: [B, 3] 物体初始速度
                - hand_vel_glb_init: [B, 2, 3] 手部初始速度
                - contact_init: [B, 6] 接触状态+物体速度
                - has_object: [B] 是否有物体
            use_object_data: 是否使用物体数据进行Stage 3
            compute_fk: 是否计算FK方式的物体位置
        
        Returns:
            结果字典，包含各阶段的预测输出
        """
        human_imu = data_dict["human_imu"]
        batch_size, seq_len = human_imu.shape[:2]
        
        results = {}
        
        # Stage 1: VelocityContact - 预测速度和接触
        vc_input_dict = {
            'human_imu': human_imu,
            'obj_imu': data_dict["obj_imu"],
            'hand_vel_glb_init': data_dict["hand_vel_glb_init"],
            'obj_vel_init': data_dict["obj_vel_init"],
            'contact_init': data_dict.get("contact_init"),
        }
        vc_out = self.velocity_contact_module(vc_input_dict)
        results.update(vc_out)
        
        # Stage 2: HumanPose - 预测人体姿态
        hp_input_dict = {
            'human_imu': human_imu,
            'v_init': data_dict["v_init"],
            'p_init': data_dict["p_init"],
        }
        if self.no_trans:
            # noTrans模式：使用GT trans
            hp_input_dict['trans_gt'] = data_dict["trans_gt"]
        else:
            # 正常模式：预测trans
            hp_input_dict['trans_init'] = data_dict["trans_init"]
        
        hp_out = self.human_pose_module(hp_input_dict)
        results.update(hp_out)
        
        # Stage 3: ObjectTrans - 预测物体位置
        has_object = data_dict.get("has_object")
        if use_object_data and (has_object is None or has_object.any()):
            ot_out = self.object_trans_module(
                hp_out["pred_hand_glb_pos"],
                vc_out["pred_hand_contact_prob"],
                data_dict["obj_trans_init"],
                obj_imu=data_dict["obj_imu"],
                human_imu=human_imu,
                obj_vel_input=vc_out["pred_obj_vel"],
                contact_init=data_dict["contact_init"],
                has_object_mask=has_object,
            )
            results.update(ot_out)
            
            # 如果需要计算FK方法的物体位置（用于比较）
            if compute_fk and "pred_lhand_obj_trans" in ot_out and "pred_rhand_obj_trans" in ot_out:
                # 使用gating weights融合左右手FK结果
                gating_weights = ot_out.get("gating_weights")
                if gating_weights is not None:
                    # gating_weights: [B, T, 3] (left, right, imu)
                    # 归一化左右手权重
                    lhand_weight = gating_weights[..., 0:1]
                    rhand_weight = gating_weights[..., 1:2]
                    fk_weight_sum = lhand_weight + rhand_weight
                    fk_weight_sum = torch.clamp(fk_weight_sum, min=1e-6)
                    lhand_weight_norm = lhand_weight / fk_weight_sum
                    rhand_weight_norm = rhand_weight / fk_weight_sum
                    
                    # 融合FK结果
                    pred_obj_trans_fk = (
                        lhand_weight_norm * ot_out["pred_lhand_obj_trans"] +
                        rhand_weight_norm * ot_out["pred_rhand_obj_trans"]
                    )
                else:
                    # 如果没有gating weights，使用接触概率
                    contact_prob = vc_out["pred_hand_contact_prob"]
                    lhand_prob = contact_prob[..., 0:1]
                    rhand_prob = contact_prob[..., 1:2]
                    prob_sum = lhand_prob + rhand_prob
                    prob_sum = torch.clamp(prob_sum, min=1e-6)
                    lhand_prob_norm = lhand_prob / prob_sum
                    rhand_prob_norm = rhand_prob / prob_sum
                    
                    pred_obj_trans_fk = (
                        lhand_prob_norm * ot_out["pred_lhand_obj_trans"] +
                        rhand_prob_norm * ot_out["pred_rhand_obj_trans"]
                    )
                
                results["pred_obj_trans_fk"] = pred_obj_trans_fk
        
        results["has_object"] = has_object
        return results


def load_model(
    config,
    device: torch.device,
    no_trans: bool = False,
    module_paths: Optional[Dict[str, str]] = None,
) -> IMUHOIModel:
    """
    加载IMUHOI模型
    
    Args:
        config: 配置对象
        device: 计算设备
        no_trans: 是否使用noTrans模式
        module_paths: 可选的模块权重路径覆盖
    
    Returns:
        加载好的IMUHOIModel实例
    """
    from utils.utils import flatten_lstm_parameters
    
    model = IMUHOIModel(config, device, no_trans=no_trans)
    model = model.to(device)
    
    # 确定预训练模块路径
    pretrained_modules = {}
    
    # 优先使用传入的module_paths
    if module_paths:
        pretrained_modules = module_paths
    else:
        # 从config中获取
        staged_cfg = getattr(config, "staged_training", {})
        if staged_cfg:
            modular_cfg = staged_cfg.get("modular_training", {}) if isinstance(staged_cfg, dict) else getattr(staged_cfg, "modular_training", {})
            if modular_cfg:
                pretrained_modules = modular_cfg.get("pretrained_modules", {}) if isinstance(modular_cfg, dict) else getattr(modular_cfg, "pretrained_modules", {})
    
    # 加载预训练模块
    if pretrained_modules:
        print("Loading pretrained modules:")
        model.load_pretrained_modules(pretrained_modules)
    
    # 优化LSTM参数布局
    flatten_lstm_parameters(model)
    model.eval()
    
    return model

