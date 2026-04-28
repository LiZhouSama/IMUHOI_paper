"""
IMUHOI完整模型 - 统一的前向推理接口
"""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix

from .velocity_contact import VelocityContactModule
from .human_pose import HumanPoseModule
from .object_trans import ObjectTransModule


class IMUHOIModel(nn.Module):
    """
    统一的IMUHOI模型，包含三个模块：
    - HumanPoseModule (Stage 1): 预测人体姿态
    - VelocityContactModule (Stage 2): 预测手和物体速度、接触概率
    - ObjectTransModule (Stage 3): 预测物体位置
    """

    @staticmethod
    def _fk_obj_trans_baseline_hard(
        pred_hand_contact_prob: torch.Tensor,  # [B, T, 3]
        pred_hand_positions: torch.Tensor,     # [B, T, 2, 3]
        obj_rotm: torch.Tensor,                # [B, T, 3, 3]
        obj_trans_init: torch.Tensor,          # [B, 3]
        contact_threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        严格复刻 trash/IMUHOI_stage_net.py::pred_obj_pos_fk 的硬阈值FK baseline，
        但用“单次时序状态机”实现（不显式构造contact_segments），输出每帧物体位置。
        """
        batch_size, seq_len, _ = pred_hand_contact_prob.shape
        device = pred_hand_contact_prob.device
        dtype = pred_hand_positions.dtype

        computed_obj_trans = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
        if seq_len == 0:
            return computed_obj_trans
        computed_obj_trans[:, 0] = obj_trans_init

        # threshold -> {0,1}，并强制第一帧为0（与原实现一致）
        l_contact = (pred_hand_contact_prob[..., 0] > contact_threshold).float()
        r_contact = (pred_hand_contact_prob[..., 1] > contact_threshold).float()
        l_contact[:, 0] = 0.0
        r_contact[:, 0] = 0.0

        # 起始接触帧：t>=1 且contact[t]=1 & contact[t-1]=0
        l_start = torch.zeros_like(l_contact)
        r_start = torch.zeros_like(r_contact)
        if seq_len > 1:
            l_start[:, 1:] = ((l_contact[:, 1:] > 0) & (l_contact[:, :-1] == 0)).float()
            r_start[:, 1:] = ((r_contact[:, 1:] > 0) & (r_contact[:, :-1] == 0)).float()

        z_unit = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        default_len = torch.tensor(0.1, device=device, dtype=dtype)

        for b in range(batch_size):
            current = -1
            obj_pos = obj_trans_init[b].clone()
            dir_local = None
            dist = None

            for t in range(seq_len):
                # new_contact：左优先（与原实现一致）
                new = -1
                if l_start[b, t] > 0:
                    new = 0
                elif r_start[b, t] > 0:
                    new = 1

                if current == 0:
                    has_contact = l_contact[b, t] > 0
                    has_contact_another = r_contact[b, t] > 0
                elif current == 1:
                    has_contact = r_contact[b, t] > 0
                    has_contact_another = l_contact[b, t] > 0
                else:
                    has_contact = False
                    has_contact_another = False

                # 状态转移规则与原实现一致：new_contact 优先，其次“断开但另一只手仍接触”的切换，再次“断开结束”
                if new != -1:
                    current = new
                    hand0 = pred_hand_positions[b, t, current, :]
                    R0 = obj_rotm[b, t]
                    vec_world = obj_pos - hand0
                    dist0 = torch.norm(vec_world)
                    if dist0 > 1e-6:
                        unit_world = vec_world / dist0
                    else:
                        unit_world = z_unit
                        dist0 = default_len
                    dir_local = R0.transpose(0, 1) @ unit_world
                    dist = dist0
                elif (not has_contact) and has_contact_another:
                    current = 1 - current
                    hand0 = pred_hand_positions[b, t, current, :]
                    R0 = obj_rotm[b, t]
                    vec_world = obj_pos - hand0
                    dist0 = torch.norm(vec_world)
                    if dist0 > 1e-6:
                        unit_world = vec_world / dist0
                    else:
                        unit_world = z_unit
                        dist0 = default_len
                    dir_local = R0.transpose(0, 1) @ unit_world
                    dist = dist0
                elif (not has_contact) and current != -1:
                    current = -1
                    dir_local = None
                    dist = None

                # 生成当前帧位置：无接触保持不动；有接触则用 hand_pos + R @ dir_local * dist
                if current == -1:
                    computed_obj_trans[b, t] = obj_pos
                else:
                    Rt = obj_rotm[b, t]
                    direction_world = Rt @ dir_local
                    obj_pos = pred_hand_positions[b, t, current, :] + direction_world * dist
                    computed_obj_trans[b, t] = obj_pos

        return computed_obj_trans

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
        
        # Stage 1: HumanPose - 先预测人体姿态
        hp_input_dict = {
            'human_imu': human_imu,
            'v_init': data_dict["v_init"],
            'p_init': data_dict["p_init"],
        }
        if self.no_trans:
            hp_input_dict['trans_gt'] = data_dict["trans_gt"]
        else:
            hp_input_dict['trans_init'] = data_dict["trans_init"]
        
        hp_out = self.human_pose_module(hp_input_dict)
        results.update(hp_out)
        
        # Stage 2: VelocityContact - 使用HPE结果估计接触
        vc_input_dict = {
            'human_imu': human_imu,
            'obj_imu': data_dict["obj_imu"],
            'hand_vel_glb_init': data_dict["hand_vel_glb_init"],
            'obj_vel_init': data_dict["obj_vel_init"],
            'contact_init': data_dict.get("contact_init"),
            'hp_out': hp_out,
        }
        vc_out = self.velocity_contact_module(vc_input_dict, hp_out=hp_out)
        results.update(vc_out)
        
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
                contact_init=data_dict.get("contact_init"),
                has_object_mask=has_object,
            )
            results.update(ot_out)
            
            # 如果需要计算FK方式的物体位置（用于比较）
            if compute_fk:
                obj_imu = data_dict["obj_imu"]
                obj_rot6d = obj_imu[..., 3:9]
                obj_rotm = rotation_6d_to_matrix(obj_rot6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
                results["pred_obj_trans_fk"] = self._fk_obj_trans_baseline_hard(
                    vc_out["pred_hand_contact_prob"],
                    hp_out["pred_hand_glb_pos"],
                    obj_rotm,
                    data_dict["obj_trans_init"],
                )
        
        results["has_object"] = has_object
        return results

    def inference(
        self,
        data_dict: Dict[str, torch.Tensor],
        gt_targets: Optional[Dict[str, torch.Tensor]] = None,
        use_object_data: bool = True,
        compute_fk: bool = False,
        interaction_use_human_pred: bool = True,
        **_,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluation-compatible inference entrypoint.

        The RNN pipeline is deterministic and does not need diffusion sampling or
        GT target conditioning.  Keep the signature aligned with the DiT
        pipeline so shared eval/visualization code can call either backend.
        """
        _ = gt_targets, interaction_use_human_pred
        return self.forward(
            data_dict,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
        )


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
