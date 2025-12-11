"""
VelocityContactModule: 预测手部速度、物体速度和接触概率
Stage 1 - 可独立训练
"""
import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

from .base import RNNWithInit
from configs import _SENSOR_NAMES


class VelocityContactModule(nn.Module):
    """
    速度和接触预测模块
    输入: human_imu, obj_imu
    输出: hand_vel, obj_vel, contact_prob
    """
    def __init__(self, cfg):
        super().__init__()
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.obj_imu_dim = getattr(cfg, "obj_imu_dim", 9)
        hidden_dim = getattr(cfg, "velocity_hidden_dim", 128)
        num_layers = getattr(cfg, "velocity_num_layers", 2)
        dropout = getattr(cfg, "velocity_dropout", 0.2)

        # 手部速度预测网络
        self.hand_vel_net = RNNWithInit(
            n_input=self.num_human_imus * self.imu_dim,
            n_output=6,  # 左右手各3维
            n_hidden=hidden_dim,
            n_init=6,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        # 物体速度预测网络
        self.obj_vel_net = RNNWithInit(
            n_input=self.obj_imu_dim,
            n_output=3,
            n_hidden=hidden_dim,
            n_init=3,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        # 接触预测网络
        contact_input_dim = 2 * self.imu_dim + self.obj_imu_dim + 6 + 3
        contact_hidden = max(hidden_dim // 2, 32)
        self.contact_net = RNNWithInit(
            n_input=contact_input_dim,
            n_output=3,  # 左手、右手、物体接触
            n_hidden=contact_hidden,
            n_init=6,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )

        self.left_hand_sensor = _SENSOR_NAMES.index("LeftForeArm")
        self.right_hand_sensor = _SENSOR_NAMES.index("RightForeArm")

    def _denormalize_imu(self, human_imu):
        """将IMU数据转换到世界坐标系"""
        batch_size, seq_len = human_imu.shape[:2]
        
        human_imu_acc = human_imu[:, :, :, :3]
        human_imu_ori = human_imu[:, :, :, 3:9]
        human_imu_ori_6d = human_imu_ori.reshape(-1, 6)
        human_imu_ori_mat = rotation_6d_to_matrix(human_imu_ori_6d).reshape(batch_size, seq_len, 6, 3, 3)
        
        R0T = human_imu_ori_mat[:, :, 0].transpose(-1, -2)
        acc_world = torch.matmul(human_imu_acc, R0T)
        acc0_world = acc_world[:, :, :1, :]
        acc_rest_mix = acc_world[:, :, 1:, :] + acc0_world
        human_imu_acc_denorm = torch.cat([acc0_world, acc_rest_mix], dim=2)
        
        human_imu_ori_denorm = torch.cat([
            human_imu_ori_mat[:, :, :1], 
            human_imu_ori_mat[:, :, :1].matmul(human_imu_ori_mat[:, :, 1:])
        ], dim=2)
        human_imu_ori_denorm_6d = matrix_to_rotation_6d(human_imu_ori_denorm)
        human_imu_denorm = torch.cat([human_imu_acc_denorm, human_imu_ori_denorm_6d], dim=-1)
        
        return human_imu_denorm

    def forward(self, data_dict: dict):
        """
        前向传播
        
        Args:
            data_dict: 包含以下键的字典:
                - human_imu: [B, T, num_imu, imu_dim] 人体IMU数据
                - obj_imu: [B, T, obj_imu_dim] 物体IMU数据
                - hand_vel_glb_init: [B, 2, 3] 初始手部速度
                - obj_vel_init: [B, 3] 初始物体速度
                - contact_init: [B, 6] 初始接触状态 (可选)
        
        Returns:
            dict: 包含预测的手部速度、物体速度和接触概率
        """
        human_imu = data_dict['human_imu']
        obj_imu = data_dict.get('obj_imu')
        hand_vel_init = data_dict['hand_vel_glb_init']
        obj_vel_init = data_dict['obj_vel_init']
        contact_init = data_dict.get('contact_init')
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B, T, num_imu, imu_dim], got {human_imu.shape}")
        batch_size, seq_len, _, _ = human_imu.shape
        device = human_imu.device
        dtype = human_imu.dtype

        # 处理物体IMU
        if obj_imu is None:
            obj_imu = torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        else:
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(batch_size, seq_len, -1)
            elif obj_imu.dim() != 3:
                raise ValueError(f"obj_imu must be [B, T, obj_dim], got {obj_imu.shape}")

        # 处理初始化向量
        if hand_vel_init.dim() != 3 or hand_vel_init.shape[1:] != (2, 3):
            raise ValueError(f"hand_vel_init must be [B,2,3], got {hand_vel_init.shape}")
        hand_vel_init_vec = hand_vel_init.reshape(batch_size, -1)

        if obj_vel_init.dim() == 1:
            obj_vel_init_vec = obj_vel_init.unsqueeze(0).expand(batch_size, -1)
        elif obj_vel_init.dim() == 2:
            obj_vel_init_vec = obj_vel_init
        else:
            raise ValueError(f"obj_vel_init must be [B,3] or [3], got {obj_vel_init.shape}")

        # 去规范化IMU数据
        human_imu_denorm = self._denormalize_imu(human_imu)
        human_imu_denorm_flat = human_imu_denorm.reshape(batch_size, seq_len, -1)

        # 预测手部速度
        hand_vel_flat = self.hand_vel_net((human_imu_denorm_flat, hand_vel_init_vec))
        hand_vel = hand_vel_flat.view(batch_size, seq_len, 2, 3)

        # 预测物体速度
        obj_vel = self.obj_vel_net((obj_imu, obj_vel_init_vec))

        # 构建接触预测输入
        hand_imu_feat = human_imu_denorm[:, :, [self.left_hand_sensor, self.right_hand_sensor], :].reshape(
            batch_size, seq_len, -1
        )
        contact_input = torch.cat(
            (hand_imu_feat, obj_imu, hand_vel.view(batch_size, seq_len, -1), obj_vel),
            dim=-1,
        )

        # 处理接触初始化
        if contact_init is None:
            contact_init_vec = torch.cat(
                (torch.zeros(batch_size, 3, device=device, dtype=dtype), obj_vel_init_vec), 
                dim=-1
            )
        else:
            if contact_init.dim() == 1:
                contact_init_vec = contact_init.unsqueeze(0).expand(batch_size, -1)
            else:
                contact_init_vec = contact_init

        # 预测接触概率
        contact_logits = self.contact_net((contact_input, contact_init_vec))
        contact_prob = torch.sigmoid(contact_logits)

        return {
            "pred_hand_glb_vel": hand_vel,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": contact_logits,
            "pred_hand_contact_prob": contact_prob,
        }

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        """返回空输出"""
        return {
            "pred_hand_glb_vel": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "pred_obj_vel": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_hand_contact_logits": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_hand_contact_prob": torch.zeros(batch_size, seq_len, 3, device=device),
        }

