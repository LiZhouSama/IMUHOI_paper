"""
ObjectTransModule: 预测物体位置
Stage 3 - 依赖Stage1和Stage2的输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from .base import RNNWithInit
from configs import FRAME_RATE, _SENSOR_NAMES


class ObjectTransModule(nn.Module):
    """
    物体位置预测模块
    基于手部位置、接触概率和物体IMU预测物体位置
    """
    def __init__(self, cfg):
        super().__init__()
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        hidden_dim_multiplier = getattr(cfg, "hidden_dim_multiplier", 1)

        # Gating网络参数
        self.gating_prior_beta = getattr(cfg, "gating_prior_beta", 5.0)
        self.gating_temperature = getattr(cfg, "gating_temperature", 5.0)
        self.gating_smoothing_enabled = getattr(cfg, "gating_smoothing_enabled", False)
        self.gating_smoothing_alpha = getattr(cfg, "gating_smoothing_alpha", 0.6)
        self.gating_max_change = getattr(cfg, "gating_max_change", 0.25)
        
        # 速度校正参数
        self.vel_static_threshold = getattr(cfg, "vel_static_threshold", 0.3)
        self.vel_min_hand_speed = getattr(cfg, "vel_min_hand_speed", 0.02)

        n_fk_branch_input = 34
        n_gating_input = 9

        # 左手FK预测头
        self.lhand_fk_head = RNNWithInit(
            n_input=n_fk_branch_input,
            n_output=4,  # 方向(3) + 长度(1)
            n_hidden=128 * hidden_dim_multiplier,
            n_init=4,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        
        # 右手FK预测头
        self.rhand_fk_head = RNNWithInit(
            n_input=n_fk_branch_input,
            n_output=4,
            n_hidden=128 * hidden_dim_multiplier,
            n_init=4,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        
        # Gating网络
        self.gating_head = RNNWithInit(
            n_input=n_gating_input,
            n_output=3,  # 左手FK、右手FK、IMU积分
            n_hidden=64 * hidden_dim_multiplier,
            n_init=6,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=0.2,
        )

    @staticmethod
    def _unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return x / norm

    @staticmethod
    def _softplus_positive(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-4

    def _smooth_gating_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """平滑gating权重，减少跳变"""
        if self.training or (not self.gating_smoothing_enabled) or weights.size(1) < 2:
            return weights

        LHAND_FK, RHAND_FK, IMU_BRANCH = 0, 1, 2
        smoothed_weights = weights.clone()
        prev_smoothed = weights[:, 0, :]

        for t in range(1, weights.size(1)):
            current_weights = weights[:, t, :]
            prev_dominant = prev_smoothed.argmax(dim=-1)
            curr_dominant = current_weights.argmax(dim=-1)
            frame_weights = current_weights.clone()

            transition_mask = prev_dominant != curr_dominant
            if transition_mask.any():
                for b in range(weights.size(0)):
                    if not transition_mask[b]:
                        continue
                    prev_dom = prev_dominant[b].item()
                    curr_dom = curr_dominant[b].item()
                    need_smoothing = (
                        (prev_dom == IMU_BRANCH and curr_dom in [LHAND_FK, RHAND_FK]) or
                        (prev_dom in [LHAND_FK, RHAND_FK] and curr_dom in [LHAND_FK, RHAND_FK])
                    )
                    if need_smoothing:
                        frame_weights[b, :] = (
                            self.gating_smoothing_alpha * prev_smoothed[b, :] +
                            (1.0 - self.gating_smoothing_alpha) * current_weights[b, :]
                        )
                        if self.gating_max_change > 0:
                            weight_change = frame_weights[b, :] - prev_smoothed[b, :]
                            change_norm = torch.norm(weight_change)
                            if change_norm > self.gating_max_change:
                                weight_change = weight_change * (self.gating_max_change / (change_norm + 1e-8))
                                frame_weights[b, :] = prev_smoothed[b, :] + weight_change
                        frame_weights[b, :] = F.softmax(
                            torch.log(frame_weights[b, :] + 1e-8) * self.gating_temperature, dim=-1
                        )
            smoothed_weights[:, t, :] = frame_weights
            prev_smoothed = frame_weights

        return smoothed_weights

    def _build_fk_inputs(self, obj_rot6d, hand_pos, hand_contact_scalar, obj_imu9, hand_imu9, obj_vel3, obj_rot_delta3):
        return torch.cat([obj_rot6d, hand_pos, hand_contact_scalar, obj_imu9, hand_imu9, obj_vel3, obj_rot_delta3], dim=2)

    def _build_gating_inputs(self, contact_prob3, obj_vel3, obj_imu_acc3):
        return torch.cat([contact_prob3, obj_vel3, obj_imu_acc3], dim=2)

    def _rot6d_delta(self, rot6d: torch.Tensor) -> torch.Tensor:
        B, T, _ = rot6d.shape
        R = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(B, T, 3, 3)
        rel = torch.matmul(R[:, 1:].transpose(-1, -2), R[:, :-1])
        aa = matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(B, T-1, 3)
        aa = F.pad(aa, (0, 0, 1, 0))
        return aa

    def _compute_hand_velocity(self, hand_pos: torch.Tensor) -> torch.Tensor:
        """从手部位置计算速度 [B, T, 3] -> [B, T, 3]"""
        vel = torch.zeros_like(hand_pos)
        if hand_pos.size(1) > 1:
            vel[:, 1:] = (hand_pos[:, 1:] - hand_pos[:, :-1]) * FRAME_RATE
        return vel

    def _correct_obj_velocity(
        self,
        v_imu: torch.Tensor,      # [B, T, 3]
        v_lhand: torch.Tensor,    # [B, T, 3]
        v_rhand: torch.Tensor,    # [B, T, 3]
        p_left: torch.Tensor,     # [B, T, 1]
        p_right: torch.Tensor,    # [B, T, 1]
    ) -> torch.Tensor:
        """
        两阶段速度校正：
        1. 静止校正：无接触时物体静止
        2. 接触方向校正：接触时物体速度在手方向上的分量=手速度
        """
        p_max = torch.maximum(p_left, p_right)
        
        # === 阶段1：静止校正因子 ===
        # p_max < threshold 时开始衰减到0
        static_factor = torch.clamp(p_max / self.vel_static_threshold, 0, 1)
        
        # === 阶段2：接触方向校正 ===
        def direction_correct(v_obj, v_hand, p_contact):
            """在手运动方向上校正物体速度"""
            v_hand_speed = v_hand.norm(dim=-1, keepdim=True)
            hand_moving = (v_hand_speed > self.vel_min_hand_speed).float()
            v_hand_dir = v_hand / v_hand_speed.clamp_min(1e-6)
            
            # v_obj在手方向上的投影
            proj_scalar = (v_obj * v_hand_dir).sum(dim=-1, keepdim=True)
            
            # 校正: 去掉v_obj在手方向的分量，换成手速度
            v_corrected = v_obj - proj_scalar * v_hand_dir + v_hand
            
            # 只有手在运动且有接触时才应用
            w = p_contact * hand_moving
            return v_obj * (1 - w) + v_corrected * w
        
        # 分别用左右手校正
        v_lcorr = direction_correct(v_imu, v_lhand, p_left)
        v_rcorr = direction_correct(v_imu, v_rhand, p_right)
        
        # 按接触概率融合
        total_p = p_left + p_right + 1e-6
        v_contact_corrected = (p_left / total_p) * v_lcorr + (p_right / total_p) * v_rcorr
        
        # 最终：静止因子控制整体幅度
        return static_factor * v_contact_corrected

    def _compute_init_dir_len(self, hand_pos_0, obj_rotm_0, obj_pos_0):
        vec_world = obj_pos_0 - hand_pos_0
        lb0 = vec_world.norm(dim=-1, keepdim=True)
        unit_world = self._unit_vector(vec_world)
        obj_Rt = obj_rotm_0.transpose(-1, -2)
        oe0 = torch.bmm(obj_Rt, unit_world.unsqueeze(-1)).squeeze(-1)
        return oe0, lb0

    def forward(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        obj_imu: torch.Tensor = None,
        human_imu: torch.Tensor = None,
        obj_vel_input: torch.Tensor = None,
        contact_init: torch.Tensor = None,
        has_object_mask: torch.Tensor = None,
    ):
        """
        前向传播
        
        Args:
            hand_positions: [B, T, 2, 3] 手部位置 (来自HumanPoseModule)
            pred_hand_contact_prob: [B, T, 3] 接触概率 (来自VelocityContactModule)
            obj_trans_init: [B, 3] 初始物体位置
            obj_imu: [B, T, 9] 物体IMU数据
            human_imu: [B, T, num_imu, imu_dim] 人体IMU数据
            obj_vel_input: [B, T, 3] 物体速度 (来自VelocityContactModule)
            contact_init: [B, 6] 初始接触状态
            has_object_mask: [B] 是否有物体
        
        Returns:
            dict: 包含预测的物体位置和相关信息
        """
        if hand_positions is None:
            raise ValueError("hand_positions cannot be None")
        if hand_positions.dim() == 3:
            bs, seq_len, _ = hand_positions.shape
            hand_positions = hand_positions.view(bs, seq_len, 2, 3)
        elif hand_positions.dim() == 4:
            bs, seq_len = hand_positions.shape[:2]
        else:
            raise ValueError(f"Unexpected hand_positions shape {hand_positions.shape}")

        device = hand_positions.device
        dtype = hand_positions.dtype
        lhand_position = hand_positions[:, :, 0, :]
        rhand_position = hand_positions[:, :, 1, :]

        # 处理输入
        if obj_imu is None:
            obj_imu = torch.zeros(bs, seq_len, self.imu_dim, device=device, dtype=dtype)
        else:
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(bs, seq_len, -1)

        if human_imu is None:
            human_imu = torch.zeros(bs, seq_len, self.num_human_imus * self.imu_dim, device=device, dtype=dtype)
        if human_imu.dim() == 3 and human_imu.shape[-1] == self.num_human_imus * self.imu_dim:
            human_imu = human_imu.view(bs, seq_len, self.num_human_imus, self.imu_dim)

        obj_rot = obj_imu[:, :, 3:9]
        obj_rot_delta = self._rot6d_delta(obj_rot)
        obj_rotm = rotation_6d_to_matrix(obj_rot.reshape(-1, 6)).reshape(bs, seq_len, 3, 3)
        obj_imu_acc = obj_imu[:, :, :3]

        l_idx = _SENSOR_NAMES.index("LeftForeArm")
        r_idx = _SENSOR_NAMES.index("RightForeArm")
        lhand_imu9 = human_imu[:, :, l_idx, :]
        rhand_imu9 = human_imu[:, :, r_idx, :]

        if obj_vel_input is None:
            obj_vel_input = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)

        pL = pred_hand_contact_prob[:, :, 0:1]
        pR = pred_hand_contact_prob[:, :, 1:2]

        # 计算手速度并校正物体速度
        lhand_vel = self._compute_hand_velocity(lhand_position)
        rhand_vel = self._compute_hand_velocity(rhand_position)
        obj_vel_corrected = self._correct_obj_velocity(obj_vel_input, lhand_vel, rhand_vel, pL, pR)

        # 构建FK输入
        fk_l_input = self._build_fk_inputs(obj_rot, lhand_position, pL, obj_imu, lhand_imu9, obj_vel_input, obj_rot_delta)
        fk_r_input = self._build_fk_inputs(obj_rot, rhand_position, pR, obj_imu, rhand_imu9, obj_vel_input, obj_rot_delta)

        # 计算初始方向和长度
        obj_pos_0 = obj_trans_init
        obj_R_0 = obj_rotm[:, 0, :, :]
        l_hand_0 = lhand_position[:, 0, :]
        r_hand_0 = rhand_position[:, 0, :]
        l_oe0, l_lb0 = self._compute_init_dir_len(l_hand_0, obj_R_0, obj_pos_0)
        r_oe0, r_lb0 = self._compute_init_dir_len(r_hand_0, obj_R_0, obj_pos_0)

        l_init_vec = torch.cat((l_oe0, l_lb0), dim=-1)
        r_init_vec = torch.cat((r_oe0, r_lb0), dim=-1)

        if contact_init is None:
            contact_init_vec = torch.cat((torch.zeros(bs, 3, device=device, dtype=dtype), obj_vel_input[:, 0, :]), dim=-1)
        else:
            if contact_init.dim() == 1:
                contact_init_vec = contact_init.unsqueeze(0).expand(bs, -1)
            else:
                contact_init_vec = contact_init

        # FK预测
        l_fk_out = self.lhand_fk_head((fk_l_input, l_init_vec))
        r_fk_out = self.rhand_fk_head((fk_r_input, r_init_vec))
        l_dir = self._unit_vector(l_fk_out[:, :, :3])
        r_dir = self._unit_vector(r_fk_out[:, :, :3])
        l_len = self._softplus_positive(l_fk_out[:, :, 3])
        r_len = self._softplus_positive(r_fk_out[:, :, 3])

        # 转换到世界坐标系
        obj_rotm_flat = obj_rotm.reshape(bs * seq_len, 3, 3)
        l_dir_world = torch.bmm(obj_rotm_flat, l_dir.reshape(bs * seq_len, 3, 1)).reshape(bs, seq_len, 3)
        r_dir_world = torch.bmm(obj_rotm_flat, r_dir.reshape(bs * seq_len, 3, 1)).reshape(bs, seq_len, 3)
        l_pos_fk = lhand_position + l_dir_world * l_len.unsqueeze(-1)
        r_pos_fk = rhand_position + r_dir_world * r_len.unsqueeze(-1)

        # Gating预测
        gating_input = self._build_gating_inputs(pred_hand_contact_prob, obj_vel_input, obj_imu_acc)
        gate_logits = self.gating_head((gating_input, contact_init_vec))
        prior_im = 1.0 - torch.maximum(pL.squeeze(-1), pR.squeeze(-1))
        prior = torch.stack([pL.squeeze(-1), pR.squeeze(-1), prior_im], dim=-1)
        gate_logits = gate_logits + self.gating_prior_beta * torch.log(prior + 1e-6)
        weights_raw = F.softmax(gate_logits / self.gating_temperature, dim=-1)
        weights = self._smooth_gating_weights(weights_raw)

        # 融合位置（使用校正后的速度）
        fused_pos = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)
        dt = 1.0 / FRAME_RATE
        for t in range(seq_len):
            prev_pos = fused_pos[:, t - 1, :] if t > 0 else obj_trans_init
            pos_imu_integrated = prev_pos + obj_vel_corrected[:, t, :] * dt
            fused_pos[:, t, :] = (
                weights[:, t, 0:1] * l_pos_fk[:, t, :] +
                weights[:, t, 1:2] * r_pos_fk[:, t, :] +
                weights[:, t, 2:3] * pos_imu_integrated
            )

        # 计算速度和加速度
        vel_from_pos = torch.zeros_like(fused_pos)
        acc_from_pos = torch.zeros_like(fused_pos)
        if seq_len > 1:
            vel_from_pos[:, 1:] = (fused_pos[:, 1:] - fused_pos[:, :-1]) * FRAME_RATE
        if seq_len > 2:
            acc_from_pos[:, 2:] = (fused_pos[:, 2:] - 2 * fused_pos[:, 1:-1] + fused_pos[:, :-2]) * (FRAME_RATE**2)

        # 应用mask
        if has_object_mask is not None:
            if has_object_mask.dim() > 1:
                has_object_mask = has_object_mask.view(bs)
            mask = has_object_mask.to(dtype=dtype, device=device).view(bs, 1, 1)
            fused_pos = fused_pos * mask
            vel_from_pos = vel_from_pos * mask
            acc_from_pos = acc_from_pos * mask
            weights = weights * mask
            weights_raw = weights_raw * mask
            l_pos_fk = l_pos_fk * mask
            r_pos_fk = r_pos_fk * mask
            l_dir = l_dir * mask
            r_dir = r_dir * mask
            l_len = l_len * mask.squeeze(-1)
            r_len = r_len * mask.squeeze(-1)
            l_oe0 = l_oe0 * mask.squeeze(-1)
            r_oe0 = r_oe0 * mask.squeeze(-1)
            l_lb0 = (l_lb0 * mask.squeeze(-1).unsqueeze(-1)).squeeze(-1)
            r_lb0 = (r_lb0 * mask.squeeze(-1).unsqueeze(-1)).squeeze(-1)
        else:
            l_lb0 = l_lb0.squeeze(-1)
            r_lb0 = r_lb0.squeeze(-1)

        return {
            "pred_obj_trans": fused_pos,
            "gating_weights": weights,
            "gating_weights_raw": weights_raw,
            "pred_obj_vel_from_posdiff": vel_from_pos,
            "pred_obj_acc_from_posdiff": acc_from_pos,
            "obj_vel_input": obj_vel_input,
            "obj_vel_corrected": obj_vel_corrected,
            "pred_lhand_obj_direction": l_dir,
            "pred_rhand_obj_direction": r_dir,
            "pred_lhand_lb": l_len,
            "pred_rhand_lb": r_len,
            "pred_lhand_obj_trans": l_pos_fk,
            "pred_rhand_obj_trans": r_pos_fk,
            "init_lhand_oe_ho": l_oe0,
            "init_rhand_oe_ho": r_oe0,
            "init_lhand_lb": l_lb0,
            "init_rhand_lb": r_lb0,
            "gating_smoothing_applied": (not self.training) and self.gating_smoothing_enabled,
        }

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        """返回空输出"""
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_dir = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_scalar = torch.zeros(batch_size, seq_len, device=device)
        zeros_weights = torch.zeros(batch_size, seq_len, 3, device=device)
        return {
            "pred_obj_trans": zeros_pos,
            "gating_weights": zeros_weights,
            "gating_weights_raw": zeros_weights,
            "pred_obj_vel_from_posdiff": zeros_pos,
            "pred_obj_acc_from_posdiff": zeros_pos,
            "obj_vel_input": zeros_pos,
            "obj_vel_corrected": zeros_pos,
            "pred_lhand_obj_direction": zeros_dir,
            "pred_rhand_obj_direction": zeros_dir,
            "pred_lhand_lb": zeros_scalar,
            "pred_rhand_lb": zeros_scalar,
            "pred_lhand_obj_trans": zeros_pos,
            "pred_rhand_obj_trans": zeros_pos,
            "init_lhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_rhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_lhand_lb": torch.zeros(batch_size, device=device),
            "init_rhand_lb": torch.zeros(batch_size, device=device),
            "gating_smoothing_applied": False,
        }

