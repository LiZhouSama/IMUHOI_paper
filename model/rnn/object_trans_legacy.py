"""
ObjectTransModule: 预测物体位置
Stage 3 - 依赖Stage1和Stage2的输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

from .base import RNN, RNNWithInit
from .online import (
    append_stream_data,
    infer_batch_seq,
    normalize_inference_mode,
    resolve_online_window,
    select_time_context,
    slice_time_dict,
    slice_time_value,
    take_latest_frame,
    TimeDictAccumulator,
)
from configs import FRAME_RATE, _REDUCED_POSE_NAMES, _SENSOR_NAMES


class LegacyObjectTransModule(nn.Module):
    """
    Legacy multi-head ObjectTransModule.

    This intentionally shares the current OT data path, fusion recurrence,
    online prefix contract, and human refinement head. The only architectural
    difference is that the current shared predictor is split into independent
    left-FK, right-FK, and four-way gating RNN heads.

    No mesh prior, observation encoder, or privileged geometry input is used.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.imu_dim = int(getattr(cfg, "imu_dim", 9))
        self.obj_imu_dim = int(max(getattr(cfg, "obj_imu_dim", self.imu_dim), 9))
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        hidden_dim_multiplier = getattr(cfg, "hidden_dim_multiplier", 1)

        self.refine_pose_dim = len(_REDUCED_POSE_NAMES) * 6

        n_fk_branch_input = 6 + 3 + 1 + self.obj_imu_dim + self.imu_dim + 3 + 3
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
            n_output=4,  # 左手FK、右手FK、IMU速度积分、静止保持
            n_hidden=64 * hidden_dim_multiplier,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=0.2,
        )

        n_refine_input = self.refine_pose_dim + 22
        self.refine_head = RNN(
            n_input=n_refine_input,
            n_output=self.refine_pose_dim + 3,
            n_hidden=128 * hidden_dim_multiplier,
            n_rnn_layer=2,
            bidirectional=False,
            dropout=0.2,
        )
        nn.init.zeros_(self.refine_head.linear2.weight)
        nn.init.zeros_(self.refine_head.linear2.bias)

    @staticmethod
    def _unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
        return x / norm

    @staticmethod
    def _softplus_positive(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x) + 1e-4

    def _build_fk_inputs(self, obj_rot6d, hand_pos, hand_contact_scalar, obj_imu9, hand_imu9, obj_vel3, obj_rot_delta3):
        return torch.cat(
            [obj_rot6d, hand_pos, hand_contact_scalar, obj_imu9, hand_imu9, obj_vel3, obj_rot_delta3],
            dim=2,
        )

    def _build_gating_inputs(self, contact_prob3, obj_vel3, obj_imu_acc3):
        return torch.cat([contact_prob3, obj_vel3, obj_imu_acc3], dim=2)

    @staticmethod
    def _pad_or_trim_last_dim(x: torch.Tensor, out_dim: int) -> torch.Tensor:
        if x.shape[-1] == out_dim:
            return x
        if x.shape[-1] > out_dim:
            return x[..., :out_dim]
        pad_shape = (*x.shape[:-1], out_dim - x.shape[-1])
        return torch.cat((x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)), dim=-1)

    def _prepare_obj_imu(self, obj_imu: torch.Tensor, batch_size: int, seq_len: int, device, dtype) -> torch.Tensor:
        if not isinstance(obj_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        out = obj_imu.to(device=device, dtype=dtype)
        if out.dim() == 4:
            out = out.reshape(batch_size, seq_len, -1)
        if out.dim() == 2:
            out = out.unsqueeze(0)
        if out.shape[0] == 1 and batch_size > 1:
            out = out.expand(batch_size, -1, -1)
        if out.dim() != 3 or out.shape[:2] != (batch_size, seq_len):
            return torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        return self._pad_or_trim_last_dim(out, self.obj_imu_dim)

    def _prepare_human_imu(self, human_imu: torch.Tensor, batch_size: int, seq_len: int, device, dtype) -> torch.Tensor:
        if not isinstance(human_imu, torch.Tensor):
            return torch.zeros(batch_size, seq_len, self.num_human_imus, self.imu_dim, device=device, dtype=dtype)
        out = human_imu.to(device=device, dtype=dtype)
        if out.dim() == 3 and out.shape[-1] == self.num_human_imus * self.imu_dim:
            out = out.view(batch_size, seq_len, self.num_human_imus, self.imu_dim)
        if out.dim() != 4 or out.shape[:2] != (batch_size, seq_len):
            return torch.zeros(batch_size, seq_len, self.num_human_imus, self.imu_dim, device=device, dtype=dtype)
        if out.shape[2] != self.num_human_imus or out.shape[3] != self.imu_dim:
            flat = out.reshape(batch_size, seq_len, -1)
            flat = self._pad_or_trim_last_dim(flat, self.num_human_imus * self.imu_dim)
            out = flat.view(batch_size, seq_len, self.num_human_imus, self.imu_dim)
        return out

    def _rot6d_delta(self, rot6d: torch.Tensor) -> torch.Tensor:
        B, T, _ = rot6d.shape
        R = rotation_6d_to_matrix(rot6d.reshape(-1, 6)).reshape(B, T, 3, 3)
        rel = torch.matmul(R[:, 1:].transpose(-1, -2), R[:, :-1])
        aa = matrix_to_axis_angle(rel.reshape(-1, 3, 3)).reshape(B, T-1, 3)
        aa = F.pad(aa, (0, 0, 1, 0))
        return aa

    def _compute_init_dir_len(self, hand_pos_0, obj_rotm_0, obj_pos_0):
        vec_world = obj_pos_0 - hand_pos_0
        lb0 = vec_world.norm(dim=-1, keepdim=True)
        unit_world = self._unit_vector(vec_world)
        obj_Rt = obj_rotm_0.transpose(-1, -2)
        oe0 = torch.bmm(obj_Rt, unit_world.unsqueeze(-1)).squeeze(-1)
        return oe0, lb0

    def _prepare_refine_pose(self, human_pose_input, batch_size, seq_len, device, dtype):
        if not isinstance(human_pose_input, torch.Tensor):
            return None
        pose = human_pose_input.to(device=device, dtype=dtype)
        if pose.dim() == 4:
            pose = pose.reshape(batch_size, seq_len, -1)
        if pose.dim() == 3 and pose.shape[:2] == (batch_size, seq_len) and pose.shape[-1] == self.refine_pose_dim:
            return pose
        return None

    def _prepare_refine_root(self, root_trans_input, batch_size, seq_len, device, dtype):
        if not isinstance(root_trans_input, torch.Tensor):
            return None
        root = root_trans_input.to(device=device, dtype=dtype)
        if root.dim() == 2:
            root = root.unsqueeze(1).expand(batch_size, seq_len, 3)
        if root.dim() == 3 and root.shape[:2] == (batch_size, seq_len) and root.shape[-1] == 3:
            return root
        return None

    def _compute_human_refinement(
        self,
        human_pose_input,
        root_trans_input,
        fused_pos,
        obj_vel_feature,
        weights,
        pred_hand_contact_prob,
        lhand_position,
        rhand_position,
        has_object_mask,
        enable_refine,
    ):
        if not enable_refine:
            return {}
        batch_size, seq_len = fused_pos.shape[:2]
        device = fused_pos.device
        dtype = fused_pos.dtype
        base_pose = self._prepare_refine_pose(human_pose_input, batch_size, seq_len, device, dtype)
        base_root = self._prepare_refine_root(root_trans_input, batch_size, seq_len, device, dtype)
        if base_pose is None or base_root is None:
            return {}

        refine_input = torch.cat(
            (
                fused_pos,
                obj_vel_feature,
                weights,
                pred_hand_contact_prob,
                lhand_position,
                rhand_position,
                base_root,
                base_pose,
            ),
            dim=-1,
        )
        residual = self.refine_head(refine_input)
        pose_delta = residual[:, :, : self.refine_pose_dim]
        root_trans_delta = residual[:, :, self.refine_pose_dim :]
        if has_object_mask is not None:
            mask = has_object_mask
            if mask.dim() > 1:
                mask = mask.view(batch_size)
            mask = mask.to(device=device, dtype=dtype).view(batch_size, 1, 1)
            pose_delta = pose_delta * mask
            root_trans_delta = root_trans_delta * mask

        return {
            "pose_delta": pose_delta,
            "root_trans_delta": root_trans_delta,
            "refined_pose": base_pose + pose_delta,
            "refined_root_trans": base_root + root_trans_delta,
        }

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
        human_pose_input: torch.Tensor = None,
        root_trans_input: torch.Tensor = None,
        enable_refine: bool = True,
        known_obj_trans_prefix: torch.Tensor = None,
        known_gating_weights_prefix: torch.Tensor = None,
        **_,
    ):
        """
        前向传播
        
        Args:
            hand_positions: [B, T, 2, 3] 手部位置 (来自HumanPoseModule)
            pred_hand_contact_prob: [B, T, 3] 接触概率 (左/右无条件接触 + 物体运动，来自VelocityContactModule)
            obj_trans_init: [B, 3] 初始物体位置
            obj_imu: [B, T, 9] 物体IMU数据
            human_imu: [B, T, num_imu, imu_dim] 人体IMU数据
            obj_vel_input: [B, T, 3] 物体速度 (来自VelocityContactModule)
            contact_init: [B, 3] 初始接触状态
            has_object_mask: [B] 是否有物体
            human_pose_input: [B, T, num_reduced*6] 人体姿态基线，用于残差微调
            root_trans_input: [B, T, 3] 根节点位移基线，用于残差微调
            enable_refine: 是否输出人体姿态/root trans微调结果
        
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
        obj_imu = self._prepare_obj_imu(obj_imu, bs, seq_len, device, dtype)
        human_imu = self._prepare_human_imu(human_imu, bs, seq_len, device, dtype)

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
        p_move = pred_hand_contact_prob[:, :, 2:3]

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
            contact_init_vec = torch.zeros(bs, 3, device=device, dtype=dtype)
        else:
            if contact_init.dim() == 1:
                contact_init_vec = contact_init.unsqueeze(0).expand(bs, -1)
            else:
                contact_init_vec = contact_init
            if contact_init_vec.shape[-1] > 3:
                contact_init_vec = contact_init_vec[..., :3]
            if contact_init_vec.shape[-1] != 3:
                raise ValueError(f"contact_init must have last dim 3, got {contact_init_vec.shape}")

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
        weights_raw = F.softmax(gate_logits, dim=-1)
        # smoothing逻辑已取消：训练和推理都直接使用四路softmax权重。
        weights = weights_raw

        # 融合位置。online模式下，窗口prefix来自已知历史结果，
        # 只预测窗口内尚未知的尾部帧。
        fused_steps = []
        imu_steps = []
        static_steps = []
        prefix_len = 0
        if isinstance(known_obj_trans_prefix, torch.Tensor):
            prefix = known_obj_trans_prefix.to(device=device, dtype=dtype)
            if prefix.dim() == 2:
                prefix = prefix.unsqueeze(1)
            if prefix.dim() == 3 and prefix.shape[0] == bs and prefix.shape[-1] == 3:
                prefix_len = min(int(prefix.shape[1]), seq_len)
        dt = 1.0 / FRAME_RATE
        for t in range(seq_len):
            if t < prefix_len:
                fused_steps.append(prefix[:, t, :])
                imu_steps.append(torch.zeros(bs, 3, device=device, dtype=dtype))
                static_steps.append(torch.zeros(bs, 3, device=device, dtype=dtype))
                continue

            prev_pos = fused_steps[-1] if t > 0 else obj_trans_init
            pos_imu_integrated = prev_pos + obj_vel_input[:, t, :] * dt
            pos_static = prev_pos
            imu_steps.append(pos_imu_integrated)
            static_steps.append(pos_static)
            fused_steps.append(
                weights[:, t, 0:1] * l_pos_fk[:, t, :] +
                weights[:, t, 1:2] * r_pos_fk[:, t, :] +
                weights[:, t, 2:3] * pos_imu_integrated +
                weights[:, t, 3:4] * pos_static
            )

        fused_pos = torch.stack(fused_steps, dim=1)
        imu_integrated_pos = torch.stack(imu_steps, dim=1)
        static_pos = torch.stack(static_steps, dim=1)

        # 计算速度和加速度
        vel_from_pos = torch.zeros_like(fused_pos)
        if seq_len > 1:
            vel_from_pos = torch.cat(
                (vel_from_pos[:, :1], (fused_pos[:, 1:] - fused_pos[:, :-1]) * FRAME_RATE),
                dim=1,
            )
        acc_from_pos = torch.zeros_like(fused_pos)
        if seq_len > 2:
            acc_from_pos = torch.cat(
                (
                    acc_from_pos[:, :2],
                    (fused_pos[:, 2:] - 2 * fused_pos[:, 1:-1] + fused_pos[:, :-2]) * (FRAME_RATE**2),
                ),
                dim=1,
            )

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
            imu_integrated_pos = imu_integrated_pos * mask
            static_pos = static_pos * mask
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

        results = {
            "pred_obj_trans": fused_pos,
            "gating_weights": weights,
            "gating_weights_raw": weights_raw,
            "pred_obj_vel_from_posdiff": vel_from_pos,
            "pred_obj_acc_from_posdiff": acc_from_pos,
            "obj_vel_input": obj_vel_input,
            "obj_vel_corrected": obj_vel_input,
            "pred_lhand_obj_direction": l_dir,
            "pred_rhand_obj_direction": r_dir,
            "pred_lhand_lb": l_len,
            "pred_rhand_lb": r_len,
            "pred_lhand_obj_trans": l_pos_fk,
            "pred_rhand_obj_trans": r_pos_fk,
            "pred_imu_obj_trans": imu_integrated_pos,
            "pred_static_obj_trans": static_pos,
            "init_lhand_oe_ho": l_oe0,
            "init_rhand_oe_ho": r_oe0,
            "init_lhand_lb": l_lb0,
            "init_rhand_lb": r_lb0,
            "gating_contact_prob": pred_hand_contact_prob,
            "gating_smoothing_applied": False,
        }
        results.update(
            self._compute_human_refinement(
                human_pose_input,
                root_trans_input,
                fused_pos,
                obj_vel_input,
                weights,
                pred_hand_contact_prob,
                lhand_position,
                rhand_position,
                has_object_mask,
                enable_refine,
            )
        )
        return results

    @staticmethod
    def _slice_optional_time(value, start: int, end: int, batch_size: int, seq_len: int):
        return slice_time_value(value, start, end, batch_size, seq_len)

    def _inference_online_sequence(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        online_window: int,
        **kwargs,
    ):
        if hand_positions.dim() == 3:
            batch_size, seq_len, _ = hand_positions.shape
        elif hand_positions.dim() == 4:
            batch_size, seq_len = hand_positions.shape[:2]
        else:
            raise ValueError(f"Unexpected hand_positions shape {hand_positions.shape}")

        if seq_len <= online_window:
            return self.forward(
                hand_positions,
                pred_hand_contact_prob,
                obj_trans_init,
                **kwargs,
            )

        def _slice_kwargs(start: int, end: int):
            return {
                key: self._slice_optional_time(value, start, end, batch_size, seq_len)
                for key, value in kwargs.items()
                if key != "known_obj_trans_prefix"
            }

        warmup_len = int(online_window)
        warmup_out = self.forward(
            hand_positions[:, :warmup_len],
            pred_hand_contact_prob[:, :warmup_len],
            obj_trans_init,
            **_slice_kwargs(0, warmup_len),
        )
        history = warmup_out
        history_acc = TimeDictAccumulator(history, seq_len)
        history = history_acc.current()

        for end in range(warmup_len + 1, seq_len + 1):
            start = end - warmup_len
            prefix = select_time_context(history, start, end - 1).get("pred_obj_trans")
            if isinstance(prefix, torch.Tensor) and prefix.shape[1] > 0:
                step_obj_trans_init = prefix[:, 0]
            else:
                step_obj_trans_init = obj_trans_init
            window_out = self.forward(
                hand_positions[:, start:end],
                pred_hand_contact_prob[:, start:end],
                step_obj_trans_init,
                known_obj_trans_prefix=prefix,
                **_slice_kwargs(start, end),
            )
            latest = take_latest_frame(window_out, batch_size, end - start)
            history = history_acc.append(latest)

        return history_acc.current()

    def inference(
        self,
        hand_positions: torch.Tensor,
        pred_hand_contact_prob: torch.Tensor,
        obj_trans_init: torch.Tensor,
        inference_mode: str = "offline",
        online_window: int = None,
        online_state: dict = None,
        return_online_state: bool = False,
        **kwargs,
    ):
        mode = normalize_inference_mode(inference_mode)
        if mode == "offline":
            output = self.forward(
                hand_positions,
                pred_hand_contact_prob,
                obj_trans_init,
                **kwargs,
            )
            if return_online_state:
                return output, online_state or {}
            return output

        window = resolve_online_window(self.cfg, online_window)
        stream_dict = {
            "hand_positions": hand_positions,
            "pred_hand_contact_prob": pred_hand_contact_prob,
            **{key: value for key, value in kwargs.items() if key != "known_obj_trans_prefix"},
        }
        if isinstance(online_state, dict) and isinstance(online_state.get("data_dict"), dict):
            run_data, previous_len = append_stream_data(online_state["data_dict"], stream_dict, sequence_key="hand_positions")
            hand_positions = run_data["hand_positions"]
            pred_hand_contact_prob = run_data["pred_hand_contact_prob"]
            for key in stream_dict.keys():
                if key not in {"hand_positions", "pred_hand_contact_prob"}:
                    kwargs[key] = run_data.get(key)
        else:
            previous_len = 0
            run_data = stream_dict

        output = self._inference_online_sequence(
            hand_positions,
            pred_hand_contact_prob,
            obj_trans_init,
            window,
            **kwargs,
        )
        state = {"data_dict": run_data, "outputs": output}
        if return_online_state:
            if previous_len > 0:
                batch_size, seq_len = infer_batch_seq(run_data, key="hand_positions")
                output = slice_time_dict(output, previous_len, seq_len, batch_size, seq_len)
            return output, state
        return output

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        """返回空输出"""
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_dir = torch.zeros(batch_size, seq_len, 3, device=device)
        zeros_scalar = torch.zeros(batch_size, seq_len, device=device)
        zeros_weights = torch.zeros(batch_size, seq_len, 4, device=device)
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
            "pred_imu_obj_trans": zeros_pos,
            "pred_static_obj_trans": zeros_pos,
            "gating_contact_prob": torch.zeros(batch_size, seq_len, 3, device=device),
            "init_lhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_rhand_oe_ho": torch.zeros(batch_size, 3, device=device),
            "init_lhand_lb": torch.zeros(batch_size, device=device),
            "init_rhand_lb": torch.zeros(batch_size, device=device),
            "gating_smoothing_applied": False,
        }
