"""
IMUHOI完整模型 - 统一的前向推理接口
"""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from .velocity_contact import VelocityContactModule
from .human_pose import HumanPoseModule
from .object_trans import ObjectTransModule
from .online import (
    append_stream_data,
    concat_time_dicts,
    infer_batch_seq,
    merge_latest_context,
    normalize_inference_mode,
    resolve_online_window,
    select_time_context,
    slice_time_dict,
    take_latest_frame,
    TimeDictAccumulator,
    update_data_inits_from_history,
)
from utils.human_pose import select_hand_anchor_positions, select_wrist_positions


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

    @staticmethod
    def _resolve_hoi_hand_positions(hp_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the palm anchor positions used by VC/OT geometry."""
        if isinstance(hp_out, dict):
            palm_pos = hp_out.get("pred_palm_glb_pos")
            if isinstance(palm_pos, torch.Tensor):
                return palm_pos
            hand_pos = hp_out.get("pred_hand_glb_pos")
            if isinstance(hand_pos, torch.Tensor):
                return hand_pos
        raise KeyError("hp_out must contain pred_palm_glb_pos or pred_hand_glb_pos")

    @staticmethod
    def _resolve_stage1_hoi_hand_positions(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Use the pre-refinement hand anchors when they are retained in history."""
        if isinstance(outputs, dict):
            for key in ("stage1_pred_palm_glb_pos", "stage1_pred_hand_glb_pos"):
                hand_pos = outputs.get(key)
                if isinstance(hand_pos, torch.Tensor):
                    return hand_pos
        return IMUHOIModel._resolve_hoi_hand_positions(outputs)
    
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
                    module_strict = False if name in {"velocity_contact", "object_trans"} else strict
                    load_checkpoint(module, path, self.device, strict=module_strict)
                    print(f"Loaded {name} from {path}")
            else:
                if path:
                    print(f"Warning: {name} checkpoint not found at {path}")

    def _promote_refined_human_outputs(
        self,
        results: Dict[str, torch.Tensor],
        ot_out: Dict[str, torch.Tensor],
        data_dict: Dict[str, torch.Tensor],
    ) -> None:
        refined_pose = ot_out.get("refined_pose")
        refined_root = ot_out.get("refined_root_trans")
        if not isinstance(refined_pose, torch.Tensor) or not isinstance(refined_root, torch.Tensor):
            return

        for key in (
            "p_pred",
            "root_trans_pred",
            "root_vel_pred",
            "root_vel_local_pred",
            "pred_joints_local",
            "pred_joints_global",
            "pred_hand_glb_pos",
            "pred_palm_glb_pos",
            "pred_full_pose_rotmat",
            "pred_full_pose_6d",
        ):
            if key in results and f"stage1_{key}" not in results:
                results[f"stage1_{key}"] = results[key]

        results["p_pred"] = refined_pose
        results["root_trans_pred"] = refined_root

        hp_module = self.human_pose_module
        human_imu = data_dict["human_imu"]
        batch_size, seq_len = human_imu.shape[:2]
        device = refined_pose.device
        dtype = refined_pose.dtype

        root_vel = None
        if hasattr(hp_module, "_compute_root_velocity_from_trans"):
            root_vel = hp_module._compute_root_velocity_from_trans(refined_root)
            if isinstance(root_vel, torch.Tensor):
                results["root_vel_pred"] = root_vel

        try:
            orientation_6d = human_imu[..., -6:].to(device=device, dtype=dtype)
            orientation_mat = rotation_6d_to_matrix(orientation_6d.reshape(-1, 6)).reshape(
                batch_size, seq_len, human_imu.shape[2], 3, 3
            )
            if root_vel is not None:
                root_rot = orientation_mat[:, :, 0]
                results["root_vel_local_pred"] = torch.matmul(
                    root_rot.transpose(-1, -2),
                    root_vel.unsqueeze(-1),
                ).squeeze(-1)

            if getattr(hp_module, "body_model", None) is not None and getattr(hp_module, "body_model_device", None) != device:
                hp_module.body_model = hp_module.body_model.to(device)
                hp_module.body_model_device = device

            joints_local = None
            if hasattr(hp_module, "_compute_fk_joints_batched"):
                joints_local = hp_module._compute_fk_joints_batched(refined_pose, orientation_mat.clone())
            if isinstance(joints_local, torch.Tensor):
                results["pred_joints_local"] = joints_local
                joints_global = joints_local + refined_root.unsqueeze(2)
                results["pred_joints_global"] = joints_global
                if joints_global.shape[2] > 21:
                    results["pred_hand_glb_pos"] = select_wrist_positions(joints_global)
                    results["pred_palm_glb_pos"] = select_hand_anchor_positions(joints_global)

            if hasattr(hp_module, "_reduced_glb_6d_to_full_glb_mat"):
                bt = batch_size * seq_len
                reduced = refined_pose.reshape(bt, -1, 6)
                orientation_flat = orientation_mat[:, :, :6].reshape(bt, 6, 3, 3)
                full_flat = hp_module._reduced_glb_6d_to_full_glb_mat(reduced, orientation_flat)
                full_rotmats = full_flat.reshape(batch_size, seq_len, 24, 3, 3)
                results["pred_full_pose_rotmat"] = full_rotmats
                results["pred_full_pose_6d"] = matrix_to_rotation_6d(full_flat.reshape(-1, 3, 3)).reshape(
                    batch_size, seq_len, 24, 6
                )
        except Exception as exc:
            print(f"Warning: failed to recompute refined human outputs: {exc}")

    def _resolve_refine_human(self, refine_human: Optional[bool]) -> bool:
        if refine_human is None:
            return bool(getattr(self.cfg, "enable_ot_refine", False))
        return bool(refine_human)
    
    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
        refine_human: Optional[bool] = None,
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
            refine_human: 是否使用ObjectTrans输出的人体姿态/root trans微调结果
        
        Returns:
            结果字典，包含各阶段的预测输出
        """
        refine_human = self._resolve_refine_human(refine_human)
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
            'obj_trans_init': data_dict["obj_trans_init"],
            'contact_init': data_dict.get("contact_init"),
            'hp_out': hp_out,
        }
        vc_out = self.velocity_contact_module(vc_input_dict, hp_out=hp_out)
        results.update(vc_out)
        hoi_hand_positions = self._resolve_hoi_hand_positions(hp_out)
        
        # Stage 3: ObjectTrans - 预测物体位置
        has_object = data_dict.get("has_object")
        if use_object_data and (has_object is None or has_object.any()):
            ot_out = self.object_trans_module(
                hoi_hand_positions,
                vc_out["pred_hand_contact_prob"],
                data_dict["obj_trans_init"],
                obj_imu=data_dict["obj_imu"],
                human_imu=human_imu,
                obj_vel_input=vc_out["pred_obj_vel"],
                contact_init=data_dict.get("contact_init"),
                has_object_mask=has_object,
                human_pose_input=hp_out.get("p_pred"),
                root_trans_input=hp_out.get("root_trans_pred"),
                enable_refine=refine_human,
            )
            results.update(ot_out)
            if refine_human:
                self._promote_refined_human_outputs(results, ot_out, data_dict)
            
            # 如果需要计算FK方式的物体位置（用于比较）
            if compute_fk:
                obj_imu = data_dict["obj_imu"]
                obj_rot6d = obj_imu[..., 3:9]
                obj_rotm = rotation_6d_to_matrix(obj_rot6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
                results["pred_obj_trans_fk"] = self._fk_obj_trans_baseline_hard(
                    vc_out["pred_hand_contact_prob"],
                    results.get("pred_palm_glb_pos", hoi_hand_positions),
                    obj_rotm,
                    data_dict["obj_trans_init"],
                )
        
        results["has_object"] = has_object
        return results

    def _build_hp_input_dict(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hp_input_dict = {
            "human_imu": data_dict["human_imu"],
            "v_init": data_dict["v_init"],
            "p_init": data_dict["p_init"],
        }
        if self.no_trans:
            hp_input_dict["trans_gt"] = data_dict["trans_gt"]
        else:
            hp_input_dict["trans_init"] = data_dict["trans_init"]
        return hp_input_dict

    def _compute_fk_output(
        self,
        results: Dict[str, torch.Tensor],
        vc_out: Dict[str, torch.Tensor],
        data_dict: Dict[str, torch.Tensor],
        obj_trans_init: torch.Tensor,
    ) -> None:
        human_imu = data_dict["human_imu"]
        batch_size, seq_len = human_imu.shape[:2]
        obj_imu = data_dict["obj_imu"]
        obj_rot6d = obj_imu[..., 3:9]
        obj_rotm = rotation_6d_to_matrix(obj_rot6d.reshape(-1, 6)).reshape(batch_size, seq_len, 3, 3)
        hand_pos = self._resolve_hoi_hand_positions(results)
        results["pred_obj_trans_fk"] = self._fk_obj_trans_baseline_hard(
            vc_out["pred_hand_contact_prob"],
            hand_pos,
            obj_rotm,
            obj_trans_init,
        )

    @staticmethod
    def _merge_stage_context(prefix: Dict[str, torch.Tensor], latest: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        stage_prefix = {key: prefix[key] for key in latest.keys() if key in prefix}
        return merge_latest_context(stage_prefix, latest)

    @staticmethod
    def _merge_raw_human_pose_context(
        prefix: Dict[str, torch.Tensor],
        latest: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Merge Stage-1 history without feeding OT-refined poses back upstream.

        OT refinement is an output correction, not an additional causal input to
        HP, VC, or the OT predictor.  Online history may contain promoted
        refined values, while the original Stage-1 values are kept under
        ``stage1_*``.  Prefer those values so online inputs match training.
        """
        raw_prefix = {}
        for key in latest:
            stage1_key = f"stage1_{key}"
            if stage1_key in prefix:
                raw_prefix[key] = prefix[stage1_key]
            elif key in prefix:
                raw_prefix[key] = prefix[key]
        return merge_latest_context(raw_prefix, latest)

    @staticmethod
    def _stage1_history_for_inits(history: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Expose retained raw Stage-1 values under their ordinary init keys."""
        if not isinstance(history, dict):
            return history
        raw_history = dict(history)
        for key, value in history.items():
            if key.startswith("stage1_"):
                raw_history[key[len("stage1_"):]] = value
        return raw_history

    @staticmethod
    def _append_stream_window(
        previous: Optional[Dict[str, torch.Tensor]],
        current: Dict[str, torch.Tensor],
        max_len: int,
        sequence_key: str = "human_imu",
    ) -> Dict[str, torch.Tensor]:
        """Append a chunk and retain only the causal sliding window."""
        merged, _ = append_stream_data(previous, current, sequence_key=sequence_key)
        batch_size, seq_len = infer_batch_seq(merged, key=sequence_key)
        if seq_len <= max_len:
            return merged
        return slice_time_dict(merged, seq_len - max_len, seq_len, batch_size, seq_len)

    @staticmethod
    def _uses_stateful_ot(module: nn.Module) -> bool:
        mode_fn = getattr(module, "online_mode", None)
        return callable(mode_fn) and mode_fn() == "stateful"

    def _inference_online_sequence(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
        refine_human: Optional[bool] = None,
        online_window: int = 120,
    ) -> Dict[str, torch.Tensor]:
        if (
            use_object_data
            and self._uses_stateful_ot(self.object_trans_module)
        ):
            return self._inference_online_sequence_stateful_ot(
                data_dict,
                use_object_data=use_object_data,
                compute_fk=compute_fk,
                refine_human=refine_human,
                online_window=online_window,
            )

        refine_human = self._resolve_refine_human(refine_human)
        batch_size, seq_len = infer_batch_seq(data_dict)
        if seq_len <= online_window:
            return self.forward(
                data_dict,
                use_object_data=use_object_data,
                compute_fk=compute_fk,
                refine_human=refine_human,
            )

        warmup_len = int(online_window)
        warmup_data = slice_time_dict(data_dict, 0, warmup_len, batch_size, seq_len)
        history = self.forward(
            warmup_data,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
        )
        history_acc = TimeDictAccumulator(history, seq_len)
        history = history_acc.current()

        for end in range(warmup_len + 1, seq_len + 1):
            start = end - warmup_len
            window_data = slice_time_dict(data_dict, start, end, batch_size, seq_len)
            window_data = update_data_inits_from_history(
                window_data,
                self._stage1_history_for_inits(history),
                index=start - 1,
            )
            window_len = end - start
            prefix = select_time_context(history, start, end - 1)

            hp_out_raw = self.human_pose_module.forward(self._build_hp_input_dict(window_data))
            hp_latest = take_latest_frame(hp_out_raw, batch_size, window_len)
            hp_context = self._merge_raw_human_pose_context(prefix, hp_latest)

            vc_input_dict = {
                "human_imu": window_data["human_imu"],
                "obj_imu": window_data["obj_imu"],
                "hand_vel_glb_init": window_data["hand_vel_glb_init"],
                "obj_vel_init": window_data["obj_vel_init"],
                "obj_trans_init": window_data["obj_trans_init"],
                "contact_init": window_data.get("contact_init"),
                "hp_out": hp_context,
            }
            vc_out_raw = self.velocity_contact_module.forward(vc_input_dict, hp_out=hp_context)
            vc_latest = take_latest_frame(vc_out_raw, batch_size, window_len)
            vc_context = self._merge_stage_context(prefix, vc_latest)

            results = {}
            results.update(hp_context)
            results.update(vc_context)

            has_object = window_data.get("has_object")
            if use_object_data and (has_object is None or has_object.any()):
                obj_prefix = prefix.get("pred_obj_trans")
                gate_prefix = prefix.get("gating_weights")
                if isinstance(obj_prefix, torch.Tensor) and obj_prefix.shape[1] > 0:
                    ot_obj_trans_init = obj_prefix[:, 0]
                else:
                    ot_obj_trans_init = window_data["obj_trans_init"]
                ot_out = self.object_trans_module.forward(
                    self._resolve_hoi_hand_positions(hp_context),
                    vc_context["pred_hand_contact_prob"],
                    ot_obj_trans_init,
                    obj_imu=window_data["obj_imu"],
                    human_imu=window_data["human_imu"],
                    obj_vel_input=vc_context["pred_obj_vel"],
                    contact_init=window_data.get("contact_init"),
                    has_object_mask=has_object,
                    human_pose_input=hp_context.get("p_pred"),
                    root_trans_input=hp_context.get("root_trans_pred"),
                    enable_refine=refine_human,
                    known_obj_trans_prefix=obj_prefix,
                    known_gating_weights_prefix=gate_prefix,
                )
                results.update(ot_out)
                if refine_human:
                    self._promote_refined_human_outputs(results, ot_out, window_data)
                if compute_fk:
                    self._compute_fk_output(results, vc_context, window_data, ot_obj_trans_init)

            results["has_object"] = has_object
            latest = take_latest_frame(results, batch_size, window_len)
            history = history_acc.append(latest)

        return history_acc.current()

    def _inference_stateful_ot_stream(
        self,
        data_dict: Dict[str, torch.Tensor],
        *,
        online_state: Optional[dict],
        use_object_data: bool,
        compute_fk: bool,
        refine_human: Optional[bool],
        online_window: int,
    ):
        """Run bounded-memory online inference with a persistent OT predictor.

        HP, VC, and the optional refine head deliberately retain the existing
        sliding-window semantics.  Only the shared ObjectTrans predictor keeps
        its learned LSTM state and fused OT state across calls, which isolates
        the experimental variable without silently changing upstream models.
        """
        refine_human = self._resolve_refine_human(refine_human)
        batch_size, chunk_len = infer_batch_seq(data_dict)
        if chunk_len < 1:
            raise ValueError("Stateful online inference requires at least one frame.")

        state = dict(online_state or {})
        input_window = state.get("input_window")
        output_window = state.get("output_window")
        prediction_state = state.get("ot_prediction_state")

        for name, window_data, sequence_key in (
            ("input_window", input_window, "human_imu"),
            ("output_window", output_window, "p_pred"),
        ):
            if not isinstance(window_data, dict):
                continue
            state_batch, _ = infer_batch_seq(window_data, key=sequence_key)
            if state_batch != batch_size:
                raise ValueError(
                    f"Stateful online {name} batch size ({state_batch}) does not match "
                    f"the current chunk ({batch_size}); reset online_state between streams."
                )
        if (
            isinstance(output_window, dict)
            and isinstance(output_window.get("pred_obj_trans"), torch.Tensor)
            and prediction_state is None
        ):
            raise ValueError(
                "Stateful online state has prior outputs but no OT prediction state; "
                "reset online_state instead of mixing incompatible streams."
            )

        emitted = []
        for frame_index in range(chunk_len):
            frame_data = slice_time_dict(data_dict, frame_index, frame_index + 1, batch_size, chunk_len)
            input_window = self._append_stream_window(input_window, frame_data, online_window, sequence_key="human_imu")
            _, window_len = infer_batch_seq(input_window)

            history = output_window if isinstance(output_window, dict) else {}
            history_len = 0
            if history:
                history_batch, history_len = infer_batch_seq(history, key="p_pred")
                if history_batch != batch_size:
                    raise ValueError("Stateful online output history has an incompatible batch size.")

            # Once the input window advances, its predecessor is the first
            # frame retained in output_window.  Before that, all frames still
            # share the original start state and no init update is needed.
            has_predecessor = history_len >= online_window
            window_data = input_window
            if has_predecessor:
                window_data = update_data_inits_from_history(
                    window_data,
                    self._stage1_history_for_inits(history),
                    index=0,
                )
                prefix = select_time_context(history, 1, history_len)
            else:
                prefix = history

            hp_out_raw = self.human_pose_module.forward(self._build_hp_input_dict(window_data))
            hp_latest = take_latest_frame(hp_out_raw, batch_size, window_len)
            hp_context = self._merge_raw_human_pose_context(prefix, hp_latest)

            vc_input_dict = {
                "human_imu": window_data["human_imu"],
                "obj_imu": window_data["obj_imu"],
                "hand_vel_glb_init": window_data["hand_vel_glb_init"],
                "obj_vel_init": window_data["obj_vel_init"],
                "obj_trans_init": window_data["obj_trans_init"],
                "contact_init": window_data.get("contact_init"),
                "hp_out": hp_context,
            }
            vc_out_raw = self.velocity_contact_module.forward(vc_input_dict, hp_out=hp_context)
            vc_latest = take_latest_frame(vc_out_raw, batch_size, window_len)
            vc_context = self._merge_stage_context(prefix, vc_latest)

            results: Dict[str, torch.Tensor] = {}
            results.update(hp_context)
            results.update(vc_context)
            has_object = window_data.get("has_object")
            if use_object_data and (has_object is None or has_object.any()):
                ot_latest, prediction_state = self.object_trans_module.forward(
                    self._resolve_hoi_hand_positions(hp_latest),
                    vc_latest["pred_hand_contact_prob"],
                    window_data["obj_trans_init"],
                    obj_imu=window_data["obj_imu"][:, -1:],
                    human_imu=window_data["human_imu"][:, -1:],
                    obj_vel_input=vc_latest["pred_obj_vel"],
                    contact_init=window_data.get("contact_init"),
                    has_object_mask=has_object,
                    enable_refine=False,
                    compute_refine=False,
                    prediction_state=prediction_state,
                    return_prediction_state=True,
                )
                ot_context = self._merge_stage_context(prefix, ot_latest)
                ot_context.update(
                    self.object_trans_module.compute_human_refinement_from_prediction(
                        ot_context,
                        self._resolve_hoi_hand_positions(hp_context),
                        vc_context["pred_hand_contact_prob"],
                        obj_vel_input=vc_context["pred_obj_vel"],
                        human_pose_input=hp_context.get("p_pred"),
                        root_trans_input=hp_context.get("root_trans_pred"),
                        has_object_mask=has_object,
                        enable_refine=refine_human,
                    )
                )
                results.update(ot_context)
                if refine_human:
                    self._promote_refined_human_outputs(results, ot_context, window_data)
                if compute_fk:
                    self._compute_fk_output(results, vc_context, window_data, data_dict["obj_trans_init"])

            results["has_object"] = has_object
            latest = take_latest_frame(results, batch_size, window_len)
            output_window = self._append_stream_window(output_window, latest, online_window, sequence_key="p_pred")
            emitted.append(latest)

        state = {
            "input_window": input_window,
            "output_window": output_window,
            "ot_prediction_state": prediction_state,
        }
        return concat_time_dicts(emitted), state

    def _inference_online_sequence_stateful_ot(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
        refine_human: Optional[bool] = None,
        online_window: int = 120,
    ) -> Dict[str, torch.Tensor]:
        """Run one complete sequence through the same persistent streaming path."""
        output, _ = self._inference_stateful_ot_stream(
            data_dict,
            online_state=None,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
            online_window=online_window,
        )
        return output

    def inference(
        self,
        data_dict: Dict[str, torch.Tensor],
        gt_targets: Optional[Dict[str, torch.Tensor]] = None,
        use_object_data: bool = True,
        compute_fk: bool = False,
        interaction_use_human_pred: bool = True,
        refine_human: Optional[bool] = None,
        inference_mode: str = "offline",
        online_window: Optional[int] = None,
        online_state: Optional[dict] = None,
        return_online_state: bool = False,
        **_,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluation-compatible inference entrypoint.

        The RNN pipeline is deterministic and does not need diffusion sampling or
        GT target conditioning.  Keep the signature aligned with the DiT
        pipeline so shared eval/visualization code can call either backend.
        """
        _ = gt_targets, interaction_use_human_pred
        mode = normalize_inference_mode(inference_mode)
        if mode == "offline":
            output = self.forward(
                data_dict,
                use_object_data=use_object_data,
                compute_fk=compute_fk,
                refine_human=refine_human,
            )
            if return_online_state:
                return output, online_state or {}
            return output

        window = resolve_online_window(self.cfg, online_window)
        if use_object_data and self._uses_stateful_ot(self.object_trans_module):
            output, state = self._inference_stateful_ot_stream(
                data_dict,
                online_state=online_state,
                use_object_data=use_object_data,
                compute_fk=compute_fk,
                refine_human=refine_human,
                online_window=window,
            )
            if return_online_state:
                return output, state
            return output

        run_data, previous_len = append_stream_data(
            online_state.get("data_dict") if isinstance(online_state, dict) else None,
            data_dict,
        )
        output = self._inference_online_sequence(
            run_data,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
            online_window=window,
        )
        state = {"data_dict": run_data, "outputs": output}
        if return_online_state:
            if previous_len > 0:
                batch_size, seq_len = infer_batch_seq(run_data)
                output = slice_time_dict(output, previous_len, seq_len, batch_size, seq_len)
            return output, state
        return output


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
