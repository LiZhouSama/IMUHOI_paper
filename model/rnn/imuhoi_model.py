"""
IMUHOI完整模型 - 统一的前向推理接口
"""
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from .velocity_contact import (
    VelocityContactModule,
    build_velocity_contact_module,
    normalize_velocity_contact_arch,
)
from .human_pose import HumanPoseModule
from .object_trans import ObjectTransModule
from .object_trans_v3 import V3ObjectTransModule
from .object_trans_legacy import LegacyObjectTransModule
from .object_trans_IMUHOI import (
    IMUHOIObjectTransModule,
    is_imuhoi_object_trans_arch,
)
from .object_trans_IMUHOI_4heads import IMUHOI4HeadsObjectTransModule
from .object_trans_legacy_mesh import LegacyMeshObjectTransModule
from .object_trans_mergeHeads_obsEncoder import (
    ObjectTransModule as MergeHeadsObsObjectTransModule,
)
from .lb_vq_diffusion import FrozenLBPriorPredictor
from .online import (
    append_stream_data,
    concat_time_dicts,
    infer_batch_seq,
    merge_latest_context,
    normalize_inference_mode,
    resolve_online_lookahead,
    resolve_online_window,
    select_time_context,
    slice_time_dict,
    TimeDictAccumulator,
    update_data_inits_from_history,
)
from configs import FRAME_RATE
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

    def __init__(
        self,
        cfg,
        device,
        no_trans: bool = False,
        object_trans_arch: Optional[str] = None,
        velocity_contact_arch: Optional[str] = None,
    ):
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
        self.velocity_contact_arch = normalize_velocity_contact_arch(
            velocity_contact_arch or getattr(cfg, "velocity_contact_arch", "current")
        )
        self.object_trans_arch = str(
            object_trans_arch or getattr(cfg, "object_trans_arch", "current")
        ).strip().lower()
        
        # 创建三个模块
        self.velocity_contact_module = (
            VelocityContactModule(cfg)
            if self.velocity_contact_arch == "current"
            else build_velocity_contact_module(cfg, velocity_contact_arch=self.velocity_contact_arch)
        )
        self.human_pose_module = HumanPoseModule(cfg, device, no_trans=no_trans)
        if self.object_trans_arch == "legacy":
            self.object_trans_module = LegacyObjectTransModule(cfg)
        elif self.object_trans_arch == "imuhoi":
            self.object_trans_module = IMUHOIObjectTransModule(cfg)
        elif self.object_trans_arch == "imuhoi_4heads":
            self.object_trans_module = IMUHOI4HeadsObjectTransModule(cfg)
        elif self.object_trans_arch == "legacy_mesh":
            self.object_trans_module = LegacyMeshObjectTransModule(cfg)
        elif self.object_trans_arch == "mergeheads_obs":
            self.object_trans_module = MergeHeadsObsObjectTransModule(cfg)
        elif self.object_trans_arch == "current":
            self.object_trans_module = ObjectTransModule(cfg)
        elif self.object_trans_arch == "v2":
            from model.rnn.object_trans_v2 import V2ObjectTransModule

            self.object_trans_module = V2ObjectTransModule(cfg)
        elif self.object_trans_arch == "v3":
            self.object_trans_module = V3ObjectTransModule(cfg)
        else:
            raise ValueError(
                "object_trans_arch must be 'current', 'v2', 'v3', 'legacy', 'imuhoi', "
                "'imuhoi_4heads', "
                "'legacy_mesh', or 'mergeHeads_obs', "
                f"got {self.object_trans_arch!r}"
            )
        self.lb_prior_predictor = None
        if bool(getattr(cfg, "enable_ot_lb_prior", False)):
            if self.object_trans_arch in {"current", "v2", "v3"}:
                self.lb_prior_predictor = FrozenLBPriorPredictor(cfg)
            elif self.object_trans_arch != "mergeheads_obs":
                raise ValueError("l_b VQ/Diffusion prior supports object_trans_arch=current, v2, or v3 only")

    def _predict_lb_prior(self, hp_out, vc_out, obj_imu):
        predictor = getattr(self, "lb_prior_predictor", None)
        if predictor is None:
            return None
        return predictor(hp_out, vc_out, obj_imu)

    def _online_lb_stats_enabled(self) -> bool:
        """Contact-segment l_b statistics measured from the causal history.

        Enabled by default for the current ObjectTrans architecture whenever
        the frozen VQ/diffusion prior is not in use.  Offline single-pass
        inference intentionally stays cold (no history -> init-geometry
        fallback inside ObjectTrans), matching the training-time drop/short
        prefix augmentation.
        """
        if getattr(self, "lb_prior_predictor", None) is not None:
            return False
        if not isinstance(self.object_trans_module, ObjectTransModule):
            return False
        return bool(getattr(self.cfg, "object_trans_lb_stats", True))

    def _compute_online_lb_stats(self, history) -> Optional[torch.Tensor]:
        """Per-hand [q10, q50, q90] of ||pred_obj_trans - hand|| on contact frames."""
        if not isinstance(history, dict) or not history:
            return None
        pred_obj_trans = history.get("pred_obj_trans")
        contact_prob = history.get("pred_hand_contact_prob")
        if not isinstance(pred_obj_trans, torch.Tensor) or not isinstance(contact_prob, torch.Tensor):
            return None
        try:
            hand_positions = self._resolve_stage1_hoi_hand_positions(history)
        except KeyError:
            return None
        return ObjectTransModule.compute_online_lb_quantiles(
            pred_obj_trans,
            hand_positions,
            contact_prob,
            contact_threshold=float(getattr(self.cfg, "lb_stats_contact_threshold", 0.5)),
            min_contact_frames=int(getattr(self.cfg, "lb_stats_min_frames", 10)),
            # Long sequences: statistics measured from the model's own history
            # feed back into the bone predictor; a bounded window limits how
            # much accumulated error the prior can carry (default keeps the
            # historical 2048-frame cap).
            max_history_frames=int(getattr(self.cfg, "lb_stats_max_history_frames", 2048)),
            tail_frames=int(getattr(self.cfg, "lb_stats_tail_frames", 0)),
        )

    def _apply_vc_segment_contact_if_enabled(
        self,
        vc_out: Dict[str, torch.Tensor],
        contact_init: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Adapt endpoint-aware VC output before OT consumes contact.

        Endpoint segment decoding is inference-only and non-differentiable.
        Keep training on the frame-level contact head, matching the staged VC
        loss design, and remain compatible with VC implementations that do not
        expose the optional decoder.
        """
        if self.training or not bool(getattr(self.cfg, "contact_from_segments", False)):
            return vc_out
        apply_segment_contact = getattr(
            self.velocity_contact_module,
            "_apply_segment_contact",
            None,
        )
        if not callable(apply_segment_contact):
            return vc_out
        return apply_segment_contact(vc_out, contact_init=contact_init)

    # Diagnostic oracles (eval_batch_online --oracle_vc_*): temporal [B, T, 3]
    # inputs whose finite entries replace the VelocityContact output before
    # anything downstream reads it; NaN entries keep the prediction.
    VC_ORACLE_KEYS = (
        ("oracle_vc_contact", "pred_hand_contact_prob"),
        ("oracle_vc_obj_vel", "pred_obj_vel"),
    )

    @classmethod
    def _apply_vc_oracle(
        cls,
        vc_out: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if not any(isinstance(data.get(key), torch.Tensor) for key, _ in cls.VC_ORACLE_KEYS):
            return vc_out
        vc_out = dict(vc_out)
        for oracle_key, output_key in cls.VC_ORACLE_KEYS:
            oracle = data.get(oracle_key)
            pred = vc_out.get(output_key)
            if not isinstance(oracle, torch.Tensor) or not isinstance(pred, torch.Tensor):
                continue
            oracle = oracle[:, : pred.shape[1]].to(device=pred.device, dtype=pred.dtype)
            head = pred[..., : oracle.shape[-1]]
            head = torch.where(torch.isfinite(oracle), oracle, head)
            vc_out[output_key] = torch.cat((head, pred[..., oracle.shape[-1]:]), dim=-1)
        contact = vc_out.get("pred_hand_contact_prob")
        if isinstance(data.get("oracle_vc_contact"), torch.Tensor) and isinstance(contact, torch.Tensor):
            if isinstance(vc_out.get("pred_obj_move_prob"), torch.Tensor):
                vc_out["pred_obj_move_prob"] = contact[..., 2:3]
            if isinstance(vc_out.get("pred_hand_contact_prob_marginal"), torch.Tensor):
                vc_out["pred_hand_contact_prob_marginal"] = contact[..., :2]
            vc_out.pop("pred_hand_contact_prob_frame", None)
        return vc_out

    def _vc_window_contact_init(self, window_data: Dict[str, torch.Tensor]):
        """VelocityContact's window initializer; diagnostic overrides only.

        ``eval_vc_contact_init``: gt (GT contact at the window start, from
        ``oracle_vc_contact_init``), neutral (0.5) or zero.  ObjectTrans keeps
        the published initializer either way.
        """
        init = window_data.get("contact_init")
        mode = getattr(self.cfg, "eval_vc_contact_init", None)
        if not mode or not isinstance(init, torch.Tensor):
            return init
        if mode == "gt":
            return window_data["oracle_vc_contact_init"][:, 0].to(init)
        if mode == "neutral":
            return torch.full_like(init, 0.5)
        if mode == "zero":
            return torch.zeros_like(init)
        raise ValueError(f"unknown eval_vc_contact_init {mode!r}")

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

    def _resolve_ot_hand_positions(
        self,
        hp_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Use wrists for the EgoIMU IMUHOI OT and palms for newer OTs."""
        if is_imuhoi_object_trans_arch(
            getattr(self, "object_trans_arch", "current")
        ):
            wrist_pos = hp_out.get("pred_hand_glb_pos")
            if isinstance(wrist_pos, torch.Tensor):
                return wrist_pos
            raise KeyError("IMUHOI ObjectTrans requires pred_hand_glb_pos")
        return self._resolve_hoi_hand_positions(hp_out)

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
        derive: bool = True,
    ) -> None:
        """Publish the OT-refined pose and root, keeping Stage-1 under ``stage1_*``.

        ``derive=False`` skips the quantities computed from them (root
        velocity, FK joints, full-pose rotations); the online loop does those
        once for the whole published sequence in
        ``_derive_refined_human_outputs`` because no later window reads them.
        """
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
        if derive:
            self._derive_refined_human_outputs(results, data_dict)

    # Frame rows per FK call when the whole published sequence is derived at
    # once (batch x time); bounds the transient memory of the batched FK.
    DERIVE_CHUNK_ROWS = 16384

    def _derive_refined_human_outputs(
        self,
        results: Dict[str, torch.Tensor],
        data_dict: Dict[str, torch.Tensor],
        chunk_rows: Optional[int] = None,
    ) -> None:
        """Root velocity, FK joints and full-pose rotations of the published pose.

        Everything here is per frame apart from the root velocity (a backward
        difference of ``root_trans_pred``) and the sensor-orientation smoothing,
        which runs over all of ``data_dict["human_imu"]``: that input may extend
        past the published frames, as the lookahead does, so every frame sees
        the same neighbours it saw inside its own window.
        """
        refined_pose = results["p_pred"]
        refined_root = results["root_trans_pred"]
        hp_module = self.human_pose_module
        human_imu = data_dict["human_imu"]
        batch_size, seq_len = refined_pose.shape[:2]
        device = refined_pose.device
        dtype = refined_pose.dtype

        root_vel = None
        if hasattr(hp_module, "_compute_root_velocity_from_trans"):
            root_vel = hp_module._compute_root_velocity_from_trans(refined_root)
            if isinstance(root_vel, torch.Tensor):
                results["root_vel_pred"] = root_vel

        try:
            # Same (optionally smoothed) sensor orientations as the HumanPose FK, so
            # the refined skeleton does not re-import the raw orientation noise.
            if hasattr(hp_module, "fk_orientation_6d"):
                orientation_6d = hp_module.fk_orientation_6d(human_imu)
            else:
                orientation_6d = human_imu[..., -6:]
            orientation_6d = orientation_6d[:, :seq_len].to(device=device, dtype=dtype)
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

            step = seq_len
            if chunk_rows:
                step = max(1, int(chunk_rows) // max(batch_size, 1))
            joints_parts, full_parts = [], []
            for start in range(0, seq_len, step):
                pose_part = refined_pose[:, start:start + step]
                orient_part = orientation_mat[:, start:start + step]
                part_len = int(pose_part.shape[1])
                if hasattr(hp_module, "_compute_fk_joints_batched"):
                    joints_parts.append(hp_module._compute_fk_joints_batched(
                        pose_part,
                        orient_part.clone(),
                        data_dict.get("body_model_type"),
                    ))
                if hasattr(hp_module, "_reduced_glb_6d_to_full_glb_mat"):
                    bt = batch_size * part_len
                    full_parts.append(hp_module._reduced_glb_6d_to_full_glb_mat(
                        pose_part.reshape(bt, -1, 6),
                        orient_part[:, :, :6].reshape(bt, 6, 3, 3),
                    ).reshape(batch_size, part_len, 24, 3, 3))

            if joints_parts and all(isinstance(part, torch.Tensor) for part in joints_parts):
                joints_local = torch.cat(joints_parts, dim=1)
                results["pred_joints_local"] = joints_local
                joints_global = joints_local + refined_root.unsqueeze(2)
                results["pred_joints_global"] = joints_global
                if joints_global.shape[2] > 21:
                    results["pred_hand_glb_pos"] = select_wrist_positions(joints_global)
                    results["pred_palm_glb_pos"] = select_hand_anchor_positions(joints_global)

            if full_parts:
                full_rotmats = torch.cat(full_parts, dim=1)
                results["pred_full_pose_rotmat"] = full_rotmats
                results["pred_full_pose_6d"] = matrix_to_rotation_6d(full_rotmats.reshape(-1, 3, 3)).reshape(
                    batch_size, seq_len, 24, 6
                )
        except Exception as exc:
            print(f"Warning: failed to recompute refined human outputs: {exc}")

    def _resolve_refine_human(self, refine_human: Optional[bool]) -> bool:
        if is_imuhoi_object_trans_arch(
            getattr(self, "object_trans_arch", "current")
        ):
            return False
        if refine_human is None:
            return bool(getattr(self.cfg, "enable_ot_refine", False))
        return bool(refine_human)
    
    def forward(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
        refine_human: Optional[bool] = None,
        feedback_control=None,
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
            'body_model_type': data_dict.get('body_model_type'),
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
        vc_out = self._apply_vc_segment_contact_if_enabled(
            vc_out,
            data_dict.get("contact_init"),
        )
        vc_out = self._apply_vc_oracle(vc_out, data_dict)
        results.update(vc_out)
        hoi_hand_positions = self._resolve_ot_hand_positions(hp_out)
        
        # Stage 3: ObjectTrans - 预测物体位置
        has_object = data_dict.get("has_object")
        if use_object_data and (has_object is None or has_object.any()):
            lb_prior_quantiles = self._predict_lb_prior(
                hp_out, vc_out, data_dict["obj_imu"]
            )
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
                lb_prior_quantiles=lb_prior_quantiles,
                enable_refine=refine_human,
                feedback_control=feedback_control,
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
            "body_model_type": data_dict.get("body_model_type"),
        }
        if self.no_trans:
            hp_input_dict["trans_gt"] = data_dict["trans_gt"]
        else:
            hp_input_dict["trans_init"] = data_dict["trans_init"]
        for key in (
            "prev_foot_pos",
        ):
            if key in data_dict:
                hp_input_dict[key] = data_dict[key]
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
        hand_pos = self._resolve_ot_hand_positions(results)
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

    def _finalize_hp_online_frame(
        self,
        frame: Dict[str, torch.Tensor],
        history: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        finalizer = getattr(self.human_pose_module, "finalize_online_frame", None)
        if not callable(finalizer):
            return frame
        return finalizer(frame, self._stage1_history_for_inits(history))

    @staticmethod
    def _build_feedback_control(
        data_dict: Dict[str, torch.Tensor],
        start: int,
        end: int,
        mode: str = "normal",
        perturbation=None,
        injection_frame: Optional[int] = None,
        target_frame: Optional[int] = None,
    ):
        mode = str(mode or "normal").strip().lower()
        if mode == "normal":
            return None
        if mode not in {"gt_prev", "perturb"}:
            raise ValueError("feedback_mode must be one of normal, gt_prev, perturb")

        human_imu = data_dict.get("human_imu")
        if not isinstance(human_imu, torch.Tensor):
            raise ValueError("feedback diagnostics require tensor human_imu")
        batch_size = int(human_imu.shape[0])
        window_len = int(end - start)
        if window_len < 1:
            return None
        device = human_imu.device
        dtype = human_imu.dtype

        if mode == "gt_prev":
            obj_trans_gt = data_dict.get("obj_trans_gt")
            obj_trans_init = data_dict.get("obj_trans_init")
            if not isinstance(obj_trans_gt, torch.Tensor):
                raise ValueError("gt_prev feedback requires data_dict['obj_trans_gt']")
            if not isinstance(obj_trans_init, torch.Tensor):
                raise ValueError("gt_prev feedback requires data_dict['obj_trans_init']")
            obj_trans_gt = obj_trans_gt.to(device=device, dtype=dtype)
            obj_trans_init = obj_trans_init.to(device=device, dtype=dtype)
            if obj_trans_gt.dim() == 2:
                obj_trans_gt = obj_trans_gt.unsqueeze(0)
            if obj_trans_init.dim() == 1:
                obj_trans_init = obj_trans_init.unsqueeze(0)
            if (
                obj_trans_gt.dim() != 3
                or obj_trans_gt.shape[0] != batch_size
                or obj_trans_gt.shape[-1] != 3
                or obj_trans_init.shape != (batch_size, 3)
            ):
                raise ValueError("feedback GT object positions must have shape [B,T,3]")

            total_len = int(obj_trans_gt.shape[1])
            global_indices = torch.arange(
                int(start), int(end), device=device, dtype=torch.long
            )

            def _lookup(indices):
                valid = (indices >= 0) & (indices < total_len)
                clamped = indices.clamp(0, max(total_len - 1, 0))
                values = obj_trans_gt.index_select(1, clamped)
                initial = obj_trans_init.unsqueeze(1).expand(-1, indices.numel(), -1)
                return torch.where(valid.view(1, -1, 1), values, initial)

            previous = _lookup(global_indices - 1)
            previous_previous = _lookup(global_indices - 2)
            return {
                "mode": "gt_prev",
                "prev_obj_trans": previous,
                "prev_prev_obj_trans": previous_previous,
                "prev_obj_delta": previous - previous_previous,
            }

        delta = torch.zeros(
            batch_size, window_len, 3, device=device, dtype=dtype
        )
        injection_mask = torch.zeros(
            batch_size, window_len, device=device, dtype=torch.bool
        )
        local_injection = None
        if injection_frame is not None:
            injection_frame = int(injection_frame)
            if target_frame is None:
                if int(start) == 0 and int(start) <= injection_frame < int(end):
                    local_injection = injection_frame - int(start)
            elif int(target_frame) == injection_frame:
                local_injection = int(target_frame) - int(start)
        if local_injection is not None and 0 <= local_injection < window_len:
            value = torch.as_tensor(perturbation, device=device, dtype=dtype)
            if value.dim() == 1:
                value = value.unsqueeze(0).expand(batch_size, -1)
            elif value.dim() == 2 and value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, -1)
            if value.shape != (batch_size, 3):
                raise ValueError(
                    f"feedback perturbation must have shape [{batch_size},3], "
                    f"got {tuple(value.shape)}"
                )
            delta[:, local_injection] = value
            injection_mask[:, local_injection] = True
        return {
            "mode": "perturb",
            "perturbation": delta,
            "injection_mask": injection_mask,
        }

    @staticmethod
    def _stack_chunk_inputs(dicts):
        """Stack per-window input dicts along the batch dimension."""
        if len(dicts) == 1:
            return dicts[0]
        stacked = {}
        for key in dicts[0]:
            values = [item.get(key) for item in dicts]
            if all(isinstance(value, torch.Tensor) for value in values):
                stacked[key] = torch.cat(values, dim=0)
            elif all(isinstance(value, (list, tuple)) for value in values):
                stacked[key] = [entry for value in values for entry in value]
            elif all(isinstance(value, dict) for value in values):
                stacked[key] = IMUHOIModel._stack_chunk_inputs(values)
            else:
                stacked[key] = values[0]
        return stacked

    @staticmethod
    def _split_chunk_output(output, count, batch_size):
        """Split a chunk-batched module output back into per-window dicts."""
        if count == 1:
            return [output]
        rows = []
        for index in range(count):
            row = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor) and int(value.shape[0]) == count * batch_size:
                    row[key] = value[index * batch_size:(index + 1) * batch_size]
                elif isinstance(value, dict):
                    row[key] = IMUHOIModel._split_chunk_output(value, count, batch_size)[index]
                else:
                    row[key] = value
            rows.append(row)
        return rows

    def _published_hp_prefix(self, published_raw, staged_stack, start, chunk_start, index, latest):
        """Raw stage-1 HP history over ``[start, target)`` for one chunk window.

        Frames before the chunk come from the accumulated history, where the
        stage-1 aliases hold the values HP itself published; frames inside the
        chunk come from the frames this chunk has already finalized.
        """
        parts = []
        if start < chunk_start:
            history_slice = select_time_context(published_raw, start, chunk_start)
            parts.append({key: history_slice[key] for key in latest if key in history_slice})
        if index > 0:
            parts.append(
                {key: value[:, :index] for key, value in staged_stack.items() if key in latest}
            )
        if not parts:
            return {}
        return concat_time_dicts(parts)

    def _prepare_online_chunk(
        self,
        data_dict,
        history,
        targets,
        window,
        lookahead,
        batch_size,
        seq_len,
        feedback_mode,
        feedback_perturbation,
        feedback_injection_frame,
    ):
        """Run HumanPose and VelocityContact once for a whole chunk of targets.

        Every window in the chunk reads its initializers from frames at or
        before ``targets[0] - 1``, so all of them are known before the chunk
        starts and the two window forwards can run as one batch.  Only the
        published-frame finalization and the ObjectTrans scan stay sequential,
        because they consume the frame published just before them.
        """
        chunk_start = int(targets[0])
        plans = []
        for target in targets:
            start = target - window + 1
            end = target + lookahead + 1
            window_data = slice_time_dict(data_dict, start, end, batch_size, seq_len)
            window_data = update_data_inits_from_history(
                window_data,
                self._stage1_history_for_inits(history),
                index=start - 1,
                contact_index=start,
                trans_index=start,
                obj_vel_diff_frames=int(getattr(self.cfg, "online_obj_vel_init_diff_frames", 0) or 0),
                hp_init_smoothing=getattr(self.human_pose_module, "smooth_history_init", None),
                fps=float(getattr(self.cfg, "frame_rate", FRAME_RATE)),
            )
            plans.append(
                {
                    "target": target,
                    "start": start,
                    "window_data": window_data,
                    "window_len": end - start,
                    "target_local": target - start,
                    "feedback_control": self._build_feedback_control(
                        data_dict,
                        start,
                        end,
                        mode=feedback_mode,
                        perturbation=feedback_perturbation,
                        injection_frame=feedback_injection_frame,
                        target_frame=target,
                    ),
                }
            )

        count = len(plans)
        hp_inputs = [self._build_hp_input_dict(plan["window_data"]) for plan in plans]
        hp_out = self.human_pose_module.forward(self._stack_chunk_inputs(hp_inputs))
        for plan, row in zip(plans, self._split_chunk_output(hp_out, count, batch_size)):
            plan["hp_raw"] = row

        # The published frame integrates the root from the frame before it, so
        # this chain cannot be batched; it is one ground correction per frame.
        finalize_history = history
        staged = []
        for plan in plans:
            frame = slice_time_dict(
                plan["hp_raw"],
                plan["target_local"],
                plan["target_local"] + 1,
                batch_size,
                plan["window_len"],
            )
            frame = self._finalize_hp_online_frame(frame, finalize_history)
            plan["hp_frame"] = frame
            staged.append(frame)
            finalize_history = frame

        published_raw = self._stage1_history_for_inits(history)
        staged_stack = concat_time_dicts(staged)
        vc_inputs = []
        for index, plan in enumerate(plans):
            hp_future = slice_time_dict(
                plan["hp_raw"],
                plan["target_local"] + 1,
                plan["window_len"],
                batch_size,
                plan["window_len"],
            )
            latest = concat_time_dicts([plan["hp_frame"], hp_future])
            prefix = self._published_hp_prefix(
                published_raw, staged_stack, plan["start"], chunk_start, index, latest
            )
            hp_context = merge_latest_context(prefix, latest)
            plan["hp_context"] = hp_context
            window_data = plan["window_data"]
            vc_inputs.append(
                {
                    "human_imu": window_data["human_imu"],
                    "obj_imu": window_data["obj_imu"],
                    "hand_vel_glb_init": window_data["hand_vel_glb_init"],
                    "obj_vel_init": window_data["obj_vel_init"],
                    "obj_trans_init": window_data["obj_trans_init"],
                    "contact_init": self._vc_window_contact_init(window_data),
                    "hp_out": hp_context,
                }
            )
        stacked_vc_input = self._stack_chunk_inputs(vc_inputs)
        vc_out = self.velocity_contact_module.forward(
            stacked_vc_input, hp_out=stacked_vc_input["hp_out"]
        )
        for plan, row in zip(plans, self._split_chunk_output(vc_out, count, batch_size)):
            plan["vc_out_raw"] = row
        return plans

    def _inference_online_sequence(
        self,
        data_dict: Dict[str, torch.Tensor],
        use_object_data: bool = True,
        compute_fk: bool = False,
        refine_human: Optional[bool] = None,
        online_window: int = 120,
        online_lookahead: int = 5,
        progress_callback=None,
        feedback_mode: str = "normal",
        feedback_perturbation=None,
        feedback_injection_frame: Optional[int] = None,
        online_chunk: int = 1,
    ) -> Dict[str, torch.Tensor]:
        refine_human = self._resolve_refine_human(refine_human)
        batch_size, seq_len = infer_batch_seq(data_dict)
        window = int(online_window)
        lookahead = int(online_lookahead)
        publish_len = max(seq_len - lookahead, 0)
        if publish_len == 0:
            feedback_control = self._build_feedback_control(
                data_dict,
                0,
                seq_len,
                mode=feedback_mode,
                perturbation=feedback_perturbation,
                injection_frame=feedback_injection_frame,
            )
            template = self.forward(
                data_dict,
                use_object_data=use_object_data,
                compute_fk=compute_fk,
                refine_human=refine_human,
                feedback_control=feedback_control,
            )
            return slice_time_dict(template, 0, 0, batch_size, seq_len)

        # Keep the legacy warmup block and append lookahead frames only as
        # future context.  Consequently lookahead=0 is the original online
        # path, not a separately maintained approximation of it.
        warmup_publish_len = min(window, publish_len)
        warmup_input_len = warmup_publish_len + lookahead
        warmup_data = slice_time_dict(
            data_dict, 0, warmup_input_len, batch_size, seq_len
        )
        warmup_feedback = self._build_feedback_control(
            data_dict,
            0,
            warmup_input_len,
            mode=feedback_mode,
            perturbation=feedback_perturbation,
            injection_frame=feedback_injection_frame,
        )
        warmup_raw = self.forward(
            warmup_data,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
            feedback_control=warmup_feedback,
        )
        history = slice_time_dict(
            warmup_raw,
            0,
            warmup_publish_len,
            batch_size,
            warmup_input_len,
        )
        history_acc = TimeDictAccumulator(history, publish_len)
        history = history_acc.current()
        if callable(progress_callback):
            progress_callback(warmup_publish_len, publish_len)

        # Windows whose initializers all predate the chunk can run their
        # HumanPose and VelocityContact forwards as one batch; the sequential
        # part is then the published-frame finalization plus ObjectTrans.
        chunk = max(1, min(int(online_chunk), max(1, window - 1)))
        deferred_refine = False
        for chunk_start in range(warmup_publish_len, publish_len, chunk):
            plans = self._prepare_online_chunk(
                data_dict,
                history,
                list(range(chunk_start, min(chunk_start + chunk, publish_len))),
                window,
                lookahead,
                batch_size,
                seq_len,
                feedback_mode,
                feedback_perturbation,
                feedback_injection_frame,
            )
            for plan in plans:
                target = plan["target"]
                start = plan["start"]
                window_data = plan["window_data"]
                window_len = plan["window_len"]
                target_local = plan["target_local"]
                prefix = select_time_context(history, start, target)
                feedback_control = plan["feedback_control"]
                hp_context = plan["hp_context"]
                vc_out_raw = self._apply_vc_segment_contact_if_enabled(
                    plan["vc_out_raw"],
                    window_data.get("contact_init"),
                )
                vc_out_raw = self._apply_vc_oracle(vc_out_raw, window_data)
                vc_current_future = slice_time_dict(
                    vc_out_raw, target_local, window_len, batch_size, window_len
                )
                vc_context = self._merge_stage_context(prefix, vc_current_future)

                results = {}
                results.update(hp_context)
                results.update(vc_context)

                has_object = window_data.get("has_object")
                if use_object_data and (has_object is None or has_object.any()):
                    lb_prior_quantiles = self._predict_lb_prior(
                        hp_context, vc_context, window_data["obj_imu"]
                    )
                    if lb_prior_quantiles is None and self._online_lb_stats_enabled():
                        # Cross-segment statistics use the full accumulated
                        # history, not just the current window prefix.
                        lb_prior_quantiles = self._compute_online_lb_stats(history)
                    obj_prefix = prefix.get("pred_obj_trans")
                    gate_prefix = prefix.get("gating_weights")
                    if isinstance(obj_prefix, torch.Tensor) and obj_prefix.shape[1] > 0:
                        ot_obj_trans_init = obj_prefix[:, 0]
                    else:
                        ot_obj_trans_init = window_data["obj_trans_init"]
                    ot_extra_kwargs = {}
                    if getattr(self.object_trans_module, "SUPPORTS_FAST_KNOWN_PREFIX", False):
                        # Only frame ``target_local`` is published from this window
                        # and every OT branch is causal, so the teacher-forced
                        # prefix can run as one batched LSTM call and the lookahead
                        # tail does not need to be scanned at all.
                        ot_extra_kwargs["fast_known_prefix"] = True
                        ot_extra_kwargs["scan_end"] = target_local + 1
                    segment_state_keys = getattr(
                        self.object_trans_module, "SEGMENT_STATE_OUTPUT_KEYS", None
                    )
                    if segment_state_keys and isinstance(prefix, dict):
                        # Segment-latching OTs publish their latch state; hand it
                        # back so the window continues the segment instead of
                        # re-deriving it from the published trajectory.
                        ot_extra_kwargs["known_segment_state_prefix"] = {
                            key: prefix.get(key) for key in segment_state_keys
                        }
                    ot_out = self.object_trans_module.forward(
                        self._resolve_ot_hand_positions(hp_context),
                        vc_context["pred_hand_contact_prob"],
                        ot_obj_trans_init,
                        obj_imu=window_data["obj_imu"],
                        human_imu=window_data["human_imu"],
                        obj_vel_input=vc_context["pred_obj_vel"],
                        contact_init=window_data.get("contact_init"),
                        has_object_mask=has_object,
                        human_pose_input=hp_context.get("p_pred"),
                        root_trans_input=hp_context.get("root_trans_pred"),
                        lb_prior_quantiles=lb_prior_quantiles,
                        enable_refine=refine_human,
                        known_obj_trans_prefix=obj_prefix,
                        known_gating_weights_prefix=gate_prefix,
                        feedback_control=feedback_control,
                        **ot_extra_kwargs,
                    )
                    results.update(ot_out)
                    if refine_human:
                        # Later windows read only the stage1_* values, so the
                        # refined skeleton is output-only: one FK call per
                        # published frame (host/sync bound, whatever its
                        # length) becomes one call after the loop.  The FK
                        # object baseline still needs refined palms per window.
                        self._promote_refined_human_outputs(
                            results, ot_out, window_data, derive=compute_fk
                        )
                        deferred_refine = deferred_refine or not compute_fk
                    if compute_fk:
                        self._compute_fk_output(results, vc_context, window_data, ot_obj_trans_init)

                results["has_object"] = has_object
                frame = slice_time_dict(
                    results, target_local, target_local + 1, batch_size, window_len
                )
                history = history_acc.append(frame)
                if callable(progress_callback):
                    progress_callback(target + 1, publish_len)

        output = history_acc.current()
        if deferred_refine:
            # The warmup block was derived inside its own forward; redoing it
            # here is per frame and gives the same values.
            self._derive_refined_human_outputs(
                output, data_dict, chunk_rows=self.DERIVE_CHUNK_ROWS
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
        online_lookahead: Optional[int] = None,
        online_progress_callback=None,
        online_chunk: Optional[int] = None,
        online_state: Optional[dict] = None,
        return_online_state: bool = False,
        feedback_mode: str = "normal",
        feedback_perturbation=None,
        feedback_injection_frame: Optional[int] = None,
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
            seq_len = int(data_dict["human_imu"].shape[1])
            feedback_control = self._build_feedback_control(
                data_dict,
                0,
                seq_len,
                mode=feedback_mode,
                perturbation=feedback_perturbation,
                injection_frame=feedback_injection_frame,
            )
            output = self.forward(
                data_dict,
                use_object_data=use_object_data,
                compute_fk=compute_fk,
                refine_human=refine_human,
                feedback_control=feedback_control,
            )
            if return_online_state:
                return output, online_state or {}
            return output

        window = resolve_online_window(self.cfg, online_window)
        lookahead = resolve_online_lookahead(self.cfg, online_lookahead)
        # Windows whose initializers predate the chunk share one HP/VC forward.
        # 1 keeps the original one-window-per-frame schedule.
        chunk = online_chunk if online_chunk is not None else getattr(self.cfg, "online_chunk", 1)
        previous_output_len = 0
        if isinstance(online_state, dict):
            previous_output_len = int(online_state.get("emitted_len", 0))
        run_data, _ = append_stream_data(
            online_state.get("data_dict") if isinstance(online_state, dict) else None,
            data_dict,
        )
        output = self._inference_online_sequence(
            run_data,
            use_object_data=use_object_data,
            compute_fk=compute_fk,
            refine_human=refine_human,
            online_window=window,
            online_lookahead=lookahead,
            progress_callback=online_progress_callback,
            feedback_mode=feedback_mode,
            feedback_perturbation=feedback_perturbation,
            feedback_injection_frame=feedback_injection_frame,
            online_chunk=int(chunk or 1),
        )
        _, output_len = infer_batch_seq(output, key="p_pred")
        state = {"data_dict": run_data, "outputs": output, "emitted_len": output_len}
        if return_online_state:
            if previous_output_len > 0:
                batch_size, _ = infer_batch_seq(run_data)
                output = slice_time_dict(
                    output,
                    previous_output_len,
                    output_len,
                    batch_size,
                    output_len,
                )
            return output, state
        return output


def load_model(
    config,
    device: torch.device,
    no_trans: bool = False,
    module_paths: Optional[Dict[str, str]] = None,
    object_trans_arch: Optional[str] = None,
    velocity_contact_arch: Optional[str] = None,
) -> IMUHOIModel:
    """
    加载IMUHOI模型
    
    Args:
        config: 配置对象
        device: 计算设备
        no_trans: 是否使用noTrans模式
        module_paths: 可选的模块权重路径覆盖
        velocity_contact_arch: VC架构；current包含boundary分支，legacy不包含
    
    Returns:
        加载好的IMUHOIModel实例
    """
    from utils.utils import flatten_lstm_parameters
    
    model = IMUHOIModel(
        config,
        device,
        no_trans=no_trans,
        object_trans_arch=object_trans_arch,
        velocity_contact_arch=velocity_contact_arch,
    )
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
