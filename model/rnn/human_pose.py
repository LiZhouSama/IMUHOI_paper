"""
HumanPoseModule: 预测人体姿态和根节点位移
Stage 1 - 可独立训练
支持noTrans模式：不预测根节点位移，使用GT位移
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from .base import RNN, RNNWithInit, SubPoser
from configs import (
    FRAME_RATE,
    _SENSOR_NAMES,
    _SENSOR_VEL_NAMES,
    _REDUCED_POSE_NAMES,
    _REDUCED_INDICES,
    _IGNORED_INDICES,
    _SENSOR_ROT_INDICES,
    _VEL_SELECTION_INDICES,
)


class HumanPoseModule(nn.Module):
    """
    人体姿态预测模块
    
    Args:
        cfg: 配置对象
        device: 设备
        no_trans: 是否禁用根节点位移预测（使用GT位移）
    """
    def __init__(self, cfg, device, no_trans=False):
        super().__init__()
        self.device = device
        self.no_trans = no_trans
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        n_hidden = getattr(cfg, "human_pose_hidden", 200)
        num_layer = getattr(cfg, "human_pose_layers", 2)
        dropout = getattr(cfg, "human_pose_dropout", 0.2)
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.sensor_names = list(_SENSOR_NAMES)
        self.v_names = list(_SENSOR_VEL_NAMES)
        self.p_names = list(_REDUCED_POSE_NAMES)
        n_glb = 6

        # 姿态预测器配置
        self.posers_config = [
            {
                "sensor": ["Root", "LeftLowerLeg", "RightLowerLeg", "Head"],
                "velocity": ["Root", "LeftFoot", "RightFoot", "Head"],
                "pose": ["LeftHip", "RightHip"],
            },
            {
                "sensor": ["Root", "Head"],
                "velocity": ["Root", "Head"],
                "pose": ["Spine1", "Spine2", "Spine3", "Neck"],
            },
            {
                "sensor": ["Root", "LeftForeArm", "RightForeArm"],
                "velocity": ["LeftHand", "RightHand"],
                "pose": ["LeftCollar", "RightCollar", "LeftShoulder", "RightShoulder"],
            },
        ]

        # 构建子姿态预测器
        self.posers = nn.ModuleList()
        for config in self.posers_config:
            n_sensor = len(config["sensor"])
            n_input = n_sensor * self.imu_dim + n_glb
            v_output = len(config["velocity"]) * 3
            p_output = len(config["pose"]) * 6
            self.posers.append(
                SubPoser(
                    n_input=n_input,
                    v_output=v_output,
                    p_output=p_output,
                    n_hidden=n_hidden,
                    num_layer=num_layer,
                    dropout=dropout,
                    extra_dim=n_glb,
                )
            )

        # 全局特征网络
        human_feature_dim = self.num_human_imus * self.imu_dim
        self.glb = RNN(
            n_input=human_feature_dim,
            n_output=n_glb,
            n_hidden=36,
            n_rnn_layer=1,
            dropout=dropout,
        )

        # 根节点位移预测网络（仅在非noTrans模式下使用）
        if not no_trans:
            n_feet_vel = 6
            n_feet_imu = 2 * self.imu_dim
            n_torso_vel = 18
            n_torso_imu = 2 * self.imu_dim

            self.tran_b1 = RNN(
                n_input=n_feet_vel + n_feet_imu + n_glb,
                n_output=2,
                n_hidden=64,
                n_rnn_layer=2,
                bidirectional=True,
                dropout=dropout,
            )
            self.tran_b2 = RNN(
                n_input=n_torso_vel + n_torso_imu + n_glb,
                n_output=3,
                n_hidden=128,
                n_rnn_layer=2,
                bidirectional=False,
                dropout=dropout,
            )
            self.torso_joints = torch.tensor([0, 3, 6, 9, 12, 15], dtype=torch.long)
            self.prob_threshold = (0.5, 0.9)
            self.gravity_velocity = torch.tensor([0.0, -0.018, 0.0], dtype=torch.float32)
            self.floor_height = 0.0
            self.prevent_floor_penetration = True
            self._foot_joint_indices = (7, 8)

        # SMPL相关
        self.smpl_parents = torch.tensor(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
            dtype=torch.long,
        )
        self.body_model = None
        self.body_model_device = None
        body_model_path = getattr(cfg, "body_model_path", None)
        if body_model_path is None:
            raise ValueError("body_model_path is not set")
        try:
            self.body_model = BodyModel(bm_fname=body_model_path, num_betas=16)
            self.body_model.eval()
            for param in self.body_model.parameters():
                param.requires_grad_(False)
            self.body_model_device = torch.device("cpu")
        except Exception as exc:
            print(f"加载Body Model失败: {exc}")
            exit()

        if isinstance(_VEL_SELECTION_INDICES, torch.Tensor):
            self.vel_indices = _VEL_SELECTION_INDICES.tolist()
        else:
            self.vel_indices = list(_VEL_SELECTION_INDICES)

        self._generate_indices_list()
        self.hand_joint_indices = (20, 21)

    def _find_indices(self, names, pool):
        return [pool.index(name) for name in names]

    def _generate_indices_list(self):
        self.indices = []
        for config in self.posers_config:
            self.indices.append({
                "sensor_indices": self._find_indices(config["sensor"], self.sensor_names),
                "v_indices": self._find_indices(config["velocity"], self.v_names),
                "p_indices": self._find_indices(config["pose"], self.p_names),
            })

    def _prob_to_weight(self, p):
        p_clamped = p.clamp(self.prob_threshold[0], self.prob_threshold[1])
        return (p_clamped - self.prob_threshold[0]) / (self.prob_threshold[1] - self.prob_threshold[0] + 1e-8)

    def _global2local(self, global_rotmats, parents):
        batch_size, num_joints, _, _ = global_rotmats.shape
        local_rotmats = torch.zeros_like(global_rotmats)
        local_rotmats[:, 0] = global_rotmats[:, 0]
        for i in range(1, num_joints):
            parent_idx = parents[i]
            R_parent = global_rotmats[:, parent_idx]
            R_parent_inv = R_parent.transpose(-1, -2)
            local_rotmats[:, i] = torch.matmul(R_parent_inv, global_rotmats[:, i])
        local_rotmats[:, _IGNORED_INDICES] = torch.eye(
            3, device=global_rotmats.device, dtype=global_rotmats.dtype
        ).view(1, 1, 3, 3).repeat(batch_size, len(_IGNORED_INDICES), 1, 1)
        return local_rotmats

    def _reduced_glb_6d_to_full_glb_mat(self, glb_reduced_pose, orientation):
        root_rotation = orientation[:, 0]
        reduced_rot = rotation_6d_to_matrix(glb_reduced_pose.reshape(-1, 6)).reshape(
            glb_reduced_pose.shape[0], len(_REDUCED_POSE_NAMES), 3, 3
        )
        reduced_rot_global = torch.matmul(root_rotation.unsqueeze(1), reduced_rot)
        orientation_global = orientation.clone()
        orientation_global[:, 1:] = torch.matmul(root_rotation.unsqueeze(1), orientation[:, 1:])
        dtype = glb_reduced_pose.dtype
        device = glb_reduced_pose.device
        full_pose = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).repeat(
            glb_reduced_pose.shape[0], 24, 1, 1
        )
        full_pose[:, _REDUCED_INDICES] = reduced_rot_global
        full_pose[:, _SENSOR_ROT_INDICES] = orientation_global
        ignored_parents = self.smpl_parents[_IGNORED_INDICES]
        full_pose[:, _IGNORED_INDICES] = full_pose[:, ignored_parents]
        return full_pose

    def _compute_fk_joints_batched(self, glb_p_out_tensor: torch.Tensor, orientation: torch.Tensor):
        if self.body_model is None:
            return None

        batch_size, seq_len, _ = glb_p_out_tensor.shape
        device = glb_p_out_tensor.device
        BT = batch_size * seq_len

        glb_pose = glb_p_out_tensor.reshape(BT, len(_REDUCED_POSE_NAMES), 6)
        orientation = orientation[:, :, : len(_SENSOR_ROT_INDICES)].reshape(BT, len(_SENSOR_ROT_INDICES), 3, 3)
        full_glb = self._reduced_glb_6d_to_full_glb_mat(glb_pose, orientation)
        local_pose = self._global2local(full_glb, self.smpl_parents.tolist())
        pose_aa = matrix_to_axis_angle(local_pose.reshape(-1, 3, 3)).reshape(BT, 24, 3)

        try:
            with torch.no_grad():
                body_out = self.body_model(
                    pose_body=pose_aa[:, 1:22].reshape(BT, 63),
                    root_orient=pose_aa[:, 0].reshape(BT, 3),
                )
            joints = body_out.Jtr[:, :24, :]
            return joints.reshape(batch_size, seq_len, 24, 3)
        except Exception as exc:
            print(f"FK计算失败: {exc}")
            return None

    def _compute_torso_velocity_batched(self, joints_pos: torch.Tensor):
        if joints_pos is None:
            return None
        batch_size, seq_len, _, _ = joints_pos.shape
        device = joints_pos.device

        if isinstance(self.torso_joints, torch.Tensor):
            torso_idx = self.torso_joints.to(device=device, dtype=torch.long)
        else:
            torso_idx = torch.tensor(self.torso_joints, device=device, dtype=torch.long)
        torso_pos = torch.index_select(joints_pos, dim=2, index=torso_idx).contiguous()

        torso_vel = torch.zeros_like(torso_pos)
        if seq_len > 1:
            torso_vel[:, 1:] = (torso_pos[:, 1:] - torso_pos[:, :-1]) * self.fps
            torso_vel[:, 0] = torso_vel[:, 1]
        return torso_vel.view(batch_size, seq_len, -1)

    def _compute_root_velocity_from_trans(self, trans: torch.Tensor):
        if trans is None:
            return None
        if trans.dim() == 2:
            trans = trans.unsqueeze(1)
        vel = torch.zeros_like(trans)
        if trans.size(1) > 1:
            vel[:, 1:] = (trans[:, 1:] - trans[:, :-1]) * self.fps
            vel[:, 0] = vel[:, 1]
        return vel

    def _apply_floor_penetration(self, root_velocities: torch.Tensor, joints_pos: torch.Tensor):
        if (not getattr(self, "prevent_floor_penetration", False)) or joints_pos is None:
            return root_velocities
        batch_size, seq_len, _ = root_velocities.shape
        device = root_velocities.device
        axis = 1
        foot_start, foot_end = self._foot_joint_indices

        if joints_pos.size(2) <= foot_end:
            return root_velocities

        floor = torch.as_tensor(self.floor_height, device=device, dtype=root_velocities.dtype)
        foot_y = joints_pos[:, :, foot_start : foot_end + 1, axis]
        foot_min_y = foot_y.amin(dim=2)
        vel_axis = root_velocities[..., axis]
        vel_cumsum = torch.cumsum(vel_axis, dim=1)
        adjustment = F.relu(floor - foot_min_y - vel_cumsum)
        adjustment_max = torch.cummax(adjustment, dim=1).values
        adjustment_prev = F.pad(adjustment_max[:, :-1], (1, 0))
        delta = adjustment_max - adjustment_prev
        corrected = root_velocities.clone()
        corrected[..., axis] = vel_axis + delta
        return corrected

    def _integrate_root_velocity(self, root_velocities: torch.Tensor):
        return torch.cumsum(root_velocities, dim=1) / self.fps

    def _fuse_velocities_batched(
        self,
        root_vel_local: torch.Tensor,
        contact_logits: torch.Tensor,
        joints_pos: torch.Tensor,
        root_rotation: torch.Tensor,
        vel_scale: float = 3.0,
    ):
        device = root_vel_local.device
        batch_size, seq_len, _ = root_vel_local.shape
        tran_b2_vel = torch.matmul(root_rotation, root_vel_local.unsqueeze(-1)).squeeze(-1)
        tran_b2_vel = tran_b2_vel * vel_scale / self.fps

        lfoot_pos = joints_pos[:, :, 7, :]
        rfoot_pos = joints_pos[:, :, 8, :]
        lfoot_vel = torch.zeros_like(lfoot_pos)
        rfoot_vel = torch.zeros_like(rfoot_pos)
        if seq_len > 1:
            lfoot_vel[:, 1:] = lfoot_pos[:, :-1] - lfoot_pos[:, 1:]
            rfoot_vel[:, 1:] = rfoot_pos[:, :-1] - rfoot_pos[:, 1:]

        contact_idx = contact_logits.argmax(dim=-1)
        left_mask = (contact_idx == 0).unsqueeze(-1)
        tran_b1_vel = torch.where(left_mask, lfoot_vel, rfoot_vel)
        gravity = self.gravity_velocity.to(device=device, dtype=root_vel_local.dtype).view(1, 1, 3)
        tran_b1_vel = (tran_b1_vel + gravity) / self.fps

        contact_sigmoid = torch.sigmoid(contact_logits)
        contact_max = contact_sigmoid.max(dim=-1).values
        weight = self._prob_to_weight(contact_max).unsqueeze(-1)

        fused = (1.0 - weight) * tran_b2_vel + weight * tran_b1_vel
        fused = fused * self.fps
        return fused

    def forward(self, data_dict: dict):
        """
        前向传播
        
        Args:
            data_dict: 包含以下键的字典:
                - human_imu: [B, T, num_imu, imu_dim] 人体IMU数据
                - v_init: [B, num_vel, 3] 初始速度
                - p_init: [B, num_pose, 6] 初始姿态
                - trans_init: [B, 3] 初始根节点位置 (仅非noTrans模式)
                - trans_gt: [B, T, 3] GT根节点位置 (仅noTrans模式)
        
        Returns:
            dict: 包含预测的姿态、速度和手部位置
        """
        human_imu = data_dict['human_imu']
        v_init = data_dict['v_init']
        p_init = data_dict['p_init']
        trans_init = data_dict.get('trans_init')
        trans_gt = data_dict.get('trans_gt')
        if human_imu.dim() != 4:
            raise ValueError(f"human_imu must be [B, T, num_imu, imu_dim], got {human_imu.shape}")
        batch_size, seq_len, _, _ = human_imu.shape
        device = human_imu.device

        if self.body_model is not None and (self.body_model_device != device):
            self.body_model = self.body_model.to(device)
            self.body_model_device = device

        human_flat = human_imu.reshape(batch_size, seq_len, -1)
        s_glb = self.glb(human_flat)

        v_components = []
        p_components = []
        v_lower = None

        for poser_idx, poser in enumerate(self.posers):
            indices = self.indices[poser_idx]
            sensor_feat = human_imu[:, :, indices["sensor_indices"], :].reshape(batch_size, seq_len, -1)
            poser_input = torch.cat((sensor_feat, s_glb), dim=-1)
            v_init_sub = v_init[:, indices["v_indices"], :].reshape(batch_size, -1)
            p_init_sub = p_init[:, indices["p_indices"], :].reshape(batch_size, -1)

            v_i, p_i = poser(poser_input, v_init_sub, p_init_sub)
            v_components.append(v_i)
            p_components.append(p_i)

            if poser_idx == 0:
                v_lower = v_i

        v_pred = torch.cat(v_components, dim=-1)
        p_pred = torch.cat(p_components, dim=-1)

        orientation_6d = human_imu[..., -6:]
        orientation_mat = rotation_6d_to_matrix(orientation_6d.reshape(-1, 6)).reshape(
            batch_size, seq_len, self.num_human_imus, 3, 3
        )
        root_R = orientation_mat[:, :, 0]

        joints_pos = None
        full_glb_rotmats = None
        full_glb_rot6d = None
        if self.body_model is not None:
            joints_pos = self._compute_fk_joints_batched(p_pred, orientation_mat.clone())
            try:
                BT = batch_size * seq_len
                glb_pose_flat = p_pred.reshape(BT, len(_REDUCED_POSE_NAMES), 6)
                orientation_flat = orientation_mat[:, :, : len(_SENSOR_ROT_INDICES)].reshape(
                    BT, len(_SENSOR_ROT_INDICES), 3, 3
                )
                full_glb_rotmats_flat = self._reduced_glb_6d_to_full_glb_mat(glb_pose_flat, orientation_flat)
                full_glb_rotmats = full_glb_rotmats_flat.reshape(batch_size, seq_len, 24, 3, 3)
                full_glb_rot6d = matrix_to_rotation_6d(full_glb_rotmats_flat.reshape(-1, 3, 3)).reshape(
                    batch_size, seq_len, 24, 6
                )
            except Exception as exc:
                print(f"Failed to compute full pose rotations: {exc}")
                full_glb_rotmats = None
                full_glb_rot6d = None

        results = {
            "v_pred": v_pred,
            "p_pred": p_pred,
        }

        if self.no_trans:
            # noTrans模式：使用GT位移
            if trans_gt is None:
                trans_gt = torch.zeros(batch_size, seq_len, 3, device=device, dtype=human_imu.dtype)
            else:
                trans_gt = trans_gt.to(device=device, dtype=human_imu.dtype)
                if trans_gt.dim() == 2:
                    trans_gt = trans_gt.unsqueeze(1).expand(batch_size, seq_len, 3)

            if joints_pos is not None:
                lhand = joints_pos[:, :, self.hand_joint_indices[0], :] + trans_gt
                rhand = joints_pos[:, :, self.hand_joint_indices[1], :] + trans_gt
                pred_hand_glb_pos = torch.stack((lhand, rhand), dim=2)
            else:
                pred_hand_glb_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=human_imu.dtype)

            root_vel_pred = self._compute_root_velocity_from_trans(trans_gt)
            if root_vel_pred is None:
                root_vel_pred = torch.zeros(batch_size, seq_len, 3, device=device, dtype=human_imu.dtype)

            results.update({
                "pred_hand_glb_pos": pred_hand_glb_pos,
                "root_vel_pred": root_vel_pred,
                "root_trans_pred": trans_gt,
            })
        else:
            # 普通模式：预测根节点位移
            if joints_pos is not None:
                torso_vel = self._compute_torso_velocity_batched(joints_pos)
            else:
                torso_vel = torch.zeros(batch_size, seq_len, 18, device=device, dtype=human_imu.dtype)

            feet_vel = v_lower[:, :, 3:9]
            feet_imu = human_imu[:, :, [1, 2], :].reshape(batch_size, seq_len, -1)
            torso_imu = human_imu[:, :, [0, 3], :].reshape(batch_size, seq_len, -1)

            contact_pred = self.tran_b1(torch.cat((feet_vel, feet_imu, s_glb), dim=-1))
            root_vel_local_pred = self.tran_b2(torch.cat((torso_vel, torso_imu, s_glb), dim=-1))

            if joints_pos is not None:
                root_vel_pred = self._fuse_velocities_batched(root_vel_local_pred, contact_pred, joints_pos, root_R)
            else:
                root_vel_pred = torch.matmul(root_R, root_vel_local_pred.unsqueeze(-1)).squeeze(-1) / self.fps

            root_vel_pred = self._apply_floor_penetration(root_vel_pred, joints_pos)
            root_trans_delta = self._integrate_root_velocity(root_vel_pred)

            if trans_init is None:
                trans_init = torch.zeros(batch_size, 3, device=device, dtype=human_imu.dtype)
            root_trans_pred = root_trans_delta + trans_init.unsqueeze(1)

            if joints_pos is not None:
                lhand = joints_pos[:, :, self.hand_joint_indices[0], :] + root_trans_pred
                rhand = joints_pos[:, :, self.hand_joint_indices[1], :] + root_trans_pred
                pred_hand_glb_pos = torch.stack((lhand, rhand), dim=2)
            else:
                pred_hand_glb_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=human_imu.dtype)

            results.update({
                "contact_pred": contact_pred,
                "root_vel_local_pred": root_vel_local_pred,
                "root_vel_pred": root_vel_pred,
                "root_trans_pred": root_trans_pred,
                "pred_hand_glb_pos": pred_hand_glb_pos,
            })

        if joints_pos is not None:
            results["pred_joints_local"] = joints_pos
            if "root_trans_pred" in results:
                results["pred_joints_global"] = joints_pos + results["root_trans_pred"].unsqueeze(2)
        else:
            results["pred_joints_local"] = torch.zeros(batch_size, seq_len, 24, 3, device=device, dtype=human_imu.dtype)
            if "root_trans_pred" in results:
                results["pred_joints_global"] = torch.zeros(batch_size, seq_len, 24, 3, device=device, dtype=human_imu.dtype)

        if full_glb_rotmats is not None:
            results["pred_full_pose_rotmat"] = full_glb_rotmats
            results["pred_full_pose_6d"] = full_glb_rot6d
        else:
            results["pred_full_pose_rotmat"] = torch.zeros(batch_size, seq_len, 24, 3, 3, device=device, dtype=human_imu.dtype)
            results["pred_full_pose_6d"] = torch.zeros(batch_size, seq_len, 24, 6, device=device, dtype=human_imu.dtype)

        return results

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device, no_trans: bool = False):
        """返回空输出"""
        results = {
            "v_pred": torch.zeros(batch_size, seq_len, len(_SENSOR_VEL_NAMES) * 3, device=device),
            "p_pred": torch.zeros(batch_size, seq_len, len(_REDUCED_POSE_NAMES) * 6, device=device),
            "pred_hand_glb_pos": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "root_vel_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "root_trans_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_joints_local": torch.zeros(batch_size, seq_len, 24, 3, device=device),
            "pred_joints_global": torch.zeros(batch_size, seq_len, 24, 3, device=device),
            "pred_full_pose_rotmat": torch.zeros(batch_size, seq_len, 24, 3, 3, device=device),
            "pred_full_pose_6d": torch.zeros(batch_size, seq_len, 24, 6, device=device),
        }
        if not no_trans:
            results.update({
                "contact_pred": torch.zeros(batch_size, seq_len, 2, device=device),
                "root_vel_local_pred": torch.zeros(batch_size, seq_len, 3, device=device),
                "root_vel_pred": torch.zeros(batch_size, seq_len, 3, device=device),
                "root_trans_pred": torch.zeros(batch_size, seq_len, 3, device=device),
            })
        return results
