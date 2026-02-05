"""
VelocityContactModule: predict hand/object velocity and contact cues.
"""
import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

from .base import RNNWithInit
from configs import FRAME_RATE, _SENSOR_NAMES
from utils.utils import _central_diff, _smooth_acceleration


class VelocityContactModule(nn.Module):
    """
    Speed and contact prediction.
    Input: human_imu, obj_imu, hp_out (optional human-pose outputs).
    Output: hand/object velocity, contact probabilities, interaction boundary.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_human_imus = getattr(cfg, "num_human_imus", len(_SENSOR_NAMES))
        self.imu_dim = getattr(cfg, "imu_dim", 9)
        self.obj_imu_dim = getattr(cfg, "obj_imu_dim", self.imu_dim)
        if self.obj_imu_dim != self.imu_dim:
            raise ValueError("Shared velocity predictor requires obj_imu_dim to match imu_dim.")
        hidden_dim = getattr(cfg, "velocity_hidden_dim", 128)
        num_layers = getattr(cfg, "velocity_num_layers", 2)
        dropout = getattr(cfg, "velocity_dropout", 0.2)
        self.fps = float(getattr(cfg, "frame_rate", FRAME_RATE))

        # Body-part grouped encoders for interaction boundary prediction
        self.body_group_indices = {
            "left_arm": [13, 16, 18, 20, 22],
            "right_arm": [14, 17, 19, 21, 23],
            "torso": [0, 3, 6, 9, 12, 15],
            "left_leg": [1, 4, 7, 10],
            "right_leg": [2, 5, 8, 11],
        }
        group_hidden = getattr(cfg, "boundary_group_hidden", max(hidden_dim // 2, 32))
        boundary_hidden = getattr(cfg, "boundary_hidden_dim", hidden_dim)
        self.boundary_hidden_dim = boundary_hidden
        self.group_encoders = nn.ModuleDict()
        self.group_norms = nn.ModuleDict()
        for name, idx in self.body_group_indices.items():
            group_input_dim = len(idx) * (6 + 3 + 3) + 3  # pose6d + vel + acc + root vel
            self.group_norms[name] = nn.LayerNorm(group_input_dim)
            self.group_encoders[name] = nn.GRU(
                input_size=group_input_dim,
                hidden_size=group_hidden,
                num_layers=1,
                batch_first=True,
            )
        boundary_input_dim = group_hidden * len(self.body_group_indices)
        self.boundary_rnn = nn.GRU(
            input_size=boundary_input_dim,
            hidden_size=boundary_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.boundary_head = nn.Linear(boundary_hidden, 2)
        self.boundary_input_norm = nn.LayerNorm(boundary_input_dim)
        self.boundary_hidden_norm = nn.LayerNorm(boundary_hidden)

        # Velocity predictor (shared for hands/obj)
        self.vel_net = RNNWithInit(
            n_input=self.imu_dim,
            n_output=3,
            n_hidden=hidden_dim,
            n_init=3,
            n_rnn_layer=num_layers,
            bidirectional=False,
            dropout=dropout,
        )

        # Object move probability
        contact_input_dim = 2 * self.imu_dim + self.obj_imu_dim
        contact_hidden = max(hidden_dim // 2, 32)
        self.obj_move_net = RNNWithInit(
            n_input=contact_input_dim,
            n_output=1,  # object moving or not
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )
        # Hand contact conditioned on object moving
        hand_contact_input_dim = 2 * (3 + 3) + 3 + self.obj_imu_dim + boundary_hidden  # vel+acc per hand + root vel + obj imu + boundary features
        self.hand_contact_net = RNNWithInit(
            n_input=hand_contact_input_dim,
            n_output=2,  # left/right conditional prob
            n_hidden=contact_hidden,
            n_init=3,
            n_rnn_layer=1,
            bidirectional=False,
            dropout=dropout,
        )
        self.hand_contact_input_norm = nn.LayerNorm(hand_contact_input_dim)

        self.left_hand_sensor = _SENSOR_NAMES.index("LeftForeArm")
        self.right_hand_sensor = _SENSOR_NAMES.index("RightForeArm")

    def _denormalize_imu(self, human_imu: torch.Tensor):
        """Convert IMU data back to world frame."""
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

    def _prepare_pose_streams(self, hp_out, device, dtype, batch_size, seq_len):
        pose_6d = None
        joint_pos = None
        root_vel = None
        hand_pos = None
        if isinstance(hp_out, dict):
            pose_6d = hp_out.get("pred_full_pose_6d")
            joint_pos = hp_out.get("pred_joints_local")
            root_vel = hp_out.get("root_vel_pred")
            hand_pos = hp_out.get("pred_hand_glb_pos")

        if pose_6d is None:
            pose_6d = torch.zeros(batch_size, seq_len, 24, 6, device=device, dtype=dtype)
        if joint_pos is None:
            joint_pos = torch.zeros(batch_size, seq_len, 24, 3, device=device, dtype=dtype)
        if root_vel is None:
            root_vel = torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)
        if hand_pos is None:
            hand_pos = torch.zeros(batch_size, seq_len, 2, 3, device=device, dtype=dtype)

        dt = 1.0 / self.fps
        joint_vel = _central_diff(joint_pos, dt)
        joint_acc = _smooth_acceleration(joint_pos, self.fps, smooth_n=4)
        hand_vel = _central_diff(hand_pos, dt)
        hand_acc = _smooth_acceleration(hand_pos, self.fps, smooth_n=4)

        return {
            "pose_6d": pose_6d,
            "joint_vel": joint_vel,
            "joint_acc": joint_acc,
            "root_vel": root_vel,
            "hand_vel": hand_vel,
            "hand_acc": hand_acc,
        }

    def _encode_body_groups(self, pose_streams):
        boundary_features = []
        pose_6d = pose_streams["pose_6d"]
        joint_vel = pose_streams["joint_vel"]
        joint_acc = pose_streams["joint_acc"]
        root_vel = pose_streams["root_vel"]
        batch_size, seq_len = root_vel.shape[:2]

        for name, encoder in self.group_encoders.items():
            idx = self.body_group_indices[name]
            pose_feat = pose_6d[:, :, idx, :].reshape(batch_size, seq_len, -1)
            vel_feat = joint_vel[:, :, idx, :].reshape(batch_size, seq_len, -1)
            acc_feat = joint_acc[:, :, idx, :].reshape(batch_size, seq_len, -1)
            group_input = torch.cat((pose_feat, vel_feat, acc_feat, root_vel), dim=-1)
            group_input = self.group_norms[name](group_input)
            group_out, _ = encoder(group_input)
            boundary_features.append(group_out)

        boundary_input = torch.cat(boundary_features, dim=-1)
        boundary_input = self.boundary_input_norm(boundary_input)
        boundary_out, _ = self.boundary_rnn(boundary_input)
        boundary_out_norm = self.boundary_hidden_norm(boundary_out)
        boundary_logits = self.boundary_head(boundary_out_norm)
        boundary_prob = torch.sigmoid(boundary_logits)
        return boundary_logits, boundary_prob, boundary_out_norm

    def forward(self, data_dict: dict, hp_out: dict = None):
        """
        Forward pass.
        Args:
            data_dict: contains human_imu, obj_imu, hand_vel_glb_init, obj_vel_init, contact_init.
            hp_out: optional human pose outputs (pose/vel/acc) for boundary/contact prediction.
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

        # Object IMU
        if obj_imu is None:
            obj_imu = torch.zeros(batch_size, seq_len, self.obj_imu_dim, device=device, dtype=dtype)
        else:
            if obj_imu.dim() == 4:
                obj_imu = obj_imu.reshape(batch_size, seq_len, -1)
            elif obj_imu.dim() != 3:
                raise ValueError(f"obj_imu must be [B, T, obj_dim], got {obj_imu.shape}")
        if obj_imu.shape[-1] != self.imu_dim:
            raise ValueError(f"Shared velocity net expects obj_imu feature dim {self.imu_dim}, got {obj_imu.shape[-1]}")

        # Init vectors
        if hand_vel_init.dim() != 3 or hand_vel_init.shape[1:] != (2, 3):
            raise ValueError(f"hand_vel_init must be [B,2,3], got {hand_vel_init.shape}")
        l_hand_vel_init = hand_vel_init[:, 0, :]
        r_hand_vel_init = hand_vel_init[:, 1, :]

        if obj_vel_init.dim() == 1:
            obj_vel_init_vec = obj_vel_init.unsqueeze(0).expand(batch_size, -1)
        elif obj_vel_init.dim() == 2:
            obj_vel_init_vec = obj_vel_init
        else:
            raise ValueError(f"obj_vel_init must be [B,3] or [3], got {obj_vel_init.shape}")

        # Denormalize IMU
        human_imu_denorm = self._denormalize_imu(human_imu)

        # Hand velocities
        l_hand_imu = human_imu_denorm[:, :, self.left_hand_sensor, :]
        r_hand_imu = human_imu_denorm[:, :, self.right_hand_sensor, :]
        l_hand_vel = self.vel_net((l_hand_imu, l_hand_vel_init))
        r_hand_vel = self.vel_net((r_hand_imu, r_hand_vel_init))
        hand_vel = torch.stack((l_hand_vel, r_hand_vel), dim=2)

        # Object velocity
        obj_vel = self.vel_net((obj_imu, obj_vel_init_vec))

        # Contact input (IMU-based)
        hand_imu_feat = human_imu_denorm[:, :, [self.left_hand_sensor, self.right_hand_sensor], :].reshape(
            batch_size, seq_len, -1
        )
        contact_input_imu = torch.cat((hand_imu_feat, obj_imu), dim=-1)

        # Contact init
        if contact_init is None:
            contact_init_vec = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        else:
            if contact_init.dim() == 1:
                contact_init_vec = contact_init.unsqueeze(0).expand(batch_size, -1)
            else:
                contact_init_vec = contact_init
            if contact_init_vec.shape[-1] > 3:
                contact_init_vec = contact_init_vec[..., :3]
            if contact_init_vec.shape[-1] != 3:
                raise ValueError(f"contact_init must have last dim 3, got {contact_init_vec.shape}")

        # Object move prob
        obj_move_logits = self.obj_move_net((contact_input_imu, contact_init_vec))
        obj_move_prob = torch.sigmoid(obj_move_logits)

        # Boundary prediction from HPE outputs
        pose_streams = self._prepare_pose_streams(hp_out or data_dict.get("hp_out"), device, dtype, batch_size, seq_len)
        boundary_logits, boundary_prob, boundary_out = self._encode_body_groups(pose_streams)

        # Hand contact using HPE-derived velocity/acc plus boundary hidden states
        hand_dyn_feat = torch.cat(
            (
                pose_streams["hand_vel"].reshape(batch_size, seq_len, -1),
                pose_streams["hand_acc"].reshape(batch_size, seq_len, -1),
            ),
            dim=-1,
        )
        root_vel_feat = pose_streams["root_vel"]
        hand_contact_input = torch.cat((hand_dyn_feat, root_vel_feat, obj_imu, boundary_out), dim=-1)
        hand_contact_input = self.hand_contact_input_norm(hand_contact_input)

        hand_contact_logits = self.hand_contact_net((hand_contact_input, contact_init_vec))
        hand_contact_prob_cond = torch.sigmoid(hand_contact_logits)

        # Unconditional hand contact = object move prob * conditional
        hand_contact_prob = obj_move_prob * hand_contact_prob_cond

        # Concatenated outputs for compatibility
        contact_logits = torch.cat((hand_contact_logits, obj_move_logits), dim=-1)
        contact_prob = torch.cat((hand_contact_prob, obj_move_prob), dim=-1)

        return {
            "pred_hand_glb_vel": hand_vel,
            "pred_obj_vel": obj_vel,
            "pred_hand_contact_logits": contact_logits,
            "pred_hand_contact_prob": contact_prob,
            "pred_obj_move_prob": obj_move_prob,
            "pred_hand_contact_prob_cond": hand_contact_prob_cond,
            "pred_hand_contact_logits_cond": hand_contact_logits,
            "pred_interaction_boundary_logits": boundary_logits,
            "pred_interaction_boundary_prob": boundary_prob,
        }

    @staticmethod
    def empty_output(batch_size: int, seq_len: int, device: torch.device):
        zeros_pos = torch.zeros(batch_size, seq_len, 3, device=device)
        return {
            "pred_hand_glb_vel": torch.zeros(batch_size, seq_len, 2, 3, device=device),
            "pred_obj_vel": zeros_pos.clone(),
            "pred_hand_contact_logits": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_hand_contact_prob": torch.zeros(batch_size, seq_len, 3, device=device),
            "pred_obj_move_prob": torch.zeros(batch_size, seq_len, 1, device=device),
            "pred_hand_contact_prob_cond": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_hand_contact_logits_cond": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_interaction_boundary_logits": torch.zeros(batch_size, seq_len, 2, device=device),
            "pred_interaction_boundary_prob": torch.zeros(batch_size, seq_len, 2, device=device),
        }
