"""
HumanPoseModule的损失函数
"""
import torch
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

from configs import _SENSOR_VEL_NAMES, _REDUCED_POSE_NAMES


class HumanPoseLoss:
    """人体姿态预测的损失计算"""
    
    # 训练阶段使用的损失键
    LOSS_KEYS = {
        'vel_root',
        'pose_reduced',
        'root_vel_local',
        'root_vel',
        'root_trans',
        'hand_pos',
        'contact',
    }
    
    # 测试阶段用于模型选择的损失键 (可自定义子集)
    TEST_LOSS_KEYS = {'pose_reduced', 'root_trans'}
    
    def __init__(self, weights=None, no_trans=False):
        """
        Args:
            weights: dict, 各损失项的权重，默认为1.0
            no_trans: bool, 是否为noTrans模式
        """
        self.weights = weights or {}
        self.no_trans = no_trans
    
    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)
    
    def compute_loss(self, pred_dict, batch, device):
        """计算损失"""
        human_imu = batch['human_imu'].to(device)
        dtype = human_imu.dtype
        bs, seq = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)
        root_ori_gt = rotation_6d_to_matrix(human_imu[:, :, 0, -6:])
        
        losses = {key: zero.clone() for key in self.LOSS_KEYS}
        
        # 获取GT数据
        trans_gt = batch.get('trans')
        if isinstance(trans_gt, torch.Tensor):
            trans_gt = trans_gt.to(device)
        else:
            trans_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)
        
        root_vel_gt = batch.get('root_vel')
        if isinstance(root_vel_gt, torch.Tensor):
            root_vel_gt = root_vel_gt.to(device)
        else:
            root_vel_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)
        
        root_vel_local_gt = root_ori_gt.transpose(-1, -2).matmul(root_vel_gt.unsqueeze(-1)).squeeze(-1)
        
        sensor_vel_root_gt = batch.get('sensor_vel_root')
        if isinstance(sensor_vel_root_gt, torch.Tensor):
            sensor_vel_root_gt = sensor_vel_root_gt.to(device)
            if sensor_vel_root_gt.dim() == 3:
                sensor_vel_root_gt = sensor_vel_root_gt.unsqueeze(0).expand(bs, -1, -1, -1)
        else:
            sensor_vel_root_gt = torch.zeros(bs, seq, len(_SENSOR_VEL_NAMES), 3, device=device, dtype=dtype)
        
        ori_root_reduced_gt = batch.get('ori_root_reduced')
        if isinstance(ori_root_reduced_gt, torch.Tensor):
            ori_root_reduced_gt = ori_root_reduced_gt.to(device)
        else:
            ori_root_reduced_gt = None
        
        position_global_gt = batch.get('position_global')
        if isinstance(position_global_gt, torch.Tensor):
            position_global_gt = position_global_gt.to(device)
        else:
            position_global_gt = None
        
        # 计算各项损失
        if 'v_pred' in pred_dict:
            v_pred = pred_dict['v_pred'].view(bs, seq, -1, 3)
            vel_indices = [0, 1, 2, 3, 0, 3, 4, 5]
            target_vel = sensor_vel_root_gt[:, :, vel_indices, :]
            losses['vel_root'] = F.mse_loss(v_pred, target_vel)
        
        if 'p_pred' in pred_dict and ori_root_reduced_gt is not None:
            pose_gt_6d = matrix_to_rotation_6d(
                ori_root_reduced_gt.reshape(-1, 3, 3)
            ).reshape(bs, seq, len(_REDUCED_POSE_NAMES), 6)
            p_pred = pred_dict['p_pred'].view(bs, seq, len(_REDUCED_POSE_NAMES), 6)
            losses['pose_reduced'] = F.mse_loss(p_pred, pose_gt_6d)
        
        # 非noTrans模式下的额外损失
        if not self.no_trans:
            if 'root_vel_local_pred' in pred_dict:
                losses['root_vel_local'] = F.mse_loss(pred_dict['root_vel_local_pred'], root_vel_local_gt)
            
            if 'root_vel_pred' in pred_dict:
                losses['root_vel'] = F.mse_loss(pred_dict['root_vel_pred'], root_vel_gt)
            
            if 'root_trans_pred' in pred_dict:
                losses['root_trans'] = F.mse_loss(pred_dict['root_trans_pred'], trans_gt)
        
        # 手部位置损失
        if 'pred_hand_glb_pos' in pred_dict and position_global_gt is not None:
            hand_pos_gt = torch.stack([
                position_global_gt[:, :, 20, :],
                position_global_gt[:, :, 21, :],
            ], dim=2)
            losses['hand_pos'] = F.mse_loss(pred_dict['pred_hand_glb_pos'], hand_pos_gt)

        if 'contact_pred' in pred_dict:
            lfoot_contact_gt = batch.get('lfoot_contact')
            rfoot_contact_gt = batch.get('rfoot_contact')
            if isinstance(lfoot_contact_gt, torch.Tensor) and isinstance(rfoot_contact_gt, torch.Tensor):
                lfoot_contact_gt = lfoot_contact_gt.to(device=device, dtype=dtype)
                rfoot_contact_gt = rfoot_contact_gt.to(device=device, dtype=dtype)
                if lfoot_contact_gt.dim() == 1:
                    lfoot_contact_gt = lfoot_contact_gt.unsqueeze(0).expand(bs, -1)
                if rfoot_contact_gt.dim() == 1:
                    rfoot_contact_gt = rfoot_contact_gt.unsqueeze(0).expand(bs, -1)
                if lfoot_contact_gt.shape[:2] == (bs, seq) and rfoot_contact_gt.shape[:2] == (bs, seq):
                    contact_target = torch.stack([lfoot_contact_gt, rfoot_contact_gt], dim=-1).clamp(0.0, 1.0)
                    contact_pred = pred_dict['contact_pred'].to(device=device, dtype=dtype)
                    if contact_pred.shape == contact_target.shape:
                        losses['contact'] = F.binary_cross_entropy_with_logits(contact_pred, contact_target)
        
        # 加权求和
        total_loss = zero.clone()
        weighted_losses = {}
        for key, loss in losses.items():
            weight = self.weights.get(key, 1.0)
            # noTrans模式下，root相关损失权重为0
            if self.no_trans and key in {'root_vel_local', 'root_vel', 'root_trans'}:
                weight = 0.0
            weighted_losses[key] = loss * weight
            total_loss = total_loss + weighted_losses[key]
        
        return total_loss, losses, weighted_losses
    
    def compute_test_loss(self, pred_dict, batch, device):
        """计算测试损失（用于模型选择）"""
        total_loss, losses, _ = self.compute_loss(pred_dict, batch, device)
        
        # 只选择测试损失键
        test_losses = {k: v for k, v in losses.items() if k in self.TEST_LOSS_KEYS}
        
        # 重新计算测试总损失
        test_total = sum(test_losses.values())
        
        return test_total, test_losses
    
    @classmethod
    def get_loss_keys(cls):
        """返回损失键列表"""
        return list(cls.LOSS_KEYS)

