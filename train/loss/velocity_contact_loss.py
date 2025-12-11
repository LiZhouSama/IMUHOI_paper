"""
VelocityContactModule的损失函数
"""
import torch
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix

from configs import _SENSOR_POS_INDICES


class VelocityContactLoss:
    """速度和接触预测的损失计算"""
    
    # 损失键定义
    LOSS_KEYS = {'obj_vel', 'hand_vel', 'hand_contact'}
    
    def __init__(self, weights=None):
        """
        Args:
            weights: dict, 各损失项的权重，默认为1.0
        """
        self.weights = weights or {}
    
    def __call__(self, pred_dict, batch, device):
        """
        计算损失
        
        Args:
            pred_dict: 模型预测字典
            batch: 数据批次
            device: 计算设备
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        return self.compute_loss(pred_dict, batch, device)
    
    def compute_loss(self, pred_dict, batch, device):
        """计算损失"""
        human_imu = batch['human_imu'].to(device)
        dtype = human_imu.dtype
        bs, seq = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)
        
        losses = {key: zero.clone() for key in self.LOSS_KEYS}
        
        # 获取GT数据
        sensor_vel_glb_gt = batch.get('sensor_vel_glb')
        if isinstance(sensor_vel_glb_gt, torch.Tensor):
            sensor_vel_glb_gt = sensor_vel_glb_gt.to(device)
            if sensor_vel_glb_gt.dim() == 3:
                sensor_vel_glb_gt = sensor_vel_glb_gt.unsqueeze(0).expand(bs, -1, -1, -1)
        else:
            sensor_vel_glb_gt = torch.zeros(bs, seq, len(_SENSOR_POS_INDICES), 3, device=device, dtype=dtype)
        
        obj_vel_gt = batch.get('obj_vel')
        if isinstance(obj_vel_gt, torch.Tensor):
            obj_vel_gt = obj_vel_gt.to(device)
        else:
            obj_vel_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)
        
        lhand_contact_gt = batch.get('lhand_contact')
        if isinstance(lhand_contact_gt, torch.Tensor):
            lhand_contact_gt = lhand_contact_gt.to(device).bool()
        else:
            lhand_contact_gt = torch.zeros(bs, seq, device=device, dtype=torch.bool)
        
        rhand_contact_gt = batch.get('rhand_contact')
        if isinstance(rhand_contact_gt, torch.Tensor):
            rhand_contact_gt = rhand_contact_gt.to(device).bool()
        else:
            rhand_contact_gt = torch.zeros(bs, seq, device=device, dtype=torch.bool)
        
        obj_contact_gt = batch.get('obj_contact')
        if isinstance(obj_contact_gt, torch.Tensor):
            obj_contact_gt = obj_contact_gt.to(device).bool()
        else:
            obj_contact_gt = torch.zeros(bs, seq, device=device, dtype=torch.bool)
        
        hand_contact_gt = torch.stack([
            lhand_contact_gt.float(),
            rhand_contact_gt.float(),
            obj_contact_gt.float(),
        ], dim=-1)
        
        # 计算各项损失
        if 'pred_obj_vel' in pred_dict:
            losses['obj_vel'] = F.mse_loss(pred_dict['pred_obj_vel'], obj_vel_gt)
        
        if 'pred_hand_glb_vel' in pred_dict:
            hand_indices = [-2, -1]
            gt_hand_vel = sensor_vel_glb_gt[:, :, hand_indices, :]
            losses['hand_vel'] = F.mse_loss(pred_dict['pred_hand_glb_vel'], gt_hand_vel)
        
        if 'pred_hand_contact_prob' in pred_dict:
            losses['hand_contact'] = F.binary_cross_entropy(
                pred_dict['pred_hand_contact_prob'], hand_contact_gt
            )
        
        # 加权求和
        total_loss = zero.clone()
        weighted_losses = {}
        for key, loss in losses.items():
            weight = self.weights.get(key, 1.0)
            weighted_losses[key] = loss * weight
            total_loss = total_loss + weighted_losses[key]
        
        return total_loss, losses, weighted_losses
    
    @classmethod
    def get_loss_keys(cls):
        """返回损失键列表"""
        return list(cls.LOSS_KEYS)

