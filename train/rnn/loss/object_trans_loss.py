"""
ObjectTransModule的损失函数
"""
import torch
import torch.nn.functional as F


def _masked_mse(pred, target, mask, zero):
    """带mask的MSE损失"""
    if mask is None:
        return F.mse_loss(pred, target)
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return zero.clone()
    return F.mse_loss(pred[mask_bool], target[mask_bool])


def _masked_mean(values, mask, zero):
    """带mask的均值"""
    if mask is None:
        return values.mean()
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return zero.clone()
    return values[mask_bool].mean()


class ObjectTransLoss:
    """物体位置预测的损失计算"""
    
    # 训练阶段使用的损失键
    LOSS_KEYS = {
        'obj_trans',
        'lhand_obj_direction',
        'rhand_obj_direction',
        'lhand_lb',
        'rhand_lb',
        'hoi_error_l',
        'hoi_error_r',
        'obj_vel_cons',
        'obj_acc_cons',
    }
    
    # 测试阶段用于模型选择的损失键
    TEST_LOSS_KEYS = {
        'lhand_obj_direction',
        'rhand_obj_direction',
        'lhand_lb',
        'rhand_lb',
        'hoi_error_l',
        'hoi_error_r',
    }
    
    def __init__(self, weights=None):
        """
        Args:
            weights: dict, 各损失项的权重，默认为1.0
        """
        self.weights = weights or {}
    
    def __call__(self, pred_dict, batch, device):
        return self.compute_loss(pred_dict, batch, device)
    
    def compute_loss(self, pred_dict, batch, device):
        """计算损失"""
        human_imu = batch['human_imu'].to(device)
        dtype = human_imu.dtype
        bs, seq = human_imu.shape[:2]
        zero = human_imu.new_tensor(0.0)
        
        losses = {key: zero.clone() for key in self.LOSS_KEYS}
        
        # 获取GT数据
        obj_vel_gt = batch.get('obj_vel')
        if isinstance(obj_vel_gt, torch.Tensor):
            obj_vel_gt = obj_vel_gt.to(device)
        else:
            obj_vel_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)
        
        obj_imu_gt = batch.get('obj_imu')
        if isinstance(obj_imu_gt, torch.Tensor):
            obj_imu_gt = obj_imu_gt.to(device)
        else:
            obj_imu_gt = None
        
        obj_trans_gt = batch.get('obj_trans')
        if isinstance(obj_trans_gt, torch.Tensor):
            obj_trans_gt = obj_trans_gt.to(device)
        else:
            obj_trans_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)
        
        position_global_gt = batch.get('position_global')
        if isinstance(position_global_gt, torch.Tensor):
            position_global_gt = position_global_gt.to(device)
        else:
            position_global_gt = None
        
        lhand_dir_gt = batch.get('lhand_obj_direction')
        if isinstance(lhand_dir_gt, torch.Tensor):
            lhand_dir_gt = lhand_dir_gt.to(device)
        else:
            lhand_dir_gt = None
        
        rhand_dir_gt = batch.get('rhand_obj_direction')
        if isinstance(rhand_dir_gt, torch.Tensor):
            rhand_dir_gt = rhand_dir_gt.to(device)
        else:
            rhand_dir_gt = None
        
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
        
        # 构建has_object_mask
        has_object = batch.get('has_object')
        if isinstance(has_object, torch.Tensor):
            has_object_mask = has_object.to(device=device, dtype=torch.bool)
            if has_object_mask.dim() == 0:
                has_object_mask = has_object_mask.unsqueeze(0)
            if has_object_mask.dim() == 1:
                has_object_mask = has_object_mask.unsqueeze(1).expand(has_object_mask.shape[0], seq)
            else:
                has_object_mask = has_object_mask.bool()
        elif isinstance(has_object, (list, tuple)):
            has_object_mask = torch.tensor(has_object, device=device, dtype=torch.bool).unsqueeze(1).expand(-1, seq)
        elif isinstance(has_object, bool):
            has_object_mask = torch.full((bs, seq), has_object, device=device, dtype=torch.bool)
        else:
            has_object_mask = torch.ones(bs, seq, device=device, dtype=torch.bool)
        
        if has_object_mask.shape[0] != bs:
            has_object_mask = has_object_mask[0].unsqueeze(0).expand(bs, seq)
        
        # 只有在有物体时才计算损失
        if not has_object_mask.any():
            total_loss = zero.clone()
            weighted_losses = {key: zero.clone() for key in self.LOSS_KEYS}
            return total_loss, losses, weighted_losses
        
        obj_mask = has_object_mask
        
        # 计算各项损失
        if 'pred_obj_trans' in pred_dict:
            losses['obj_trans'] = _masked_mse(pred_dict['pred_obj_trans'], obj_trans_gt, obj_mask, zero)
        
        if 'pred_obj_vel_from_posdiff' in pred_dict:
            losses['obj_vel_cons'] = _masked_mse(pred_dict['pred_obj_vel_from_posdiff'], obj_vel_gt, obj_mask, zero)
        
        if 'pred_obj_acc_from_posdiff' in pred_dict and obj_imu_gt is not None:
            losses['obj_acc_cons'] = _masked_mse(pred_dict['pred_obj_acc_from_posdiff'], obj_imu_gt[:, :, :3], obj_mask, zero)
        
        if position_global_gt is not None:
            lhand_pos_gt = position_global_gt[:, :, 20, :]
            rhand_pos_gt = position_global_gt[:, :, 21, :]
            lb_l_gt = torch.norm(obj_trans_gt - lhand_pos_gt, dim=-1)
            lb_r_gt = torch.norm(obj_trans_gt - rhand_pos_gt, dim=-1)
            mask_l = (lhand_contact_gt & obj_mask)
            mask_r = (rhand_contact_gt & obj_mask)
            
            if 'pred_lhand_lb' in pred_dict:
                losses['lhand_lb'] = _masked_mse(pred_dict['pred_lhand_lb'], lb_l_gt, mask_l, zero)
            
            if 'pred_rhand_lb' in pred_dict:
                losses['rhand_lb'] = _masked_mse(pred_dict['pred_rhand_lb'], lb_r_gt, mask_r, zero)
            
            if 'pred_lhand_obj_direction' in pred_dict and lhand_dir_gt is not None:
                losses['lhand_obj_direction'] = _masked_mse(pred_dict['pred_lhand_obj_direction'], lhand_dir_gt, mask_l, zero)
            
            if 'pred_rhand_obj_direction' in pred_dict and rhand_dir_gt is not None:
                losses['rhand_obj_direction'] = _masked_mse(pred_dict['pred_rhand_obj_direction'], rhand_dir_gt, mask_r, zero)
            
            # HOI误差
            if 'pred_lhand_obj_direction' in pred_dict and 'pred_lhand_lb' in pred_dict and lhand_dir_gt is not None:
                if mask_l.any():
                    vec_gt_l = lhand_dir_gt * lb_l_gt.unsqueeze(-1)
                    vec_pred_l = pred_dict['pred_lhand_obj_direction'] * pred_dict['pred_lhand_lb'].unsqueeze(-1)
                    diff_l = torch.norm(vec_pred_l - vec_gt_l, dim=-1)
                    losses['hoi_error_l'] = _masked_mean(diff_l, mask_l, zero)
            
            if 'pred_rhand_obj_direction' in pred_dict and 'pred_rhand_lb' in pred_dict and rhand_dir_gt is not None:
                if mask_r.any():
                    vec_gt_r = rhand_dir_gt * lb_r_gt.unsqueeze(-1)
                    vec_pred_r = pred_dict['pred_rhand_obj_direction'] * pred_dict['pred_rhand_lb'].unsqueeze(-1)
                    diff_r = torch.norm(vec_pred_r - vec_gt_r, dim=-1)
                    losses['hoi_error_r'] = _masked_mean(diff_r, mask_r, zero)
        
        # 加权求和
        total_loss = zero.clone()
        weighted_losses = {}
        for key, loss in losses.items():
            weight = self.weights.get(key, 1.0)
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

