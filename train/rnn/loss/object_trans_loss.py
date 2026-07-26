"""
ObjectTransModule的损失函数
"""
import torch
import torch.nn.functional as F
from utils.rotation_conversions import matrix_to_rotation_6d
from utils.human_pose import select_hand_anchor_positions
from configs import FRAME_RATE, _REDUCED_POSE_NAMES


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
        'lhand_lb_cons',
        'rhand_lb_cons',
        'hoi_error_l',
        'hoi_error_r',
        'obj_vel_cons',
        'refine_pose',
        'refine_root_trans',
        'gate_weak',
    }

    # 测试阶段记录/上报的损失键
    TEST_LOSS_KEYS = {
        'obj_trans',
        'lhand_obj_direction',
        'rhand_obj_direction',
        'lhand_lb',
        'rhand_lb',
        'lhand_lb_cons',
        'rhand_lb_cons',
        'hoi_error_l',
        'hoi_error_r',
        'refine_pose',
        'refine_root_trans',
    }

    # 模型选择（best.pt）只累加 OT 核心项：refine_pose/refine_root_trans 的
    # 训练权重很大（服务人体微调头），计入选择总分会主导 checkpoint 的选取，
    # 而在线部署关心的是 OT 本身的质量。refine 项仍在 TEST_LOSS_KEYS 中记录。
    MODEL_SELECTION_KEYS = {
        'obj_trans',
        'lhand_obj_direction',
        'rhand_obj_direction',
        'lhand_lb',
        'rhand_lb',
        'lhand_lb_cons',
        'rhand_lb_cons',
        'hoi_error_l',
        'hoi_error_r',
    }
    
    def __init__(self, weights=None):
        """
        Args:
            weights: dict, 各损失项的权重，默认为1.0
        """
        self.weights = weights or {}

    @staticmethod
    def _build_gate_weak_target(contact_prob, eps=1e-6):
        """Build soft targets for [left_fk, right_fk, imu_velocity, static_hold]."""
        p_l = contact_prob[..., 0:1].clamp(0.0, 1.0)
        p_r = contact_prob[..., 1:2].clamp(0.0, 1.0)
        p_move = contact_prob[..., 2:3].clamp(0.0, 1.0)

        p_l_cond = (p_l / p_move.clamp_min(eps)).clamp(0.0, 1.0)
        p_r_cond = (p_r / p_move.clamp_min(eps)).clamp(0.0, 1.0)
        no_hand_cond = (1.0 - p_l_cond) * (1.0 - p_r_cond)

        scores = torch.cat(
            (
                p_l,
                p_r,
                p_move * no_hand_cond,
                1.0 - p_move,
            ),
            dim=-1,
        ).clamp_min(0.0)
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    
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
        
        obj_trans_gt = batch.get('obj_trans')
        if isinstance(obj_trans_gt, torch.Tensor):
            obj_trans_gt = obj_trans_gt.to(device)
        else:
            obj_trans_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

        trans_gt = batch.get('trans')
        if isinstance(trans_gt, torch.Tensor):
            trans_gt = trans_gt.to(device=device, dtype=dtype)
            if trans_gt.dim() == 2:
                trans_gt = trans_gt.unsqueeze(0).expand(bs, -1, -1)
            if trans_gt.shape[:2] != (bs, seq):
                trans_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)
        else:
            trans_gt = torch.zeros(bs, seq, 3, device=device, dtype=dtype)

        ori_root_reduced_gt = batch.get('ori_root_reduced')
        pose_gt_6d = None
        if isinstance(ori_root_reduced_gt, torch.Tensor):
            ori_root_reduced_gt = ori_root_reduced_gt.to(device=device, dtype=dtype)
            if ori_root_reduced_gt.dim() == 4:
                ori_root_reduced_gt = ori_root_reduced_gt.unsqueeze(0)
            if ori_root_reduced_gt.shape[0] == 1 and bs > 1:
                ori_root_reduced_gt = ori_root_reduced_gt.expand(bs, -1, -1, -1, -1)
            if (
                ori_root_reduced_gt.shape[:3] == (bs, seq, len(_REDUCED_POSE_NAMES))
                and ori_root_reduced_gt.shape[-2:] == (3, 3)
            ):
                pose_gt_6d = matrix_to_rotation_6d(
                    ori_root_reduced_gt.reshape(-1, 3, 3)
                ).reshape(bs, seq, len(_REDUCED_POSE_NAMES) * 6)
        
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

        gate_logits = pred_dict.get('gate_logits') if isinstance(pred_dict, dict) else None
        gating_weights = pred_dict.get('gating_weights') if isinstance(pred_dict, dict) else None
        gating_contact_prob = pred_dict.get('gating_contact_prob') if isinstance(pred_dict, dict) else None
        # 训练时优先用 GT 接触构建 gate 弱监督目标：若目标来自输入的（预测）
        # 接触概率，gate 只会被回归成"VC 说什么就信什么"，VC 出错时被一并强化；
        # 用 GT 目标才能让 gate 借助其他信号纠正错误的接触估计。
        gate_contact_source = None
        obj_contact_gt = batch.get('obj_contact')
        if (
            isinstance(batch.get('lhand_contact'), torch.Tensor)
            and isinstance(batch.get('rhand_contact'), torch.Tensor)
            and isinstance(obj_contact_gt, torch.Tensor)
            and lhand_contact_gt.shape == (bs, seq)
            and rhand_contact_gt.shape == (bs, seq)
        ):
            obj_contact_bool = obj_contact_gt.to(device).bool()
            if obj_contact_bool.shape == (bs, seq):
                gate_contact_source = torch.stack(
                    (
                        lhand_contact_gt.to(dtype=dtype),
                        rhand_contact_gt.to(dtype=dtype),
                        obj_contact_bool.to(dtype=dtype),
                    ),
                    dim=-1,
                )
        if gate_contact_source is None and (
            isinstance(gating_contact_prob, torch.Tensor)
            and gating_contact_prob.shape[:2] == (bs, seq)
            and gating_contact_prob.shape[-1] >= 3
        ):
            gate_contact_source = gating_contact_prob[..., :3].to(device=device, dtype=dtype)
        if gate_contact_source is not None:
            target = self._build_gate_weak_target(gate_contact_source).detach()
            smooth = float(self.weights.get('gate_weak_target_smoothing', 0.1))
            if smooth > 0.0:
                smooth = min(max(smooth, 0.0), 1.0)
                target = (1.0 - smooth) * target + smooth / target.shape[-1]
            if (
                isinstance(gate_logits, torch.Tensor)
                and gate_logits.shape[:2] == (bs, seq)
                and gate_logits.shape[-1] == 4
            ):
                gate_log_prob = F.log_softmax(gate_logits.to(device=device, dtype=dtype), dim=-1)
                gate_ce = -(target * gate_log_prob).sum(dim=-1)
                losses['gate_weak'] = _masked_mean(gate_ce, obj_mask, zero)
            elif (
                isinstance(gating_weights, torch.Tensor)
                and gating_weights.shape[:2] == (bs, seq)
                and gating_weights.shape[-1] == 4
            ):
                gate_log_prob = torch.log(gating_weights.to(device=device, dtype=dtype).clamp_min(1e-6))
                gate_ce = -(target * gate_log_prob).sum(dim=-1)
                losses['gate_weak'] = _masked_mean(gate_ce, obj_mask, zero)

        if pose_gt_6d is not None and 'refined_pose' in pred_dict:
            refined_pose = pred_dict['refined_pose']
            if refined_pose.dim() == 4:
                refined_pose = refined_pose.reshape(bs, seq, -1)
            if refined_pose.shape == pose_gt_6d.shape:
                losses['refine_pose'] = _masked_mse(refined_pose, pose_gt_6d, obj_mask, zero)

        if 'refined_root_trans' in pred_dict:
            refined_root_trans = pred_dict['refined_root_trans']
            if refined_root_trans.shape == trans_gt.shape:
                losses['refine_root_trans'] = _masked_mse(refined_root_trans, trans_gt, obj_mask, zero)
        
        if position_global_gt is not None:
            hand_pos_gt = select_hand_anchor_positions(position_global_gt)
            lhand_pos_gt = hand_pos_gt[:, :, 0, :]
            rhand_pos_gt = hand_pos_gt[:, :, 1, :]
            lb_l_gt = torch.norm(obj_trans_gt - lhand_pos_gt, dim=-1)
            lb_r_gt = torch.norm(obj_trans_gt - rhand_pos_gt, dim=-1)
            mask_l = (lhand_contact_gt & obj_mask)
            mask_r = (rhand_contact_gt & obj_mask)
            
            if 'pred_lhand_lb' in pred_dict:
                losses['lhand_lb'] = _masked_mse(pred_dict['pred_lhand_lb'], lb_l_gt, mask_l, zero)

            if 'pred_rhand_lb' in pred_dict:
                losses['rhand_lb'] = _masked_mse(pred_dict['pred_rhand_lb'], lb_r_gt, mask_r, zero)

            # 接触段内的骨长时序一致性：对齐 GT 的帧间差分而不是惩罚 |Δlb| 本身。
            # 刚性抓握时 GT 差分≈0（等价于恒定先验）；手在物体上滑动时 GT 差分
            # 非零，预测被引导跟随滑动而不是被错误地压平。差分乘 FRAME_RATE
            # 换算成 m/s，与 obj_vel_cons 同量纲。
            if seq > 1:
                if 'pred_lhand_lb' in pred_dict:
                    pred_delta_l = (pred_dict['pred_lhand_lb'][:, 1:] - pred_dict['pred_lhand_lb'][:, :-1]) * FRAME_RATE
                    gt_delta_l = (lb_l_gt[:, 1:] - lb_l_gt[:, :-1]) * FRAME_RATE
                    cons_mask_l = mask_l[:, 1:] & mask_l[:, :-1]
                    losses['lhand_lb_cons'] = _masked_mse(pred_delta_l, gt_delta_l, cons_mask_l, zero)

                if 'pred_rhand_lb' in pred_dict:
                    pred_delta_r = (pred_dict['pred_rhand_lb'][:, 1:] - pred_dict['pred_rhand_lb'][:, :-1]) * FRAME_RATE
                    gt_delta_r = (lb_r_gt[:, 1:] - lb_r_gt[:, :-1]) * FRAME_RATE
                    cons_mask_r = mask_r[:, 1:] & mask_r[:, :-1]
                    losses['rhand_lb_cons'] = _masked_mse(pred_delta_r, gt_delta_r, cons_mask_r, zero)
            
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
            default_weight = 0.0 if key == 'gate_weak' else 1.0
            weight = self.weights.get(key, default_weight)
            weighted_losses[key] = loss * weight
            total_loss = total_loss + weighted_losses[key]
        
        return total_loss, losses, weighted_losses
    
    def compute_test_loss(self, pred_dict, batch, device):
        """计算测试损失：上报 TEST_LOSS_KEYS，模型选择只累加 MODEL_SELECTION_KEYS。"""
        total_loss, losses, weighted_losses = self.compute_loss(pred_dict, batch, device)

        # 上报的测试分量
        test_losses = {k: v for k, v in weighted_losses.items() if k in self.TEST_LOSS_KEYS}

        # 模型选择总分只看 OT 核心项，避免 refine 大权重主导 best.pt
        selection_values = [v for k, v in test_losses.items() if k in self.MODEL_SELECTION_KEYS]
        if selection_values:
            test_total = sum(selection_values)
        else:
            human_imu = batch['human_imu'].to(device)
            test_total = human_imu.new_tensor(0.0)

        return test_total, test_losses
    
    @classmethod
    def get_loss_keys(cls):
        """返回损失键列表"""
        return list(cls.LOSS_KEYS)
