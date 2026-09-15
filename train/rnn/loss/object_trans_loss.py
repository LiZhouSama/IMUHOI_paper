"""
ObjectTransModule的损失函数
"""
import torch
import torch.nn.functional as F
from utils.rotation_conversions import matrix_to_rotation_6d
from utils.human_pose import select_hand_anchor_positions
from utils.object_fusion import build_contact_fusion_weights
from configs import FRAME_RATE, _REDUCED_POSE_NAMES


def resolve_object_trans_loss_weights(cfg):
    """Merge the arch-scoped ObjectTrans weights over the shared ``loss_weights``.

    ``loss_weights`` is shared by every stage and every ObjectTrans architecture,
    but the weights each architecture wants are not the same: ``current`` fuses
    through a *learned* four-way gate and is trained with the metric-aligned
    ``hoi_fused`` term, while v2/v3 drive the fusion with weights that are a fixed
    function of the contact probabilities and gain nothing from it.  Only the block
    named by ``cfg.object_trans_arch`` is applied, so changing one architecture's
    recipe never moves another's.  Architectures without a block keep the shared
    weights unchanged.
    """
    weights = dict(getattr(cfg, "loss_weights", None) or {})
    per_arch = getattr(cfg, "object_trans_arch_loss_weights", None) or {}
    if not isinstance(per_arch, dict):
        raise ValueError(
            "object_trans_arch_loss_weights must map an ObjectTrans arch name to a "
            f"weight block, got {type(per_arch).__name__}"
        )
    arch = str(getattr(cfg, "object_trans_arch", "current") or "current").strip().lower()
    overrides = per_arch.get(arch)
    if overrides is None:
        return weights
    if not isinstance(overrides, dict):
        raise ValueError(
            f"object_trans_arch_loss_weights[{arch!r}] must be a mapping of weights, "
            f"got {type(overrides).__name__}"
        )
    weights.update(dict(overrides))
    return weights


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
        'bigrasp_common',
        'bigrasp_proposal',
        'bigrasp_delta',
        'bigrasp_quality',
        'prior_hoi_l',
        'prior_hoi_r',
        'state_conf',
        'grasp_mode',
        'lhand_obj_direction',
        'rhand_obj_direction',
        'lhand_lb',
        'rhand_lb',
        'lhand_lb_cons',
        'rhand_lb_cons',
        'hoi_error_l',
        'hoi_error_r',
        'hoi_fused_l',
        'hoi_fused_r',
        'obj_vel_cons',
        'refine_pose',
        'refine_root_trans',
        'gate_weak',
    }

    # 测试阶段记录/上报的损失键
    TEST_LOSS_KEYS = {
        'obj_trans',
        'bigrasp_common',
        'bigrasp_proposal',
        'bigrasp_delta',
        'bigrasp_quality',
        'prior_hoi_l',
        'prior_hoi_r',
        'state_conf',
        'grasp_mode',
        'lhand_obj_direction',
        'rhand_obj_direction',
        'lhand_lb',
        'rhand_lb',
        'lhand_lb_cons',
        'rhand_lb_cons',
        'hoi_error_l',
        'hoi_error_r',
        'hoi_fused_l',
        'hoi_fused_r',
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
        'hoi_fused_l',
        'hoi_fused_r',
    }
    # Zero unless the config asks for them: the official HOI metric compares the
    # *fused* object relative to the hand OT consumed, on GT contact frames.
    # ``hoi_error_*`` only supervises the FK branch's bone, so a fusion that
    # leans on the IMU/static branches is never told what the metric wants.
    _ZERO_DEFAULT_WEIGHT_KEYS = (
        'gate_weak', 'hoi_fused_l', 'hoi_fused_r',
        'bigrasp_common', 'bigrasp_proposal', 'bigrasp_delta', 'bigrasp_quality',
        # state-trust terms (object_trans_state_trust): the state-free prior
        # FK's metric-aligned error and the BCE of the per-frame state
        # confidence against "state within state_conf_threshold of GT".
        'prior_hoi_l', 'prior_hoi_r', 'state_conf',
        # grasp-latent contact modes (object_trans_grasp_modes): BCE of the
        # per-hand rigidity against "GT bone speed below grasp_mode_speed_threshold".
        'grasp_mode',
    )
    
    def __init__(self, weights=None):
        """
        Args:
            weights: dict, 各损失项的权重，默认为1.0
        """
        self.weights = weights or {}
        self.state_conf_threshold = float((weights or {}).get('state_conf_threshold', 0.10))
        self.grasp_mode_speed_threshold = float((weights or {}).get('grasp_mode_speed_threshold', 0.05))
        self.bigrasp_quality_temperature = max(float(self.weights.get('bigrasp_quality_temperature', 0.05)), 1e-6)

    @staticmethod
    def _build_gate_weak_target(contact_prob, eps=1e-6):
        """Build soft targets for [left_fk, right_fk, imu_velocity, static_hold]."""
        return build_contact_fusion_weights(contact_prob, eps=eps)
    
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
        loss_mask = pred_dict.get('ot_loss_mask')
        if isinstance(loss_mask, torch.Tensor):
            if loss_mask.shape != obj_mask.shape:
                raise ValueError("ot_loss_mask must have shape [B,T]")
            obj_mask = obj_mask & loss_mask.to(device=device, dtype=torch.bool)
        
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
        if gate_contact_source is not None and not bool(pred_dict.get('gating_is_fixed', False)):
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

            # Metric-aligned HOI: fused object minus the hand OT consumed, against
            # GT object minus GT palm, on GT contact frames (metres, L2 per frame).
            hands_in = pred_dict.get('ot_hand_positions')
            fused = pred_dict.get('pred_obj_trans')
            if (
                isinstance(hands_in, torch.Tensor)
                and isinstance(fused, torch.Tensor)
                and hands_in.shape == (bs, seq, 2, 3)
                and fused.shape == (bs, seq, 3)
            ):
                hands_in = hands_in.to(device=device, dtype=fused.dtype)
                rel_gt_l = obj_trans_gt - lhand_pos_gt
                rel_gt_r = obj_trans_gt - rhand_pos_gt
                diff_fused_l = torch.norm((fused - hands_in[:, :, 0]) - rel_gt_l, dim=-1)
                diff_fused_r = torch.norm((fused - hands_in[:, :, 1]) - rel_gt_r, dim=-1)
                losses['hoi_fused_l'] = _masked_mean(diff_fused_l, mask_l, zero)
                losses['hoi_fused_r'] = _masked_mean(diff_fused_r, mask_r, zero)
                # State-free prior FK (state trust): same metric-aligned target.
                prior_l = pred_dict.get('prior_lhand_obj_trans')
                prior_r = pred_dict.get('prior_rhand_obj_trans')
                if isinstance(prior_l, torch.Tensor) and prior_l.shape == (bs, seq, 3):
                    diff_prior_l = torch.norm((prior_l - hands_in[:, :, 0]) - rel_gt_l, dim=-1)
                    losses['prior_hoi_l'] = _masked_mean(diff_prior_l, mask_l, zero)
                if isinstance(prior_r, torch.Tensor) and prior_r.shape == (bs, seq, 3):
                    diff_prior_r = torch.norm((prior_r - hands_in[:, :, 1]) - rel_gt_r, dim=-1)
                    losses['prior_hoi_r'] = _masked_mean(diff_prior_r, mask_r, zero)

            common_vector = pred_dict.get('bigrasp_vector_world')
            if isinstance(common_vector, torch.Tensor):
                both_mask = mask_l & mask_r
                target_vector = obj_trans_gt - hand_pos_gt.mean(dim=2)
                losses['bigrasp_common'] = _masked_mean((common_vector - target_vector).norm(dim=-1), both_mask, zero)
                proposal_vector = pred_dict['bigrasp_proposal_world']
                losses['bigrasp_proposal'] = _masked_mean((proposal_vector - target_vector).norm(dim=-1), both_mask, zero)
                if seq > 1:
                    pred_delta = common_vector[:, 1:] - common_vector[:, :-1]
                    target_delta = target_vector[:, 1:] - target_vector[:, :-1]
                    losses['bigrasp_delta'] = _masked_mean(
                        ((pred_delta - target_delta) * FRAME_RATE).norm(dim=-1), both_mask[:, 1:] & both_mask[:, :-1], zero,
                    )
                candidates = pred_dict['bigrasp_candidates'].detach()
                contact = torch.stack((mask_l, mask_r), dim=-1).to(candidates.dtype)
                relative_target = obj_trans_gt.unsqueeze(2) - hand_pos_gt
                relative_prediction = candidates.unsqueeze(3) - hands_in.detach().unsqueeze(2)
                candidate_error = (relative_prediction - relative_target.unsqueeze(2)).norm(dim=-1)
                candidate_error = (candidate_error * contact.unsqueeze(2)).sum(dim=-1) / contact.sum(dim=-1).clamp_min(1).unsqueeze(-1)
                target_quality = torch.softmax(-candidate_error / self.bigrasp_quality_temperature, dim=-1).detach()
                quality_kl = F.kl_div(F.log_softmax(pred_dict['bigrasp_quality_logits'].float(), dim=-1), target_quality.float(), reduction='none').sum(dim=-1)
                losses['bigrasp_quality'] = _masked_mean(quality_kl, both_mask, zero)
        # State confidence (state trust): the state used at frame t is the fused
        # object of frame t-1 (window init at t=0); it is "trustworthy" when it
        # lies within state_conf_threshold metres of the GT object at t-1.
        conf_logit = pred_dict.get('state_confidence_logit')
        state_prev = pred_dict.get('ot_state_prev_obj_trans')
        if (
            isinstance(conf_logit, torch.Tensor)
            and isinstance(state_prev, torch.Tensor)
            and conf_logit.shape == (bs, seq)
            and state_prev.shape == (bs, seq, 3)
        ):
            gt_prev = torch.cat((obj_trans_gt[:, :1], obj_trans_gt[:, :-1]), dim=1)
            state_err = torch.norm(state_prev.to(obj_trans_gt.dtype) - gt_prev, dim=-1)
            target = (state_err < self.state_conf_threshold).to(conf_logit.dtype)
            losses['state_conf'] = F.binary_cross_entropy_with_logits(conf_logit.float(), target.float())

        # Grasp-latent contact modes: the predicted per-hand rigidity P(rigid)
        # is supervised by whether the GT object-frame bone (dir * length) moves
        # slower than grasp_mode_speed_threshold m/s between consecutive GT
        # contact frames (rigid grasp) or faster (sliding / re-settling).
        # BCE on the logits: F.binary_cross_entropy on probabilities is
        # rejected under autocast (bf16/fp16 mixed precision).
        rigid_prob = pred_dict.get('grasp_rigid_logit')
        if (
            isinstance(rigid_prob, torch.Tensor)
            and rigid_prob.shape == (bs, seq, 2)
            and seq > 1
            and position_global_gt is not None
            and lhand_dir_gt is not None
            and rhand_dir_gt is not None
        ):
            hand_pos_gt = select_hand_anchor_positions(position_global_gt)
            bones_gt = torch.stack(
                (
                    lhand_dir_gt * torch.norm(obj_trans_gt - hand_pos_gt[:, :, 0, :], dim=-1, keepdim=True),
                    rhand_dir_gt * torch.norm(obj_trans_gt - hand_pos_gt[:, :, 1, :], dim=-1, keepdim=True),
                ),
                dim=2,
            )  # [B,T,2,3]
            bone_speed = (bones_gt[:, 1:] - bones_gt[:, :-1]).norm(dim=-1) * FRAME_RATE  # [B,T-1,2]
            rigid_target = (bone_speed < self.grasp_mode_speed_threshold).to(rigid_prob.dtype)
            contact_gt = torch.stack((lhand_contact_gt, rhand_contact_gt), dim=-1)  # [B,T,2]
            mode_mask = contact_gt[:, 1:] & contact_gt[:, :-1] & obj_mask[:, 1:].unsqueeze(-1)
            if mode_mask.any():
                bce = F.binary_cross_entropy_with_logits(rigid_prob[:, 1:].float(), rigid_target.float(), reduction='none')
                losses['grasp_mode'] = bce[mode_mask].mean()

        # 加权求和
        total_loss = zero.clone()
        weighted_losses = {}
        for key, loss in losses.items():
            default_weight = 0.0 if key in self._ZERO_DEFAULT_WEIGHT_KEYS else 1.0
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
