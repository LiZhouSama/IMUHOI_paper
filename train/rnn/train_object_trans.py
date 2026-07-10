"""
ObjectTransModule训练脚本 (Stage 3) + 可选自动联合训练
依赖Stage1和Stage2的预训练权重

用法:
    # 仅训练Stage3 (Object Trans)
    python train_object_trans.py --cfg config.yaml --vc_ckpt path/to/vc.pt --hp_ckpt path/to/hp.pt

    # 先训练Stage3，结束后在同一输出目录自动联合微调ObjectTrans + VelocityContact
    python train_object_trans.py --cfg config.yaml --vc_ckpt path/to/vc.pt --hp_ckpt path/to/hp.pt --joint_train
"""
import copy
import os
import sys
import glob
import torch
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import VelocityContactModule, HumanPoseModule, ObjectTransModule
from train.rnn.loss import VelocityContactLoss, ObjectTransLoss
from train.rnn.scheduled_inputs import (
    build_gt_human_pose_outputs,
    build_gt_velocity_contact_outputs,
    prediction_mix_probability,
    sample_mix_dict,
    sample_mix_tensor,
)
from train.rnn.train_utils import (
    get_base_args,
    merge_config,
    setup_seed,
    setup_device,
    create_save_dir,
    create_dataloaders,
    build_model_input_dict,
    flatten_lstm_parameters,
    save_config_snapshot,
    resolve_resume_checkpoint,
    get_model_state_dict,
    load_model_state_from_checkpoint,
    load_checkpoint,
    call_model_inference,
    _capture_rng_state,
    _restore_rng_state,
    _torch_load_checkpoint,
)
from torch.cuda.amp.grad_scaler import GradScaler
try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:
    SummaryWriter = None


def get_args():
    """获取命令行参数"""
    parser = get_base_args()
    parser.description = 'ObjectTransModule训练 (Stage 3) + 联合训练'
    
    # Stage3特有参数
    parser.add_argument('--vc_ckpt', type=str, default=None, 
                        help='VelocityContactModule权重路径')
    parser.add_argument('--hp_ckpt', type=str, default=None, 
                        help='HumanPoseModule权重路径')
    parser.add_argument('--ot_ckpt', type=str, default=None,
                        help='保留兼容；自动joint训练固定使用本次ObjectTrans训练得到的best.pt')
    parser.add_argument('--joint_train', action='store_true', 
                        help='Stage3训练完成后，自动在同一输出目录进行ObjectTrans + VelocityContact联合训练')
    parser.add_argument('--ablate_vc_boundary', action='store_true', default=None,
                        help='训练时将VelocityContact boundary输出置零')
    parser.add_argument('--ablate_ot_obs_encoder', action='store_true', default=None,
                        help='训练时将ObjectTrans obs_encoder输出置零')
	    
    return parser.parse_args()


def find_latest_checkpoint(base_dir, module_name):
    """自动搜索最新的检查点文件"""
    patterns = [
        os.path.join(base_dir, f'{module_name}*', 'best.pt'),
        # os.path.join(base_dir, f'{module_name}*', 'final.pt'),
        # os.path.join(base_dir, 'modules', f'{module_name}_best.pt'),
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        return None
    
    # 按修改时间排序，返回最新的
    all_files.sort(key=os.path.getmtime, reverse=True)
    return all_files[0]


def _cfg_get(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _resolve_pretrained_ckpt(cfg, cli_path, module_key, fallback_attr=None):
    """Resolve ckpt path: explicit CLI > cfg attr > pretrained_modules > auto-find."""
    if cli_path:
        return cli_path, "cli"
    if fallback_attr:
        cfg_path = getattr(cfg, fallback_attr, None)
        if cfg_path:
            return cfg_path, f"config.{fallback_attr}"
    pretrained_modules = getattr(cfg, "pretrained_modules", None)
    cfg_path = _cfg_get(pretrained_modules, module_key)
    if cfg_path:
        return cfg_path, f"config.pretrained_modules.{module_key}"
    return None, None


class Stage3JointTrainer:
    """Stage3和联合训练器"""
    
    def __init__(self, cfg, vc_model, hp_model, ot_model, train_loader, test_loader=None, joint_train=False):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.joint_train = joint_train
        self.vc_model = vc_model.to(self.device)
        self.hp_model = hp_model.to(self.device)
        self.ot_model = ot_model.to(self.device)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.resume_path = resolve_resume_checkpoint(getattr(cfg, "resume_dir", None))
        self.resume_checkpoint = None
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.n_iter = 0
	        
        self.vc_trainable = bool(joint_train)

        # 设置训练/冻结状态：HP始终冻结；VC仅在joint阶段参与训练
        print("Stage3模式：冻结HumanPose，训练ObjectTrans")
        for param in self.hp_model.parameters():
            param.requires_grad = False
        for param in self.vc_model.parameters():
            param.requires_grad = self.vc_trainable
        for param in self.ot_model.parameters():
            param.requires_grad = True
        if joint_train:
            print("联合训练模式：从epoch 0开始微调VelocityContact + ObjectTrans")

        if self.resume_path:
            print(f"Resume checkpoint: {self.resume_path}")
            self.resume_checkpoint = _torch_load_checkpoint(self.resume_path, map_location=self.device)
            self._restore_model_weights(self.resume_checkpoint)
	        
        # 多GPU包装
        if cfg.use_multi_gpu:
            print(f'多GPU训练: {cfg.gpus}')
            self.vc_model = torch.nn.DataParallel(self.vc_model, device_ids=cfg.gpus)
            self.hp_model = torch.nn.DataParallel(self.hp_model, device_ids=cfg.gpus)
            self.ot_model = torch.nn.DataParallel(self.ot_model, device_ids=cfg.gpus)
        
        flatten_lstm_parameters(self.vc_model)
        flatten_lstm_parameters(self.hp_model)
        flatten_lstm_parameters(self.ot_model)
        
        # 优化器
        base_lr = cfg.lr
        if cfg.use_multi_gpu:
            base_lr = base_lr * len(cfg.gpus)

        param_groups = [
            {
                "params": [p for p in self.ot_model.parameters() if p.requires_grad],
                "lr": base_lr,
                "name": "object_trans",
            }
        ]
        if joint_train:
            param_groups.append(
                {
                    "params": [p for p in self.vc_model.parameters() if p.requires_grad],
                    "lr": base_lr,
                    "name": "velocity_contact",
                }
            )

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=cfg.milestones, gamma=cfg.gamma
        )
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        if self.resume_checkpoint is not None:
            self._restore_training_state(self.resume_checkpoint)

        ot_trainable = sum(p.numel() for p in self.ot_model.parameters() if p.requires_grad)
        vc_params = sum(p.numel() for p in self.vc_model.parameters())
        print(f"ObjectTrans可训练参数数量: {ot_trainable}, lr={base_lr}")
        if joint_train:
            print(f"VelocityContact可训练参数数量: {vc_params}, lr={base_lr}")
        
        # 损失函数
        loss_weights = getattr(cfg, 'loss_weights', {})
        self.vc_loss_fn = VelocityContactLoss(weights=loss_weights) if joint_train else None
        self.ot_loss_fn = ObjectTransLoss(weights=loss_weights)
        
        # TensorBoard
        self.writer = None
        if cfg.use_tensorboard and not cfg.debug and SummaryWriter is not None:
            log_dir = os.path.join(cfg.save_dir, 'tensorboard_logs')
            self.writer = SummaryWriter(log_dir=log_dir)
        elif cfg.use_tensorboard and not cfg.debug:
            print('TensorBoard is unavailable; continue without SummaryWriter.')
        
    def _restore_model_weights(self, checkpoint):
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Resume checkpoint must be a dict: {self.resume_path}")

        ot_key = "object_trans_state_dict" if checkpoint.get("object_trans_state_dict") is not None else None
        load_model_state_from_checkpoint(
            self.ot_model,
            checkpoint,
            strict=False,
            state_key=ot_key,
            name="ObjectTrans",
        )
        print(f"恢复ObjectTrans权重: {self.resume_path}")

        if checkpoint.get("velocity_contact_state_dict") is not None:
            load_model_state_from_checkpoint(
                self.vc_model,
                checkpoint,
                strict=False,
                state_key="velocity_contact_state_dict",
                name="VelocityContact",
            )
            print(f"恢复VelocityContact权重: {self.resume_path}")

    def _restore_training_state(self, checkpoint):
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except Exception as exc:
                print(f"警告：恢复 optimizer_state_dict 失败，将只恢复权重: {exc}")
        else:
            print("警告：resume checkpoint 缺少 optimizer_state_dict，将只恢复权重。")

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except Exception as exc:
                print(f"警告：恢复 scheduler_state_dict 失败，学习率调度器将从当前配置重新开始: {exc}")
        else:
            print("警告：resume checkpoint 缺少 scheduler_state_dict，学习率调度器将从当前配置重新开始。")

        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state is not None:
            try:
                self.scaler.load_state_dict(scaler_state)
            except Exception as exc:
                print(f"警告：恢复 scaler_state_dict 失败: {exc}")

        completed_epoch = int(checkpoint.get("epoch", -1))
        self.start_epoch = completed_epoch + 1
        self.n_iter = int(checkpoint.get("n_iter", self.start_epoch * max(len(self.train_loader), 1)))
        self.best_loss = self._resolve_resume_best_loss(checkpoint)
        _restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"恢复训练状态: completed_epoch={completed_epoch}, "
            f"next_epoch={self.start_epoch}, n_iter={self.n_iter}, best_loss={self.best_loss:.6f}"
        )

    def _resolve_resume_best_loss(self, checkpoint):
        if isinstance(checkpoint.get("best_loss"), (int, float)):
            return float(checkpoint["best_loss"])
        if os.path.basename(self.resume_path or "") == "best.pt" and isinstance(checkpoint.get("loss"), (int, float)):
            return float(checkpoint["loss"])

        best_path = os.path.join(self.cfg.save_dir, "best.pt")
        if os.path.isfile(best_path):
            try:
                best_checkpoint = _torch_load_checkpoint(best_path, map_location="cpu")
                if isinstance(best_checkpoint, dict) and isinstance(best_checkpoint.get("loss"), (int, float)):
                    return float(best_checkpoint["loss"])
            except Exception as exc:
                print(f"警告：读取 best.pt 的 best_loss 失败: {exc}")
        return float("inf")

    def forward_all(self, data_dict, batch=None, epoch: int = 0, training: bool = True):
        """完整的前向传播"""
        self.hp_model.eval()
        if self.vc_trainable and training:
            self.vc_model.train()
        else:
            self.vc_model.eval()

        with torch.no_grad():
            hp_out = call_model_inference(self.hp_model, data_dict, inference_mode="offline")

        gt_hp = None
        if training and batch is not None:
            pred_prob = prediction_mix_probability(epoch, self.cfg)
            gt_hp = build_gt_human_pose_outputs(batch, self.device, dtype=hp_out['pred_hand_glb_pos'].dtype)
            hp_for_vc = sample_mix_dict(
                gt_hp,
                hp_out,
                (
                    "p_pred",
                    "pred_full_pose_6d",
                    "pred_joints_local",
                    "pred_joints_global",
                    "pred_hand_glb_pos",
                    "pred_palm_glb_pos",
                    "root_vel_pred",
                    "root_trans_pred",
                ),
                pred_prob,
            )
        else:
            hp_for_vc = hp_out

        vc_grad_enabled = self.vc_trainable and training
        with torch.set_grad_enabled(vc_grad_enabled):
            vc_out = call_model_inference(self.vc_model, data_dict, hp_out=hp_for_vc, inference_mode="offline")

        if training and batch is not None:
            pred_prob = prediction_mix_probability(epoch, self.cfg)
            if gt_hp is None:
                gt_hp = build_gt_human_pose_outputs(batch, self.device, dtype=hp_out['pred_hand_glb_pos'].dtype)
            gt_vc = build_gt_velocity_contact_outputs(batch, self.device, dtype=vc_out['pred_obj_vel'].dtype)
            hand_positions, ot_mask = sample_mix_tensor(
                gt_hp.get('pred_palm_glb_pos', gt_hp['pred_hand_glb_pos']),
                hp_out.get('pred_palm_glb_pos', hp_out['pred_hand_glb_pos']),
                pred_prob,
                return_mask=True,
            )
            contact_prob = sample_mix_tensor(
                gt_vc['pred_hand_contact_prob'],
                vc_out['pred_hand_contact_prob'],
                pred_prob,
                mask=ot_mask,
            )
            obj_vel_input = sample_mix_tensor(
                gt_vc['pred_obj_vel'],
                vc_out['pred_obj_vel'],
                pred_prob,
                mask=ot_mask,
            )
            human_pose_input = sample_mix_tensor(
                gt_hp['p_pred'],
                hp_out['p_pred'],
                pred_prob,
                mask=ot_mask,
            )
            root_trans_input = sample_mix_tensor(
                gt_hp['root_trans_pred'],
                hp_out['root_trans_pred'],
                pred_prob,
                mask=ot_mask,
            )
        else:
            hand_positions = hp_out.get('pred_palm_glb_pos', hp_out['pred_hand_glb_pos'])
            contact_prob = vc_out['pred_hand_contact_prob']
            obj_vel_input = vc_out['pred_obj_vel']
            human_pose_input = hp_out.get('p_pred')
            root_trans_input = hp_out.get('root_trans_pred')

        ot_out = call_model_inference(
            self.ot_model,
            hand_positions,
            contact_prob,
            data_dict['obj_trans_init'],
            obj_imu=data_dict['obj_imu'],
            human_imu=data_dict['human_imu'],
            obj_vel_input=obj_vel_input,
            contact_init=data_dict['contact_init'],
            has_object_mask=data_dict['has_object'],
            human_pose_input=human_pose_input,
            root_trans_input=root_trans_input,
            obj_points_canonical=data_dict.get('obj_points_canonical'),
            obj_rot_gt=data_dict.get('obj_rot_gt'),
            obj_trans_gt=data_dict.get('obj_trans_gt'),
            obj_scale_gt=data_dict.get('obj_scale_gt'),
            enable_refine=getattr(self.cfg, "enable_ot_refine", False),
            inference_mode="offline",
        )

        results = {}
        results.update(vc_out)
        results.update(hp_out)
        results.update(ot_out)

        return results, vc_out, hp_out, ot_out

    def train_epoch(self, epoch):
            """训练一个epoch"""
            self.vc_model.train() if self.vc_trainable else self.vc_model.eval()
            self.hp_model.eval()
            self.ot_model.train()
            
            total_loss = 0
            loss_components = {}
            
            train_iter = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)
            
            for batch in train_iter:
                data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)
                
                self.optimizer.zero_grad()
                
                results, vc_out, hp_out, ot_out = self.forward_all(data_dict, batch=batch, epoch=epoch, training=True)
                
                # 计算损失
                losses = {}
                ot_losses = {}
                vc_losses = {}

                if self.joint_train and self.vc_trainable and self.vc_loss_fn:
                    vc_loss, vc_losses, vc_weighted_losses = self.vc_loss_fn(vc_out, batch, self.device)
                    losses.update({f'vc_{k}': v for k, v in vc_weighted_losses.items()})
                else:
                    vc_loss = torch.tensor(0.0, device=self.device)
                    vc_weighted_losses = {}

                ot_loss, ot_losses, ot_weighted_losses = self.ot_loss_fn(ot_out, batch, self.device)
                losses.update({f'ot_{k}': v for k, v in ot_weighted_losses.items()})

                # 总损失
                batch_loss = vc_loss + ot_loss
                
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += batch_loss.item()
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[key] = loss_components.get(key, 0) + value.item()
                
                # 更新进度条
                postfix = {'loss': batch_loss.item(), 'ot': ot_loss.item()}
                if self.joint_train and self.vc_trainable:
                    postfix['vc'] = vc_loss.item()
                train_iter.set_postfix(postfix)
                
                if self.writer is not None:
                    self.writer.add_scalar('train/total_loss', batch_loss.item(), self.n_iter)
                    self.writer.add_scalar('train/ot_loss', ot_loss.item(), self.n_iter)
                    if self.joint_train and self.vc_trainable:
                        self.writer.add_scalar('train/vc_loss', vc_loss.item(), self.n_iter)
                    for key, value in ot_weighted_losses.items():
                        if isinstance(value, torch.Tensor):
                            self.writer.add_scalar(f'train/ot/{key}', value.item(), self.n_iter)
                    if self.joint_train and self.vc_trainable:
                        for key, value in vc_weighted_losses.items():
                            if isinstance(value, torch.Tensor):
                                self.writer.add_scalar(f'train/vc/{key}', value.item(), self.n_iter)
                
                self.n_iter += 1
            
            total_loss /= len(self.train_loader)
            for key in loss_components:
                loss_components[key] /= len(self.train_loader)
            
            return total_loss, loss_components
    
    def evaluate(self, epoch):
        """评估"""
        if self.test_loader is None:
            return None, {}
        
        self.vc_model.eval()
        self.hp_model.eval()
        self.ot_model.eval()
        
        total_loss = 0
        loss_components = {}
        
        with torch.no_grad():
            test_iter = tqdm(self.test_loader, desc=f'Test {epoch}', leave=False)
            
            for batch in test_iter:
                data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=False)
                results, vc_out, hp_out, ot_out = self.forward_all(data_dict, batch=batch, epoch=epoch, training=False)

                # 计算测试损失
                vc_loss = torch.tensor(0.0, device=self.device)
                vc_losses = {}
                if self.joint_train and self.vc_trainable and self.vc_loss_fn:
                    vc_loss, vc_losses, _ = self.vc_loss_fn(vc_out, batch, self.device)

                ot_loss, ot_losses = self.ot_loss_fn.compute_test_loss(ot_out, batch, self.device)

                combined_loss = ot_loss + vc_loss
                total_loss += combined_loss.item()
                for key, value in ot_losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[key] = loss_components.get(key, 0) + value.item()
                for key, value in vc_losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[f"vc_{key}"] = loss_components.get(f"vc_{key}", 0) + value.item()
        
        total_loss /= len(self.test_loader)
        for key in loss_components:
            loss_components[key] /= len(self.test_loader)
        
        return total_loss, loss_components
    
    def train(self):
        """完整训练循环"""
        max_epoch = self.cfg.epochs
        train_loss = 0.0
        if self.start_epoch >= max_epoch:
            print(f"Resume checkpoint already reached epoch {self.start_epoch - 1}; target epochs={max_epoch}.")
            if self.writer is not None:
                self.writer.close()
            return
	        
        for epoch in range(self.start_epoch, max_epoch):
            train_loss, train_components = self.train_epoch(epoch)
	            
            print(f'\rEpoch {epoch}, Train Loss: {train_loss:.4f}', end='')
            
            if epoch % 10 == 0 and self.test_loader is not None:
                test_loss, test_components = self.evaluate(epoch)
                
                if test_loss is not None:
                    print(f', Test Loss: {test_loss:.4f}')
	                    
                    if test_loss < self.best_loss:
                        self.best_loss = test_loss
                        self._save_models(
                            epoch,
                            test_loss,
                            'best',
                            {
                                'test_components': test_components,
                                'best_loss': self.best_loss,
                                'n_iter': self.n_iter,
                            },
                        )
                        print(f'新的最佳测试损失: {self.best_loss:.4f}')
                    
                    if self.writer is not None:
                        self.writer.add_scalar('test/total_loss', test_loss, self.n_iter)
                        for key, value in test_components.items():
                            if key in ObjectTransLoss.TEST_LOSS_KEYS:
                                self.writer.add_scalar(f'test/ot/{key}', value, self.n_iter)
                            elif key.startswith('vc_'):
                                self.writer.add_scalar(f'test/vc/{key[3:]}', value, self.n_iter)
                            else:
                                self.writer.add_scalar(f'test/{key}', value, self.n_iter)
            else:
                print()
                if self.test_loader is None and train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_models(
                        epoch,
                        train_loss,
                        'best',
                        {
                            'train_components': train_components,
                            'best_loss': self.best_loss,
                            'n_iter': self.n_iter,
                        },
                    )
                    print(f'新的最佳训练损失: {self.best_loss:.4f}')
	            
            self.scheduler.step()
            self._save_models(
                epoch,
                train_loss,
                'last',
                {
                    'train_components': train_components,
                    'best_loss': self.best_loss,
                    'n_iter': self.n_iter,
                },
            )
	        
        # 保存最终模型
        self._save_models(
            max_epoch - 1,
            train_loss,
            'final',
            {
                'best_loss': self.best_loss,
                'n_iter': self.n_iter,
            },
        )
	        
        if self.writer is not None:
            self.writer.close()
	    
    def _write_checkpoint(self, path, checkpoint):
        tmp_path = f"{path}.tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def _build_joint_checkpoint(self, epoch, loss, extra=None):
        ot_state = get_model_state_dict(self.ot_model)
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': ot_state,
            'module_state_dict': ot_state,
            'object_trans_state_dict': ot_state,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'rng_state': _capture_rng_state(),
        }
        if self.joint_train:
            checkpoint['velocity_contact_state_dict'] = get_model_state_dict(self.vc_model)
        if extra:
            checkpoint.update(extra)
        return checkpoint

    def _save_module_checkpoint(self, path, state_dict, epoch, loss):
        self._write_checkpoint(
            path,
            {
                'epoch': epoch,
                'module_state_dict': state_dict,
                'model_state_dict': state_dict,
                'loss': loss,
                'best_loss': self.best_loss,
                'n_iter': self.n_iter,
            },
        )

    def _save_models(self, epoch, loss, prefix, extra=None):
        """保存模型"""
        checkpoint = self._build_joint_checkpoint(epoch, loss, extra=extra)

        if self.joint_train:
            # joint阶段与OT阶段共用目录，但不覆盖OT的best.pt/last.pt/final.pt。
            ot_path = os.path.join(self.cfg.save_dir, f'{prefix}_object_trans.pt')
            vc_path = os.path.join(self.cfg.save_dir, f'{prefix}_velocity_contact.pt')
            self._save_module_checkpoint(ot_path, checkpoint['object_trans_state_dict'], epoch, loss)
            self._save_module_checkpoint(vc_path, checkpoint['velocity_contact_state_dict'], epoch, loss)
        else:
            checkpoint_path = os.path.join(self.cfg.save_dir, f'{prefix}.pt')
            self._write_checkpoint(checkpoint_path, checkpoint)

        print(f'保存模型: {prefix}')


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _load_object_trans_best_for_joint(ot_model, best_path, device):
    if not os.path.isfile(best_path):
        raise FileNotFoundError(f"自动joint训练需要ObjectTrans训练结果，但未找到: {best_path}")
    checkpoint = _torch_load_checkpoint(best_path, map_location=device)
    state_key = "object_trans_state_dict" if isinstance(checkpoint, dict) and checkpoint.get("object_trans_state_dict") is not None else None
    load_model_state_from_checkpoint(
        _unwrap_model(ot_model),
        checkpoint,
        strict=False,
        state_key=state_key,
        name="ObjectTrans",
    )
    print(f"joint训练起点加载ObjectTrans最佳权重: {best_path}")


def _run_auto_joint_train(cfg, trainer, train_loader, test_loader, base_lr):
    joint_epochs = int(getattr(cfg, "joint_train_epochs", 0))
    if joint_epochs <= 0:
        raise ValueError("启用 --joint_train 时，配置 joint_train_epochs 必须大于0")

    joint_milestones = list(getattr(cfg, "joint_train_milestones", []))
    joint_cfg = copy.deepcopy(cfg)
    joint_cfg.joint_train = True
    joint_cfg.epochs = joint_epochs
    joint_cfg.milestones = joint_milestones
    joint_cfg.lr = float(base_lr) * 0.1
    joint_cfg.resume_dir = None

    best_ot_path = os.path.join(cfg.save_dir, "best.pt")
    _load_object_trans_best_for_joint(trainer.ot_model, best_ot_path, trainer.device)
    save_config_snapshot(
        joint_cfg,
        extra={
            "joint_train_auto_started_from": best_ot_path,
            "joint_train_lr_source": "lr * 0.1",
        },
        filename="joint_train_config.yaml",
    )

    print("=" * 50)
    print("自动联合训练: ObjectTrans + VelocityContact")
    print(f"训练轮数: {joint_cfg.epochs}")
    print(f"学习率: {joint_cfg.lr}")
    print(f"milestones: {joint_cfg.milestones}")
    print(f"保存目录: {joint_cfg.save_dir}")
    print("=" * 50)

    joint_trainer = Stage3JointTrainer(
        joint_cfg,
        _unwrap_model(trainer.vc_model),
        _unwrap_model(trainer.hp_model),
        _unwrap_model(trainer.ot_model),
        train_loader,
        test_loader,
        joint_train=True,
    )
    joint_trainer.train()
    return joint_trainer


def main():
    """主函数"""
    args = get_args()
    cfg = merge_config(args)
    cfg.ablate_vc_boundary = (
        bool(args.ablate_vc_boundary)
        if args.ablate_vc_boundary is not None
        else bool(getattr(cfg, "ablate_vc_boundary", False))
    )
    cfg.ablate_ot_obs_encoder = (
        bool(args.ablate_ot_obs_encoder)
        if args.ablate_ot_obs_encoder is not None
        else bool(getattr(cfg, "ablate_ot_obs_encoder", False))
    )
    if cfg.ablate_ot_obs_encoder:
        cfg.cond_mode_probs = [0.0, 1.0, 0.0]
    auto_joint_train = bool(args.joint_train) or bool(getattr(cfg, "joint_train", False))
    base_train_lr = float(cfg.lr)
    cfg.joint_train = auto_joint_train
		    
    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
	    
    module_name = 'object_trans'
    ablation_suffix = ""
    if cfg.ablate_vc_boundary:
        ablation_suffix += "_vc_boundary_zero"
    if cfg.ablate_ot_obs_encoder:
        ablation_suffix += "_ot_obs_encoder_zero"
    module_name = f"{module_name}{ablation_suffix}"
    save_dir = create_save_dir(cfg, module_name)
    save_config_snapshot(cfg)
    
    mode_str = "Stage 3 + 自动联合训练" if auto_joint_train else "Stage 3"
    trans_str = "noTrans" if cfg.no_trans else "普通"
    
    print("=" * 50)
    print(f"{mode_str}: ObjectTransModule训练 ({trans_str}模式)")
    print(f"设备: {cfg.device}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"训练轮数: {cfg.epochs}")
    print(f"noTrans模式: {cfg.no_trans}")
    print(f"VC boundary ablation: {'enabled' if cfg.ablate_vc_boundary else 'disabled'}")
    print(f"OT obs_encoder ablation: {'enabled' if cfg.ablate_ot_obs_encoder else 'disabled'}")
    print(f"保存目录: {save_dir}")
    print("=" * 50)
    
    # 查找/加载预训练权重
    vc_ckpt, vc_source = _resolve_pretrained_ckpt(cfg, args.vc_ckpt, 'velocity_contact', 'vc_ckpt')
    hp_ckpt, hp_source = _resolve_pretrained_ckpt(cfg, args.hp_ckpt, 'human_pose', 'hp_ckpt')
    ot_ckpt, ot_source = _resolve_pretrained_ckpt(cfg, args.ot_ckpt, 'object_trans', 'ot_ckpt')

    base_save_dir = os.path.dirname(save_dir)
    if vc_ckpt is None:
        vc_ckpt = find_latest_checkpoint(base_save_dir, 'velocity_contact')
        vc_source = "auto"
    print(f"VelocityContact权重({vc_source or 'missing'}): {vc_ckpt}")
    if hp_ckpt is None:
        hp_ckpt = find_latest_checkpoint(base_save_dir, 'human_pose')
        hp_source = "auto"
    print(f"HumanPose权重({hp_source or 'missing'}): {hp_ckpt}")
    if ot_ckpt:
        print(f"ObjectTrans权重配置({ot_source or 'missing'})已记录但不会作为自动joint起点: {ot_ckpt}")
	    
    cfg.vc_ckpt = vc_ckpt
    cfg.hp_ckpt = hp_ckpt
    cfg.ot_ckpt = ot_ckpt
    cfg.joint_train = auto_joint_train
    save_config_snapshot(cfg)

    if vc_ckpt is None or hp_ckpt is None:
        print("警告: 未找到预训练权重，将使用随机初始化")
        print("建议先训练Stage1和Stage2，或使用--auto_find_ckpt自动搜索")
    
    # 创建数据加载器
    train_loader, test_loader = create_dataloaders(cfg)
    
    if train_loader is None or len(train_loader) == 0:
        print("错误: 无法创建训练数据加载器")
        return
    
    # 创建模型
    device = torch.device(cfg.device)
    
    vc_model = VelocityContactModule(cfg)
    if vc_ckpt and os.path.exists(vc_ckpt):
        load_checkpoint(vc_model, vc_ckpt, device, strict=False)
    elif vc_ckpt:
        print(f"警告: VelocityContact权重不存在: {vc_ckpt}")

    hp_model = HumanPoseModule(cfg, device, no_trans=cfg.no_trans)
    if hp_ckpt and os.path.exists(hp_ckpt):
        load_checkpoint(hp_model, hp_ckpt, device)
    elif hp_ckpt:
        print(f"警告: HumanPose权重不存在: {hp_ckpt}")
    
    ot_model = ObjectTransModule(cfg)
    if getattr(cfg, "resume_dir", None):
        print("检测到 --resume_dir，ObjectTrans权重将由训练器从断点恢复。")
    
    print(f"VelocityContact参数量: {sum(p.numel() for p in vc_model.parameters())}")
    print(f"HumanPose参数量: {sum(p.numel() for p in hp_model.parameters())}")
    print(f"ObjectTrans参数量: {sum(p.numel() for p in ot_model.parameters())}")
    
    # 创建训练器
    trainer = Stage3JointTrainer(
        cfg, vc_model, hp_model, ot_model,
        train_loader, test_loader,
        joint_train=False
    )
    
    # 开始训练
    trainer.train()

    if auto_joint_train:
        _run_auto_joint_train(cfg, trainer, train_loader, test_loader, base_train_lr)
    
    print(f"\n训练完成！模型保存到: {save_dir}")


if __name__ == "__main__":
    main()
