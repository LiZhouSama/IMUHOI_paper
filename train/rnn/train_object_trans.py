"""
ObjectTransModule训练脚本 (Stage 3) + 可选自动联合训练
依赖Stage1和Stage2的预训练权重

用法:
    # 仅训练Stage3 (Object Trans)
    python train_object_trans.py --cfg config.yaml --vc_ckpt path/to/vc.pt --hp_ckpt path/to/hp.pt

    # 先训练Stage3，结束后自动联合微调；IMUHOI为HumanPose + ObjectTrans，
    # 其他架构为VelocityContact + ObjectTrans
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

from model import HumanPoseModule, ObjectTransModule
from model.rnn.velocity_contact import (
    VELOCITY_CONTACT_ARCH_CHOICES,
    build_velocity_contact_module,
    normalize_velocity_contact_arch,
)
from model.rnn.object_trans_legacy import LegacyObjectTransModule
from model.rnn.object_trans_IMUHOI import (
    IMUHOIObjectTransModule,
    is_imuhoi_object_trans_arch,
)
from model.rnn.object_trans_IMUHOI_4heads import (
    IMUHOI4HeadsObjectTransModule,
)
from model.rnn.object_trans_legacy_mesh import LegacyMeshObjectTransModule
from model.rnn.object_trans_v2 import V2ObjectTransModule
from model.rnn import V3ObjectTransModule
from model.rnn.object_trans_mergeHeads_obsEncoder import (
    ObjectTransModule as MergeHeadsObsObjectTransModule,
)
from model.rnn.lb_vq_diffusion import FrozenLBPriorPredictor
from train.rnn.loss import (
    HumanPoseLoss,
    VelocityContactLoss,
    V2VelocityContactLoss,
    ObjectTransLoss,
    IMUHOIObjectTransLoss,
    resolve_object_trans_loss_weights,
)
from train.rnn.scheduled_inputs import (
    build_gt_human_pose_outputs,
    build_gt_velocity_contact_outputs,
    prediction_mix_probability,
    sample_gt_lb_prior_stats,
    sample_mix_dict,
    sample_mix_tensor,
)
from train.rnn.train_utils import (
    should_stop_early as _should_stop_early,
    get_base_args,
    merge_config,
    setup_seed,
    setup_device,
    create_save_dir,
    create_dataloaders,
    prime_dataloader_workers,
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
    autocast,
    resolve_amp_settings,
)
from torch.nn.utils import clip_grad_norm_
from train.rnn.ot_history_training import forward_with_ot_history
from train.rnn.train_utils import resumed_last_improvement_epoch as _resumed_last_improvement_epoch
try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:
    SummaryWriter = None


OBJECT_TRANS_ARCH_CHOICES = (
    'current',
    'v2',
    'v3',
    'legacy',
    'legacy_mesh',
    'mergeHeads_obs',
    'imuhoi',
    'imuhoi_4heads',
)


def _normalize_object_trans_arch(value):
    normalized = str(value).strip().lower()
    if normalized == 'mergeheads_obs':
        return 'mergeHeads_obs'
    return normalized


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
    parser.add_argument('--lb_vq_ckpt', type=str, default=None,
                        help='冻结的 l_b VQ checkpoint；默认读取 lb_vq_prior.vq_checkpoint')
    parser.add_argument('--lb_diffusion_ckpt', type=str, default=None,
                        help='冻结的 l_b diffusion checkpoint；默认读取 lb_vq_prior.diffusion_checkpoint')
    parser.add_argument('--joint_train', action='store_true', 
                        help=(
                            'Stage3训练完成后自动联合训练；IMUHOI解冻'
                            'HumanPose + ObjectTrans并冻结VelocityContact，'
                            '其他架构保持VelocityContact + ObjectTrans'
                        ))
    parser.add_argument(
        '--object_trans_arch', '--object-trans-arch',
        dest='object_trans_arch',
        type=_normalize_object_trans_arch,
        choices=OBJECT_TRANS_ARCH_CHOICES,
        default=None,
        help=(
            'ObjectTrans架构；v2为固定VC权重，v3为不确定性感知的分层FK融合；'
            'legacy为四路无prior多head，IMUHOI为EgoIMU'
            '原始三路融合，IMUHOI_4heads为增加静止头的四路融合，'
            'legacy_mesh为ObsEncoder多head，'
            'mergeHeads_obs为共享head+ObsEncoder'
        ),
    )
    parser.add_argument(
        '--velocity_contact_arch', '--velocity-contact-arch',
        dest='velocity_contact_arch',
        type=str.lower,
        choices=VELOCITY_CONTACT_ARCH_CHOICES,
        default=None,
        help='VelocityContact架构；legacy不包含interaction boundary模块',
    )
    parser.add_argument(
        '--object_trans_state_feedback', '--object-trans-state-feedback',
        dest='object_trans_state_feedback',
        choices=['none', 'fused'],
        default=None,
        help='覆盖 OT 融合状态反馈开关；none=shared-head无反馈，fused=显式OT state feedback',
    )

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


def _joint_trainable_modules(object_trans_arch, joint_train):
    """Return (hp_trainable, vc_trainable) for the requested joint stage."""
    if not joint_train:
        return False, False
    if is_imuhoi_object_trans_arch(object_trans_arch):
        return True, False
    return False, True


class Stage3JointTrainer:
    """Stage3和联合训练器"""

    # Joint checkpoint selection deliberately follows the deployment target.
    # Values are the weighted loss components produced by the configured loss.
    JOINT_CHECKPOINT_SELECTION_KEYS = (
        "hoi_error_l",
        "hoi_error_r",
        "obj_trans",
    )
    # Added to the selection score only when the config gives them weight, so
    # the metric-aligned HOI terms steer checkpoint choice exactly when they
    # steer training.
    OPTIONAL_JOINT_SELECTION_KEYS = ("hoi_fused_l", "hoi_fused_r")

    def __init__(
        self,
        cfg,
        vc_model,
        hp_model,
        ot_model,
        train_loader,
        test_loader=None,
        joint_train=False,
        lb_prior_model=None,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.joint_train = joint_train
        self.imuhoi_arch = is_imuhoi_object_trans_arch(
            getattr(cfg, "object_trans_arch", "current")
        )
        self.joint_checkpoint_selection_keys = self.JOINT_CHECKPOINT_SELECTION_KEYS
        loss_weight_cfg = resolve_object_trans_loss_weights(cfg)
        optional_keys = tuple(
            key
            for key in self.OPTIONAL_JOINT_SELECTION_KEYS
            if float(loss_weight_cfg.get(key, 0.0) or 0.0) > 0.0
        )
        if optional_keys:
            self.joint_checkpoint_selection_keys = (
                self.JOINT_CHECKPOINT_SELECTION_KEYS + optional_keys
            )
        self.vc_model = vc_model.to(self.device)
        self.hp_model = hp_model.to(self.device)
        self.ot_model = ot_model.to(self.device)
        self.lb_prior_model = (
            lb_prior_model.to(self.device) if lb_prior_model is not None else None
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        resume_dir = getattr(cfg, "resume_dir", None)
        if joint_train and resume_dir:
            self.resume_path = os.path.join(resume_dir, "last_joint.pt")
            if not os.path.isfile(self.resume_path):
                raise FileNotFoundError(f"Joint resume checkpoint not found: {self.resume_path}")
        else:
            self.resume_path = resolve_resume_checkpoint(resume_dir)
        self.resume_checkpoint = None
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.n_iter = 0

        # The original IMUHOI joint stage fine-tunes HumanPose + ObjectTrans.
        # Other architectures retain the repository's existing VC + OT joint
        # training behavior.
        self.hp_trainable, self.vc_trainable = _joint_trainable_modules(
            getattr(cfg, "object_trans_arch", "current"),
            joint_train,
        )

        # 设置训练/冻结状态
        print("Stage3模式：训练ObjectTrans")
        for param in self.hp_model.parameters():
            param.requires_grad = self.hp_trainable
        for param in self.vc_model.parameters():
            param.requires_grad = self.vc_trainable
        for param in self.ot_model.parameters():
            param.requires_grad = True
        if self.lb_prior_model is not None:
            self.lb_prior_model.eval()
            for param in self.lb_prior_model.parameters():
                param.requires_grad = False
            print(
                "冻结 l_b VQ/Diffusion prior: "
                f"sample_steps={self.lb_prior_model.sample_steps}"
            )

        # 在线接触段 l_b 统计先验（diffusion prior 未启用时的默认来源）：
        # 训练时从 GT 序列的随机因果前缀模拟"已观测历史"的分位数统计。
        self.lb_stats_enabled = (
            self.lb_prior_model is None
            and bool(getattr(cfg, "object_trans_lb_stats", True))
            and str(getattr(cfg, "object_trans_arch", "current")).strip().lower() in {"current", "v2", "v3"}
        )
        self.lb_stats_min_frames = int(getattr(cfg, "lb_stats_min_frames", 10))
        self.lb_stats_noise_std = float(getattr(cfg, "lb_stats_noise_std", 0.02))
        self.lb_stats_drop_prob = float(getattr(cfg, "lb_stats_drop_prob", 0.3))
        if self.lb_stats_enabled:
            print(
                "启用在线 l_b 统计先验模拟: "
                f"min_frames={self.lb_stats_min_frames}, "
                f"noise_std={self.lb_stats_noise_std}, "
                f"drop_prob={self.lb_stats_drop_prob}"
            )
        if joint_train:
            print(
                "联合训练checkpoint选择（按配置权重）: "
                + " + ".join(self.joint_checkpoint_selection_keys)
            )
            if self.imuhoi_arch:
                print(
                    "IMUHOI联合训练模式：冻结VelocityContact，"
                    "从epoch 0开始微调HumanPose + ObjectTrans"
                )
            else:
                print(
                    "联合训练模式：冻结HumanPose，"
                    "从epoch 0开始微调VelocityContact + ObjectTrans"
                )

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
        if self.vc_trainable:
            param_groups.append(
                {
                    "params": [p for p in self.vc_model.parameters() if p.requires_grad],
                    "lr": base_lr,
                    "name": "velocity_contact",
                }
            )
        if self.hp_trainable:
            param_groups.append(
                {
                    "params": [
                        p for p in self.hp_model.parameters()
                        if p.requires_grad
                    ],
                    "lr": base_lr,
                    "name": "human_pose",
                }
            )

        # foreach: same update, multi-tensor kernels.  See create_optimizer
        # in train/rnn/train_utils.py for the measurement.
        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=cfg.weight_decay, foreach=True
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=cfg.milestones, gamma=cfg.gamma
        )
        self.amp_enabled, self.amp_dtype = resolve_amp_settings(cfg, self.device)
        print(f"AMP: enabled={self.amp_enabled}, dtype={self.amp_dtype}")
        self.gradient_clip = float(getattr(cfg, "gradient_clip", 1.0))
        if self.resume_checkpoint is not None:
            self._restore_training_state(self.resume_checkpoint)

        ot_trainable = sum(p.numel() for p in self.ot_model.parameters() if p.requires_grad)
        vc_params = sum(p.numel() for p in self.vc_model.parameters())
        print(f"ObjectTrans可训练参数数量: {ot_trainable}, lr={base_lr}")
        if joint_train:
            if self.hp_trainable:
                hp_params = sum(
                    p.numel()
                    for p in self.hp_model.parameters()
                    if p.requires_grad
                )
                print(f"HumanPose可训练参数数量: {hp_params}, lr={base_lr}")
                print("VelocityContact冻结")
            else:
                print(f"VelocityContact可训练参数数量: {vc_params}, lr={base_lr}")

        # 损失函数。VC/HP 用共享权重；OT 额外套上本架构的权重块。
        loss_weights = getattr(cfg, 'loss_weights', {})
        ot_loss_weights = resolve_object_trans_loss_weights(cfg)
        if self.imuhoi_arch:
            imuhoi_weights = getattr(
                cfg,
                "imuhoi_loss_weights",
                IMUHOIObjectTransLoss.DEFAULT_WEIGHTS,
            )
            self.vc_loss_fn = None
            self.hp_loss_fn = (
                HumanPoseLoss(
                    weights={
                        **dict(loss_weights),
                        # EgoIMU IMUHOI does not supervise the auxiliary
                        # HumanPose foot-contact head in its joint stage.
                        "contact": 0.0,
                    },
                    no_trans=bool(getattr(cfg, "no_trans", False)),
                )
                if self.hp_trainable
                else None
            )
            self.ot_loss_fn = IMUHOIObjectTransLoss(
                weights=imuhoi_weights
            )
        else:
            if joint_train:
                self.vc_loss_fn = (
                    V2VelocityContactLoss(weights=loss_weights)
                    if str(getattr(cfg, "velocity_contact_arch", "current"))
                    .strip()
                    .lower()
                    == "v2"
                    else VelocityContactLoss(
                        weights=loss_weights,
                        anticipation_horizon=getattr(
                            cfg, "v4_anticipation_horizon", None
                        ),
                    )
                )
            else:
                self.vc_loss_fn = None
            self.hp_loss_fn = None
            self.ot_loss_fn = ObjectTransLoss(weights=ot_loss_weights)

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
        if checkpoint.get("human_pose_state_dict") is not None:
            load_model_state_from_checkpoint(
                self.hp_model,
                checkpoint,
                strict=False,
                state_key="human_pose_state_dict",
                name="HumanPose",
            )
            print(f"恢复HumanPose权重: {self.resume_path}")

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
        if self.joint_train:
            saved_keys = checkpoint.get("checkpoint_selection_keys")
            expected_keys = list(self.joint_checkpoint_selection_keys)
            if saved_keys != expected_keys:
                print(
                    "警告：resume joint checkpoint 的选择标准与当前配置不一致；"
                    "保留训练状态，但重置 best_loss 并按新标准重新选优。"
                )
                return float("inf")
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

    @classmethod
    def _joint_checkpoint_selection_loss(cls, weighted_losses):
        """Return the legacy weighted score used by current/v2 training."""
        missing = [
            key
            for key in cls.JOINT_CHECKPOINT_SELECTION_KEYS
            if key not in weighted_losses
        ]
        if missing:
            raise KeyError(
                "Joint checkpoint selection is missing loss components: "
                + ", ".join(missing)
            )
        return torch.stack(
            [
                weighted_losses[key]
                for key in cls.JOINT_CHECKPOINT_SELECTION_KEYS
            ]
        ).sum()

    def _configured_joint_checkpoint_selection_loss(self, weighted_losses):
        """Return the deployment score configured for this OT architecture."""
        missing = [
            key
            for key in self.joint_checkpoint_selection_keys
            if key not in weighted_losses
        ]
        if missing:
            raise KeyError(
                "Joint checkpoint selection is missing loss components: "
                + ", ".join(missing)
            )
        return torch.stack(
            [
                weighted_losses[key]
                for key in self.joint_checkpoint_selection_keys
            ]
        ).sum()

    def _build_rest_state_perturbation(self, batch, data_dict, training):
        """Observable mid-window state corruption for the state-trust confidence.

        With probability ``object_trans_rest_perturb_prob`` a window gets its
        ObjectTrans feedback state shifted by N(0, ``object_trans_rest_perturb_std``)
        at one random frame where the GT object is at rest (|v| below
        ``object_trans_rest_perturb_vel`` m/s).  The shift persists through the
        state until the model corrects it, which is exactly the long-horizon
        failure (a drifted resting object met by a hand), and it is visible to
        the consistency features, so the confidence head gets informative
        negatives beyond the window-init corruption.  Returns a
        ``feedback_control`` dict (mode "perturb") or None.
        """
        from configs import FRAME_RATE

        std = float(getattr(self.cfg, "object_trans_rest_perturb_std", 0.0) or 0.0)
        prob = float(getattr(self.cfg, "object_trans_rest_perturb_prob", 0.5))
        vel_thr = float(getattr(self.cfg, "object_trans_rest_perturb_vel", 0.02))
        if not training or std <= 0.0 or prob <= 0.0 or batch is None:
            return None
        obj_trans = batch.get("obj_trans")
        if not isinstance(obj_trans, torch.Tensor) or obj_trans.dim() != 3 or obj_trans.shape[1] < 12:
            return None
        obj_trans = obj_trans.to(self.device)
        batch_size, seq_len = obj_trans.shape[:2]
        speed = (obj_trans[:, 1:] - obj_trans[:, :-1]).norm(dim=-1) * FRAME_RATE  # [B,T-1]
        rest = torch.cat((speed[:, :1], speed), dim=1) < vel_thr                   # [B,T]
        rest[:, :5] = False
        rest[:, -5:] = False
        has_object = data_dict.get("has_object")
        if isinstance(has_object, torch.Tensor):
            rest = rest & has_object.to(rest.device).view(batch_size, 1).bool()
        select = (torch.rand(batch_size, device=obj_trans.device) < prob) & rest.any(dim=1)
        if not bool(select.any()):
            return None
        # one random rest frame per selected window
        scores = torch.rand(batch_size, seq_len, device=obj_trans.device) * rest.float()
        inject = scores.argmax(dim=1)
        injection_mask = torch.zeros(batch_size, seq_len, device=obj_trans.device, dtype=torch.bool)
        injection_mask[torch.arange(batch_size, device=obj_trans.device), inject] = select
        perturbation = torch.zeros(batch_size, seq_len, 3, device=obj_trans.device, dtype=obj_trans.dtype)
        perturbation[injection_mask] = torch.randn(int(injection_mask.sum()), 3, device=obj_trans.device, dtype=obj_trans.dtype) * std
        return {"mode": "perturb", "perturbation": perturbation, "injection_mask": injection_mask}

    def _perturb_object_init(self, obj_trans_init, contact_init, has_object):
        """Noisy object init for windows that start inside a contact segment.

        Training windows start from the ground-truth object position, so a
        window that begins mid-segment gets an exact hand->object bone for
        free and the head never learns to distrust it.  Online inference hands
        later windows its own published position instead, whose error grows
        with the accumulated hand drift.  With probability
        ``object_trans_init_noise_prob`` the init of such a window is offset by
        Gaussian noise (``object_trans_init_noise_std`` metres).  Windows that
        start without contact are left alone: a wrong static hold before the
        first contact cannot be corrected by any branch and would only teach
        the motion heads to hallucinate.
        """
        std = float(getattr(self.cfg, "object_trans_init_noise_std", 0.0) or 0.0)
        prob = float(getattr(self.cfg, "object_trans_init_noise_prob", 0.5))
        if std <= 0.0 or prob <= 0.0 or not isinstance(obj_trans_init, torch.Tensor):
            return obj_trans_init
        batch_size = obj_trans_init.shape[0]
        select = torch.rand(batch_size, device=obj_trans_init.device) < prob
        if isinstance(contact_init, torch.Tensor) and contact_init.shape[0] == batch_size:
            select = select & (contact_init[..., :2] > 0.5).any(dim=-1)
        if isinstance(has_object, torch.Tensor) and has_object.shape[0] == batch_size:
            select = select & has_object.to(device=select.device, dtype=torch.bool)
        noise = torch.randn_like(obj_trans_init) * std
        return obj_trans_init + noise * select.to(obj_trans_init.dtype).unsqueeze(-1)

    def forward_all(self, data_dict, batch=None, epoch: int = 0, training: bool = True):
        """完整的前向传播"""
        if self.hp_trainable and training:
            self.hp_model.train()
        else:
            self.hp_model.eval()
        if self.vc_trainable and training:
            self.vc_model.train()
        else:
            self.vc_model.eval()

        with torch.set_grad_enabled(self.hp_trainable and training):
            hp_out = call_model_inference(self.hp_model, data_dict, inference_mode="offline")

        gt_hp = None
        mix_mask = None
        if training and batch is not None and not self.imuhoi_arch:
            pred_prob = prediction_mix_probability(epoch, self.cfg)
            gt_hp = build_gt_human_pose_outputs(batch, self.device, dtype=hp_out['pred_hand_glb_pos'].dtype)
            # 整条 VC->OT 输入链共用同一个 per-sample mask：否则会出现
            # "VC 吃预测 HP 算出的 contact + OT 吃 GT 手位置" 这类部署时
            # 不存在的自相矛盾组合。
            hp_for_vc, mix_mask = sample_mix_dict(
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
                return_mask=True,
            )
        else:
            hp_for_vc = hp_out

        vc_grad_enabled = self.vc_trainable and training
        with torch.set_grad_enabled(vc_grad_enabled):
            vc_out = call_model_inference(self.vc_model, data_dict, hp_out=hp_for_vc, inference_mode="offline")

        if training and batch is not None and not self.imuhoi_arch:
            pred_prob = prediction_mix_probability(epoch, self.cfg)
            if gt_hp is None:
                gt_hp = build_gt_human_pose_outputs(batch, self.device, dtype=hp_out['pred_hand_glb_pos'].dtype)
            gt_vc = build_gt_velocity_contact_outputs(batch, self.device, dtype=vc_out['pred_obj_vel'].dtype)
            if mix_mask is None:
                gt_hand_positions = gt_hp["pred_palm_glb_pos"]
                pred_hand_positions = hp_out["pred_palm_glb_pos"]
                hand_positions, mix_mask = sample_mix_tensor(
                    gt_hand_positions,
                    pred_hand_positions,
                    pred_prob,
                    return_mask=True,
                )
            else:
                gt_hand_positions = gt_hp["pred_palm_glb_pos"]
                pred_hand_positions = hp_out["pred_palm_glb_pos"]
                hand_positions = sample_mix_tensor(
                    gt_hand_positions,
                    pred_hand_positions,
                    pred_prob,
                    mask=mix_mask,
                )
            contact_prob = sample_mix_tensor(
                gt_vc['pred_hand_contact_prob'],
                vc_out['pred_hand_contact_prob'],
                pred_prob,
                mask=mix_mask,
            )
            obj_vel_input = sample_mix_tensor(
                gt_vc['pred_obj_vel'],
                vc_out['pred_obj_vel'],
                pred_prob,
                mask=mix_mask,
            )
            human_pose_input = sample_mix_tensor(
                gt_hp['p_pred'],
                hp_out['p_pred'],
                pred_prob,
                mask=mix_mask,
            )
            root_trans_input = sample_mix_tensor(
                gt_hp['root_trans_pred'],
                hp_out['root_trans_pred'],
                pred_prob,
                mask=mix_mask,
            )
        else:
            hand_positions = hp_out["pred_palm_glb_pos"]
            contact_prob = vc_out['pred_hand_contact_prob']
            obj_vel_input = vc_out['pred_obj_vel']
            human_pose_input = hp_out.get('p_pred')
            root_trans_input = hp_out.get('root_trans_pred')

        lb_prior_quantiles = None
        if self.lb_prior_model is not None:
            self.lb_prior_model.eval()
            with torch.no_grad():
                lb_prior_quantiles = self.lb_prior_model(
                    hp_out,
                    vc_out,
                    data_dict['obj_imu'],
                )
        elif self.lb_stats_enabled and training and batch is not None:
            # 用 GT 序列的随机因果前缀模拟在线接触段统计（含噪声/丢弃增广）。
            # 评估分支保持 None：offline 单趟推理没有历史，等价于冷启动，
            # 训练中的 drop/短前缀样本已覆盖这一分布。
            lb_prior_quantiles = sample_gt_lb_prior_stats(
                batch,
                self.device,
                hp_out['pred_hand_glb_pos'].dtype,
                min_contact_frames=self.lb_stats_min_frames,
                noise_std=self.lb_stats_noise_std,
                drop_prob=self.lb_stats_drop_prob,
            )

        obj_trans_init = data_dict['obj_trans_init']
        feedback_control = None
        if training and batch is not None:
            obj_trans_init = self._perturb_object_init(
                obj_trans_init, data_dict.get('contact_init'), data_dict.get('has_object')
            )
            feedback_control = self._build_rest_state_perturbation(batch, data_dict, training)

        ot_kwargs = dict(
            obj_imu=data_dict['obj_imu'],
            human_imu=data_dict['human_imu'],
            obj_vel_input=obj_vel_input,
            contact_init=data_dict['contact_init'],
            has_object_mask=data_dict['has_object'],
            human_pose_input=human_pose_input,
            root_trans_input=root_trans_input,
            lb_prior_quantiles=lb_prior_quantiles,
            enable_refine=False if self.imuhoi_arch else getattr(self.cfg, "enable_ot_refine", False),
        )
        if feedback_control is not None:
            ot_kwargs['feedback_control'] = feedback_control
        history_window = int(getattr(self.cfg, 'ot_history_training_window', 0)) if training else 0
        if history_window:
            # Regime mixing: the history path never supervises the cold-start
            # window (GT init, no published prefix) that the online protocol
            # also runs, and its LSTM init network gets ~no gradient there.
            # Use it only on a fraction of the batches (ot_history_training_prob)
            # and only after ot_history_training_start_epoch, so the ordinary
            # GT-init regime keeps training alongside the own-rollout regime.
            history_prob = float(getattr(self.cfg, 'ot_history_training_prob', 1.0))
            history_start = int(getattr(self.cfg, 'ot_history_training_start_epoch', 0))
            if int(epoch) < history_start or torch.rand(()).item() >= history_prob:
                history_window = 0
        if history_window:
            ot_out = forward_with_ot_history(self.ot_model, hand_positions, contact_prob, obj_trans_init, history_window, **ot_kwargs)
        else:
            ot_out = call_model_inference(
                self.ot_model,
            hand_positions,
            contact_prob,
            obj_trans_init,
                inference_mode="offline", **ot_kwargs,
            )

        results = {}
        results.update(vc_out)
        results.update(hp_out)
        results.update(ot_out)

        return results, vc_out, hp_out, ot_out

    def train_epoch(self, epoch):
            """训练一个epoch"""
            self.vc_model.train() if self.vc_trainable else self.vc_model.eval()
            self.hp_model.train() if self.hp_trainable else self.hp_model.eval()
            self.ot_model.train()

            total_loss = 0
            loss_components = {}

            train_iter = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)

            for batch in train_iter:
                data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    results, vc_out, hp_out, ot_out = self.forward_all(
                        data_dict, batch=batch, epoch=epoch, training=True
                    )

                    losses = {}
                    ot_losses = {}
                    vc_losses = {}
                    hp_losses = {}

                    if self.joint_train and self.vc_trainable and self.vc_loss_fn:
                        vc_loss, vc_losses, vc_weighted_losses = self.vc_loss_fn(vc_out, batch, self.device)
                        losses.update({f'vc_{k}': v for k, v in vc_weighted_losses.items()})
                    else:
                        vc_loss = torch.tensor(0.0, device=self.device)
                        vc_weighted_losses = {}

                    if self.hp_trainable and self.hp_loss_fn:
                        hp_loss, hp_losses, hp_weighted_losses = self.hp_loss_fn(
                            hp_out,
                            batch,
                            self.device,
                        )
                        losses.update(
                            {
                                f"hp_{key}": value
                                for key, value in hp_weighted_losses.items()
                            }
                        )
                    else:
                        hp_loss = torch.tensor(0.0, device=self.device)
                        hp_weighted_losses = {}

                    ot_loss, ot_losses, ot_weighted_losses = self.ot_loss_fn(
                        ot_out, batch, self.device
                    )
                    losses.update({f'ot_{k}': v for k, v in ot_weighted_losses.items()})
                    batch_loss = hp_loss + vc_loss + ot_loss

                batch_loss.backward()
                if self.gradient_clip > 0:
                    clip_grad_norm_(
                        [
                            param
                            for group in self.optimizer.param_groups
                            for param in group["params"]
                        ],
                        max_norm=self.gradient_clip,
                    )
                self.optimizer.step()

                total_loss += batch_loss.item()
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[key] = loss_components.get(key, 0) + value.item()

                # 更新进度条
                postfix = {'loss': batch_loss.item(), 'ot': ot_loss.item()}
                if self.joint_train and self.vc_trainable:
                    postfix['vc'] = vc_loss.item()
                if self.hp_trainable:
                    postfix['hp'] = hp_loss.item()
                train_iter.set_postfix(postfix)

                if self.writer is not None:
                    self.writer.add_scalar('train/total_loss', batch_loss.item(), self.n_iter)
                    self.writer.add_scalar('train/ot_loss', ot_loss.item(), self.n_iter)
                    if self.joint_train and self.vc_trainable:
                        self.writer.add_scalar('train/vc_loss', vc_loss.item(), self.n_iter)
                    if self.hp_trainable:
                        self.writer.add_scalar(
                            'train/hp_loss', hp_loss.item(), self.n_iter
                        )
                    for key, value in ot_weighted_losses.items():
                        if isinstance(value, torch.Tensor):
                            self.writer.add_scalar(f'train/ot/{key}', value.item(), self.n_iter)
                    if self.joint_train and self.vc_trainable:
                        for key, value in vc_weighted_losses.items():
                            if isinstance(value, torch.Tensor):
                                self.writer.add_scalar(f'train/vc/{key}', value.item(), self.n_iter)
                    if self.hp_trainable:
                        for key, value in hp_weighted_losses.items():
                            if isinstance(value, torch.Tensor):
                                self.writer.add_scalar(
                                    f'train/hp/{key}',
                                    value.item(),
                                    self.n_iter,
                                )

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
                data_dict = build_model_input_dict(
                    batch, self.cfg, self.device,
                    add_noise=bool(getattr(self.cfg, "validation_add_imu_noise", False)),
                )
                with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    results, vc_out, hp_out, ot_out = self.forward_all(
                        data_dict, batch=batch, epoch=epoch, training=False
                    )

                    vc_loss = torch.tensor(0.0, device=self.device)
                    vc_losses = {}
                    if self.joint_train and self.vc_trainable and self.vc_loss_fn:
                        vc_loss, vc_losses, _ = self.vc_loss_fn(vc_out, batch, self.device)

                    hp_loss = torch.tensor(0.0, device=self.device)
                    hp_losses = {}
                    if self.hp_trainable and self.hp_loss_fn:
                        _, _, hp_weighted = self.hp_loss_fn(
                            hp_out,
                            batch,
                            self.device,
                        )
                        hp_losses = {
                            key: value
                            for key, value in hp_weighted.items()
                            if key in self.hp_loss_fn.TEST_LOSS_KEYS
                        }
                        if hp_losses:
                            hp_loss = sum(hp_losses.values())

                    if self.joint_train:
                        _, _, ot_weighted_losses = self.ot_loss_fn.compute_loss(
                            ot_out, batch, self.device
                        )
                        report_keys = set(self.ot_loss_fn.TEST_LOSS_KEYS).union(
                            self.joint_checkpoint_selection_keys
                        )
                        ot_losses = {
                            key: value
                            for key, value in ot_weighted_losses.items()
                            if key in report_keys
                        }
                        combined_loss = self._configured_joint_checkpoint_selection_loss(
                            ot_weighted_losses
                        )
                    else:
                        ot_loss, ot_losses = self.ot_loss_fn.compute_test_loss(
                            ot_out, batch, self.device
                        )
                        combined_loss = ot_loss + vc_loss + hp_loss
                total_loss += combined_loss.item()
                for key, value in ot_losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[key] = loss_components.get(key, 0) + value.item()
                for key, value in vc_losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[f"vc_{key}"] = loss_components.get(f"vc_{key}", 0) + value.item()
                for key, value in hp_losses.items():
                    if isinstance(value, torch.Tensor):
                        loss_components[f"hp_{key}"] = (
                            loss_components.get(f"hp_{key}", 0)
                            + value.item()
                        )

        total_loss /= len(self.test_loader)
        for key in loss_components:
            loss_components[key] /= len(self.test_loader)
        if self.joint_train:
            loss_components["checkpoint_selection_total"] = total_loss

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

        # Evaluation cadence follows the config (``eval_interval``; the joint
        # phase passes ``joint_train_eval_interval``).  Early stopping: stop when
        # the checkpoint-selection score on the test split has not improved for
        # ``early_stopping_patience_epochs`` epochs (0 = off), never before
        # ``early_stopping_min_epochs``.
        eval_interval = max(1, int(getattr(self.cfg, "eval_interval", 10) or 10))
        patience_epochs = int(getattr(self.cfg, "early_stopping_patience_epochs", 0) or 0)
        min_epochs = int(getattr(self.cfg, "early_stopping_min_epochs", 0) or 0)
        last_improvement_epoch = _resumed_last_improvement_epoch(
            self.cfg.save_dir, self.start_epoch, "best_joint.pt" if self.joint_train else "best.pt"
        )
        stopped_early_at = None
        for epoch in range(self.start_epoch, max_epoch):
            train_loss, train_components = self.train_epoch(epoch)

            print(f'\rEpoch {epoch}, Train Loss: {train_loss:.4f}', end='')

            if epoch % eval_interval == 0 and self.test_loader is not None:
                test_loss, test_components = self.evaluate(epoch)

                if test_loss is not None:
                    print(f', Test Loss: {test_loss:.4f}')

                    if test_loss < self.best_loss:
                        self.best_loss = test_loss
                        last_improvement_epoch = epoch
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
                            if key in self.ot_loss_fn.TEST_LOSS_KEYS:
                                self.writer.add_scalar(f'test/ot/{key}', value, self.n_iter)
                            elif key.startswith('vc_'):
                                self.writer.add_scalar(f'test/vc/{key[3:]}', value, self.n_iter)
                            elif key.startswith('hp_'):
                                self.writer.add_scalar(f'test/hp/{key[3:]}', value, self.n_iter)
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
            if self.should_stop_early(epoch, last_improvement_epoch, patience_epochs, min_epochs, self.test_loader is not None):
                stopped_early_at = epoch
                print(
                    f"Early stopping at epoch {epoch}: no test-metric improvement since epoch "
                    f"{last_improvement_epoch} (patience {patience_epochs} epochs, best {self.best_loss:.4f})"
                )
                break

        # 保存最终模型
        self._save_models(
            max_epoch - 1 if stopped_early_at is None else stopped_early_at,
            train_loss,
            'final',
            {
                'best_loss': self.best_loss,
                'n_iter': self.n_iter,
                'stopped_early_at': stopped_early_at,
            },
        )

        if self.writer is not None:
            self.writer.close()

    @staticmethod
    def should_stop_early(epoch, last_improvement_epoch, patience_epochs, min_epochs, has_test_loader=True):
        """Early-stopping rule shared by every stage (implementation in train_utils)."""
        return _should_stop_early(epoch, last_improvement_epoch, patience_epochs, min_epochs, has_test_loader)

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
            'rng_state': _capture_rng_state(),
        }
        if self.joint_train:
            if self.hp_trainable:
                checkpoint['human_pose_state_dict'] = get_model_state_dict(
                    self.hp_model
                )
            if self.vc_trainable:
                checkpoint['velocity_contact_state_dict'] = (
                    get_model_state_dict(self.vc_model)
                )
        if extra:
            checkpoint.update(extra)
        return checkpoint

    def _save_module_checkpoint(self, path, state_dict, epoch, loss, extra=None):
        checkpoint = {
            'epoch': epoch,
            'module_state_dict': state_dict,
            'model_state_dict': state_dict,
            'loss': loss,
            'best_loss': self.best_loss,
            'n_iter': self.n_iter,
        }
        if extra:
            checkpoint.update(extra)
        self._write_checkpoint(path, checkpoint)

    def _save_models(self, epoch, loss, prefix, extra=None):
        """保存模型"""
        extra = dict(extra or {})
        if self.joint_train:
            extra.update(
                {
                    'checkpoint_selection_keys': list(
                        self.joint_checkpoint_selection_keys
                    ),
                    'checkpoint_selection_uses_weighted_losses': True,
                }
            )
        checkpoint = self._build_joint_checkpoint(epoch, loss, extra=extra)

        if self.joint_train:
            self._write_checkpoint(
                os.path.join(self.cfg.save_dir, f'{prefix}_joint.pt'),
                checkpoint,
            )
            # joint阶段与OT阶段共用目录，但不覆盖OT的best.pt/last.pt/final.pt。
            ot_path = os.path.join(self.cfg.save_dir, f'{prefix}_object_trans.pt')
            self._save_module_checkpoint(
                ot_path,
                checkpoint['object_trans_state_dict'],
                epoch,
                loss,
                extra=extra,
            )
            if self.hp_trainable:
                hp_path = os.path.join(
                    self.cfg.save_dir,
                    f'{prefix}_human_pose.pt',
                )
                self._save_module_checkpoint(
                    hp_path,
                    checkpoint['human_pose_state_dict'],
                    epoch,
                    loss,
                    extra=extra,
                )
            if self.vc_trainable:
                vc_path = os.path.join(
                    self.cfg.save_dir,
                    f'{prefix}_velocity_contact.pt',
                )
                self._save_module_checkpoint(
                    vc_path,
                    checkpoint['velocity_contact_state_dict'],
                    epoch,
                    loss,
                    extra=extra,
                )
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
    # Early stopping is phase-specific: the OT phase decays the lr only at
    # epoch 100 (milestones), so its patience window must reach past the first
    # milestone; the joint phase decays every 10 epochs.
    if getattr(cfg, "joint_train_early_stopping_patience_epochs", None) is not None:
        joint_cfg.early_stopping_patience_epochs = int(cfg.joint_train_early_stopping_patience_epochs)
    if getattr(cfg, "joint_train_early_stopping_min_epochs", None) is not None:
        joint_cfg.early_stopping_min_epochs = int(cfg.joint_train_early_stopping_min_epochs)
    joint_cfg.eval_interval = max(
        1, int(getattr(cfg, "joint_train_eval_interval", cfg.eval_interval))
    )
    joint_cfg.lr = float(base_lr) * 0.1
    joint_amp_enabled = getattr(cfg, "joint_train_amp_enabled", None)
    if joint_amp_enabled is not None:
        joint_cfg.amp_enabled = bool(joint_amp_enabled)
    joint_cfg.resume_dir = (
        cfg.save_dir
        if getattr(cfg, "resume_dir", None)
        and os.path.isfile(os.path.join(cfg.save_dir, "last_joint.pt"))
        else None
    )

    best_ot_path = os.path.join(cfg.save_dir, "best.pt")
    _load_object_trans_best_for_joint(trainer.ot_model, best_ot_path, trainer.device)
    save_config_snapshot(
        joint_cfg,
        extra={
            "joint_train_auto_started_from": best_ot_path,
            "joint_train_lr_source": "lr * 0.1",
            "joint_checkpoint_selection_keys": list(
                trainer.joint_checkpoint_selection_keys
            ),
            "joint_checkpoint_selection_uses_weighted_losses": True,
        },
        filename="joint_train_config.yaml",
    )

    print("=" * 50)
    if is_imuhoi_object_trans_arch(
        getattr(cfg, "object_trans_arch", "")
    ):
        print("自动联合训练: HumanPose + IMUHOI ObjectTrans")
    else:
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
        lb_prior_model=trainer.lb_prior_model,
    )
    joint_trainer.train()
    return joint_trainer


def main():
    """主函数"""
    args = get_args()
    cfg = merge_config(args)
    cfg.velocity_contact_arch = normalize_velocity_contact_arch(
        args.velocity_contact_arch
        if args.velocity_contact_arch is not None
        else getattr(cfg, "velocity_contact_arch", "current")
    )
    if args.object_trans_state_feedback is not None:
        cfg.object_trans_state_feedback = args.object_trans_state_feedback == 'fused'
    elif not hasattr(cfg, 'object_trans_state_feedback'):
        cfg.object_trans_state_feedback = False
    if args.object_trans_arch is not None:
        cfg.object_trans_arch = args.object_trans_arch
    elif not hasattr(cfg, 'object_trans_arch'):
        cfg.object_trans_arch = 'current'
    normalized_ot_arch = str(cfg.object_trans_arch).strip().lower()
    if normalized_ot_arch == 'mergeheads_obs':
        cfg.object_trans_arch = 'mergeHeads_obs'
    elif normalized_ot_arch == 'imuhoi':
        cfg.object_trans_arch = 'imuhoi'
    elif normalized_ot_arch == 'imuhoi_4heads':
        cfg.object_trans_arch = 'imuhoi_4heads'
    if cfg.object_trans_arch == 'mergeHeads_obs':
        # Persist the architecture's resolved fixed dimensions so the run
        # snapshot describes the actual checkpoint shapes.
        cfg.interaction_code_dim = MergeHeadsObsObjectTransModule.INTERACTION_CODE_DIM
        cfg.object_trans_hidden_dim = MergeHeadsObsObjectTransModule.PREDICTOR_HIDDEN_DIM
    auto_joint_train = bool(args.joint_train) or bool(getattr(cfg, "joint_train", False))
    base_train_lr = float(cfg.lr)
    cfg.joint_train = auto_joint_train

    prior_cfg = getattr(cfg, 'lb_vq_prior', None)
    enable_lb_prior = bool(getattr(cfg, 'enable_ot_lb_prior', False))
    if cfg.object_trans_arch == 'mergeHeads_obs' and enable_lb_prior:
        print("mergeHeads_obs使用内置ObsEncoder；已关闭外部l_b VQ/Diffusion prior。")
        enable_lb_prior = False
        cfg.enable_ot_lb_prior = False
    if enable_lb_prior and cfg.object_trans_arch not in {'current', 'v2', 'v3'}:
        raise ValueError("l_b VQ/Diffusion prior supports object_trans_arch=current, v2, or v3 only")
    if enable_lb_prior:
        if prior_cfg is None:
            raise ValueError("enable_ot_lb_prior=true requires an lb_vq_prior config block")
        if args.lb_vq_ckpt:
            prior_cfg.vq_checkpoint = args.lb_vq_ckpt
        if args.lb_diffusion_ckpt:
            prior_cfg.diffusion_checkpoint = args.lb_diffusion_ckpt

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)

    if cfg.object_trans_arch == 'legacy':
        module_name = 'object_trans_legacy'
    elif cfg.object_trans_arch == 'v2':
        module_name = 'object_trans_v2'
    elif cfg.object_trans_arch == 'v3':
        module_name = 'object_trans_v3'
    elif cfg.object_trans_arch == 'imuhoi':
        module_name = 'object_trans_IMUHOI'
    elif cfg.object_trans_arch == 'imuhoi_4heads':
        module_name = 'object_trans_IMUHOI_4heads'
    elif cfg.object_trans_arch == 'legacy_mesh':
        module_name = 'object_trans_legacy_mesh'
    elif cfg.object_trans_arch == 'mergeHeads_obs':
        module_name = 'object_trans_mergeHeads_obs'
    else:
        module_name = 'object_trans'
    if cfg.velocity_contact_arch == "legacy":
        module_name = f"{module_name}_vc_legacy"
    elif cfg.velocity_contact_arch in {"v2", "v3", "v4"}:
        module_name = f"{module_name}_vc_{cfg.velocity_contact_arch}"
    elif cfg.velocity_contact_arch == "imuhoi":
        module_name = f"{module_name}_vc_IMUHOI"
    elif cfg.velocity_contact_arch.startswith("imuhoi_cond"):
        module_name = f"{module_name}_vc_{cfg.velocity_contact_arch}"
    save_dir = create_save_dir(cfg, module_name)
    save_config_snapshot(cfg)

    mode_str = "Stage 3 + 自动联合训练" if auto_joint_train else "Stage 3"
    trans_str = "noTrans" if cfg.no_trans else "普通"

    print("=" * 50)
    print(f"{mode_str}: ObjectTransModule训练 ({trans_str}模式)")
    print(f"ObjectTrans架构: {cfg.object_trans_arch}")
    print(f"设备: {cfg.device}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"训练轮数: {cfg.epochs}")
    print(f"noTrans模式: {cfg.no_trans}")
    print(f"VelocityContact架构: {cfg.velocity_contact_arch}")
    print(
        "OT state feedback: "
        f"{'enabled' if bool(getattr(cfg, 'object_trans_state_feedback', False)) else 'disabled'}"
    )
    print(f"保存目录: {save_dir}")
    print("=" * 50)

    # 查找/加载预训练权重
    vc_ckpt, vc_source = _resolve_pretrained_ckpt(cfg, args.vc_ckpt, 'velocity_contact', 'vc_ckpt')
    hp_ckpt, hp_source = _resolve_pretrained_ckpt(cfg, args.hp_ckpt, 'human_pose', 'hp_ckpt')
    ot_ckpt, ot_source = _resolve_pretrained_ckpt(cfg, args.ot_ckpt, 'object_trans', 'ot_ckpt')

    base_save_dir = os.path.dirname(save_dir)
    if vc_ckpt is None:
        vc_search_name = (
            "velocity_contact_IMUHOI"
            if cfg.velocity_contact_arch == "imuhoi"
            else "velocity_contact"
        )
        vc_ckpt = find_latest_checkpoint(base_save_dir, vc_search_name)
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
    prime_dataloader_workers(train_loader, test_loader)

    if train_loader is None or len(train_loader) == 0:
        print("错误: 无法创建训练数据加载器")
        return

    # 创建模型
    device = torch.device(cfg.device)

    vc_model = build_velocity_contact_module(cfg)
    if vc_ckpt and os.path.exists(vc_ckpt):
        load_checkpoint(vc_model, vc_ckpt, device, strict=False)
    elif vc_ckpt:
        print(f"警告: VelocityContact权重不存在: {vc_ckpt}")

    hp_model = HumanPoseModule(cfg, device, no_trans=cfg.no_trans)
    if hp_ckpt and os.path.exists(hp_ckpt):
        load_checkpoint(hp_model, hp_ckpt, device)
    elif hp_ckpt:
        print(f"警告: HumanPose权重不存在: {hp_ckpt}")

    if cfg.object_trans_arch == 'legacy':
        ot_model_cls = LegacyObjectTransModule
    elif cfg.object_trans_arch == 'imuhoi':
        ot_model_cls = IMUHOIObjectTransModule
    elif cfg.object_trans_arch == 'imuhoi_4heads':
        ot_model_cls = IMUHOI4HeadsObjectTransModule
    elif cfg.object_trans_arch == 'legacy_mesh':
        ot_model_cls = LegacyMeshObjectTransModule
    elif cfg.object_trans_arch == 'mergeHeads_obs':
        ot_model_cls = MergeHeadsObsObjectTransModule
    elif cfg.object_trans_arch == 'v2':
        ot_model_cls = V2ObjectTransModule
    elif cfg.object_trans_arch == 'v3':
        ot_model_cls = V3ObjectTransModule
    else:
        ot_model_cls = ObjectTransModule
    ot_model = ot_model_cls(cfg)
    lb_prior_model = FrozenLBPriorPredictor(cfg) if enable_lb_prior else None
    if getattr(cfg, "resume_dir", None):
        print("检测到 --resume_dir，ObjectTrans权重将由训练器从断点恢复。")

    print(f"VelocityContact参数量: {sum(p.numel() for p in vc_model.parameters())}")
    print(f"HumanPose参数量: {sum(p.numel() for p in hp_model.parameters())}")
    print(f"ObjectTrans参数量: {sum(p.numel() for p in ot_model.parameters())}")
    if lb_prior_model is not None:
        print(
            "冻结 l_b prior 参数量: "
            f"{sum(p.numel() for p in lb_prior_model.parameters())}"
        )

    # 创建训练器
    trainer = Stage3JointTrainer(
        cfg, vc_model, hp_model, ot_model,
        train_loader, test_loader,
        joint_train=False,
        lb_prior_model=lb_prior_model,
    )

    # 开始训练
    trainer.train()

    if auto_joint_train:
        _run_auto_joint_train(cfg, trainer, train_loader, test_loader, base_train_lr)

    print(f"\n训练完成！模型保存到: {save_dir}")


if __name__ == "__main__":
    main()
