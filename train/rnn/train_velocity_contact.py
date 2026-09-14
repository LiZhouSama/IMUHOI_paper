"""
VelocityContactModule standalone training (Stage 2).
Freeze a pretrained HumanPoseModule (Stage 1) to supply hp_out for VC training.
"""
from __future__ import annotations
import os
import sys
import torch
from tqdm import tqdm

# add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model import HumanPoseModule
from model.rnn.velocity_contact import (
    VELOCITY_CONTACT_ARCH_CHOICES,
    build_velocity_contact_module,
    normalize_velocity_contact_arch,
)
from train.rnn.loss import (
    VelocityContactLoss,
    V2VelocityContactLoss,
    IMUHOIVelocityContactLoss,
)
from torch.nn.utils import clip_grad_norm_
from train.rnn.train_utils import (
    get_base_args,
    merge_config,
    setup_seed,
    setup_device,
    create_save_dir,
    create_dataloaders,
    prime_dataloader_workers,
    build_model_input_dict,
    build_optimizer_and_scheduler,
    load_model_state_from_checkpoint,
    save_checkpoint,
    save_config_snapshot,
    BaseTrainer,
    autocast,
    load_checkpoint,
    call_model_inference,
)
from train.rnn.scheduled_inputs import (
    build_gt_human_pose_outputs,
    prediction_mix_probability,
    sample_mix_dict,
)


def _cfg_get(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _resolve_hp_ckpt(cfg, cli_hp_ckpt=None):
    """Prefer explicit CLI, then cfg.hp_ckpt, then pretrained_modules.human_pose."""
    if cli_hp_ckpt:
        return cli_hp_ckpt
    cfg_hp_ckpt = getattr(cfg, "hp_ckpt", None)
    if cfg_hp_ckpt:
        return cfg_hp_ckpt
    pretrained_modules = getattr(cfg, "pretrained_modules", None)
    return _cfg_get(pretrained_modules, "human_pose")


def _milestone_lr_lambda(milestones, gamma):
    milestones = tuple(sorted(int(value) for value in milestones))
    gamma = float(gamma)

    def lr_lambda(epoch):
        return gamma ** sum(int(epoch) >= milestone for milestone in milestones)

    return lr_lambda


def _vc_parameter_group(name):
    if name.startswith(
        ("vel_net.", "hand_vel_net.", "obj_vel_net.")
    ):
        return "velocity"
    if name.startswith("contact_endpoint_"):
        return "endpoint"
    return "contact"


def _vc_endpoint_finetune_stage(target):
    """Whether this architecture wants a final endpoint-only fine-tune stage.

    An architecture whose endpoint head reads features owned by the contact
    group (VC v4 reads boundary features only) gains nothing from a stage that
    freezes that group, so it opts out and the joint stage runs to the end.
    """
    return bool(
        hasattr(target, "contact_endpoint_head")
        and getattr(target, "SUPPORTS_ENDPOINT_FINETUNE_STAGE", True)
    )


def _staged_lr_lambda(group_name, cfg, endpoint_stage=True):
    velocity_epochs = int(getattr(cfg, "vc_velocity_stage_epochs", 120))
    joint_epochs = int(getattr(cfg, "vc_joint_stage_epochs", 60))
    endpoint_start = velocity_epochs + joint_epochs
    gamma = float(getattr(cfg, "vc_group_lr_gamma", getattr(cfg, "gamma", 0.2)))

    velocity_milestones = tuple(
        sorted(int(value) for value in getattr(
            cfg,
            "vc_velocity_lr_milestones",
            [60, 100],
        ))
    )
    contact_milestones = tuple(
        sorted(int(value) for value in getattr(
            cfg,
            "vc_contact_lr_milestones",
            [20, 35, 50],
        ))
    )
    endpoint_joint_milestones = tuple(
        sorted(int(value) for value in getattr(
            cfg,
            "vc_endpoint_joint_lr_milestones",
            [30, 50],
        ))
    )
    endpoint_finetune_milestones = tuple(
        sorted(int(value) for value in getattr(
            cfg,
            "vc_endpoint_finetune_lr_milestones",
            [50, 90],
        ))
    )
    endpoint_finetune_scale = float(
        getattr(cfg, "vc_endpoint_finetune_lr_scale", 0.2)
    )

    def decay(relative_epoch, milestones):
        return gamma ** sum(relative_epoch >= milestone for milestone in milestones)

    def lr_lambda(epoch):
        epoch = int(epoch)
        if epoch < velocity_epochs:
            if group_name != "velocity":
                return 0.0
            return decay(epoch, velocity_milestones)
        if endpoint_stage and epoch >= endpoint_start:
            if group_name != "endpoint":
                return 0.0
            relative_epoch = epoch - endpoint_start
            return endpoint_finetune_scale * decay(
                relative_epoch,
                endpoint_finetune_milestones,
            )
        # Joint stage; without an endpoint tail it runs to the last epoch.
        relative_epoch = epoch - velocity_epochs
        if group_name == "contact":
            return decay(relative_epoch, contact_milestones)
        if group_name == "endpoint":
            return decay(relative_epoch, endpoint_joint_milestones)
        return 0.0

    return lr_lambda


def build_velocity_contact_optimizer_and_scheduler(model, cfg):
    """Use one optimizer/checkpoint with task-specific VC learning-rate decay."""
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    staged_training = (
        bool(getattr(cfg, "vc_staged_training", False))
        and hasattr(target, "contact_endpoint_head")
    )
    if not staged_training and not bool(getattr(cfg, "vc_grouped_lr", False)):
        return build_optimizer_and_scheduler(model, cfg)

    base_lr = float(cfg.lr)
    if bool(getattr(cfg, "use_multi_gpu", False)):
        base_lr *= len(getattr(cfg, "gpus", []))
    weight_decay = float(cfg.weight_decay)

    grouped = {"contact": [], "velocity": [], "endpoint": []}
    for name, parameter in target.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = _vc_parameter_group(name)
        grouped[group_name].append(parameter)

    parameter_groups = []
    active_names = []
    for group_name in ("contact", "velocity", "endpoint"):
        parameters = grouped[group_name]
        if not parameters:
            continue
        parameter_groups.append({
            "params": parameters,
            "lr": base_lr,
            "group_name": group_name,
        })
        active_names.append(group_name)

    # foreach: same update, multi-tensor kernels.  See create_optimizer in
    # train/rnn/train_utils.py for the measurement.
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=base_lr,
        weight_decay=weight_decay,
        foreach=True,
    )
    gamma = float(getattr(cfg, "vc_group_lr_gamma", getattr(cfg, "gamma", 0.2)))
    endpoint_stage = _vc_endpoint_finetune_stage(target)
    if staged_training:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                _staged_lr_lambda(group_name, cfg, endpoint_stage=endpoint_stage)
                for group_name in active_names
            ],
        )
    else:
        default_milestones = list(getattr(cfg, "milestones", []))
        milestone_map = {
            "contact": list(getattr(cfg, "vc_contact_lr_milestones", [20, 30, 40])),
            "velocity": list(getattr(cfg, "vc_velocity_lr_milestones", default_milestones)),
            "endpoint": list(getattr(cfg, "vc_endpoint_lr_milestones", default_milestones)),
        }
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                _milestone_lr_lambda(milestone_map[group_name], gamma)
                for group_name in active_names
            ],
        )

    counts = {
        group_name: sum(parameter.numel() for parameter in grouped[group_name])
        for group_name in active_names
    }
    print(f"VC参数组: {counts}")
    if staged_training:
        velocity_epochs = int(getattr(cfg, "vc_velocity_stage_epochs", 120))
        joint_epochs = int(getattr(cfg, "vc_joint_stage_epochs", 60))
        if endpoint_stage:
            print(
                "VC三阶段学习: "
                f"velocity=[0,{velocity_epochs}), "
                f"joint=[{velocity_epochs},{velocity_epochs + joint_epochs}), "
                f"endpoint=[{velocity_epochs + joint_epochs},{int(cfg.epochs)})"
            )
        else:
            print(
                "VC两阶段学习（该架构的endpoint头只读boundary特征，不做endpoint单独微调）: "
                f"velocity=[0,{velocity_epochs}), "
                f"joint=[{velocity_epochs},{int(cfg.epochs)})"
            )
    else:
        print(
            "VC分组学习率里程碑: "
            + ", ".join(
                f"{group_name}={milestone_map[group_name]}"
                for group_name in active_names
            )
            + f", gamma={gamma}"
        )
    return optimizer, scheduler


def get_args():
    parser = get_base_args()
    parser.description = 'VelocityContactModule training (Stage 2)'
    parser.add_argument(
        '--hp_ckpt',
        type=str,
        default=None,
        help='Path to pretrained HumanPoseModule (frozen to provide hp_out).',
    )
    parser.add_argument(
        '--velocity_contact_arch', '--velocity-contact-arch',
        dest='velocity_contact_arch',
        type=str.lower,
        choices=VELOCITY_CONTACT_ARCH_CHOICES,
        default=None,
        help=(
            'VelocityContact架构；legacy不包含interaction boundary模块，'
            'v2使用手物速度→单向obj_move→直接双手接触的级联；'
            'v3在imuhoi_cond_boundary上加双向obj_move；'
            'v4在v3基础上把人体姿态先验变成受监督模块：朝向归一化+分模态归一化+'
            '双向boundary，boundary同时喂给obj_move，并加端点/anticipation两个'
            '纯姿态辅助头；'
            'imuhoi恢复EgoIMU的全身手速度/独立物体速度/联合接触网络；'
            'imuhoi_cond*为从该基线逐项增加条件接触、几何、boundary和endpoint的消融。'
        ),
    )
    return parser.parse_args()


class VelocityContactTrainer(BaseTrainer):
    """Trainer for VelocityContactModule."""

    STAGE_LOSS_KEYS = {
        "velocity": {"obj_vel", "hand_vel"},
        "joint": {
            "obj_move",
            "hand_contact_cond",
            "hand_contact_marginal",
            "contact_endpoint",
            "contact_endpoint_order",
            "endpoint_contact_consistency",
            # The anticipation head lives in the contact group with the body
            # encoder it supervises, so it is trained here and not in the
            # endpoint-only stage.
            "contact_anticipation",
        },
        "endpoint": {
            "contact_endpoint",
            "contact_endpoint_order",
            "endpoint_contact_consistency",
        },
        "all": set(VelocityContactLoss.LOSS_KEYS),
    }

    GROUP_MODULE_NAMES = {
        # hand_vel_net/obj_vel_net are the IMUHOI-family split of vel_net; without
        # them a frozen velocity group keeps its dropout active and injects noise
        # into the velocities the contact heads consume.
        "velocity": ("vel_net", "hand_vel_net", "obj_vel_net"),
        "contact": (
            "group_encoders",
            "group_norms",
            "group_pose_norms",
            "group_vel_norms",
            "group_acc_norms",
            "root_vel_norm",
            "boundary_rnn",
            "boundary_fusion",
            "boundary_input_norm",
            "boundary_hidden_norm",
            "contact_anticipation_norm",
            "contact_anticipation_head",
            "obj_move_net",
            "hand_contact_net",
            "interaction_input_norm",
            "interaction_history_net",
            "interaction_future_proj",
            "interaction_future_conv",
            "interaction_fusion",
            "obj_move_head",
            "hand_contact_head",
        ),
        "endpoint": (
            "contact_endpoint_input_norm",
            "contact_endpoint_rnn",
            "contact_endpoint_head",
        ),
    }

    def __init__(self, cfg, model, loss_fn, train_loader, test_loader=None, hp_model=None):
        target = model.module if isinstance(model, torch.nn.DataParallel) else model
        self.staged_training = (
            bool(getattr(cfg, "vc_staged_training", False))
            and hasattr(target, "contact_endpoint_head")
        )
        self.endpoint_finetune_stage = _vc_endpoint_finetune_stage(target)
        super().__init__(cfg, model, loss_fn, train_loader, test_loader)
        self.hp_model = hp_model
        self.current_epoch = 0
        self.current_stage = None
        self.velocity_stage_best_loss = float("inf")
        self.velocity_stage_best_state = None
        if (
            self.staged_training
            and self.endpoint_finetune_stage
            and self.endpoint_stage_start >= int(cfg.epochs)
        ):
            raise ValueError(
                "VC staged training leaves no endpoint-only epochs: "
                f"velocity={self.velocity_stage_epochs}, "
                f"joint={self.joint_stage_epochs}, total={int(cfg.epochs)}"
            )
        if self.staged_training and self.velocity_stage_epochs >= int(cfg.epochs):
            raise ValueError(
                "VC staged training leaves no joint epochs: "
                f"velocity={self.velocity_stage_epochs}, total={int(cfg.epochs)}"
            )
        if isinstance(self.resume_checkpoint, dict):
            self.velocity_stage_best_loss = float(
                self.resume_checkpoint.get(
                    "velocity_stage_best_loss",
                    self.velocity_stage_best_loss,
                )
            )
            self.velocity_stage_best_state = self.resume_checkpoint.get(
                "velocity_stage_best_state",
            )
        if self.hp_model is not None:
            self.hp_model.eval()
            for p in self.hp_model.parameters():
                p.requires_grad_(False)

    def build_optimizer_and_scheduler(self):
        return build_velocity_contact_optimizer_and_scheduler(self.model, self.cfg)

    @property
    def velocity_stage_epochs(self):
        return int(getattr(self.cfg, "vc_velocity_stage_epochs", 120))

    @property
    def joint_stage_epochs(self):
        return int(getattr(self.cfg, "vc_joint_stage_epochs", 60))

    @property
    def endpoint_stage_start(self):
        if not self.endpoint_finetune_stage:
            # No endpoint tail: the joint stage owns every remaining epoch.
            return int(self.cfg.epochs)
        return self.velocity_stage_epochs + self.joint_stage_epochs

    def stage_for_epoch(self, epoch):
        if not self.staged_training:
            return "all"
        if int(epoch) < self.velocity_stage_epochs:
            return "velocity"
        if int(epoch) < self.endpoint_stage_start:
            return "joint"
        return "endpoint"

    def _target_model(self):
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def _configure_stage(self, stage):
        active_groups = {
            "velocity": {"velocity"},
            "joint": {"contact", "endpoint"},
            "endpoint": {"endpoint"},
            "all": {"velocity", "contact", "endpoint"},
        }[stage]
        target = self._target_model()
        for name, parameter in target.named_parameters():
            parameter.requires_grad_(_vc_parameter_group(name) in active_groups)

        self.model.train()
        for group_name, module_names in self.GROUP_MODULE_NAMES.items():
            train_group = group_name in active_groups
            for module_name in module_names:
                module = getattr(target, module_name, None)
                if module is not None:
                    module.train(train_group)

        if stage != self.current_stage:
            trainable = sum(
                parameter.numel()
                for parameter in target.parameters()
                if parameter.requires_grad
            )
            print(f"VC训练阶段切换: {stage}, 可训练参数={trainable}")
            self.current_stage = stage

    def _contact_schedule_epoch(self, epoch):
        if not self.staged_training:
            return int(epoch)
        return max(0, int(epoch) - self.velocity_stage_epochs)

    def _capture_parameter_group(self, group_name):
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self._target_model().named_parameters()
            if _vc_parameter_group(name) == group_name
        }

    def _restore_parameter_group(self, state):
        if not isinstance(state, dict):
            return False
        restored = 0
        with torch.no_grad():
            for name, parameter in self._target_model().named_parameters():
                value = state.get(name)
                if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
                    restored += 1
        return restored > 0

    def _restore_joint_best_for_endpoint_finetune(self):
        best_path = os.path.join(self.cfg.save_dir, "best.pt")
        if not os.path.isfile(best_path):
            print("VC endpoint微调前未找到joint best.pt，使用联合阶段末尾权重。")
            return
        try:
            try:
                checkpoint = torch.load(
                    best_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(best_path, map_location=self.device)
            if checkpoint.get("vc_stage") != "joint":
                print("VC best.pt并非联合阶段检查点，不覆盖当前模型。")
                return
            load_model_state_from_checkpoint(
                self.model,
                checkpoint,
                strict=False,
                name="VelocityContact endpoint finetune init",
            )
            for parameter_group in self.optimizer.param_groups:
                if parameter_group.get("group_name") != "endpoint":
                    continue
                for parameter in parameter_group["params"]:
                    self.optimizer.state.pop(parameter, None)
            print("VC endpoint微调从联合阶段的统一best.pt恢复，并重置endpoint优化器状态。")
        except Exception as exc:
            print(f"警告：恢复联合阶段best.pt失败，使用当前权重: {exc}")

    def model_forward(
        self,
        data_dict,
        batch=None,
        epoch=None,
        training=None,
    ):
        """Use frozen HP/GT scheduled hp_out, then forward VC."""
        training = self.model.training if training is None else bool(training)
        epoch = self.current_epoch if epoch is None else int(epoch)
        velocity_only = (
            self.staged_training
            and self.stage_for_epoch(epoch) == "velocity"
        )
        if velocity_only:
            return self.model(
                data_dict,
                hp_out=None,
                velocity_only=True,
            )

        if (
            str(getattr(self.cfg, "velocity_contact_arch", "current"))
            .strip()
            .lower()
            == "imuhoi"
        ):
            # Keep the complete VC training path independent of HP tensors,
            # matching EgoIMU rather than merely ignoring hp_out in the model.
            if training:
                return self.model(data_dict, hp_out=None)
            return call_model_inference(
                self.model,
                data_dict,
                hp_out=None,
                inference_mode="offline",
            )

        target = self._target_model()
        requires_hp = bool(getattr(target, "requires_hp", True))
        pred_hp_out = None
        if requires_hp and self.hp_model is not None:
            with torch.no_grad():
                pred_hp_out = call_model_inference(self.hp_model, data_dict, inference_mode="offline")

        hp_out = pred_hp_out
        if requires_hp and batch is not None:
            gt_hp_out = build_gt_human_pose_outputs(
                batch,
                self.device,
                dtype=data_dict['human_imu'].dtype,
            )
            if pred_hp_out is None:
                hp_out = gt_hp_out
            elif training:
                hp_out = sample_mix_dict(
                    gt_hp_out,
                    pred_hp_out,
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
                    prediction_mix_probability(
                        self._contact_schedule_epoch(epoch),
                        self.cfg,
                    ),
                )
        if training:
            # Keep training on differentiable frame-level outputs. Segment
            # decoding is an inference-only, non-differentiable operation.
            return self.model(data_dict, hp_out=hp_out)
        return call_model_inference(
            self.model,
            data_dict,
            hp_out=hp_out,
            inference_mode="offline",
        )

    def train_epoch(self, epoch):
        """Override to add gradient clipping for stability."""
        self.current_epoch = epoch
        stage = self.stage_for_epoch(epoch)
        self._configure_stage(stage)
        active_loss_keys = (
            set(self.loss_fn.LOSS_KEYS)
            if stage == "all"
            else self.STAGE_LOSS_KEYS[stage]
        )
        total_loss = 0
        loss_components = {}

        if self.writer is not None:
            for parameter_group in self.optimizer.param_groups:
                group_name = parameter_group.get("group_name", "all")
                self.writer.add_scalar(
                    f"lr/{group_name}",
                    float(parameter_group["lr"]),
                    epoch,
                )

        train_iter = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)

        for batch in train_iter:
            data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                pred_dict = self.model_forward(data_dict, batch=batch, epoch=epoch, training=True)

                total_loss_tensor, losses, weighted_losses = self.loss_fn(
                    pred_dict,
                    batch,
                    self.device,
                    active_loss_keys=active_loss_keys,
                )

            if self.scaler.is_enabled():
                self.scaler.scale(total_loss_tensor).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                total_loss_tensor.backward()

            clip_grad_norm_(
                (
                    parameter
                    for parameter in self.model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=float(getattr(self.cfg, "gradient_clip", 1.0)),
            )
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            total_loss_val = total_loss_tensor.item()
            total_loss += total_loss_val
            for key, value in weighted_losses.items():
                if isinstance(value, torch.Tensor):
                    loss_components[key] = loss_components.get(key, 0) + value.item()

            postfix = {'loss': total_loss_val}
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.item() != 0.0:
                    postfix[key] = value.item()
            train_iter.set_postfix(postfix)

            if self.writer is not None:
                self.writer.add_scalar('train/total_loss', total_loss_val, self.n_iter)
                for key, value in weighted_losses.items():
                    if isinstance(value, torch.Tensor):
                        self.writer.add_scalar(f'train/{key}', value.item(), self.n_iter)

            self.n_iter += 1

        total_loss /= len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)

        return total_loss, loss_components

    def evaluate_all_losses(self, epoch):
        """Evaluate once and derive both stage and unified checkpoint metrics."""
        if self.test_loader is None:
            return None, None, {}

        self.model.eval()
        weighted_sums = {}
        with torch.no_grad():
            test_iter = tqdm(self.test_loader, desc=f"Test {epoch}", leave=False)
            for batch in test_iter:
                data_dict = build_model_input_dict(
                    batch,
                    self.cfg,
                    self.device,
                    add_noise=False,
                )
                pred_dict = self.model_forward(
                    data_dict,
                    batch=batch,
                    epoch=epoch,
                    training=False,
                )
                _, _, weighted_losses = self.loss_fn(
                    pred_dict,
                    batch,
                    self.device,
                )
                for key, value in weighted_losses.items():
                    if isinstance(value, torch.Tensor):
                        weighted_sums[key] = (
                            weighted_sums.get(key, 0.0) + float(value.item())
                        )

        num_batches = len(self.test_loader)
        components = {
            key: value / num_batches
            for key, value in weighted_sums.items()
        }
        stage = self.stage_for_epoch(epoch)
        stage_keys = (
            set(self.loss_fn.LOSS_KEYS)
            if stage == "all"
            else self.STAGE_LOSS_KEYS[stage]
        )
        stage_loss = sum(
            components.get(key, 0.0)
            for key in stage_keys
        )
        unified_loss = None
        if stage != "velocity":
            unified_loss = sum(
                components.get(key, 0.0)
                for key in self.loss_fn.TEST_LOSS_KEYS
            )
        return unified_loss, stage_loss, components

    def train(self):
        """Train velocity, joint contact, then endpoint in one checkpoint."""
        max_epoch = int(self.cfg.epochs)
        train_loss = 0.0
        if self.start_epoch >= max_epoch:
            print(
                f"Resume checkpoint already reached epoch {self.start_epoch - 1}; "
                f"target epochs={max_epoch}."
            )
            if self.writer is not None:
                self.writer.close()
            return self.model

        for epoch in range(self.start_epoch, max_epoch):
            stage = self.stage_for_epoch(epoch)
            if (
                self.staged_training
                and stage == "joint"
                and epoch == self.velocity_stage_epochs
            ):
                if self._restore_parameter_group(self.velocity_stage_best_state):
                    print(
                        "VC联合阶段从验证集最优velocity参数开始，随后冻结velocity。"
                    )
                else:
                    print(
                        "VC联合阶段未找到velocity验证快照，冻结当前velocity参数。"
                    )
                self.velocity_stage_best_state = None
            if (
                self.staged_training
                and stage == "endpoint"
                and epoch == self.endpoint_stage_start
            ):
                self._restore_joint_best_for_endpoint_finetune()

            train_loss, train_components = self.train_epoch(epoch)
            print(
                f"\rEpoch {epoch} [{stage}], Train Loss: {train_loss:.4f}",
                end="",
            )

            eval_interval = max(1, int(getattr(self.cfg, "eval_interval", 10)))
            if epoch % eval_interval == 0 and self.test_loader is not None:
                test_loss, stage_test_loss, test_components = (
                    self.evaluate_all_losses(epoch)
                )
                if test_loss is None:
                    print(f", Stage Loss: {stage_test_loss:.4f}")
                else:
                    print(
                        f", Test Loss: {test_loss:.4f}, "
                        f"Stage Loss: {stage_test_loss:.4f}"
                    )

                if stage == "velocity":
                    if stage_test_loss < self.velocity_stage_best_loss:
                        self.velocity_stage_best_loss = stage_test_loss
                        self.velocity_stage_best_state = (
                            self._capture_parameter_group("velocity")
                        )
                        print(
                            "新的最佳velocity阶段损失: "
                            f"{self.velocity_stage_best_loss:.4f}"
                        )
                elif test_loss is not None and test_loss < self.best_loss:
                    self.best_loss = test_loss
                    save_path = os.path.join(self.cfg.save_dir, "best.pt")
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        save_path,
                        test_loss,
                        {
                            "test_components": test_components,
                            "best_loss": self.best_loss,
                            "n_iter": self.n_iter,
                            "vc_stage": stage,
                        },
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                    )
                    print(f"新的统一最佳测试损失: {self.best_loss:.4f}")

                if self.writer is not None:
                    if test_loss is not None:
                        self.writer.add_scalar(
                            "test/total_loss",
                            test_loss,
                            self.n_iter,
                        )
                    self.writer.add_scalar(
                        "test/stage_loss",
                        stage_test_loss,
                        self.n_iter,
                    )
                    for key, value in test_components.items():
                        self.writer.add_scalar(
                            f"test/{key}",
                            value,
                            self.n_iter,
                        )
            else:
                print()

            self.scheduler.step()
            last_info = {
                "train_components": train_components,
                "best_loss": self.best_loss,
                "n_iter": self.n_iter,
                "vc_stage": stage,
            }
            if stage == "velocity":
                last_info.update({
                    "velocity_stage_best_loss": self.velocity_stage_best_loss,
                    "velocity_stage_best_state": self.velocity_stage_best_state,
                })
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                os.path.join(self.cfg.save_dir, "last.pt"),
                train_loss,
                last_info,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )

        save_checkpoint(
            self.model,
            self.optimizer,
            max_epoch - 1,
            os.path.join(self.cfg.save_dir, "final.pt"),
            train_loss,
            {
                "best_loss": self.best_loss,
                "n_iter": self.n_iter,
                "vc_stage": self.stage_for_epoch(max_epoch - 1),
            },
            scheduler=self.scheduler,
            scaler=self.scaler,
        )

        if self.writer is not None:
            self.writer.close()
        return self.model


def main():
    args = get_args()
    cfg = merge_config(args)
    cfg.hp_ckpt = _resolve_hp_ckpt(cfg, getattr(args, "hp_ckpt", None))
    cfg.velocity_contact_arch = normalize_velocity_contact_arch(
        args.velocity_contact_arch
        if args.velocity_contact_arch is not None
        else getattr(cfg, "velocity_contact_arch", "current")
    )
    if cfg.velocity_contact_arch == "imuhoi":
        # The EgoIMU VC is deliberately independent of HumanPose.
        cfg.hp_ckpt = None
        # EgoIMU trains hand/object velocity and three-channel contact
        # together; the newer endpoint-specific VC curriculum does not apply.
        cfg.vc_staged_training = False

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    module_name = {
        "current": "velocity_contact",
        "legacy": "velocity_contact_legacy",
        "v2": "velocity_contact_v2",
        "imuhoi": "velocity_contact_IMUHOI",
    }.get(
        cfg.velocity_contact_arch,
        f"velocity_contact_{cfg.velocity_contact_arch}",
    )
    save_dir = create_save_dir(cfg, module_name)
    save_config_snapshot(cfg)

    print("=" * 50)
    print("Stage 2: VelocityContactModule training")
    print(f"Device: {cfg.device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Learning rate: {cfg.lr}")
    if getattr(cfg, "pretrained_ckpt", None):
        print(f"Pretrained: {cfg.pretrained_ckpt}")
    if getattr(cfg, "hp_ckpt", None):
        print(f"HumanPose checkpoint: {cfg.hp_ckpt}")
    print(f"VelocityContact架构: {cfg.velocity_contact_arch}")
    print(f"Save dir: {save_dir}")
    print("=" * 50)

    # dataloaders
    train_loader, test_loader = create_dataloaders(cfg)
    prime_dataloader_workers(train_loader, test_loader)
    if train_loader is None or len(train_loader) == 0:
        print("Error: failed to create train dataloader")
        return

    # VC model
    model = build_velocity_contact_module(cfg)
    model = model.to(cfg.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # frozen HP model (Stage 1) for hp_out
    hp_model = None
    if cfg.hp_ckpt and bool(getattr(model, "requires_hp", True)):
        device = torch.device(cfg.device)
        try:
            hp_model = HumanPoseModule(cfg, device, no_trans=cfg.no_trans)
            if os.path.exists(cfg.hp_ckpt):
                load_checkpoint(hp_model, cfg.hp_ckpt, device, strict=False)
                print(f"Loaded HumanPose checkpoint: {cfg.hp_ckpt}")
            else:
                print(f"Warning: HumanPose checkpoint not found at {cfg.hp_ckpt}, fallback to GT hp_out.")
            hp_model = hp_model.to(device)
        except Exception as exc:
            print(f"Warning: failed to init/load HumanPose; fallback to GT hp_out. {exc}")
            hp_model = None

    # loss
    if cfg.velocity_contact_arch == "imuhoi":
        loss_weights = getattr(
            cfg,
            "imuhoi_loss_weights",
            IMUHOIVelocityContactLoss.DEFAULT_WEIGHTS,
        )
        loss_fn = IMUHOIVelocityContactLoss(weights=loss_weights)
    elif cfg.velocity_contact_arch == "v2":
        loss_weights = getattr(cfg, 'loss_weights', {})
        loss_fn = V2VelocityContactLoss(weights=loss_weights)
    else:
        loss_weights = getattr(cfg, 'loss_weights', {})
        loss_fn = VelocityContactLoss(
            weights=loss_weights,
            anticipation_horizon=getattr(cfg, 'v4_anticipation_horizon', None),
        )

    # trainer
    trainer = VelocityContactTrainer(cfg, model, loss_fn, train_loader, test_loader, hp_model=hp_model)

    # train
    model = trainer.train()

    print(f"\nTraining complete! Models saved to: {save_dir}")


if __name__ == "__main__":
    main()
