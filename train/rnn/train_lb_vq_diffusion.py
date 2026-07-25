"""Train the standalone mesh-aware l_b VQ prior and observation diffusion.

The ObjectTrans model is not imported or modified by this script.

Examples:
    # Train VQ first, then train diffusion from the best VQ checkpoint.
    python train/rnn/train_lb_vq_diffusion.py \
        --cfg configs/IMUHOI_train_rnn.yaml --stage all

    # Train only diffusion from an existing VQ checkpoint.
    python train/rnn/train_lb_vq_diffusion.py \
        --cfg configs/IMUHOI_train_rnn.yaml --stage diffusion \
        --vq_ckpt outputs/IMUHOI_RNN_2/lb_vq_diffusion_xxx/vq_best.pt
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import HumanPoseModule, VelocityContactModule
from model.rnn.lb_vq_diffusion import (
    DEFAULT_LB_QUANTILES,
    LBLatentDiffusion,
    LBVQAutoEncoder,
    build_lb_observation_sequence,
    build_lb_privileged_sequence,
    compute_lb_quantile_targets,
    masked_lb_quantile_loss,
    masked_lb_quantile_mae,
)
from train.rnn.train_utils import (
    build_model_input_dict,
    call_model_inference,
    create_dataloaders,
    create_save_dir,
    get_base_args,
    get_model_state_dict,
    load_checkpoint,
    merge_config,
    save_config_snapshot,
    setup_device,
    setup_seed,
)

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:
    SummaryWriter = None


def get_args():
    parser = get_base_args()
    parser.description = "Train standalone l_b VQ-VAE and observation-conditioned latent diffusion"
    parser.add_argument("--stage", choices=("vq", "diffusion", "all"), default="all")
    parser.add_argument("--hp_ckpt", type=str, default=None, help="Frozen HumanPose checkpoint")
    parser.add_argument("--vc_ckpt", type=str, default=None, help="Frozen VelocityContact checkpoint")
    parser.add_argument("--vq_ckpt", type=str, default=None, help="VQ checkpoint required by diffusion-only stage")
    parser.add_argument("--vq_epochs", type=int, default=None)
    parser.add_argument("--diffusion_epochs", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=None, help="DDIM steps used by validation sampling")
    return parser.parse_args()


def _prior_cfg(cfg):
    return getattr(cfg, "lb_vq_prior", cfg)


def _cfg_value(cfg, name, default):
    block = _prior_cfg(cfg)
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _mapping_get(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _resolve_frontend_checkpoint(cfg, explicit_path: Optional[str], key: str) -> str:
    path = explicit_path
    if not path:
        path = _mapping_get(getattr(cfg, "pretrained_modules", None), key)
    if not path:
        raise ValueError(
            f"A pretrained {key} checkpoint is required because this experiment must use actual HP/VC inference."
        )
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{key} checkpoint does not exist: {path}")
    return path


def _load_stage_state(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
    state = checkpoint.get("model_state_dict", checkpoint.get("module_state_dict"))
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    return checkpoint


def _save_stage_checkpoint(
    path: str,
    stage: str,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    metrics: Dict[str, float],
    extra: Optional[Dict] = None,
) -> None:
    payload = {
        "stage": stage,
        "epoch": int(epoch),
        "model_state_dict": get_model_state_dict(model),
        "module_state_dict": get_model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "metrics": dict(metrics),
        "loss": float(metrics.get("loss", float("inf"))),
    }
    if extra:
        payload.update(extra)
    temporary = f"{path}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class _MetricAverager:
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}

    def update(self, values: Dict[str, object], weight: int) -> None:
        weight = int(max(1, weight))
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = float(value.detach().item())
            elif not isinstance(value, (int, float)):
                continue
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * weight
            self.weights[key] = self.weights.get(key, 0.0) + weight

    def mean(self) -> Dict[str, float]:
        return {
            key: self.sums[key] / max(self.weights.get(key, 0.0), 1.0)
            for key in self.sums
        }


class FrozenObservationFrontend:
    """Run the exact frozen HP -> VC inference chain used by OT."""

    def __init__(self, cfg, hp_model: torch.nn.Module, vc_model: torch.nn.Module):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.hp_model = _freeze(hp_model.to(self.device))
        self.vc_model = _freeze(vc_model.to(self.device))
        quantiles = _cfg_value(cfg, "quantiles", DEFAULT_LB_QUANTILES)
        self.quantiles = tuple(float(value) for value in quantiles)
        self.min_contact_frames = int(_cfg_value(cfg, "min_contact_frames", 3))

    @torch.no_grad()
    def build(self, batch: Dict, *, add_noise: bool) -> Optional[Dict[str, torch.Tensor]]:
        self.hp_model.eval()
        self.vc_model.eval()
        data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=add_noise)
        hp_out = call_model_inference(self.hp_model, data_dict, inference_mode="offline")
        vc_out = call_model_inference(
            self.vc_model,
            data_dict,
            hp_out=hp_out,
            inference_mode="offline",
        )

        observation = build_lb_observation_sequence(hp_out, vc_out, data_dict["obj_imu"])
        privileged = build_lb_privileged_sequence(
            hp_out,
            vc_out,
            data_dict["obj_rot_gt"],
            data_dict["obj_trans_gt"],
            data_dict.get("obj_scale_gt"),
        )

        lhand_lb = batch["lhand_lb"].to(device=self.device, dtype=observation.dtype)
        rhand_lb = batch["rhand_lb"].to(device=self.device, dtype=observation.dtype)
        lhand_contact = batch["lhand_contact"].to(device=self.device, dtype=torch.bool)
        rhand_contact = batch["rhand_contact"].to(device=self.device, dtype=torch.bool)
        targets, valid_hands, contact_counts = compute_lb_quantile_targets(
            lhand_lb,
            rhand_lb,
            lhand_contact,
            rhand_contact,
            has_object=data_dict.get("has_object"),
            quantiles=self.quantiles,
            min_contact_frames=self.min_contact_frames,
        )
        valid_samples = valid_hands.any(dim=1)
        if not bool(valid_samples.any().item()):
            return None

        mesh_points = data_dict["obj_points_canonical"]
        obj_scale = data_dict.get("obj_scale_gt")
        if isinstance(obj_scale, torch.Tensor):
            scale = obj_scale.to(device=mesh_points.device, dtype=mesh_points.dtype)
            if scale.dim() == 1:
                scale = scale.unsqueeze(0)
            # Scale is effectively constant for one object sequence.  Median is
            # robust to the tiny frame-wise numerical variation in processed data.
            scale = scale.median(dim=1).values.view(-1, 1, 1)
            mesh_points = mesh_points * scale
        return {
            "observation": observation[valid_samples],
            "privileged": privileged[valid_samples],
            "mesh_points": mesh_points[valid_samples],
            "targets": targets[valid_samples],
            "valid_hands": valid_hands[valid_samples],
            "contact_counts": contact_counts[valid_samples],
        }


class LBVQTrainer:
    def __init__(self, cfg, model, frontend, train_loader, test_loader, writer=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.frontend = frontend
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.writer = writer
        self.epochs = int(_cfg_value(cfg, "vq_epochs", 100))
        learning_rate = float(_cfg_value(cfg, "vq_lr", 2e-4))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=float(_cfg_value(cfg, "vq_weight_decay", cfg.weight_decay)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.epochs, 1),
            eta_min=learning_rate * 0.05,
        )
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        self.gradient_clip = float(_cfg_value(cfg, "gradient_clip", 1.0))
        self.recon_weight = float(_cfg_value(cfg, "vq_recon_weight", 10.0))
        self.usage_weight = float(_cfg_value(cfg, "vq_usage_weight", 0.01))
        self.smooth_weight = float(_cfg_value(cfg, "vq_smooth_weight", 0.01))

    def _run_epoch(self, loader, epoch: int, training: bool) -> Dict[str, float]:
        self.model.train(training)
        metrics = _MetricAverager()
        iterator = tqdm(loader, desc=f"VQ {'train' if training else 'eval'} {epoch}", leave=False)
        for batch in iterator:
            prepared = self.frontend.build(batch, add_noise=training)
            if prepared is None:
                continue
            batch_size = int(prepared["targets"].shape[0])
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with autocast(enabled=self.device.type == "cuda"):
                    output = self.model(prepared["privileged"], prepared["mesh_points"])
                    recon_loss = masked_lb_quantile_loss(
                        output["lb_quantiles"],
                        prepared["targets"],
                        prepared["valid_hands"],
                    )
                    if output["z_e"].shape[1] > 1:
                        latent_smooth = (output["z_e"][:, 1:] - output["z_e"][:, :-1]).square().mean()
                    else:
                        latent_smooth = output["z_e"].sum() * 0.0
                    loss = (
                        self.recon_weight * recon_loss
                        + output["codebook_loss"]
                        + output["commitment_loss"]
                        + self.usage_weight * output["usage_loss"]
                        + self.smooth_weight * latent_smooth
                    )

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            quantile_mae = masked_lb_quantile_mae(
                output["lb_quantiles"], prepared["targets"], prepared["valid_hands"]
            )
            values = {
                "loss": loss,
                "recon": recon_loss,
                "quantile_mae_cm": quantile_mae * 100.0,
                "codebook": output["codebook_loss"],
                "commitment": output["commitment_loss"],
                "usage": output["usage_loss"],
                "latent_smooth": latent_smooth,
                "quantization_error": output["quantization_error"],
                "perplexity": output["perplexity"],
                "active_code_fraction": output["active_code_fraction"],
            }
            if not training:
                vq_model = _unwrap(self.model)
                latents = output["z_q"].detach()
                mean_latents = latents.mean(dim=(0, 1), keepdim=True).expand_as(latents)
                mean_prediction = vq_model.decode_latents(mean_latents)["lb_quantiles"]
                values["mean_latent_mae_cm"] = (
                    masked_lb_quantile_mae(
                        mean_prediction, prepared["targets"], prepared["valid_hands"]
                    )
                    * 100.0
                )
                if batch_size > 1:
                    shuffled_prediction = vq_model.decode_latents(
                        torch.roll(latents, shifts=1, dims=0)
                    )["lb_quantiles"]
                    values["shuffled_latent_mae_cm"] = (
                        masked_lb_quantile_mae(
                            shuffled_prediction,
                            prepared["targets"],
                            prepared["valid_hands"],
                        )
                        * 100.0
                    )
            metrics.update(values, batch_size)
            iterator.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}",
                mae_cm=f"{float((quantile_mae * 100.0).detach().item()):.2f}",
                ppl=f"{float(output['perplexity'].detach().item()):.1f}",
            )
        return metrics.mean()

    def train(self, save_dir: str) -> str:
        best_path = os.path.join(save_dir, "vq_best.pt")
        last_path = os.path.join(save_dir, "vq_last.pt")
        best_score = float("inf")
        for epoch in range(self.epochs):
            train_metrics = self._run_epoch(self.train_loader, epoch, training=True)
            eval_metrics = (
                self._run_epoch(self.test_loader, epoch, training=False)
                if self.test_loader is not None
                else train_metrics
            )
            self.scheduler.step()
            _log_epoch("vq", epoch, train_metrics, eval_metrics, self.writer, save_dir)
            _save_stage_checkpoint(
                last_path,
                "vq",
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch,
                eval_metrics,
            )
            score = float(eval_metrics.get("loss", float("inf")))
            if score < best_score:
                best_score = score
                _save_stage_checkpoint(
                    best_path,
                    "vq",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch,
                    eval_metrics,
                )
                print(f"VQ best updated: epoch={epoch}, loss={best_score:.6f}")
        _load_stage_state(_unwrap(self.model), best_path, self.device)
        return best_path


class LBDiffusionTrainer:
    def __init__(self, cfg, model, vq_model, frontend, train_loader, test_loader, writer=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.vq_model = _freeze(vq_model.to(self.device))
        self.frontend = frontend
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.writer = writer
        self.epochs = int(_cfg_value(cfg, "diffusion_epochs", 100))
        learning_rate = float(_cfg_value(cfg, "diffusion_lr", 2e-4))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=float(_cfg_value(cfg, "diffusion_weight_decay", cfg.weight_decay)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.epochs, 1),
            eta_min=learning_rate * 0.05,
        )
        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        self.gradient_clip = float(_cfg_value(cfg, "gradient_clip", 1.0))
        self.latent_weight = float(_cfg_value(cfg, "diffusion_latent_weight", 1.0))
        self.decode_weight = float(_cfg_value(cfg, "diffusion_decode_weight", 5.0))
        self.codebook_weight = float(_cfg_value(cfg, "diffusion_codebook_weight", 0.1))
        self.sample_steps = int(_cfg_value(cfg, "diffusion_sample_steps", 50))
        self.eval_sample_batches = int(_cfg_value(cfg, "eval_sample_batches", 1))

    def _target_latents(self, prepared: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.vq_model.encode(prepared["privileged"], prepared["mesh_points"])
        return encoded["z_q"].detach()

    def _run_epoch(self, loader, epoch: int, training: bool) -> Dict[str, float]:
        self.model.train(training)
        self.vq_model.eval()
        metrics = _MetricAverager()
        iterator = tqdm(loader, desc=f"Diff {'train' if training else 'eval'} {epoch}", leave=False)
        sampled_batches = 0
        for batch in iterator:
            prepared = self.frontend.build(batch, add_noise=training)
            if prepared is None:
                continue
            target_latents = self._target_latents(prepared)
            batch_size = int(target_latents.shape[0])
            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with autocast(enabled=self.device.type == "cuda"):
                    output = self.model(prepared["observation"], target_latents)
                    prediction_type = _unwrap(self.model).prediction_type
                    if prediction_type == "eps":
                        latent_loss = F.mse_loss(output["eps_pred"], output["noise"])
                    else:
                        latent_loss = F.mse_loss(output["x0_pred"], target_latents)

                    decoded = self.vq_model.decode_latents(output["x0_pred"])
                    decode_loss = masked_lb_quantile_loss(
                        decoded["lb_quantiles"],
                        prepared["targets"],
                        prepared["valid_hands"],
                    )
                    nearest, _, _ = self.vq_model.quantizer.nearest(output["x0_pred"])
                    codebook_distance = F.mse_loss(output["x0_pred"], nearest.detach())
                    loss = (
                        self.latent_weight * latent_loss
                        + self.decode_weight * decode_loss
                        + self.codebook_weight * codebook_distance
                    )

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            denoise_mae = masked_lb_quantile_mae(
                decoded["lb_quantiles"], prepared["targets"], prepared["valid_hands"]
            )
            values = {
                "loss": loss,
                "latent": latent_loss,
                "decode": decode_loss,
                "codebook_distance": codebook_distance,
                "denoise_quantile_mae_cm": denoise_mae * 100.0,
            }

            if not training and sampled_batches < self.eval_sample_batches:
                diffusion = _unwrap(self.model)
                latent_len = target_latents.shape[1]
                initial_noise = torch.zeros(
                    batch_size,
                    latent_len,
                    diffusion.latent_dim,
                    device=self.device,
                    dtype=target_latents.dtype,
                )
                sampled = diffusion.sample(
                    prepared["observation"],
                    steps=self.sample_steps,
                    eta=0.0,
                    initial_noise=initial_noise,
                )["latents"]
                continuous = self.vq_model.decode_latents(sampled, project_to_codebook=False)["lb_quantiles"]
                projected = self.vq_model.decode_latents(sampled, project_to_codebook=True)["lb_quantiles"]
                values["sample_quantile_mae_cm"] = (
                    masked_lb_quantile_mae(
                        continuous, prepared["targets"], prepared["valid_hands"]
                    )
                    * 100.0
                )
                values["sample_projected_mae_cm"] = (
                    masked_lb_quantile_mae(
                        projected, prepared["targets"], prepared["valid_hands"]
                    )
                    * 100.0
                )
                sampled_batches += 1

            metrics.update(values, batch_size)
            iterator.set_postfix(
                loss=f"{float(loss.detach().item()):.4f}",
                mae_cm=f"{float((denoise_mae * 100.0).detach().item()):.2f}",
            )
        return metrics.mean()

    def train(self, save_dir: str, vq_checkpoint: str) -> str:
        best_path = os.path.join(save_dir, "diffusion_best.pt")
        best_sample_path = os.path.join(save_dir, "diffusion_best_sample.pt")
        last_path = os.path.join(save_dir, "diffusion_last.pt")
        best_score = float("inf")
        best_sample_score = float("inf")
        for epoch in range(self.epochs):
            train_metrics = self._run_epoch(self.train_loader, epoch, training=True)
            eval_metrics = (
                self._run_epoch(self.test_loader, epoch, training=False)
                if self.test_loader is not None
                else train_metrics
            )
            self.scheduler.step()
            _log_epoch("diffusion", epoch, train_metrics, eval_metrics, self.writer, save_dir)
            extra = {"vq_checkpoint": vq_checkpoint}
            _save_stage_checkpoint(
                last_path,
                "diffusion",
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                epoch,
                eval_metrics,
                extra=extra,
            )
            score = float(eval_metrics.get("loss", float("inf")))
            if score < best_score:
                best_score = score
                _save_stage_checkpoint(
                    best_path,
                    "diffusion",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch,
                    eval_metrics,
                    extra=extra,
                )
                print(f"Diffusion best updated: epoch={epoch}, loss={best_score:.6f}")
            sample_score = float(eval_metrics.get("sample_quantile_mae_cm", float("inf")))
            if math.isfinite(sample_score) and sample_score < best_sample_score:
                best_sample_score = sample_score
                _save_stage_checkpoint(
                    best_sample_path,
                    "diffusion",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    epoch,
                    eval_metrics,
                    extra=extra,
                )
                print(
                    "Diffusion sample best updated: "
                    f"epoch={epoch}, sample_mae_cm={best_sample_score:.6f}"
                )
        return best_path


def _log_epoch(stage, epoch, train_metrics, eval_metrics, writer, save_dir):
    train_text = ", ".join(f"{key}={value:.6f}" for key, value in sorted(train_metrics.items()))
    eval_text = ", ".join(f"{key}={value:.6f}" for key, value in sorted(eval_metrics.items()))
    print(f"[{stage}] epoch={epoch} train: {train_text}")
    print(f"[{stage}] epoch={epoch} eval:  {eval_text}")
    if writer is not None:
        for key, value in train_metrics.items():
            writer.add_scalar(f"{stage}/train_{key}", value, epoch)
        for key, value in eval_metrics.items():
            writer.add_scalar(f"{stage}/eval_{key}", value, epoch)
    record = {
        "stage": stage,
        "epoch": int(epoch),
        "train": train_metrics,
        "eval": eval_metrics,
    }
    with open(os.path.join(save_dir, "metrics.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = get_args()
    cfg = merge_config(args)
    if not hasattr(cfg, "lb_vq_prior"):
        raise ValueError("Config must define an lb_vq_prior block")

    if args.batch_size is None:
        cfg.batch_size = int(_cfg_value(cfg, "batch_size", cfg.batch_size))
    if args.num_workers is None:
        cfg.num_workers = int(_cfg_value(cfg, "num_workers", cfg.num_workers))
    if cfg.debug:
        cfg.batch_size = min(int(cfg.batch_size), 4)
    if args.lr is not None:
        cfg.lb_vq_prior.vq_lr = float(args.lr)
        cfg.lb_vq_prior.diffusion_lr = float(args.lr)
    if args.vq_epochs is not None:
        cfg.lb_vq_prior.vq_epochs = int(args.vq_epochs)
    if args.diffusion_epochs is not None:
        cfg.lb_vq_prior.diffusion_epochs = int(args.diffusion_epochs)
    if args.sample_steps is not None:
        cfg.lb_vq_prior.diffusion_sample_steps = int(args.sample_steps)
    cfg.lb_vq_prior.max_seq_len = max(
        int(_cfg_value(cfg, "max_seq_len", 0)),
        int(cfg.train.window),
        int(cfg.test.window),
    )
    cfg.deterministic_test_windows = bool(
        _cfg_value(cfg, "deterministic_test_windows", True)
    )
    cfg.lb_vq_stage = args.stage

    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    save_dir = create_save_dir(cfg, f"lb_vq_diffusion_{args.stage}")

    hp_ckpt = _resolve_frontend_checkpoint(cfg, args.hp_ckpt, "human_pose")
    vc_ckpt = _resolve_frontend_checkpoint(cfg, args.vc_ckpt, "velocity_contact")
    cfg.hp_ckpt = hp_ckpt
    cfg.vc_ckpt = vc_ckpt
    cfg.vq_ckpt = args.vq_ckpt or _cfg_value(cfg, "vq_checkpoint", None)
    save_config_snapshot(cfg)

    print("=" * 72)
    print("Standalone l_b VQ + latent diffusion training")
    print(f"stage: {args.stage}")
    print(f"device: {cfg.device}")
    print(f"batch size: {cfg.batch_size}")
    print(f"HP checkpoint: {hp_ckpt}")
    print(f"VC checkpoint: {vc_ckpt}")
    print(f"save dir: {save_dir}")
    print("ObjectTransModule is not instantiated by this script.")
    print("=" * 72)

    train_loader, test_loader = create_dataloaders(cfg)
    if train_loader is None or len(train_loader) == 0:
        raise RuntimeError("Training dataloader is empty")

    device = torch.device(cfg.device)
    hp_model = HumanPoseModule(cfg, device, no_trans=cfg.no_trans)
    load_checkpoint(hp_model, hp_ckpt, device, strict=True)
    vc_model = VelocityContactModule(cfg)
    load_checkpoint(vc_model, vc_ckpt, device, strict=False)
    frontend = FrozenObservationFrontend(cfg, hp_model, vc_model)

    writer = None
    if bool(getattr(cfg, "use_tensorboard", False)) and not cfg.debug and SummaryWriter is not None:
        writer = SummaryWriter(os.path.join(save_dir, "tensorboard_logs"))

    vq_model = LBVQAutoEncoder(cfg)
    vq_checkpoint = cfg.vq_ckpt
    if args.stage in {"vq", "all"}:
        vq_trainer = LBVQTrainer(cfg, vq_model, frontend, train_loader, test_loader, writer=writer)
        vq_checkpoint = vq_trainer.train(save_dir)
        # The VQ optimizer/scaler can otherwise keep a full extra copy of its
        # states alive throughout diffusion training.
        del vq_trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    diffusion_checkpoint = None
    if args.stage in {"diffusion", "all"}:
        if not vq_checkpoint:
            raise ValueError("--vq_ckpt or lb_vq_prior.vq_checkpoint is required for diffusion stage")
        vq_checkpoint = os.path.normpath(vq_checkpoint)
        if not os.path.isfile(vq_checkpoint):
            raise FileNotFoundError(f"VQ checkpoint does not exist: {vq_checkpoint}")
        if args.stage == "diffusion":
            _load_stage_state(vq_model, vq_checkpoint, device)
        diffusion_model = LBLatentDiffusion(cfg)
        diffusion_trainer = LBDiffusionTrainer(
            cfg,
            diffusion_model,
            vq_model,
            frontend,
            train_loader,
            test_loader,
            writer=writer,
        )
        diffusion_checkpoint = diffusion_trainer.train(save_dir, vq_checkpoint)

    if writer is not None:
        writer.close()
    print("Training complete")
    print(f"VQ checkpoint: {vq_checkpoint}")
    if diffusion_checkpoint:
        print(f"Diffusion checkpoint: {diffusion_checkpoint}")


if __name__ == "__main__":
    main()
