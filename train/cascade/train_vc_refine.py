#!/usr/bin/env python3
"""Cascade stage 4: train VC-refine with HP, VC and the feedback-free OT frozen.

Per training window (the same 60-frame windows, noise and loaders as every RNN
stage):

    HP -> VC (first pass) -> [synthetic VC errors] -> OT (first pass)
       -> VC-refine -> OT-refine (same OT weights, refined contacts)

Losses: VelocityContact's contact terms on the refined contacts against the GT
labels (obj_move, conditional hand contact, marginal hand contact, with the VC
recipe's weights) -- the primary objective, contact correction -- plus a smaller
auxiliary position term on the OT-refine output in the units of the reported
metrics: the per-frame object L2 error over all frames (so a false contact's
drag counts, not only GT contact frames) and the metric-aligned HOI error on GT
contact frames.  It reaches VC-refine through the fixed fusion weights and the
predictor's contact input; OT stays frozen.  (With the OT recipe's weights,
obj MSE x5 + HOI x10, the HOI term alone was ~10x the contact loss and scores
only GT contact frames, i.e. it rewards more contact and never sees a drag.)

Two deliberate departures from a GT-initialised window, both because VC-refine
judges contacts against the object state and the state is drifted online:

* window-init noise: the object state both OT passes start from is offset by
  sigma * N(0, I), sigma log-uniform in [cascade_vcr_init_noise_min, _max]
  (per axis), on a fraction cascade_vcr_init_noise_prob of the windows;
  validation uses a fixed draw per batch so its loss is comparable over epochs.
  (Online published-object error: BEHAVE median 22 cm, OMOMO 12 cm.)
* synthetic VC errors (training only): on cascade_vcr_corrupt_prob of the windows
  one false contact segment (incl. static-object ones), one missed segment or
  a shifted onset/release is written into the first-pass VC logits before OT
  runs, so the first pass carries the drag a wrong contact causes.  In-sample
  VC is too accurate on its own training windows to teach the correction.
"""
from __future__ import annotations

import math
import os
import sys

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from model.cascade.cascade_model import CascadeIMUHOIModel  # noqa: E402
from model.rnn.velocity_contact import normalize_velocity_contact_arch  # noqa: E402
from train.rnn.loss import ObjectTransLoss, VelocityContactLoss  # noqa: E402
from train.rnn.scheduled_inputs import sample_gt_lb_prior_stats  # noqa: E402
from train.rnn.train_utils import (  # noqa: E402
    _capture_rng_state,
    _restore_rng_state,
    _torch_load_checkpoint,
    autocast,
    build_model_input_dict,
    create_dataloaders,
    create_save_dir,
    flatten_lstm_parameters,
    get_base_args,
    load_checkpoint,
    merge_config,
    resolve_amp_settings,
    resumed_last_improvement_epoch,
    save_config_snapshot,
    setup_device,
    setup_seed,
    should_stop_early,
)

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except Exception:  # noqa: BLE001
    SummaryWriter = None

CONTACT_LOSS_KEYS = ("obj_move", "hand_contact_cond", "hand_contact_marginal")


def get_args():
    parser = get_base_args()
    parser.description = "Cascade VC-refine training (HP/VC/OT frozen)"
    parser.add_argument("--hp_ckpt", type=str, required=True)
    parser.add_argument("--vc_ckpt", type=str, required=True)
    parser.add_argument("--ot_ckpt", type=str, required=True, help="feedback-free OT (cascade stage 3) checkpoint")
    parser.add_argument("--velocity_contact_arch", type=str, default="v5")
    return parser.parse_args()


# --------------------------------------------------------------------------- data
def gt_contacts(batch, device):
    keys = ("lhand_contact", "rhand_contact", "obj_contact")
    return torch.stack([batch[key].to(device).bool() for key in keys], dim=-1)


def corrupt_contact_logits(logits, gt, prob, generator=None):
    """Write one synthetic VC error into a fraction of the windows.

    logits [B,T,3] = (left | move, right | move, move) as produced by VC; gt [B,T,3]
    bool (left, right, move).  Ops (one per selected window): false segment on a
    hand (move logit raised too where the object is static), missed segment, or
    a 3-12 frame shift of one hand's logit track.  Segment logits are pulled to
    a random confident target through a 3-frame ramp rather than offset, so the
    error is effective whatever VC's own confidence was.
    """
    batch_size, seq_len, _ = logits.shape
    device = logits.device

    def rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    def randint(low, high, shape):
        return torch.randint(low, high, shape, device=device, generator=generator)

    rows = torch.arange(batch_size, device=device)
    frames = torch.arange(seq_len, device=device).unsqueeze(0)
    selected = rand(batch_size) < float(prob)
    op = randint(0, 3, (batch_size,))  # 0 false segment, 1 missed segment, 2 shift
    hand = randint(0, 2, (batch_size,))
    gt_hand = gt[rows, :, hand]  # [B,T]
    gt_move = gt[..., 2]
    eligible = torch.where((op == 0).unsqueeze(1), ~gt_hand, gt_hand)
    has_eligible = eligible.any(dim=1)
    start = (rand(batch_size, seq_len) * eligible.to(logits.dtype)).argmax(dim=1, keepdim=True)
    length = randint(4, 31, (batch_size, 1))
    offset = frames - start
    ramp = torch.minimum(offset + 1, length - offset).to(logits.dtype).div(3.0).clamp(0.0, 1.0)
    envelope = ramp * ((offset >= 0) & (offset < length) & eligible).to(logits.dtype)
    segment_rows = selected & has_eligible & (op < 2)
    envelope = envelope * segment_rows.to(logits.dtype).unsqueeze(1)

    out = logits.clone()
    hand_track = logits[rows, :, hand]  # [B,T]
    target = torch.where(op == 0, 1.0 + 3.0 * rand(batch_size), -(1.0 + 3.0 * rand(batch_size))).unsqueeze(1)
    new_track = hand_track + envelope * (target - hand_track)
    # Static false contact: the object-move logit must rise too, or p_move * p_hand stays low.
    move_target = (1.0 + 3.0 * rand(batch_size)).unsqueeze(1)
    move_env = envelope * ((op == 0).unsqueeze(1) & ~gt_move).to(logits.dtype)
    out[..., 2] = logits[..., 2] + move_env * (move_target - logits[..., 2])

    shift = randint(3, 13, (batch_size, 1)) * torch.where(rand(batch_size, 1) < 0.5, -1, 1)
    source = (frames - shift).clamp(0, seq_len - 1)
    shifted_track = torch.gather(hand_track, 1, source)
    shift_rows = (selected & (op == 2)).unsqueeze(1)
    new_track = torch.where(shift_rows, shifted_track, new_track)
    out[rows, :, hand] = new_track
    return out


def contact_outputs_from_logits(logits):
    """VC contact dict (the keys the cascade and the losses read) from (cond L, cond R, move) logits."""
    move = torch.sigmoid(logits[..., 2:3])
    cond = torch.sigmoid(logits[..., :2])
    return {
        "pred_hand_contact_logits": logits,
        "pred_hand_contact_prob": torch.cat((move * cond, move), dim=-1),
        "pred_obj_move_prob": move,
        "pred_hand_contact_prob_cond": cond,
        "pred_hand_contact_logits_cond": logits[..., :2],
        "pred_hand_contact_prob_marginal": move * cond,
    }


def perturb_init(obj_trans_init, prob, sigma_min, sigma_max, generator=None):
    if prob <= 0.0 or sigma_max <= 0.0:
        return obj_trans_init
    batch_size = obj_trans_init.shape[0]
    device, dtype = obj_trans_init.device, obj_trans_init.dtype
    select = torch.rand(batch_size, device=device, generator=generator) < float(prob)
    log_sigma = math.log(sigma_min) + torch.rand(batch_size, device=device, generator=generator) * (
        math.log(sigma_max) - math.log(sigma_min)
    )
    noise = torch.randn(batch_size, 3, device=device, generator=generator, dtype=dtype)
    noise = noise * log_sigma.exp().to(dtype).unsqueeze(-1) * select.to(dtype).unsqueeze(-1)
    return obj_trans_init + noise


def object_l2(pred_obj_trans, batch, device):
    """Mean per-frame object position error (metres) over windows that have an object."""
    gt = batch["obj_trans"].to(device=device, dtype=torch.float32)
    error = (pred_obj_trans.float() - gt).norm(dim=-1)
    has_object = batch.get("has_object")
    if isinstance(has_object, torch.Tensor):
        mask = has_object.to(device=device, dtype=torch.float32)
        mask = mask.reshape(error.shape) if mask.numel() == error.numel() else mask.reshape(-1, 1).expand_as(error)
        return (error * mask).sum() / mask.sum().clamp_min(1.0)
    return error.mean()


def binary_counts(pred_prob, target, threshold=0.5):
    pred = pred_prob > threshold
    tp = (pred & target).sum().item()
    fp = (pred & ~target).sum().item()
    fn = (~pred & target).sum().item()
    return tp, fp, fn


def deterministic_train_mode(module):
    """Train mode with every dropout disabled: eval numerics, but cuDNN can backpropagate.

    The frozen OT-refine pass must pass gradients to the contact input through
    its LSTMs/GRU, and cuDNN's RNN backward refuses modules in eval mode.  The
    OT has no batch norm and no other train/eval-dependent branch in this
    configuration, so this is numerically its eval forward.
    """
    module.train()
    for sub in module.modules():
        if isinstance(sub, torch.nn.Dropout):
            sub.p = 0.0
        if isinstance(sub, torch.nn.RNNBase):
            sub.dropout = 0.0
    return module


# ------------------------------------------------------------------------ trainer
class VCRefineTrainer:
    def __init__(self, cfg, model, train_loader, test_loader):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.train_loader, self.test_loader = train_loader, test_loader
        for name, module in self.model.named_children():
            for param in module.parameters():
                param.requires_grad = name == "vc_refine_module"
        if bool(getattr(cfg, "object_trans_nocontact_lock_vertical", False)):
            raise ValueError("deterministic_train_mode assumes the OT has no inference-only branch")
        deterministic_train_mode(self.model.object_trans_module)
        params = [p for p in self.model.vc_refine_module.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay), foreach=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=list(cfg.milestones), gamma=float(cfg.gamma))
        self.amp_enabled, self.amp_dtype = resolve_amp_settings(cfg, self.device)
        self.gradient_clip = float(getattr(cfg, "gradient_clip", 1.0))
        loss_weights = dict(getattr(cfg, "loss_weights", {}) or {})
        self.contact_loss = VelocityContactLoss(weights=loss_weights, anticipation_horizon=getattr(cfg, "v4_anticipation_horizon", None))
        position_weights = {key: 0.0 for key in ObjectTransLoss.LOSS_KEYS}
        position_weights.update(
            hoi_fused_l=float(getattr(cfg, "cascade_vcr_hoi_weight", 0.5)),
            hoi_fused_r=float(getattr(cfg, "cascade_vcr_hoi_weight", 0.5)),
        )
        self.position_loss = ObjectTransLoss(weights=position_weights)
        self.obj_l2_weight = float(getattr(cfg, "cascade_vcr_obj_l2_weight", 0.5))
        self.contact_weight = float(getattr(cfg, "cascade_vcr_contact_weight", 1.0))
        self.corrupt_prob = float(getattr(cfg, "cascade_vcr_corrupt_prob", 0.5))
        self.noise_prob = float(getattr(cfg, "cascade_vcr_init_noise_prob", 0.8))
        self.noise_min = float(getattr(cfg, "cascade_vcr_init_noise_min", 0.02))
        self.noise_max = float(getattr(cfg, "cascade_vcr_init_noise_max", 0.30))
        self.lb_min_frames = int(getattr(cfg, "lb_stats_min_frames", 10))
        self.lb_noise_std = float(getattr(cfg, "lb_stats_noise_std", 0.02))
        self.lb_drop_prob = float(getattr(cfg, "lb_stats_drop_prob", 0.3))
        self.start_epoch, self.best_loss, self.n_iter = 0, float("inf"), 0
        self.writer = None
        if getattr(cfg, "use_tensorboard", False) and not cfg.debug and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=os.path.join(cfg.save_dir, "tensorboard_logs"))
        resume_dir = getattr(cfg, "resume_dir", None)
        if resume_dir and os.path.isfile(os.path.join(resume_dir, "last.pt")):
            checkpoint = _torch_load_checkpoint(os.path.join(resume_dir, "last.pt"), map_location=self.device)
            self.model.vc_refine_module.load_state_dict(checkpoint["module_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = int(checkpoint["epoch"]) + 1
            self.best_loss = float(checkpoint.get("best_loss", float("inf")))
            self.n_iter = int(checkpoint.get("n_iter", 0))
            if checkpoint.get("rng_state") is not None:
                _restore_rng_state(checkpoint["rng_state"])
            print(f"Resumed VC-refine from {resume_dir} at epoch {self.start_epoch}")

    def _frozen_forward(self, data_dict):
        model = self.model
        with torch.no_grad():
            hp_out = model.human_pose_module(model._build_hp_input_dict(data_dict))
            vc_out = model.velocity_contact_module(
                {
                    "human_imu": data_dict["human_imu"],
                    "obj_imu": data_dict["obj_imu"],
                    "hand_vel_glb_init": data_dict["hand_vel_glb_init"],
                    "obj_vel_init": data_dict["obj_vel_init"],
                    "obj_trans_init": data_dict["obj_trans_init"],
                    "contact_init": data_dict.get("contact_init"),
                    "hp_out": hp_out,
                },
                hp_out=hp_out,
            )
        return hp_out, vc_out

    def _step(self, batch, training, generator=None):
        data_dict = build_model_input_dict(batch, self.cfg, self.device, add_noise=training)
        model = self.model
        model.human_pose_module.eval()
        model.velocity_contact_module.eval()
        model.vc_refine_module.train(training)
        with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            hp_out, vc_out = self._frozen_forward(data_dict)
            gt = gt_contacts(batch, self.device)
            logits = vc_out["pred_hand_contact_logits"].detach()
            if training and self.corrupt_prob > 0.0:
                logits = corrupt_contact_logits(logits.float(), gt, self.corrupt_prob).to(logits.dtype)
            vc_first = dict(vc_out)
            vc_first.update(contact_outputs_from_logits(logits))
            obj_init = perturb_init(data_dict["obj_trans_init"], self.noise_prob, self.noise_min, self.noise_max, generator)
            lb_prior = None
            if training and model._online_lb_stats_enabled():
                lb_prior = sample_gt_lb_prior_stats(
                    batch, self.device, hp_out["pred_hand_glb_pos"].dtype, min_contact_frames=self.lb_min_frames,
                    noise_std=self.lb_noise_std, drop_prob=self.lb_drop_prob,
                )
            ot_first, refined, ot_refined = model.cascade_window(
                data_dict, hp_out, vc_first, obj_init, lb_prior, refine_human=False
            )
            _, _, contact_weighted = self.contact_loss.compute_loss(
                refined, batch, self.device, active_loss_keys=CONTACT_LOSS_KEYS
            )
            _, _, position_weighted = self.position_loss.compute_loss(ot_refined, batch, self.device)
        losses = {f"contact_{key}": self.contact_weight * value.float() for key, value in contact_weighted.items()}
        losses.update({f"pos_{key}": value.float() for key, value in position_weighted.items()
                       if key in ("hoi_fused_l", "hoi_fused_r")})
        if self.obj_l2_weight > 0.0:
            losses["pos_obj_l2"] = self.obj_l2_weight * object_l2(ot_refined["pred_obj_trans"], batch, self.device)
        total = sum(losses.values())
        diagnostics = {}
        if not training:
            with torch.no_grad():
                hand_gt = gt[..., :2]
                for tag, probs in (("first", vc_first["pred_hand_contact_prob"]), ("refined", refined["pred_hand_contact_prob"])):
                    diagnostics[f"counts_{tag}"] = binary_counts(probs[..., :2].float(), hand_gt)
                obj_gt = batch["obj_trans"].to(self.device).float()
                for tag, out in (("first", ot_first), ("refined", ot_refined)):
                    diagnostics[f"obj_err_{tag}"] = (out["pred_obj_trans"].float() - obj_gt).norm(dim=-1).mean().item()
        return total, losses, diagnostics

    def train_epoch(self, epoch):
        totals, count = {}, 0
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f"VCR epoch {epoch}", leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            loss, losses, _ = self._step(batch, training=True)
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.vc_refine_module.parameters(), self.gradient_clip)
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            if self.writer is not None:
                self.writer.add_scalar("train/total_loss", loss.item(), self.n_iter)
            self.n_iter += 1
        return total_loss / max(count, 1), {key: value / max(count, 1) for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, epoch):
        if self.test_loader is None:
            return None, {}
        totals, count, total_loss = {}, 0, 0.0
        counts = {"first": [0, 0, 0], "refined": [0, 0, 0]}
        obj_err = {"first": 0.0, "refined": 0.0}
        for index, batch in enumerate(tqdm(self.test_loader, desc=f"VCR test {epoch}", leave=False)):
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(getattr(self.cfg, "seed", 42)) * 1000 + index)
            loss, losses, diagnostics = self._step(batch, training=False, generator=generator)
            total_loss += loss.item()
            count += 1
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            for tag in counts:
                for slot, value in enumerate(diagnostics[f"counts_{tag}"]):
                    counts[tag][slot] += value
                obj_err[tag] += diagnostics[f"obj_err_{tag}"]
        components = {key: value / max(count, 1) for key, value in totals.items()}
        for tag, (tp, fp, fn) in counts.items():
            components[f"hand_f1_{tag}"] = 2 * tp / max(2 * tp + fp + fn, 1)
            components[f"hand_fp_frames_{tag}"] = fp
            components[f"hand_fn_frames_{tag}"] = fn
            components[f"obj_err_cm_{tag}"] = 100.0 * obj_err[tag] / max(count, 1)
        return total_loss / max(count, 1), components

    def _save(self, prefix, epoch, loss, extra=None):
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "best_loss": self.best_loss,
            "n_iter": self.n_iter,
            "module_state_dict": self.model.vc_refine_module.state_dict(),
            "model_state_dict": self.model.vc_refine_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "rng_state": _capture_rng_state(),
        }
        checkpoint.update(extra or {})
        path = os.path.join(self.cfg.save_dir, f"{prefix}.pt")
        torch.save(checkpoint, f"{path}.tmp")
        os.replace(f"{path}.tmp", path)

    def train(self):
        cfg = self.cfg
        eval_interval = max(1, int(getattr(cfg, "eval_interval", 1)))
        patience = int(getattr(cfg, "early_stopping_patience_epochs", 0) or 0)
        min_epochs = int(getattr(cfg, "early_stopping_min_epochs", 0) or 0)
        last_improvement = resumed_last_improvement_epoch(cfg.save_dir, self.start_epoch, "best.pt")
        stopped_at, train_loss = None, 0.0
        for epoch in range(self.start_epoch, int(cfg.epochs)):
            train_loss, train_components = self.train_epoch(epoch)
            message = f"Epoch {epoch} train {train_loss:.4f}"
            if self.writer is not None:
                for key, value in train_components.items():
                    self.writer.add_scalar(f"train/{key}", value, self.n_iter)
            if epoch % eval_interval == 0 and self.test_loader is not None:
                test_loss, components = self.evaluate(epoch)
                message += (
                    f" | test {test_loss:.4f} F1 {components['hand_f1_first']:.4f}->{components['hand_f1_refined']:.4f}"
                    f" FP {components['hand_fp_frames_first']}->{components['hand_fp_frames_refined']}"
                    f" FN {components['hand_fn_frames_first']}->{components['hand_fn_frames_refined']}"
                    f" obj {components['obj_err_cm_first']:.2f}->{components['obj_err_cm_refined']:.2f} cm"
                )
                if self.writer is not None:
                    self.writer.add_scalar("test/total_loss", test_loss, self.n_iter)
                    for key, value in components.items():
                        self.writer.add_scalar(f"test/{key}", value, self.n_iter)
                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                    last_improvement = epoch
                    self._save("best", epoch, test_loss, {"test_components": components})
                    message += " *"
            print(message, flush=True)
            self.scheduler.step()
            self._save("last", epoch, train_loss)
            if should_stop_early(epoch, last_improvement, patience, min_epochs, self.test_loader is not None):
                stopped_at = epoch
                print(f"Early stopping at epoch {epoch} (best {self.best_loss:.4f} at {last_improvement})", flush=True)
                break
        final_epoch = int(cfg.epochs) - 1 if stopped_at is None else stopped_at
        self._save("final", final_epoch, train_loss, {"stopped_early_at": stopped_at})
        if self.writer is not None:
            self.writer.close()


def main():
    args = get_args()
    cfg = merge_config(args)
    cfg.velocity_contact_arch = normalize_velocity_contact_arch(args.velocity_contact_arch)
    cfg.object_trans_arch = "v2"
    cfg.object_trans_state_feedback = False
    setup_seed(cfg.seed)
    cfg = setup_device(cfg)
    create_save_dir(cfg, "cascade_vc_refine")
    cfg.hp_ckpt, cfg.vc_ckpt, cfg.ot_ckpt = args.hp_ckpt, args.vc_ckpt, args.ot_ckpt
    save_config_snapshot(cfg)
    train_loader, test_loader = create_dataloaders(cfg)
    device = torch.device(cfg.device)
    model = CascadeIMUHOIModel(cfg, device, no_trans=bool(cfg.no_trans), velocity_contact_arch=cfg.velocity_contact_arch)
    load_checkpoint(model.human_pose_module, args.hp_ckpt, device)
    load_checkpoint(model.velocity_contact_module, args.vc_ckpt, device, strict=False)
    load_checkpoint(model.object_trans_module, args.ot_ckpt, device, strict=False)
    flatten_lstm_parameters(model)
    trainable = sum(p.numel() for p in model.vc_refine_module.parameters())
    print(f"VC-refine parameters: {trainable}; lr {cfg.lr}; epochs {cfg.epochs}; milestones {list(cfg.milestones)}")
    print(
        "corrupt_prob", getattr(cfg, "cascade_vcr_corrupt_prob", 0.5),
        "init_noise", getattr(cfg, "cascade_vcr_init_noise_prob", 0.8),
        getattr(cfg, "cascade_vcr_init_noise_min", 0.02), getattr(cfg, "cascade_vcr_init_noise_max", 0.30),
        flush=True,
    )
    VCRefineTrainer(cfg, model, train_loader, test_loader).train()
    print(f"VC-refine training finished: {cfg.save_dir}")


if __name__ == "__main__":
    main()
