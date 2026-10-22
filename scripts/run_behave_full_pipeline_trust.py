#!/usr/bin/env python3
"""Full BEHAVE pipeline (HP -> VC-v4 -> ObjectTrans+joint) for the state-trust comparison.

Unlike ``run_omomo_cano_ot_variants.py`` (which reuses the canonical OMOMO HP/VC
checkpoints), every stage here is trained on BEHAVE, so the long-horizon
comparison is measured **in domain**: training and validation use the BEHAVE
segment split, the reported numbers use the BEHAVE full-length test sequences
(median 1424 frames).

Schedule scaling (the reason this is not a thin wrapper).  The lr milestones,
the epoch budget and the early-stopping patience of ``configs/IMUHOI_train_rnn.yaml``
were tuned on OMOMO, where ``batch_size x windows_per_sequence`` yields 23
optimizer steps per epoch.  BEHAVE's training split is ~1/3 the size, so the
same settings would decay the learning rate after 1/3 of the optimizer steps and
stop training long before the schedule intends.  This script therefore computes

    scale = steps_per_epoch(reference dataset) / steps_per_epoch(train dataset)

and multiplies every epoch-like quantity by it, which keeps the lr schedule,
the evaluation cadence and the patience windows identical **in optimizer-step
space**.  Early stopping is then placed past the first decay: a MultiStepLR run
usually improves again right after a milestone, so a patience window that closes
before it would stop a run that was about to improve.  For the staged VC trainer
the window is placed past the last stage transition for the same reason.

    python scripts/run_behave_full_pipeline_trust.py \
        --experiment_root outputs/IMUHOI++/RNN_cano/behave_full_trust_20260911 \
        --variants cur_hoi_p800 cur_hoi_trust_mild cur_hoi_trust cur_hoi_trust_perturb \
        --max_parallel 2
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGE_PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
DEFAULT_CFG = PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml"
DATASET = "behave"                   # overridden by --dataset
REFERENCE_DATASET = "omomo"          # the dataset the config's schedule was tuned on
VC_ARCH = "v4"                       # overridden by --vc_arch

# Variant recipes: identical to the OMOMO state-trust comparison so the two
# experiments can be read side by side.
HOI_ALIGNED = {"obj_trans": 1.0, "hoi_fused_l": 10.0, "hoi_fused_r": 10.0}
TRUST_LOSSES = {**HOI_ALIGNED, "prior_hoi_l": 5.0, "prior_hoi_r": 5.0, "state_conf": 1.0}
EAG_LOSSES = {**HOI_ALIGNED, "prior_hoi_l": 5.0, "prior_hoi_r": 5.0}
VARIANTS = {
    "cur_hoi_p800": {
        "label": "baseline: cur_hoi recipe (HOI-aligned fused loss), no state trust",
        "overrides": {"loss_weights": dict(HOI_ALIGNED)},
    },
    "cur_hoi_trust_noaug": {
        "label": "state-trust module only (prior head + consistency features + confidence), no corruption",
        "overrides": {"loss_weights": dict(TRUST_LOSSES), "object_trans_state_trust": True},
    },
    "cur_hoi_trust_mild": {
        "label": "state trust + mild window-init corruption (0.10 m on 25% of windows)",
        "overrides": {
            "loss_weights": dict(TRUST_LOSSES),
            "object_trans_state_trust": True,
            "object_trans_init_noise_std": 0.10,
            "object_trans_init_noise_prob": 0.25,
        },
    },
    "cur_hoi_trust": {
        "label": "state trust + window-init corruption (0.20 m on 50% of windows)",
        "overrides": {
            "loss_weights": dict(TRUST_LOSSES),
            "object_trans_state_trust": True,
            "object_trans_init_noise_std": 0.20,
            "object_trans_init_noise_prob": 0.5,
        },
    },
    "cur_hoi_eag_modes": {
        "label": "cur_hoi_eag + contact modes (per-hand rigidity scaling the increment, supervised by GT bone speed)",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED, "prior_hoi_l": 5.0, "prior_hoi_r": 5.0, "grasp_mode": 1.0},
            "object_trans_grasp_latent": True,
            "object_trans_grasp_modes": True,
        },
    },
    "cur_hoi_p800_histmix": {
        "label": "cur_hoi_p800 + mixed-regime rollout-aligned training (hist prob 0.5 from reference epoch 20, schedule-scaled)",
        # Epoch-like variant overrides are written in reference-dataset epochs and
        # scaled in variant() (VARIANT_SCALED_SCALARS), like the shared schedule.
        "overrides": {"loss_weights": dict(HOI_ALIGNED), "train_data_window": 90, "ot_history_training_window": 60,
                      "ot_history_training_prob": 0.5, "ot_history_training_start_epoch": 20},
    },
    # Forearm-vs-object rotation as a predictor input (no architecture change):
    # B = R_obj^T R_forearm drives the in-segment bone change and was not derivable
    # from the old inputs (root-relative forearm orientation, no root orientation).
    # EgoIMU-style baseline: the IMUHOI VC (independent of HumanPose) + the
    # four-head IMUHOI ObjectTrans.  Its joint phase fine-tunes HumanPose and
    # freezes VC (train_object_trans.py::_joint_trainable_modules), the OT human
    # refine is off, and the losses come from `imuhoi_loss_weights`.  Run it with
    # --vc_arch imuhoi in its own experiment root.
    "vc_imuhoi_ot_4heads": {
        "label": "EgoIMU baseline: VC-IMUHOI + ObjectTrans IMUHOI_4heads (joint phase fine-tunes HP, VC frozen)",
        "ot_arch": "imuhoi_4heads",
        "overrides": {},
    },
    # The original three-branch IMUHOI ObjectTrans of the conference paper
    # (left FK, right FK, IMU integration; no static hold), same recipe as above.
    "vc_imuhoi_ot_3branch": {
        "label": "IMUHOI paper baseline: VC-IMUHOI + three-branch ObjectTrans IMUHOI (joint phase fine-tunes HP, VC frozen)",
        "ot_arch": "imuhoi",
        "overrides": {},
    },
    # Feedback-free predictor: the published object is no longer fed back into
    # the shared LSTM each frame (the IMUHOI bone heads have no such loop).  The
    # IMU/static candidates still integrate from the previous published position.
    "cur_hoi_nofb": {
        "label": "cur_hoi_p800 with object_trans_state_feedback off (feedback-free predictor)",
        "overrides": {"loss_weights": dict(HOI_ALIGNED), "object_trans_state_feedback": False},
    },
    # Fixed contact-driven fusion instead of the learned gate: ObjectTrans v2 is the
    # current pipeline (state feedback included) with weights
    # [p_L, p_R, p_move * P(no hand | move), 1 - p_move] from VelocityContact.
    "v2_fixed_fusion": {
        "label": "ObjectTrans v2: current predictor + fixed VC-driven fusion weights (no learned gate)",
        "ot_arch": "v2",
        "overrides": {},
    },
    "v2_armrot_biclose": {
        "label": "v2 fixed VC fusion + forearm/object rotation + explicit bimanual closure candidate",
        "ot_arch": "v2",
        "overrides": {
            "loss_weights": {"bimanual_hoi": 5.0, "bimanual_prior": 1.0},
            "object_trans_forearm_obj_rotation": True,
            "object_trans_bimanual_closure": True,
        },
    },
    # Plan A (2026-09-22): the state feedback reads an anchor recursion fused with
    # the fixed v2 contact weights (sits on the FK candidates during contact, no
    # IMU-integration drift); the published output keeps the learned gate.
    "cur_hoi_anchorfb": {
        "label": "cur_hoi_p800 + anchor-state feedback (object_trans_feedback_source=anchor)",
        "overrides": {"loss_weights": dict(HOI_ALIGNED), "object_trans_feedback_source": "anchor"},
    },
    "cur_hoi_anchorfb_bonecons": {
        "label": "anchorfb + bone-vector delta consistency loss (GT-delta-matched, contact frames)",
        "overrides": {"loss_weights": {**HOI_ALIGNED, "lhand_bone_cons": 1.0, "rhand_bone_cons": 1.0},
                      "object_trans_feedback_source": "anchor"},
    },
    "cur_hoi_armrot": {
        "label": "cur_hoi_p800 + forearm-vs-object rotation input (B, its per-frame delta, and B(0)^T B(t))",
        "overrides": {"loss_weights": dict(HOI_ALIGNED), "object_trans_forearm_obj_rotation": True},
    },
    "cur_hoi_biclose_armrot": {
        "label": "biclose + forearm-vs-object rotation input",
        "overrides": {"loss_weights": {**HOI_ALIGNED, "bimanual_hoi": 5.0, "bimanual_prior": 1.0},
                      "object_trans_bimanual_closure": True, "object_trans_forearm_obj_rotation": True},
    },
    "cur_hoi_biclose": {
        "label": "cur_hoi + explicit bimanual closure (state-free hand-pair-frame candidate, closure-aware 5-way gate)",
        "overrides": {"loss_weights": {**HOI_ALIGNED, "bimanual_hoi": 5.0, "bimanual_prior": 1.0}, "object_trans_bimanual_closure": True},
    },
    "cur_hoi_trust_perturb": {
        "label": "state trust + init corruption + rest-phase state perturbation (0.20 m)",
        "overrides": {
            "loss_weights": dict(TRUST_LOSSES),
            "object_trans_state_trust": True,
            "object_trans_init_noise_std": 0.20,
            "object_trans_init_noise_prob": 0.5,
            "object_trans_rest_perturb_std": 0.20,
            "object_trans_rest_perturb_prob": 0.5,
        },
    },
    # ---- 2026-09-14: event-anchored grasp latent (OT structure), see
    # scripts/run_omomo_cano_ot_variants.py for the recipe rationale.
    "cur_hoi_eag": {
        "label": "event-anchored grasp latent (onset anchoring, learned increments, release), no corruption",
        "overrides": {"loss_weights": dict(EAG_LOSSES), "object_trans_grasp_latent": True},
    },
    "cur_hoi_eag_mild": {
        "label": "grasp latent + mild window-init corruption (0.10 m on 25% of windows)",
        "overrides": {
            "loss_weights": dict(EAG_LOSSES),
            "object_trans_grasp_latent": True,
            "object_trans_init_noise_std": 0.10,
            "object_trans_init_noise_prob": 0.25,
        },
    },
    "cur_hoi_eag_res": {
        "label": "grasp latent with anchored bounded residual (no integration; bone = onset anchor + 0.10 m * tanh)",
        "overrides": {
            "loss_weights": dict(EAG_LOSSES),
            "object_trans_grasp_latent": True,
            "object_trans_grasp_increment_mode": "residual",
            "object_trans_grasp_residual_bound": 0.10,
        },
    },
    "cur_hoi_p800_hist": {
        "label": "baseline + rollout-aligned training (data window 90, own-rollout prefix 60)",
        "overrides": {"loss_weights": dict(HOI_ALIGNED), "train_data_window": 90, "ot_history_training_window": 60},
    },
    "cur_hoi_eag_hist": {
        "label": "grasp latent + rollout-aligned training (data window 90, own-rollout prefix 60)",
        "overrides": {"loss_weights": dict(EAG_LOSSES), "object_trans_grasp_latent": True,
                      "train_data_window": 90, "ot_history_training_window": 60},
    },
    "cur_hoi_eag_modes": {
        "label": "grasp latent + contact modes (per-hand rigidity scaling the increment, supervised by GT bone speed)",
        "overrides": {
            "loss_weights": {**EAG_LOSSES, "grasp_mode": 1.0},
            "object_trans_grasp_latent": True,
            "object_trans_grasp_modes": True,
        },
    },
}

# Epoch-like config keys.  Everything here is expressed in epochs and therefore
# has to follow the steps-per-epoch scaling; anything not listed (lr, gamma,
# weights, thresholds) is scale invariant.
SCALED_SCALARS = (
    "epochs",
    "joint_train_epochs",
    "input_mix_end_epoch",
    "eval_interval",
    "joint_train_eval_interval",
    "vc_velocity_stage_epochs",
    "vc_joint_stage_epochs",
    "ot_history_training_start_epoch",
)
# Epoch-like keys that may appear in a variant's overrides (applied after
# scale_schedule, so variant() scales them itself).
VARIANT_SCALED_SCALARS = ("ot_history_training_start_epoch",)
SCALED_LISTS = (
    "milestones",
    "joint_train_milestones",
    "vc_velocity_lr_milestones",
    "vc_contact_lr_milestones",
    "vc_endpoint_lr_milestones",
    "vc_endpoint_joint_lr_milestones",
    "vc_endpoint_finetune_lr_milestones",
)
# Early-stopping windows on the reference dataset (epochs), scaled like the rest.
REF_PATIENCE = 40
REF_JOINT_PATIENCE = 30

METRICS = (
    ("obj_trans_err_fusion_refine", "Object published (cm)"),
    ("hoi_err_fusion", "HOI published (cm)"),
    ("obj_jitter", "Object jitter (mm/frame^2)"),
    ("mpjpe", "MPJPE (cm)"),
    ("root_trans_err", "Root (cm)"),
    ("contact_f1_lhand", "Left F1"),
    ("contact_f1_rhand", "Right F1"),
)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


def deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def count_sequences(cfg: dict, dataset: str) -> int:
    entry = (cfg.get("datasets") or {}).get(dataset)
    if not entry:
        raise KeyError(f"config has no datasets.{dataset}")
    train_dir = PROJECT_ROOT / str(entry["train_path"])
    if not train_dir.is_dir():
        raise FileNotFoundError(train_dir)
    return sum(1 for path in train_dir.iterdir() if path.suffix == ".pt")


def steps_per_epoch(cfg: dict, dataset: str) -> int:
    """Optimizer steps per epoch: windows drawn per epoch // batch (drop_last)."""
    windows = count_sequences(cfg, dataset) * max(1, int(cfg.get("windows_per_sequence", 1)))
    return max(1, windows // int(cfg["batch_size"]))


def scale_schedule(cfg: dict, scale: float) -> dict:
    out = copy.deepcopy(cfg)
    for key in SCALED_SCALARS:
        if key in out and isinstance(out[key], (int, float)):
            out[key] = max(1, int(round(float(out[key]) * scale)))
    for key in SCALED_LISTS:
        if isinstance(out.get(key), (list, tuple)):
            out[key] = [max(1, int(round(float(value) * scale))) for value in out[key]]
    return out


def find_completed_stage(root: Path, prefix: str, suffix: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if path.is_dir() and path.name.startswith(prefix) and path.name.endswith(suffix):
            if (path / "final.pt").is_file() or (path / "final_object_trans.pt").is_file():
                return path
    return None


def find_interrupted_stage(root: Path, prefix: str, suffix: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if path.is_dir() and path.name.startswith(prefix) and path.name.endswith(suffix):
            if (path / "last.pt").is_file():
                return path
    return None


def checkpoint_epoch(path: Path) -> Optional[int]:
    """Epoch recorded in a checkpoint, or None if it cannot be read."""
    if not path.is_file():
        return None
    try:
        import torch  # local import: the orchestrator itself needs no GPU

        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    return int(epoch) if isinstance(epoch, (int, float)) else None


def find_object_trans_stage(root: Path, suffix: str) -> Optional[Path]:
    """Newest ObjectTrans stage directory for this variant, finished or not."""
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if path.is_dir() and path.name.endswith(suffix) and (path / "last.pt").is_file():
            return path
    return None


def find_joint_checkpoints(root: Path, suffix: str) -> Optional[Path]:
    """Stage dir of a FINISHED joint run, else None.

    The joint phase is finished only when the trainer wrote its final
    ObjectTrans checkpoint; best_* alone exist from the first joint evaluation
    on, and treating those as "done" evaluated half-trained joint runs after an
    interruption (2026-09-19 WSL restart).  Which of VC / HP is also written
    depends on the arch (current: VC fine-tuned, HP frozen; imuhoi: the other
    way round), so only ObjectTrans is required here.
    """
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if not (path.is_dir() and path.name.endswith(suffix)):
            continue
        if (path / "best_object_trans.pt").is_file() and (path / "final_object_trans.pt").is_file():
            return path
    return None


def read_metrics(eval_json: Path) -> dict:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    if isinstance(results, dict) and len(results) == 1:
        results = next(iter(results.values()))
    return {key: value for key, value in results.items() if isinstance(value, (int, float))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--run_tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_profile", choices=("current", "egoimu_gaussian"), default="current",
                        help="egoimu_gaussian: train/validation feature Gaussian std 0.1; clean independent eval.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=48)
    parser.add_argument("--max_parallel", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--schedule_scale",
        default="auto",
        help="auto = steps_per_epoch(reference)/steps_per_epoch(train); or an explicit float (1 = keep the config).",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Attempts per ObjectTrans stage.  Transient CUDA faults "
             "(\"unspecified launch failure\" / CUBLAS_STATUS_EXECUTION_FAILED) hit this box "
             "whenever two trainings share the GPU; each retry resumes from last.pt, so a "
             "fault costs at most the current epoch.",
    )
    parser.add_argument("--check_gpu_before_stage", action="store_true",
                        help="Print all GPU memory levels before each training/evaluation subprocess.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_hp", type=Path, default=None, help="reuse an existing HumanPose checkpoint")
    parser.add_argument("--skip_vc", type=Path, default=None, help="reuse an existing VelocityContact checkpoint")
    parser.add_argument("--set", dest="shared_overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Root-level config override applied to every stage of this experiment root "
                             "(YAML value), e.g. --set human_pose_smoothing_half_width=4")
    parser.add_argument("--vc_arch", default=VC_ARCH,
                        help="VelocityContact architecture for this run (v4, imuhoi, ...); it is trained from scratch, "
                             "so a different arch needs its own experiment root")
    parser.add_argument("--dataset", default="behave",
                        help="training/evaluation dataset key under config.datasets (behave, omomo, ...); the schedule "
                             "is scaled against REFERENCE_DATASET (scale 1 for omomo)")
    return parser.parse_args()


def configure_noise_profile(cfg: dict, profile: str, seed: int) -> tuple[list[str], list[str]]:
    """Resolve training and independent-eval noise without changing default runs."""
    if profile == "current":
        return ["--simulate_imu_noise"], ["--simulate_imu_noise", "--simulate_imu_noise_seed", str(seed)]
    if profile != "egoimu_gaussian":
        raise ValueError(f"Unknown noise profile: {profile}")
    cfg.update({"simulate_imu_noise": False, "validation_add_imu_noise": True,
                "imu_acc_noise_std": 0.1, "imu_rot_noise_std": 0.1,
                "obj_imu_acc_noise_std": 0.1, "obj_imu_rot_noise_std": 0.1})
    return ["--no_simulate_imu_noise"], []


def main() -> int:
    global DATASET, VC_ARCH
    args = parse_args()
    DATASET = str(args.dataset).strip().lower()
    VC_ARCH = str(args.vc_arch).strip().lower()
    unknown = [name for name in args.variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)}")
    cfg_path = (args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg).resolve()
    root = (args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root).resolve()
    if not SAGE_PYTHON.is_file() or not cfg_path.is_file():
        raise FileNotFoundError(f"{SAGE_PYTHON} / {cfg_path}")

    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    ref_steps = steps_per_epoch(base_cfg, REFERENCE_DATASET)
    train_steps = steps_per_epoch(base_cfg, DATASET)
    scale = (ref_steps / train_steps) if args.schedule_scale == "auto" else float(args.schedule_scale)

    shared_cfg = scale_schedule(base_cfg, scale)
    shared_cfg.update({
        "save_dir": str(root),
        "train_datasets": [DATASET],
        "velocity_contact_arch": VC_ARCH,
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": args.seed,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "pretrained_modules": {"human_pose": None, "velocity_contact": None, "object_trans": None},
        # WSL: the pin-memory thread segfaults on full-size batches; pinning only
        # affects the host->device copy speed.
        "dataloader_pin_memory": False,
    })
    shared_overrides = {}
    for item in args.shared_overrides:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        shared_overrides[key.strip()] = yaml.safe_load(value)
    shared_cfg.update(shared_overrides)
    training_noise_args, evaluation_noise_args = configure_noise_profile(shared_cfg, args.noise_profile, args.seed)

    # Early stopping, placed past the first lr decay (and, for the staged VC
    # trainer, past the last stage transition).
    patience = max(1, int(round(REF_PATIENCE * scale)))
    joint_patience = max(1, int(round(REF_JOINT_PATIENCE * scale)))
    first_milestone = min(shared_cfg["milestones"]) if shared_cfg.get("milestones") else 0
    ot_min_epochs = min(first_milestone + patience, shared_cfg["epochs"] - 1)
    vc_stage_end = int(shared_cfg.get("vc_velocity_stage_epochs", 0)) + int(shared_cfg.get("vc_joint_stage_epochs", 0))
    vc_min_epochs = min(vc_stage_end + patience, shared_cfg["epochs"] - 1)
    joint_min_epochs = min(joint_patience, shared_cfg["joint_train_epochs"] - 1)
    schedule = {
        "reference_dataset": REFERENCE_DATASET,
        "reference_steps_per_epoch": ref_steps,
        "train_steps_per_epoch": train_steps,
        "schedule_scale": round(scale, 4),
        "batch_size": shared_cfg["batch_size"],
        "windows_per_sequence": shared_cfg["windows_per_sequence"],
        "train_sequences": count_sequences(base_cfg, DATASET),
        "epochs": shared_cfg["epochs"],
        "milestones": shared_cfg["milestones"],
        "joint_train_epochs": shared_cfg["joint_train_epochs"],
        "joint_train_milestones": shared_cfg["joint_train_milestones"],
        "eval_interval": shared_cfg["eval_interval"],
        "joint_train_eval_interval": shared_cfg["joint_train_eval_interval"],
        "vc_velocity_stage_epochs": shared_cfg.get("vc_velocity_stage_epochs"),
        "vc_joint_stage_epochs": shared_cfg.get("vc_joint_stage_epochs"),
        "early_stopping": {
            "patience_epochs": patience,
            "hp_min_epochs": ot_min_epochs,
            "vc_min_epochs": vc_min_epochs,
            "ot_min_epochs": ot_min_epochs,
            "joint_patience_epochs": joint_patience,
            "joint_min_epochs": joint_min_epochs,
        },
    }

    if args.dry_run:
        print(json.dumps({"experiment_root": str(root), "schedule": schedule,
                          "variants": {name: VARIANTS[name] for name in args.variants}}, indent=2, ensure_ascii=False))
        return 0

    if root.exists() and not args.resume:
        raise FileExistsError(f"Experiment root already exists; use --resume or another path: {root}")
    root.mkdir(parents=True, exist_ok=True)

    # Stage-specific early-stopping settings (HP/VC read the same keys as the OT trainer).
    hp_cfg = copy.deepcopy(shared_cfg)
    hp_cfg.update({"early_stopping_patience_epochs": patience, "early_stopping_min_epochs": ot_min_epochs})
    vc_cfg = copy.deepcopy(shared_cfg)
    vc_cfg.update({"early_stopping_patience_epochs": patience, "early_stopping_min_epochs": vc_min_epochs})
    ot_common = {
        "early_stopping_patience_epochs": patience,
        "early_stopping_min_epochs": ot_min_epochs,
        "joint_train_early_stopping_patience_epochs": joint_patience,
        "joint_train_early_stopping_min_epochs": joint_min_epochs,
    }
    hp_cfg_path, vc_cfg_path = root / "hp_config.yaml", root / "vc_config.yaml"
    write_yaml(hp_cfg_path, hp_cfg)
    write_yaml(vc_cfg_path, vc_cfg)
    write_yaml(root / "shared_config.yaml", shared_cfg)

    manifest_path = root / "experiment_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.resume and manifest_path.is_file()
        else {
            "status": "running", "started_at": now(), "source_config": str(cfg_path), "dataset": DATASET,
            "seed": args.seed, "velocity_contact_arch": VC_ARCH, "schedule": schedule,
            "shared_overrides": shared_overrides,
            "variants": {name: VARIANTS[name] for name in args.variants},
            "gpu": args.gpu, "max_parallel": args.max_parallel, "stages": {}, "pipelines": {},
        }
    )
    manifest["schedule"] = schedule
    manifest["noise_profile"] = args.noise_profile
    write_json(manifest_path, manifest)

    root_log = root / "orchestrator.log"
    lock = threading.Lock()
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONUNBUFFERED": "1", "TMPDIR": "/tmp",
                "TMP": "/tmp", "TEMP": "/tmp", "IMUHOI_PERSISTENT_WORKERS": "0",
                "IMUHOI_SKIP_DATALOADER_PRIME": "1"})

    def log(message: str) -> None:
        with lock, root_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")
        print(f"[{now()}] {message}", flush=True)

    def commit(**fields) -> None:
        with lock:
            manifest.update(fields)
            write_json(manifest_path, manifest)

    def update(section: str, name: str, **fields) -> None:
        with lock:
            manifest.setdefault(section, {}).setdefault(name, {}).update(fields)
            write_json(manifest_path, manifest)

    def run_stage(command, log_path: Path) -> None:
        if args.check_gpu_before_stage:
            subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
                            "--format=csv,noheader"], check=True)
        log("RUN " + " ".join(str(part) for part in command))
        # Append: a retried stage must not overwrite the traceback of the failed attempt.
        with log_path.open("a", encoding="utf-8") as output:
            result = subprocess.run([str(part) for part in command], cwd=PROJECT_ROOT, env=env,
                                    stdout=output, stderr=subprocess.STDOUT)
        log(f"END rc={result.returncode} log={log_path}")
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); see {log_path}")

    # ---- stage 1: HumanPose on BEHAVE ------------------------------------------------
    if args.skip_hp is not None:
        hp_ckpt = args.skip_hp.resolve()
        update("stages", "human_pose", status="reused", checkpoint=str(hp_ckpt))
    else:
        hp_root = root / "hp"
        hp_root.mkdir(parents=True, exist_ok=True)
        hp_suffix = f"{DATASET}_hp_{args.run_tag}"
        update("stages", "human_pose", status="running", started_at=now())
        stage = find_completed_stage(hp_root, "human_pose", hp_suffix)
        if stage is None:
            command = [SAGE_PYTHON, PROJECT_ROOT / "train/rnn/train_human_pose.py", "--cfg", hp_cfg_path,
                       "--save_dir", hp_root, "--run_suffix", hp_suffix, "--seed", args.seed,
                       "--train_datasets", DATASET, "--num_workers", args.num_workers,
                       *training_noise_args, "--epochs", hp_cfg["epochs"]]
            interrupted = find_interrupted_stage(hp_root, "human_pose", hp_suffix)
            if interrupted is not None:
                command += ["--resume_dir", interrupted]
            run_stage(command, hp_root / "human_pose.log")
            stage = find_completed_stage(hp_root, "human_pose", hp_suffix)
        if stage is None:
            raise FileNotFoundError("BEHAVE HumanPose training produced no completed stage")
        hp_ckpt = stage / "best.pt"
        update("stages", "human_pose", status="complete", checkpoint=str(hp_ckpt), save_dir=str(stage), finished_at=now())

    # ---- stage 2: VelocityContact v4 on BEHAVE ---------------------------------------
    if args.skip_vc is not None:
        vc_ckpt = args.skip_vc.resolve()
        update("stages", "velocity_contact", status="reused", checkpoint=str(vc_ckpt))
    else:
        vc_root = root / f"vc_{VC_ARCH}"  # keeps two archs apart inside one root (legacy roots used "vc_v4")
        vc_root.mkdir(parents=True, exist_ok=True)
        vc_suffix = f"{DATASET}_vc_{VC_ARCH}_{args.run_tag}"
        update("stages", "velocity_contact", status="running", started_at=now())
        stage = find_completed_stage(vc_root, "velocity_contact", vc_suffix)
        if stage is None:
            command = [SAGE_PYTHON, PROJECT_ROOT / "train/rnn/train_velocity_contact.py", "--cfg", vc_cfg_path,
                       "--save_dir", vc_root, "--run_suffix", vc_suffix, "--seed", args.seed,
                       "--train_datasets", DATASET, "--num_workers", args.num_workers,
                       *training_noise_args, "--epochs", vc_cfg["epochs"],
                       "--velocity_contact_arch", VC_ARCH, "--hp_ckpt", hp_ckpt]
            interrupted = find_interrupted_stage(vc_root, "velocity_contact", vc_suffix)
            if interrupted is not None:
                command += ["--resume_dir", interrupted]
            run_stage(command, vc_root / "velocity_contact.log")
            stage = find_completed_stage(vc_root, "velocity_contact", vc_suffix)
        if stage is None:
            raise FileNotFoundError("BEHAVE VelocityContact training produced no completed stage")
        # best.pt is written on the first test-loss improvement; fall back to the
        # final checkpoint so an arch without a test-selectable best still works.
        vc_ckpt = stage / "best.pt" if (stage / "best.pt").is_file() else stage / "final.pt"
        update("stages", "velocity_contact", status="complete", checkpoint=str(vc_ckpt), save_dir=str(stage), finished_at=now())

    commit(reused_checkpoints={"hp_ckpt": str(hp_ckpt), "vc_ckpt": str(vc_ckpt)})

    # ---- stage 3: one ObjectTrans + joint run per variant, then full-sequence eval ----
    def evaluate(tag: str, cfg_file: Path, hp_path: Path, vc_path: Path, ot_path: Path, ot_arch: str,
                 out_json: Path, log_path: Path) -> None:
        if out_json.is_file():
            return
        run_stage([SAGE_PYTHON, PROJECT_ROOT / "eval_batch_online.py", "--config", cfg_file,
                   "--dataset", DATASET, "--hp_ckpt", hp_path, "--vc_ckpt", vc_path, "--ot_ckpt", ot_path,
                   "--velocity_contact_arch", VC_ARCH, "--object_trans_arch", ot_arch,
                   *evaluation_noise_args, "--seed", args.seed,
                   "--num_workers", args.num_workers, "--online_batch_size", args.eval_batch_size,
                   "--output_json", out_json], log_path)

    def variant(name: str) -> bool:
        spec = VARIANTS[name]
        pipeline_root = root / name
        pipeline_root.mkdir(parents=True, exist_ok=True)
        suffix = f"{DATASET}_ot_{name}_{args.run_tag}"
        ot_arch = str(spec.get("ot_arch", "current"))
        config = deep_update(copy.deepcopy(shared_cfg), {"save_dir": str(pipeline_root), "object_trans_arch": ot_arch})
        config.update(ot_common)
        config = deep_update(config, spec["overrides"])
        for key in VARIANT_SCALED_SCALARS:
            if isinstance(spec["overrides"].get(key), (int, float)):
                config[key] = max(0, int(round(float(spec["overrides"][key]) * scale)))
        variant_cfg = pipeline_root / "config.yaml"
        write_yaml(variant_cfg, config)
        update("pipelines", name, variant=name, label=spec["label"], overrides=spec["overrides"],
               config=str(variant_cfg), status="running", phase="object_trans_joint_train", started_at=now())
        try:
            joint = find_joint_checkpoints(pipeline_root, suffix)
            attempt = 0
            while joint is None:
                attempt += 1
                if attempt > max(1, int(args.max_attempts)):
                    raise RuntimeError(f"{name}: ObjectTrans stage failed {attempt - 1} times")
                command = [SAGE_PYTHON, PROJECT_ROOT / "train/rnn/train_object_trans.py", "--cfg", variant_cfg,
                           "--save_dir", pipeline_root, "--run_suffix", suffix, "--seed", args.seed,
                           "--train_datasets", DATASET, "--num_workers", args.num_workers, *training_noise_args,
                           "--epochs", config["epochs"], "--joint_train", "--velocity_contact_arch", VC_ARCH,
                           "--object_trans_arch", ot_arch, "--hp_ckpt", hp_ckpt, "--vc_ckpt", vc_ckpt]
                stage = find_object_trans_stage(pipeline_root, suffix)
                if stage is not None:
                    command += ["--resume_dir", stage]
                    # The ObjectTrans phase already ran to its end (early stop or
                    # budget) but the joint phase did not finish: capping --epochs
                    # at the resumed epoch makes the trainer skip straight to the
                    # joint phase instead of training the frozen-VC stage again.
                    if (stage / "final.pt").is_file():
                        last_epoch = checkpoint_epoch(stage / "last.pt")
                        if last_epoch is not None:
                            command[command.index("--epochs") + 1] = last_epoch + 1
                    log(f"{name}: attempt {attempt} resumes from {stage.name}")
                log_path = pipeline_root / (
                    "object_trans_joint_train.log" if attempt == 1 else f"object_trans_joint_train_attempt{attempt}.log"
                )
                try:
                    run_stage(command, log_path)
                except RuntimeError as exc:
                    log(f"{name}: attempt {attempt} failed ({exc})")
                    if attempt >= max(1, int(args.max_attempts)):
                        raise
                joint = find_joint_checkpoints(pipeline_root, suffix)
            stage_dir = joint
            # Whichever of VC / HP the joint phase fine-tuned is taken from the
            # stage dir; the other one stays the stage-2 / reused checkpoint.
            def ckpt_set(tag: str):
                pick = lambda name_, fallback: (stage_dir / name_) if (stage_dir / name_).is_file() else fallback  # noqa: E731
                return (pick(f"{tag}_human_pose.pt", hp_ckpt), pick(f"{tag}_velocity_contact.pt", vc_ckpt),
                        stage_dir / f"{tag}_object_trans.pt")
            best_hp, best_vc, best_ot = ckpt_set("best")
            update("pipelines", name, phase="eval", stage_dir=str(stage_dir), object_trans_arch=ot_arch,
                   velocity_contact_arch=VC_ARCH, joint_hp_ckpt=str(best_hp), joint_vc_ckpt=str(best_vc),
                   joint_ot_ckpt=str(best_ot))
            # Long-horizon metrics move with the checkpoint (the segment-window
            # score is a noisy proxy), so report both best and final.
            best_json = pipeline_root / f"eval_{DATASET}_best.json"
            evaluate(name, variant_cfg, best_hp, best_vc, best_ot, ot_arch, best_json,
                     pipeline_root / f"eval_{DATASET}_best.log")
            final_json = pipeline_root / f"eval_{DATASET}_final.json"
            final_hp, final_vc, final_ot = ckpt_set("final")
            if final_ot.is_file():
                evaluate(name, variant_cfg, final_hp, final_vc, final_ot, ot_arch, final_json,
                         pipeline_root / f"eval_{DATASET}_final.log")
            update("pipelines", name, status="complete", phase="complete", finished_at=now(),
                   metrics_best=read_metrics(best_json),
                   metrics_final=read_metrics(final_json) if final_json.is_file() else None)
            return True
        except Exception as exc:  # noqa: BLE001
            update("pipelines", name, status="failed", phase="failed", error=str(exc), finished_at=now())
            log(f"FAILED {name}: {exc}")
            return False

    results = {}
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(variant, name): name for name in args.variants}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    commit(results=results, finished_at=now(), status="complete" if all(results.values()) else "failed")

    header = "| variant | ckpt | " + " | ".join(label for _, label in METRICS) + " |"
    lines = [header, "|---|---|" + "---|" * len(METRICS)]
    for name in args.variants:
        entry = manifest["pipelines"].get(name, {})
        for tag, key in (("best", "metrics_best"), ("final", "metrics_final")):
            metrics = entry.get(key) or {}
            if metrics:
                lines.append(f"| {name} | {tag} | " + " | ".join(
                    "" if metrics.get(metric) is None else f"{metrics[metric]:.4f}" for metric, _ in METRICS) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
