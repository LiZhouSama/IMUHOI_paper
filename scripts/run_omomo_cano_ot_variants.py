#!/usr/bin/env python3
"""Train/eval ObjectTrans variants on the canonical OMOMO run (shared HP + VC-v4).

Reuses the canonical baseline `outputs/IMUHOI++/RNN_cano/omomo_cano_20260907`:
the shared HumanPose checkpoint and the VC-v4 checkpoint.  Each variant names
an ObjectTrans arch plus config overrides (nested keys such as
``loss_weights.obj_trans`` are merged, not replaced), is trained with
``--joint_train`` (VC fine-tuned alongside, as in the baseline), and evaluated
with ``eval_batch_online.py`` under the baseline protocol (window 60,
lookahead 10, seed 42, simulated IMU noise, 1173 sequences).

    python scripts/run_omomo_cano_ot_variants.py --experiment_root outputs/... \
        --variants v3_gate cur_gate --max_parallel 2
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

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGE_PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
DEFAULT_CFG = PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml"
BASELINE_ROOT = PROJECT_ROOT / "outputs/IMUHOI++/RNN_cano/omomo_cano_20260907"
HP_CKPT = BASELINE_ROOT / "hp/human_pose_09072119_omomo_cano/best.pt"
VC_CKPT = BASELINE_ROOT / "vc_v4/velocity_contact_v4_09072238_omomo_cano_vc_v4/best.pt"
VC_ARCH = "v4"

GATE_ON = {
    "object_trans_static_gate": "both",
    "object_trans_static_gate_pmove_threshold": 0.5,
}
# HOI-aligned supervision: the official metric is the fused object relative to
# the hand on GT contact frames, so supervise exactly that and stop asking the
# bone to chase the (hand-error-compensated) absolute object position.
HOI_ALIGNED = {
    "loss_weights": {
        "obj_trans": 1.0,
        "hoi_fused_l": 10.0,
        "hoi_fused_r": 10.0,
    }
}
VARIANTS = {
    "legacy_mesh": {
        "arch": "legacy_mesh",
        "label": "ot-legacy_mesh + FP32 joint VC-v4 fine-tuning, baseline losses",
        "overrides": {"joint_train_amp_enabled": False},
    },
    "v3_gate": {"arch": "v3", "label": "ot-v3 + static gate both", "overrides": {**GATE_ON}},
    "cur_gate": {"arch": "current", "label": "ot-current + static gate both", "overrides": {**GATE_ON}},
    "v3_gate_hoi": {
        "arch": "v3",
        "label": "ot-v3 + static gate + HOI-aligned fused loss (obj_trans 5->1)",
        "overrides": {**GATE_ON, **HOI_ALIGNED},
    },
    "cur_gate_hoi": {
        "arch": "current",
        "label": "ot-current + static gate + HOI-aligned fused loss (obj_trans 5->1)",
        "overrides": {**GATE_ON, **HOI_ALIGNED},
    },
    "cur_hoi": {
        "arch": "current",
        "label": "ot-current + HOI-aligned fused loss (obj_trans 5->1), no static gate",
        "overrides": {**HOI_ALIGNED},
    },
    "v3_gate_hoi_nofb": {
        "arch": "v3",
        "label": "ot-v3 + static gate + HOI-aligned loss + no state feedback",
        "overrides": {**GATE_ON, **HOI_ALIGNED, "object_trans_state_feedback": False},
    },
    # Exposure-bias probes: the online protocol re-initialises every window from
    # the model's own (drifted) published object, but training always starts a
    # window from the GT object.  Inject Gaussian drift into the window init
    # (init geometry + initial feedback state) on half of the windows.
    "cur_hoi_drift15": {
        "arch": "current",
        "label": "cur_hoi + window-init drift N(0, 0.15 m) on 50% of windows",
        "overrides": {**HOI_ALIGNED, "object_trans_init_noise_std": 0.15, "object_trans_init_noise_prob": 0.5},
    },
    "cur_hoi_drift30": {
        "arch": "current",
        "label": "cur_hoi + window-init drift N(0, 0.30 m) on 50% of windows",
        "overrides": {**HOI_ALIGNED, "object_trans_init_noise_std": 0.30, "object_trans_init_noise_prob": 0.5},
    },
    # ---- 2026-09-10: state trust (object-state confidence) -------------------
    # cur_hoi_p800   : the cur_hoi recipe under the source config's training
    #                  protocol (--train_protocol config: batch 800 x 4 windows,
    #                  early stopping) -- the baseline for that protocol.
    # cur_hoi_trust  : + state-free prior FK head, consistency features,
    #                  supervised state confidence gating the state-derived
    #                  inputs, and observable window-init corruption (0.2 m on
    #                  half of the windows) so the confidence has negatives.
    # cur_hoi_trust_noaug : the same module without the corruption (ablation).
    "cur_hoi_p800": {
        "arch": "current",
        "label": "cur_hoi recipe under the config protocol (800 x 4 windows, early stopping)",
        "overrides": {**HOI_ALIGNED},
    },
    "cur_hoi_trust": {
        "arch": "current",
        "label": "cur_hoi + state trust (prior head, consistency features, supervised confidence) + init corruption 0.2 m",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0, "state_conf": 1.0},
            "object_trans_state_trust": True,
            "object_trans_init_noise_std": 0.20,
            "object_trans_init_noise_prob": 0.5,
        },
    },
    "cur_hoi_trust_perturb": {
        "arch": "current",
        "label": "cur_hoi_trust + observable rest-phase state perturbation (0.2 m at a random GT-rest frame, 50% of windows)",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0, "state_conf": 1.0},
            "object_trans_state_trust": True,
            "object_trans_init_noise_std": 0.20,
            "object_trans_init_noise_prob": 0.5,
            "object_trans_rest_perturb_std": 0.20,
            "object_trans_rest_perturb_prob": 0.5,
        },
    },
    "cur_hoi_trust_perturb40": {
        "arch": "current",
        "label": "cur_hoi_trust + rest-phase state perturbation 0.4 m (separable from the prior's own error) + state_conf weight 2, threshold 0.3 m",
        "overrides": {
            # The state-free prior is ~20 cm off on OMOMO, so a 10 cm confidence
            # threshold is below what the innovation can resolve; ask "is the
            # state grossly wrong (>30 cm)?" instead, with 40 cm perturbations.
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0, "state_conf": 2.0, "state_conf_threshold": 0.30},
            "object_trans_state_trust": True,
            "object_trans_init_noise_std": 0.20,
            "object_trans_init_noise_prob": 0.5,
            "object_trans_rest_perturb_std": 0.40,
            "object_trans_rest_perturb_prob": 0.5,
        },
    },
    "cur_hoi_trust_mild": {
        "arch": "current",
        "label": "cur_hoi_trust with mild init corruption (0.10 m on 25% of windows): operating point closer to the OMOMO baseline",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0, "state_conf": 1.0},
            "object_trans_state_trust": True,
            "object_trans_init_noise_std": 0.10,
            "object_trans_init_noise_prob": 0.25,
        },
    },
    "cur_hoi_trust_noaug": {
        "arch": "current",
        "label": "cur_hoi + state trust module, no init corruption",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0, "state_conf": 1.0},
            "object_trans_state_trust": True,
        },
    },
    # ---- 2026-09-14: event-anchored grasp latent (OT structure) -------------
    # The hand->object bone becomes a segment state: anchored at contact
    # onsets by fusing the observed geometry with a state-free posture prior,
    # propagated by learned increments, released to the prior.  The published
    # object enters the grasp geometry only at onsets (model/rnn/grasp_state.py).
    "cur_hoi_eag": {
        "arch": "current",
        "label": "cur_hoi + event-anchored grasp latent (onset anchoring, learned increments, release), no corruption",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0},
            "object_trans_grasp_latent": True,
        },
    },
    "cur_hoi_eag_mild": {
        "arch": "current",
        "label": "cur_hoi_eag + mild window-init corruption (0.10 m on 25% of windows) so the onset fusion sees wrong states",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0},
            "object_trans_grasp_latent": True,
            "object_trans_init_noise_std": 0.10,
            "object_trans_init_noise_prob": 0.25,
        },
    },
    # 2026-09-15: the integrating latent random-walks on streams longer than
    # the training window (OMOMO FK HOI 8.1 -> 12.8 cm over a segment vs
    # 7.6 -> 10.0 for the baseline); residual mode keeps the onset anchor
    # constant and predicts a bounded, non-integrated deviation from it.
    "cur_hoi_eag_res": {
        "arch": "current",
        "label": "cur_hoi_eag with anchored bounded residual (no integration; bone = onset anchor + 0.10 m * tanh)",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0},
            "object_trans_grasp_latent": True,
            "object_trans_grasp_increment_mode": "residual",
            "object_trans_grasp_residual_bound": 0.10,
        },
    },
    # ---- 2026-09-15: rollout-aligned (online-protocol-aligned) training ----
    # 90-frame data windows; the first 60 frames are predicted without grad
    # (the model's own rollout), the next window is trained conditioned on
    # that published prefix (train/rnn/ot_history_training.py), so the trust
    # decision (observed geometry vs prior, feedback vs FK) sees real drift
    # instead of GT-initialised windows.  Same 800x4 + joint + early-stopping
    # protocol as cur_hoi_p800 / cur_hoi_eag.
    "cur_hoi_p800_hist": {
        "arch": "current",
        "label": "cur_hoi_p800 + rollout-aligned training (data window 90, own-rollout prefix 60)",
        "overrides": {**HOI_ALIGNED, "train_data_window": 90, "ot_history_training_window": 60},
    },
    "cur_hoi_eag_hist": {
        "arch": "current",
        "label": "cur_hoi_eag + rollout-aligned training (data window 90, own-rollout prefix 60)",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0},
            "object_trans_grasp_latent": True,
            "train_data_window": 90,
            "ot_history_training_window": 60,
        },
    },
    "cur_hoi_eag_modes": {
        "arch": "current",
        # residual mode lost on both datasets (OMOMO 11.20, BEHAVE 38.3), so
        # the modes variant scales the INTEGRATED increment of cur_hoi_eag.
        "label": "cur_hoi_eag + contact modes (per-hand rigidity scaling the increment, supervised by GT bone speed)",
        "overrides": {
            "loss_weights": {**HOI_ALIGNED["loss_weights"], "prior_hoi_l": 5.0, "prior_hoi_r": 5.0, "grasp_mode": 1.0},
            "object_trans_grasp_latent": True,
            "object_trans_grasp_modes": True,
        },
    },
}

METRICS = (
    ("obj_trans_err_fusion_refine", "Object published (cm)"),
    ("obj_trans_err_fusion_raw", "Object raw (cm)"),
    ("hoi_err_fusion", "HOI published (cm)"),
    ("obj_jitter", "Object jitter (mm/frame^2)"),
    ("mpjpe", "MPJPE (cm)"),
    ("root_trans_err", "Root (cm)"),
    ("contact_f1_lhand", "Left F1"),
    ("contact_f1_rhand", "Right F1"),
    ("contact_f1_obj", "Object F1"),
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
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
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


def find_joint_checkpoints(root: Path, suffix: str):
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if not (path.is_dir() and path.name.endswith(suffix)):
            continue
        vc = path / "best_velocity_contact.pt"
        ot = path / "best_object_trans.pt"
        complete = all((path / filename).is_file() for filename in (
            "final_object_trans.pt", "final_velocity_contact.pt",
        ))
        if complete and vc.is_file() and ot.is_file():
            return vc, ot
    return None


def find_ot_resume_dir(root: Path, suffix: str):
    if root.is_dir():
        for path in sorted(root.iterdir(), reverse=True):
            if path.is_dir() and path.name.endswith(suffix) and (path / "last.pt").is_file():
                return path
    return None


def checkpoint_epoch(path: Path):
    """Epoch stored in a trainer checkpoint (None if unreadable)."""
    try:
        import torch  # local import: the orchestrator itself does not need torch otherwise
        payload = torch.load(path, map_location="cpu")
    except Exception:  # noqa: BLE001
        return None
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    return int(epoch) if isinstance(epoch, (int, float)) else None


def read_metrics(eval_json: Path) -> dict:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    if isinstance(results, dict) and len(results) == 1:
        results = next(iter(results.values()))
    return {key: json_safe(results.get(key)) for key, _ in METRICS}


def format_summary(rows: dict) -> str:
    best = {}
    for key, _ in METRICS:
        values = [metrics[key] for metrics in rows.values() if metrics.get(key) is not None]
        if len(values) > 1:
            best[key] = (max if key.startswith("contact_f1_") else min)(values)
    lines = ["| variant | " + " | ".join(label for _, label in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for name, metrics in rows.items():
        cells = []
        for key, _ in METRICS:
            value = metrics.get(key)
            cell = "" if value is None else f"{value:.4f}"
            if value is not None and value == best.get(key):
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--experiment_root", type=Path, required=True)
    parser.add_argument("--baseline_eval", type=Path, default=None)
    parser.add_argument("--run_tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cpu_threads", type=int, default=6)
    parser.add_argument("--ot_epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=48)
    parser.add_argument("--max_parallel", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument(
        "--train_protocol",
        choices=("pinned", "config"),
        default="pinned",
        help="pinned: batch 200 x 1 window (20260907 baseline protocol); config: keep the source config's batch_size / windows_per_sequence.",
    )
    parser.add_argument("--early_stopping_patience_epochs", type=int, default=0, help="0 = off; applies to both phases")
    parser.add_argument("--early_stopping_min_epochs", type=int, default=0)
    parser.add_argument("--joint_early_stopping_patience_epochs", type=int, default=None, help="joint phase override (default: same as OT phase)")
    parser.add_argument("--joint_early_stopping_min_epochs", type=int, default=None, help="joint phase override (default: same as OT phase)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_attempts", type=int, default=3,
                        help="retry a crashed ObjectTrans/joint stage from its last.pt this many times (transient CUDA faults)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.cpu_threads < 1:
        parser.error("--cpu_threads must be positive")
    unknown = [name for name in args.variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)}")
    cfg_path = (args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg).resolve()
    root = (args.experiment_root if args.experiment_root.is_absolute() else PROJECT_ROOT / args.experiment_root).resolve()
    for path in (SAGE_PYTHON, cfg_path, HP_CKPT, VC_CKPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_metrics = None
    if args.baseline_eval is not None:
        args.baseline_eval = (PROJECT_ROOT / args.baseline_eval).resolve()
        baseline_metrics = read_metrics(args.baseline_eval)

    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if list(base_cfg.get("train_datasets", [])) != ["omomo"]:
        raise ValueError("The source config must select train_datasets: [omomo].")
    shared_cfg = copy.deepcopy(base_cfg)
    shared_cfg.update({
        "save_dir": str(root),
        "train_datasets": ["omomo"],
        "velocity_contact_arch": VC_ARCH,
        "simulate_imu_noise": True,
        "simulate_imu_noise_seed": args.seed,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "pretrained_modules": {"human_pose": None, "velocity_contact": None, "object_trans": None},
        # WSL: the DataLoader pin-memory thread segfaults (SIGSEGV inside
        # Tensor.pin_memory) on the full-size batches of this run; the same
        # command trains fine with pinning off, and pinning only affects H2D
        # copy speed.
        "dataloader_pin_memory": False,
        # Match the 20260907 canonical baseline protocol (its saved configs
        # trained with batch 200 x 1 window per sequence); the source config
        # was raised to 800 x 4 on 2026-09-08, which quadruples the work per
        # epoch and would make the 300+100-epoch comparison unequal.
    })
    if args.train_protocol == "pinned":
        shared_cfg.update({"batch_size": 200, "windows_per_sequence": 1})
    if args.device == "cpu":
        shared_cfg.update({"device": "cpu", "use_multi_gpu": False, "gpus": []})
    shared_cfg.update({
        "early_stopping_patience_epochs": int(args.early_stopping_patience_epochs),
        "early_stopping_min_epochs": int(args.early_stopping_min_epochs),
    })
    if args.joint_early_stopping_patience_epochs is not None:
        shared_cfg["joint_train_early_stopping_patience_epochs"] = int(args.joint_early_stopping_patience_epochs)
    if args.joint_early_stopping_min_epochs is not None:
        shared_cfg["joint_train_early_stopping_min_epochs"] = int(args.joint_early_stopping_min_epochs)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu) if args.device == "cuda" else "", "PYTHONUNBUFFERED": "1",
                "TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp", "IMUHOI_PERSISTENT_WORKERS": "0",
                "IMUHOI_SKIP_DATALOADER_PRIME": "1"})
    if args.device == "cpu":
        env.update({key: str(args.cpu_threads) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    if args.dry_run:
        print(json.dumps({"experiment_root": str(root), "variants": {n: VARIANTS[n] for n in args.variants},
                          "hp_ckpt": str(HP_CKPT), "vc_ckpt": str(VC_CKPT), "ot_epochs": args.ot_epochs,
                          "execution_device": args.device, "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
                          "cpu_threads": args.cpu_threads if args.device == "cpu" else None,
                          "joint_train_epochs": base_cfg.get("joint_train_epochs")}, indent=2))
        return 0
    if root.exists() and not args.resume:
        raise FileExistsError(f"Experiment root already exists; use --resume or another path: {root}")
    root.mkdir(parents=True, exist_ok=True)
    shared_cfg_path = root / "shared_config.yaml"
    write_yaml(shared_cfg_path, shared_cfg)
    manifest_path = root / "experiment_manifest.json"
    root_log = root / "orchestrator.log"
    lock = threading.Lock()
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if args.resume and manifest_path.is_file() else {
            "status": "running", "started_at": now(), "source_config": str(cfg_path),
            "shared_config": str(shared_cfg_path), "dataset": "omomo", "seed": args.seed,
            "velocity_contact_arch": VC_ARCH, "baseline_root": str(BASELINE_ROOT),
            "reused_checkpoints": {"hp_ckpt": str(HP_CKPT), "vc_ckpt": str(VC_CKPT)},
            "variants": {n: VARIANTS[n] for n in args.variants}, "ot_epochs": args.ot_epochs,
            "gpu": args.gpu, "max_parallel": args.max_parallel, "pipelines": {},
            "train_protocol": args.train_protocol,
            "batch_size": shared_cfg.get("batch_size"), "windows_per_sequence": shared_cfg.get("windows_per_sequence"),
            "early_stopping_patience_epochs": int(args.early_stopping_patience_epochs),
            "early_stopping_min_epochs": int(args.early_stopping_min_epochs),
        }
    )
    manifest["status"] = "running"
    manifest["execution_device"] = args.device
    manifest["cpu_threads"] = args.cpu_threads if args.device == "cpu" else None
    manifest["variants"] = {name: VARIANTS[name] for name in args.variants}
    if baseline_metrics is not None:
        manifest["baseline_eval_json"] = str(args.baseline_eval)
        manifest["baseline_metrics"] = baseline_metrics
    if args.resume:
        manifest["resumed_at"] = now()
        manifest.pop("finished_at", None)
        manifest.pop("results", None)
    write_json(manifest_path, manifest)

    def log(message: str) -> None:
        with lock, root_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")

    def update_pipeline(name: str, **fields) -> None:
        with lock:
            if fields.get("status") == "running":
                pipeline = manifest["pipelines"].setdefault(name, {})
                pipeline.pop("error", None)
                pipeline.pop("finished_at", None)
            manifest["pipelines"].setdefault(name, {}).update(fields)
            write_json(manifest_path, manifest)

    def run_stage(command, log_path: Path) -> None:
        log("RUN " + " ".join(command))
        print(f"[{now()}] RUN {' '.join(command)}", flush=True)
        subprocess.run([
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], cwd=PROJECT_ROOT, env=env, check=True)
        with log_path.open("a", encoding="utf-8") as output:
            output.write(f"\n[{now()}] RUN {' '.join(command)}\n")
            output.flush()
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        log(f"END returncode={result.returncode} log={log_path}")
        print(f"[{now()}] END rc={result.returncode} log={log_path}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); see {log_path}")

    def variant(name: str) -> bool:
        spec = VARIANTS[name]
        arch = spec["arch"]
        pipeline_root = root / name
        pipeline_root.mkdir(parents=True, exist_ok=True)
        suffix = f"omomo_cano_ot_{arch}_{name}_{args.run_tag}"
        config = deep_update(copy.deepcopy(shared_cfg), {"save_dir": str(pipeline_root), "object_trans_arch": arch})
        config = deep_update(config, spec["overrides"])
        variant_cfg = pipeline_root / "config.yaml"
        write_yaml(variant_cfg, config)
        update_pipeline(name, variant=name, label=spec["label"], overrides=spec["overrides"], object_trans_arch=arch,
                        velocity_contact_arch=VC_ARCH, config=str(variant_cfg), status="running", phase="object_trans_joint_train")
        try:
            joint = find_joint_checkpoints(pipeline_root, suffix)
            attempt = 0
            while joint is None:
                attempt += 1
                if attempt > max(1, int(args.max_attempts)):
                    raise RuntimeError(f"{name}: ObjectTrans stage failed {attempt - 1} times")
                cmd = [str(SAGE_PYTHON), str(PROJECT_ROOT / "train/rnn/train_object_trans.py"), "--cfg", str(variant_cfg),
                       "--save_dir", str(pipeline_root), "--run_suffix", suffix, "--seed", str(args.seed),
                       "--train_datasets", "omomo", "--num_workers", str(args.num_workers), "--simulate_imu_noise",
                       "--epochs", str(args.ot_epochs), "--joint_train", "--velocity_contact_arch", VC_ARCH,
                       "--object_trans_arch", arch, "--hp_ckpt", str(HP_CKPT), "--vc_ckpt", str(VC_CKPT)]
                # Resume from last.pt when asked to, or after a crash (transient
                # CUDA faults under concurrent GPU jobs).
                resume_dir = find_ot_resume_dir(pipeline_root, suffix) if (args.resume or attempt > 1) else None
                if resume_dir is not None:
                    cmd += ["--resume_dir", str(resume_dir)]
                    if (resume_dir / "final.pt").is_file():
                        # The ObjectTrans phase already ended (early stop or budget)
                        # but the joint phase did not finish: cap --epochs at the
                        # resumed epoch so the trainer skips straight to joint.
                        last_epoch = checkpoint_epoch(resume_dir / "last.pt")
                        if last_epoch is not None:
                            cmd[cmd.index("--epochs") + 1] = str(last_epoch + 1)
                    update_pipeline(
                        name, resume_dir=str(resume_dir), attempt=attempt,
                        joint_restarted=not (resume_dir / "last_joint.pt").is_file(),
                        joint_amp_enabled=config.get("joint_train_amp_enabled", config.get("amp_enabled", True)),
                    )
                if spec["overrides"].get("object_trans_state_feedback") is False:
                    cmd += ["--object_trans_state_feedback", "none"]
                log_path = pipeline_root / (
                    "object_trans_joint_train.log" if attempt == 1 else f"object_trans_joint_train_attempt{attempt}.log"
                )
                try:
                    run_stage(cmd, log_path)
                except RuntimeError as exc:
                    log(f"{name}: attempt {attempt} failed ({exc})")
                    if attempt >= max(1, int(args.max_attempts)):
                        raise
                joint = find_joint_checkpoints(pipeline_root, suffix)
            joint_vc, joint_ot = joint
            update_pipeline(name, phase="eval", joint_vc_ckpt=str(joint_vc), joint_ot_ckpt=str(joint_ot))
            eval_json = pipeline_root / "eval_batch_online.json"
            if not eval_json.is_file():
                run_stage([str(SAGE_PYTHON), str(PROJECT_ROOT / "eval_batch_online.py"), "--config", str(variant_cfg),
                           "--dataset", "omomo", "--hp_ckpt", str(HP_CKPT), "--vc_ckpt", str(joint_vc), "--ot_ckpt", str(joint_ot),
                           "--velocity_contact_arch", VC_ARCH, "--object_trans_arch", arch, "--simulate_imu_noise",
                           "--simulate_imu_noise_seed", str(args.seed), "--seed", str(args.seed), "--num_workers", str(args.num_workers),
                           "--online_batch_size", str(args.eval_batch_size), "--output_json", str(eval_json)],
                          pipeline_root / "eval_batch_online.log")
            update_pipeline(name, status="complete", phase="complete", eval_json=str(eval_json), metrics=read_metrics(eval_json), finished_at=now())
            return True
        except Exception as exc:  # noqa: BLE001
            update_pipeline(name, status="failed", phase="failed", error=str(exc), finished_at=now())
            log(f"FAILED {name}: {exc}")
            return False

    results = {}
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        futures = {pool.submit(variant, name): name for name in args.variants}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    manifest["results"] = results
    manifest["finished_at"] = now()
    manifest["status"] = "complete" if all(results.values()) else "failed"
    write_json(manifest_path, manifest)
    # summary table
    rows = {}
    if baseline_metrics is not None:
        rows["ot_current_baseline"] = baseline_metrics
    for name in args.variants:
        rows[name] = manifest["pipelines"].get(name, {}).get("metrics", {})
    summary = format_summary(rows)
    if args.device == "cpu":
        summary += f"\nThis execution used CPU ({args.cpu_threads} threads); resumed runs retain earlier checkpoint training.\n"
    for name in args.variants:
        if VARIANTS[name]["overrides"].get("joint_train_amp_enabled") is False:
            summary += f"\n{name}: joint training uses FP32 (AMP disabled). Historical baseline precision/backend may differ.\n"
    (root / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
