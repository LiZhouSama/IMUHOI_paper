#!/usr/bin/env python3
"""Train ObjectTrans variants on OMOMO and compare them against the archived ot-v2.

Everything upstream of ObjectTrans is frozen and reused from the archived
`omomo_baseline_v4_ot_v2_shared_hp_20260826` run: the shared HumanPose
checkpoint and the VC-v4 checkpoint.  Only the OT stage (with `--joint_train`,
so VC is fine-tuned alongside) is retrained, which is what isolates the OT
change.

Each variant names its ObjectTrans arch and a set of config overrides (see
`VARIANTS`).  Eval-only variants re-evaluate the archived ot-v2 checkpoints
under a different config (for example with the object-static gate switched
on) so a single eval-time change can be attributed without retraining.

The ot-v2 reference is *re-evaluated* from its archived checkpoints rather than
quoted from the archived JSON, so all rows come from the same eval code and
the same test directory listing.
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
BASELINE_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN2/omomo_baseline_v4_ot_v2_shared_hp_20260826"
)
VC_ARCH = "v4"

# name -> {"arch": ObjectTrans arch, "label": summary label,
#          "train": True to train (False = eval archived ot-v2 ckpts under this config),
#          "overrides": config overrides}
GATE_ON = {
    "object_trans_static_gate": "both",
    "object_trans_static_gate_pmove_threshold": 0.5,
}
INIT_NOISE = {"object_trans_init_noise_std": 0.06, "object_trans_init_noise_prob": 0.5}
VARIANTS = {
    "v3_gate_noise": {
        "arch": "v3",
        "label": "ot-v3 (soft latch + Rrel) + static gate + init noise",
        "train": True,
        "overrides": {**GATE_ON, **INIT_NOISE},
    },
    "v3_gate": {
        "arch": "v3",
        "label": "ot-v3 (soft latch + Rrel) + static gate",
        "train": True,
        "overrides": {**GATE_ON},
    },
    "v2_gate_noise": {
        "arch": "v2",
        "label": "ot-v2 retrained + static gate + init noise",
        "train": True,
        "overrides": {**GATE_ON, **INIT_NOISE},
    },
    "v2_gate_eval": {
        "arch": "v2",
        "label": "ot-v2 archived ckpt + static gate (eval only)",
        "train": False,
        "overrides": {**GATE_ON},
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
    tmp.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.replace(tmp, path)


def find_best(root: Path, prefix: str, suffix: str) -> Path | None:
    if not root.is_dir():
        return None
    matches = [
        path / "best.pt"
        for path in sorted(root.iterdir())
        if path.is_dir()
        and path.name.startswith(prefix)
        and path.name.endswith(suffix)
        and (path / "best.pt").is_file()
    ]
    return matches[-1] if matches else None


def find_joint_checkpoints(root: Path, suffix: str) -> tuple[Path, Path] | None:
    if not root.is_dir():
        return None
    for path in sorted(root.iterdir(), reverse=True):
        if not (path.is_dir() and path.name.endswith(suffix)):
            continue
        vc = path / "best_velocity_contact.pt"
        ot = path / "best_object_trans.pt"
        if vc.is_file() and ot.is_file():
            return vc, ot
    return None


def read_metrics(eval_json: Path) -> dict:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    if isinstance(results, dict) and len(results) == 1:
        results = next(iter(results.values()))
    return {key: json_safe(results.get(key)) for key, _ in METRICS}


def resolve_baseline_checkpoints() -> dict:
    manifest = json.loads(
        (BASELINE_ROOT / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    pipeline = manifest["pipelines"][VC_ARCH]
    paths = {
        "hp_ckpt": Path(manifest["hp_ckpt"]),
        "vc_ckpt": Path(pipeline["vc_ckpt"]),
        "v2_joint_vc_ckpt": Path(pipeline["joint_vc_ckpt"]),
        "v2_joint_ot_ckpt": Path(pipeline["joint_ot_ckpt"]),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument(
        "--experiment_root",
        type=Path,
        default=Path("outputs/IMUHOI++/RNN2/omomo_ot_v3_ablation"),
    )
    parser.add_argument("--run_tag", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ot_epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_parallel", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument(
        "--skip_v2_reeval",
        action="store_true",
        help="Quote the archived ot-v2 numbers instead of re-running its eval.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    unknown = [name for name in args.variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)}")
    cfg_path = (
        args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg
    ).resolve()
    root = (
        args.experiment_root
        if args.experiment_root.is_absolute()
        else PROJECT_ROOT / args.experiment_root
    ).resolve()
    for path in (SAGE_PYTHON, cfg_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if list(base_cfg.get("train_datasets", [])) != ["omomo"]:
        raise ValueError("The source config must select train_datasets: [omomo].")
    checkpoints = resolve_baseline_checkpoints()

    shared_cfg = copy.deepcopy(base_cfg)
    shared_cfg.update(
        {
            "save_dir": str(root),
            "train_datasets": ["omomo"],
            "velocity_contact_arch": VC_ARCH,
            "simulate_imu_noise": True,
            "simulate_imu_noise_seed": args.seed,
            "seed": args.seed,
            "num_workers": args.num_workers,
            "pretrained_modules": {
                "human_pose": None,
                "velocity_contact": None,
                "object_trans": None,
            },
        }
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_root": str(root),
                    "variants": {name: VARIANTS[name] for name in args.variants},
                    "velocity_contact_arch": VC_ARCH,
                    "reused": {k: str(v) for k, v in checkpoints.items()},
                    "ot_epochs": args.ot_epochs,
                    "joint_train_epochs": base_cfg.get("joint_train_epochs"),
                    "max_parallel": args.max_parallel,
                    "skip_v2_reeval": args.skip_v2_reeval,
                },
                indent=2,
            )
        )
        return 0

    if root.exists() and not args.resume:
        raise FileExistsError(
            f"Experiment root already exists; use --resume or another path: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)
    shared_cfg_path = root / "shared_config.yaml"
    write_yaml(shared_cfg_path, shared_cfg)
    manifest_path = root / "experiment_manifest.json"
    root_log = root / "orchestrator.log"
    lock = threading.Lock()
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.resume and manifest_path.is_file()
        else {
            "status": "running",
            "started_at": now(),
            "source_config": str(cfg_path),
            "shared_config": str(shared_cfg_path),
            "dataset": "omomo",
            "seed": args.seed,
            "velocity_contact_arch": VC_ARCH,
            "baseline_root": str(BASELINE_ROOT),
            "reused_checkpoints": {k: str(v) for k, v in checkpoints.items()},
            "variants": {name: VARIANTS[name] for name in args.variants},
            "ot_epochs": args.ot_epochs,
            "gpu": args.gpu,
            "max_parallel": args.max_parallel,
            "pipelines": {},
        }
    )
    write_json(manifest_path, manifest)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": "/tmp",
            "TMP": "/tmp",
            "TEMP": "/tmp",
            "IMUHOI_PERSISTENT_WORKERS": "0",
            "IMUHOI_SKIP_DATALOADER_PRIME": "1",
        }
    )

    def log(message: str) -> None:
        with lock, root_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{now()}] {message}\n")

    def update_pipeline(name: str, **fields) -> None:
        with lock:
            manifest["pipelines"].setdefault(name, {}).update(fields)
            write_json(manifest_path, manifest)

    def run_stage(command: list[str], log_path: Path) -> None:
        message = "RUN " + " ".join(command)
        log(message)
        print(f"[{now()}] {message}", flush=True)
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                command, cwd=PROJECT_ROOT, env=env, stdout=output,
                stderr=subprocess.STDOUT,
            )
        log(f"END returncode={result.returncode} log={log_path}")
        print(f"[{now()}] END rc={result.returncode} log={log_path}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Stage failed ({result.returncode}); see {log_path}")

    def eval_command(config: Path, vc_ckpt: Path, ot_ckpt: Path, ot_arch: str,
                     out_json: Path) -> list[str]:
        return [
            str(SAGE_PYTHON), str(PROJECT_ROOT / "eval_batch_online.py"),
            "--config", str(config),
            "--dataset", "omomo",
            "--hp_ckpt", str(checkpoints["hp_ckpt"]),
            "--vc_ckpt", str(vc_ckpt),
            "--ot_ckpt", str(ot_ckpt),
            "--velocity_contact_arch", VC_ARCH,
            "--object_trans_arch", ot_arch,
            "--simulate_imu_noise",
            "--simulate_imu_noise_seed", str(args.seed),
            "--seed", str(args.seed),
            "--num_workers", str(args.num_workers),
            "--online_batch_size", str(args.eval_batch_size),
            "--output_json", str(out_json),
        ]

    def variant(name: str) -> bool:
        spec = VARIANTS[name]
        arch = spec["arch"]
        pipeline_root = root / name
        pipeline_root.mkdir(parents=True, exist_ok=True)
        suffix = f"omomo_ot_{arch}_{name}_{args.run_tag}"
        config = copy.deepcopy(shared_cfg)
        config.update(
            {"save_dir": str(pipeline_root), "object_trans_arch": arch, **spec["overrides"]}
        )
        variant_cfg = pipeline_root / "config.yaml"
        write_yaml(variant_cfg, config)
        update_pipeline(
            name,
            variant=name,
            label=spec["label"],
            overrides=spec["overrides"],
            object_trans_arch=arch,
            velocity_contact_arch=VC_ARCH,
            config=str(variant_cfg),
            status="running",
            phase="object_trans_joint_train" if spec["train"] else "eval",
        )
        try:
            if not spec["train"]:
                joint = (checkpoints["v2_joint_vc_ckpt"], checkpoints["v2_joint_ot_ckpt"])
                if arch != "v2":
                    raise ValueError(f"{name}: eval-only variants reuse the archived ot-v2 checkpoints")
            else:
                joint = find_joint_checkpoints(pipeline_root, suffix)
            if joint is None:
                run_stage(
                    [
                        str(SAGE_PYTHON),
                        str(PROJECT_ROOT / "train/rnn/train_object_trans.py"),
                        "--cfg", str(variant_cfg),
                        "--save_dir", str(pipeline_root),
                        "--run_suffix", suffix,
                        "--seed", str(args.seed),
                        "--train_datasets", "omomo",
                        "--num_workers", str(args.num_workers),
                        "--simulate_imu_noise",
                        "--epochs", str(args.ot_epochs),
                        "--joint_train",
                        "--velocity_contact_arch", VC_ARCH,
                        "--object_trans_arch", arch,
                        "--hp_ckpt", str(checkpoints["hp_ckpt"]),
                        "--vc_ckpt", str(checkpoints["vc_ckpt"]),
                    ],
                    pipeline_root / "object_trans_joint_train.log",
                )
                joint = find_joint_checkpoints(pipeline_root, suffix)
                if joint is None:
                    raise FileNotFoundError(f"{name}: no joint checkpoints produced")
            joint_vc, joint_ot = joint
            update_pipeline(
                name, phase="eval",
                joint_vc_ckpt=str(joint_vc), joint_ot_ckpt=str(joint_ot),
            )

            eval_json = pipeline_root / "eval_batch_online.json"
            if not eval_json.is_file():
                run_stage(
                    eval_command(variant_cfg, joint_vc, joint_ot, arch, eval_json),
                    pipeline_root / "eval_batch_online.log",
                )
            update_pipeline(
                name, status="complete", phase="complete",
                eval_json=str(eval_json), metrics=read_metrics(eval_json),
                finished_at=now(),
            )
            return True
        except Exception as exc:  # noqa: BLE001 - recorded in the manifest
            update_pipeline(
                name, status="failed", phase="failed", error=str(exc),
                finished_at=now(),
            )
            log(f"VARIANT FAILED {name}: {exc}")
            return False

    def v2_reference() -> bool:
        """Re-evaluate the archived ot-v2 checkpoints under today's eval code."""
        name = "ot_v2_reference"
        pipeline_root = root / name
        pipeline_root.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(shared_cfg)
        config.update({"save_dir": str(pipeline_root), "object_trans_arch": "v2"})
        variant_cfg = pipeline_root / "config.yaml"
        write_yaml(variant_cfg, config)
        update_pipeline(
            name, variant=name, object_trans_arch="v2",
            velocity_contact_arch=VC_ARCH, config=str(variant_cfg),
            status="running", phase="eval",
            joint_vc_ckpt=str(checkpoints["v2_joint_vc_ckpt"]),
            joint_ot_ckpt=str(checkpoints["v2_joint_ot_ckpt"]),
        )
        try:
            eval_json = pipeline_root / "eval_batch_online.json"
            if not eval_json.is_file():
                run_stage(
                    eval_command(
                        variant_cfg,
                        checkpoints["v2_joint_vc_ckpt"],
                        checkpoints["v2_joint_ot_ckpt"],
                        "v2",
                        eval_json,
                    ),
                    pipeline_root / "eval_batch_online.log",
                )
            update_pipeline(
                name, status="complete", phase="complete",
                eval_json=str(eval_json), metrics=read_metrics(eval_json),
                finished_at=now(),
            )
            return True
        except Exception as exc:  # noqa: BLE001
            update_pipeline(
                name, status="failed", phase="failed", error=str(exc),
                finished_at=now(),
            )
            log(f"V2 REFERENCE FAILED: {exc}")
            return False

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {executor.submit(variant, name): name for name in args.variants}
        results = {
            futures[future]: bool(future.result()) for future in as_completed(futures)
        }
    if not args.skip_v2_reeval:
        results["ot_v2_reference"] = v2_reference()

    manifest["results"] = results
    manifest["finished_at"] = now()
    manifest["status"] = "complete" if all(results.values()) else "failed"
    write_json(manifest_path, manifest)
    write_summary(root, manifest)
    log(f"FINISHED status={manifest['status']}")
    return 0 if manifest["status"] == "complete" else 1


def write_summary(root: Path, manifest: dict) -> None:
    pipelines = manifest.get("pipelines", {})
    order = ["ot_v2_reference"] + [name for name in VARIANTS if name in pipelines]
    labels = {"ot_v2_reference": "ot-v2 (archived ckpt, re-evaluated)"}
    labels.update({name: VARIANTS[name]["label"] for name in VARIANTS})
    rows = [name for name in order if name in pipelines]
    lines = [
        "# OMOMO ObjectTrans v3 ablation",
        "",
        f"Generated {now()} from `{root}`.",
        "",
        "Shared upstream: the HumanPose checkpoint and VC-v4 checkpoint from",
        f"`{manifest.get('baseline_root')}`. Only ObjectTrans was retrained",
        "(`--joint_train`, so VC is fine-tuned with it). Eval is",
        "`eval_batch_online.py` on the OMOMO test split, seed 42, simulated IMU noise.",
        "",
        "| Metric | " + " | ".join(labels[name] for name in rows) + " |",
        "| --- | " + " | ".join("---" for _ in rows) + " |",
    ]
    for key, label in METRICS:
        cells = []
        for name in rows:
            value = manifest["pipelines"][name].get("metrics", {}).get(key)
            cells.append("n/a" if value is None else f"{value:.4f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    for name in rows:
        entry = manifest["pipelines"][name]
        overrides = entry.get("overrides")
        if overrides:
            lines.append(f"- **{labels[name]}**: `{overrides}`")
    lines.append("")
    (root / "ot_v3_ablation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
