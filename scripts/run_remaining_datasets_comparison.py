#!/usr/bin/env python3
"""Train the remaining canonical datasets and summarize both full pipelines.

Use WSL-native SAGE. All stages run serially; no shortened/debug training.
Each dataset shares its initially trained HP, but joint stages follow the
respective published recipes (VC+OT versus HP+OT). Test-window checkpoint
selection is inherited from the existing runner, not an independent validation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
RUNNER = ROOT / "scripts/run_behave_full_pipeline_trust.py"
DEFAULT_ROOT = ROOT / "outputs/IMUHOI++/RNN_cano/remaining3_v5ar_vs_imuhoi3b_20261010"
DATASETS = ("imhd", "pahoi", "hodome")
METHODS = {
    "proposed": ("v5", "v2_armrot_biclose", "HP + VC-v5ar + OT-v2-arm-bic"),
    "imuhoi3b": ("imuhoi", "vc_imuhoi_ot_3branch", "HP + VC-IMUHOI + OT-IMUHOI (3 branches)"),
}
METRICS = (
    ("obj_trans_err_fusion_refine", "Object (cm)"),
    ("hoi_err_fusion", "HOI (cm)"),
    ("mpjpe", "MPJPE (cm)"),
    ("root_trans_err", "Root (cm)"),
    ("contact_f1_lhand", "Left F1"),
    ("contact_f1_rhand", "Right F1"),
    ("contact_f1_obj", "Object F1"),
    ("obj_jitter", "Object jitter (mm/frame^2)"),
)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def select_gpu(requested=None):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    print(result.stdout, flush=True)
    available = []
    for line in result.stdout.splitlines():
        index, total, used, free, utilization = [int(x.strip()) for x in line.split(",")]
        if requested is None or index == requested:
            available.append((free, -utilization, index))
    if not available:
        raise RuntimeError("Requested GPU was not found")
    free, _, index = max(available)
    if free < 6000:
        raise RuntimeError(f"GPU {index} has only {free} MiB free; require at least 6000 MiB")
    print(f"Selected GPU {index}: {free} MiB free", flush=True)
    return index


def source_ids(files):
    return {re.sub(r"_seg\d+$", "", p.stem) for p in files}


def audit_dataset(cfg, dataset):
    entry = cfg["datasets"][dataset]
    paths = {key: ROOT / entry[key] for key in ("train_path", "test_path", "full_sequence_test_path")}
    files = {key: sorted(path.glob("*.pt")) for key, path in paths.items()}
    for key, values in files.items():
        if not values:
            raise FileNotFoundError(f"Empty/missing {dataset} {key}: {paths[key]}")
    train_ids = source_ids(files["train_path"])
    for key in ("test_path", "full_sequence_test_path"):
        overlap = train_ids & source_ids(files[key])
        if overlap:
            raise ValueError(f"Train/test source overlap in {dataset}: {sorted(overlap)[:5]}")
    n_windows = len(files["train_path"]) * cfg["windows_per_sequence"]
    if n_windows < cfg["batch_size"]:
        raise ValueError(f"{dataset}: drop_last would yield zero training batches")
    return {
        "body_model_type": entry["body_model_type"],
        "counts": {key: len(values) for key, values in files.items()},
        "paths": {key: str(path) for key, path in paths.items()},
        "train_source_sequences": len(train_ids),
        "train_steps_per_epoch": n_windows // cfg["batch_size"],
        "source_overlap": 0,
    }


def prepare(root, datasets, seed):
    base_path = ROOT / "configs/IMUHOI_train_rnn.yaml"
    reference_path = ROOT / "outputs/IMUHOI++/RNN_cano/vc_v5_20260930_ot_omomo_v5ar/v2_armrot_biclose/config.yaml"
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    reference = yaml.safe_load(reference_path.read_text(encoding="utf-8"))
    # Copy the exact trained v5ar switches, not the reference run's scaled schedule.
    v5 = {key: value for key, value in reference.items() if key.startswith("v5_")}
    if v5.get("v5_init_noise_prob") != 0.5 or v5.get("v5_heading_normalize_imu") is not False:
        raise ValueError("Reference is not the v5ar recipe")
    cfg.update(v5)
    cfg["pretrained_modules"] = {"human_pose": None, "velocity_contact": None, "object_trans": None}
    cfg["dataloader_pin_memory"] = False
    for model, path in cfg["body_model_paths"].items():
        if not (ROOT / path).is_file():
            raise FileNotFoundError(f"Missing {model} body model: {path}")
    audits = {ds: audit_dataset(cfg, ds) for ds in datasets}
    ref_n = len(list((ROOT / cfg["datasets"]["omomo"]["train_path"]).glob("*.pt")))
    ref_steps = ref_n * cfg["windows_per_sequence"] // cfg["batch_size"]
    for audit in audits.values():
        scale = ref_steps / audit["train_steps_per_epoch"]
        audit["schedule_scale"] = scale
        audit["max_stage_epochs"] = round(cfg["epochs"] * scale)
        audit["max_joint_epochs"] = round(cfg["joint_train_epochs"] * scale)
    provenance = {
        "base_config": str(base_path), "v5ar_reference": str(reference_path), "v5ar_switches": v5,
        "seed": seed, "datasets": audits, "reference_steps_per_epoch": ref_steps,
        "hp_smoothing_half_width": cfg.get("human_pose_smoothing_half_width", 0),
        "online_obj_vel_init_diff_frames": cfg.get("online_obj_vel_init_diff_frames", 0),
        "training_protocol": "existing test-window checkpoint selection; full-sequence online evaluation",
    }
    return cfg, provenance


def summary(root, jobs):
    records = []
    lines = ["# IMHD / PAHOI / HODome: VC-v5ar + OT-v2-arm-bic vs original IMUHOI", "",
             f"Updated: {now()}", "",
             "Seed 42 by default; per-dataset training from scratch; identical initial HP, canonical data,",
             "IMU-noise seed, HP smoothing (half-width 7), object-velocity init (diff 3), window/lookahead,",
             "and optimizer-step-scaled budgets. Each recipe keeps its own joint-training targets:",
             "proposed fine-tunes VC+OT with HP frozen; IMUHOI fine-tunes HP+OT with VC frozen.", "",
             "**Selection caveat:** `best` is chosen on the existing test-window loss, not a separate",
             "validation split. `final` is also reported but early stopping still observes test-window loss.",
             "Single-seed pipeline comparison; not an isolated architecture ablation or a clean held-out benchmark.", "",
             "A dash denotes unavailable metrics, never zero. Partial rows are explicitly marked.", ""]
    header = ["Dataset", "Method", "Checkpoint", "Status"] + [label for _, label in METRICS]
    lines += ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for job in jobs:
        manifest_path = Path(job["root"]) / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        pipe = manifest.get("pipelines", {}).get(job["variant"], {})
        phase = pipe.get("phase")
        if not phase:
            running = [name for name, stage in manifest.get("stages", {}).items() if stage.get("status") == "running"]
            phase = ",".join(running)
        status = (job["status"] if job["status"] == "failed" else
                  pipe.get("status", manifest.get("status", job["status"])))
        if status == "running" and phase:
            status += ":" + phase
        for tag in ("best", "final"):
            metrics = pipe.get("metrics_" + tag) or {}
            clean = {key: value if isinstance(value, (int, float)) and math.isfinite(value) else None for key, value in metrics.items()}
            record = {"dataset": job["dataset"], "method": job["method"], "checkpoint": tag,
                      "status": status, "manifest": str(manifest_path), "config": pipe.get("config"),
                      **{key: clean.get(key) for key, _ in METRICS}}
            records.append(record)
            numbers = ["-" if record[key] is None else f"{record[key]:.3f}" for key, _ in METRICS]
            lines.append("| " + " | ".join([job["dataset"].upper(), job["method"], tag, status] + numbers) + " |")
    lines += ["", "## Relative error reduction vs IMUHOI", "", "(baseline - proposed) / baseline; positive means improved. Only completed paired rows are used.", "",
              "| Dataset | Checkpoint | Object reduction | HOI reduction |", "|---|---|---|---|"]
    for ds in dict.fromkeys(job["dataset"] for job in jobs):
        for tag in ("best", "final"):
            pair = {r["method"]: r for r in records if r["dataset"] == ds and r["checkpoint"] == tag and r["status"] == "complete"}
            if set(pair) != set(METHODS):
                continue
            gains = []
            for key in ("obj_trans_err_fusion_refine", "hoi_err_fusion"):
                p, b = pair["proposed"][key], pair["imuhoi3b"][key]
                gains.append("-" if p is None or b is None or b == 0 else f"{100 * (b-p)/b:.2f}%")
            lines.append("| " + " | ".join([ds.upper(), tag] + gains) + " |")
    lines += ["", "## Provenance", "", "- `recipe.yaml`: frozen base configuration plus exact v5ar switches.",
              "- `provenance.json`: dataset sizes, schedules, configuration origin, and source-file hashes.",
              "- `queue_status.json`: job states, commands, failures, and resume state.",
              "- `<dataset>/<method>/experiment_manifest.json`: selected checkpoints and full precision metrics.",
              "- `<dataset>/<method>/<variant>/eval_<dataset>_{best,final}.json`: complete full-sequence results.", ""]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "metrics.json", records)
    with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Use WSL-native SAGE: {PYTHON}")
    root = args.root.resolve()
    if args.summarize_only:
        state = json.loads((root / "queue_status.json").read_text())
        summary(root, state["jobs"])
        return 0
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("Duplicate datasets")
    cfg, provenance = prepare(root, args.datasets, args.seed)
    if args.dry_run:
        print(json.dumps(provenance, indent=2))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recipe_path = root / "recipe.yaml"
        if recipe_path.exists() and (root / "provenance.json").exists():
            cfg = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
            old = json.loads((root / "provenance.json").read_text())
            if old["seed"] != args.seed or list(old["datasets"]) != args.datasets:
                raise ValueError("Resume datasets/seed differ from the frozen experiment")
        else:
            recipe_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            paths = [RUNNER, Path(__file__).resolve(), ROOT / "configs/IMUHOI_train_rnn.yaml"]
            paths += sorted((ROOT / "model/rnn").glob("*.py"))
            paths += sorted((ROOT / "train/rnn").rglob("*.py"))
            paths += [ROOT / "eval_batch_online.py", ROOT / "eval_IMUHOI.py", ROOT / "dataset/dataset_IMUHOI.py"]
            provenance["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            provenance["created_at"] = now()
            write_json(root / "provenance.json", provenance)
        jobs = [{"dataset": ds, "method": method, "variant": spec[1], "root": str(root / ds / method), "status": "queued"}
                for ds in args.datasets for method, spec in METHODS.items()]
        state = {"status": "prepared", "updated_at": now(), "jobs": jobs}
        write_json(root / "queue_status.json", state)
        summary(root, jobs)
        if args.prepare_only:
            print(root, flush=True)
            return 0
        failures = []
        for job in jobs:
            ds, method = job["dataset"], job["method"]
            try:
                gpu = select_gpu(args.gpu)
                command = [str(PYTHON), str(RUNNER), "--cfg", str(recipe_path), "--experiment_root", job["root"],
                           "--dataset", ds, "--vc_arch", METHODS[method][0], "--variants", job["variant"],
                           "--run_tag", "remaining3_20261010", "--seed", str(args.seed), "--gpu", str(gpu),
                           "--num_workers", "2", "--eval_batch_size", "48", "--max_parallel", "1",
                           "--schedule_scale", "auto", "--resume", "--check_gpu_before_stage"]
                if method == "imuhoi3b":
                    hp_manifest = json.loads((root / ds / "proposed/experiment_manifest.json").read_text())
                    hp = hp_manifest["stages"]["human_pose"]
                    hp_path = Path(hp["checkpoint"])
                    if hp["status"] not in ("complete", "reused") or not hp_path.is_file():
                        raise RuntimeError(f"No completed shared HP for {ds}")
                    command += ["--skip_hp", str(hp_path)]
                job.update(status="running", started_at=now(), gpu=gpu, command=command)
                state.update(status="running", updated_at=now())
                write_json(root / "queue_status.json", state)
                summary(root, jobs)
                print(f"START {ds}/{method}: {' '.join(command)}", flush=True)
                log_path = root / f"{ds}_{method}.log"
                with log_path.open("a", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, "PYTHONUNBUFFERED": "1"})
                if result.returncode:
                    raise RuntimeError(f"Exit {result.returncode}; inspect {log_path}")
                manifest = json.loads((Path(job["root"]) / "experiment_manifest.json").read_text())
                if manifest["status"] != "complete":
                    raise RuntimeError("Runner returned without a completed manifest")
                job.update(status="complete", finished_at=now())
            except Exception as exc:
                job.update(status="failed", error=str(exc), finished_at=now())
                failures.append(f"{ds}/{method}: {exc}")
                print(f"FAILED {failures[-1]}", flush=True)
            state["updated_at"] = now()
            write_json(root / "queue_status.json", state)
            summary(root, jobs)
        state.update(status="failed" if failures else "complete", finished_at=now(), failures=failures)
        write_json(root / "queue_status.json", state)
        summary(root, jobs)
        print(f"FINISHED {state['status']}: {root / 'summary.md'}", flush=True)
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
