#!/usr/bin/env python3
"""OMOMO noise-protocol transfer, baseline first, with a gated IMUHOI++ run.

Only scalar noise settings are read from EgoIMU; no module or checkpoint is
imported from that repository. Numerical proximity is not protocol equivalence.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_behave_full_pipeline_trust import configure_noise_profile

PYTHON = Path("/home/l/anaconda3/envs/SAGE/bin/python")
RUNNER = ROOT / "scripts/run_behave_full_pipeline_trust.py"
DEFAULT_ROOT = ROOT / "outputs/IMUHOI++/paper_noise_omomo_20261016"
VARIANTS = {"baseline": ("imuhoi", "vc_imuhoi_ot_3branch"),
            "proposed": ("v5", "v2_armrot_biclose")}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_gate(metrics, targets, tolerance):
    """Require finite nonnegative values for every registered metric, final only."""
    checks = {}
    for key, target in targets.items():
        value = metrics.get(key)
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        checks[key] = {"value": value if valid else None, "target": target,
                       "upper_limit": target * (1 + tolerance),
                       "pass": bool(valid and value <= target * (1 + tolerance))}
    return {"passed": bool(checks) and all(v["pass"] for v in checks.values()),
            "checks": checks, "meaning": "numerical threshold only; not strict paper reproduction"}


def select_gpu(minimum):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    print(result.stdout, flush=True)
    rows = [tuple(int(v.strip()) for v in line.split(",")) for line in result.stdout.splitlines()]
    index, free, used, utilization = max(rows, key=lambda row: (row[1], -row[3]))
    if free < minimum:
        raise RuntimeError(f"GPU {index} has {free} MiB free; need {minimum} MiB. Retry when available.")
    return index


def audit_data(cfg):
    import torch

    entry = cfg["datasets"]["omomo"]
    audit, ids = {}, {}
    for split in ("train", "test"):
        folder = ROOT / entry[f"{split}_path"]
        files = sorted(folder.glob("*.pt"))
        if not files:
            raise FileNotFoundError(folder)
        ids[split] = {re.sub(r"_seg\d+$", "", p.stem) for p in files}
        sample = torch.load(files[0], map_location="cpu", weights_only=False)
        metadata = {k: sample.get(k) for k in ("canonical_body_model_gender", "canonical_betas",
                                               "canonical_skeleton_version", "contact_postprocess_version")}
        if metadata["canonical_body_model_gender"] != "neutral" or metadata["canonical_betas"] != "zero":
            raise ValueError(f"Not canonical data: {files[0]}")
        if any(sample.get(k) is not None for k in ("human_imu_real", "obj_imu_real")):
            raise ValueError("This protocol requires synthetic OMOMO IMUs")
        inventory = [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files]
        audit[split] = {"path": str(folder.relative_to(ROOT)), "count": len(files), "source_count": len(ids[split]),
                        "inventory": inventory, "sample_metadata": metadata}
    overlap = ids["train"] & ids["test"]
    if overlap:
        raise ValueError(f"Source sequence leakage: {sorted(overlap)[:5]}")
    audit["optimizer_steps_per_epoch"] = audit["train"]["count"] * cfg["windows_per_sequence"] // cfg["batch_size"]
    if audit["optimizer_steps_per_epoch"] < 1:
        raise ValueError("drop_last would give no training batches")
    return audit


def source_hashes():
    paths = [Path(__file__).resolve(), RUNNER, ROOT / "utils/utils.py", ROOT / "utils/hoi_metrics.py",
             ROOT / "dataset/dataset_IMUHOI.py", ROOT / "process/imu_noise.py",
             ROOT / "process/preprocess.py", ROOT / "eval_IMUHOI.py", ROOT / "eval_batch_online.py"]
    paths += list((ROOT / "model/rnn").glob("*.py"))
    paths += list((ROOT / "train/rnn").rglob("*.py"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


def prepare(root, protocol, upgrade_eval_protocol=False):
    cfg = yaml.safe_load((ROOT / protocol["base_config"]).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg.update(protocol["v5ar"])
    configure_noise_profile(cfg, protocol["noise_profile"], protocol["seed"])
    cfg.update({"train_datasets": ["omomo"], "dataloader_pin_memory": False,
                "pretrained_modules": {"human_pose": None, "velocity_contact": None, "object_trans": None}})
    # Preserve training data and recipe; evaluate the same weights in both modes.
    original_cfg = ROOT.parent / "EgoIMU/outputs/IMUHOI/omomo/training_config.yaml"
    contents = original_cfg.read_text(encoding="utf-8")
    values = {}
    for key in ("imu_noise_std", "obj_imu_noise_std"):
        match = re.search(rf"^{key}:\s*([0-9.eE+-]+)\s*$", contents, re.MULTILINE)
        if not match or float(match.group(1)) != 0.1:
            raise ValueError(f"Noise provenance mismatch for {key}")
        values[key] = float(match.group(1))
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                  "noise_source": str(original_cfg), "noise_source_sha256": sha256(original_cfg),
                  "extracted_noise_values": values, "dataset": audit_data(cfg), "source_sha256": source_hashes(),
                  "selection_caveat": "Existing test-window selection/early stopping; no independent validation split.",
                  "scope": "Canonical dataset; paper comparison offline, secondary online; transferred noise protocol."}
    root.mkdir(parents=True, exist_ok=True)
    recipe = root / "recipe.yaml"
    if recipe.exists():
        old = json.loads((root / "provenance.json").read_text())
        if upgrade_eval_protocol:
            # This migration only changes evaluation orchestration, never training.
            previous_protocol = dict(protocol)
            previous_protocol.pop("gate_inference_mode", None)
            previous_protocol.pop("evaluation_modes", None)
            changed_sources = {key for key in set(old["source_sha256"]) | set(provenance["source_sha256"])
                               if old["source_sha256"].get(key) != provenance["source_sha256"].get(key)}
            allowed_sources = {"scripts/run_omomo_paper_noise.py"}
            if old["protocol"] != previous_protocol or not changed_sources <= allowed_sources:
                raise ValueError("Evaluation migration may not change the training recipe or training code")
            for key in ("dataset", "noise_source_sha256"):
                if old[key] != provenance[key]:
                    raise ValueError(f"Evaluation migration changed {key}")
            if yaml.safe_load(recipe.read_text()) != cfg:
                raise ValueError("Evaluation migration changed training configuration")
            archive = root / "provenance.before_offline_correction.json"
            if archive.exists():
                raise FileExistsError(archive)
            write_json(archive, old)
            provenance["protocol_revision"] = "User correction: paper metrics are offline; retain online as additional output."
            write_json(root / "provenance.json", provenance)
            old = provenance
        for key in ("protocol", "dataset", "source_sha256", "noise_source_sha256"):
            if old[key] != provenance[key]:
                raise ValueError(f"Frozen experiment changed ({key}); use a new experiment root")
        if yaml.safe_load(recipe.read_text()) != cfg:
            raise ValueError("Base recipe changed; use a new experiment root")
    else:
        recipe.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        write_json(root / "provenance.json", provenance)
    return recipe, provenance


def read_pipeline(root, method):
    path = root / method / "experiment_manifest.json"
    if not path.exists():
        return {}, {}
    manifest = json.loads(path.read_text())
    return manifest, manifest.get("pipelines", {}).get(VARIANTS[method][1], {})


def evaluation_path(root, method, mode, checkpoint):
    folder = root / method / VARIANTS[method][1]
    name = f"eval_omomo_offline_{checkpoint}.json" if mode == "offline" else f"eval_omomo_{checkpoint}.json"
    return folder / name


def read_evaluation(root, method, mode, checkpoint):
    path = evaluation_path(root, method, mode, checkpoint)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    expected = "offline" if mode == "offline" else "batch_online"
    if payload.get("inference_mode") != expected:
        raise ValueError(f"Wrong evaluation mode in {path}")
    metrics = payload.get("results", {}).get("omomo", {})
    return {key: float(value) for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(value)}


def offline_command(root, method, protocol, checkpoint):
    manifest, pipeline = read_pipeline(root, method)
    stage = Path(pipeline["stage_dir"])
    hp = stage / f"{checkpoint}_human_pose.pt"
    vc = stage / f"{checkpoint}_velocity_contact.pt"
    ot = stage / f"{checkpoint}_object_trans.pt"
    if not hp.is_file():
        hp = Path(manifest["stages"]["human_pose"]["checkpoint"])
    if not vc.is_file():
        vc = Path(manifest["stages"]["velocity_contact"]["checkpoint"])
    for path in (hp, vc, ot):
        if not path.is_file():
            raise FileNotFoundError(path)
    return [str(PYTHON), str(ROOT / "eval_IMUHOI.py"), "--config", pipeline["config"],
            "--dataset", "omomo", "--inference_mode", "offline", "--hp_ckpt", str(hp),
            "--vc_ckpt", str(vc), "--ot_ckpt", str(ot),
            "--velocity_contact_arch", VARIANTS[method][0], "--object_trans_arch", pipeline["object_trans_arch"],
            "--num_workers", str(protocol["num_workers"]), "--seed", str(protocol["seed"]),
            "--output_json", str(evaluation_path(root, method, "offline", checkpoint))]


def evaluate_offline(root, method, protocol):
    for checkpoint in ("best", "final"):
        output = evaluation_path(root, method, "offline", checkpoint)
        command = offline_command(root, method, protocol, checkpoint)
        receipt = output.with_suffix(".command.json")
        if output.exists():
            if not receipt.exists() or json.loads(receipt.read_text()) != command:
                raise ValueError(f"Cached evaluation has different checkpoint paths: {output}")
            read_evaluation(root, method, "offline", checkpoint)
            continue
        gpu = select_gpu(protocol["minimum_free_gpu_mib"])
        write_json(receipt, command)
        print(f"Offline eval {method}/{checkpoint}: {output}", flush=True)
        with output.with_suffix(".log").open("a") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"})
        if not read_evaluation(root, method, "offline", checkpoint):
            raise RuntimeError(f"Offline evaluation returned no finite metrics: {output}")


def wait_for_existing_baseline(root, pid):
    """Adopt only the verified experiment runner; keep its active training intact."""
    cmdline = Path(f"/proc/{pid}/cmdline")
    expected_root = str(root / "baseline").encode()
    if not cmdline.exists():
        raise RuntimeError(f"Baseline PID {pid} is no longer running; resume without --adopt_baseline_pid")
    identity = cmdline.read_bytes()
    if str(RUNNER).encode() not in identity.split(b"\0") or expected_root not in identity.split(b"\0"):
        raise ValueError("Refusing to adopt an unrelated process")
    write_json(root / "adopted_baseline.json", {"pid": pid, "command": identity.decode().split("\0")})
    print(f"Adopting baseline runner PID {pid}; training continues unchanged.", flush=True)
    while cmdline.exists():
        try:
            if cmdline.read_bytes() != identity:
                break
        except FileNotFoundError:
            break
        time.sleep(30)
    _, pipeline = read_pipeline(root, "baseline")
    if pipeline.get("status") != "complete":
        raise RuntimeError("Adopted baseline stopped before completion; resume without --adopt_baseline_pid")


def report(root, protocol, state):
    rows = []
    for method in VARIANTS:
        _, pipeline = read_pipeline(root, method)
        for mode in protocol["evaluation_modes"]:
            for checkpoint in ("best", "final"):
                metrics = read_evaluation(root, method, mode, checkpoint)
                rows.append({"method": method, "mode": mode, "checkpoint": checkpoint,
                             "status": "complete" if metrics else ("pending_evaluation" if pipeline.get("status") == "complete"
                                                                      else pipeline.get("status", "pending")),
                             "metrics": metrics, "output_json": str(evaluation_path(root, method, mode, checkpoint))})
    write_json(root / "results.json", {"state": state, "rows": rows})
    lines = ["# OMOMO: paper noise protocol", "", f"State: {state['status']}", "",
             "Paper comparison and numerical gate: OFFLINE. ONLINE is reported separately on the same checkpoints.",
             "Current canonical data. Gaussian sigma 0.1 on all 9 IMU features during training",
             "and trainer validation; independent eval is clean. Existing test-window checkpoint selection",
             "and early stopping are retained; this is a diagnostic replication, not an unbiased benchmark.", "",
             "| Method | Mode | Checkpoint | State | Object cm | HOI cm | MPJPE cm | Root cm |", "|---|---|---|---|---|---|---|---|"]
    keys = list(protocol["paper_targets_cm"])
    lines.append("| Paper Table 2 | offline | published | reference | " + " | ".join(str(protocol["paper_targets_cm"][k]) for k in keys) + " |")
    for row in rows:
        values = [f"{row['metrics'][k]:.4f}" if k in row["metrics"] else "pending" for k in keys]
        lines.append("| " + " | ".join([row["method"], row["mode"], row["checkpoint"], row["status"], *values]) + " |")
    if "gate" in state:
        lines += ["", f"OFFLINE final-checkpoint numerical gate passed: {state['gate']['passed']}",
                  "All four errors must be <= 110% of the reference. Better numbers also pass; this does not prove",
                  "paper-protocol equivalence. Jitter is excluded because derivative order/units differ."]
    for mode in protocol["evaluation_modes"]:
        final = {r["method"]: r["metrics"] for r in rows if r["mode"] == mode and r["checkpoint"] == "final" and r["metrics"]}
        if set(final) == set(VARIANTS):
            lines += ["", f"## {mode}: final-checkpoint improvement (baseline - proposed) / baseline", ""]
            for key in keys:
                b, p = final["baseline"].get(key), final["proposed"].get(key)
                if b is not None and p is not None and b > 0:
                    lines.append(f"- {key}: {100 * (b - p) / b:.2f}%")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_method(root, recipe, protocol, method, hp_checkpoint=None):
    gpu = select_gpu(protocol["minimum_free_gpu_mib"])
    arch, variant = VARIANTS[method]
    command = [str(PYTHON), str(RUNNER), "--cfg", str(recipe), "--experiment_root", str(root / method),
               "--dataset", "omomo", "--vc_arch", arch, "--variants", variant, "--noise_profile", protocol["noise_profile"],
               "--run_tag", "paper_noise", "--seed", str(protocol["seed"]), "--gpu", str(gpu),
               "--num_workers", str(protocol["num_workers"]), "--eval_batch_size", str(protocol["eval_batch_size"]),
               "--max_parallel", "1", "--max_attempts", "1", "--check_gpu_before_stage", "--resume"]
    if hp_checkpoint:
        command += ["--skip_hp", hp_checkpoint]
    write_json(root / f"{method}_command.json", command)
    print(f"Starting {method}; log: {root / (method + '.log')}", flush=True)
    with (root / f"{method}.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/experiments/omomo_paper_noise_20261016.yaml")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--upgrade_eval_protocol", action="store_true",
                        help="Archive initial provenance and migrate this run to the user's offline-primary correction.")
    parser.add_argument("--adopt_baseline_pid", type=int, default=None,
                        help="Wait for this existing baseline runner instead of launching duplicate training.")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f"Use {PYTHON}")
    root = args.root.resolve()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("gate_inference_mode") != "offline" or protocol.get("evaluation_modes") != ["offline", "online"]:
        raise ValueError("This experiment requires offline primary evaluation plus online outputs")
    if args.summarize_only:
        report(root, protocol, json.loads((root / "status.json").read_text()))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        select_gpu(0 if args.adopt_baseline_pid else protocol["minimum_free_gpu_mib"])
        recipe, provenance = prepare(root, protocol, upgrade_eval_protocol=args.upgrade_eval_protocol)
        state = {"status": "prepared", "strict_paper_reproduction": False, "gate_inference_mode": "offline"}
        def commit():
            write_json(root / "status.json", state)
            report(root, protocol, state)
        commit()
        print(json.dumps({"root": str(root), "train_sequences": provenance["dataset"]["train"]["count"],
                          "test_sequences": provenance["dataset"]["test"]["count"],
                          "steps_per_epoch": provenance["dataset"]["optimizer_steps_per_epoch"]}), flush=True)
        if args.prepare_only:
            return 0
        try:
            state["status"] = "baseline_running"
            commit()
            if args.adopt_baseline_pid:
                wait_for_existing_baseline(root, args.adopt_baseline_pid)
            else:
                run_method(root, recipe, protocol, "baseline")
            manifest, baseline = read_pipeline(root, "baseline")
            if baseline.get("status") != "complete":
                raise RuntimeError("Baseline did not complete")
            state["status"] = "baseline_offline_eval"
            commit()
            evaluate_offline(root, "baseline", protocol)
            state["gate"] = numerical_gate(read_evaluation(root, "baseline", "offline", protocol["gate_checkpoint"]), protocol["paper_targets_cm"],
                                            protocol["numerical_gate_relative_tolerance"])
            if not state["gate"]["passed"]:
                state["status"] = "baseline_complete_gate_not_met"
                commit()
                return 0
            state["status"] = "proposed_running"
            commit()
            # Reuse only this experiment's initial HP, never the baseline joint HP.
            hp = manifest["stages"]["human_pose"]["checkpoint"]
            run_method(root, recipe, protocol, "proposed", hp_checkpoint=hp)
            _, proposed = read_pipeline(root, "proposed")
            if proposed.get("status") != "complete":
                raise RuntimeError("Proposed pipeline did not complete")
            state["status"] = "proposed_offline_eval"
            commit()
            evaluate_offline(root, "proposed", protocol)
            state["status"] = "comparison_complete"
            commit()
        except Exception as exc:
            state.update(status="failed", error=str(exc))
            commit()
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
