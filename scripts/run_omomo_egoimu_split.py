#!/usr/bin/env python3
"""Train current OMOMO preprocessing under the saved EgoIMU split and seed.

Data files are independent byte-identical copies of current preprocessed data.
No EgoIMU executable code or checkpoint is imported. The old run is preserved.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_omomo_paper_noise as queue

DEFAULT_ROOT = ROOT / "outputs/IMUHOI++/egoimu_split_seed10_omomo_20261016"
DEFAULT_CONFIG = ROOT / "configs/experiments/omomo_egoimu_split_seed10_20261016.yaml"


def reference_seed(path):
    matches = re.findall(r"^seed:\s*(\d+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Need exactly one saved scalar training seed")
    return int(matches[0])


def validate_split(info):
    train, test = info["train_files"], info["test_files"]
    all_names = train + test
    if len(set(all_names)) != len(all_names):
        raise ValueError("Duplicate sequence ID or train/test overlap in saved split")
    if len(all_names) != info["total_files"]:
        raise ValueError("Saved split count mismatch")
    if any(Path(name).name != name or not name.endswith(".pt") for name in all_names):
        raise ValueError("Invalid sequence filename in split")
    return {"train": sorted(train), "test": sorted(test)}


def materialize_split(unified, old_root, split_info, data_root):
    import torch

    splits = validate_split(split_info)
    required = set(splits["train"] + splits["test"])
    available = {p.name for p in unified.glob("*.pt")}
    if available != required:
        raise ValueError(f"Current unified IDs differ: missing {sorted(required-available)}, extra {sorted(available-required)}")
    need_bytes = sum((unified / name).stat().st_size for name in required)
    if shutil.disk_usage(data_root.parent).free < need_bytes + 1024 ** 3:
        raise RuntimeError("Insufficient space for independent data copies")
    inventory, identity_sets = {}, {}
    for split, names in splits.items():
        folder = data_root / split
        folder.mkdir(parents=True, exist_ok=True)
        if {p.name for p in folder.glob("*.pt")} - set(names):
            raise ValueError(f"Unexpected files in restored {split}")
        entries, identities = [], set()
        for i, name in enumerate(names):
            source, target = unified / name, folder / name
            old_source = old_root / split / name
            old, current = (torch.load(p, map_location="cpu", weights_only=False) for p in (old_source, source))
            if str(old["seq_name"]) != str(current["seq_name"]) or len(old["obj_trans"]) != len(current["obj_trans"]):
                raise ValueError(f"Source identity or length mismatch: {split}/{name}")
            if current.get("canonical_body_model_gender") != "neutral" or current.get("canonical_betas") != "zero":
                raise ValueError(f"Not the current canonical preprocessing: {name}")
            identity = str(current["seq_name"])
            if identity in identities:
                raise ValueError(f"Duplicate sequence identity in {split}: {identity}")
            identities.add(identity)
            digest = queue.sha256(source)
            if not target.exists():
                temporary = target.with_suffix(".pt.tmp")
                shutil.copy2(source, temporary)
                temporary.replace(target)
            if queue.sha256(target) != digest:
                raise ValueError(f"Restored copy differs from current preprocessing: {target}")
            entries.append({"name": name, "seq_name": identity, "frames": len(current["obj_trans"]),
                            "sha256": digest, "bytes": source.stat().st_size})
            if (i + 1) % 400 == 0:
                print(f"Identity and byte audit {split}: {i+1}/{len(names)}", flush=True)
        inventory[split], identity_sets[split] = entries, identities
    if identity_sets["train"] & identity_sets["test"]:
        raise ValueError("Train/test overlap by actual sequence identity")
    queue.write_json(data_root / "split_info.json", {**split_info, "restoration": "byte-identical current preprocessing copies"})
    queue.write_json(data_root / "inventory.json", inventory)
    return inventory


def training_recipe(previous, data_root, seed):
    cfg = copy.deepcopy(previous)
    cfg.update({"seed": seed, "simulate_imu_noise_seed": seed, "eval_hoi_protocol": "egoimu"})
    for key in ("omomo", "omomo_full"):
        entry = cfg["datasets"][key]
        entry.update({"train_path": str((data_root / "train").relative_to(ROOT)),
                      "test_path": str((data_root / "test").relative_to(ROOT)),
                      "debug_path": str((data_root / "train").relative_to(ROOT)),
                      "full_sequence_test_path": str((data_root / "test").relative_to(ROOT))})
    cfg["pretrained_modules"] = {"human_pose": None, "velocity_contact": None, "object_trans": None}
    return cfg


def sources():
    return {**queue.source_hashes(), str(Path(__file__).resolve().relative_to(ROOT)): queue.sha256(Path(__file__).resolve())}


def prepare(root, settings, settings_path):
    previous_path = ROOT / settings["previous_recipe"]
    original_config = (ROOT / settings["egoimu_training_config"]).resolve()
    original_split = (ROOT / settings["egoimu_split"]).resolve()
    unified = ROOT / settings["current_unified"]
    split_info = json.loads(original_split.read_text())
    splits = validate_split(split_info)
    seed = reference_seed(original_config)
    if seed != settings["seed"] or split_info["seed"] != settings["split_seed"]:
        raise ValueError("Configured seeds differ from saved EgoIMU experiment")
    previous = yaml.safe_load(previous_path.read_text(encoding="utf-8"))
    if previous["simulate_imu_noise"] or previous["imu_rot_noise_std"] != 0.1:
        raise ValueError("Previous recipe is not the completed Gaussian-noise experiment")
    cfg = training_recipe(previous, root / "data", seed)
    protocol = yaml.safe_load((ROOT / settings["previous_protocol"]).read_text(encoding="utf-8"))
    protocol.update({"seed": seed, "split_seed": split_info["seed"], "eval_hoi_protocol": "egoimu"})
    fingerprint = {"settings": settings, "settings_sha256": queue.sha256(settings_path),
                   "previous_recipe_sha256": queue.sha256(previous_path),
                   "egoimu_training_config_sha256": queue.sha256(original_config),
                   "egoimu_split_sha256": queue.sha256(original_split), "source_sha256": sources()}
    provenance_file = root / "provenance.json"
    if provenance_file.exists():
        provenance = json.loads(provenance_file.read_text())
        if provenance["fingerprint"] != fingerprint:
            raise ValueError("Frozen source/config/reference changed; use a new experiment root")
        if yaml.safe_load((root / "recipe.yaml").read_text()) != cfg:
            raise ValueError("Frozen recipe changed")
        inventory = json.loads((root / "data/inventory.json").read_text())
        for split, entries in inventory.items():
            if {entry["name"] for entry in entries} != set(splits[split]):
                raise ValueError("Cached identity inventory differs from reference split")
            for entry in entries:
                if any(queue.sha256(p) != entry["sha256"] for p in (unified / entry["name"], root / "data" / split / entry["name"])):
                    raise ValueError(f"Preprocessed tensors changed: {split}/{entry['name']}")
        return root / "recipe.yaml", protocol
    inventory = materialize_split(unified, original_split.parent, split_info, root / "data")
    cfg_path = root / "recipe.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint,
                  "seed": seed, "split_seed": split_info["seed"], "eval_hoi_protocol": "egoimu",
                  "counts": {split: len(entries) for split, entries in inventory.items()},
                  "data_source": str(unified), "split_source": str(original_split),
                  "training_config_changes": ["OMOMO train/test paths", "training seed 42 -> 10", "inactive noise seed 42 -> 10", "evaluation-only HOI protocol"],
                  "selection_caveat": "Same existing test-window checkpoint selection and early stopping as previous run.",
                  "comparison_caveat": "Split and seed both change; this is not a split-only causal ablation.",
                  "hoi_definition": "GT wrist 20/21 contacts, idle root 0, bimanual frames twice, then equal per-sequence mean"}
    queue.write_json(provenance_file, provenance)
    queue.write_json(root / "protocol.json", protocol)
    return cfg_path, protocol


def report(root, protocol, state):
    queue.report(root, protocol, state)
    path = root / "summary.md"
    header = ("Saved EgoIMU split: 4705 train / 1177 test. Training seed=10; reference split seed=42.\n"
              "Primary HOI: EgoIMU wrist contacts + idle-root relative error; contact-only palm HOI is also saved.\n"
              "Current preprocessing tensors are byte-identical copies. Baseline starts from a new HP.\n\n")
    path.write_text(header + path.read_text(encoding="utf-8"), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--settings", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != queue.PYTHON.resolve():
        raise RuntimeError(f"Use {queue.PYTHON}")
    root = args.root.resolve()
    if args.summarize_only:
        report(root, json.loads((root / "protocol.json").read_text()), json.loads((root / "status.json").read_text()))
        return 0
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue.select_gpu(8000)
        settings = yaml.safe_load(args.settings.read_text(encoding="utf-8"))
        queue.write_json(root / "status.json", {"status": "preparing_data", "seed": settings["seed"],
                                                "split_seed": settings["split_seed"], "eval_hoi_protocol": "egoimu"})
        try:
            recipe, protocol = prepare(root, settings, args.settings)
        except Exception as exc:
            queue.write_json(root / "status.json", {"status": "preparation_failed", "error": str(exc)})
            raise
        state = {"status": "prepared", "seed": protocol["seed"], "split_seed": protocol["split_seed"],
                 "gate_inference_mode": "offline", "eval_hoi_protocol": "egoimu", "strict_paper_reproduction": False}
        def commit():
            queue.write_json(root / "status.json", state)
            report(root, protocol, state)
        commit()
        if args.prepare_only:
            return 0
        try:
            for method in ("baseline", "proposed"):
                state["status"] = method + "_running"
                commit()
                hp = None
                if method == "proposed":
                    manifest, _ = queue.read_pipeline(root, "baseline")
                    hp = manifest["stages"]["human_pose"]["checkpoint"]
                queue.run_method(root, recipe, protocol, method, hp_checkpoint=hp)
                _, pipeline = queue.read_pipeline(root, method)
                if pipeline.get("status") != "complete":
                    raise RuntimeError(f"{method} pipeline incomplete")
                state["status"] = method + "_offline_eval"
                commit()
                queue.evaluate_offline(root, method, protocol)
                # Both independent evaluation modes must record the selected HOI protocol.
                for mode in ("offline", "online"):
                    for checkpoint in ("best", "final"):
                        payload = json.loads(queue.evaluation_path(root, method, mode, checkpoint).read_text())
                        if payload.get("hoi_metric_protocol") != "egoimu":
                            raise ValueError(f"Wrong HOI protocol: {method} {mode} {checkpoint}")
                if method == "baseline":
                    state["gate"] = queue.numerical_gate(queue.read_evaluation(root, method, "offline", "final"),
                                                         protocol["paper_targets_cm"], protocol["numerical_gate_relative_tolerance"])
                    if not state["gate"]["passed"]:
                        state["status"] = "baseline_complete_gate_not_met"
                        commit()
                        return 0
            state["status"] = "comparison_complete"
            commit()
        except Exception as exc:
            state.update(status="failed", error=str(exc))
            commit()
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
