#!/usr/bin/env python3
"""Run the explicitly authorized IMUHOI++ control after baseline gate failure."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_omomo_paper_noise as queue
from scripts import run_omomo_egoimu_split as restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=restored.DEFAULT_ROOT)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != queue.PYTHON.resolve():
        raise RuntimeError(f"Use {queue.PYTHON}")
    root = args.root.resolve()
    with (root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue.select_gpu(8000)
        provenance = json.loads((root / "provenance.json").read_text())
        if provenance["fingerprint"]["source_sha256"] != restored.sources():
            raise ValueError("Frozen baseline source changed; inspect before claiming identical conditions")
        protocol = json.loads((root / "protocol.json").read_text())
        cfg = yaml.safe_load((root / "recipe.yaml").read_text())
        if cfg["seed"] != 10 or cfg["eval_hoi_protocol"] != "egoimu" or protocol["seed"] != 10:
            raise ValueError("Unexpected seed or HOI policy")
        manifest, baseline = queue.read_pipeline(root, "baseline")
        if baseline.get("status") != "complete":
            raise ValueError("Baseline has not completed")
        hp = Path(manifest["stages"]["human_pose"]["checkpoint"])
        if not hp.is_file():
            raise FileNotFoundError(hp)
        if hp == Path(baseline["joint_hp_ckpt"]):
            raise ValueError("Reuse the original HP, not the baseline joint-finetuned HP")
        for mode in ("offline", "online"):
            for checkpoint in ("best", "final"):
                payload = json.loads(queue.evaluation_path(root, "baseline", mode, checkpoint).read_text())
                if payload.get("hoi_metric_protocol") != "egoimu":
                    raise ValueError("Baseline evaluation policy mismatch")
        inventory = json.loads((root / "data/inventory.json").read_text())
        for split, entries in inventory.items():
            folder = root / "data" / split
            if {p.name for p in folder.glob("*.pt")} != {entry["name"] for entry in entries}:
                raise ValueError(f"Data membership changed: {split}")
            for i, entry in enumerate(entries):
                if queue.sha256(folder / entry["name"]) != entry["sha256"]:
                    raise ValueError(f"Preprocessed data changed: {split}/{entry['name']}")
                if (i + 1) % 1000 == 0:
                    print(f"Frozen data check {split}: {i+1}/{len(entries)}", flush=True)
        authorization = {"requested_at": datetime.now(timezone.utc).isoformat(),
                         "instruction": "User explicitly requested IMUHOI++ under the same conditions, independently of baseline gate.",
                         "seed": 10, "hp_checkpoint": str(hp), "hp_sha256": queue.sha256(hp),
                         "recipe_sha256": queue.sha256(root / "recipe.yaml"),
                         "continuation_source_sha256": queue.sha256(Path(__file__).resolve()),
                         "frozen_baseline_sources_verified": True,
                         "data_counts": {split: len(entries) for split, entries in inventory.items()}}
        receipt = root / "proposed_authorization_20261017.json"
        if receipt.exists():
            old = json.loads(receipt.read_text())
            for key in ("hp_sha256", "recipe_sha256", "continuation_source_sha256"):
                if old[key] != authorization[key]:
                    raise ValueError(f"Continuation changed: {key}")
        else:
            queue.write_json(receipt, authorization)
        state = json.loads((root / "status.json").read_text())
        archive = root / "status.before_proposed_authorization_20261017.json"
        if not archive.exists():
            queue.write_json(archive, state)
        state.update(status="proposed_running", proposed_explicitly_authorized=True,
                     proposed_authorization=str(receipt))
        def commit():
            queue.write_json(root / "status.json", state)
            restored.report(root, protocol, state)
            path = root / "summary.md"
            path.write_text("IMUHOI++ explicitly authorized on 2026-10-17 despite the unmet baseline gate.\n\n" + path.read_text())
        commit()
        try:
            queue.run_method(root, root / "recipe.yaml", protocol, "proposed", hp_checkpoint=str(hp))
            _, pipeline = queue.read_pipeline(root, "proposed")
            if pipeline.get("status") != "complete":
                raise RuntimeError("Proposed pipeline incomplete")
            state["status"] = "proposed_offline_eval"
            commit()
            queue.evaluate_offline(root, "proposed", protocol)
            for mode in ("offline", "online"):
                for checkpoint in ("best", "final"):
                    payload = json.loads(queue.evaluation_path(root, "proposed", mode, checkpoint).read_text())
                    if payload.get("hoi_metric_protocol") != "egoimu":
                        raise ValueError("Proposed evaluation policy mismatch")
            state["status"] = "comparison_complete"
            commit()
        except Exception as exc:
            state.update(status="failed", error=str(exc))
            commit()
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
