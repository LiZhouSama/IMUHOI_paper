#!/usr/bin/env python3
"""Build the external-HP cache of one (dataset, comparison method) for the HP plug-in experiment.

Runs ``Comparisons/eval_comparison.py`` -- protocol v3, the code path behind the published
baseline numbers -- with the main pipeline's IMU noise (seed 42) over every directory the
IMUHOI++ trainers and the online evaluator read for this dataset:

* ``train_path``              training windows of VC / OT / joint,
* ``test_path``               the trainers' validation windows (best-checkpoint selection),
* ``full_sequence_test_path`` the online evaluation (identical to the published baseline eval).

and stores every sequence's online local pose and root translation (``--save_hp_cache``,
layout in ``utils/external_hp_cache.py``).  DIP has no translation estimator: as in the
published comparison table its root is the ground truth (``--dip_root gt``; ``fixed`` keeps
its first-frame root).  The full-test eval doubles as a reproduction check of the baseline
against the published eval JSON.  ``manifest.json`` is written last (status "complete"), so an
interrupted build is resumed directory by directory.

    python scripts/hp_plugin/build_hp_cache.py --dataset omomo --method tip \
        --cache_dir outputs/IMUHOI++/RNN_cano/hp_plugin_20261105/cache/omomo/tip
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.external_hp_cache import (  # noqa: E402
    CACHE_FORMAT,
    CACHE_VERSION,
    MANIFEST_NAME,
    cache_entry_path,
    directory_key,
    external_hp_cache_key,
)

METHODS = ("dip18", "tip", "transpose", "globalpose")
DATASETS = ("omomo", "behave", "imhd", "pahoi", "hodome")
COMPARISON_ROOTS = {
    "omomo": "outputs/Comparisons/omomo_cano_20261023/{method}",
    "default": "outputs/Comparisons/all4_cano_20261023/{dataset}/{method}",
}
CHECK_METRICS = ("obj_err_cm", "hoi_err_cm", "human_pos_err_cm", "root_trans_err_cm", "human_ang_err_deg")
CHECK_TOLERANCE = 0.05  # cross-GPU float differences; a different noise draw moves these by ~1 SE


def comparison_dir(dataset: str, method: str) -> Path:
    template = COMPARISON_ROOTS.get(dataset, COMPARISON_ROOTS["default"])
    return PROJECT_ROOT / template.format(dataset=dataset, method=method)


def published_eval_json(dataset: str, method: str, dip_root: str) -> Path:
    suffix = "_gtroot" if method == "dip18" and dip_root == "gt" else ""
    return comparison_dir(dataset, method) / f"eval_{dataset}_test_online{suffix}.json"


def dataset_directories(cfg_path: Path, dataset: str) -> list[tuple[str, Path]]:
    """(split label, directory) of every data directory the pipeline reads, deduplicated."""
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    entry = (cfg.get("datasets") or {}).get(dataset)
    if not entry:
        raise KeyError(f"{cfg_path} has no datasets.{dataset}")
    wanted = (
        ("train", entry.get("train_path")),
        ("test", entry.get("test_path")),
        ("test", entry.get("full_sequence_test_path") or entry.get("test_path")),
    )
    seen, result = set(), []
    for label, relative in wanted:
        if not relative:
            continue
        path = Path(relative)
        path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"datasets.{dataset} directory does not exist: {path}")
        if path in seen:
            continue
        seen.add(path)
        result.append((label, path))
    return result


def full_sequence_test_dir(cfg_path: Path, dataset: str) -> Path:
    entry = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})["datasets"][dataset]
    relative = Path(entry.get("full_sequence_test_path") or entry["test_path"])
    return (relative if relative.is_absolute() else PROJECT_ROOT / relative).resolve()


def project_relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def missing_entries(cache_dir: Path, directory: Path) -> list[str]:
    return [
        str(path)
        for path in sorted(directory.glob("*.pt"))
        if not cache_entry_path(cache_dir, external_hp_cache_key(path)).is_file()
    ]


def eval_json_path(cache_dir: Path, directory: Path) -> Path:
    return cache_dir / "eval" / (directory_key(directory).replace("/", "__") + ".json")


def run_directory(args, cache_dir: Path, label: str, directory: Path, checkpoint: Path, out_json: Path) -> None:
    command = [
        args.python, "-m", "Comparisons.eval_comparison",
        "--method", args.method, "--checkpoint", str(checkpoint),
        "--cfg", str(args.comparison_cfg), "--dataset", args.dataset,
        "--split", label, "--data_dir", str(directory),
        "--simulate_imu_noise", "--simulate_imu_noise_seed", "42", "--seed", "42",
        "--device", args.device, "--num_workers", str(args.num_workers),
        "--physics_workers", str(args.physics_workers),
        "--save_hp_cache", str(cache_dir), "--output_json", str(out_json),
    ]
    if args.method == "dip18" and args.dip_root == "gt":
        command.append("--gt_root_trans")
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def reference_check(dataset: str, method: str, dip_root: str, full_json: Path) -> dict:
    """Per-sequence comparison of the rebuilt full-test eval with the published baseline eval.

    DIP, TransPose and GlobalPose (incl. its physics) are deterministic: on the machine that
    made the published evals the rebuilt per-sequence metrics are bit-identical (checked
    2026-11-05 on six OMOMO sequences), elsewhere they differ only by GPU float noise.  TIP keeps
    its 0.8 history dropout on at test time (released behaviour), so its predictions depend on
    the RNG state, i.e. on the order sequences are processed (glob order differs between file
    systems): only the aggregate over the whole test set is expected to agree, within sampling
    noise.  The check reports; it does not stop the build.
    """
    reference = published_eval_json(dataset, method, dip_root)
    if not reference.is_file():
        print(f"note: no published baseline eval at {reference}; skipping the reproduction check")
        return {"reference": project_relative(reference), "available": False}
    published = json.loads(reference.read_text())
    rebuilt = json.loads(full_json.read_text())
    pub_seq = {row["seq_name"]: row["metrics"] for row in published.get("per_sequence", [])}
    new_seq = {row["seq_name"]: row["metrics"] for row in rebuilt.get("per_sequence", [])}
    common = sorted(set(pub_seq) & set(new_seq))
    same_set = bool(common) and set(pub_seq) == set(new_seq)
    per_sequence, aggregate = {}, {}
    for name in CHECK_METRICS:
        diffs = sorted(
            abs(new_seq[seq][name] - pub_seq[seq][name])
            for seq in common
            if new_seq[seq].get(name) is not None and pub_seq[seq].get(name) is not None
        )
        if diffs:
            per_sequence[name] = {"median_abs": diffs[len(diffs) // 2], "max_abs": diffs[-1]}
        if same_set and rebuilt["metrics"].get(name) is not None and published["metrics"].get(name) is not None:
            aggregate[name] = rebuilt["metrics"][name] - published["metrics"][name]
    stochastic = method == "tip"
    if stochastic:
        passed = all(abs(value) <= 10 * CHECK_TOLERANCE for value in aggregate.values()) if aggregate else None
    else:
        passed = all(stats["median_abs"] <= 0.01 for stats in per_sequence.values()) and all(
            abs(value) <= CHECK_TOLERANCE for value in aggregate.values()
        )
    print(f"baseline reproduction vs {project_relative(reference)} ({len(common)} common sequences, "
          f"{'same' if same_set else 'different'} sequence set{', stochastic TIP inference' if stochastic else ''}):")
    for name in CHECK_METRICS:
        line = f"  {name:20s}"
        if name in per_sequence:
            line += f" per-seq |d| median {per_sequence[name]['median_abs']:.2e} max {per_sequence[name]['max_abs']:.2e}"
        if name in aggregate:
            line += f"  aggregate d {aggregate[name]:+.4f}"
        print(line)
    print("  PASS" if passed else "  not assessed (subset)" if passed is None else
          "  WARNING: larger than float noise -- check the noise keys / data / checkpoint before training")
    return {
        "reference": project_relative(reference),
        "available": True,
        "common_sequences": len(common),
        "same_sequence_set": same_set,
        "stochastic_inference": stochastic,
        "per_sequence": per_sequence,
        "aggregate_deltas": aggregate,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="default: the published selected.pt of this dataset and method")
    parser.add_argument("--cfg", type=Path, default=PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml",
                        help="training config whose datasets.<dataset> paths the cache must cover")
    parser.add_argument("--comparison_cfg", type=Path, default=PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml",
                        help="--cfg of eval_comparison (the published baseline evals used this default)")
    parser.add_argument("--dip_root", choices=("gt", "fixed"), default="gt",
                        help="DIP root translation: ground truth as in the published table, or its fixed first frame")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--physics_workers", type=int, default=12, help="GlobalPose physics CPU processes")
    parser.add_argument("--force", action="store_true", help="re-run every directory")
    args = parser.parse_args()

    cache_dir = (args.cache_dir if args.cache_dir.is_absolute() else PROJECT_ROOT / args.cache_dir).resolve()
    checkpoint = args.checkpoint or comparison_dir(args.dataset, args.method) / "selected.pt"
    checkpoint = (checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"comparison checkpoint not found: {checkpoint}")
    cfg_path = (args.cfg if args.cfg.is_absolute() else PROJECT_ROOT / args.cfg).resolve()
    directories = dataset_directories(cfg_path, args.dataset)
    full_test_dir = full_sequence_test_dir(cfg_path, args.dataset)

    manifest_path = cache_dir / MANIFEST_NAME
    if manifest_path.is_file():
        manifest_path.unlink()  # invalid until every directory is complete again
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries = {}
    online_windows, translation_modes = [], set()
    for label, directory in directories:
        out_json = eval_json_path(cache_dir, directory)
        complete = out_json.is_file() and not missing_entries(cache_dir, directory)
        if args.force or not complete:
            out_json.parent.mkdir(parents=True, exist_ok=True)
            run_directory(args, cache_dir, label, directory, checkpoint, out_json)
        missing = missing_entries(cache_dir, directory)
        if missing:
            raise RuntimeError(f"{len(missing)} sequences of {directory} have no cache entry, e.g. {missing[:3]}")
        report = json.loads(out_json.read_text())
        online_windows.append(report.get("online_window") or {})
        translation_modes.update(report.get("translation_mode") or [])
        entries[directory_key(directory)] = {
            "split": label,
            "num_sequences": report.get("num_sequences"),
            "eval_json": str(out_json.relative_to(cache_dir)),
        }
        print(f"{directory_key(directory)}: {report.get('num_sequences')} sequences cached")

    latency = max(int(window.get("future_frames", 0) or 0) for window in online_windows)
    check = reference_check(args.dataset, args.method, args.dip_root, eval_json_path(cache_dir, full_test_dir))
    manifest = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "status": "complete",
        "method": args.method,
        "dataset": args.dataset,
        "checkpoint": project_relative(checkpoint),
        "protocol": "Comparisons paper_v3 online inference, simulate_imu_noise seed 42",
        "latency_frames": latency,
        "online_window": online_windows[0],
        "translation_mode": sorted(translation_modes),
        "dip_root": args.dip_root if args.method == "dip18" else None,
        "directories": entries,
        "full_sequence_test": directory_key(full_test_dir),
        "baseline_reproduction": check,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
    }
    tmp = manifest_path.with_name(MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)
    print(f"wrote {manifest_path} (latency {latency} frames, translation {sorted(translation_modes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
