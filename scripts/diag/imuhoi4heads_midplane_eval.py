"""Reproduce official VC-IMUHOI / OT-IMUHOI_4heads online evaluation and measure hand midplanes."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/analysis/imuhoi4heads_midplane_20261002"
BASE = ROOT / "outputs/IMUHOI++/RNN_cano"
RUNS = {
    "OMOMO_20260920": ("omomo", "relabel_20260920_omomo_imuhoi"),
    "BEHAVE_20260920": ("behave", "relabel_20260920_behave_imuhoi"),
    "BEHAVE_hpsmooth_20260922": ("behave", "relabel_20260922_behave_hpsmooth_imuhoi"),
}
RODS = {
    "omomo": {"mop", "floorlamp", "clothesstand", "tripod", "vacuum"},
    "behave": {"boxlong", "yogamat"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[np.ndarray]) -> dict:
    v = np.concatenate(values)
    return {
        "frames": int(v.size),
        "mean_cm": float(v.mean()),
        "median_cm": float(np.median(v)),
        "p90_cm": float(np.percentile(v, 90)),
        "within_2cm_pct": float(np.mean(v <= 2) * 100),
        "within_5cm_pct": float(np.mean(v <= 5) * 100),
        "within_10cm_pct": float(np.mean(v <= 10) * 100),
    }


def analyze(cache: Path, dataset: str) -> dict:
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    pools = defaultdict(list)
    sequences = defaultdict(set)
    invalid = defaultdict(int)
    for record in payload["records"]:
        n = int(record["publish_len"])
        if n == 0 or not record.get("has_object", True):
            continue
        palms = np.asarray(record["pred_palm_glb_pos"][:n], dtype=np.float64)
        gt_palms = np.asarray(record["gt_palm_pos_eval"][:n], dtype=np.float64)
        gt_obj = np.asarray(record["gt_obj_trans"][:n], dtype=np.float64)
        pred = np.asarray(record["pred_obj_trans_published"][:n], dtype=np.float64)
        raw = np.asarray(record["pred_obj_trans"][:n], dtype=np.float64)
        contacts = np.asarray(record["gt_lhand_contact"][:n], bool) & np.asarray(record["gt_rhand_contact"][:n], bool)
        pred_contacts = np.asarray(record["pred_hand_contact_prob"][:n, :2], dtype=np.float64)
        pred_contacts = (pred_contacts >= 0.5).all(axis=1)
        normal = palms[:, 1] - palms[:, 0]
        span = np.linalg.norm(normal, axis=1)
        gt_normal = gt_palms[:, 1] - gt_palms[:, 0]
        gt_span = np.linalg.norm(gt_normal, axis=1)
        finite = np.isfinite(normal).all(axis=1) & np.isfinite(gt_normal).all(axis=1)
        finite &= np.isfinite(pred).all(axis=1) & np.isfinite(raw).all(axis=1)
        finite &= np.isfinite(gt_obj).all(axis=1) & np.isfinite(palms).all(axis=(1, 2))
        finite &= np.isfinite(gt_palms).all(axis=(1, 2))
        valid = finite & (span > 1e-6) & (gt_span > 1e-6)
        invalid["all_bimanual"] += int((contacts & ~valid).sum())
        direction = normal / np.maximum(span[:, None], 1e-8)
        gt_direction = gt_normal / np.maximum(gt_span[:, None], 1e-8)
        mid = palms.mean(axis=1)
        gt_mid = gt_palms.mean(axis=1)
        measures = {
            "pred_published_to_pred_plane": np.abs(np.sum((pred - mid) * direction, axis=1)) * 100,
            "pred_raw_to_pred_plane": np.abs(np.sum((raw - mid) * direction, axis=1)) * 100,
            "gt_to_gt_plane": np.abs(np.sum((gt_obj - gt_mid) * gt_direction, axis=1)) * 100,
            "gt_to_pred_plane": np.abs(np.sum((gt_obj - mid) * direction, axis=1)) * 100,
        }
        object_name = str(record["obj_name"])
        for group, mask in (
            ("all_bimanual", contacts & valid),
            ("ordinary_bimanual", contacts & valid & (object_name not in RODS[dataset])),
            ("predicted_bimanual", pred_contacts & valid),
        ):
            if not mask.any():
                continue
            sequences[group].add(str(record["seq_file"]))
            for measure, values in measures.items():
                pools[(group, measure)].append(values[mask])
    return {
        "record_count": len(payload["records"]),
        "invalid_bimanual_frames": dict(invalid),
        "groups": {
            group: {
                "sequences": len(sequences[group]),
                **{measure: summarize(pools[(group, measure)]) for measure in (
                    "pred_published_to_pred_plane", "pred_raw_to_pred_plane", "gt_to_gt_plane", "gt_to_pred_plane"
                )},
            }
            for group in ("all_bimanual", "ordinary_bimanual", "predicted_bimanual")
        },
    }


def run_one(name: str, dataset: str, folder: str) -> dict:
    source = BASE / folder
    manifest = json.loads((source / "experiment_manifest.json").read_text())
    pipeline = manifest["pipelines"]["vc_imuhoi_ot_4heads"]
    assert pipeline["object_trans_arch"] == "imuhoi_4heads"
    assert pipeline["velocity_contact_arch"] == "imuhoi"
    original = json.loads((source / "vc_imuhoi_ot_4heads" / f"eval_{dataset}_best.json").read_text())
    assert original["object_trans_arch"] == "imuhoi_4heads"
    assert original["velocity_contact_arch"] == "imuhoi"
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest / "online_predictions.pt"
    result_path = dest / "official_test.json"
    log_path = dest / "official_test.log"
    checkpoints = {
        "hp": Path(pipeline["joint_hp_ckpt"]),
        "vc": Path(pipeline["joint_vc_ckpt"]),
        "ot": Path(pipeline["joint_ot_ckpt"]),
    }
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_original_ot_plane.py"),
        "--capture_predictions", str(cache), "--config", pipeline["config"],
        "--dataset", dataset, "--hp_ckpt", str(checkpoints["hp"]),
        "--vc_ckpt", str(checkpoints["vc"]), "--ot_ckpt", str(checkpoints["ot"]),
        "--velocity_contact_arch", "imuhoi", "--object_trans_arch", "imuhoi_4heads",
        "--simulate_imu_noise", "--simulate_imu_noise_seed", "42", "--seed", "42",
        "--num_workers", "2", "--online_batch_size", "48", "--online_chunk", "16",
        "--online_window", "60", "--online_lookahead", "10", "--output_json", str(result_path),
    ]
    for flag, key in (("--human_pose_smoothing_half_width", "human_pose_smoothing_half_width"),
                      ("--online_obj_vel_init_diff_frames", "online_obj_vel_init_diff_frames")):
        if key in original:
            command.extend((flag, str(original[key])))
    if not cache.exists() or not result_path.exists():
        with log_path.open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(result_path.read_text())
    baseline_obj = original["results"][dataset]["obj_trans_err_fusion"]
    reproduced_obj = result["results"][dataset]["obj_trans_err_fusion"]
    if abs(baseline_obj - reproduced_obj) > 0.02:
        raise RuntimeError(f"{name}: online object error disagrees: {baseline_obj} vs {reproduced_obj}")
    analysis = analyze(cache, dataset)
    analysis.update({
        "dataset": dataset, "source": str(source), "cache": str(cache),
        "config": pipeline["config"], "checkpoint_sha256": {key: sha256(value) for key, value in checkpoints.items()},
        "official_object_cm": reproduced_obj, "original_official_object_cm": baseline_obj,
        "official_hoi_cm": result["results"][dataset]["hoi_err_fusion"],
        "command": command,
    })
    (dest / "summary.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    return analysis


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, (dataset, folder) in RUNS.items():
        print(f"Starting {name}", flush=True)
        summaries[name] = run_one(name, dataset, folder)
        (OUT / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
        for group in ("all_bimanual", "ordinary_bimanual"):
            value = summaries[name]["groups"][group]["pred_published_to_pred_plane"]
            print(name, group, value, flush=True)


if __name__ == "__main__":
    main()
