"""Measure official VC-v4 + OT-v2 armrot/biclose online test tracks against predicted hand midplanes."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from imuhoi4heads_midplane_eval import analyze, sha256

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/IMUHOI++/RNN_cano"
OUT = ROOT / "outputs/analysis/armrot_biclose_midplane_20261002"
FOLDERS = {
    "omomo": "v2_armrot_biclose_20260923_omomo",
    "behave": "v2_armrot_biclose_20260923_behave",
}


def capture(dataset: str) -> dict:
    directory = BASE / FOLDERS[dataset]
    manifest = json.loads((directory / "experiment_manifest.json").read_text())
    pipeline = manifest["pipelines"]["v2_armrot_biclose"]
    assert pipeline["object_trans_arch"] == "v2"
    assert pipeline["velocity_contact_arch"] == "v4"
    published = json.loads((directory / "v2_armrot_biclose" / f"eval_{dataset}_final.json").read_text())
    assert published["object_trans_arch"] == "v2"
    assert published["velocity_contact_arch"] == "v4"
    dest = OUT / (dataset.upper() + "_final")
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest / "online_predictions.pt"
    result_file = dest / "official_test.json"
    stage = Path(pipeline["stage_dir"])
    hp = Path(pipeline["joint_hp_ckpt"])
    vc = stage / "final_velocity_contact.pt"
    ot = stage / "final_object_trans.pt"
    command = [sys.executable, str(ROOT / "scripts/evaluate_original_ot_plane.py"),
               "--capture_predictions", str(cache), "--config", pipeline["config"],
               "--dataset", dataset, "--hp_ckpt", str(hp), "--vc_ckpt", str(vc), "--ot_ckpt", str(ot),
               "--velocity_contact_arch", "v4", "--object_trans_arch", "v2", "--simulate_imu_noise",
               "--simulate_imu_noise_seed", "42", "--seed", "42", "--num_workers", "2",
               "--online_batch_size", "48", "--online_chunk", "16", "--online_window", "60",
               "--online_lookahead", "10", "--output_json", str(result_file)]
    for flag, key in (("--human_pose_smoothing_half_width", "human_pose_smoothing_half_width"),
                      ("--online_obj_vel_init_diff_frames", "online_obj_vel_init_diff_frames")):
        if key in published:
            command.extend((flag, str(published[key])))
    if not cache.exists() or not result_file.exists():
        with (dest / "official_test.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    rerun = json.loads(result_file.read_text())
    original_cm = float(published["results"][dataset]["obj_trans_err_fusion"])
    rerun_cm = float(rerun["results"][dataset]["obj_trans_err_fusion"])
    if abs(original_cm - rerun_cm) > 0.02:
        raise RuntimeError(f"{dataset} official final score differs: {original_cm} vs {rerun_cm}")
    analysis = analyze(cache, dataset)
    analysis.update({"source": str(directory), "cache": str(cache), "official_object_cm": original_cm,
                     "rerun_official_object_cm": rerun_cm, "command": command,
                     "source_checkpoint_hashes": {"hp": sha256(hp), "vc": sha256(vc), "ot": sha256(ot)}})
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("omomo", "behave"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    analysis = capture(args.dataset)
    name = args.dataset.upper() + "_final"
    (OUT / f"{name}.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False))
    for group in ("ordinary_bimanual", "all_bimanual"):
        print(name, group, analysis["groups"][group]["pred_published_to_pred_plane"], flush=True)
    print("official_object_cm", analysis["official_object_cm"], flush=True)


if __name__ == "__main__":
    main()
