"""Re-evaluate every row of the five-dataset comparison with per-joint rotation errors.

2026-10-25: eval_IMUHOI.py computed MPJRE with pytorch3d 0.7.3's
matrix_to_axis_angle, which returns non-canonical axis-angles (angle > pi) for
~20% of rotations and inflates the axis-difference MPJRE of IMUHOI++ / IMUHOI.
The evaluator now uses the canonical float64 conversion (as the baselines'
Comparisons/common/metrics.py always did) and reports per-joint errors with
--per_joint_rot_error.  This script re-runs the same checkpoints and settings
as the all5 tables:

  main rows   eval_batch_online.py (all5_v5ar_vs_imuhoi3b_20261012 sources)
  baselines   Comparisons.eval_comparison on selected.pt (OMOMO GlobalPose
              without physics: 4 h with it, MPJRE equal to 0.01 deg on OMOMO)

Outputs: outputs/Comparisons/per_joint_mpjre_20261025/<dataset>/<method>.json
Usage: python scripts/run_per_joint_mpjre_20261025.py [--jobs 3]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = "/home/l/anaconda3/envs/SAGE/bin/python"
MAIN = ROOT / "outputs/IMUHOI++/RNN_cano/all5_v5ar_vs_imuhoi3b_20261012/metrics.json"
OUT = ROOT / "outputs/Comparisons/per_joint_mpjre_20261025"
NOISE = ["--simulate_imu_noise", "--simulate_imu_noise_seed", "42"]


def main_jobs() -> list[tuple[Path, list[str]]]:
    jobs = []
    for entry in json.loads(MAIN.read_text()):
        if entry["checkpoint"] != "best":
            continue
        parts = Path(entry["eval_json"]).parts
        pipeline, experiment = parts[-2], ROOT.joinpath(*parts[:-2])
        manifest = json.loads((experiment / "experiment_manifest.json").read_text())["pipelines"][pipeline]
        dataset = entry["dataset"]
        out = OUT / dataset / f"{entry['method']}.json"
        jobs.append((out, [
            PY, "eval_batch_online.py", "--config", str(experiment / pipeline / "config.yaml"),
            "--dataset", dataset, "--hp_ckpt", manifest["joint_hp_ckpt"],
            "--vc_ckpt", manifest["joint_vc_ckpt"], "--ot_ckpt", manifest["joint_ot_ckpt"],
            "--velocity_contact_arch", manifest["velocity_contact_arch"],
            "--object_trans_arch", manifest["object_trans_arch"], *NOISE, "--seed", "42",
            "--num_workers", "2", "--online_batch_size", "48", "--per_joint_rot_error",
            "--output_json", str(out),
        ]))
    return jobs


def baseline_jobs() -> list[tuple[Path, list[str]]]:
    jobs = []
    for dataset in ("omomo", "behave", "imhd", "pahoi", "hodome"):
        run_root = (
            ROOT / "outputs/Comparisons/omomo_cano_20261023"
            if dataset == "omomo"
            else ROOT / "outputs/Comparisons/all4_cano_20261023" / dataset
        )
        for method in ("dip18", "tip", "transpose", "globalpose"):
            out = OUT / dataset / f"{method}.json"
            extra = ["--no_globalpose_physics"] if (dataset, method) == ("omomo", "globalpose") else []
            jobs.append((out, [
                PY, "-m", "Comparisons.eval_comparison", "--method", method,
                "--checkpoint", str(run_root / method / "selected.pt"), "--dataset", dataset,
                "--split", "test", *NOISE, "--per_joint_rot_error", *extra, "--output_json", str(out),
            ]))
    return jobs


def run(job: tuple[Path, list[str]]) -> str:
    out, command = job
    if out.is_file():
        return f"skip {out.relative_to(ROOT)}"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".log"), "w") as log:
        rc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    return f"rc={rc} {out.relative_to(ROOT)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    # Long OMOMO main-pipeline evaluations first.
    jobs = sorted(main_jobs(), key=lambda j: "omomo" not in str(j[0])) + baseline_jobs()
    with ThreadPoolExecutor(args.jobs) as pool:
        for message in pool.map(run, jobs):
            print(message, flush=True)
    print("all done", flush=True)
