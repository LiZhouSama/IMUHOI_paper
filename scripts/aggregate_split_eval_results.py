import argparse
import json
from pathlib import Path


METRICS = [
    ("mpjpe", "MPJPE (cm)"),
    ("mpjre_angle", "MPJRE (deg)"),
    ("root_trans_err", "Root trans (cm)"),
    ("stage1_root_trans_err", "Stage1 root (cm)"),
    ("jitter", "Jitter (mm/frame^2)"),
    ("obj_trans_err_fusion_raw", "Object fusion raw (cm)"),
    ("obj_trans_err_fusion_refine", "Object fusion refine/published (cm)"),
    ("obj_trans_err_imu", "Object IMU (cm)"),
    ("hoi_err_fusion", "HOI fusion (cm)"),
    ("hoi_err_imu", "HOI IMU (cm)"),
    ("contact_f1_lhand", "Left hand F1"),
    ("contact_f1_rhand", "Right hand F1"),
    ("contact_f1_obj", "Object F1"),
]

MODELS = {
    "baseline": ("omomo_baseline_v3_new_hp_20260823", "baseline"),
    "v3": ("omomo_baseline_v3_new_hp_20260823", "v3"),
    "v4": ("omomo_v4_pose_prior_20260823", "08_v4_supervised_pose_prior"),
}

DATASETS = ("omomo", "behave", "pahoi")


def load_result(project_root, model, dataset):
    run_dir, subdir = MODELS[model]
    path = (
        project_root
        / "outputs"
        / "IMUHOI++"
        / "RNN"
        / run_dir
        / subdir
        / f"eval_{dataset}_split.json"
    )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if dataset not in payload.get("results", {}):
        raise ValueError(f"{path} does not contain results for {dataset}")
    return path, payload


def format_value(value):
    return "NaN" if value != value else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    entries = []
    for model in MODELS:
        for dataset in DATASETS:
            source_path, payload = load_result(args.project_root, model, dataset)
            entries.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "source": str(source_path),
                    "config": payload.get("config"),
                    "settings": {
                        "online_window": payload.get("resolved_online_window"),
                        "online_lookahead": payload.get("resolved_online_lookahead"),
                        "seed": payload.get("seed"),
                        "simulate_imu_noise": payload.get("simulate_imu_noise"),
                        "simulate_imu_noise_seed": payload.get("simulate_imu_noise_seed"),
                        "velocity_contact_arch": payload.get("velocity_contact_arch"),
                    },
                    "metrics": payload["results"][dataset],
                }
            )

    report = {
        "description": "IMUHOI split-test evaluation comparison",
        "datasets": list(DATASETS),
        "models": list(MODELS),
        "entries": entries,
    }
    json_path = args.output_stem.with_suffix(".json")
    md_path = args.output_stem.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2, allow_nan=True)
        handle.write("\n")

    headers = ["Dataset", "Model"] + [label for _, label in METRICS]
    lines = [
        "# IMUHOI Split-Test Evaluation Comparison",
        "",
        "Settings: window 60, lookahead 10, seed 42, simulated IMU noise enabled, object feedback fused.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for entry in entries:
        values = [entry["dataset"], entry["model"]]
        values.extend(format_value(entry["metrics"].get(key, float("nan"))) for key, _ in METRICS)
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "FK object/HOI metrics are unavailable in these runs and remain NaN.",
            "",
            f"Source JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Validated {len(entries)}/9 model-dataset entries")


if __name__ == "__main__":
    main()