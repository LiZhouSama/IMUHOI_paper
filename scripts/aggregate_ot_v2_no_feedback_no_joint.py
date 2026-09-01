"""Compare joint/feedback controls for the OT-v2 no-feedback experiment."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN2/omomo_baseline_v4_ot_v2_no_feedback_no_joint_20260829"
)
JOINT_EVAL_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN2/full_dataset_baseline_v3_ot_v2_eval_20260827"
)
PIPELINES = {"baseline": "imuhoi", "v4": "v4"}
DATASETS = {"omomo_full": 1176, "pahoi_full": 113}
VARIANTS = (
    "feedback_joint",
    "feedback_no_joint",
    "no_feedback_no_joint",
)
METRICS = (
    "mpjpe",
    "root_trans_err",
    "obj_jitter",
    "obj_trans_err_fusion_raw",
    "obj_trans_err_fusion",
    "hoi_err_fusion",
    "contact_f1_lhand",
    "contact_f1_rhand",
    "contact_f1_obj",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metrics_from(path: Path, dataset: str) -> dict:
    payload = load_json(path)
    metrics = payload.get("results", {}).get(dataset)
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing {dataset} metrics in {path}")
    return payload, {key: float(metrics[key]) for key in METRICS}


def delta(left: dict, right: dict) -> dict:
    return {key: float(left[key]) - float(right[key]) for key in METRICS}


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe(item) for item in value]
    return value


def fmt(value) -> str:
    return f"{float(value):.4f}"


def read_eval_duration(path: Path) -> float:
    match = re.search(r"评估耗时:\s*([0-9.]+)秒", path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Evaluation duration not found in {path}")
    return float(match.group(1))


def aggregate() -> dict:
    manifest = load_json(EXPERIMENT_ROOT / "experiment_manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["object_trans_arch"] == "v2"
    assert manifest["object_trans_state_feedback"] is False
    assert manifest["joint_train"] is False

    results = {}
    for pipeline, vc_arch in PIPELINES.items():
        pipeline_root = EXPERIMENT_ROOT / pipeline
        results[pipeline] = {}
        for dataset, expected_count in DATASETS.items():
            joint_path = JOINT_EVAL_ROOT / f"{pipeline}_{dataset}.json"
            old_path = pipeline_root / f"eval_old_feedback_no_joint_{dataset}.json"
            new_path = pipeline_root / f"eval_new_no_feedback_no_joint_{dataset}.json"
            joint_payload, joint_metrics = metrics_from(joint_path, dataset)
            old_payload, old_metrics = metrics_from(old_path, dataset)
            new_payload, new_metrics = metrics_from(new_path, dataset)

            assert joint_payload["object_trans_arch"] == "v2"
            assert old_payload["object_trans_arch"] == "v2"
            assert new_payload["object_trans_arch"] == "v2"
            assert joint_payload["velocity_contact_arch"] == vc_arch
            assert old_payload["velocity_contact_arch"] == vc_arch
            assert new_payload["velocity_contact_arch"] == vc_arch
            assert joint_payload["object_trans_state_feedback"] is True
            assert old_payload["object_trans_state_feedback"] is True
            assert new_payload["object_trans_state_feedback"] is False
            assert joint_payload["simulate_imu_noise_seed"] == 42
            assert old_payload["simulate_imu_noise_seed"] == 42
            assert new_payload["simulate_imu_noise_seed"] == 42

            old_records = load_json(
                pipeline_root / f"eval_old_feedback_no_joint_{dataset}_sequences.json"
            )["datasets"][dataset]
            new_records = load_json(
                pipeline_root / f"eval_new_no_feedback_no_joint_{dataset}_sequences.json"
            )["datasets"][dataset]
            assert len(old_records) == expected_count
            assert len(new_records) == expected_count
            assert {Path(row["seq_path"]).name for row in old_records} == {
                Path(row["seq_path"]).name for row in new_records
            }

            for key in (
                "contact_f1_lhand",
                "contact_f1_rhand",
                "contact_f1_obj",
                "obj_trans_err_imu",
            ):
                old_value = float(old_payload["results"][dataset][key])
                new_value = float(new_payload["results"][dataset][key])
                assert old_value == new_value, (pipeline, dataset, key)

            variants = {
                "feedback_joint": joint_metrics,
                "feedback_no_joint": old_metrics,
                "no_feedback_no_joint": new_metrics,
            }
            old_duration = read_eval_duration(
                pipeline_root / f"eval_old_feedback_no_joint_{dataset}.log"
            )
            new_duration = read_eval_duration(
                pipeline_root / f"eval_new_no_feedback_no_joint_{dataset}.log"
            )
            results[pipeline][dataset] = {
                "sequence_count": expected_count,
                "variants": variants,
                "evaluation_duration_seconds": {
                    "feedback_no_joint": old_duration,
                    "no_feedback_no_joint": new_duration,
                    "speedup_percent": (old_duration - new_duration) / old_duration * 100.0,
                },
                "deltas": {
                    "remove_joint_keep_feedback": delta(
                        variants["feedback_no_joint"], variants["feedback_joint"]
                    ),
                    "remove_feedback_without_joint": delta(
                        variants["no_feedback_no_joint"],
                        variants["feedback_no_joint"],
                    ),
                    "requested_vs_feedback_joint": delta(
                        variants["no_feedback_no_joint"], variants["feedback_joint"]
                    ),
                },
            }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "shared_human_pose_checkpoint": manifest["shared_hp_checkpoint"],
            "object_trans_arch": "v2",
            "seed": 42,
            "simulate_imu_noise_seed": 42,
            "online_window": 60,
            "online_lookahead": 10,
            "new_training": {
                "object_trans_state_feedback": False,
                "joint_train": False,
                "epochs": 300,
            },
            "comparison_note": (
                "feedback_joint uses the prior joint VC/OT checkpoints; "
                "feedback_no_joint and no_feedback_no_joint use the same frozen Stage2 VC."
            ),
        },
        "pipelines": results,
    }


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# OT-v2 state-feedback and joint-training comparison",
        "",
        "All rows use the same HumanPose checkpoint, full-sequence evaluation, window 60, "
        "lookahead 10, and deterministic simulated IMU noise with seed 42.",
        "",
        "`feedback_joint` uses the previous joint VC/OT checkpoints. The two OT-only rows "
        "use the same frozen Stage2 VC, so their contact F1 and IMU object baseline are "
        "asserted to be numerically identical.",
        "",
        "## Metrics",
        "",
        "| Dataset | Pipeline | Variant | Object raw | Object published | HOI | Obj jitter | Root |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for pipeline, datasets in summary["pipelines"].items():
        for dataset, result in datasets.items():
            for variant in VARIANTS:
                values = result["variants"][variant]
                lines.append(
                    f"| {dataset} | {pipeline} | {variant} | "
                    f"{fmt(values['obj_trans_err_fusion_raw'])} | "
                    f"{fmt(values['obj_trans_err_fusion'])} | "
                    f"{fmt(values['hoi_err_fusion'])} | "
                    f"{fmt(values['obj_jitter'])} | "
                    f"{fmt(values['root_trans_err'])} |"
                )

    lines.extend(
        [
            "",
            "## Effect of disabling feedback under OT-only training",
            "",
            "Delta is `no_feedback_no_joint - feedback_no_joint`; negative is better.",
            "",
            "| Dataset | Pipeline | Object raw delta | Object delta | HOI delta | Obj jitter delta | Root delta |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pipeline, datasets in summary["pipelines"].items():
        for dataset, result in datasets.items():
            values = result["deltas"]["remove_feedback_without_joint"]
            lines.append(
                f"| {dataset} | {pipeline} | "
                f"{fmt(values['obj_trans_err_fusion_raw'])} | "
                f"{fmt(values['obj_trans_err_fusion'])} | "
                f"{fmt(values['hoi_err_fusion'])} | "
                f"{fmt(values['obj_jitter'])} | "
                f"{fmt(values['root_trans_err'])} |"
            )

    lines.extend(
        [
            "",
            "## Requested configuration vs previous feedback+joint result",
            "",
            "Delta is `no_feedback_no_joint - feedback_joint`; this combines removing "
            "feedback with removing joint training and reverting to the frozen Stage2 VC.",
            "",
            "| Dataset | Pipeline | Object delta | HOI delta | Obj jitter delta | Root delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for pipeline, datasets in summary["pipelines"].items():
        for dataset, result in datasets.items():
            values = result["deltas"]["requested_vs_feedback_joint"]
            lines.append(
                f"| {dataset} | {pipeline} | "
                f"{fmt(values['obj_trans_err_fusion'])} | "
                f"{fmt(values['hoi_err_fusion'])} | "
                f"{fmt(values['obj_jitter'])} | "
                f"{fmt(values['root_trans_err'])} |"
            )

    lines.extend(
        [
            "",
            "## Evaluation speed",
            "",
            "| Dataset | Pipeline | Feedback seconds | No-feedback seconds | Speedup |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for pipeline, datasets in summary["pipelines"].items():
        for dataset, result in datasets.items():
            duration = result["evaluation_duration_seconds"]
            lines.append(
                f"| {dataset} | {pipeline} | "
                f"{duration['feedback_no_joint']:.2f} | "
                f"{duration['no_feedback_no_joint']:.2f} | "
                f"{duration['speedup_percent']:.2f}% |"
            )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Disabling state feedback activates the vectorized window-level LSTM fast path, "
            "but it does not improve accuracy in this experiment. Both pipelines regress on "
            "OMOMO object and HOI errors relative to their feedback-enabled OT-only controls. "
            "On PAHOI, baseline also regresses; v4 has slightly worse object error but slightly "
            "better HOI. The speed-oriented no-feedback default is therefore a compute/accuracy "
            "tradeoff rather than a free quality improvement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = aggregate()
    json_path = EXPERIMENT_ROOT / "ot_v2_no_feedback_no_joint_comparison.json"
    markdown_path = EXPERIMENT_ROOT / "ot_v2_no_feedback_no_joint_comparison.md"
    json_path.write_text(
        json.dumps(safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(markdown_path, summary)
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
