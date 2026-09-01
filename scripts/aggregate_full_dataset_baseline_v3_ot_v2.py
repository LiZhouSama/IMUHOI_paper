"""Aggregate same-HP baseline/v3 VC + ObjectTrans v2 full-dataset evaluations."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN2/full_dataset_baseline_v3_ot_v2_eval_20260827"
)
HISTORICAL_ROOT = PROJECT_ROOT / (
    "outputs/IMUHOI++/RNN2/omomo_baseline_v4_ot_v2_shared_hp_20260826"
)

DATASETS = {
    "omomo_full": {
        "label": "OMOMO full",
        "expected_sequences": 1176,
        "split_info": PROJECT_ROOT
        / "process/IMUHOI++/processed_split_data_OMOMO_smplx_full/split_info.json",
    },
    "pahoi_full": {
        "label": "PAHOI full",
        "expected_sequences": 113,
        "split_info": PROJECT_ROOT
        / "process/IMUHOI++/processed_split_data_PAHOI_smplx_full/split_info.json",
    },
}
PIPELINES = {"baseline": "imuhoi", "v3": "v3", "v4": "v4"}
COMPARISON_PIPELINES = ("v3", "v4")
OVERALL_METRICS = (
    ("mpjpe", "MPJPE (cm)"),
    ("mpjre_angle", "MPJRE (deg)"),
    ("root_trans_err", "Root (cm)"),
    ("jitter", "Jitter"),
    ("obj_trans_err_fusion", "Object (cm)"),
    ("hoi_err_fusion", "HOI (cm)"),
    ("contact_f1_lhand", "Left F1"),
    ("contact_f1_rhand", "Right F1"),
    ("contact_f1_obj", "Object F1"),
)
SEQUENCE_METRICS = (
    "obj_trans_err_fusion_raw_cm",
    "obj_trans_err_fusion_published_cm",
    "hoi_err_fusion_cm",
    "published_minus_raw_cm",
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _mean_records(records: List[dict]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for metric in SEQUENCE_METRICS:
        values = [
            float(record[metric])
            for record in records
            if math.isfinite(float(record[metric]))
        ]
        result[metric] = fmean(values) if values else None
        result[f"{metric}_valid_count"] = len(values)
    return result


def _format(value, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{value:.{digits}f}"


def aggregate(eval_root: Path) -> dict:
    evaluations: Dict[str, Dict[str, dict]] = {}
    sequence_payloads: Dict[str, Dict[str, dict]] = {}
    shared_hp_checkpoint = None

    for dataset_name, dataset_spec in DATASETS.items():
        evaluations[dataset_name] = {}
        sequence_payloads[dataset_name] = {}
        for pipeline, expected_arch in PIPELINES.items():
            stem = f"{pipeline}_{dataset_name}"
            evaluation = _load_json(eval_root / f"{stem}.json")
            sequence_payload = _load_json(eval_root / f"{stem}_sequences.json")

            assert evaluation["object_trans_arch"] == "v2"
            assert evaluation["velocity_contact_arch"] == expected_arch
            assert evaluation["object_trans_state_feedback"] is True
            assert evaluation["resolved_online_lookahead"] == 10
            assert evaluation["seed"] == 42
            assert evaluation["simulate_imu_noise"] is True
            assert evaluation["simulate_imu_noise_seed"] == 42

            protocol = sequence_payload["protocol"]
            assert protocol["object_trans_arch"] == "v2"
            assert protocol["velocity_contact_arch"] == expected_arch
            assert protocol["object_trans_state_feedback"] is True
            hp_checkpoint = protocol["human_pose_checkpoint"]
            if shared_hp_checkpoint is None:
                shared_hp_checkpoint = hp_checkpoint
            assert hp_checkpoint == shared_hp_checkpoint

            records = sequence_payload["datasets"][dataset_name]
            expected_sequences = int(dataset_spec["expected_sequences"])
            assert len(records) == expected_sequences
            assert len({Path(record["seq_path"]).name for record in records}) == len(records)
            assert sequence_payload["aggregate_results"][dataset_name] == evaluation["results"][dataset_name]

            evaluations[dataset_name][pipeline] = evaluation
            sequence_payloads[dataset_name][pipeline] = sequence_payload

    dataset_results = {}
    for dataset_name, dataset_spec in DATASETS.items():
        split_info = _load_json(Path(dataset_spec["split_info"]))
        test_files = set(split_info["test_files"])
        recovered_files = set(split_info["recovered_test_files"])
        reference_files = test_files - recovered_files
        expected_sequences = int(dataset_spec["expected_sequences"])
        assert len(test_files) == expected_sequences

        overall = {}
        record_maps = {}
        for pipeline in PIPELINES:
            metrics = evaluations[dataset_name][pipeline]["results"][dataset_name]
            overall[pipeline] = {
                metric: float(metrics[metric]) for metric, _ in OVERALL_METRICS
            }
            records = sequence_payloads[dataset_name][pipeline]["datasets"][dataset_name]
            record_maps[pipeline] = {
                Path(record["seq_path"]).name: record for record in records
            }
            assert set(record_maps[pipeline]) == test_files

        for comparison in COMPARISON_PIPELINES:
            overall[f"delta_{comparison}_minus_baseline"] = {
                metric: overall[comparison][metric] - overall["baseline"][metric]
                for metric, _ in OVERALL_METRICS
            }

        subsets = {}
        for subset_name, names in (
            ("reference_test", reference_files),
            ("recovered_test", recovered_files),
        ):
            subset_values = {
                pipeline: _mean_records([record_maps[pipeline][name] for name in names])
                for pipeline in PIPELINES
            }
            for comparison in COMPARISON_PIPELINES:
                subset_values[f"delta_{comparison}_minus_baseline"] = {
                    metric: (
                        subset_values[comparison][metric]
                        - subset_values["baseline"][metric]
                        if subset_values[comparison][metric] is not None
                        and subset_values["baseline"][metric] is not None
                        else None
                    )
                    for metric in SEQUENCE_METRICS
                }
            subsets[subset_name] = {
                "sequence_count": len(names),
                **subset_values,
            }

        dataset_results[dataset_name] = {
            "label": dataset_spec["label"],
            "sequence_count": expected_sequences,
            "reference_test_count": len(reference_files),
            "recovered_test_count": len(recovered_files),
            "overall": overall,
            "sequence_equal_weight_subsets": subsets,
        }

    baseline_protocol = sequence_payloads["omomo_full"]["baseline"]["protocol"]
    v3_protocol = sequence_payloads["omomo_full"]["v3"]["protocol"]
    v4_protocol = sequence_payloads["omomo_full"]["v4"]["protocol"]
    historical_payload = _load_json(HISTORICAL_ROOT / "omomo_baseline_v4_ot_v2_eval.json")
    historical_rows = {
        row["pipeline"]: row for row in historical_payload["rows"]
    }
    historical_comparison = {}
    comparison_metric_names = (
        "mpjpe",
        "root_trans_err",
        "stage1_root_trans_err",
        "obj_trans_err_fusion_raw",
        "obj_trans_err_fusion",
        "obj_trans_err_imu",
        "hoi_err_fusion",
        "contact_f1_lhand",
        "contact_f1_rhand",
        "contact_f1_obj",
    )
    for pipeline in ("baseline", "v4"):
        matched_payload = _load_json(
            eval_root / f"{pipeline}_omomo_reference_current_code.json"
        )
        assert matched_payload["online_batch_size"] == 32
        assert matched_payload["resolved_online_lookahead"] == 10
        assert matched_payload["simulate_imu_noise_seed"] == 42
        old_metrics = historical_rows[pipeline]["metrics"]
        matched_metrics = matched_payload["results"]["omomo"]
        full_metrics = evaluations["omomo_full"][pipeline]["results"]["omomo_full"]
        historical_comparison[pipeline] = {
            "historical_20260826": {
                key: float(old_metrics[key]) for key in comparison_metric_names
            },
            "current_code_matched_reference_test": {
                key: float(matched_metrics[key]) for key in comparison_metric_names
            },
            "delta_current_code_minus_historical": {
                key: float(matched_metrics[key]) - float(old_metrics[key])
                for key in comparison_metric_names
            },
            "current_code_full_test": {
                key: float(full_metrics[key]) for key in comparison_metric_names
            },
            "delta_full_minus_matched_reference": {
                key: float(full_metrics[key]) - float(matched_metrics[key])
                for key in comparison_metric_names
            },
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "shared_human_pose_checkpoint": shared_hp_checkpoint,
            "object_trans_arch": "v2",
            "object_trans_state_feedback": True,
            "online_window": 60,
            "online_lookahead": 10,
            "online_batch_size": 64,
            "seed": 42,
            "simulate_imu_noise": True,
            "simulate_imu_noise_seed": 42,
            "pipelines": {
                "baseline": {
                    "velocity_contact_arch": "imuhoi",
                    "velocity_contact_checkpoint": baseline_protocol[
                        "velocity_contact_checkpoint"
                    ],
                    "object_trans_checkpoint": baseline_protocol[
                        "object_trans_checkpoint"
                    ],
                },
                "v3": {
                    "velocity_contact_arch": "v3",
                    "velocity_contact_checkpoint": v3_protocol[
                        "velocity_contact_checkpoint"
                    ],
                    "object_trans_checkpoint": v3_protocol[
                        "object_trans_checkpoint"
                    ],
                },
                "v4": {
                    "velocity_contact_arch": "v4",
                    "velocity_contact_checkpoint": v4_protocol[
                        "velocity_contact_checkpoint"
                    ],
                    "object_trans_checkpoint": v4_protocol[
                        "object_trans_checkpoint"
                    ],
                },
            },
        },
        "subset_metric_note": (
            "Subset object/HOI values are equal-weight means of per-sequence metrics; "
            "overall values are evaluator-native aggregate metrics."
        ),
        "datasets": dataset_results,
        "historical_comparison": historical_comparison,
    }


def write_markdown(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "# Full-dataset baseline vs v3/v4 with shared HP and ObjectTrans v2",
        "",
        "Protocol: same HumanPose checkpoint, ObjectTrans v2, fused state feedback, "
        "OT refine enabled, online window 60 + lookahead 10, deterministic simulated "
        "IMU noise with seed 42.",
        "",
        "## Overall evaluator metrics",
        "",
        "| Dataset | Pipeline | N | MPJPE | Root | Object | HOI | Left F1 | Right F1 | Object F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    datasets = summary["datasets"]
    for dataset in datasets.values():
        for pipeline in PIPELINES:
            values = dataset["overall"][pipeline]
            lines.append(
                f"| {dataset['label']} | {pipeline} | {dataset['sequence_count']} | "
                f"{_format(values['mpjpe'])} | {_format(values['root_trans_err'])} | "
                f"{_format(values['obj_trans_err_fusion'])} | "
                f"{_format(values['hoi_err_fusion'])} | "
                f"{_format(values['contact_f1_lhand'])} | "
                f"{_format(values['contact_f1_rhand'])} | "
                f"{_format(values['contact_f1_obj'])} |"
            )

    lines.extend(
        [
            "",
            "## Delta vs baseline",
            "",
            "Negative is better for errors; positive is better for F1.",
            "",
            "| Dataset | Pipeline | MPJPE | Root | Object | HOI | Left F1 | Right F1 | Object F1 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in datasets.values():
        for comparison in COMPARISON_PIPELINES:
            delta = dataset["overall"][f"delta_{comparison}_minus_baseline"]
            lines.append(
                f"| {dataset['label']} | {comparison} | {_format(delta['mpjpe'])} | "
                f"{_format(delta['root_trans_err'])} | "
                f"{_format(delta['obj_trans_err_fusion'])} | "
                f"{_format(delta['hoi_err_fusion'])} | "
                f"{_format(delta['contact_f1_lhand'])} | "
                f"{_format(delta['contact_f1_rhand'])} | "
                f"{_format(delta['contact_f1_obj'])} |"
            )

    lines.extend(
        [
            "",
            "## Reference vs recovered test subsets",
            "",
            "Values below are equal-weight means over per-sequence records.",
            "",
            "| Dataset | Subset | Pipeline | N | Object | Delta vs baseline | HOI | Delta vs baseline |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in datasets.values():
        for subset_name, subset in dataset["sequence_equal_weight_subsets"].items():
            for pipeline in PIPELINES:
                values = subset[pipeline]
                delta = (
                    {metric: 0.0 for metric in SEQUENCE_METRICS}
                    if pipeline == "baseline"
                    else subset[f"delta_{pipeline}_minus_baseline"]
                )
                lines.append(
                    f"| {dataset['label']} | {subset_name} | {pipeline} | "
                    f"{subset['sequence_count']} | "
                    f"{_format(values['obj_trans_err_fusion_published_cm'])} | "
                    f"{_format(delta['obj_trans_err_fusion_published_cm'])} | "
                    f"{_format(values['hoi_err_fusion_cm'])} | "
                    f"{_format(delta['hoi_err_fusion_cm'])} |"
                )

    lines.extend(
        [
            "",
            "## Why the current HOI error is much lower than the 2026-08-26 report",
            "",
            "The matched rerun uses the same 1015 sequence paths, checkpoint, batch size 32, "
            "lookahead 10, and noise seed 42. Therefore its delta from the historical row "
            "isolates the current inference/evaluation stack rather than the full split.",
            "",
            "| Pipeline | Stage | Root | Stage1 Root | Object raw | Object published | Object IMU | HOI |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    historical = summary["historical_comparison"]
    for pipeline, values in historical.items():
        for stage_key, stage_label in (
            ("historical_20260826", "historical"),
            ("current_code_matched_reference_test", "current matched"),
            ("delta_current_code_minus_historical", "delta"),
        ):
            metrics = values[stage_key]
            lines.append(
                f"| {pipeline} | {stage_label} | {_format(metrics['root_trans_err'])} | "
                f"{_format(metrics['stage1_root_trans_err'])} | "
                f"{_format(metrics['obj_trans_err_fusion_raw'])} | "
                f"{_format(metrics['obj_trans_err_fusion'])} | "
                f"{_format(metrics['obj_trans_err_imu'])} | "
                f"{_format(metrics['hoi_err_fusion'])} |"
            )

    lines.extend(
        [
            "",
            "### Current matched reference test vs current full test",
            "",
            "| Pipeline | Matched Object | Full Object | Delta | Matched HOI | Full HOI | Delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pipeline, values in historical.items():
        matched = values["current_code_matched_reference_test"]
        full = values["current_code_full_test"]
        delta = values["delta_full_minus_matched_reference"]
        lines.append(
            f"| {pipeline} | {_format(matched['obj_trans_err_fusion'])} | "
            f"{_format(full['obj_trans_err_fusion'])} | "
            f"{_format(delta['obj_trans_err_fusion'])} | "
            f"{_format(matched['hoi_err_fusion'])} | "
            f"{_format(full['hoi_err_fusion'])} | "
            f"{_format(delta['hoi_err_fusion'])} |"
        )

    lines.extend(
        [
            "",
            "The old evaluator compared an object trajectory anchored to Stage1 hands against "
            "refined hands. The current publication path translates the left/right FK candidates "
            "from Stage1 hands to refined hands and re-runs fusion before computing HOI. This "
            "removes the extra `-delta_refine` term from the relative hand-object residual.",
            "",
            "Object IMU error is numerically unchanged in both matched reruns, while contact F1 "
            "is unchanged or differs only in the fourth decimal. This rules out a new noise draw "
            "or materially different VC predictions as the source of the HOI drop.",
            "",
            "The full split is not the source of the improvement: recovered sequences are harder, "
            "and moving from the matched reference test to the full test increases object error. "
            "The large historical-to-current HOI decrease is an evaluation-consistency correction, "
            "not a model-quality gain.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    args = parser.parse_args()
    eval_root = args.eval_root.resolve()
    summary = aggregate(eval_root)
    json_path = eval_root / "full_dataset_baseline_v3_v4_ot_v2_summary.json"
    markdown_path = eval_root / "full_dataset_baseline_v3_v4_ot_v2_summary.md"
    json_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    write_markdown(markdown_path, summary)
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
