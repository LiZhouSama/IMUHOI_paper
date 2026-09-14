#!/usr/bin/env python3
"""CPU-only calibration AFTER the existing HOI publisher; no model inference.

fit --dump train_online.pt --split-manifest development_split/split_manifest.json
    --output calibration.json [--report fit_report.json] [--grid-points 41]
evaluate --dump online.pt --calibration calibration.json --output evaluation.json

The manifest source must be an existing train directory, identical to the cache
source. Every manifest row must be present exactly once, with disjoint train /
validation groups. Only TRAIN selects lambda on an ascending uniform [0, 1] grid
(ties within 1e-9 cm prefer smaller lambda). Validation may only accept that ONE
candidate if HOI improves and object error does not worsen, or fall back to zero.
Evaluate never searches, rejects or changes the locked lambda, even on test data.

All optimization/scoring uses the SAME final pred_palm_glb_pos and canonical
GT gt_palm_pos_eval. The four audit HOIs, in order, are raw/OT-hand/dataset-GT,
raw/OT-hand/eval-GT, raw/final-hand/eval-GT, published/final-hand/eval-GT.
The exact object-only oracle allows arbitrary per-frame object positions; it is
not a causal estimator and imposes no initial-position or smoothness constraint.

Metric formulas match eval: sequence-mean cm; HOI averages GT contact hand-frames within
each sequence (both hands count twice), then sequences with contact. Jitter is
mean second-difference norm in mm/frame^2, without FPS scaling, on the named
trajectory (baseline=published; raw_final=raw). The existing evaluator reports
raw jitter, so compare that with raw_final, not published jitter. Undefined metrics
are JSON null. Inputs are trusted local tensor-only torch.save dump dictionaries;
loading expects a trusted local research cache and uses map_location='cpu'. No checkpoints,
source datasets, networks or CUDA contexts are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

METHOD = "hoi_publisher_scalar_v1"
EPS_CM = 1e-9
PROTOCOL_KEYS = (
    "mode", "variant", "object_trans_arch", "velocity_contact_arch",
    "online_window", "online_lookahead", "seed", "simulate_imu_noise",
    "refine_human", "module_paths", "checkpoint_sha256",
)
FOUR_WAY_KEYS = (
    "raw_ot_palm_dataset_gt", "raw_ot_palm_eval_gt",
    "raw_final_palm_eval_gt", "published_final_palm_eval_gt",
)


def _lambda(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("lambda must be a finite number in [0, 1]")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("lambda must be a finite number in [0, 1]")
    return value


def calibrate_positions(published: torch.Tensor, raw: torch.Tensor,
                        lambda_value: float) -> torch.Tensor:
    """Apply the fixed publisher correction only; never change the human/state."""
    value = _lambda(lambda_value)
    if published.shape != raw.shape:
        raise ValueError("raw and published object shapes must agree")
    if value == 0.0:
        return published  # Exact identity, not an arithmetic approximation.
    if value == 1.0:
        return raw
    return published + value * (raw - published)


def hoi_error_cm(pred_obj, gt_obj, pred_palm, gt_palm, contact) -> float | None:
    if not bool(contact.any()):
        return None
    residual = (pred_obj[:, None] - pred_palm) - (gt_obj[:, None] - gt_palm)
    return float(torch.linalg.vector_norm(residual, dim=-1)[contact].mean()) * 100.0


def object_only_oracle_cm(pred_palm, gt_palm, contact) -> float | None:
    """Exact unrestricted object-only minimum of sequence HOI, not a predictor.

    A single contact has minimum zero. With two contacts the sum of distances
    to the two desired object positions is at least their separation, attained
    anywhere on the segment between them. Normalize by ALL contact hand-frames.
    """
    count = int(contact.sum())
    if count == 0:
        return None
    hand_error = pred_palm - gt_palm
    difference = hand_error[:, 0] - hand_error[:, 1]
    both = contact.all(dim=-1)
    return float(torch.linalg.vector_norm(difference[both], dim=-1).sum()) * 100.0 / count


def _tensor(record, key, shape, binary=False):
    value = record.get(key)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{record.get('seq_file', '?')}: {key} must have shape {shape}")
    value = value.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{key} contains non-finite values")
    if binary:
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError(f"{key} must contain binary contact labels")
        return value.bool()
    return value


def prepare_record(record: dict) -> dict:
    """Strict schema: never silently substitute raw GT or Stage-1 for final FK."""
    length = record.get("publish_len")
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("Each record requires non-negative integer publish_len")
    seq_file = record.get("seq_file")
    if not isinstance(seq_file, str) or not seq_file or Path(seq_file).suffix != ".pt":
        raise ValueError("Each record requires an explicit .pt seq_file")
    ot_key = "ot_hand_positions" if "ot_hand_positions" in record else "stage1_pred_palm_glb_pos"
    return {
        "file": Path(seq_file).name, "seq_file": seq_file,
        "seq_name": str(record.get("seq_name", "")),
        "object": str(record.get("obj_name", "")), "frames": length,
        "has_object": bool(record.get("has_object", True)),
        "raw": _tensor(record, "pred_obj_trans", (length, 3)),
        "published": _tensor(record, "pred_obj_trans_published", (length, 3)),
        "gt_obj": _tensor(record, "gt_obj_trans", (length, 3)),
        "final_palm": _tensor(record, "pred_palm_glb_pos", (length, 2, 3)),
        "ot_palm": _tensor(record, ot_key, (length, 2, 3)),
        "gt_eval": _tensor(record, "gt_palm_pos_eval", (length, 2, 3)),
        "gt_dataset": _tensor(record, "gt_palm_pos", (length, 2, 3)),
        "contact": torch.stack((
            _tensor(record, "gt_lhand_contact", (length,), binary=True),
            _tensor(record, "gt_rhand_contact", (length,), binary=True),
        ), dim=-1),
    }


def _online_records(payload):
    if not isinstance(payload, dict) or payload.get("mode") != "online":
        raise ValueError("Only an online cache is allowed")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Cache provenance must be a dictionary")
    for metadata in (payload, provenance):
        if metadata.get("feedback_mode", "normal") != "normal":
            raise ValueError("GT/perturbed feedback caches are forbidden; require normal feedback")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Cache records must be a nonempty list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Cache records must be dictionaries")
    return records


def _source_path(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must explicitly identify a directory")
    return Path(value).expanduser().resolve()


def _in_source(value, source, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} requires an explicit file path")
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else source / path).resolve()
    if path.parent != source or path.suffix != ".pt":
        raise ValueError(f"{label} is outside the manifest train source: {value}")
    return path


def training_partition(payload: dict, manifest: dict) -> tuple[list, list, dict]:
    """Validate complete provenance/membership BEFORE reading any GT tensors."""
    records = _online_records(payload)
    source = _source_path(manifest.get("source"), "Manifest source")
    if source.name != "train" or not source.is_dir():
        raise ValueError("Manifest source must be an existing train directory, never test")
    if _source_path(payload.get("data_path"), "Cache data_path") != source:
        raise ValueError("Cache source must equal the manifest train directory; test caches are forbidden")
    for metadata in (payload, payload.get("provenance", {})):
        if metadata.get("split", "train") != "train":
            raise ValueError("Declared cache split must be train, never test")
        if "data_path" in metadata and _source_path(metadata["data_path"], "Cache provenance") != source:
            raise ValueError("Cache provenance disagrees with the manifest train source")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest rows must be nonempty")
    membership, groups = {}, {"train": set(), "validation": set()}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Manifest rows must be dictionaries")
        name, group = row.get("file"), row.get("group")
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix != ".pt":
            raise ValueError("Manifest file must be a .pt basename")
        if name in membership:
            raise ValueError(f"Duplicate manifest file: {name}")
        partition = row.get("partition")
        partition = "validation" if partition == "val" else partition
        if partition not in groups:
            raise ValueError("Manifest partition must be train or validation/val, never test")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Manifest group must be a nonempty string")
        expected = _in_source(name, source, "Manifest file")
        if "path" in row and _in_source(row["path"], source, "Manifest path") != expected:
            raise ValueError("Manifest path disagrees with its file")
        membership[name] = {"partition": partition, "group": group}
        groups[partition].add(group)
    if not all(groups.values()) or groups["train"] & groups["validation"]:
        raise ValueError("Both partitions need nonempty, disjoint groups")
    indexed = {}
    for record in records:
        path = _in_source(record.get("seq_file"), source, "Cache seq_file")
        if path.name in indexed:
            raise ValueError(f"Duplicate cache file: {path.name}")
        if path.name not in membership:
            raise ValueError(f"Unexpected cache file outside manifest: {path.name}")
        indexed[path.name] = record
    missing = set(membership) - set(indexed)
    if missing:
        raise ValueError(f"Cache is missing {len(missing)} manifest files; partial fitting is forbidden")
    prepared = {name: prepare_record(indexed[name]) for name in sorted(indexed)}
    partitions = {
        partition: [prepared[name] for name in prepared if membership[name]["partition"] == partition]
        for partition in groups
    }
    return partitions["train"], partitions["validation"], membership


def _mean(values):
    valid = [value for value in values if value is not None]
    return math.fsum(valid) / len(valid) if valid else None


def _metrics(record, position):
    if not record["has_object"] or record["frames"] == 0:
        return {"obj_err_cm": None, "hoi_err_cm": None, "obj_jitter_mm_per_frame2": None}
    return {
        "obj_err_cm": float(torch.linalg.vector_norm(position - record["gt_obj"], dim=-1).mean()) * 100.0,
        "hoi_err_cm": hoi_error_cm(position, record["gt_obj"], record["final_palm"], record["gt_eval"], record["contact"]),
        "obj_jitter_mm_per_frame2": (
            float(torch.linalg.vector_norm(position[2:] - 2 * position[1:-1] + position[:-2], dim=-1).mean()) * 1000.0
            if record["frames"] >= 3 else None
        ),
    }


def _summary(rows):
    metric_sets = ("baseline", "raw_final", "candidate", "calibrated", "hoi_four_way_cm")
    result = {
        "sequences": len(rows),
        "object_metric_sequences": sum(row["baseline"]["obj_err_cm"] is not None for row in rows),
        "hoi_metric_sequences": sum(row["baseline"]["hoi_err_cm"] is not None for row in rows),
        "contact_hand_frames": sum(row["contact_hand_frames"] for row in rows),
        "oracle_object_only_hoi_cm": _mean(row["oracle_object_only_hoi_cm"] for row in rows),
    }
    for name in metric_sets:
        result[name] = {key: _mean(row[name][key] for row in rows) for key in rows[0][name]} if rows else {}
    baseline, calibrated = result["baseline"].get("hoi_err_cm"), result["calibrated"].get("hoi_err_cm")
    bound = result["oracle_object_only_hoi_cm"]
    result["oracle_max_baseline_hoi_reduction_cm"] = baseline - bound if baseline is not None and bound is not None else None
    result["calibrated_minus_oracle_hoi_cm"] = calibrated - bound if calibrated is not None and bound is not None else None
    return result


def score_records(records, lambda_value, membership=None, candidate_lambda=None):
    """Report fixed-lambda metrics; this function does not fit or select anything."""
    lambda_value = _lambda(lambda_value)
    candidate_lambda = lambda_value if candidate_lambda is None else _lambda(candidate_lambda)
    rows = []
    for record in records:
        active = record["has_object"] and record["frames"] > 0
        p, h, gt, c = record["raw"], record["ot_palm"], record["gt_obj"], record["contact"]
        four = (
            hoi_error_cm(p, gt, h, record["gt_dataset"], c),
            hoi_error_cm(p, gt, h, record["gt_eval"], c),
            hoi_error_cm(p, gt, record["final_palm"], record["gt_eval"], c),
            hoi_error_cm(record["published"], gt, record["final_palm"], record["gt_eval"], c),
        ) if active else (None,) * 4
        row = {key: record[key] for key in ("file", "seq_file", "seq_name", "object", "frames")}
        row.update((membership or {}).get(record["file"], {"partition": "evaluation", "group": None}))
        row.update({
            "contact_hand_frames": int(c.sum()) if active else 0,
            "bimanual_frames": int(c.all(-1).sum()) if active else 0,
            "baseline": _metrics(record, record["published"]),
            "raw_final": _metrics(record, p),
            "candidate": _metrics(record, calibrate_positions(record["published"], p, candidate_lambda)),
            "calibrated": _metrics(record, calibrate_positions(record["published"], p, lambda_value)),
            "hoi_four_way_cm": dict(zip(FOUR_WAY_KEYS, four)),
            "oracle_object_only_hoi_cm": object_only_oracle_cm(record["final_palm"], record["gt_eval"], c) if active else None,
        })
        rows.append(row)
    return {
        "lambda": lambda_value, "candidate_lambda": candidate_lambda,
        "summary": _summary(rows),
        "by_object": {name: _summary([row for row in rows if row["object"] == name]) for name in sorted({r["object"] for r in rows})},
        "per_sequence": rows,
    }


def _train_objective(records, value):
    return _mean(
        hoi_error_cm(calibrate_positions(r["published"], r["raw"], value), r["gt_obj"], r["final_palm"], r["gt_eval"], r["contact"])
        for r in records if r["has_object"] and r["frames"] > 0
    )


def _protocol(payload):
    protocol = {key: payload[key] for key in PROTOCOL_KEYS if key in payload}
    provenance = payload.get("provenance", {})
    # Schema-v2 dumps supply content hashes. Compare contents, not relocation,
    # timestamps, sequence selection, source split or elapsed runtime.
    if "checkpoints" in provenance:
        protocol["checkpoint_sha256"] = {
            name: info["sha256"] for name, info in provenance["checkpoints"].items()
        }
    if "config" in provenance:
        protocol["config_sha256"] = provenance["config"]["sha256"]
    if "source_files" in provenance:
        protocol["source_sha256"] = {
            name: info["sha256"] for name, info in provenance["source_files"].items()
        }
    if "gt_palm_pos_eval" in provenance:
        target = provenance["gt_palm_pos_eval"]
        protocol["eval_gt_geometry"] = {
            key: target[key] for key in ("definition", "body_model_type", "betas", "hands") if key in target
        }
        protocol["eval_gt_geometry"]["body_model_sha256"] = target["body_model"]["sha256"]
    for key in ("initialization", "feedback_mode", "published_object_trans"):
        if key in provenance:
            protocol[key] = provenance[key]
    return protocol


def _json_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fit_calibration(payload, manifest, grid_points=41, provenance=None):
    if isinstance(grid_points, bool) or not isinstance(grid_points, int) or not 2 <= grid_points <= 1001:
        raise ValueError("grid_points must be an integer between 2 and 1001")
    train, validation, membership = training_partition(payload, manifest)
    grid = [{"lambda": index / (grid_points - 1), "train_hoi_cm": _train_objective(train, index / (grid_points - 1))} for index in range(grid_points)]
    if grid[0]["train_hoi_cm"] is None:
        raise ValueError("Train partition has no scored GT contact hand-frames")
    best = grid[0]
    for candidate in grid[1:]:
        if candidate["train_hoi_cm"] < best["train_hoi_cm"] - EPS_CM:
            best = candidate
    fitted = best["lambda"]
    # Exactly one validation candidate; never select another grid value here.
    validation_check = score_records(validation, fitted)["summary"]
    baseline, candidate = validation_check["baseline"], validation_check["calibrated"]
    if baseline["hoi_err_cm"] is None:
        raise ValueError("Validation partition has no scored GT contact hand-frames")
    hoi_improved = candidate["hoi_err_cm"] < baseline["hoi_err_cm"] - EPS_CM
    obj_not_worse = candidate["obj_err_cm"] <= baseline["obj_err_cm"] + EPS_CM
    accepted = fitted != 0.0 and hoi_improved and obj_not_worse
    selected = fitted if accepted else 0.0
    reason = "accepted_train_candidate" if accepted else (
        "train_selected_zero" if fitted == 0.0 else
        "validation_hoi_not_improved" if not hoi_improved else "validation_object_worsened"
    )
    lock = {
        "schema_version": 1, "method": METHOD, "locked": True,
        "lambda": selected, "train_fitted_lambda": fitted, "accepted": accepted,
        "protocol": _protocol(payload),
        "provenance": {**(provenance or {}), "source": str(Path(manifest["source"]).resolve()),
                       "membership_sha256": _json_hash(membership)},
        "selection": {
            "rule": "Train-only uniform grid argmin of canonical final-hand sequence-mean HOI; smallest lambda tie-break. Validation accepts this candidate only if HOI improves and object error does not worsen; otherwise lambda=0. No validation retuning.",
            "grid_points": grid_points, "comparison_tolerance_cm": EPS_CM,
            "train_sequences": len(train), "validation_sequences": len(validation),
            "train_baseline_hoi_cm": grid[0]["train_hoi_cm"], "train_candidate_hoi_cm": best["train_hoi_cm"],
            "validation_baseline": baseline, "validation_candidate": candidate,
            "validation_hoi_improved": hoi_improved, "validation_object_not_worse": obj_not_worse,
            "reason": reason,
        },
    }
    lock["calibration_id"] = _json_hash(lock)
    report = {
        "method": METHOD, "calibration_id": lock["calibration_id"],
        "selection": lock["selection"], "train_grid": grid,
        "train": score_records(train, selected, membership, fitted),
        "validation": score_records(validation, selected, membership, fitted),
        "oracle_scope": "Unrestricted per-frame object-only minimum with final palms fixed; GT is diagnostic only, not an inference input. No causal/initialization/smoothness constraints.",
    }
    return lock, report


def validate_calibration(calibration):
    if not isinstance(calibration, dict) or calibration.get("schema_version") != 1 or calibration.get("method") != METHOD or calibration.get("locked") is not True:
        raise ValueError("Expected a locked scalar publisher calibration")
    if calibration.get("calibration_id") != _json_hash({k: v for k, v in calibration.items() if k != "calibration_id"}):
        raise ValueError("Calibration integrity check failed; do not edit the locked JSON")
    return _lambda(calibration.get("lambda"))


def evaluate_calibration(payload, calibration):
    """Test GT is used only for reporting; never refit, reject or tune lambda."""
    value = validate_calibration(calibration)
    records = _online_records(payload)
    protocol = _protocol(payload)
    for key, expected in calibration["protocol"].items():
        if key not in protocol or protocol[key] != expected:
            raise ValueError(f"Evaluation cache protocol mismatch: {key}")
    prepared = [prepare_record(record) for record in records]
    if len({r["file"] for r in prepared}) != len(prepared):
        raise ValueError("Duplicate evaluation cache file")
    report = score_records(prepared, value)
    report.update({"method": METHOD, "calibration_id": calibration["calibration_id"],
                   "protocol": protocol, "data_path": payload.get("data_path"),
                   "selection_performed": False})
    return report


def _file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_json_new(path, payload):
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit on manifest train rows; validate once; save a locked scalar")
    fit.add_argument("--dump", required=True)
    fit.add_argument("--split-manifest", "--split_manifest", required=True)
    fit.add_argument("--output", required=True, help="New small locked calibration JSON; never overwritten")
    fit.add_argument("--report", help="Detailed report; default <output stem>.report.json")
    fit.add_argument("--grid-points", "--grid_points", type=int, default=41)
    evaluate = commands.add_parser("evaluate", help="Score a cache using the locked lambda, without selection")
    evaluate.add_argument("--dump", required=True)
    evaluate.add_argument("--calibration", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    # This is a trusted, locally generated research cache with provenance, not a weights file.
    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    cache_info = {"cache_path": str(Path(args.dump).resolve()), "cache_sha256": _file_hash(args.dump)}
    if args.command == "fit":
        manifest_path = Path(args.split_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output = Path(args.output)
        report_path = Path(args.report) if args.report else output.with_name(output.stem + ".report.json")
        if output.resolve() == report_path.resolve() or output.exists() or report_path.exists():
            raise ValueError("Calibration/report outputs must be distinct new files")
        lock, report = fit_calibration(payload, manifest, args.grid_points, {
            **cache_info, "split_manifest": str(manifest_path.resolve()), "split_sha256": _file_hash(manifest_path),
        })
        _write_json_new(output, lock)
        _write_json_new(report_path, report)
        print(json.dumps({"lambda": lock["lambda"], "train_fitted_lambda": lock["train_fitted_lambda"],
                          "reason": lock["selection"]["reason"], "calibration": str(output), "report": str(report_path)}, allow_nan=False))
    else:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        report = evaluate_calibration(payload, calibration)
        report["cache"] = cache_info
        _write_json_new(args.output, report)
        print(json.dumps({"lambda": report["lambda"], "summary": report["summary"], "report": args.output}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
