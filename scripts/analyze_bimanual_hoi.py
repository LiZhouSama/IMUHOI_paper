"""Measure bimanual-only HOI budgets and causal geometric replays from online dumps."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.hoi_metrics import compute_hoi_error_at_gt_contact
from utils.rotation_conversions import rotation_6d_to_matrix


def mean_or_nan(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def contact_runs(mask):
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))


def relative_errors(obj, hands, gt_obj, gt_hands):
    return (obj - gt_obj)[:, None, :] - (hands - gt_hands)


def hoi_score(obj, hands, gt_obj, gt_hands, contact):
    errors = np.linalg.norm(relative_errors(obj, hands, gt_obj, gt_hands), axis=-1)
    return mean_or_nan(errors[contact]) * 100.0


def correct_hand_difference(hands, target_difference, mask, gain=1.0):
    corrected = hands.copy()
    correction = 0.5 * gain * (target_difference - (hands[:, 0] - hands[:, 1]))
    corrected[mask, 0] += correction[mask]
    corrected[mask, 1] -= correction[mask]
    return corrected


def causal_hand_constraint(hands, rotations, active, alpha=0.1, gain=0.5):
    """Keep the palm midpoint fixed; use only current/past predicted geometry."""
    difference_local = np.einsum("tji,tj->ti", rotations, hands[:, 0] - hands[:, 1])
    target_local = difference_local.copy()
    for start, stop in contact_runs(active):
        state = difference_local[start].copy()
        for frame_index in range(start, stop):
            state = state + alpha * (difference_local[frame_index] - state)
            target_local[frame_index] = state
    target_world = np.einsum("tij,tj->ti", rotations, target_local)
    return correct_hand_difference(hands, target_world, active, gain)


def analyze_record(record, threshold=0.15):
    def array(key):
        return record[key].detach().cpu().numpy().astype(np.float64, copy=False)

    obj = array("pred_obj_trans_published")
    hands = array("pred_palm_glb_pos")
    gt_obj = array("gt_obj_trans")
    gt_hands = array("gt_palm_pos_eval")
    contact = np.column_stack((array("gt_lhand_contact").reshape(-1),
                               array("gt_rhand_contact").reshape(-1))).astype(bool)
    both = contact.all(axis=1)
    predicted_both = (array("pred_hand_contact_prob")[:, :2] >= threshold).all(axis=1)
    rotations = rotation_6d_to_matrix(record["gt_obj_imu"][:, 3:9]).cpu().numpy()
    gt_rotations = rotation_6d_to_matrix(record["gt_obj_rot"]).cpu().numpy()
    gt_difference = gt_hands[:, 0] - gt_hands[:, 1]
    errors = relative_errors(obj, hands, gt_obj, gt_hands)
    common = errors.mean(axis=1)
    differential = 0.5 * (errors[:, 0] - errors[:, 1])
    contact_count = int(contact.sum())
    both_count = int(both.sum())
    result = {
        "sequence": record.get("seq_file", record.get("seq_name", "")),
        "object": record.get("obj_name", ""),
        "frames": len(obj),
        "contact_hand_frames": contact_count,
        "any_contact_frames": int(contact.any(axis=1).sum()),
        "both_frames": both_count,
        "predicted_both_frames": int(predicted_both.sum()),
        "true_positive_both_frames": int((both & predicted_both).sum()),
        "bimanual_metric_share": 2.0 * both_count / contact_count if contact_count else float("nan"),
        "both_common_cm": mean_or_nan(np.linalg.norm(common[both], axis=-1)) * 100.0,
        "both_differential_cm": mean_or_nan(np.linalg.norm(differential[both], axis=-1)) * 100.0,
        "both_baseline_cm": mean_or_nan(np.linalg.norm(errors[both], axis=-1)) * 100.0,
        "scenarios": {},
    }
    oracle_object = obj.copy()
    oracle_object[both] -= common[both]
    variants = {
        "baseline": (obj, hands),
        "oracle_diff_25": (obj, correct_hand_difference(hands, gt_difference, both, 0.25)),
        "oracle_diff_50": (obj, correct_hand_difference(hands, gt_difference, both, 0.5)),
        "oracle_diff_100": (obj, correct_hand_difference(hands, gt_difference, both)),
        "oracle_object_only_both": (oracle_object, hands),
        "causal_soft": (obj, causal_hand_constraint(hands, rotations, predicted_both)),
        "causal_hard": (obj, causal_hand_constraint(hands, rotations, predicted_both, 0.0, 1.0)),
        "causal_soft_gt_gate": (obj, causal_hand_constraint(hands, rotations, both)),
    }
    for name, (variant_obj, variant_hands) in variants.items():
        result["scenarios"][name] = {
            "hoi_cm": hoi_score(variant_obj, variant_hands, gt_obj, gt_hands, contact),
            "palm_contact_cm": mean_or_nan(np.linalg.norm(variant_hands - gt_hands, axis=-1)[contact]) * 100.0,
            "object_cm": mean_or_nan(np.linalg.norm(variant_obj - gt_obj, axis=-1)) * 100.0,
        }
    official = compute_hoi_error_at_gt_contact(
        torch.from_numpy(obj), torch.from_numpy(gt_obj), torch.from_numpy(hands),
        torch.from_numpy(gt_hands), torch.from_numpy(contact[:, 0]), torch.from_numpy(contact[:, 1]),
    )
    baseline = result["scenarios"]["baseline"]["hoi_cm"]
    if contact_count and not np.isclose(official, baseline, atol=1e-8):
        raise AssertionError((official, baseline))
    if contact_count and result["scenarios"]["oracle_diff_100"]["hoi_cm"] > baseline + 1e-8:
        raise AssertionError("Exact antisymmetric correction increased paired HOI")
    gt_local = np.einsum("tji,tj->ti", gt_rotations, gt_difference)
    onset_residuals = []
    fitted_residuals = []
    segment_lengths = []
    for start, stop in contact_runs(both):
        segment = gt_local[start:stop]
        segment_lengths.append(int(stop - start))
        onset_residuals.extend((0.5 * np.linalg.norm(segment - segment[0], axis=-1) * 100.0).tolist())
        fitted_residuals.extend((0.5 * np.linalg.norm(segment - segment.mean(axis=0), axis=-1) * 100.0).tolist())
    result["gt_onset_rigidity_half_cm"] = mean_or_nan(onset_residuals)
    result["gt_fitted_rigidity_half_cm"] = mean_or_nan(fitted_residuals)
    result["segment_lengths"] = segment_lengths
    return result


def summarize(records):
    summary = {"sequence_count": len(records)}
    for key in ("frames", "contact_hand_frames", "any_contact_frames", "both_frames",
                "predicted_both_frames", "true_positive_both_frames"):
        summary[key] = sum(record[key] for record in records)
    summary["valid_hoi_sequences"] = sum(record["contact_hand_frames"] > 0 for record in records)
    summary["both_contact_frame_fraction"] = summary["both_frames"] / max(summary["any_contact_frames"], 1)
    summary["both_contact_hand_frame_fraction"] = 2 * summary["both_frames"] / max(summary["contact_hand_frames"], 1)
    summary["predicted_both_precision"] = summary["true_positive_both_frames"] / max(summary["predicted_both_frames"], 1)
    summary["predicted_both_recall"] = summary["true_positive_both_frames"] / max(summary["both_frames"], 1)
    for key in ("bimanual_metric_share", "both_common_cm", "both_differential_cm", "both_baseline_cm",
                "gt_onset_rigidity_half_cm", "gt_fitted_rigidity_half_cm"):
        summary[key] = mean_or_nan([record[key] for record in records])
    lengths = [length for record in records for length in record["segment_lengths"]]
    summary["bimanual_segments"] = len(lengths)
    summary["segment_frames_median"] = float(np.median(lengths)) if lengths else None
    summary["scenarios"] = {}
    for name in records[0]["scenarios"]:
        row = {metric: mean_or_nan([record["scenarios"][name][metric] for record in records])
               for metric in ("hoi_cm", "palm_contact_cm", "object_cm")}
        gains = np.array([record["scenarios"]["baseline"]["hoi_cm"] - record["scenarios"][name]["hoi_cm"]
                          for record in records])
        gains = gains[np.isfinite(gains)]
        row["hoi_gain_cm"] = mean_or_nan(gains)
        if gains.size:
            generator = np.random.default_rng(42)
            bootstrap = generator.choice(gains, size=(2000, len(gains)), replace=True).mean(axis=1)
            row["gain_sequence_bootstrap_95_cm"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        summary["scenarios"][name] = row
    return summary


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", action="append", required=True, help="dataset=online_dump.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"threshold": args.threshold, "causal_alpha": 0.1, "causal_gain": 0.5,
              "aggregation": "mean of per-sequence GT-contact hand-frame means; skip undefined HOI",
              "scope": "frozen published object, free palm corrections; no IK or model rerun",
              "datasets": {}}
    for specification in args.dump:
        dataset, source = specification.split("=", 1)
        payload = torch.load(source, map_location="cpu", weights_only=False)
        records = [analyze_record(record, args.threshold) for record in payload["records"]
                   if record.get("has_object", True) and record["publish_len"] > 0]
        protocol = {key: payload.get(key) for key in ("dataset", "variant", "mode", "online_window",
                    "online_lookahead", "seed", "simulate_imu_noise", "refine_human", "data_path")}
        summary = summarize(records)
        report["datasets"][dataset] = {"source": source, "protocol": protocol,
                                       "summary": summary, "sequences": records}
        print(dataset, json.dumps(json_safe(summary), ensure_ascii=True), flush=True)
        del payload
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()