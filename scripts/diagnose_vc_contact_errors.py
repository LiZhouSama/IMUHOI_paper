"""Localize VC contact errors in time, relative to ground-truth transitions.

Aggregate contact F1 is dominated by segment interiors (~73% of frames on
OMOMO are hand-contact frames), so it cannot show what a transition-aware
change did.  This script reuses the batch-online prediction path from
``eval_batch_online`` and then bins every frame by its signed distance to the
nearest ground-truth contact onset/offset, reporting false-positive and
false-negative rates per bin for one or more VC checkpoints.

Use it to answer, for a change aimed at interaction boundaries:
  * did the error actually move at the boundaries, or only in the interiors?
  * are new false positives concentrated in the pre-onset window (which would
    implicate an anticipation objective leaking into the contact head)?

Runs are given as ``NAME=ARCH:CKPT`` and may append ``:zero_boundary`` to
replace the pose-prior hidden state with zeros, which measures how much the
contact heads actually depend on it.  For architectures that expose pose-only
auxiliary heads it also reports their average precision against the frame base
rate, so "the pose stream encodes upcoming transitions" becomes a number.

Example::

    python scripts/diagnose_vc_contact_errors.py \
        --hp_ckpt .../hp/best.pt \
        --run v3=v3:.../v3/.../best_velocity_contact.pt \
        --run v4=v4:.../v4/.../best_velocity_contact.pt \
        --run v4_zeroed=v4:.../v4/.../best_velocity_contact.pt:zero_boundary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import copy

import eval_batch_online as ebo
from dataset.dataset_IMUHOI import IMUDataset
from eval_IMUHOI import _set_eval_seed, _unwrap_model, get_default_dataset_config
from model.rnn.imuhoi_model import load_model
from model.rnn.online import resolve_online_lookahead, resolve_online_window
from utils.dataset_config import get_full_sequence_test_path, resolve_dataset_path
from utils.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/IMUHOI_train_rnn.yaml")
    parser.add_argument("--hp_ckpt", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=ARCH:VC_CKPT",
        help="Named VC checkpoint, e.g. v4=v4:/path/best_velocity_contact.pt",
    )
    parser.add_argument("--dataset", default="omomo")
    parser.add_argument("--max_sequences", type=int, default=None)
    parser.add_argument("--online_batch_size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15,
                        help="Pre-onset window to report separately; match v4_anticipation_horizon.")
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def parse_run(spec: str):
    """NAME=ARCH:CKPT[:zero_boundary] -> (name, arch, path, zero_boundary)."""
    name, _, rest = spec.partition("=")
    arch, _, tail = rest.partition(":")
    ckpt, _, option = tail.partition(":")
    if not name or not arch or not ckpt:
        raise ValueError(f"--run must be NAME=ARCH:CKPT[:zero_boundary], got {spec!r}")
    if option and option != "zero_boundary":
        raise ValueError(f"unknown run option {option!r}")
    return name, arch, Path(ckpt), option == "zero_boundary"


def zero_boundary_features(velocity_contact):
    """Replace the pose-prior hidden state with zeros, keeping every shape.

    The contact heads still receive a boundary block of the same width, so any
    metric change is exactly the contribution the pose prior was making.
    """
    original = velocity_contact._encode_body_groups

    def zeroed(pose_streams):
        return torch.zeros_like(original(pose_streams))

    velocity_contact._encode_body_groups = zeroed
    return velocity_contact


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, computed without sklearn."""
    if labels.sum() == 0 or labels.size == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    true_positives = np.cumsum(labels)
    precision = true_positives / np.arange(1, labels.size + 1)
    return float((precision * labels).sum() / labels.sum())


def score_quantiles(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"count": 0}
    values = np.quantile(scores, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(scores.size),
        "p01": float(values[0]),
        "p05": float(values[1]),
        "p25": float(values[2]),
        "p50": float(values[3]),
        "p75": float(values[4]),
        "p95": float(values[5]),
        "p99": float(values[6]),
    }


def signed_distance_to_transitions(mask: np.ndarray) -> np.ndarray:
    """Signed frame distance to the nearest onset/offset of a boolean channel.

    Negative before a transition, positive after, ``+inf`` when the channel
    never changes state in the sequence.
    """
    length = int(mask.shape[0])
    if length < 2:
        return np.full(length, np.inf)
    changes = np.nonzero(mask[1:] != mask[:-1])[0] + 1
    if changes.size == 0:
        return np.full(length, np.inf)
    frames = np.arange(length)
    delta = frames[:, None] - changes[None, :]
    nearest = np.argmin(np.abs(delta), axis=1)
    return delta[frames, nearest].astype(np.float64)


BINS = (
    (-np.inf, -30, "pre  <-30"),
    (-30, -15, "pre  -30..-16"),
    (-15, -5, "pre  -15..-6"),
    (-5, 0, "pre   -5..-1"),
    (0, 5, "post   0..4"),
    (5, 15, "post   5..14"),
    (15, 30, "post  15..29"),
    (30, np.inf, "post >=30"),
)


def main():
    args = parse_args()
    runs = [parse_run(spec) for spec in args.run]
    config = load_config(str(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_configs = get_default_dataset_config(config=config)
    if args.dataset not in dataset_configs:
        raise SystemExit(
            f"Unknown dataset {args.dataset!r}; available: {sorted(dataset_configs)}"
        )
    data_path = resolve_dataset_path(
        get_full_sequence_test_path(dataset_configs[args.dataset]), PROJECT_ROOT
    )
    window = config.test.get("window", 60)
    dataset = IMUDataset(
        data_dir=str(data_path),
        window_size=window,
        debug=False,
        simulate_imu_noise=True,
        simulate_imu_noise_seed=42,
        min_obj_contact_frames=0,
        full_sequence=True,
        resolve_bimanual_contact_conflicts=config.get(
            "resolve_bimanual_contact_conflicts", False
        ),
    )
    if args.max_sequences:
        dataset.sequence_info = dataset.sequence_info[: args.max_sequences]
    print(f"sequences: {len(dataset)}")

    online_window = resolve_online_window(config, None)
    online_lookahead = resolve_online_lookahead(config, None)
    hand_threshold = float(getattr(config, "hand_contact_threshold", 0.15))

    report = {}
    for name, arch, ckpt, zero_boundary in runs:
        print(f"\n=== {name} ({arch}{', boundary zeroed' if zero_boundary else ''}) ===")
        run_config = copy.deepcopy(config)
        run_config.velocity_contact_arch = arch
        module_paths = {
            "human_pose": str(args.hp_ckpt),
            "velocity_contact": str(ckpt),
        }
        model = _unwrap_model(
            load_model(
                run_config,
                device,
                no_trans=False,
                module_paths=module_paths,
                object_trans_arch="current",
                velocity_contact_arch=arch,
            )
        ).eval()
        if zero_boundary:
            zero_boundary_features(model.velocity_contact_module)

        _set_eval_seed(42)
        with torch.inference_mode():
            pred_cache, sample_cache = ebo._batched_online_predict_dataset(
                model,
                dataset,
                run_config,
                device,
                online_batch_size=args.online_batch_size,
                online_window=online_window,
                online_lookahead=online_lookahead,
                use_object_data=False,
                compute_fk=False,
                refine_human=False,
                sort_desc=True,
                eval_imu_noise=False,
                eval_imu_noise_seed=None,
            )

        stats = collect(pred_cache, sample_cache, hand_threshold, args.horizon)
        report[name] = stats
        print_stats(stats)
        del model
        torch.cuda.empty_cache()

    if len(report) > 1:
        print("\n=== hard-negative false-positive rate by distance to onset ===")
        names = list(report)
        header = "bin".ljust(16) + "".join(n.rjust(12) for n in names)
        print(header)
        for _, _, label in BINS:
            row = label.ljust(16)
            for n in names:
                value = report[n]["hard_neg_fp_by_bin"].get(label)
                row += ("-" if value is None else f"{value:.4f}").rjust(12)
            print(row)

    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output_json}")


def collect(pred_cache, sample_cache, hand_threshold, horizon):
    totals = {
        "frames": 0,
        "hard_neg_frames": 0,
        "hard_neg_fp": 0,
        "pre_onset_frames": 0,
        "pre_onset_fp": 0,
        "pre_onset_hard_neg_frames": 0,
        "pre_onset_hard_neg_fp": 0,
    }
    bin_hard_neg = {label: [0, 0] for _, _, label in BINS}
    bin_fn = {label: [0, 0] for _, _, label in BINS}
    # Pose-only auxiliary heads: do the boundary features carry the signal?
    probe_scores = {"anticipation": [], "endpoint": []}
    probe_labels = {"anticipation": [], "endpoint": []}
    # Threshold-free contact quality, so a calibration shift is not mistaken
    # for a loss of discriminative power at the fixed 0.15 operating point.
    contact_scores = []
    contact_targets = []
    contact_hard_neg_mask = []

    for sample in sample_cache:
        key = sample.get("seq_path") or sample.get("seq_file")
        pred = pred_cache.get(key)
        if pred is None:
            continue
        contact = pred.get("pred_hand_contact_prob")
        if contact is None:
            continue
        contact = np.asarray(contact.squeeze(0) if contact.dim() == 3 else contact)
        length = contact.shape[0]

        def gt(name):
            value = sample.get(name)
            value = np.asarray(value)
            if value.ndim > 1:
                value = value.reshape(-1)
            return value[:length].astype(bool)

        lhand, rhand, obj = gt("lhand_contact"), gt("rhand_contact"), gt("obj_contact")
        pred_l = contact[:, 0] > hand_threshold
        pred_r = contact[:, 1] > hand_threshold
        pred_any = pred_l | pred_r
        gt_any = lhand | rhand

        hard_neg = obj & ~gt_any
        distance = signed_distance_to_transitions(gt_any)
        # Pre-onset: the horizon frames leading into a contact start.
        onsets = np.nonzero(gt_any[1:] & ~gt_any[:-1])[0] + 1
        pre_onset = np.zeros(length, dtype=bool)
        for onset in onsets:
            pre_onset[max(0, onset - horizon):onset] = True

        totals["frames"] += length
        totals["hard_neg_frames"] += int(hard_neg.sum())
        totals["hard_neg_fp"] += int(pred_any[hard_neg].sum())
        totals["pre_onset_frames"] += int(pre_onset.sum())
        totals["pre_onset_fp"] += int((pred_any & ~gt_any)[pre_onset].sum())
        pre_hard = pre_onset & hard_neg
        totals["pre_onset_hard_neg_frames"] += int(pre_hard.sum())
        totals["pre_onset_hard_neg_fp"] += int(pred_any[pre_hard].sum())

        for low, high, label in BINS:
            in_bin = (distance >= low) & (distance < high) & np.isfinite(distance)
            selected = in_bin & hard_neg
            bin_hard_neg[label][0] += int(pred_any[selected].sum())
            bin_hard_neg[label][1] += int(selected.sum())
            positives = in_bin & gt_any
            bin_fn[label][0] += int((~pred_any)[positives].sum())
            bin_fn[label][1] += int(positives.sum())

        contact_scores.append(np.maximum(contact[:, 0], contact[:, 1]))
        contact_targets.append(gt_any)
        contact_hard_neg_mask.append(hard_neg)

        endpoint_gt = sample.get("contact_endpoint")
        if endpoint_gt is not None:
            endpoint_gt = np.asarray(endpoint_gt).reshape(-1, 6)[:length]
            endpoint_prob = pred.get("pred_contact_endpoint_prob")
            if endpoint_prob is not None:
                endpoint_prob = np.asarray(
                    endpoint_prob.squeeze(0) if endpoint_prob.dim() == 3 else endpoint_prob
                )[:length]
                probe_scores["endpoint"].append(endpoint_prob.reshape(-1))
                probe_labels["endpoint"].append((endpoint_gt > 0.5).reshape(-1))
            anticipation_prob = pred.get("pred_contact_anticipation_prob")
            if anticipation_prob is not None and length > horizon:
                anticipation_prob = np.asarray(
                    anticipation_prob.squeeze(0)
                    if anticipation_prob.dim() == 3
                    else anticipation_prob
                )[:length]
                # Target: an endpoint of the same channel inside (t, t+horizon].
                target = np.zeros_like(endpoint_gt, dtype=bool)
                hard = endpoint_gt > 0.5
                for shift in range(1, horizon + 1):
                    target[: length - shift] |= hard[shift:]
                usable = slice(0, length - horizon)
                probe_scores["anticipation"].append(
                    anticipation_prob[usable].reshape(-1)
                )
                probe_labels["anticipation"].append(target[usable].reshape(-1))

    def rate(pair):
        return pair[0] / pair[1] if pair[1] else None

    probes = {}
    for probe_name in probe_scores:
        if not probe_scores[probe_name]:
            continue
        scores = np.concatenate(probe_scores[probe_name])
        labels = np.concatenate(probe_labels[probe_name]).astype(np.float64)
        probes[probe_name] = {
            "average_precision": average_precision(scores, labels),
            "base_rate": float(labels.mean()),
            "lift": (
                average_precision(scores, labels) / float(labels.mean())
                if labels.mean() > 0
                else None
            ),
        }

    contact_quality = {}
    if contact_scores:
        scores = np.concatenate(contact_scores)
        targets = np.concatenate(contact_targets).astype(np.float64)
        hard = np.concatenate(contact_hard_neg_mask)
        contact_quality["average_precision"] = average_precision(scores, targets)
        contact_quality["base_rate"] = float(targets.mean())
        # Positives against hard negatives only: the discrimination the fixed
        # threshold is actually failing on.
        selected = (targets > 0.5) | hard
        contact_quality["hard_negative_average_precision"] = average_precision(
            scores[selected], targets[selected]
        )
        contact_quality["score_quantiles"] = {
            "positive_any_hand": score_quantiles(scores[targets > 0.5]),
            "hard_negative": score_quantiles(scores[hard]),
            "other_negative": score_quantiles(
                scores[(targets <= 0.5) & ~hard]
            ),
        }
        threshold_sweep = {}
        for threshold in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            predicted = scores > threshold
            true_positive = float((predicted & (targets > 0.5)).sum())
            predicted_positive = float(predicted.sum())
            positive_count = float((targets > 0.5).sum())
            threshold_sweep[str(threshold)] = {
                "f1": (
                    2.0 * true_positive
                    / (predicted_positive + positive_count)
                    if predicted_positive + positive_count
                    else None
                ),
                "precision": (
                    true_positive / predicted_positive
                    if predicted_positive
                    else None
                ),
                "recall": (
                    true_positive / positive_count if positive_count else None
                ),
                "predicted_positive_rate": float(predicted.mean()),
                "hard_negative_fp_rate": float(predicted[hard].mean()),
            }
        contact_quality["threshold_sweep"] = threshold_sweep
        best = {"f1": -1.0}
        for threshold in np.arange(0.05, 0.96, 0.05):
            predicted = scores > threshold
            true_positive = float((predicted & (targets > 0.5)).sum())
            if true_positive == 0:
                continue
            precision = true_positive / float(predicted.sum())
            recall = true_positive / float((targets > 0.5).sum())
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "threshold": round(float(threshold), 2),
                    "precision": precision,
                    "recall": recall,
                    "hard_neg_fp_rate": float(predicted[hard].mean()),
                }
        contact_quality["best_threshold"] = best

    return {
        "totals": totals,
        "contact_quality": contact_quality,
        "pose_only_probes": probes,
        "hard_neg_fp_rate": rate(
            (totals["hard_neg_fp"], totals["hard_neg_frames"])
        ),
        "pre_onset_fp_rate": rate(
            (totals["pre_onset_fp"], totals["pre_onset_frames"])
        ),
        "pre_onset_hard_neg_fp_rate": rate(
            (totals["pre_onset_hard_neg_fp"], totals["pre_onset_hard_neg_frames"])
        ),
        "hard_neg_fp_by_bin": {k: rate(v) for k, v in bin_hard_neg.items()},
        "fn_by_bin": {k: rate(v) for k, v in bin_fn.items()},
    }


def print_stats(stats):
    totals = stats["totals"]
    print(f"  frames                     : {totals['frames']}")
    print(f"  hard-negative frames       : {totals['hard_neg_frames']} "
          f"({totals['hard_neg_frames'] / max(totals['frames'], 1):.2%})")
    print(f"  hard-negative FP rate      : {stats['hard_neg_fp_rate']}")
    print(f"  pre-onset FP rate          : {stats['pre_onset_fp_rate']}")
    print(f"  pre-onset hard-neg FP rate : {stats['pre_onset_hard_neg_fp_rate']}")
    quality = stats.get("contact_quality") or {}
    if quality:
        print(f"  contact AP (any hand)      : {quality['average_precision']:.4f} "
              f"(base rate {quality['base_rate']:.4f})")
        print("  contact AP vs hard negs    : "
              f"{quality['hard_negative_average_precision']:.4f}")
        best = quality["best_threshold"]
        print(f"  best-threshold F1          : {best['f1']:.4f} @ thr={best['threshold']} "
              f"(P={best['precision']:.4f} R={best['recall']:.4f} "
              f"hardFP={best['hard_neg_fp_rate']:.4f})")
        quantiles = quality.get("score_quantiles", {})
        if quantiles:
            print(
                "  score medians (positive/hard-negative/other-negative): "
                f"{quantiles['positive_any_hand']['p50']:.4f} / "
                f"{quantiles['hard_negative']['p50']:.4f} / "
                f"{quantiles['other_negative']['p50']:.4f}"
            )
    for name, probe in stats.get("pose_only_probes", {}).items():
        lift = probe["lift"]
        print(
            f"  pose-only {name:13s}: AP={probe['average_precision']:.4f} "
            f"base_rate={probe['base_rate']:.4f} "
            f"lift={'n/a' if lift is None else f'{lift:.2f}x'}"
        )


if __name__ == "__main__":
    main()
