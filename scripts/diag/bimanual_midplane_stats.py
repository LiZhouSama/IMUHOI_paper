"""Measure object distance to the perpendicular bisector of two hand anchors.

Read unified sequences once, preserving saved contact labels. No model inference
or dataset mutation is performed. Run with the WSL SAGE interpreter.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


DATASETS = {"OMOMO": "processed_split_data_OMOMO_full",
            "BEHAVE": "processed_split_data_BEHAVE"}
THRESHOLDS_CM = (0.1, 1, 2, 5, 10, 20)


def geometry(hands, obj):
    delta = hands[:, 1] - hands[:, 0]
    span = torch.linalg.vector_norm(delta, dim=-1)
    offset = obj - hands.mean(dim=1)
    signed = (offset * delta).sum(-1) / span.clamp_min(1e-12)
    finite = torch.isfinite(hands).all(-1).all(-1) & torch.isfinite(obj).all(-1)
    valid = finite & (span > 1e-6)
    # Independent equal-distance identity verifies the plane formula.
    radii_sq = ((obj[:, None] - hands) ** 2).sum(-1)
    alternate = (radii_sq[:, 0] - radii_sq[:, 1]) / (2 * span.clamp_min(1e-12))
    torch.testing.assert_close(signed[valid], alternate[valid], atol=1e-9, rtol=1e-8)
    return torch.stack((signed.abs(), signed, span,
                        torch.linalg.vector_norm(offset, dim=-1)), dim=-1), valid


def summarize(values):
    if len(values) == 0:
        return {"frames": 0}
    cm = values[:, 0] * 100
    result = {"frames": len(values), "mean_cm": float(cm.mean()),
              "rms_cm": float(np.sqrt(np.mean(cm ** 2))),
              "signed_mean_cm": float(values[:, 1].mean() * 100),
              "span_median_cm": float(np.median(values[:, 2]) * 100),
              "midpoint_distance_median_cm": float(np.median(values[:, 3]) * 100),
              "normalized_distance_median": float(np.median(values[:, 0] / values[:, 2]))}
    for q in (50, 75, 90, 95, 99, 100):
        result[f"p{q}_cm"] = float(np.percentile(cm, q))
    for threshold in THRESHOLDS_CM:
        result[f"within_{threshold:g}cm_pct"] = float(np.mean(cm <= threshold) * 100)
    for ratio in (0.05, 0.1, 0.2):
        result[f"within_{ratio:g}_span_pct"] = float(
            np.mean(values[:, 0] <= ratio * values[:, 2]) * 100)
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, dataset, device):
    data_root = root / DATASETS[dataset]
    paths = sorted((data_root / "unified").glob("*.pt"))
    if not paths:
        raise ValueError(f"No input sequences: {data_root}")
    splits = {s: {p.name for p in (data_root / s).glob("*.pt")}
              for s in ("train", "test")}
    assert not splits["train"] & splits["test"], "Overlapping train/test filenames"
    assert (splits["train"] | splits["test"]) <= {p.name for p in paths}
    groups = defaultdict(list)
    objects = defaultdict(list)
    rows, frame_parts = [], []
    names = Counter()
    counts = Counter(files=len(paths))
    invalid_files = []
    unassigned = []
    for index, path in enumerate(paths):
        data = torch.load(path, map_location="cpu", weights_only=False)
        name = str(data.get("seq_name", path.stem))
        names[name] += 1
        split = next((s for s in splits if path.name in splits[s]), "unassigned")
        if split == "unassigned":
            unassigned.append(path.name)
        obj_name = str(data["obj_name"])
        pos = torch.as_tensor(data["position_global_full_gt_world"],
                              dtype=torch.float64, device=device)
        obj = torch.as_tensor(data["obj_trans"], dtype=torch.float64,
                              device=device).reshape(-1, 3)
        t = len(pos)
        assert pos.shape == (t, 24, 3) and obj.shape == (t, 3), path
        labels = {}
        for key in ("lhand_contact", "rhand_contact", "obj_contact",
                    "lhand_contact_raw", "rhand_contact_raw"):
            value = torch.as_tensor(data[key], device=device).reshape(-1)
            assert value.shape == (t,) and torch.all((value == 0) | (value == 1)), (path, key)
            labels[key] = value.bool()
        both = labels["lhand_contact"] & labels["rhand_contact"]
        raw_both = labels["lhand_contact_raw"] & labels["rhand_contact_raw"]
        palm, valid = geometry(pos[:, [22, 23]], obj)
        wrist, wrist_valid = geometry(pos[:, [20, 21]], obj)
        mask = both & valid
        invalid = int((both & ~valid).sum())
        counts.update(total_frames=t, bimanual_frames=int(both.sum()),
                      valid_bimanual_frames=int(mask.sum()), invalid_bimanual_frames=invalid,
                      sequences_with_bimanual=int(bool(mask.any())))
        if invalid:
            invalid_files.append({"file": path.name, "frames": invalid})
        arr = palm[mask].cpu().numpy()
        for group, gmask, vals in (
            ("all", mask, palm), (split, mask, palm),
            ("raw_labels", raw_both & valid, palm),
            ("object_moving", mask & labels["obj_contact"], palm),
            ("object_static", mask & ~labels["obj_contact"], palm),
            ("wrists", both & wrist_valid, wrist),
            ("span_at_least_10cm", mask & (palm[:, 2] >= 0.1), palm),
        ):
            groups[group].append(vals[gmask].cpu().numpy())
        objects[obj_name].append(arr)
        rows.append({"dataset": dataset, "file": path.name, "seq_name": name,
                     "object": obj_name, "split": split, "total_frames": t,
                     "bimanual_label_frames": int(both.sum()),
                     "invalid_frames": invalid, **summarize(arr)})
        frame_parts.append(np.column_stack((np.full(len(arr), index),
                                           torch.where(mask)[0].cpu().numpy(), arr)))
        if (index + 1) % 500 == 0:
            print(f"{dataset}: {index + 1}/{len(paths)} files", flush=True)
    arrays = {k: np.concatenate(v) for k, v in groups.items()}
    object_stats = [{"dataset": dataset, "object": k, **summarize(np.concatenate(v))}
                    for k, v in objects.items()]
    nonempty = [r for r in rows if r["frames"]]
    metadata = {**dict(counts), "input": str(data_root / "unified"),
                "unique_seq_names": len(names),
                "repeated_seq_names": {k: v for k, v in names.items() if v > 1},
                "unassigned_files": unassigned, "invalid_files": invalid_files,
                "groups": {k: summarize(v) for k, v in arrays.items()},
                "sequence_equal_weight": {
                    "mean_of_sequence_means_cm": float(np.mean([r["mean_cm"] for r in nonempty])),
                    "mean_of_sequence_within_5cm_pct": float(np.mean([r["within_5cm_pct"] for r in nonempty])),
                    "mean_of_sequence_within_10cm_pct": float(np.mean([r["within_10cm_pct"] for r in nonempty]))}}
    return metadata, rows, object_stats, arrays["all"], np.concatenate(frame_parts)


def plot(output, all_values, object_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for (name, arr), color in zip(all_values.items(), ("#2166ac", "#d6604d")):
        cm = np.sort(arr[:, 0] * 100)
        axes[0].plot(cm, 100 * np.arange(1, len(cm) + 1) / len(cm), label=name, color=color)
        axes[1].hist(cm, bins=np.arange(0, 61, 1), weights=np.full(len(cm), 100 / len(cm)),
                     histtype="step", linewidth=1.8, color=color, label=name)
    for threshold in (2, 5, 10, 20):
        axes[0].axvline(threshold, color="gray", alpha=0.2)
    axes[0].set(xlim=(0, 40), ylim=(0, 100), xlabel="Distance to bisector plane (cm)",
                ylabel="Bimanual frames within distance (%)", title="Cumulative distribution")
    axes[1].set(xlim=(0, 60), xlabel="Distance to bisector plane (cm)",
                ylabel="All bimanual frames per 1 cm bin (%)", title="Distance distribution (tail beyond 60 cm omitted)")
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.legend()
    fig.suptitle("Object position vs. perpendicular bisector of GT palm anchors")
    fig.savefig(output / "distance_distribution.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        rows = sorted((r for r in object_rows if r["dataset"] == dataset and r["frames"]),
                      key=lambda r: r["p50_cm"])
        y = np.arange(len(rows))
        ax.hlines(y, [r["p50_cm"] for r in rows], [r["p90_cm"] for r in rows], color="#9db6c8")
        ax.scatter([r["p50_cm"] for r in rows], y, label="Median", color="#2166ac")
        ax.scatter([r["p90_cm"] for r in rows], y, label="P90", color="#d6604d", marker="|")
        ax.set_yticks(y, [f"{r['object']} (n={r['frames']:,})" for r in rows])
        ax.invert_yaxis()
        ax.set(title=dataset, xlabel="Distance to bisector plane (cm)")
        ax.grid(axis="x", alpha=0.2)
        ax.legend()
    fig.savefig(output / "by_object.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("process/IMUHOI++cano"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/bimanual_midplane_20260928"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    # Exact on-plane, off-plane, and coincident-hand examples.
    hands = torch.tensor([[[-1., 0, 0], [1., 0, 0]]] * 2 + [[[0., 0, 0]] * 2],
                         dtype=torch.float64, device=device)
    obj = torch.tensor([[0., 3, 4], [0.05, 3, 4], [1., 2, 3]], dtype=torch.float64, device=device)
    values, valid = geometry(hands, obj)
    torch.testing.assert_close(values[:2, 0], torch.tensor([0., 0.05], dtype=torch.float64, device=device))
    assert valid.tolist() == [True, True, False]
    rotated = torch.tensor([[0., -1, 0], [1., 0, 0], [0., 0, 1]], dtype=torch.float64, device=device)
    moved, _ = geometry(hands @ rotated.T + 7, obj @ rotated.T + 7)
    torch.testing.assert_close(values[:2], moved[:2])
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"definition": "abs((obj - (left + right)/2) dot (right-left))/norm(right-left)",
              "hand_joints": [22, 23], "object_field": "obj_trans",
              "contact": "saved lhand_contact AND rhand_contact",
              "degenerate_span_threshold_m": 1e-6, "datasets": {}}
    sequence_rows, object_rows, arrays = [], [], {}
    for dataset in DATASETS:
        stats, seqs, objs, arr, frame_data = analyze(args.data_root, dataset, device)
        result["datasets"][dataset] = stats
        sequence_rows.extend(seqs)
        object_rows.extend(objs)
        arrays[dataset] = arr
        np.savez_compressed(args.output / f"{dataset}_frames.npz", values=frame_data,
                            columns=np.array(["file_index", "frame_index", "distance_m", "signed_distance_m",
                                              "hand_span_m", "object_midpoint_distance_m"]),
                            files=np.array([row["file"] for row in seqs]))
        print(dataset, json.dumps(stats, ensure_ascii=False), flush=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output / "per_sequence.csv", sequence_rows)
    write_csv(args.output / "per_object.csv", object_rows)
    write_csv(args.output / "per_group.csv", [{"dataset": d, "group": k, **v}
              for d, s in result["datasets"].items() for k, v in s["groups"].items()])
    plot(args.output, arrays, object_rows)
    print(f"Saved analysis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
