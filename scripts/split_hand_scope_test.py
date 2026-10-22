"""Create a traceable long-sequence hand-interaction evaluation subset.

Uses saved labels only. This is a scope-filtered benchmark, not a replacement
for the full test set. All ranges are zero-based, end-exclusive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

STATIC_ARRAY_KEYS = {"obj_points_canonical", "betas", "canonical_betas"}


def true_runs(mask):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]


def find_cuts(moving, hand, min_motion_frames=60, max_static_gap=15):
    """Merge brief static gaps only inside a continuous no-hand interval."""
    moving = np.asarray(moving, dtype=bool).reshape(-1)
    hand = np.asarray(hand, dtype=bool).reshape(-1)
    if moving.shape != hand.shape:
        raise ValueError("Movement and hand labels must have equal lengths")
    if min_motion_frames < 1 or max_static_gap < 0:
        raise ValueError("Invalid cut thresholds")
    cuts = []
    for outer_a, outer_b in true_runs(~hand):
        episodes = true_runs(moving[outer_a:outer_b])
        groups = []
        for a, b in episodes:
            a, b = a + outer_a, b + outer_a
            if groups and a - groups[-1][1] <= max_static_gap:
                groups[-1][1] = b
            else:
                groups.append([a, b])
        for a, b in groups:
            count = int(moving[a:b].sum())
            if count >= min_motion_frames:
                cuts.append({"start": a, "end": b, "frames": b-a,
                             "moving_frames": count, "static_gap_frames": b-a-count})
    return cuts


def retained_intervals(length, cuts, min_frames=120):
    if min_frames < 1:
        raise ValueError("Minimum retained length must be positive")
    keep, short = [], []
    cursor = 0
    for cut in [*cuts, {"start": length, "end": length}]:
        a, b = cursor, cut["start"]
        if not 0 <= a <= b <= cut["end"] <= length:
            raise ValueError("Cuts must be ordered, disjoint and in bounds")
        if b > a:
            (keep if b-a >= min_frames else short).append((a, b))
        cursor = cut["end"]
    return keep, short


def temporal_keys(data, length):
    return [k for k, v in data.items() if k not in STATIC_ARRAY_KEYS
            and isinstance(v, (torch.Tensor, np.ndarray))
            and v.ndim > 0 and v.shape[0] == length]


def slice_sequence(data, start, end):
    length = len(data["obj_trans"])
    if not 0 <= start < end <= length:
        raise ValueError("Invalid slice bounds")
    keys = set(temporal_keys(data, length))
    result = {}
    for k, v in data.items():
        if k in keys:
            v = v[start:end].clone() if isinstance(v, torch.Tensor) else v[start:end].copy()
        result[k] = v
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def process(args):
    source, out = Path(args.input_dir), Path(args.output_dir)
    files = sorted(source.glob("*.pt"))
    if not files:
        raise ValueError(f"No source files in {source}")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    params = {"min_motion_frames": args.min_motion_frames, "max_static_gap": args.max_static_gap,
              "min_retained_frames": args.min_frames, "fps": args.fps}
    manifest = {"status": "building", "input_dir": str(source), "output_dir": str(out),
                "range_convention": "zero-based end-exclusive", "parameters": params,
                "rule": "Saved obj_contact AND neither hand; bridge only short static gaps without hand contact; threshold counts moving frames, not gap frames.",
                "benchmark_scope": "hand-interaction applicable subset; short unheld motion remains",
                "sequences": []}
    manifest_path = out.parent / "split_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for path in files:
        before = digest(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        length = len(data["obj_trans"])
        labels = {}
        for key in ("obj_contact", "lhand_contact", "rhand_contact"):
            v = torch.as_tensor(data[key]).reshape(-1)
            if len(v) != length or not bool(((v == 0) | (v == 1)).all()):
                raise ValueError(f"Invalid binary label {path}: {key}")
            labels[key] = v.bool().numpy()
        moving = labels["obj_contact"]
        hand = labels["lhand_contact"] | labels["rhand_contact"]
        cuts = find_cuts(moving, hand, args.min_motion_frames, args.max_static_gap)
        keep, short = retained_intervals(length, cuts, args.min_frames)
        row = {"source_file": path.name, "source_sha256": before, "source_frames": length,
               "temporal_keys": temporal_keys(data, length), "cuts": cuts,
               "discarded_short": [{"start": a, "end": b, "frames": b-a} for a,b in short],
               "retained": []}
        for index, (a, b) in enumerate(keep):
            name = f"{path.stem}_hand{index:03d}_f{a:06d}-{b:06d}"
            result = slice_sequence(data, a, b)
            result["seq_name"] = name
            result["hand_scope_split"] = {"version": 1, "source_file": path.name,
                "source_seq_name": data.get("seq_name", path.stem), "source_sha256": before,
                "source_start_frame": a, "source_end_frame_exclusive": b,
                "source_frames": length, "parameters": params}
            target = out / f"{name}.pt"
            torch.save(result, target)
            loaded = torch.load(target, map_location="cpu", weights_only=False)
            if not equal(result, loaded):
                raise AssertionError(f"Saved output differs: {target}")
            for key, v in data.items():
                if key == "seq_name":
                    continue
                expected = v[a:b] if key in row["temporal_keys"] else v
                if not equal(expected, loaded[key]):
                    raise AssertionError(f"Source/slice mismatch: {target} {key}")
            if find_cuts(moving[a:b], hand[a:b], args.min_motion_frames, args.max_static_gap):
                raise AssertionError(f"Qualifying unheld interval remains: {target}")
            row["retained"].append({"file": target.name, "start": a, "end": b, "frames": b-a,
                                    "remaining_short_unheld_motion_frames": int((moving[a:b] & ~hand[a:b]).sum())})
        if digest(path) != before:
            raise AssertionError(f"Source changed during processing: {path}")
        accounted = sum(b-a for a,b in keep+short) + sum(c["frames"] for c in cuts)
        if accounted != length:
            raise AssertionError("Frame accounting mismatch")
        manifest["sequences"].append(row)
    rows = manifest["sequences"]
    lengths = [r["frames"] for s in rows for r in s["retained"]]
    manifest["summary"] = {
        "source_sequences": len(rows), "source_frames": sum(s["source_frames"] for s in rows),
        "affected_sequences": sum(bool(s["cuts"]) for s in rows),
        "cut_intervals": sum(len(s["cuts"]) for s in rows),
        "cut_frames": sum(c["frames"] for s in rows for c in s["cuts"]),
        "discarded_short_intervals": sum(len(s["discarded_short"]) for s in rows),
        "discarded_short_frames": sum(c["frames"] for s in rows for c in s["discarded_short"]),
        "output_sequences": len(lengths), "retained_frames": sum(lengths),
        "min_frames": min(lengths) if lengths else None, "max_frames": max(lengths) if lengths else None,
        "median_frames": float(np.median(lengths)) if lengths else None,
        "fully_dropped_sequences": [s["source_file"] for s in rows if not s["retained"]],
        "remaining_short_unheld_motion_frames": sum(r["remaining_short_unheld_motion_frames"] for s in rows for r in s["retained"]),
    }
    manifest["status"] = "complete"
    manifest["validation"] = "All saved fields checked against exact source slices; static fields preserved; SHA256 source unchanged; frame accounting exact; no qualifying cuts remain. No model evaluation performed."
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# BEHAVE hand-interaction long-sequence subset", "", "All ranges: zero-based, end-exclusive. Original data unchanged.",
             "", "Parameters: " + json.dumps(params), "", "```json", json.dumps(manifest["summary"], indent=2), "```", "",
             "## Complete source-to-result list", "", "| Source | Original frames | Retained [start,end): frames | Cut [start,end) | Discarded short [start,end) |", "|---|---:|---|---|---|"]
    for s in rows:
        ranges = lambda xs: "; ".join(f'[{x["start"]},{x["end"]}): {x["frames"]}' for x in xs) or "-"
        lines.append(f'| {Path(s["source_file"]).stem} | {s["source_frames"]} | {ranges(s["retained"])} | {ranges(s["cuts"])} | {ranges(s["discarded_short"])} |')
    lines.extend(["", "## Output filenames", ""])
    lines.extend(f'- `{r["file"]}` ({r["frames"]} frames)' for s in rows for r in s["retained"])
    (out.parent / "split_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-motion-frames", type=int, default=60)
    parser.add_argument("--max-static-gap", type=int, default=15)
    parser.add_argument("--min-frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30)
    process(parser.parse_args())
