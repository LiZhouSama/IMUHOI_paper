#!/usr/bin/env python3
"""Compare saved OMOMO results, without importing or executing EgoIMU code."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dataset.dataset_IMUHOI import _build_human_imu_root, _build_object_imu

OLD = ROOT.parent / "EgoIMU/process/processed_split_data_OMOMO"
NEW = ROOT / "process/IMUHOI++cano/processed_split_data_OMOMO_full"
OUT = ROOT / "outputs/analysis/omomo_preprocessing_audit_20261016"


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def tensor(data, key):
    return torch.as_tensor(data[key]).float()


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)), "p95": float(np.quantile(values, 0.95)), "max": float(values.max())}


def collect_index(root):
    result = {}
    for split in ("train", "test"):
        for path in sorted((root / split).glob("*.pt")):
            if path.name in result:
                raise ValueError(f"Duplicated file ID: {root} {path.name}")
            result[path.name] = (split, path)
    return result


def geometry(data):
    p = tensor(data, "obj_trans")
    delta = p[1:] - p[:-1]
    speed = torch.linalg.norm(delta, dim=-1) * 30
    hands = tensor(data, "lhand_contact").bool() | tensor(data, "rhand_contact").bool()
    return {"frames": len(p), "seconds": len(p) / 30, "path_length_m": float(torch.linalg.norm(delta, dim=-1).sum()),
            "mean_speed_m_s": float(speed.mean()) if len(speed) else 0.0,
            "extent_m": float(torch.linalg.norm(p.max(0).values - p.min(0).values)),
            "hand_contact_fraction": float(hands.float().mean()),
            "object_move_fraction": float(tensor(data, "obj_contact").mean())}


def compare(name, old, new, row):
    if str(old["seq_name"]) != str(new["seq_name"]):
        row["identity_match"] = False
        return
    p_old, p_new = tensor(old, "position_global_full_gt_world"), tensor(new, "position_global_full_gt_world")
    if len(p_old) != len(p_new):
        row["frame_count_match"] = False
        return
    row.update(identity_match=True, frame_count_match=True)
    obj_old, obj_new = tensor(old, "obj_trans"), tensor(new, "obj_trans")
    shift = (obj_new - obj_old).mean(0)
    row["object_constant_shift_y_cm"] = float(shift[1]) * 100
    row["object_shift_removed_max_m"] = float(((obj_new - obj_old) - shift).abs().max())
    row["object_rotation_max_abs"] = float((tensor(old, "obj_rot") - tensor(new, "obj_rot")).abs().max())
    row["object_scale_max_abs"] = float((tensor(old, "obj_scale") - tensor(new, "obj_scale")).abs().max())
    row["rotation_body22_max_abs"] = float((tensor(old, "rotation_global")[:, :22] - tensor(new, "rotation_global")[:, :22]).abs().max())
    row["trans_shift_removed_max_m"] = float(((tensor(new, "trans") - tensor(old, "trans")) - shift).abs().max())
    row["body22_shift_removed_mean_cm"] = float(torch.linalg.norm(p_new[:, :22] - p_old[:, :22] - shift, dim=-1).mean()) * 100
    relative_old = p_old[:, :22] - p_old[:, :1]
    relative_new = p_new[:, :22] - p_new[:, :1]
    row["body22_root_relative_mean_cm"] = float(torch.linalg.norm(relative_new - relative_old, dim=-1).mean()) * 100
    old_hand_vec = obj_old[:, None] - p_old[:, [20, 21]]
    new_wrist_vec = obj_new[:, None] - p_new[:, [20, 21]]
    new_palm_vec = obj_new[:, None] - p_new[:, [22, 23]]
    row["hand_object_vector_wrist_to_wrist_delta_cm"] = float(torch.linalg.norm(new_wrist_vec - old_hand_vec, dim=-1).mean()) * 100
    row["hand_object_vector_old_wrist_to_new_palm_delta_cm"] = float(torch.linalg.norm(new_palm_vec - old_hand_vec, dim=-1).mean()) * 100
    row["new_wrist_to_palm_cm"] = float(torch.linalg.norm(p_new[:, [22, 23]] - p_new[:, [20, 21]], dim=-1).mean()) * 100
    for key in ("lhand_contact", "rhand_contact", "obj_contact", "lfoot_contact", "rfoot_contact"):
        a, b = tensor(old, key).bool(), tensor(new, key).bool()
        row[f"{key}_old_positive"] = int(a.sum())
        row[f"{key}_new_positive"] = int(b.sum())
        row[f"{key}_disagree_frames"] = int((a != b).sum())
        raw = new.get(key + "_raw")
        if raw is not None:
            row[f"{key}_old_vs_new_raw_disagree_frames"] = int((a != torch.as_tensor(raw).bool()).sum())
    old_imu = _build_human_imu_root(p_old, tensor(old, "rotation_global"), [0, 7, 8, 15, 20, 21], [0, 4, 5, 15, 18, 19], 30)
    old_obj_imu = _build_object_imu(tensor(old, "obj_rot"), obj_old, 30)
    new_imu, new_obj_imu = tensor(new, "human_imu"), tensor(new, "obj_imu")
    row["human_acc_feature_rmse_m_s2"] = float((old_imu[..., :3] - new_imu[..., :3]).square().mean().sqrt())
    row["human_rot6d_feature_max_abs"] = float((old_imu[..., 3:] - new_imu[..., 3:]).abs().max())
    row["object_acc_feature_rmse_m_s2"] = float((old_obj_imu[..., :3] - new_obj_imu[..., :3]).square().mean().sqrt())
    row["object_rot6d_feature_max_abs"] = float((old_obj_imu[..., 3:] - new_obj_imu[..., 3:]).abs().max())


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    old_index, new_index = collect_index(OLD), collect_index(NEW)
    matrix = Counter((old_index.get(name, ("missing",))[0], new_index.get(name, ("missing",))[0])
                     for name in old_index.keys() | new_index.keys())
    print("FILE_SPLIT_MATRIX", dict(matrix), flush=True)
    old_info = json.loads((OLD / "split_info.json").read_text())
    new_info = json.loads((NEW / "split_info.json").read_text())
    rows, datasets = [], {"old": defaultdict(list), "new": defaultdict(list)}
    names = sorted(old_index.keys() | new_index.keys(), key=lambda name: int(Path(name).stem))
    for i, name in enumerate(names):
        old_entry, new_entry = old_index.get(name), new_index.get(name)
        old = load(old_entry[1]) if old_entry else None
        new = load(new_entry[1]) if new_entry else None
        row = {"file": name, "old_split": old_entry[0] if old_entry else "missing",
               "new_split": new_entry[0] if new_entry else "missing"}
        for label, data, entry in (("old", old, old_entry), ("new", new, new_entry)):
            if data is None:
                continue
            stats = geometry(data)
            row.update({f"{label}_{k}": v for k, v in stats.items()})
            row[f"{label}_seq_name"] = str(data["seq_name"])
            row[f"{label}_object"] = str(data["obj_name"])
            row[f"{label}_source_gender"] = str(data.get("gender"))
            datasets[label][entry[0]].append({**stats, "file": name, "seq_name": str(data["seq_name"]), "object": str(data["obj_name"])})
        if old is not None and new is not None:
            compare(name, old, new, row)
        rows.append(row)
        if (i + 1) % 400 == 0:
            print(f"Compared {i+1}/{len(names)}", flush=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "per_sequence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    paired = [r for r in rows if r.get("identity_match") and r.get("frame_count_match")]
    summary = {"roots": {"old": str(OLD), "new": str(NEW)},
               "split_matrix_by_filename": [{"old": a, "new": b, "n": n} for (a, b), n in sorted(matrix.items())],
               "identity_mismatches": [r for r in rows if r.get("identity_match") is False],
               "frame_count_mismatches": [r for r in rows if r.get("frame_count_match") is False],
               "unassigned_current_files": new_info.get("unassigned_files"),
               "dataset_stats": {}, "paired_statistics": {}}
    for label in datasets:
        summary["dataset_stats"][label] = {}
        for split, values in datasets[label].items():
            summary["dataset_stats"][label][split] = {"n": len(values), "total_frames": sum(v["frames"] for v in values),
                "unique_seq_names": len({v["seq_name"] for v in values}),
                "objects": dict(Counter(v["object"] for v in values)),
                "distributions": {key: describe([v[key] for v in values]) for key in ("frames", "path_length_m", "mean_speed_m_s", "extent_m", "hand_contact_fraction", "object_move_fraction")}}
        train_names = {v["seq_name"] for v in datasets[label]["train"]}
        test_names = {v["seq_name"] for v in datasets[label]["test"]}
        summary["dataset_stats"][label]["train_test_seq_name_overlap"] = sorted(train_names & test_names)
    metric_keys = [k for k in columns if not k.startswith(("old_", "new_")) and k not in ("file", "identity_match", "frame_count_match")]
    for split in ("all", "train", "test"):
        subset = paired if split == "all" else [r for r in paired if r["new_split"] == split]
        frames = sum(r["new_frames"] for r in subset)
        group = {"paired_sequences": len(subset), "frames": frames,
                 "distributions": {key: describe([r[key] for r in subset if key in r]) for key in metric_keys}}
        group["contact_rates"] = {}
        for key in ("lhand_contact", "rhand_contact", "obj_contact", "lfoot_contact", "rfoot_contact"):
            group["contact_rates"][key] = {"old_fraction": sum(r[f"{key}_old_positive"] for r in subset) / frames,
                "new_fraction": sum(r[f"{key}_new_positive"] for r in subset) / frames,
                "disagree_fraction": sum(r[f"{key}_disagree_frames"] for r in subset) / frames,
                "changed_sequences": sum(r[f"{key}_disagree_frames"] > 0 for r in subset)}
        summary["paired_statistics"][split] = group
    debug = []
    for path in sorted((OLD / "debug").glob("*.pt")):
        data = load(path)
        old_entry = old_index.get(path.name)
        new_entry = new_index.get(path.name)
        debug.append({"file": path.name, "seq_name": str(data["seq_name"]),
                      "old_split": old_entry[0] if old_entry else None, "new_split": new_entry[0] if new_entry else None,
                      "identical_to_old_split_bytes": bool(old_entry and hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(old_entry[1].read_bytes()).digest())})
    summary["ego_eval_default_debug_inventory"] = debug
    for label, info, index in (("old", old_info, old_index), ("new", new_info, new_index)):
        split_sets = {split: {n for n, e in index.items() if e[0] == split} for split in ("train", "test")}
        summary.setdefault("manifest_consistency", {})[label] = {split: {"manifest_only": sorted(set(info[f"{split}_files"]) - split_sets[split]),
                                                                                       "folder_only": sorted(split_sets[split] - set(info[f"{split}_files"]))}
                                                                      for split in ("train", "test")}
        matches = []
        if label == "new":
            candidates = sorted(old_index)
            for seed in (0, 1, 10, 42, 2024, 2025, 2026):
                shuffled = candidates.copy()
                random.Random(seed).shuffle(shuffled)
                train = set(shuffled[:int(len(shuffled) * 0.8)])
                matches.append({"seed": seed, "matches_current_train": len(train & split_sets["train"]),
                                "train_exact": train == split_sets["train"]})
        summary.setdefault("split_seed_checks", {})[label] = matches
    write(OUT / "audit.json", summary)
    write(OUT / "inventories.json", datasets)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("paired_statistics", "dataset_stats")}, ensure_ascii=False, indent=2), flush=True)
    print("SAVED", OUT, flush=True)


if __name__ == "__main__":
    main()
