"""Follow-up line/segment analysis using cached bimanual frame geometry and meshes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = {"OMOMO": "processed_split_data_OMOMO_full", "BEHAVE": "processed_split_data_BEHAVE"}
ROD_HANDLE_CANDIDATES = {"clothesstand", "floorlamp", "mop", "tripod", "vacuum"}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def derive(v):
    plane, signed, span, radius = v[:, 2], v[:, 3], v[:, 4], v[:, 5]
    squared = radius ** 2 - plane ** 2
    assert np.all(squared >= -1e-10)
    line = np.sqrt(np.maximum(squared, 0))
    extension = plane > span / 2
    segment = np.hypot(line, np.maximum(plane - span / 2, 0))
    projection_t = 0.5 + signed / span
    return plane, line, segment, extension, projection_t


def stats(v):
    if not len(v):
        return {"frames": 0}
    plane, line, segment, extension, projection_t = derive(v)
    out = {"frames": len(v), "projection_on_extension_pct": float(extension.mean() * 100),
           "span_median_cm": float(np.median(v[:, 4]) * 100)}
    angles = np.degrees(np.arctan2(line, plane))
    out["midpoint_object_vs_hand_line_angle_median_deg"] = float(np.median(angles))
    for name, distance in (("plane", plane), ("line", line), ("segment", segment)):
        out[name + "_mean_cm"] = float(distance.mean() * 100)
        for q in (50, 90, 95):
            out[f"{name}_p{q}_cm"] = float(np.percentile(distance, q) * 100)
        for threshold in (2, 5, 10, 20):
            out[f"{name}_within_{threshold}cm_pct"] = float((distance <= threshold / 100).mean() * 100)
    for threshold in (5, 10, 20):
        near = line <= threshold / 100
        out[f"plane_or_line_within_{threshold}cm_pct"] = float(((plane <= threshold / 100) | near).mean() * 100)
        out[f"extension_among_line_within_{threshold}cm_pct"] = float(extension[near].mean() * 100) if near.any() else None
    return out


def mesh_shape(dataset, name):
    if dataset == "OMOMO":
        path = Path("datasets/OMOMO/captured_objects") / f"{name}_cleaned_simplified.obj"
    else:
        folder = Path("datasets/BEHAVE/objects") / name
        paths = sorted(folder.glob("*.ply")) + sorted(folder.glob("*.obj"))
        paths = sorted(paths, key=lambda p: ("f1000" not in p.stem, len(p.name), p.name))
        if not paths:
            raise FileNotFoundError(folder)
        path = paths[0]
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    # Surface samples make PCA less sensitive to unequal mesh tessellation.
    np.random.seed(20260929)
    pts, _ = trimesh.sample.sample_surface(mesh, 20000)
    pts -= pts.mean(axis=0)
    values, vectors = np.linalg.eigh(pts.T @ pts / len(pts))
    values, vectors = values[::-1], vectors[:, ::-1]
    local = pts @ vectors
    extent = np.ptp(local, axis=0)
    shape = {"mesh_path": str(path), "pca_sd_major_over_second": float(np.sqrt(values[0] / values[1])),
             "pca_sd_second_over_third": float(np.sqrt(values[1] / values[2])),
             "pca_major_extent_native_units": float(extent[0]),
             "pca_second_extent_native_units": float(extent[1]),
             "pca_third_extent_native_units": float(extent[2]),
             "pca_major_variance_pct": float(values[0] / values.sum() * 100)}
    return shape, local


def validate_source(dataset, filename, v, device):
    data = torch.load(Path("process/IMUHOI++cano") / DATASETS[dataset] / "unified" / filename,
                      map_location="cpu", weights_only=False)
    both = torch.as_tensor(data["lhand_contact"]).bool() & torch.as_tensor(data["rhand_contact"]).bool()
    frames = v[:, 1].astype(int)
    assert np.array_equal(torch.where(both)[0].numpy(), frames), (dataset, filename, "contact mismatch")
    pos = torch.as_tensor(data["position_global_full_gt_world"], dtype=torch.float64, device=device)[frames]
    obj = torch.as_tensor(data["obj_trans"], dtype=torch.float64, device=device)[frames]
    left, right = pos[:, 22], pos[:, 23]
    delta = right - left
    span = delta.norm(dim=-1)
    normal = delta / span[:, None]
    relative = obj - (left + right) / 2
    signed = (relative * normal).sum(-1)
    line = (relative - signed[:, None] * normal).norm(dim=-1)
    t = ((obj - left) * delta).sum(-1) / span ** 2
    segment = (obj - (left + t.clamp(0, 1)[:, None] * delta)).norm(dim=-1)
    plane_np, line_np, segment_np, _, _ = derive(v)
    for expected, actual in ((v[:, 3], signed), (v[:, 4], span), (v[:, 5], relative.norm(dim=-1)),
                             (line_np, line), (segment_np, segment)):
        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=1e-8, rtol=1e-8)
    return {"dataset": dataset, "file": filename, "frames_checked": len(v)}


def figures(output, rows, full, meshes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        rr = sorted([r for r in rows if r["dataset"] == dataset and r["group"] == "all"], key=lambda r: r["plane_p50_cm"])
        y = np.arange(len(rr))
        ax.scatter([r["plane_p50_cm"] for r in rr], y, color="#d6604d", label="Plane median", marker="s")
        ax.scatter([r["line_p50_cm"] for r in rr], y, color="#2166ac", label="Infinite line median")
        ax.hlines(y, [r["plane_p50_cm"] for r in rr], [r["line_p50_cm"] for r in rr], color="#bbb", zorder=0)
        ax.set_yticks(y, [r["object"] for r in rr])
        ax.invert_yaxis()
        ax.set(title=dataset, xlabel="Distance (cm)", xlim=(0, None))
        ax.grid(axis="x", alpha=0.2)
        ax.legend()
    fig.suptitle("Bimanual frames: object reference point vs. hand bisector plane / hand line")
    fig.savefig(output / "plane_vs_line_by_object.png", dpi=170)
    plt.close(fig)
    candidates = [("OMOMO", n) for n in ("mop", "clothesstand", "tripod", "vacuum", "floorlamp")]
    candidates += [("BEHAVE", "yogamat"), ("BEHAVE", "boxlong"), ("OMOMO", "woodchair")]
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    for idx, (dataset, name) in enumerate(candidates):
        ax = fig.add_subplot(2, 4, idx + 1, projection="3d")
        xyz = meshes[(dataset, name)][::5]
        ax.scatter(xyz[:, 1], xyz[:, 2], xyz[:, 0], s=0.3, color="#355f83", alpha=0.55)
        extent = np.ptp(xyz, axis=0)
        ax.set_box_aspect([extent[1], extent[2], extent[0]])
        ax.set_title(f"{dataset}: {name}", fontsize=11)
        ax.set_axis_off()
        ax.view_init(elev=15, azim=35)
    fig.suptitle("Local meshes; longest PCA axis vertical (each panel scaled independently)")
    fig.savefig(output / "object_mesh_shapes.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for name in ("mop", "clothesstand", "tripod", "vacuum", "floorlamp"):
        v = full[("OMOMO", name)]
        plane, line, _, _, _ = derive(v)
        for ax, mask in ((axes[0], np.ones(len(v), dtype=bool)), (axes[1], plane > 0.1)):
            arr = np.sort(line[mask] * 100)
            ax.plot(arr, np.arange(1, len(arr) + 1) / len(arr) * 100, label=name)
    for ax, title in zip(axes, ("All bimanual frames", "Only frames >10 cm from the plane")):
        ax.set(xlim=(0, 40), ylim=(0, 100), title=title, xlabel="Distance to infinite hand line (cm)", ylabel="Frames within distance (%)")
        ax.axvline(5, color="gray", alpha=0.3)
        ax.axvline(10, color="gray", alpha=0.3)
        ax.grid(alpha=0.15)
        ax.legend()
    fig.savefig(output / "rod_candidates_line_cdf.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/analysis/bimanual_midplane_20260928"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/bimanual_line_by_object_20260929"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    # Unit examples: on-line between hands; on-line extension; off-line.
    examples = np.array([[0, 0, 0, 0, 2, 0], [0, 0, 2, 2, 2, 2], [0, 0, 2, 2, 2, np.sqrt(13)]], dtype=float)
    _, line, segment, extension, _ = derive(examples)
    np.testing.assert_allclose(line, [0, 0, 3])
    np.testing.assert_allclose(segment, [0, 1, np.sqrt(10)])
    assert extension.tolist() == [False, True, True]
    with (args.source / "per_sequence.csv").open(encoding="utf-8-sig") as f:
        sequences = list(csv.DictReader(f))
    object_by_file = {(r["dataset"], r["file"]): r["object"] for r in sequences}
    rows, shape_rows, checks, seq_rows = [], [], [], []
    full, meshes, dataset_summaries = {}, {}, {}
    for dataset in DATASETS:
        cached = np.load(args.source / f"{dataset}_frames.npz")
        v, files = cached["values"], cached["files"]
        names = np.array([object_by_file[(dataset, str(f))] for f in files])
        frame_names = names[v[:, 0].astype(int)]
        for name in sorted(set(frame_names)):
            arr = v[frame_names == name]
            full[(dataset, name)] = arr
            shape, meshes[(dataset, name)] = mesh_shape(dataset, name)
            shape_rows.append({"dataset": dataset, "object": name, **shape})
            first_index = int(arr[0, 0])
            checks.append(validate_source(dataset, str(files[first_index]), arr[arr[:, 0] == first_index], device))
            for group, mask in (("all", np.ones(len(arr), bool)), ("plane_gt_5cm", arr[:, 2] > .05),
                                ("plane_gt_10cm", arr[:, 2] > .1), ("plane_gt_20cm", arr[:, 2] > .2),
                                ("span_ge_10cm", arr[:, 4] >= .1)):
                rows.append({"dataset": dataset, "object": name, "group": group, **stats(arr[mask])})
        for index in np.unique(v[:, 0]).astype(int):
            arr = v[v[:, 0] == index]
            seq_rows.append({"dataset": dataset, "file": str(files[index]), "object": str(names[index]), **stats(arr)})
        elongated_names = [r["object"] for r in shape_rows if r["dataset"] == dataset and r["pca_sd_major_over_second"] >= 3]
        candidate_mask = np.isin(frame_names, elongated_names)
        ds = {"all": stats(v), "elongated_objects": stats(v[candidate_mask]), "other_objects": stats(v[~candidate_mask])}
        for cm in (5, 10, 20):
            bad = v[:, 2] > cm / 100
            ds[f"plane_gt_{cm}cm"] = {**stats(v[bad]), "elongated_object_share_pct": float(candidate_mask[bad].mean() * 100)}
        ds["elongated_classes"] = elongated_names
        ds["elongated_frame_share_pct"] = float(candidate_mask.mean() * 100)
        dataset_summaries[dataset] = ds
        plane, line, segment, extension, t = derive(v)
        np.savez_compressed(args.output / f"{dataset}_line_frames.npz",
                            values=np.column_stack((v[:, :2], plane, line, segment, t, extension.astype(float))),
                            columns=np.array(["file_index", "frame_index", "plane_m", "line_m", "segment_m", "projection_t", "on_extension"]), files=files)
    write_csv(args.output / "per_object_groups.csv", rows)
    write_csv(args.output / "per_sequence.csv", seq_rows)
    write_csv(args.output / "mesh_shape.csv", shape_rows)
    figures(args.output, rows, full, meshes)
    summary = {"source_snapshot": str(args.source), "source_summary_sha256": hashlib.sha256((args.source / "summary.json").read_bytes()).hexdigest(),
               "definition": {"line": "sqrt(midpoint_distance^2 - plane_distance^2)",
                              "segment": "sqrt(line_distance^2 + max(plane_distance-span/2,0)^2)",
                              "extension": "projection t<0 or t>1; t=0.5+signed_plane_distance/span"},
               "validation": checks, "datasets": dataset_summaries,
               "rod_handle_candidates_OMOMO": sorted(ROD_HANDLE_CANDIDATES),
               "classification_note": "Elongated = surface PCA standard deviation ratio major/second >=3 (descriptive threshold); semantic rod/shaft interpretation additionally checked in mesh figure. Mesh extents use native units without per-frame obj_scale, not physical centimeters."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in rows:
        if r["group"] in ("all", "plane_gt_10cm"):
            print(r["dataset"], r["object"], r["group"], json.dumps(r), flush=True)
    print("Saved", args.output)


if __name__ == "__main__":
    main()

