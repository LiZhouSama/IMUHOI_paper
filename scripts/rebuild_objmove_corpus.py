"""Rebuild the canonical IMUHOI corpus with the jitter-robust obj_move criterion.

The reference corpus (process/IMUHOI++cano) is re-generated dataset by dataset
with the current preprocessing code into a new root that mirrors its layout:

    <new_root>/processed_split_data_<D>/{unified,train,test,split_info.json}
    <new_root>/processed_seg_data_<D>_full/{train,test}      (BEHAVE/HODOME/IMHD)

Steps per dataset:
  1. run the dataset preprocessing script into ``unified/`` (OMOMO additionally
     gets the reference corpus's min-foot-joint-y=0 floor shift);
  2. carry ``obj_points_canonical`` over from the reference file of the same
     name (it is an unseeded random surface sample, so re-sampling would change
     a model input unrelated to obj_move);
  3. split ``unified/`` by the reference split_info train/test file lists;
  4. re-segment with the reference seg_sample settings (seed 42, 150-300);
  5. compare every new file with its reference file key by key and write a
     JSON report (``rebuild_report_<D>.json`` under the new root).

Usage:
    python scripts/rebuild_objmove_corpus.py --datasets BEHAVE HODOME \
        --new-root process/IMUHOI++cano_objmove
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from process.preprocess import (  # noqa: E402
    CONTACT_POSTPROCESS_VERSION,
    OBJ_MOTION_HALF_WINDOW_FRAMES,
    OBJ_MOTION_LOWPASS_HZ,
    OBJ_MOTION_ROT_THRESHOLD_DEG,
    OBJ_MOTION_TRANS_THRESHOLD_M,
    OBJ_MOVE_MIN_RUN_FRAMES,
)

# split-dir name, preprocessing command (output dir appended by the driver), segmented?
DATASETS = {
    "OMOMO": {
        "split_dir": "processed_split_data_OMOMO_full",
        "cmd": [
            "process/preprocess.py",
            "--keep-object-displacement-without-hand-contact",
            "--no-resolve-bimanual-contact-conflicts",
        ],
        "output_flag": "--save_dir_unified",
        "segment": False,
        # The reference OMOMO corpus was additionally shifted vertically so that the
        # lowest foot joint sits at y=0; reproduce it (verified bit-exact).
        "floor_align": True,
    },
    "BEHAVE": {
        "split_dir": "processed_split_data_BEHAVE",
        "cmd": ["process/preprocess_BEHAVE.py", "--keep-object-displacement-without-hand-contact"],
        "output_flag": "--output_dir",
        "segment": True,
    },
    "HODOME": {
        "split_dir": "processed_split_data_HODOME",
        "cmd": ["process/preprocess_HODOME.py", "--keep-object-displacement-without-hand-contact"],
        "output_flag": "--output_dir",
        "segment": True,
    },
    "IMHD": {
        "split_dir": "processed_split_data_IMHD",
        "cmd": ["process/preprocess_IMHD.py", "--keep-object-displacement-without-hand-contact"],
        "output_flag": "--output_dir",
        "segment": True,
    },
    "PAHOI": {
        "split_dir": "processed_split_data_PAHOI_full",
        "cmd": ["process/preprocess_PAHOI.py", "--keep-object-displacement-without-hand-contact"],
        "output_flag": "--output_dir",
        "segment": False,
    },
}

SEG_ARGS = [
    "--min_len", "150", "--max_len", "300", "--seed", "42",
    "--body-model-type", "smplh", "--keep-object-displacement-without-hand-contact",
]

# Keys expected to change because of the obj_move criterion (and what derives from it).
CONTACT_DERIVED_KEYS = {
    "lhand_contact", "rhand_contact", "obj_contact",
    "lhand_contact_raw", "rhand_contact_raw", "obj_contact_raw",
    "lhand_lb", "rhand_lb", "lhand_obj_direction", "rhand_obj_direction",
    "contact_endpoint", "contact_endpoint_gauss", "contact_postprocess_version",
    "obj_motion_lowpass_hz", "obj_motion_half_window_frames",
    "obj_motion_trans_threshold_m", "obj_motion_rot_threshold_deg",
    "obj_move_min_run_frames",
}


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _run(cmd, log_path):
    print(f"$ {' '.join(cmd)}  (log: {log_path})", flush=True)
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}), see {log_path}")
    print(f"  done in {(time.time() - started) / 60:.1f} min", flush=True)


def carry_over_object_points(new_unified: Path, ref_unified: Path):
    replaced, missing = 0, []
    for path in sorted(new_unified.glob("*.pt")):
        ref_path = ref_unified / path.name
        if not ref_path.exists():
            missing.append(path.name)
            continue
        data, ref = _load(path), _load(ref_path)
        new_points, ref_points = data.get("obj_points_canonical"), ref.get("obj_points_canonical")
        if new_points is None or ref_points is None:
            continue
        if tuple(new_points.shape) != tuple(ref_points.shape):
            raise ValueError(f"{path.name}: obj_points_canonical shape changed")
        data["obj_points_canonical"] = ref_points.clone()
        torch.save(data, path)
        replaced += 1
    return replaced, missing


FLOOR_FOOT_JOINTS = [7, 8, 10, 11]
FLOOR_SHIFT_KEYS = ("trans", "position_global_full_gt_world", "obj_trans")


def apply_canonical_floor_alignment(new_unified: Path, ref_unified: Path):
    """Shift each sequence so min foot-joint y is 0 (the reference OMOMO step)."""
    aligned = 0
    for path in sorted(new_unified.glob("*.pt")):
        data = _load(path)
        if data.get("canonical_floor_alignment") is not None:
            continue  # already aligned; keep idempotent
        position = data["position_global_full_gt_world"]
        offset = float(position[:, FLOOR_FOOT_JOINTS, 1].min().item())
        ref_path = ref_unified / path.name
        if ref_path.exists():
            ref_offset = _load(ref_path).get("canonical_floor_offset_applied_m")
            if ref_offset is not None and abs(float(ref_offset) - offset) > 1e-6:
                raise ValueError(f"{path.name}: floor offset {offset} != reference {ref_offset}")
        for key in FLOOR_SHIFT_KEYS:
            value = torch.as_tensor(data[key]).clone()
            value[..., 1] = value[..., 1] - offset
            data[key] = value
        data["canonical_floor_alignment"] = "min_foot_joint_y_zero"
        data["canonical_floor_offset_applied_m"] = offset
        torch.save(data, path)
        aligned += 1
    return aligned


def split_by_reference(new_split_root: Path, ref_split_root: Path):
    ref_info = json.loads((ref_split_root / "split_info.json").read_text(encoding="utf-8"))
    unified = new_split_root / "unified"
    for subset in ("train", "test"):
        target = new_split_root / subset
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        names = ref_info[f"{subset}_files"]
        missing = [name for name in names if not (unified / name).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} {subset} files missing from {unified}: {missing[:5]}")
        for name in names:
            shutil.copy2(unified / name, target / name)

    info = dict(ref_info)
    info.update({
        "reference_corpus_split_info": str(ref_split_root / "split_info.json"),
        "unified_dir": str(unified.relative_to(PROJECT_ROOT)),
        "unified_total_files": len(list(unified.glob("*.pt"))),
        "contact_postprocess_version": int(CONTACT_POSTPROCESS_VERSION),
        "obj_move_criterion": {
            "lowpass_hz": OBJ_MOTION_LOWPASS_HZ,
            "half_window_frames": OBJ_MOTION_HALF_WINDOW_FRAMES,
            "trans_threshold_m": OBJ_MOTION_TRANS_THRESHOLD_M,
            "rot_threshold_deg": OBJ_MOTION_ROT_THRESHOLD_DEG,
            "min_run_frames": OBJ_MOVE_MIN_RUN_FRAMES,
            "hole_filling": False,
        },
        "obj_points_canonical_source": "carried over from reference corpus",
    })
    (new_split_root / "split_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    skip_manifest = unified / "skipped_sequences.json"
    if (ref_split_root / "skipped_sequences.json").exists() and skip_manifest.exists():
        shutil.copy2(skip_manifest, new_split_root / "skipped_sequences.json")


def _diff(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return "type"
        if a.shape != b.shape:
            return "shape"
        if a.dtype == torch.bool or b.dtype == torch.bool:
            return None if torch.equal(a.bool(), b.bool()) else "bool"
        return None if torch.equal(a, b) else float((a.double() - b.double()).abs().max())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return None if np.array_equal(np.asarray(a), np.asarray(b)) else "ndarray"
    try:
        return None if a == b else "value"
    except Exception:  # noqa: BLE001
        return "uncomparable"


def compare_dirs(new_dir: Path, ref_dir: Path):
    new_names = {p.name for p in new_dir.glob("*.pt")}
    ref_names = {p.name for p in ref_dir.glob("*.pt")}
    differing = collections.Counter()
    max_abs = {}
    frames = old_move = new_move = old_hand = new_hand = 0.0
    for name in sorted(new_names & ref_names):
        new, ref = _load(new_dir / name), _load(ref_dir / name)
        for key in set(new) | set(ref):
            if key not in new or key not in ref:
                differing[("only_new" if key in new else "only_ref") + ":" + key] += 1
                continue
            result = _diff(new[key], ref[key])
            if result is not None:
                differing[key] += 1
                if isinstance(result, float):
                    max_abs[key] = max(max_abs.get(key, 0.0), result)
        num = float(len(new["obj_contact"]))
        frames += num
        old_move += ref["obj_contact"].float().sum().item()
        new_move += new["obj_contact"].float().sum().item()
        old_hand += (ref["lhand_contact"].bool() | ref["rhand_contact"].bool()).float().sum().item()
        new_hand += (new["lhand_contact"].bool() | new["rhand_contact"].bool()).float().sum().item()
    unexpected = {
        key: count for key, count in differing.items()
        if key.split(":", 1)[-1] not in CONTACT_DERIVED_KEYS
    }
    return {
        "new_files": len(new_names),
        "ref_files": len(ref_names),
        "only_new": sorted(new_names - ref_names),
        "only_ref": sorted(ref_names - new_names),
        "differing_keys": dict(differing),
        "unexpected_differing_keys": unexpected,
        "max_abs_diff": max_abs,
        "obj_move_frac_ref": old_move / max(frames, 1.0),
        "obj_move_frac_new": new_move / max(frames, 1.0),
        "hand_contact_frac_ref": old_hand / max(frames, 1.0),
        "hand_contact_frac_new": new_hand / max(frames, 1.0),
    }


def rebuild(name: str, new_root: Path, ref_root: Path, skip_preprocess: bool):
    spec = DATASETS[name]
    new_split_root = new_root / spec["split_dir"]
    ref_split_root = ref_root / spec["split_dir"]
    unified = new_split_root / "unified"
    log_dir = new_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not skip_preprocess:
        if unified.exists() and any(unified.glob("*.pt")):
            raise FileExistsError(f"{unified} is not empty; remove it or pass --skip-preprocess")
        unified.mkdir(parents=True, exist_ok=True)
        _run(
            [sys.executable, *spec["cmd"], spec["output_flag"], str(unified)],
            log_dir / f"preprocess_{name}.log",
        )

    if spec.get("floor_align"):
        aligned = apply_canonical_floor_alignment(unified, ref_split_root / "unified")
        print(f"[{name}] canonical floor alignment applied to {aligned} files")
    replaced, missing = carry_over_object_points(unified, ref_split_root / "unified")
    print(f"[{name}] obj_points_canonical carried over for {replaced} files; no reference for {missing}")
    split_by_reference(new_split_root, ref_split_root)

    report = {"dataset": name, "points_carried_over": replaced, "no_reference": missing}
    for subset in ("unified", "train", "test"):
        report[f"split_{subset}"] = compare_dirs(new_split_root / subset, ref_split_root / subset)

    if spec["segment"]:
        seg_root = new_root / f"processed_seg_data_{name}_full"
        ref_seg_root = ref_root / f"processed_seg_data_{name}_full"
        for subset in ("train", "test"):
            target = seg_root / subset
            if target.exists():
                shutil.rmtree(target)
            _run(
                [sys.executable, "process/seg_sample.py",
                 "--input_dir", str((new_split_root / subset).relative_to(PROJECT_ROOT)),
                 "--output_dir", str(target.relative_to(PROJECT_ROOT)), *SEG_ARGS],
                log_dir / f"seg_{name}_{subset}.log",
            )
            report[f"seg_{subset}"] = compare_dirs(target, ref_seg_root / subset)

    report_path = new_root / f"rebuild_report_{name}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in report.items():
        if isinstance(value, dict) and "obj_move_frac_new" in value:
            print(
                f"[{name}] {key}: files new/ref {value['new_files']}/{value['ref_files']} | "
                f"obj_move {value['obj_move_frac_ref']:.3f}->{value['obj_move_frac_new']:.3f} | "
                f"hand {value['hand_contact_frac_ref']:.3f}->{value['hand_contact_frac_new']:.3f} | "
                f"unexpected keys {value['unexpected_differing_keys']}"
            )
    print(f"[{name}] report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), required=True)
    parser.add_argument("--new-root", default="process/IMUHOI++cano_objmove")
    parser.add_argument("--ref-root", default="process/IMUHOI++cano")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Reuse an existing unified/ and only redo the later steps.")
    args = parser.parse_args()
    new_root = (PROJECT_ROOT / args.new_root).resolve()
    ref_root = (PROJECT_ROOT / args.ref_root).resolve()
    if new_root == ref_root:
        raise ValueError("new root must differ from the reference root")
    for name in args.datasets:
        rebuild(name, new_root, ref_root, args.skip_preprocess)


if __name__ == "__main__":
    main()
