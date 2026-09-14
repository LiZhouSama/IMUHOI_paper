#!/usr/bin/env python3
"""Dump per-frame ObjectTrans predictions for offline diagnosis.

Runs one pipeline (HP + VC + OT checkpoints taken from an ablation manifest)
over an OMOMO full-sequence split (test by default) in the batched online protocol
used by ``eval_batch_online.py`` or the offline whole-sequence protocol, and
stores a compact per-sequence record with the prediction tensors that matter
for the object branch plus the ground truth needed to score them.

    python scripts/dump_ot_predictions.py \
        --manifest outputs/.../experiment_manifest.json \
        --variant latch --mode online --out /path/to/latch_online.pt

The dump intentionally mirrors the eval settings (seed, simulated IMU noise,
window, lookahead) so per-frame errors reproduce the published metrics.

For frozen online train/validation caches, use --split train --sequence_list
<list.txt> for each disjoint list generated from the training split. Relative
entries are .pt paths within the chosen split; absolute paths must stay inside
it. Text lists retain line order (blank/# lines ignored); JSON string arrays
are also accepted. Selection precedes --limit and dataset preloading. OMOMO
training prefers the configured omomo_full original-sequence train directory;
--data_path explicitly overrides it. No-options test discovery is unchanged.

Predictions use the real HP -> VC -> OT inference path, never GT-derived
per-frame pose/contact/bone features. The existing evaluation protocol's
first-frame state initialization and dataset-generated IMUs are retained and
recorded in provenance. Both raw and published/re-fused object tracks are saved.
GT targets include gt_palm_pos_eval as a backward-compatible alias of the
canonical preprocessed gt_palm_pos; no evaluation BodyModel forward is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.dataset_IMUHOI import IMUDataset  # noqa: E402
from eval_batch_online import (  # noqa: E402
    _batched_online_predict_dataset,
    _collate_sample_slices,
    _detach_to_cpu,
)
from eval_IMUHOI import (  # noqa: E402
    _build_published_object_trans,
    _set_eval_seed,
    _to_smpl24_joint_positions,
    _unwrap_model,
    get_default_dataset_config,
)
from model.rnn.imuhoi_model import load_model  # noqa: E402
from model.rnn.online import (  # noqa: E402
    resolve_online_lookahead,
    resolve_online_window,
)
from utils.ot_dump_selection import resolve_dump_data_path, select_sequence_paths  # noqa: E402
from utils.human_pose import select_hand_anchor_positions  # noqa: E402
from utils.utils import build_model_input_dict, load_config  # noqa: E402

PRED_KEYS = (
    "pred_obj_trans",
    "gating_weights",
    "pred_hand_contact_prob",
    "pred_lhand_obj_trans",
    "pred_rhand_obj_trans",
    "pred_imu_obj_trans",
    "pred_static_obj_trans",
    "pred_lhand_obj_direction",
    "pred_rhand_obj_direction",
    "pred_lhand_lb",
    "pred_rhand_lb",
    "pred_palm_glb_pos",
    "stage1_pred_palm_glb_pos",
    "pred_hand_glb_pos",
    "root_trans_pred",
    "stage1_root_trans_pred",
    "pred_obj_vel",
    # v3 extras
    "pred_lhand_obj_vector",
    "pred_rhand_obj_vector",
    "pred_lhand_segment_anchor",
    "pred_rhand_segment_anchor",
    "pred_lhand_segment_pivot",
    "pred_rhand_segment_pivot",
    "segment_contact_active",
    "segment_rel_rot6d",
    "pred_lhand_bone_proposal",
    "pred_rhand_bone_proposal",
    "pred_lhand_bone_gate",
    "pred_rhand_bone_gate",
    # state-trust extras (object_trans_state_trust)
    "state_confidence",
    "prior_lhand_obj_trans",
    "prior_rhand_obj_trans",
    "state_trust_rest_age",
    "state_trust_gap_age",
    "state_trust_rest_flag",
    "ot_hand_positions",
)
GT_KEYS = (
    "obj_trans",
    "obj_rot",
    "lhand_contact",
    "rhand_contact",
    "obj_contact",
    "trans",
    "obj_imu",
    "human_imu",
)


def _squeeze_batch(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == 1:
        return value[0].detach().cpu().clone()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return value


def _record_from(
    sample: Dict[str, Any], pred: Dict[str, Any], publish_len: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "seq_name": str(sample.get("seq_name", "")),
        "seq_file": str(sample.get("seq_file", "")),
        "obj_name": str(sample.get("obj_name", "")),
        "has_object": bool(sample.get("has_object", True)),
        "seq_len": int(sample["human_imu"].shape[0]),
        "publish_len": int(publish_len),
    }
    for key in GT_KEYS:
        value = sample.get(key)
        if isinstance(value, torch.Tensor):
            record[f"gt_{key}"] = value[:publish_len].detach().cpu().clone()
    joints = sample.get("position_global")
    if isinstance(joints, torch.Tensor):
        smpl24 = _to_smpl24_joint_positions(joints[:publish_len])
        record["gt_palm_pos"] = select_hand_anchor_positions(smpl24).detach().cpu().clone()
        # Backward-compatible alias; canonical preprocessing makes both identical.
        record["gt_palm_pos_eval"] = record["gt_palm_pos"].clone()
    for key in PRED_KEYS:
        value = pred.get(key)
        if isinstance(value, torch.Tensor):
            value = _squeeze_batch(value)
            record[key] = value[:publish_len].clone()


    # Use the evaluator's causal re-fusion, not GT or the raw OT feedback track.
    raw = record.get("pred_obj_trans")
    palm = record.get("pred_palm_glb_pos")
    if isinstance(raw, torch.Tensor):
        published = raw if raw.shape[0] == 0 else _build_published_object_trans(
            raw,
            record.get("stage1_pred_palm_glb_pos", palm),
            palm,
            record.get("pred_lhand_obj_trans"),
            record.get("pred_rhand_obj_trans"),
            record.get("pred_imu_obj_trans"),
            record.get("pred_static_obj_trans"),
            record.get("gating_weights"),
        )
        record["pred_obj_trans_published"] = published.detach().cpu().clone()

    # Cheap consistency check against predicted FK; no ground-truth joints.
    pred_joints = pred.get("pred_joints_global")
    if isinstance(palm, torch.Tensor) and isinstance(pred_joints, torch.Tensor) and publish_len:
        fk_palm = select_hand_anchor_positions(
            _to_smpl24_joint_positions(_squeeze_batch(pred_joints)[:publish_len])
        )
        if fk_palm.shape == palm.shape:
            record["pred_palm_fk_max_abs_m"] = float((fk_palm - palm).abs().max())
    return record


def _build_config(manifest: Dict[str, Any], variant: str):
    pipeline = manifest["pipelines"][variant]
    config = load_config(pipeline["config"])
    config.model_arch = "rnn"
    config.object_trans_arch = pipeline["object_trans_arch"]
    config.velocity_contact_arch = pipeline["velocity_contact_arch"]
    config.enable_ot_refine = bool(getattr(config, "enable_ot_refine", False))
    if not hasattr(config, "debug"):
        config.debug = False
    module_paths = {
        "human_pose": manifest["reused_checkpoints"]["hp_ckpt"],
        "velocity_contact": pipeline["joint_vc_ckpt"],
        "object_trans": pipeline["joint_ot_ckpt"],
    }
    return config, module_paths, pipeline


def _offline_predict(
    model, dataset: IMUDataset, config, device, refine_human: bool,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for index in tqdm(range(len(dataset)), desc="Offline", unit="seq", dynamic_ncols=True):
        sample = dict(dataset[index])
        seq_len = int(sample["human_imu"].shape[0])
        batch = _collate_sample_slices([sample], [0], [seq_len])
        data_dict = build_model_input_dict(batch, config, device, add_noise=False)
        with torch.inference_mode():
            out = model.inference(
                data_dict,
                use_object_data=True,
                compute_fk=False,
                refine_human=refine_human,
                inference_mode="offline",
            )
        records.append(_record_from(sample, _detach_to_cpu(out), seq_len))
    return records


def _file_provenance(path_like: str | Path) -> Dict[str, Any]:
    path = Path(path_like).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="omomo")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--data_path", help="Explicit full-sequence split directory (relative to project root).")
    parser.add_argument("--sequence_list", help="Text/JSON list of .pt files within the selected split; applied before --limit.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None, help="Default: CUDA if available, otherwise CPU.")
    parser.add_argument("--online_batch_size", type=int, default=32)
    parser.add_argument("--online_window", type=int, default=None)
    parser.add_argument("--online_lookahead", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_imu_noise", action="store_true", help="Disable the simulated IMU noise used by the ablation eval.")
    parser.add_argument("--limit", type=int, default=None, help="Only dump the first N sequences (smoke test).")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.online_batch_size <= 0:
        parser.error("--online_batch_size must be a positive integer")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    with open(args.manifest, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    config, module_paths, pipeline = _build_config(manifest, args.variant)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _set_eval_seed(args.seed)

    dataset_configs = get_default_dataset_config(config=config)
    data_path = resolve_dump_data_path(
        dataset_configs, args.dataset, PROJECT_ROOT, split=args.split, data_path=args.data_path,
    )
    # Preserve the legacy test glob order and batch ties for no-options calls.
    legacy_selection = (
        args.split == "test" and args.data_path is None
        and args.sequence_list is None and args.limit is None
    )
    sequence_paths = None if legacy_selection else [
        str(path) for path in select_sequence_paths(data_path, args.sequence_list, args.limit)
    ]
    path_dataset = (
        "omomo_full" if args.split == "train" and args.dataset == "omomo"
        and "omomo_full" in dataset_configs and args.data_path is None else args.dataset
    )
    dataset_cfg = dataset_configs[path_dataset]
    provenance = {
        "manifest": _file_provenance(args.manifest),
        "config": _file_provenance(pipeline["config"]),
        "resolved_config": json.loads(json.dumps(config)),
        "checkpoints": {name: _file_provenance(path) for name, path in module_paths.items()},
        "pipeline": dict(pipeline),
        "dataset": args.dataset,
        "path_dataset": path_dataset,
        "dataset_config": dict(dataset_cfg),
        "split": args.split,
        "data_path": str(data_path),
        "device": str(device),
        "sequence_list": _file_provenance(args.sequence_list) if args.sequence_list else None,
        "requested_sequence_paths": sequence_paths,
        "limit": args.limit,
        "selection_order": "legacy_dataset_discovery" if legacy_selection else (
            "sequence_list" if args.sequence_list else "lexicographic_filename"
        ),
        "online_batch_size": args.online_batch_size,
        "frozen_modules": ["human_pose", "velocity_contact", "object_trans"],
        "feature_source": "frozen_pipeline_predictions",
        "initialization": "existing_eval_first_frame_states_then_predicted_online_history",
        "ground_truth_usage": "initial_states_and_dataset_imu_synthesis_and_saved_targets_only",
        "feedback_mode": "normal",
        "published_object_trans": "eval_IMUHOI._build_published_object_trans; model-predicted FK palms",
        "source_files": {
            name: _file_provenance(PROJECT_ROOT / name) for name in (
                "scripts/dump_ot_predictions.py", "utils/ot_dump_selection.py",
                "eval_IMUHOI.py", "eval_batch_online.py",
            )
        },
    }
    test_window = config.test.get("window", config.train.get("window", 60))
    dataset = IMUDataset(
        data_dir=str(data_path),
        window_size=test_window,
        debug=False,
        simulate_imu_noise=not args.no_imu_noise,
        simulate_imu_noise_seed=args.seed,
        min_obj_contact_frames=0,
        full_sequence=True,
        sequence_paths=sequence_paths,
        resolve_bimanual_contact_conflicts=config.get("resolve_bimanual_contact_conflicts", False),
        require_canonical_skeleton=bool(getattr(config, "require_canonical_skeleton", False)),
    )
    loaded_paths = [str(Path(info["file_path"]).resolve()) for info in dataset.sequence_info]
    loaded_set = set(loaded_paths)
    skipped_paths = [path for path in (sequence_paths or []) if path not in loaded_set]
    if args.sequence_list and skipped_paths:
        raise ValueError(f"Dataset rejected explicitly selected sequences: {skipped_paths}")
    if not loaded_paths:
        raise ValueError(f"No valid sequences to cache from {data_path}")
    provenance["selected_sequences"] = [dict(info) for info in dataset.sequence_info]
    provenance["selected_sequence_paths"] = loaded_paths
    provenance["skipped_sequence_paths"] = skipped_paths
    provenance["available_file_count"] = sum(1 for path in data_path.glob("*.pt") if path.is_file())
    print(f"[Dump] {args.variant}/{args.mode}/{args.split}: {len(dataset)} sequences from {data_path}")

    model = load_model(
        config,
        device,
        module_paths=module_paths,
        object_trans_arch=config.object_trans_arch,
        velocity_contact_arch=config.velocity_contact_arch,
    )
    model = _unwrap_model(model).eval()
    model.requires_grad_(False)

    provenance["gt_palm_pos_eval"] = {
        "definition": "alias of dataset canonical position_global palm anchors",
        "body_model_forward": False,
        "canonical_body_model_gender": "neutral",
        "canonical_betas": "zero",
    }
    refine_human = bool(config.enable_ot_refine)

    online_window = resolve_online_window(config, args.online_window)
    online_lookahead = resolve_online_lookahead(config, args.online_lookahead)
    started = time.time()
    if args.mode == "online":
        with torch.inference_mode():
            pred_cache, sample_cache = _batched_online_predict_dataset(
                model,
                dataset,
                config,
                device,
                online_batch_size=args.online_batch_size,
                online_window=online_window,
                online_lookahead=online_lookahead,
                use_object_data=True,
                compute_fk=False,
                refine_human=refine_human,
                sort_desc=True,
                eval_imu_noise=False,
                eval_imu_noise_seed=None,
            )
        records = []
        for sample in sample_cache:
            pred = pred_cache[sample["_cache_key"]]
            publish_len = max(int(sample["human_imu"].shape[0]) - online_lookahead, 0)
            records.append(_record_from(sample, pred, publish_len))
    else:
        records = _offline_predict(model, dataset, config, device, refine_human)

    provenance["record_sequence_files"] = [record["seq_file"] for record in records]
    payload = {
        "schema_version": 2,
        "dataset": args.dataset,
        "split": args.split,
        "provenance": provenance,
        "variant": args.variant,
        "mode": args.mode,
        "object_trans_arch": pipeline["object_trans_arch"],
        "online_window": online_window,
        "online_lookahead": online_lookahead,
        "seed": args.seed,
        "simulate_imu_noise": not args.no_imu_noise,
        "refine_human": refine_human,
        "data_path": str(data_path),
        "elapsed_sec": time.time() - started,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"[Dump] wrote {len(records)} records to {out} in {payload['elapsed_sec']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
