#!/usr/bin/env python3
"""Batched online evaluation of the cascade refine architecture.

Thin wrapper around the stock ``eval_batch_online.py`` (same data, noise, online
protocol, metrics and JSON outputs); only the model it evaluates changes:

  --model_kind stock    the stock IMUHOIModel (re-evaluates existing baselines)
  --model_kind b0       stock online loop, ObjectTrans = feedback-free cascade OT
  --model_kind cascade  HP -> VC -> OT -> VC-refine -> OT-refine -> HP-refine

All other arguments are passed to eval_batch_online unchanged.  ``--extra_json``
receives metrics the stock evaluator does not report: object jitter of both
the raw (loop) and the published (hand-corrected) trajectory for every model
kind, and contact precision/recall/F1 of the published contacts and, for the
cascade, of VC's first-pass contacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import eval_batch_online as ebo  # noqa: E402
from eval_IMUHOI import _build_published_object_trans  # noqa: E402
from model.cascade.cascade_model import build_cascade_model  # noqa: E402


def parse_wrapper_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_kind", choices=("stock", "b0", "cascade"), default="cascade")
    parser.add_argument("--vc_refine_ckpt", type=str, default=None)
    parser.add_argument("--cascade_mode", choices=("e1", "e2", "e2p"), default=None)
    parser.add_argument("--cascade_hp_history", choices=("stage1", "refined"), default=None)
    parser.add_argument("--cascade_identity_vc_refine", action="store_true",
                        help="VC-refine disabled (identity); the cascade then reduces to the feedback-free OT")
    parser.add_argument("--extra_json", type=str, default=None)
    known, rest = parser.parse_known_args()
    return known, rest


WRAPPER, REST = parse_wrapper_args()
CAPTURED = []


def _patched_load_model(config, device, no_trans=False, module_paths=None, object_trans_arch=None,
                        velocity_contact_arch=None):
    paths = dict(module_paths or {})
    if WRAPPER.model_kind == "stock":
        return ORIGINAL_LOAD_MODEL(config, device, no_trans=no_trans, module_paths=paths,
                                   object_trans_arch=object_trans_arch, velocity_contact_arch=velocity_contact_arch)
    config.object_trans_state_feedback = False
    if WRAPPER.cascade_mode:
        config.cascade_refit_mode = WRAPPER.cascade_mode
    if WRAPPER.cascade_hp_history:
        config.cascade_hp_history = WRAPPER.cascade_hp_history
    if WRAPPER.model_kind == "cascade":
        config.cascade_vc_refine = not WRAPPER.cascade_identity_vc_refine
        if not WRAPPER.cascade_identity_vc_refine:
            if not WRAPPER.vc_refine_ckpt:
                raise ValueError("--vc_refine_ckpt is required for --model_kind cascade")
            paths["vc_refine"] = WRAPPER.vc_refine_ckpt
    print(f"[cascade-eval] kind={WRAPPER.model_kind} mode={getattr(config, 'cascade_refit_mode', 'e2')} "
          f"hp_history={getattr(config, 'cascade_hp_history', 'stage1')} "
          f"vc_refine={'identity' if WRAPPER.cascade_identity_vc_refine else WRAPPER.vc_refine_ckpt}")
    return build_cascade_model(config, device, paths, kind=WRAPPER.model_kind, no_trans=no_trans,
                               velocity_contact_arch=velocity_contact_arch)


def _capturing_predict(*args, **kwargs):
    pred_cache, sample_cache = ORIGINAL_PREDICT(*args, **kwargs)
    config = args[2] if len(args) > 2 else kwargs.get("config")
    CAPTURED.append(extra_metrics(pred_cache, sample_cache, config))
    return pred_cache, sample_cache


ORIGINAL_LOAD_MODEL = ebo.load_model
ORIGINAL_PREDICT = ebo._batched_online_predict_dataset


# ------------------------------------------------------------------ extra metrics
def _jitter(trajectory: torch.Tensor) -> float:
    if trajectory is None or trajectory.shape[0] < 3:
        return float("nan")
    acceleration = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]
    return float(acceleration.norm(dim=-1).mean().item() * 1000.0)


def _row(value):
    return value[0] if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[0] == 1 else value


def _trajectories(pred, length):
    """(raw, published) object trajectories [T,3] of one sequence."""
    get = lambda key: _row(pred.get(key))[:length] if isinstance(pred.get(key), torch.Tensor) else None  # noqa: E731
    if isinstance(pred.get("cascade_raw_obj_trans"), torch.Tensor):
        return get("cascade_raw_obj_trans"), get("pred_obj_trans")
    raw = get("pred_obj_trans")
    stage1 = get("stage1_pred_palm_glb_pos")
    refined = get("pred_palm_glb_pos")
    published = raw
    if stage1 is not None and refined is not None:
        published = _build_published_object_trans(
            raw, stage1, refined, get("pred_lhand_obj_trans"), get("pred_rhand_obj_trans"),
            get("pred_imu_obj_trans"), get("pred_static_obj_trans"), get("gating_weights"),
        )
    return raw, published


def _counts(prob, target, threshold):
    pred = prob > threshold
    return np.array([(pred & target).sum(), (pred & ~target).sum(), (~pred & target).sum()], dtype=np.int64)


def _summary(counts):
    tp, fp, fn = (int(v) for v in counts)
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "fp_frames": fp,
        "fn_frames": fn,
    }


def extra_metrics(pred_cache, sample_cache, config):
    threshold = float(getattr(config, "hand_contact_threshold", 0.5))
    jitter = {"raw": [], "published": []}
    per_sequence = []
    contact = {}
    static_fp = {}
    for sample in sample_cache:
        pred = pred_cache.get(sample["_cache_key"])
        if pred is None:
            continue
        length = int(_row(pred["pred_obj_trans"]).shape[0])
        raw, published = _trajectories(pred, length)
        record = {"seq_path": sample["_cache_key"], "obj_jitter_raw": _jitter(raw),
                  "obj_jitter_published": _jitter(published)}
        jitter["raw"].append(record["obj_jitter_raw"])
        jitter["published"].append(record["obj_jitter_published"])
        gt = np.stack([sample[key][:length].bool().numpy() for key in ("lhand_contact", "rhand_contact")], axis=-1)
        gt_move = sample["obj_contact"][:length].bool().numpy()
        static_free = (~gt_move) & ~gt.any(axis=-1)
        for tag, key in (("published", "pred_hand_contact_prob"), ("first_pass", "cascade_vc_first_pass_prob")):
            prob = pred.get(key)
            if not isinstance(prob, torch.Tensor):
                continue
            prob = _row(prob)[:length, :2].float().numpy()
            for thr_tag, thr in (("eval_thr", threshold), ("thr_0.5", 0.5)):
                bucket = contact.setdefault(tag, {}).setdefault(thr_tag, np.zeros(3, dtype=np.int64))
                bucket += _counts(prob, gt, thr)
            fp_any = (prob > 0.5).any(axis=-1) & static_free
            stats = static_fp.setdefault(tag, [0, 0])
            stats[0] += int(fp_any.sum())
            stats[1] += int(static_free.sum())
            record[f"hand_fp_frames_{tag}"] = int(((prob > threshold) & ~gt).sum())
            record[f"hand_fn_frames_{tag}"] = int(((prob <= threshold) & gt).sum())
        per_sequence.append(record)
    return {
        "hand_contact_threshold": threshold,
        "obj_jitter_raw": float(np.nanmean(jitter["raw"])) if jitter["raw"] else None,
        "obj_jitter_published": float(np.nanmean(jitter["published"])) if jitter["published"] else None,
        "contact": {tag: {thr: _summary(c) for thr, c in by_thr.items()} for tag, by_thr in contact.items()},
        "static_nohand_fp_rate_thr_0.5": {tag: v[0] / max(v[1], 1) for tag, v in static_fp.items()},
        "per_sequence": per_sequence,
    }


def main() -> int:
    sys.argv = [sys.argv[0]] + REST
    ebo.load_model = _patched_load_model
    ebo._batched_online_predict_dataset = _capturing_predict
    code = ebo.main()
    if WRAPPER.extra_json and CAPTURED:
        payload = {
            "model_kind": WRAPPER.model_kind,
            "cascade_mode": WRAPPER.cascade_mode,
            "cascade_hp_history": WRAPPER.cascade_hp_history,
            "vc_refine_ckpt": WRAPPER.vc_refine_ckpt,
            "identity_vc_refine": WRAPPER.cascade_identity_vc_refine,
            "args": REST,
            "datasets": CAPTURED,
        }
        path = Path(WRAPPER.extra_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary = {k: v for k, v in CAPTURED[0].items() if k != "per_sequence"}
        print("[cascade-eval] extra metrics:", json.dumps(summary, indent=1))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
