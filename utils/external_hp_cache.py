"""Per-sequence cache of an external human-pose estimator's online predictions.

HP plug-in experiment (2026-11-05): a comparison method (Comparisons/: DIP, TIP, TransPose,
GlobalPose) replaces our HumanPose stage while VC and OT are retrained on top of it.  Each
method runs once per sequence in its own paper-faithful online protocol
(``Comparisons/eval_comparison.py --save_hp_cache``), and its SMPL local rotations and root
translation are stored here.  ``build_model_input_dict`` slices the cached frames of every
batch window into ``ext_hp_*`` tensors, which ``model/rnn/external_human_pose.py`` turns into
the HumanPose output dict.

Layout of a cache directory::

    manifest.json                      format, method, dataset, latency, source directories
    seq/<key>                          one torch file per sequence (see ``cache_entry_path``)
    eval/<directory key>.json          the comparison eval of that directory (baseline metrics)

``<key>`` is the sequence file's path relative to the last ``/process/`` of its real path
(e.g. ``IMUHOI++cano/processed_split_data_OMOMO_full/test/x.pt``), so caches built on one
machine are valid on any checkout.  This module only needs torch; Comparisons imports it.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import torch

CACHE_FORMAT = "imuhoi_external_hp_cache"
CACHE_VERSION = 1
MANIFEST_NAME = "manifest.json"
SEQUENCE_DIR = "seq"
EXTERNAL_HP_KEYS = ("ext_hp_local_rot6d", "ext_hp_root_disp")
_ANCHOR = "/process/"


def external_hp_cache_key(file_path) -> str:
    """Machine-independent key of a processed sequence file (path below ``process/``)."""
    real = os.path.realpath(os.path.abspath(str(file_path))).replace(os.sep, "/")
    anchor = real.rfind(_ANCHOR)
    if anchor < 0:
        raise ValueError(
            f"{file_path} does not resolve below a 'process/' directory; external HP caches "
            "are keyed by the path relative to it"
        )
    return real[anchor + len(_ANCHOR):]


def cache_entry_path(cache_dir, key: str) -> Path:
    if not key or key.startswith("/") or ".." in Path(key).parts:
        raise ValueError(f"invalid external HP cache key {key!r}")
    return Path(cache_dir) / SEQUENCE_DIR / key


def directory_key(directory) -> str:
    """Key of a data directory (same convention as the sequence keys)."""
    return external_hp_cache_key(Path(directory) / "_").rsplit("/", 1)[0]


def read_manifest(cache_dir) -> Dict[str, Any]:
    path = Path(cache_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"external HP cache manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(f"{path} is not an external HP cache manifest (format={manifest.get('format')!r})")
    if int(manifest.get("version", 0)) != CACHE_VERSION:
        raise ValueError(f"{path}: unsupported cache version {manifest.get('version')!r}")
    if manifest.get("status") != "complete":
        raise RuntimeError(f"{path}: cache is not complete (status={manifest.get('status')!r})")
    return manifest


def resolve_cache_dir(path_like) -> Path:
    """Accept the cache directory or its manifest.json (used as the HP 'checkpoint' path)."""
    path = Path(str(path_like)).expanduser()
    if path.name == MANIFEST_NAME:
        path = path.parent
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / path
    return path.resolve()


def external_hp_cache_dir(cfg) -> Optional[Path]:
    value = cfg.get("external_hp_cache_dir") if isinstance(cfg, dict) else getattr(cfg, "external_hp_cache_dir", None)
    if value is None or str(value).strip() in {"", "None", "null"}:
        return None
    return resolve_cache_dir(value)


class _Entry:
    __slots__ = ("local_rot6d", "root_disp", "num_frames")

    def __init__(self, local_rot6d: torch.Tensor, root_trans: torch.Tensor):
        local_rot6d = local_rot6d.detach().to(dtype=torch.float32, device="cpu").contiguous()
        root_trans = root_trans.detach().to(dtype=torch.float32, device="cpu").contiguous()
        if local_rot6d.dim() != 3 or tuple(local_rot6d.shape[1:]) != (24, 6):
            raise ValueError(f"cached local_rot6d must be [T, 24, 6], got {tuple(local_rot6d.shape)}")
        if root_trans.shape != (local_rot6d.shape[0], 3):
            raise ValueError(f"cached root_trans must be [T, 3], got {tuple(root_trans.shape)}")
        # Per-frame root displacement d[t] = r[t] - r[t-1].  A window integrates it from its
        # own initial translation, exactly as HumanPose integrates its velocities; d[0] (only
        # used as the first frame's velocity feature) repeats d[1].
        disp = torch.zeros_like(root_trans)
        if root_trans.shape[0] > 1:
            disp[1:] = root_trans[1:] - root_trans[:-1]
            disp[0] = disp[1]
        self.local_rot6d = local_rot6d
        self.root_disp = disp
        self.num_frames = int(local_rot6d.shape[0])


class ExternalHPCache:
    """Lazily loaded per-sequence predictions of one cache directory (one per process)."""

    _instances: Dict[str, "ExternalHPCache"] = {}
    _lock = threading.Lock()

    def __init__(self, cache_dir):
        self.cache_dir = resolve_cache_dir(cache_dir)
        self.manifest = read_manifest(self.cache_dir)
        self._entries: Dict[str, _Entry] = {}
        self._keys: Dict[str, str] = {}

    @classmethod
    def get(cls, cache_dir) -> "ExternalHPCache":
        resolved = str(resolve_cache_dir(cache_dir))
        with cls._lock:
            instance = cls._instances.get(resolved)
            if instance is None:
                instance = cls(resolved)
                cls._instances[resolved] = instance
            return instance

    def key(self, seq_path) -> str:
        # realpath walks every path component (slow on WSL drives); once per file.
        seq_path = str(seq_path)
        key = self._keys.get(seq_path)
        if key is None:
            key = external_hp_cache_key(seq_path)
            self._keys[seq_path] = key
        return key

    def entry(self, seq_path) -> _Entry:
        key = self.key(seq_path)
        entry = self._entries.get(key)
        if entry is None:
            path = cache_entry_path(self.cache_dir, key)
            if not path.is_file():
                raise KeyError(
                    f"external HP cache {self.cache_dir} has no entry for {key} "
                    f"(sequence {seq_path}); rebuild the cache for this data directory"
                )
            payload = torch.load(path, map_location="cpu")
            if payload.get("key") != key:
                raise ValueError(f"{path} holds key {payload.get('key')!r}, expected {key!r}")
            entry = _Entry(payload["local_rot6d"], payload["root_trans"])
            self._entries[key] = entry
        return entry


def _per_sample(value, batch_size: int, name: str):
    if value is None:
        raise KeyError(f"external HP inputs need batch[{name!r}]")
    if isinstance(value, torch.Tensor):
        value = value.reshape(-1).tolist()
    elif isinstance(value, (str, int)):
        value = [value]
    value = list(value)
    if len(value) == 1 and batch_size > 1:
        value = value * batch_size
    if len(value) != batch_size:
        raise ValueError(f"batch[{name!r}] has {len(value)} entries for a batch of {batch_size}")
    return value


def attach_external_hp_inputs(batch, cfg, device, dtype, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
    """``ext_hp_*`` tensors ``[B, T, ...]`` for the windows of ``batch``; ``{}`` when not configured.

    Each sample reads frames ``[window_start, window_end)`` of its sequence.  A sample shorter
    than ``seq_len`` (the online evaluator repeat-pads sequences to the longest one in its
    batch) is repeat-padded the same way; padded frames are never published.
    """
    cache_dir = external_hp_cache_dir(cfg)
    if cache_dir is None:
        return {}
    cache = ExternalHPCache.get(cache_dir)
    paths = _per_sample(batch.get("seq_path"), batch_size, "seq_path")
    starts = _per_sample(batch.get("window_start"), batch_size, "window_start")
    ends = _per_sample(batch.get("window_end"), batch_size, "window_end")
    rot_rows, disp_rows = [], []
    for path, start, end in zip(paths, starts, ends):
        entry = cache.entry(path)
        start, end = int(start), int(end)
        length = end - start
        if length <= 0 or length > seq_len or start < 0 or end > entry.num_frames:
            raise ValueError(
                f"window [{start}, {end}) of {path} does not fit a batch of {seq_len} frames "
                f"or the cached sequence of {entry.num_frames} frames"
            )
        rot = entry.local_rot6d[start:end]
        disp = entry.root_disp[start:end]
        if length < seq_len:
            pad = seq_len - length
            rot = torch.cat((rot, rot[-1:].expand(pad, -1, -1)), dim=0)
            disp = torch.cat((disp, disp[-1:].expand(pad, -1)), dim=0)
        rot_rows.append(rot)
        disp_rows.append(disp)
    return {
        "ext_hp_local_rot6d": torch.stack(rot_rows).to(device=device, dtype=dtype),
        "ext_hp_root_disp": torch.stack(disp_rows).to(device=device, dtype=dtype),
    }


def write_cache_entry(cache_dir, seq_path, local_rot6d: torch.Tensor, root_trans: torch.Tensor, **extra) -> str:
    """Atomically write one sequence (used by Comparisons/eval_comparison.py --save_hp_cache)."""
    key = external_hp_cache_key(seq_path)
    path = cache_entry_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "local_rot6d": local_rot6d.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "root_trans": root_trans.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        "num_frames": int(local_rot6d.shape[0]),
        **extra,
    }
    _Entry(payload["local_rot6d"], payload["root_trans"])  # shape check before writing
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return key
