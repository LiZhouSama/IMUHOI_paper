#!/usr/bin/env python3
"""A3: downstream sensitivity to contact errors -- eval_batch_online with corrupted GT contact as the VC output.

The evaluator's own oracle (``--oracle_vc_contact all [--oracle_vc_shift K]``) replaces the VC contact channels
(left hand, right hand, object motion) by GT labels, optionally delayed by K frames.  ``--flip_frac f`` adds
flicker on top: in every sequence and channel, runs of 3-10 frames are inverted, starting with probability
f / 6.5 per frame (about a fraction f of the frames is flipped).  The flip pattern is seeded by sequence file name
and channel only, so every pipeline receives exactly the same corrupted contact.

    python scripts/vc_probe/dump_contact_oracle.py [--flip_frac 0.1] -- <eval_batch_online.py arguments> \
        --oracle_vc_contact all [--oracle_vc_shift K]
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import eval_batch_online as ebo  # noqa: E402

MIN_RUN, MAX_RUN = 3, 10
STATE = {"flip_frac": 0.0, "samples": None, "flipped": 0, "frames": 0}
_orig_group_runner = ebo._run_batched_online_group
_orig_build = ebo._build_vc_oracles


def flicker(labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Invert random 3-10 frame runs of ``labels`` so that about ``frac`` of the frames are flipped."""
    rng = np.random.default_rng(seed)
    flip = np.zeros(labels.shape[0], dtype=bool)
    starts = np.flatnonzero(rng.random(labels.shape[0]) < frac / ((MIN_RUN + MAX_RUN) / 2.0))
    for start, length in zip(starts, rng.integers(MIN_RUN, MAX_RUN + 1, size=starts.size)):
        flip[start:start + length] = True
    return np.where(flip, 1.0 - labels, labels), flip


def _build(raw_batch, spec, device, dtype):
    oracles = _orig_build(raw_batch, spec, device, dtype)
    if STATE["flip_frac"] > 0 and "oracle_vc_contact" in oracles:
        oracle = oracles["oracle_vc_contact"].float().cpu().numpy()
        for row, sample in enumerate(STATE["samples"]):
            length = int(sample["lhand_contact"].shape[0])
            stem = Path(sample["_cache_key"]).stem
            for channel in range(oracle.shape[-1]):
                values = oracle[row, :length, channel]
                finite = np.isfinite(values)
                if not finite.any():
                    continue
                seed = zlib.crc32(f"{stem}|{channel}".encode())
                flipped, mask = flicker(np.where(finite, values, 0.0), STATE["flip_frac"], seed)
                oracle[row, :length, channel] = np.where(finite, flipped, values)
                STATE["flipped"] += int((mask & finite).sum())
                STATE["frames"] += int(finite.sum())
        oracles["oracle_vc_contact"] = torch.from_numpy(oracle).to(device=device, dtype=dtype)
    return oracles


def _group_runner(model, config, device, samples, *args, **kwargs):
    STATE["samples"] = list(samples)
    return _orig_group_runner(model, config, device, samples, *args, **kwargs)


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit(__doc__)
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip_frac", type=float, default=0.0)
    own = ap.parse_args(argv[:split])
    eval_args = argv[split + 1:]
    if "--oracle_vc_contact" not in eval_args:
        raise SystemExit("pass --oracle_vc_contact all (and optionally --oracle_vc_shift K) to the evaluator")
    STATE["flip_frac"] = own.flip_frac
    ebo._build_vc_oracles = _build
    ebo._run_batched_online_group = _group_runner
    sys.argv = [str(ROOT / "eval_batch_online.py")] + eval_args
    rc = ebo.main()
    if own.flip_frac > 0:
        print(f"[A3] flicker {own.flip_frac:.3f}: flipped {STATE['flipped']} of {STATE['frames']} oracle frames "
              f"({STATE['flipped'] / max(STATE['frames'], 1):.3f})")
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
