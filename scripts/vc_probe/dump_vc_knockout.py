#!/usr/bin/env python3
"""A2: body-part knockouts of the IMUHOI++ VC (v5) and integrated-gradients attribution of its contact decisions.

Runs eval_batch_online unchanged apart from class-level patches of the v5 VelocityContact module, and saves the
published per-frame VC probabilities (like ``dump_vc_features.py --probs_only``).

Reference run (no knockout), with any of
  --stats_out PATH    means of every pose input of the VC over all its forward calls (knockout and IG baseline):
                      the five body-group encoder inputs, the boundary features b_t, the palm velocity /
                      acceleration streams and the 48 hand coupling features;
  --capture_out PATH  the exact VC input of the windows that publish a GT hand-contact onset / offset (sliding-window
                      frames [90, T_pub - 30) only);
  --ig_out PATH       after the evaluation, integrated gradients of the published hand-contact probability of those
                      windows w.r.t. the body-group encoder inputs (baseline = their means, 64 midpoint steps).
Knockout runs: --knockout {left_arm,right_arm,torso,legs,pose_prior,hp_all} --means PATH
  group names: that group's encoder input is replaced by its mean (the pose-prior pathway only);
  pose_prior:  b_t is replaced by its mean (every body group at once);
  hp_all:      pose_prior + palm velocity/acceleration + the palm-derived coupling signals (span, span rate,
               palm relative speed) replaced by their means: no HP information reaches the VC.

    python scripts/vc_probe/dump_vc_knockout.py --probs_dir OUT/probs [--knockout NAME --means M.pt]
        [--stats_out M.pt --capture_out W.pt --ig_out IG.pt] -- <eval_batch_online.py arguments>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dump_vc_features as D  # noqa: E402  (imports eval_batch_online and records its group runner)
import eval_batch_online as ebo  # noqa: E402
from model.rnn.imuhoi_model import IMUHOIModel  # noqa: E402
from model.rnn.velocity_contact_v4 import HAND_POSITION_HP_OUT_KEYS, REQUIRED_HP_OUT_KEYS  # noqa: E402
from model.rnn.velocity_contact_v5 import V5VelocityContactModule  # noqa: E402

GROUPS = ("left_arm", "right_arm", "torso", "left_leg", "right_leg")
KNOCKOUT_GROUPS = {"left_arm": ("left_arm",), "right_arm": ("right_arm",), "torso": ("torso",),
                   "legs": ("left_leg", "right_leg"), "pose_prior": (), "hp_all": ()}
# Palm-derived coupling signals (span, span rate, palm relative speed) inside the 4 x 12 multiscale layout.
HP_COUPLING_DIMS = [block * 12 + k for block in range(4) for k in (8, 9, 10)]
LOOKAHEAD, WARMUP, HALF = 10, 60, 30
HP_KEYS = tuple(k for k, _ in REQUIRED_HP_OUT_KEYS) + HAND_POSITION_HP_OUT_KEYS
INPUT_KEYS = ("human_imu", "obj_imu", "hand_vel_glb_init", "obj_vel_init", "obj_trans_init", "contact_init")

CTX = {"knockout": None, "means": None, "stats": None, "capture": False, "samples": None, "chunk": None,
       "wanted": {}, "windows": [], "module": None}

_orig_forward = V5VelocityContactModule.forward
_orig_streams = V5VelocityContactModule._prepare_pose_streams
_orig_coupling = V5VelocityContactModule._coupling_features
_orig_prep = IMUHOIModel._prepare_online_chunk


def _accumulate(name, value):
    """Running sum over every leading (batch, time) position of ``value``'s trailing feature shape."""
    stats = CTX["stats"]
    if stats is None:
        return
    flat = value.detach().double().reshape(-1, *value.shape[2:])
    entry = stats.setdefault(name, [torch.zeros(flat.shape[1:], dtype=torch.float64, device=flat.device), 0])
    entry[0] += flat.sum(0)
    entry[1] += flat.shape[0]


def _mean(name, like):
    return CTX["means"][name].to(device=like.device, dtype=like.dtype).expand_as(like)


def _streams(self, *args, **kwargs):
    streams = _orig_streams(self, *args, **kwargs)
    if getattr(self, "_a2_quiet", False):
        return streams
    for key in ("hand_vel", "hand_acc"):
        _accumulate(key, streams[key])
    if CTX["knockout"] == "hp_all":
        streams = dict(streams)
        for key in ("hand_vel", "hand_acc"):
            streams[key] = _mean(key, streams[key])
    return streams


def _coupling(self, *args, **kwargs):
    object_features, hand_features = _orig_coupling(self, *args, **kwargs)
    if getattr(self, "_a2_quiet", False):
        return object_features, hand_features
    _accumulate("hand_coupling", hand_features)
    if CTX["knockout"] == "hp_all":
        hand_features = hand_features.clone()
        mean = CTX["means"]["hand_coupling"].to(device=hand_features.device, dtype=hand_features.dtype)
        hand_features[..., HP_COUPLING_DIMS] = mean[HP_COUPLING_DIMS]
    return object_features, hand_features


def _forward(self, data_dict, hp_out=None, velocity_only=False):
    if velocity_only or getattr(self, "_a2_quiet", False):
        return _orig_forward(self, data_dict, hp_out=hp_out, velocity_only=velocity_only)
    CTX["module"] = self
    handles = []
    knockout = CTX["knockout"]
    for name in KNOCKOUT_GROUPS.get(knockout, ()):
        handles.append(self.group_encoders[name].register_forward_pre_hook(
            lambda _m, inputs, n=name: (_mean(f"grp:{n}", inputs[0]),)))
    if knockout in ("pose_prior", "hp_all"):
        handles.append(self.boundary_hidden_norm.register_forward_hook(lambda _m, _i, out: _mean("b", out)))
    if CTX["stats"] is not None:
        for name in GROUPS:
            handles.append(self.group_encoders[name].register_forward_pre_hook(
                lambda _m, inputs, n=name: _accumulate(f"grp:{n}", inputs[0])))
        handles.append(self.boundary_hidden_norm.register_forward_hook(lambda _m, _i, out: _accumulate("b", out)))
    try:
        out = _orig_forward(self, data_dict, hp_out=hp_out, velocity_only=velocity_only)
    finally:
        for handle in handles:
            handle.remove()
    if CTX["capture"] and CTX["chunk"] is not None:
        _capture(data_dict, hp_out if hp_out is not None else data_dict.get("hp_out"))
    return out


def _capture(data_dict, hp_out):
    targets, batch_size = CTX["chunk"]
    for i, target in enumerate(targets):
        for b, sample in enumerate(CTX["samples"]):
            events = CTX["wanted"].get(sample["_cache_key"], {}).get(int(target))
            if not events:
                continue
            row = i * batch_size + b
            window = {k: data_dict[k][row].detach().cpu().clone() for k in INPUT_KEYS
                      if isinstance(data_dict.get(k), torch.Tensor)}
            window["hp_out"] = {k: hp_out[k][row].detach().cpu().clone() for k in HP_KEYS
                                if isinstance(hp_out.get(k), torch.Tensor)}
            for hand, kind in events:
                CTX["windows"].append({"key": sample["_cache_key"], "file": Path(sample["_cache_key"]).stem,
                                       "target": int(target), "hand": hand, "kind": kind, "inputs": window})


def _prep(self, data_dict, history, targets, window, lookahead, batch_size, seq_len, *args, **kwargs):
    CTX["chunk"] = (list(targets), int(batch_size))
    try:
        return _orig_prep(self, data_dict, history, targets, window, lookahead, batch_size, seq_len, *args, **kwargs)
    finally:
        CTX["chunk"] = None


def _segments(binary):
    padded = np.concatenate([[False], binary, [False]])
    diff = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(diff == 1), np.flatnonzero(diff == -1) - 1))


def _group_runner(model, config, device, samples, *args, **kwargs):
    CTX["samples"] = list(samples)
    if CTX["capture"]:
        for sample in samples:
            wanted = {}
            T_pub = int(sample["lhand_contact"].shape[0]) - LOOKAHEAD
            for hand, key in enumerate(("lhand_contact", "rhand_contact")):
                labels = np.asarray(sample[key]).astype(bool)[:T_pub]
                for start, end in _segments(labels):
                    for kind, frame in (("onset", start), ("offset", end)):
                        if WARMUP + HALF <= frame < T_pub - HALF:
                            wanted.setdefault(int(frame), []).append((hand, kind))
            CTX["wanted"][sample["_cache_key"]] = wanted
    return D._group_runner(model, config, device, samples, *args, **kwargs)


def integrated_gradients(module, windows, means, steps=64):
    """IG of the published (index W-1) hand-contact probability w.r.t. each body-group encoder input."""
    module.eval()
    module._a2_quiet = True
    device = next(module.parameters()).device
    publish = WARMUP - 1
    alphas = torch.cat([torch.zeros(1), (torch.arange(steps) + 0.5) / steps, torch.ones(1)]).to(device)
    base = {n: means[f"grp:{n}"].to(device=device, dtype=torch.float32) for n in GROUPS}
    results = []
    with torch.backends.cudnn.flags(enabled=False):
        for w in windows:
            n_rows = alphas.numel()
            # The captured tensors were made under inference_mode; clone() outside it gives autograd-usable ones.
            inputs = {k: v.clone().to(device).float().unsqueeze(0).expand(n_rows, *v.shape).contiguous()
                      for k, v in w["inputs"].items() if k != "hp_out"}
            hp = {k: v.clone().to(device).float().unsqueeze(0).expand(n_rows, *v.shape).contiguous()
                  for k, v in w["inputs"]["hp_out"].items()}
            inputs["hp_out"] = hp
            leaves = {}

            def hook(_m, args, name):
                x = args[0].detach().float()
                xi = (base[name] + alphas.view(-1, 1, 1) * (x - base[name])).requires_grad_(True)
                leaves[name] = (xi, x[0])
                return (xi,)

            handles = [module.group_encoders[n].register_forward_pre_hook(
                lambda m, a, n=n: hook(m, a, n)) for n in GROUPS]
            try:
                out = _orig_forward(module, inputs, hp_out=hp)
                prob = out["pred_hand_contact_prob"][:, publish, w["hand"]].float()
                grads = torch.autograd.grad(prob[1:-1].sum(), [leaves[n][0] for n in GROUPS])
            finally:
                for handle in handles:
                    handle.remove()
            attribution = torch.stack([((leaves[n][1] - base[n]) * g[1:-1].mean(0)).sum(-1)
                                       for n, g in zip(GROUPS, grads)]).detach().cpu()  # [5, window]
            record = {k: w[k] for k in ("key", "file", "target", "hand", "kind")}
            record.update(ig=attribution, p_full=float(prob[-1]), p_base=float(prob[0]))
            results.append(record)
    module._a2_quiet = False
    return results


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit(__doc__)
    split = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs_dir", type=Path, required=True)
    ap.add_argument("--knockout", choices=sorted(KNOCKOUT_GROUPS), default=None)
    ap.add_argument("--means", type=Path, default=None)
    ap.add_argument("--stats_out", type=Path, default=None)
    ap.add_argument("--capture_out", type=Path, default=None)
    ap.add_argument("--ig_out", type=Path, default=None)
    own = ap.parse_args(argv[:split])
    if own.knockout and own.means is None:
        raise SystemExit("--knockout needs --means")
    if own.ig_out and not (own.stats_out and own.capture_out):
        raise SystemExit("--ig_out needs --stats_out and --capture_out in the same run")
    if own.knockout and (own.stats_out or own.capture_out):
        raise SystemExit("statistics / capture belong to the reference run, not to a knockout run")

    own.probs_dir.mkdir(parents=True, exist_ok=True)
    D.STATE.update(feature_dir=own.probs_dir, probs_only=True)
    CTX["knockout"] = own.knockout
    CTX["means"] = torch.load(own.means, map_location="cpu") if own.means else None
    CTX["stats"] = {} if own.stats_out else None
    CTX["capture"] = own.capture_out is not None

    V5VelocityContactModule.forward = _forward
    V5VelocityContactModule._prepare_pose_streams = _streams
    V5VelocityContactModule._coupling_features = _coupling
    IMUHOIModel._prepare_online_chunk = _prep
    ebo._run_batched_online_group = _group_runner

    sys.argv = [str(ROOT / "eval_batch_online.py")] + argv[split + 1:]
    rc = ebo.main()
    print(f"[A2] saved probabilities for {D.STATE['saved']} sequences to {own.probs_dir}"
          + (f" (knockout {own.knockout})" if own.knockout else ""))

    if own.stats_out:
        means = {k: (s / max(n, 1)).float().cpu() for k, (s, n) in CTX["stats"].items()}
        CTX["stats"] = None
        own.stats_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(means, own.stats_out)
        print(f"[A2] means of {sorted(means)} -> {own.stats_out}")
    if own.capture_out:
        torch.save(CTX["windows"], own.capture_out)
        print(f"[A2] captured {len(CTX['windows'])} event windows -> {own.capture_out}")
    if own.ig_out:
        results = integrated_gradients(CTX["module"], CTX["windows"], means)
        gap = np.array([abs(r["ig"].sum().item() - (r["p_full"] - r["p_base"])) for r in results])
        torch.save(results, own.ig_out)
        print(f"[A2] integrated gradients for {len(results)} windows -> {own.ig_out}; "
              f"completeness |sum IG - (p - p_base)| median {np.median(gap):.2e}, max {gap.max():.2e}")
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
