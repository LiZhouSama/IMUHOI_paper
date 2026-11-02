#!/usr/bin/env python3
"""Run eval_batch_online unchanged and save VC internal features at every published frame.

The VelocityContact forwards are wrapped at class level: forward hooks collect the internal
activations of each call and return them as extra ``feat_*`` keys.  The online loop already
slices every temporal VC output at the published frame of its window (and keeps frames
[0, W) of the warmup pass), so the saved features are exactly the network state that produced
each published contact decision.  The features are removed from the predictions before the
metric pass, so the evaluation JSON must reproduce the pipeline's own eval.

    python scripts/vc_probe/dump_vc_features.py --feature_dir OUT/features \
        [--random_control | --probs_only] -- <eval_batch_online.py arguments>

``--probs_only`` leaves the VC untouched and saves only its probability outputs (any VC architecture).

Saved per sequence (fp16, [T_published, D]):
  v5     feat_b (boundary features b_t), feat_grp (5 body-group GRU outputs), feat_in (group-encoder
         inputs = normalized HP pose streams), feat_hc (hand-contact LSTM), feat_om (object-motion
         BiLSTM), feat_cpl (object + hand coupling features), rand_b (b_t of a re-initialized copy)
  imuhoi feat_hc (contact LSTM), feat_cin (contact-net input: forearm/object IMU + velocities)
plus pred_hand_contact_prob [T, 3] (left, right, object motion) and, when the VC has them, the endpoint
heatmaps [T, 6] (l_start, l_end, r_start, r_end, obj_start, obj_end) and anticipation probabilities [T, 6].
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import eval_batch_online as ebo  # noqa: E402
from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule  # noqa: E402
from model.rnn.velocity_contact_v5 import V5VelocityContactModule  # noqa: E402

FEATURE_DTYPE = torch.float16
STATE = {"random_control": False, "random_seed": 1234, "saved": 0, "saved_keys": set(), "feature_dir": None,
         "probs_only": False}
PROB_KEYS = ("pred_hand_contact_prob", "pred_contact_endpoint_prob", "pred_contact_anticipation_prob")


def _output_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _hook_output(store, key):
    def hook(_module, _inputs, output):
        store.setdefault(key, []).append(_output_tensor(output).detach())
    return hook


def _hook_input(store, key):
    def hook(_module, inputs, _output):
        store.setdefault(key, []).append(inputs[0].detach())
    return hook


def _register_v5_hooks(module, store, prefix):
    handles = [module.boundary_hidden_norm.register_forward_hook(_hook_output(store, f"{prefix}b"))]
    if prefix == "feat_":
        for name, encoder in module.group_encoders.items():
            handles.append(encoder.register_forward_hook(_hook_output(store, f"grp:{name}")))
            handles.append(encoder.register_forward_hook(_hook_input(store, f"in:{name}")))
        handles.append(module.hand_contact_net.rnn.register_forward_hook(_hook_output(store, "feat_hc")))
        handles.append(module.obj_move_net.rnn.register_forward_hook(_hook_output(store, "feat_om")))
    return handles


def _single(store, key):
    values = store.get(key, [])
    if len(values) != 1:
        raise RuntimeError(f"expected exactly one activation for {key}, got {len(values)}")
    return values[0]


def _random_copy(module):
    """Same architecture and buffers, freshly initialized parameters (untrained control)."""
    device = next(module.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.inference_mode(False), torch.random.fork_rng(devices=devices):
        torch.manual_seed(STATE["random_seed"])
        clone = copy.deepcopy(module)
        for sub in clone.modules():
            if sub is not clone and hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
    clone.eval()
    return clone


_orig_v5_forward = V5VelocityContactModule.forward
_orig_v5_coupling = V5VelocityContactModule._coupling_features
_orig_imuhoi_forward = IMUHOIVelocityContactModule.forward


def _v5_coupling(self, *args, **kwargs):
    out = _orig_v5_coupling(self, *args, **kwargs)
    store = getattr(self, "_probe_store", None)
    if store is not None:
        store.setdefault("cpl", []).append(torch.cat([o.float() for o in out], dim=-1).detach())
    return out


def _v5_forward(self, data_dict, hp_out=None, velocity_only=False):
    if velocity_only or getattr(self, "_probe_inner", False):
        return _orig_v5_forward(self, data_dict, hp_out=hp_out, velocity_only=velocity_only)
    store = {}
    self._probe_store = store
    handles = _register_v5_hooks(self, store, "feat_")
    try:
        out = _orig_v5_forward(self, data_dict, hp_out=hp_out, velocity_only=velocity_only)
    finally:
        for handle in handles:
            handle.remove()
        self._probe_store = None
    out = dict(out)
    names = list(self.group_encoders.keys())
    out["feat_b"] = _single(store, "feat_b").to(FEATURE_DTYPE)
    out["feat_grp"] = torch.cat([_single(store, f"grp:{n}") for n in names], dim=-1).to(FEATURE_DTYPE)
    out["feat_in"] = torch.cat([_single(store, f"in:{n}") for n in names], dim=-1).to(FEATURE_DTYPE)
    out["feat_hc"] = _single(store, "feat_hc").to(FEATURE_DTYPE)
    out["feat_om"] = _single(store, "feat_om").to(FEATURE_DTYPE)
    out["feat_cpl"] = _single(store, "cpl").to(FEATURE_DTYPE)
    if STATE["random_control"]:
        control = getattr(self, "_probe_random_copy", None)
        if control is None:
            control = _random_copy(self)
            control._probe_inner = True
            object.__setattr__(self, "_probe_random_copy", control)  # not a registered submodule
        rstore = {}
        rhandles = _register_v5_hooks(control, rstore, "rand_")
        try:
            control(data_dict, hp_out=hp_out)
        finally:
            for handle in rhandles:
                handle.remove()
        out["rand_b"] = _single(rstore, "rand_b").to(FEATURE_DTYPE)
    return out


def _imuhoi_forward(self, data_dict, hp_out=None, velocity_only=False):
    if velocity_only:
        return _orig_imuhoi_forward(self, data_dict, hp_out=hp_out, velocity_only=velocity_only)
    store = {}
    handles = [
        self.contact_net.rnn.register_forward_hook(_hook_output(store, "feat_hc")),
        self.contact_net.linear1.register_forward_hook(_hook_input(store, "feat_cin")),
    ]
    try:
        out = _orig_imuhoi_forward(self, data_dict, hp_out=hp_out, velocity_only=velocity_only)
    finally:
        for handle in handles:
            handle.remove()
    out = dict(out)
    out["feat_hc"] = _single(store, "feat_hc").to(FEATURE_DTYPE)
    out["feat_cin"] = _single(store, "feat_cin").to(FEATURE_DTYPE)
    return out


_orig_group_runner = ebo._run_batched_online_group


def _group_runner(model, config, device, samples, *args, **kwargs):
    # The evaluator re-enters this function for sub-groups of short sequences, so an outer
    # call can receive outputs whose features a nested call has already saved.
    outputs = _orig_group_runner(model, config, device, samples, *args, **kwargs)
    feature_dir = STATE["feature_dir"]
    for sample, output in zip(samples, outputs):
        if sample["_cache_key"] in STATE["saved_keys"]:
            continue
        keys = [key for key in output if key.startswith(("feat_", "rand_"))]
        if not keys and not STATE["probs_only"]:
            raise RuntimeError("no VC features in the online output; hooks did not fire")
        record = {key: output.pop(key).squeeze(0).contiguous() for key in keys}
        for key in PROB_KEYS:
            if isinstance(output.get(key), torch.Tensor):
                record[key] = output[key].squeeze(0).float().clone()
        record["cache_key"] = sample["_cache_key"]
        record["seq_name"] = sample.get("seq_name")
        torch.save(record, feature_dir / f"{Path(sample['_cache_key']).stem}.pt")
        STATE["saved"] += 1
        STATE["saved_keys"].add(sample["_cache_key"])
    return outputs


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: dump_vc_features.py --feature_dir DIR [--random_control] -- <eval args>")
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=Path, required=True)
    parser.add_argument("--random_control", action="store_true")
    parser.add_argument("--probs_only", action="store_true", help="do not hook the VC; save probability outputs only")
    own = parser.parse_args(argv[:split])
    if own.probs_only and own.random_control:
        raise SystemExit("--probs_only and --random_control are exclusive")
    own.feature_dir.mkdir(parents=True, exist_ok=True)
    STATE["feature_dir"] = own.feature_dir
    STATE["random_control"] = own.random_control
    STATE["probs_only"] = own.probs_only

    if not own.probs_only:
        V5VelocityContactModule.forward = _v5_forward
        V5VelocityContactModule._coupling_features = _v5_coupling
        IMUHOIVelocityContactModule.forward = _imuhoi_forward
    ebo._run_batched_online_group = _group_runner

    sys.argv = [str(ROOT / "eval_batch_online.py")] + argv[split + 1:]
    rc = ebo.main()
    print(f"[vc_probe] saved features for {STATE['saved']} sequences to {own.feature_dir}")
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
