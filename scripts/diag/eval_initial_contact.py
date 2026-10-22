#!/usr/bin/env python3
"""Run eval_batch_online.py with a constant initial contact state.

The full-sequence test splits label hand contact 0 at frame 0 for every
sequence (backward-difference artifact of the preprocessing), and the online
protocol uses the frame-0 labels as the initial contact state of the warmup
window.  This wrapper replaces that initial state (``contact_init`` of the
sequence, i.e. the warmup window's VC/OT initializer) with a constant, without
touching the evaluator itself:

    python scripts/diag/eval_initial_contact.py --initial_hand_contact 0.5 \
        [--initial_obj_contact 0.5] <eval_batch_online.py arguments>

Later windows keep the published-history initializers.  The override is
recorded in the output JSON as ``initial_contact_override``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import eval_batch_online  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--initial_hand_contact", type=float, default=None)
    parser.add_argument("--initial_obj_contact", type=float, default=None)
    own, rest = parser.parse_known_args()
    if own.initial_hand_contact is None and own.initial_obj_contact is None:
        raise SystemExit("set --initial_hand_contact and/or --initial_obj_contact")
    if "--contact_init_frame" in rest:
        raise SystemExit("--contact_init_frame would overwrite the constant initial state")

    original = eval_batch_online.build_model_input_dict

    def build_with_constant_contact(*args, **kwargs):
        data_dict = original(*args, **kwargs)
        init = data_dict.get("contact_init")
        if init is not None:
            init = init.clone()
            if own.initial_hand_contact is not None:
                init[..., :2] = own.initial_hand_contact
            if own.initial_obj_contact is not None:
                init[..., 2] = own.initial_obj_contact
            data_dict["contact_init"] = init
        return data_dict

    eval_batch_online.build_model_input_dict = build_with_constant_contact
    sys.argv = ["eval_batch_online.py"] + rest
    code = eval_batch_online.main()
    if code not in (None, 0):
        return code

    if "--output_json" in rest:
        path = Path(rest[rest.index("--output_json") + 1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["initial_contact_override"] = {
            "hand": own.initial_hand_contact,
            "obj": own.initial_obj_contact,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
