#!/usr/bin/env python3
"""One-line-per-file table of online eval JSONs.

    python scripts/diag/summarize_eval_jsons.py label=path.json [...]   (dataset key is auto-detected)
"""
import json
import sys

KEYS = (("obj_trans_err_fusion_refine", "obj"), ("hoi_err_fusion", "HOI"), ("obj_jitter", "obj jit"), ("jitter", "body jit"),
        ("mpjpe", "MPJPE"), ("root_trans_err", "root"), ("contact_f1_lhand", "F1_L"), ("contact_f1_rhand", "F1_R"),
        ("contact_f1_obj", "F1_obj"))


def main():
    print("| run | " + " | ".join(label for _, label in KEYS) + " |")
    print("|---|" + "---|" * len(KEYS))
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        try:
            results = json.load(open(path))["results"]
        except (OSError, KeyError, ValueError) as exc:
            print(f"| {label} | unavailable: {exc} |")
            continue
        r = next(iter(results.values()))
        print(f"| {label} | " + " | ".join("" if r.get(k) is None else f"{r[k]:.2f}" for k, _ in KEYS) + " |")


if __name__ == "__main__":
    main()
