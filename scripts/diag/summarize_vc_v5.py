#!/usr/bin/env python3
"""Summarize the 2026-09-30 VC-v5 experiments next to the VC-v4 reference.

Rows: every eval_{ds}_{best,final}.json of the v2_armrot_biclose variant under the
VC-v4 reference roots (v2_armrot_biclose_20260923_{ds}) and the VC-v5 roots
(vc_v5_20260930_ot_{ds}_{variant}), plus the plug-in checks (standalone VC on the
existing BEHAVE OT) in vc_v5_20260930/.

    python scripts/diag/summarize_vc_v5.py [--markdown]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "outputs/IMUHOI++/RNN_cano"
KEYS = (
    ("obj_trans_err_fusion_refine", "obj"),
    ("hoi_err_fusion", "HOI"),
    ("obj_jitter", "jitter"),
    ("root_trans_err", "root"),
    ("mpjpe", "MPJPE"),
    ("contact_f1_lhand", "F1 L"),
    ("contact_f1_rhand", "F1 R"),
    ("contact_f1_obj", "F1 obj"),
)


def metrics(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    if isinstance(results, dict) and len(results) == 1:
        results = next(iter(results.values()))
    return results


def rows():
    out = []
    for ds in ("omomo", "behave"):
        ref = ROOT / f"v2_armrot_biclose_20260923_{ds}/v2_armrot_biclose"
        for tag in ("best", "final"):
            path = ref / f"eval_{ds}_{tag}.json"
            if path.is_file():
                out.append((ds, tag, "VC-v4 (09-23 reference)", metrics(path)))
        for root in sorted(ROOT.glob(f"vc_v5_20260930_ot_{ds}_*")):
            if not root.is_dir():
                continue
            variant = root.name.rsplit("_", 1)[-1]
            for tag in ("best", "final"):
                path = root / "v2_armrot_biclose" / f"eval_{ds}_{tag}.json"
                if path.is_file():
                    out.append((ds, tag, f"VC-{variant}", metrics(path)))
    for ds, tag in (("behave", "final"), ("omomo", "best")):
        for path in sorted((ROOT / "vc_v5_20260930").glob(f"eval_{ds}_plugin_*.json")):
            if path.stem.endswith("_records"):
                continue
            name = path.stem.replace(f"eval_{ds}_plugin_", "")
            out.append((ds, f"{tag} OT, plug-in", f"standalone VC-{name}", metrics(path)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    table = rows()
    if args.markdown:
        print("| dataset | checkpoint | VC | " + " | ".join(label for _, label in KEYS) + " |")
        print("|---|---|---|" + "---:|" * len(KEYS))
    for ds, tag, name, values in table:
        cells = [f"{values.get(key, float('nan')):.3f}" for key, _ in KEYS]
        if args.markdown:
            print(f"| {ds} | {tag} | {name} | " + " | ".join(cells) + " |")
        else:
            print(f"{ds:7s} {tag:18s} {name:26s} " + " ".join(f"{label} {cell}" for (_, label), cell in zip(KEYS, cells)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
