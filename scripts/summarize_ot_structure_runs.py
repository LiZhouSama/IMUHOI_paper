#!/usr/bin/env python3
"""Compare the 2026-09-14 OT structure variants (grasp latent) with the cur_hoi baselines.

OMOMO rows read ``eval_batch_online.json`` (official online protocol), BEHAVE rows read the
full-sequence ``eval_behave_best.json`` / ``eval_behave_final.json`` of the in-domain pipeline.
Deltas are relative to the baseline row of the same dataset; missing files print as "—" so the
table can be produced while runs are still in progress.

    python scripts/summarize_ot_structure_runs.py [--variants cur_hoi_eag cur_hoi_eag_mild ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs/IMUHOI++/RNN_cano"
OMOMO_BASELINE = ROOT / "omomo_cano_ot_trust_20260910_c/cur_hoi_p800/eval_batch_online.json"
OMOMO_ROOT = ROOT / "omomo_cano_ot_eag_20260914"
BEHAVE_ROOT = ROOT / "behave_full_trust_20260911"
OMOMO_KEYS = (("obj_trans_err_fusion_refine", "obj"), ("hoi_err_fusion", "HOI"), ("obj_jitter", "jitter"),
              ("mpjpe", "MPJPE"), ("root_trans_err", "root"))
BEHAVE_KEYS = (("obj_trans_err_fusion_refine", "obj"), ("hoi_err_fusion", "HOI"), ("obj_jitter", "jitter"),
               ("mpjpe", "MPJPE"), ("root_trans_err", "root"))


def load(path: Path, dataset: str):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", payload)
    return results.get(dataset) or next(iter(results.values()))


def row(name, metrics, base, keys):
    cells = []
    for key, _ in keys:
        if metrics is None or metrics.get(key) is None:
            cells.append("—")
            continue
        value = float(metrics[key])
        if base is not None and base.get(key) is not None and metrics is not base:
            cells.append(f"{value:.2f} ({value - float(base[key]):+.2f})")
        else:
            cells.append(f"{value:.2f}")
    return f"| {name} | " + " | ".join(cells) + " |"


def table(title, keys, rows):
    print(f"\n**{title}**\n")
    print("| variant | " + " | ".join(label for _, label in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for line in rows:
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["cur_hoi_eag", "cur_hoi_eag_mild", "cur_hoi_eag_modes"])
    parser.add_argument("--omomo_root", type=Path, default=OMOMO_ROOT)
    parser.add_argument("--behave_root", type=Path, default=BEHAVE_ROOT)
    parser.add_argument("--behave_reference", nargs="*", default=["cur_hoi_trust_perturb"],
                        help="extra BEHAVE rows from the same root for context")
    args = parser.parse_args()

    base_o = load(OMOMO_BASELINE, "omomo")
    rows = [row("cur_hoi_p800 (baseline)", base_o, base_o, OMOMO_KEYS)]
    for name in args.variants:
        rows.append(row(name, load(args.omomo_root / name / "eval_batch_online.json", "omomo"), base_o, OMOMO_KEYS))
    table("OMOMO (official online protocol, cm; delta vs cur_hoi_p800)", OMOMO_KEYS, rows)

    for tag in ("final", "best"):
        base_b = load(args.behave_root / "cur_hoi_p800" / f"eval_behave_{tag}.json", "behave")
        rows = [row(f"cur_hoi_p800 {tag} (baseline)", base_b, base_b, BEHAVE_KEYS)]
        for name in list(args.behave_reference) + list(args.variants):
            rows.append(row(f"{name} {tag}", load(args.behave_root / name / f"eval_behave_{tag}.json", "behave"), base_b, BEHAVE_KEYS))
        table(f"BEHAVE full sequences, in-domain pipeline, {tag} checkpoints (cm; delta vs cur_hoi_p800 {tag})", BEHAVE_KEYS, rows)


if __name__ == "__main__":
    main()
