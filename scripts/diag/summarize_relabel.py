#!/usr/bin/env python3
"""Summary table for the 2026-09-18 relabel retraining (baseline / EAG / biclose / p800_histmix on OMOMO and BEHAVE).

    python scripts/diag/summarize_relabel.py [--tag 20260918]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "outputs/IMUHOI++/RNN_cano"
VARIANTS = ("cur_hoi_p800", "cur_hoi_eag", "cur_hoi_biclose", "cur_hoi_p800_histmix", "cur_hoi_armrot", "cur_hoi_nofb")
# Variants whose VelocityContact architecture differs live in their own root.
EXTRA_ROOTS = (("relabel_20260920_{ds}_imuhoi", "vc_imuhoi_ot_4heads"),)
KEYS = (("obj_trans_err_fusion_refine", "obj"), ("hoi_err_fusion", "HOI"), ("obj_jitter", "jitter"),
        ("mpjpe", "MPJPE"), ("root_trans_err", "root"), ("contact_f1_lhand", "F1_L"), ("contact_f1_rhand", "F1_R"),
        ("contact_f1_obj", "F1_obj"))
# Pre-relabel reference numbers (OMOMO joint best / BEHAVE final), for orientation only.
OLD = {"omomo": {"cur_hoi_p800": (13.80, 9.20), "cur_hoi_eag": (14.50, 10.66), "cur_hoi_biclose": (13.51, 9.45),
                 "cur_hoi_p800_histmix": (13.17, 9.28)},
       "behave": {"cur_hoi_p800": (46.61, 40.20), "cur_hoi_eag": (41.70, 33.93), "cur_hoi_biclose": (41.60, 34.79),
                  "cur_hoi_p800_histmix": (40.22, 32.40)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="20260918")
    args = ap.parse_args()
    for ds in ("omomo", "behave"):
        root = ROOT / f"relabel_{args.tag}_{ds}"
        print(f"\n### {ds.upper()} ({root.name})")
        print("| variant | ckpt | " + " | ".join(label for _, label in KEYS) + " | old obj / HOI |")
        print("|---|---|" + "---|" * (len(KEYS) + 1))
        entries = [(root, v) for v in VARIANTS]
        entries += [(ROOT / tpl.format(ds=ds), v) for tpl, v in EXTRA_ROOTS]
        for src, v in entries:
            for tag in ("best", "final"):
                path = src / v / f"eval_{ds}_{tag}.json"
                if not path.is_file():
                    continue
                r = json.loads(path.read_text())["results"][ds]
                cells = " | ".join("" if r.get(k) is None else f"{r[k]:.2f}" for k, _ in KEYS)
                old = OLD[ds].get(v)
                print(f"| {v} | {tag} | {cells} | {'' if old is None else f'{old[0]:.2f} / {old[1]:.2f}'} |")


if __name__ == "__main__":
    main()
