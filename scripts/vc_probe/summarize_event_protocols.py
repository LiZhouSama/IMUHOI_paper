#!/usr/bin/env python3
"""Protocol sensitivity of the A1 contact-event results (event F1 at 10 frames, threshold 0.5).

Settings: every model under the unified 7/3 protocol, every model under the old 0/0 protocol, and each model
under the protocol it was trained / selected with ("native"; OMOMO v4 and IMUHOI-VC are 0/0, the rest 7/3,
and on BEHAVE every pipeline is 7/3).

    python scripts/vc_probe/summarize_event_protocols.py --out experiments/03_vc_contact_event_timing/protocol_sensitivity.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVENT_ROOT = ROOT / "outputs/IMUHOI++/RNN_cano/vc_event_timing_20261028"
SETTINGS = (
    ("OMOMO", "all 7/3", "analysis_omomo"),
    ("OMOMO", "all 0/0", "analysis_omomo_all00"),
    ("OMOMO", "native", "analysis_omomo_p00"),
    ("BEHAVE", "all 7/3 (= native)", "analysis_behave"),
    ("BEHAVE", "all 0/0", "analysis_behave_all00"),
)
EVENTS = ("hand_onset", "hand_offset", "obj_onset", "obj_offset")
MODELS = ("v5ar", "v4", "imuhoivc", "imuhoi3b")
PAIRS = ("v4-imuhoivc", "v5ar-v4", "v5ar-imuhoivc", "v5ar-imuhoi3b")


def cell(d):
    star = "*" if d[1] > 0 or d[2] < 0 else ""
    return f"{d[0]:+.3f}{star} [{d[1]:+.3f}, {d[2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lines = ["# A1 protocol sensitivity: event F1 (tolerance 10 frames, threshold 0.5)", "",
             "7/3 = HP smoothing half-width 7 + 3-frame published-trajectory velocity init; 0/0 = neither (VC's own",
             "previous output initializes its velocity). Native = the protocol each pipeline was trained and selected",
             "with: OMOMO v4 and IMUHOI-VC 0/0, OMOMO v5ar and IMUHOI 3-branch 7/3, every BEHAVE pipeline 7/3.",
             "`*` = 95 % CI (sequence bootstrap, paired) excludes 0.", "",
             "## Event F1 per model", "",
             "| dataset | setting | event | " + " | ".join(MODELS) + " |", "|---|---|---|" + "---|" * len(MODELS)]
    results = {}
    for ds, name, tag in SETTINGS:
        r = json.loads((EVENT_ROOT / tag / "results.json").read_text())
        results[(ds, name)] = r
        for ev in EVENTS:
            lines.append(f"| {ds} | {name} | {ev} | " + " | ".join(f"{r['thr'][ev][m]['f1'][2]:.3f}" for m in MODELS) + " |")
    lines += ["", "## Paired differences", "",
              "| dataset | setting | event | " + " | ".join(PAIRS) + " |", "|---|---|---|" + "---|" * len(PAIRS)]
    for (ds, name), r in results.items():
        for ev in EVENTS:
            lines.append(f"| {ds} | {name} | {ev} | " + " | ".join(cell(r["thr"][ev]["pairs"][p]["f1_tol10"]) for p in PAIRS) + " |")
    lines += ["", "## Frame-level contact F1 (left / right / object) from the eval JSONs", "",
              "| dataset | setting | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for (ds, name), r in results.items():
        fl = r["frame_level"]
        lines.append(f"| {ds} | {name} | " + " | ".join(
            f"{fl[m]['contact_f1_lhand']:.3f} / {fl[m]['contact_f1_rhand']:.3f} / {fl[m]['contact_f1_obj']:.3f}" for m in MODELS) + " |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
