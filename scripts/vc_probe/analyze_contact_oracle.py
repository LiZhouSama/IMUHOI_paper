#!/usr/bin/env python3
"""A3 analysis: object and HOI error when the VC contact output is replaced by corrupted GT contact.

Runs come from run_a3.sh (dump_contact_oracle.py): GT contact (hands + object motion) delayed by K frames
(s-15 .. s15) or with flicker (f05 / f10 / f20 = about 5 / 10 / 20 % of the frames inverted in 3-10 frame runs),
for IMUHOI++ (VC-v5ar + OT-v2) and the original three-branch IMUHOI; both get identical corrupted contact.
Degradation = error minus the error with exact GT contact (s0), per sequence; CIs bootstrap over sequences and the
pipeline comparison is paired. The real VC of each pipeline is the reference line.

    python scripts/vc_probe/analyze_contact_oracle.py --dataset behave
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_contact_events as A  # noqa: E402

RUN_ROOT = A.RUNS / "vc_contact_oracle_20261028"
SHIFTS = (-15, -10, -5, 0, 5, 10, 15)
FLIPS = (0.0, 0.05, 0.10, 0.20)
SHIFT_RUN = {k: f"s{k}" for k in SHIFTS}
FLIP_RUN = {0.0: "s0", 0.05: "f05", 0.10: "f10", 0.20: "f20"}
PIPES = {"ours": "IMUHOI++ (VC-v5ar + OT-v2)", "imuhoi3b": "IMUHOI, original 3-branch"}
COLORS = {"ours": "#2a78d6", "imuhoi3b": "#eda100"}  # same slots as the A1 figures
REAL = {"ours": A.EVENT_ROOT / "{ds}_v5ar/eval.json", "imuhoi3b": A.PILOT_ROOT / "{ds}_imuhoi/eval.json"}
METRICS = (("obj_trans_err_fusion_refine", "obj_trans_err_fusion_published_cm", "object error"),
           ("hoi_err_fusion", "hoi_err_fusion_cm", "HOI error"))


def load_run(ds, pipe, name):
    tag = f"{ds}_{pipe}_{name}"
    result = json.loads((RUN_ROOT / f"{tag}.json").read_text())["results"][ds]
    records = json.loads((RUN_ROOT / f"{tag}_records.json").read_text())["datasets"][ds]
    per_seq = {r["seq_path"]: r for r in records}
    return result, per_seq


def per_sequence(per_seq, keys, rec_key):
    return np.array([np.nan if per_seq[k][rec_key] is None else float(per_seq[k][rec_key]) for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("omomo", "behave"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ds = args.dataset
    out_dir = args.out or RUN_ROOT / f"analysis_{ds}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for pipe in PIPES:
        for name in sorted(set(SHIFT_RUN.values()) | set(FLIP_RUN.values())):
            runs[(pipe, name)] = load_run(ds, pipe, name)
    keys = sorted(runs[("ours", "s0")][1])
    W = A.boot_weights(len(keys))
    real = {pipe: json.loads(Path(str(REAL[pipe]).format(ds=ds)).read_text())["results"][ds] for pipe in PIPES}

    results = {"dataset": ds, "sequences": len(keys), "runs": {}, "real": {}, "pairs": {}}
    for pipe in PIPES:
        results["real"][pipe] = {m: real[pipe][m] for m, _, _ in METRICS}
        for (p, name), (agg, per_seq) in runs.items():
            if p != pipe:
                continue
            row = {"contact_f1": [agg["contact_f1_lhand"], agg["contact_f1_rhand"], agg["contact_f1_obj"]]}
            for metric, rec_key, _ in METRICS:
                base = per_sequence(runs[(pipe, "s0")][1], keys, rec_key)
                cur = per_sequence(per_seq, keys, rec_key)
                diff = np.nan_to_num(cur - base)
                valid = (~np.isnan(cur - base)).astype(float)
                draws = (W @ diff) / np.maximum(W @ valid, 1)
                row[metric] = {"value": agg[metric], "delta": float(np.nansum(diff) / max(valid.sum(), 1)),
                               "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
            results["runs"][f"{pipe}|{name}"] = row
    # Paired pipeline comparison of the degradation: (ours - ours@s0) - (imuhoi - imuhoi@s0).
    for name in sorted(set(SHIFT_RUN.values()) | set(FLIP_RUN.values())):
        if name == "s0":
            continue
        entry = {}
        for metric, rec_key, _ in METRICS:
            d = {}
            for pipe in PIPES:
                base = per_sequence(runs[(pipe, "s0")][1], keys, rec_key)
                d[pipe] = per_sequence(runs[(pipe, name)][1], keys, rec_key) - base
            gap = d["ours"] - d["imuhoi3b"]
            valid = (~np.isnan(gap)).astype(float)
            gap = np.nan_to_num(gap)
            draws = (W @ gap) / np.maximum(W @ valid, 1)
            entry[metric] = [float(gap.sum() / max(valid.sum(), 1)), float(np.percentile(draws, 2.5)),
                             float(np.percentile(draws, 97.5))]
        results["pairs"][name] = entry
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    plot(results, out_dir / "contact_sensitivity.png", ds.upper())
    write_summary(results, out_dir / "summary.md", ds.upper())


def plot(results, path, label):
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), facecolor=A.SURFACE)
    for row, (metric, _, name) in enumerate(METRICS):
        for col, (xs, run_of, xlabel) in enumerate(
                ((SHIFTS, SHIFT_RUN, "GT contact shifted by K frames (positive = late)"),
                 (FLIPS, FLIP_RUN, "fraction of GT contact frames flipped (3-10 frame runs)"))):
            ax = axes[row, col]
            A.style_axes(ax)
            ax.axhline(0, color=A.INK2, linewidth=0.8)
            lows, highs = [0.0], [0.0]
            for pipe in PIPES:
                rows = [results["runs"][f"{pipe}|{run_of[x]}"][metric] for x in xs]
                d = np.array([r["delta"] for r in rows])
                lo = np.array([r["ci"][0] for r in rows])
                hi = np.array([r["ci"][1] for r in rows])
                ax.fill_between(xs, lo, hi, color=COLORS[pipe], alpha=0.12, linewidth=0)
                ax.plot(xs, d, color=COLORS[pipe], linewidth=2, marker="o", markersize=4, label=PIPES[pipe])
                lows.append(lo.min())
                highs.append(hi.max())
            y0, y1 = min(lows), max(highs)
            pad = 0.08 * (y1 - y0 if y1 > y0 else 1.0)
            y0, y1 = y0 - pad, y1 + pad
            # Real-VC reference lines; one far outside the oracle curves is marked at the top edge instead.
            marks = []
            for pipe in PIPES:
                gap = results["real"][pipe][metric] - results["runs"][f"{pipe}|s0"][metric]["value"]
                inside = y0 - 2 * pad <= gap <= y1 + 4 * pad
                if inside:
                    y0, y1 = min(y0, gap - pad), max(y1, gap + pad)
                marks.append([gap, inside, pipe])
            ax.set_ylim(y0, y1)
            text_y = []
            for gap, inside, pipe in sorted(marks, key=lambda m: m[0]):
                if inside:
                    ax.axhline(gap, color=COLORS[pipe], linewidth=1.2, linestyle=(0, (4, 2)))
                    y = gap
                    text = f"real VC {gap:+.1f}"
                else:
                    y = y1 - pad
                    text = f"real VC {gap:+.1f} (off scale)"
                    ax.plot([xs[-1]], [y1], marker="^", color=COLORS[pipe], markersize=6, clip_on=False)
                for prev in text_y:
                    if abs(y - prev) < 0.07 * (y1 - y0):
                        y = prev + 0.07 * (y1 - y0)
                text_y.append(y)
                ax.annotate(text, (xs[-1], y), xytext=(6, 0), textcoords="offset points", fontsize=6.5,
                            color=A.INK, va="center", annotation_clip=False)
            ax.set_xlabel(xlabel, fontsize=7.5, color=A.INK)
            if col == 0:
                ax.set_ylabel(f"{name}: increase over exact GT contact (cm)", fontsize=7.5, color=A.INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{label}: object branch under corrupted contact (dashed = each pipeline's real VC)",
                 fontsize=9, color=A.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 0.88, 0.96))
    fig.savefig(path, dpi=180, facecolor=A.SURFACE)
    plt.close(fig)


def write_summary(results, path, label):
    lines = [f"# A3 contact-error sensitivity: {label} test", "",
             f"{results['sequences']} sequences, 7/3 protocol. Rows: GT contact (hands + object motion) shifted by K "
             "frames or with flicker; delta = error minus the error with exact GT contact (s0), sequence-bootstrap 95% CI.",
             "", "| pipeline | contact | published F1 L / R / obj | Object (cm) | delta Object | HOI (cm) | delta HOI |",
             "|---|---|---|---|---|---|---|"]
    order = ["s0"] + [SHIFT_RUN[k] for k in SHIFTS if k != 0] + [FLIP_RUN[f] for f in FLIPS if f > 0]
    for pipe in PIPES:
        real = results["real"][pipe]
        lines.append(f"| {PIPES[pipe]} | real VC | - | {real['obj_trans_err_fusion_refine']:.2f} | "
                     f"{real['obj_trans_err_fusion_refine'] - results['runs'][f'{pipe}|s0']['obj_trans_err_fusion_refine']['value']:+.2f} | "
                     f"{real['hoi_err_fusion']:.2f} | "
                     f"{real['hoi_err_fusion'] - results['runs'][f'{pipe}|s0']['hoi_err_fusion']['value']:+.2f} |")
        for name in order:
            r = results["runs"][f"{pipe}|{name}"]
            o, h = r["obj_trans_err_fusion_refine"], r["hoi_err_fusion"]
            f1 = " / ".join(f"{v:.3f}" for v in r["contact_f1"])
            cells = [f"{o['value']:.2f}", "-" if name == "s0" else f"{o['delta']:+.2f} [{o['ci'][0]:+.2f}, {o['ci'][1]:+.2f}]",
                     f"{h['value']:.2f}", "-" if name == "s0" else f"{h['delta']:+.2f} [{h['ci'][0]:+.2f}, {h['ci'][1]:+.2f}]"]
            lines.append(f"| {PIPES[pipe]} | {name} | {f1} | " + " | ".join(cells) + " |")
    lines += ["", "Real VC expressed as an equivalent flicker level (linear interpolation of the flicker curve; rough):", "",
              "| pipeline | Object | HOI |", "|---|---|---|"]
    for pipe in PIPES:
        cells = []
        for metric, _, _ in METRICS:
            gap = results["real"][pipe][metric] - results["runs"][f"{pipe}|s0"][metric]["value"]
            curve = [0.0] + [results["runs"][f"{pipe}|{FLIP_RUN[f]}"][metric]["delta"] for f in FLIPS[1:]]
            if gap <= 0:
                cells.append("none (real VC at or below exact GT contact)")
            elif all(np.diff(curve) > 0) and gap <= curve[-1]:
                cells.append(f"about {np.interp(gap, curve, FLIPS):.3f}")
            else:
                cells.append("outside the curve")
        lines.append(f"| {PIPES[pipe]} | " + " | ".join(cells) + " |")
    lines += ["", "## Degradation of IMUHOI++ minus degradation of IMUHOI (paired, cm)", "",
              "| contact | Object | HOI |", "|---|---|---|"]
    for name in order[1:]:
        p = results["pairs"][name]
        lines.append(f"| {name} | " + " | ".join(
            f"{p[m][0]:+.2f} [{p[m][1]:+.2f}, {p[m][2]:+.2f}]" for m, _, _ in METRICS) + " |")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
