#!/usr/bin/env python3
"""A2 analysis: body-part knockouts of the IMUHOI++ VC and integrated-gradients attribution of its decisions.

Knockouts (from dump_vc_knockout.py): per hand, frame-level contact F1 and event F1 (tolerance 10, threshold 0.5,
sliding-window events as in A1) of every knockout against the reference run, paired sequence bootstrap; object
and HOI errors from the eval JSONs.

Attribution: integrated gradients of the published hand-contact probability of every window that publishes a GT
hand onset / offset, w.r.t. the five body-group encoder inputs (baseline = their means). Per window the |IG| mass
is normalized; windows where the whole pose pathway changes the probability by less than 0.05, or where the IG sum
misses that change by more than 10 % (path integral not converged), are left out.

    python scripts/vc_probe/analyze_knockout.py --dataset behave
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_contact_events as A  # noqa: E402

RUN_ROOT = A.RUNS / "vc_knockout_20261028"
CONDITIONS = ("ref", "left_arm", "right_arm", "torso", "legs", "pose_prior", "hp_all")
LABELS = {"ref": "reference", "left_arm": "left arm", "right_arm": "right arm", "torso": "torso",
          "legs": "both legs", "pose_prior": "all groups (b_t)", "hp_all": "all HP inputs"}
GROUPS = ("left_arm", "right_arm", "torso", "left_leg", "right_leg")
HANDS = ("left", "right")
TOL = 10
MIN_DP = 0.05
MAX_REL_GAP = 0.10  # IG convergence: |sum IG - dp| / |dp|
PUBLISH = A.WARMUP - 1
HAND_COLORS = {"left": "#2a78d6", "right": "#eb6834"}
MODE_COLORS = {"lift": "#2a78d6", "ground": "#eb6834", "tilt": "#1baf7a"}


def load_probs(ds, files):
    out = {}
    for cond in CONDITIONS:
        directory = RUN_ROOT / f"{ds}_{cond}" / "probs"
        out[cond] = {f: torch.load(directory / f"{f}.pt", map_location="cpu", weights_only=False)
                     ["pred_hand_contact_prob"].float().numpy() for f in files}
    return out


def load_gt(ds, files):
    gt = {}
    for f in files:
        d = torch.load(A.DATA[ds] / f"{f}.pt", map_location="cpu", weights_only=False)
        T_pub = int(d["obj_trans"].shape[0]) - A.LOOKAHEAD
        gt[f] = np.stack([d[f"{c}_contact"].numpy().astype(bool)[:T_pub] for c in A.CHANNELS], axis=1)
    return gt


def per_sequence_counts(files, gt, probs, channel, kind):
    """[n_seq, 3] TP / FP / FN of events (kind) or frames (kind == 'frame') for one channel."""
    counts = np.zeros((len(files), 3))
    for i, f in enumerate(files):
        g, p = gt[f][:, channel], probs[f][:, channel] >= 0.5
        if kind == "frame":
            counts[i] = (np.sum(g & p), np.sum(~g & p), np.sum(g & ~p))
            continue
        T_pub = g.shape[0]
        g_ev = A.in_region(A.events(g, kind), T_pub)
        p_ev = A.in_region(A.events(p, kind), T_pub)
        tp, _ = A.match(g_ev, p_ev, TOL)
        counts[i] = (tp, len(p_ev) - tp, len(g_ev) - tp)
    return counts


def knockout_table(files, gt, probs, W):
    table, draws = {}, {}
    for kind in ("frame", "onset", "offset"):
        for channel, hand in enumerate(HANDS):
            ref = per_sequence_counts(files, gt, probs["ref"], channel, kind)
            ref_draw = A.f1_from(W @ ref)
            for cond in CONDITIONS:
                counts = ref if cond == "ref" else per_sequence_counts(files, gt, probs[cond], channel, kind)
                f1 = float(A.f1_from(counts.sum(0)))
                diff = A.f1_from(W @ counts) - ref_draw
                table[(kind, hand, cond)] = {"f1": f1, "delta": float(f1 - A.f1_from(ref.sum(0))),
                                             "ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
                draws[(kind, hand, cond)] = diff
    # Laterality of the arm knockouts: change of the ipsilateral hand minus change of the contralateral hand.
    for kind in ("frame", "onset", "offset"):
        for cond, ipsi, contra in (("left_arm", "left", "right"), ("right_arm", "right", "left")):
            d = draws[(kind, ipsi, cond)] - draws[(kind, contra, cond)]
            point = table[(kind, ipsi, cond)]["delta"] - table[(kind, contra, cond)]["delta"]
            table[(kind, "ipsi_minus_contra", cond)] = {"f1": float("nan"), "delta": float(point),
                                                        "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return table


def ig_records(ds, labels):
    records = torch.load(RUN_ROOT / f"{ds}_ig.pt", map_location="cpu", weights_only=False)
    by_file = collections.defaultdict(list)
    for s in labels:
        by_file[s["file"]].append(s)
    out = []
    for r in records:
        ig = r["ig"].numpy()
        mass = np.abs(ig)
        total = mass.sum()
        mode = next((s["mode"] for s in by_file[r["file"]] if s["a"] <= r["target"] <= s["b"]), "none")
        dp = r["p_full"] - r["p_base"]
        out.append({**{k: r[k] for k in ("file", "target", "hand", "kind")}, "mode": mode,
                    "dp": dp, "p": r["p_full"], "p_base": r["p_base"],
                    "rel_gap": abs(float(ig.sum()) - dp) / max(abs(dp), 1e-9),
                    "share": mass.sum(1) / max(total, 1e-12), "profile": mass.sum(0) / max(total, 1e-12),
                    "signed": ig.sum(1)})
    return out


def boot_mean(values, seq_ids, n_boot=1000, seed=0):
    """Mean of per-window values with a sequence bootstrap CI."""
    values = np.asarray(values, dtype=float)
    uniq, inv = np.unique(seq_ids, return_inverse=True)
    sums = np.zeros((len(uniq),) + values.shape[1:])
    np.add.at(sums, inv, values)
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    W = A.boot_weights(len(uniq), seed=seed)[:n_boot]
    draws = (W @ sums.reshape(len(uniq), -1)) / np.maximum(W @ counts, 1)[:, None]
    draws = draws.reshape((n_boot,) + values.shape[1:])
    return values.mean(0), np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def kept_windows(records):
    return [r for r in records if abs(r["dp"]) >= MIN_DP and r["rel_gap"] <= MAX_REL_GAP]


def ig_summary(records):
    kept = kept_windows(records)
    summary = {"windows": len(records), "kept": len(kept), "groups": {}, "laterality": {}, "before": {}, "modes": {},
               "small_dp": sum(abs(r["dp"]) < MIN_DP for r in records),
               "not_converged": sum(abs(r["dp"]) >= MIN_DP and r["rel_gap"] > MAX_REL_GAP for r in records)}
    rel = np.arange(A.WARMUP + A.LOOKAHEAD) - PUBLISH
    for kind in ("onset", "offset"):
        for h, hand in enumerate(HANDS):
            sel = [r for r in kept if r["kind"] == kind and r["hand"] == h]
            if not sel:
                continue
            ids = [r["file"] for r in sel]
            mean, lo, hi = boot_mean([r["share"] for r in sel], ids)
            summary["groups"][f"{hand}_{kind}"] = {"n": len(sel), "share": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                                                   "dp_mean": float(np.mean([r["dp"] for r in sel]))}
            ipsi, contra = (0, 1) if hand == "left" else (1, 0)
            lat = [r["share"][ipsi] - r["share"][contra] for r in sel]
            m, l, u = boot_mean(lat, ids)
            summary["laterality"][f"{hand}_{kind}"] = [float(m), float(l), float(u)]
        sel = [r for r in kept if r["kind"] == kind]
        if sel:
            ids = [r["file"] for r in sel]
            before = [r["profile"][rel < 0].sum() for r in sel]
            far = [r["profile"][rel < -A.LOOKAHEAD].sum() for r in sel]
            mb, lb, ub = boot_mean(before, ids)
            mf, lf, uf = boot_mean(far, ids)
            prof, plo, phi = boot_mean([r["profile"] for r in sel], ids)
            summary["before"][kind] = {"before_event": [float(mb), float(lb), float(ub)],
                                       "before_minus10": [float(mf), float(lf), float(uf)],
                                       "uniform_before_event": float(np.mean(rel < 0)),
                                       "profile": prof.tolist(), "profile_lo": plo.tolist(), "profile_hi": phi.tolist()}
    for mode in ("lift", "ground", "tilt"):
        sel = [r for r in kept if r["mode"] == mode]
        if len(sel) >= 10:
            mean, lo, hi = boot_mean([r["share"] for r in sel], [r["file"] for r in sel])
            summary["modes"][mode] = {"n": len(sel), "share": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    return summary


def style_axes(ax, grid_axis="y"):
    A.style_axes(ax)
    if grid_axis == "x":
        ax.grid(False)
        ax.grid(axis="x", color=A.GRID, linewidth=0.6)


def plot_knockouts(table, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), facecolor=A.SURFACE, sharey=True)
    conds = [c for c in CONDITIONS if c != "ref"]
    y = np.arange(len(conds))
    for ax, kind in zip(axes, ("frame", "onset", "offset")):
        style_axes(ax, "x")
        ax.axvline(0, color=A.INK2, linewidth=0.8)
        for k, hand in enumerate(HANDS):
            offs = (k - 0.5) * 0.28
            d = np.array([table[(kind, hand, c)]["delta"] for c in conds])
            lo = np.array([table[(kind, hand, c)]["ci"][0] for c in conds])
            hi = np.array([table[(kind, hand, c)]["ci"][1] for c in conds])
            ax.errorbar(d, y + offs, xerr=[d - lo, hi - d], fmt="o", markersize=5, color=HAND_COLORS[hand],
                        ecolor=HAND_COLORS[hand], elinewidth=1.2, capsize=0, label=f"{hand} hand")
        name = {"frame": "frame-level F1", "onset": f"onset event F1@{TOL}", "offset": f"offset event F1@{TOL}"}[kind]
        ax.set_title(f"change in {name}", fontsize=8.5, color=A.INK, loc="left")
        ax.set_xlabel("knockout minus reference", fontsize=7.5, color=A.INK)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[c] for c in conds], fontsize=7.5, color=A.INK)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=9, color=A.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path, dpi=180, facecolor=A.SURFACE)
    plt.close(fig)


def plot_ig_heatmap(records, path, title):
    kept = kept_windows(records)
    rel = np.arange(A.WARMUP + A.LOOKAHEAD) - PUBLISH
    panels = [(kind, h) for kind in ("onset", "offset") for h in range(2)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.4, 6.4), facecolor=A.SURFACE, sharex=True)
    vmax = 0.0
    grids = []
    for kind, h in panels:
        sel = [r for r in kept if r["kind"] == kind and r["hand"] == h]
        grid = np.mean([r["_mass"] for r in sel], axis=0) if sel else np.zeros((len(GROUPS), rel.size))
        grids.append((grid, len(sel)))
        vmax = max(vmax, grid.max())
    for ax, (kind, h), (grid, n) in zip(axes, panels, grids):
        ax.set_facecolor(A.SURFACE)
        im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
                       extent=(rel[0] - 0.5, rel[-1] + 0.5, len(GROUPS) - 0.5, -0.5))
        ax.axvline(0, color=A.INK, linewidth=0.8)
        ax.axvline(-A.LOOKAHEAD, color=A.INK2, linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels([g.replace("_", " ") for g in GROUPS], fontsize=7, color=A.INK)
        ax.tick_params(colors=A.INK2, labelsize=7)
        ax.set_title(f"{HANDS[h]}-hand {kind}s (n = {n})", fontsize=8, color=A.INK, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("window frame relative to the GT event (0 = published frame = event; window sees up to +10)",
                        fontsize=7.5, color=A.INK)
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.01)
    cbar.ax.tick_params(labelsize=7, colors=A.INK2)
    cbar.set_label("mean share of |IG| per frame and group", fontsize=7, color=A.INK)
    fig.suptitle(title, fontsize=9, color=A.INK, x=0.01, ha="left")
    fig.savefig(path, dpi=180, facecolor=A.SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_modes(summary, path, title):
    modes = [m for m in ("lift", "ground", "tilt") if m in summary["modes"]]
    if not modes:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.2), facecolor=A.SURFACE)
    style_axes(ax)
    x = np.arange(len(GROUPS))
    width = 0.8 / len(modes)
    for k, mode in enumerate(modes):
        s = summary["modes"][mode]
        share, lo, hi = np.array(s["share"]), np.array(s["lo"]), np.array(s["hi"])
        pos = x + (k - (len(modes) - 1) / 2) * width
        ax.bar(pos, share, width * 0.9, color=MODE_COLORS[mode], edgecolor=A.SURFACE, linewidth=1,
               label=f"{'push/drag' if mode == 'ground' else mode} (n = {s['n']})")
        ax.errorbar(pos, share, yerr=[share - lo, hi - share], fmt="none", ecolor=A.INK2, elinewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("_", " ") for g in GROUPS], fontsize=7.5, color=A.INK)
    ax.set_ylabel("share of |IG|", fontsize=7.5, color=A.INK)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title(title, fontsize=8.5, color=A.INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=A.SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("omomo", "behave"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ds = args.dataset
    out_dir = args.out or RUN_ROOT / f"analysis_{ds}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p.stem for p in (RUN_ROOT / f"{ds}_ref" / "probs").glob("*.pt"))
    gt = load_gt(ds, files)
    probs = load_probs(ds, files)
    W = A.boot_weights(len(files))
    table = knockout_table(files, gt, probs, W)
    evals = {c: json.loads((RUN_ROOT / f"{ds}_{c}" / "eval.json").read_text())["results"][ds] for c in CONDITIONS}

    labels = json.loads((A.PILOT_ROOT / f"labels_{ds}_test.json").read_text())
    records = ig_records(ds, labels)
    for r, raw in zip(records, torch.load(RUN_ROOT / f"{ds}_ig.pt", map_location="cpu", weights_only=False)):
        mass = np.abs(raw["ig"].numpy())
        r["_mass"] = mass / max(mass.sum(), 1e-12)
    summary = ig_summary(records)

    label = ds.upper()
    plot_knockouts(table, out_dir / "knockout_effects.png",
                   f"{label}: body-part knockouts of the IMUHOI++ VC (mean replacement, 95% CI)")
    plot_ig_heatmap(records, out_dir / "ig_heatmap.png",
                    f"{label}: integrated gradients of the published hand-contact probability at GT events")
    plot_modes(summary, out_dir / "ig_group_share_by_mode.png",
               f"{label}: share of |IG| per body group by interaction mode (onsets and offsets)")

    results = {"dataset": ds, "sequences": len(files),
               "knockout": {f"{k}|{h}|{c}": v for (k, h, c), v in table.items()},
               "eval": {c: {k: evals[c][k] for k in ("contact_f1_lhand", "contact_f1_rhand", "contact_f1_obj",
                                                      "obj_trans_err_fusion_refine", "hoi_err_fusion")} for c in CONDITIONS},
               "ig": summary}
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    write_summary(results, table, evals, summary, out_dir / "summary.md", label)


def write_summary(results, table, evals, summary, path, label):
    lines = [f"# A2 body-part knockouts and attribution: {label} test", "",
             f"{results['sequences']} sequences, IMUHOI++ pipeline (VC-v5ar + OT v2_armrot_biclose), 7/3 protocol.",
             "Knockout = input replaced by its mean over all VC forward calls of the reference run. Event F1: tolerance",
             f"{TOL} frames, threshold 0.5, sliding-window events only (as A1). Deltas: knockout minus reference with a",
             "paired sequence-bootstrap 95% CI.", "",
             "## Knockouts: contact F1 per hand", "",
             "| knockout | frame F1 L | frame F1 R | onset F1 L | onset F1 R | offset F1 L | offset F1 R |",
             "|---|---|---|---|---|---|---|"]
    for cond in CONDITIONS:
        cells = []
        for kind in ("frame", "onset", "offset"):
            for hand in HANDS:
                r = table[(kind, hand, cond)]
                cells.append(f"{r['f1']:.3f}" if cond == "ref" else
                             f"{r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
        lines.append(f"| {LABELS[cond]} | " + " | ".join(cells) + " |")
    lines += ["", "Arm knockouts, change of the ipsilateral hand minus change of the contralateral hand:", "",
              "| knockout | frame F1 | onset F1 | offset F1 |", "|---|---|---|---|"]
    for cond in ("left_arm", "right_arm"):
        cells = [table[(kind, "ipsi_minus_contra", cond)] for kind in ("frame", "onset", "offset")]
        lines.append(f"| {LABELS[cond]} | " + " | ".join(
            f"{r['delta']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]" for r in cells) + " |")
    lines += ["", "## Knockouts: object and HOI error (eval JSON)", "",
              "| knockout | Object (cm) | HOI (cm) | F1 obj motion |", "|---|---|---|---|"]
    ref = evals["ref"]
    for cond in CONDITIONS:
        e = evals[cond]
        if cond == "ref":
            lines.append(f"| reference | {e['obj_trans_err_fusion_refine']:.2f} | {e['hoi_err_fusion']:.2f} | {e['contact_f1_obj']:.3f} |")
        else:
            lines.append(f"| {LABELS[cond]} | {e['obj_trans_err_fusion_refine'] - ref['obj_trans_err_fusion_refine']:+.2f} | "
                         f"{e['hoi_err_fusion'] - ref['hoi_err_fusion']:+.2f} | {e['contact_f1_obj'] - ref['contact_f1_obj']:+.3f} |")
    lines += ["", "## Integrated gradients", "",
              f"{summary['windows']} windows at GT hand events; {summary['kept']} used below. Left out: "
              f"{summary['small_dp']} where the whole pose pathway changes the published probability by < {MIN_DP}, "
              f"{summary['not_converged']} where the IG sum misses that change by > {MAX_REL_GAP:.0%}.", "",
              "| events | n | mean dp (pose vs mean pose) | " + " | ".join(g.replace('_', ' ') for g in GROUPS)
              + " | ipsilateral minus contralateral arm |", "|---|---|---|" + "---|" * (len(GROUPS) + 1)]
    for key, s in summary["groups"].items():
        lat = summary["laterality"][key]
        lines.append(f"| {key.replace('_', ' ')} | {s['n']} | {s['dp_mean']:+.3f} | "
                     + " | ".join(f"{v:.3f}" for v in s["share"]) + f" | {lat[0]:+.3f} [{lat[1]:+.3f}, {lat[2]:+.3f}] |")
    lines += ["", "| events | |IG| share before the event frame | before -10 (window cannot see the event) | uniform share before the event |",
              "|---|---|---|---|"]
    for kind, b in summary["before"].items():
        lines.append(f"| {kind} | {b['before_event'][0]:.3f} [{b['before_event'][1]:.3f}, {b['before_event'][2]:.3f}] | "
                     f"{b['before_minus10'][0]:.3f} [{b['before_minus10'][1]:.3f}, {b['before_minus10'][2]:.3f}] | "
                     f"{b['uniform_before_event']:.3f} |")
    if summary["modes"]:
        lines += ["", "| mode | n | " + " | ".join(g.replace('_', ' ') for g in GROUPS) + " |",
                  "|---|---|" + "---|" * len(GROUPS)]
        for mode, s in summary["modes"].items():
            lines.append(f"| {mode} | {s['n']} | " + " | ".join(
                f"{v:.3f} [{lo:.3f}, {hi:.3f}]" for v, lo, hi in zip(s["share"], s["lo"], s["hi"])) + " |")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
