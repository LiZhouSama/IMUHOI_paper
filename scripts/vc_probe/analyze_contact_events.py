#!/usr/bin/env python3
"""A1: event-aligned contact response curves and event-level timing of the VC contact decisions.

Every GT hand-contact and object-motion segment of the test split gives an onset event (its first frame) and an
offset event (its last frame, the convention of the endpoint labels).  The published per-frame probabilities of
each model are aligned at these events (peri-event curves), binarized (threshold 0.5, and the 0.7/0.3 hysteresis
of the OT's FK tracking filter) and matched one-to-one to the GT events within a tolerance.

Only events whose +-30-frame neighbourhood lies on sliding-window frames are scored: frames [0, 60) are published
from the warmup pass, which sees inputs up to frame 69.  Predicted events are restricted to the same region.
95 % CIs bootstrap over sequences; differences between models use the same resamples (paired).

    python scripts/vc_probe/analyze_contact_events.py --dataset behave
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/IMUHOI++/RNN_cano"
EVENT_ROOT = RUNS / "vc_event_timing_20261028"
PILOT_ROOT = RUNS / "vc_probe_pilot_20261027"
DATA = {"omomo": ROOT / "process/IMUHOI++cano/processed_split_data_OMOMO_full/test",
        "behave": ROOT / "process/IMUHOI++cano/processed_split_data_BEHAVE/test"}

LOOKAHEAD, WARMUP, HALF = 10, 60, 30
TOLERANCES = (3, 5, 10, 15, 20, 30)
N_BOOT = 1000
CHANNELS = ("lhand", "rhand", "obj")
KINDS = (("hand", "onset"), ("hand", "offset"), ("obj", "onset"), ("obj", "offset"))

# model -> (label, directory pattern)
MODELS = {
    "v5ar": ("IMUHOI++ VC (v5ar) + our OT", EVENT_ROOT / "{ds}_v5ar/probs"),
    "v4": ("pose-prior VC (v4) + our OT", EVENT_ROOT / "{ds}_v4/probs"),
    "imuhoivc": ("IMUHOI VC + our OT", EVENT_ROOT / "{ds}_imuhoivc/probs"),
    "imuhoi3b": ("IMUHOI, original 3-branch", PILOT_ROOT / "{ds}_imuhoi/features"),
}
PAIRS = (("v5ar", "imuhoi3b"), ("v5ar", "imuhoivc"), ("v4", "imuhoivc"), ("v5ar", "v4"))
EVAL_JSON = {m: EVENT_ROOT / ("{ds}_%s/eval.json" % m) for m in ("v5ar", "v4", "imuhoivc")}
EVAL_JSON["imuhoi3b"] = PILOT_ROOT / "{ds}_imuhoi/eval.json"

# Reference palette (dataviz skill): 4 slots validated as adjacent pairs on #fcfcfb.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"v5ar": "#2a78d6", "v4": "#eb6834", "imuhoivc": "#1baf7a", "imuhoi3b": "#eda100"}


def segments(binary):
    """[start, end] (inclusive) of the runs of True."""
    padded = np.concatenate([[False], binary, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return list(zip(starts, ends))


def hysteresis(prob, on=0.7, off=0.3):
    state, out = False, np.zeros(prob.shape[0], dtype=bool)
    for t, value in enumerate(prob):
        state = value >= on if not state else value >= off
        out[t] = state
    return out


def events(binary, kind):
    segs = segments(binary)
    return np.array([s if kind == "onset" else e for s, e in segs], dtype=int)


def load(ds):
    files = sorted(p.stem for p in Path(str(MODELS["v5ar"][1]).format(ds=ds)).glob("*.pt"))
    gt, probs, endpoint = {}, {m: {} for m in MODELS}, {m: {} for m in MODELS}
    for f in files:
        d = torch.load(DATA[ds] / f"{f}.pt", map_location="cpu", weights_only=False)
        T_pub = int(d["obj_trans"].shape[0]) - LOOKAHEAD
        gt[f] = np.stack([d[f"{c}_contact"].numpy().astype(bool)[:T_pub] for c in CHANNELS], axis=1)
        for m, (_, pattern) in MODELS.items():
            r = torch.load(Path(str(pattern).format(ds=ds)) / f"{f}.pt", map_location="cpu", weights_only=False)
            p = r["pred_hand_contact_prob"].float().numpy()
            if p.shape[0] != T_pub:
                raise ValueError(f"{m}/{f}: {p.shape[0]} published frames, expected {T_pub}")
            probs[m][f] = p
            if "pred_contact_endpoint_prob" in r:
                endpoint[m][f] = r["pred_contact_endpoint_prob"].float().numpy()
    return files, gt, probs, endpoint


def in_region(frames, T_pub):
    return frames[(frames >= WARMUP + HALF) & (frames < T_pub - HALF)]


def channel_ids(group):
    return (0, 1) if group == "hand" else (2,)


def peri_event(files, gt, series, group, kind):
    """Rows of series[e-30 : e+31] for every GT event in the region; returns rows, sequence index per row."""
    rows, seq_idx = [], []
    for i, f in enumerate(files):
        T_pub = gt[f].shape[0]
        for c in channel_ids(group):
            for e in in_region(events(gt[f][:, c], kind), T_pub):
                rows.append(series(f, c, kind)[e - HALF:e + HALF + 1])
                seq_idx.append(i)
    return np.asarray(rows, dtype=np.float32), np.asarray(seq_idx)


def boot_weights(n_seq, seed=0):
    """[N_BOOT, n_seq] multiplicity of every sequence in each resample (sequence bootstrap)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_seq, size=(N_BOOT, n_seq))
    return np.stack([np.bincount(row, minlength=n_seq) for row in idx]).astype(np.float64)


def weighted_median_draws(values, seq_ids, W):
    if values.size == 0:
        return np.full(W.shape[0], np.nan)
    order = np.argsort(values)
    v, s = values[order], seq_ids[order]
    cw = np.cumsum(W[:, s], axis=1)
    pos = (cw >= cw[:, -1:] / 2.0).argmax(axis=1)
    return v[pos]


def curve_ci(rows, seq_idx, n_seq, W):
    per_seq_sum = np.zeros((n_seq, rows.shape[1]))
    per_seq_n = np.zeros(n_seq)
    np.add.at(per_seq_sum, seq_idx, rows)
    np.add.at(per_seq_n, seq_idx, 1)
    draws = (W @ per_seq_sum) / np.maximum(W @ per_seq_n, 1)[:, None]
    return rows.mean(0), np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def crossing(curve, kind, level=0.5):
    """First offset (interpolated) where the mean curve crosses ``level`` in the event's direction."""
    ks = np.arange(-HALF, HALF + 1)
    above = curve >= level
    for i in range(1, len(curve)):
        if (kind == "onset" and above[i] and not above[i - 1]) or (kind == "offset" and not above[i] and above[i - 1]):
            frac = (level - curve[i - 1]) / (curve[i] - curve[i - 1])
            return float(ks[i - 1] + frac)
    return float("nan")


def match(gt_frames, pred_frames, tol):
    pairs = sorted((abs(int(p) - int(g)), gi, pi) for gi, g in enumerate(gt_frames)
                   for pi, p in enumerate(pred_frames) if abs(int(p) - int(g)) <= tol)
    used_g, used_p, deltas = set(), set(), []
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        deltas.append(int(pred_frames[pi]) - int(gt_frames[gi]))
    return len(deltas), deltas


def event_stats(files, gt, probs_m, group, kind, rule):
    """Per sequence: TP/FP/FN per tolerance, matched deltas (tol 30), predicted and GT segment counts."""
    counts = np.zeros((len(files), len(TOLERANCES), 3))
    deltas = [[] for _ in files]
    nseg = np.zeros((len(files), 2))
    for i, f in enumerate(files):
        T_pub = gt[f].shape[0]
        for c in channel_ids(group):
            pred_bin = probs_m[f][:, c] >= 0.5 if rule == "thr" else hysteresis(probs_m[f][:, c])
            g_ev = in_region(events(gt[f][:, c], kind), T_pub)
            p_ev = in_region(events(pred_bin, kind), T_pub)
            nseg[i] += (len(p_ev), len(g_ev))
            for j, tol in enumerate(TOLERANCES):
                tp, dl = match(g_ev, p_ev, tol)
                counts[i, j] += (tp, len(p_ev) - tp, len(g_ev) - tp)
                if tol == 30:
                    deltas[i].extend(dl)
    return counts, deltas, nseg


def f1_from(counts):
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def summarize_events(stats, W):
    """Point estimates and bootstrap draws of F1 per tolerance, median |delta|, median delta, over-seg ratio."""
    counts, deltas, nseg = stats
    out = {"f1": f1_from(counts.sum(0)), "n_gt": int(nseg[:, 1].sum()), "n_pred": int(nseg[:, 0].sum())}
    flat = np.asarray([d for seq in deltas for d in seq], dtype=float)
    seq_ids = np.asarray([i for i, seq in enumerate(deltas) for _ in seq], dtype=int)
    out["median_abs_delta"] = float(np.median(np.abs(flat))) if flat.size else float("nan")
    out["mean_abs_delta"] = float(np.mean(np.abs(flat))) if flat.size else float("nan")
    out["median_delta"] = float(np.median(flat)) if flat.size else float("nan")
    out["within5"] = float(np.mean(np.abs(flat) <= 5)) if flat.size else float("nan")
    out["overseg"] = float(nseg[:, 0].sum() / max(nseg[:, 1].sum(), 1))
    boot_nseg = W @ nseg
    abs_sum = np.zeros(len(deltas))
    abs_n = np.zeros(len(deltas))
    np.add.at(abs_sum, seq_ids, np.abs(flat))
    np.add.at(abs_n, seq_ids, 1)
    draws = {"f1": f1_from(np.einsum("bs,stk->btk", W, counts)),
             "mean_abs_delta": (W @ abs_sum) / np.maximum(W @ abs_n, 1),
             "overseg": boot_nseg[:, 0] / np.maximum(boot_nseg[:, 1], 1),
             "median_abs_delta": weighted_median_draws(np.abs(flat), seq_ids, W)}
    return out, draws


def ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def event_axis(ax, kind):
    ax.axvspan(-LOOKAHEAD, 0, color=GRID, alpha=0.6, linewidth=0)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlim(-HALF, HALF)
    ax.set_xlabel(f"frames from GT {kind} (30 fps)", fontsize=7.5, color=INK)


def plot_curves(curves, counts, path, title, ylabel_map):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0), facecolor=SURFACE, sharex=True)
    ks = np.arange(-HALF, HALF + 1)
    for ax, (group, kind) in zip(axes.flat, KINDS):
        style_axes(ax)
        event_axis(ax, kind)
        gt_curve = curves[(group, kind)]["gt"]
        ax.plot(ks, gt_curve, color=INK2, linewidth=1.2, linestyle=(0, (4, 2)), label="GT label")
        for m in MODELS:
            mean, lo, hi = curves[(group, kind)][m]
            ax.fill_between(ks, lo, hi, color=COLORS[m], alpha=0.10, linewidth=0)
            ax.plot(ks, mean, color=COLORS[m], linewidth=2, label=MODELS[m][0])
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(ylabel_map[group], fontsize=7.5, color=INK)
        ax.set_title(f"{'hand contact' if group == 'hand' else 'object motion'} {kind}  (n = {counts[(group, kind)]})",
                     fontsize=8.5, color=INK, loc="left")
    axes[0, 0].text(-LOOKAHEAD / 2, 1.0, "window\nsees event", ha="center", va="top", fontsize=6.5, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_endpoint(curves, counts, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), facecolor=SURFACE, sharey=True)
    ks = np.arange(-HALF, HALF + 1)
    for ax, kind in zip(axes, ("onset", "offset")):
        style_axes(ax)
        event_axis(ax, kind)
        for m in ("v5ar", "v4"):
            mean, lo, hi = curves[kind][m]
            ax.fill_between(ks, lo, hi, color=COLORS[m], alpha=0.12, linewidth=0)
            ax.plot(ks, mean, color=COLORS[m], linewidth=2, label=MODELS[m][0])
        ax.set_title(f"hand {'start' if kind == 'onset' else 'end'} head at GT {kind} (n = {counts[kind]})",
                     fontsize=8.5, color=INK, loc="left")
    axes[0].set_ylabel("endpoint heatmap probability", fontsize=7.5, color=INK)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_f1(results, path, title):
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.9), facecolor=SURFACE, sharey=True)
    for ax, (group, kind) in zip(axes, KINDS):
        style_axes(ax)
        for m in MODELS:
            f1 = results["thr"][f"{group}_{kind}"][m]["f1"]
            ax.plot(TOLERANCES, f1, color=COLORS[m], linewidth=2, marker="o", markersize=3.5, label=MODELS[m][0])
        ax.set_xticks(TOLERANCES)
        ax.set_ylim(0, 1)
        ax.set_xlabel("tolerance (frames)", fontsize=7.5, color=INK)
        ax.set_title(f"{'hand' if group == 'hand' else 'object'} {kind}", fontsize=8.5, color=INK, loc="left")
    axes[0].set_ylabel("event F1 (threshold 0.5)", fontsize=7.5, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def mode_of_events(files, gt, labels, group, kind):
    """Mode of the union contact segment that contains each GT hand event (labels from mode_labels.py)."""
    by_file = collections.defaultdict(list)
    for s in labels:
        by_file[s["file"]].append(s)
    out = []
    for f in files:
        T_pub = gt[f].shape[0]
        for c in channel_ids(group):
            for e in in_region(events(gt[f][:, c], kind), T_pub):
                mode = next((s["mode"] for s in by_file[f] if s["a"] <= e <= s["b"]), "none")
                out.append(mode)
    return out


def per_mode(files, gt, probs, labels, kind, rule="thr", tol_index=2):
    """Hand-event F1 (tolerance TOLERANCES[tol_index]) and median |delta| per interaction mode."""
    by_file = collections.defaultdict(list)
    for s in labels:
        by_file[s["file"]].append(s)
    table = {}
    for m in MODELS:
        acc = collections.defaultdict(lambda: [0, 0, 0, []])
        for f in files:
            T_pub = gt[f].shape[0]
            for c in (0, 1):
                pred_bin = probs[m][f][:, c] >= 0.5 if rule == "thr" else hysteresis(probs[m][f][:, c])
                g_ev = in_region(events(gt[f][:, c], kind), T_pub)
                p_ev = in_region(events(pred_bin, kind), T_pub)
                for s in by_file[f]:
                    lo, hi = s["a"] - HALF, s["b"] + HALF
                    g_sel = g_ev[(g_ev >= s["a"]) & (g_ev <= s["b"])]
                    p_sel = p_ev[(p_ev >= lo) & (p_ev <= hi)]
                    if len(g_sel) == 0:
                        continue
                    tp, _ = match(g_sel, p_sel, TOLERANCES[tol_index])
                    _, dl = match(g_sel, p_sel, 30)
                    row = acc[s["mode"]]
                    row[0] += tp
                    row[1] += len(g_sel) - tp
                    row[2] += len(g_sel)
                    row[3].extend(dl)
        table[m] = {mode: {"recall": v[0] / max(v[2], 1), "n": v[2],
                           "median_abs_delta": float(np.median(np.abs(v[3]))) if v[3] else float("nan")}
                    for mode, v in acc.items()}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("omomo", "behave"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--alt", action="append", default=[], metavar="MODEL=RUN",
                    help="read MODEL from EVENT_ROOT/RUN instead (e.g. v4=omomo_v4_p00 for the 0/0 protocol)")
    args = ap.parse_args()
    for item in args.alt:
        model, run = item.split("=", 1)
        MODELS[model] = (MODELS[model][0] + f" [{run}]", EVENT_ROOT / run / "probs")
        EVAL_JSON[model] = EVENT_ROOT / run / "eval.json"
    ds = args.dataset
    out_dir = args.out or EVENT_ROOT / f"analysis_{ds}"
    out_dir.mkdir(parents=True, exist_ok=True)

    files, gt, probs, endpoint = load(ds)
    labels = json.loads((PILOT_ROOT / f"labels_{ds}_test.json").read_text())
    n_seq = len(files)
    W = boot_weights(n_seq)
    results = {"dataset": ds, "sequences": n_seq, "thr": {}, "hys": {}, "curves": {}, "endpoint": {}}

    def prob_series(m):
        return lambda f, c, kind: probs[m][f][:, c]

    curves, counts = {}, {}
    for group, kind in KINDS:
        gt_rows, seq_idx = peri_event(files, gt, lambda f, c, k: gt[f][:, c].astype(np.float32), group, kind)
        counts[(group, kind)] = len(gt_rows)
        curves[(group, kind)] = {"gt": gt_rows.mean(0) if len(gt_rows) else np.zeros(2 * HALF + 1)}
        key = f"{group}_{kind}"
        results["curves"][key] = {"n_events": len(gt_rows), "n_sequences": int(np.unique(seq_idx).size)}
        for m in MODELS:
            rows, idx = peri_event(files, gt, prob_series(m), group, kind)
            mean, lo, hi = curve_ci(rows, idx, n_seq, W)
            curves[(group, kind)][m] = (mean, lo, hi)
            results["curves"][key][m] = {"mean": mean.tolist(), "cross50": crossing(mean, kind),
                                         "p_at_0": float(mean[HALF]), "p_at_minus10": float(mean[HALF - 10]),
                                         "p_at_plus10": float(mean[HALF + 10])}
        for rule in ("thr", "hys"):
            stats = {m: event_stats(files, gt, probs[m], group, kind, rule) for m in MODELS}
            summ = {m: summarize_events(stats[m], W) for m in MODELS}
            res = {}
            for m, (point, draws) in summ.items():
                res[m] = {"f1": point["f1"].tolist(), "f1_ci_tol10": ci(draws["f1"][:, TOLERANCES.index(10)]),
                          "median_abs_delta": point["median_abs_delta"], "median_abs_delta_ci": ci(draws["median_abs_delta"]),
                          "mean_abs_delta": point["mean_abs_delta"], "mean_abs_delta_ci": ci(draws["mean_abs_delta"]),
                          "median_delta": point["median_delta"], "within5": point["within5"],
                          "overseg": point["overseg"], "overseg_ci": ci(draws["overseg"]),
                          "n_gt": point["n_gt"], "n_pred": point["n_pred"]}
            res["pairs"] = {}
            for a, b in PAIRS:
                da, db = summ[a][1], summ[b][1]
                j = TOLERANCES.index(10)
                res["pairs"][f"{a}-{b}"] = {
                    "f1_tol10": [float(summ[a][0]["f1"][j] - summ[b][0]["f1"][j])] + ci(da["f1"][:, j] - db["f1"][:, j]),
                    "mean_abs_delta": [summ[a][0]["mean_abs_delta"] - summ[b][0]["mean_abs_delta"]]
                    + ci(da["mean_abs_delta"] - db["mean_abs_delta"]),
                    "overseg": [summ[a][0]["overseg"] - summ[b][0]["overseg"]] + ci(da["overseg"] - db["overseg"]),
                }
            results[rule][key] = res
        print(f"[events] {key}: n={counts[(group, kind)]} " + ", ".join(
            f"{m} F1@10 {results['thr'][key][m]['f1'][TOLERANCES.index(10)]:.3f}" for m in MODELS), flush=True)

    # Endpoint heads of the pose-prior VCs at GT hand events.
    ep_curves, ep_counts = {}, {}
    for kind in ("onset", "offset"):
        ep_curves[kind] = {}
        for m in ("v5ar", "v4"):
            series = (lambda mm: lambda f, c, k: endpoint[mm][f][:, 2 * c + (0 if k == "onset" else 1)])(m)
            rows, idx = peri_event(files, gt, series, "hand", kind)
            ep_curves[kind][m] = curve_ci(rows, idx, n_seq, W)
            ep_counts[kind] = len(rows)
            mean = ep_curves[kind][m][0]
            results["endpoint"][f"hand_{kind}_{m}"] = {"mean": mean.tolist(), "peak_offset": int(np.argmax(mean) - HALF),
                                                       "peak": float(mean.max())}
    results["per_mode"] = {kind: per_mode(files, gt, probs, labels, kind) for kind in ("onset", "offset")}
    eval_f1 = {}
    for m in MODELS:
        r = json.loads(Path(str(EVAL_JSON[m]).format(ds=ds)).read_text())["results"][ds]
        eval_f1[m] = {k: r[k] for k in ("contact_f1_lhand", "contact_f1_rhand", "contact_f1_obj",
                                        "obj_trans_err_fusion_refine", "hoi_err_fusion")}
    results["frame_level"] = eval_f1
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    label = ds.upper()
    plot_curves(curves, counts, out_dir / "peri_event.png",
                f"{label}: mean published probability aligned at GT events (sliding-window frames only)",
                {"hand": "P(hand contact)", "obj": "P(object moving)"})
    plot_endpoint(ep_curves, ep_counts, out_dir / "endpoint_heads.png",
                  f"{label}: pose-prior endpoint heads at GT hand events")
    plot_f1(results, out_dir / "event_f1.png", f"{label}: event F1 vs matching tolerance")
    write_summary(results, out_dir / "summary.md", label)


def fmt_cross(value):
    """Mean-curve 0.5 crossing; 'never' when the average stays on one side of 0.5 within +-30 frames."""
    return "never" if value != value else f"{value:+.1f}"


def fmt_ci(v, c, digits=3):
    return f"{v:.{digits}f} [{c[0]:.{digits}f}, {c[1]:.{digits}f}]"


def write_summary(results, path, label):
    j10 = TOLERANCES.index(10)
    lines = [f"# A1 contact-event timing: {label} test", "",
             f"{results['sequences']} sequences; events scored on sliding-window frames only "
             f"([{WARMUP + HALF}, T_pub - {HALF})). Threshold 0.5 unless stated; hysteresis = 0.7 on / 0.3 off.", "",
             "Models: " + "; ".join(f"`{m}` = {v[0]}" for m, v in MODELS.items()), "",
             "## Events in the scored region", "",
             "| event | GT events | sequences |", "|---|---|---|"]
    for key, c in results["curves"].items():
        lines.append(f"| {key} | {c['n_events']} | {c['n_sequences']} |")
    for rule, name in (("thr", "threshold 0.5"), ("hys", "hysteresis 0.7/0.3")):
        lines += ["", f"## Event F1 by tolerance ({name})", "",
                  "| event | model | " + " | ".join(f"F1@{t}" for t in TOLERANCES) + " | F1@10 95% CI | over-seg (pred/GT) |",
                  "|---|---|" + "---|" * (len(TOLERANCES) + 2)]
        for key, res in results[rule].items():
            for m in MODELS:
                r = res[m]
                lines.append(f"| {key} | {m} | " + " | ".join(f"{v:.3f}" for v in r["f1"])
                             + f" | [{r['f1_ci_tol10'][0]:.3f}, {r['f1_ci_tol10'][1]:.3f}] | "
                             + fmt_ci(r["overseg"], r["overseg_ci"], 2) + " |")
    lines += ["", "## Timing of matched events (tolerance 30, threshold 0.5)", "",
              "Conditional on a match: a model that misses the hard events is timed on the easy ones only, so read it",
              "together with F1. The mean-curve crossing is unconditional (frame where the average probability crosses 0.5).", "",
              "| event | model | matched | median delta (pred - GT) | median abs delta | mean abs delta [95% CI] | within 5 f | mean-curve 0.5 crossing |",
              "|---|---|---|---|---|---|---|---|"]
    for key, res in results["thr"].items():
        for m in MODELS:
            r = res[m]
            matched = round(r["f1"][TOLERANCES.index(30)] * (r["n_gt"] + r["n_pred"]) / 2)
            lines.append(f"| {key} | {m} | {matched} | {r['median_delta']:+.1f} | {r['median_abs_delta']:.1f} | "
                         + fmt_ci(r["mean_abs_delta"], r["mean_abs_delta_ci"], 2)
                         + f" | {r['within5']:.2f} | {fmt_cross(results['curves'][key][m]['cross50'])} |")
    lines += ["", "## Paired differences (bootstrap over sequences, threshold 0.5)", "",
              "| event | pair | F1@10 | mean abs delta of matched events (frames) | over-seg |", "|---|---|---|---|---|"]
    for key, res in results["thr"].items():
        for pair, d in res["pairs"].items():
            lines.append(f"| {key} | {pair} | {d['f1_tol10'][0]:+.3f} [{d['f1_tol10'][1]:+.3f}, {d['f1_tol10'][2]:+.3f}] | "
                         f"{d['mean_abs_delta'][0]:+.2f} [{d['mean_abs_delta'][1]:+.2f}, {d['mean_abs_delta'][2]:+.2f}] | "
                         f"{d['overseg'][0]:+.2f} [{d['overseg'][1]:+.2f}, {d['overseg'][2]:+.2f}] |")
    lines += ["", "## Endpoint heads at GT hand events", "", "| head | model | peak probability | peak offset (frames) |",
              "|---|---|---|---|"]
    for key, r in results["endpoint"].items():
        lines.append(f"| {key} | | {r['peak']:.3f} | {r['peak_offset']:+d} |")
    lines += ["", f"## Hand events by interaction mode (recall within {TOLERANCES[2]} f, median abs delta)", "",
              "| event | model | " + " | ".join(("lift", "ground", "tilt")) + " |", "|---|---|---|---|---|"]
    for kind, table in results["per_mode"].items():
        for m in MODELS:
            cells = []
            for mode in ("lift", "ground", "tilt"):
                r = table[m].get(mode)
                cells.append(f"{r['recall']:.2f} / {r['median_abs_delta']:.1f} (n={r['n']})" if r else "-")
            lines.append(f"| hand {kind} | {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Frame-level reference (eval JSON)", "", "| model | F1 L | F1 R | F1 obj | Object (cm) | HOI (cm) |",
              "|---|---|---|---|---|---|"]
    for m, r in results["frame_level"].items():
        lines.append(f"| {m} | {r['contact_f1_lhand']:.3f} | {r['contact_f1_rhand']:.3f} | {r['contact_f1_obj']:.3f} | "
                     f"{r['obj_trans_err_fusion_refine']:.2f} | {r['hoi_err_fusion']:.2f} |")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
