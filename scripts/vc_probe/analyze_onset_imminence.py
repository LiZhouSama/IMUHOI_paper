#!/usr/bin/env python3
"""How far ahead of a GT hand-contact onset does the VC (or its pose prior) signal the coming contact?

For every hand and every non-contact frame t on sliding-window frames, ``lead`` = frames until that hand's next GT
onset. Positives at lead L are scored against idle frames (no onset within 30 frames). Published frame t has seen
inputs up to t + 10, so leads 1-10 lie inside the lookahead (the window already contains the onset) and only leads
> 10 are real-time anticipation.

Scores
  direct:  ours_ant   anticipation head, "a start of this hand within (t, t+15]" (A1 dump of the same v5ar model)
           ours_p / imuhoi_p     published hand-contact probability; imuhoi_p_h10 = IMUHOI's at t+10 (same horizon)
  probes:  linear (PCA-32 + balanced logistic regression), trained on lead 1-20 vs idle, leave-one-subject-out
           out-of-fold scores, on b_t, our hand-contact LSTM, IMUHOI's contact LSTM (at t and at t+10), the HP pose
           input and an untrained b_t.
AUC CIs: sequence bootstrap. Detection lead: per onset (30 idle frames before), the earliest frame from which the
score stays above a threshold set to 5 % false alarms on idle frames up to the onset.

    python scripts/vc_probe/analyze_onset_imminence.py --dataset behave
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_contact_events as A  # noqa: E402

PILOT = A.PILOT_ROOT
MAX_LEAD = 30
PROBE_LEAD = 20
IDLE_PER_HAND = 15000
N_BOOT = 200
FAR = 0.05
PROBES = {"ours_b": ("ours", "feat_b", 0), "ours_hc": ("ours", "feat_hc", 0), "imuhoi_hc": ("imuhoi", "feat_hc", 0),
          "imuhoi_hc_h10": ("imuhoi", "feat_hc", A.LOOKAHEAD), "ours_in": ("ours", "feat_in", 0),
          "ours_rand_b": ("ours", "rand_b", 0)}
DIRECT = ("ours_ant", "ours_p", "imuhoi_p", "imuhoi_p_h10")
LABELS = {"ours_ant": "IMUHOI++ anticipation head", "ours_p": "IMUHOI++ published contact prob.",
          "imuhoi_p": "IMUHOI published contact prob.", "imuhoi_p_h10": "IMUHOI contact prob. at t+10",
          "ours_b": "probe: IMUHOI++ pose prior b_t", "ours_hc": "probe: IMUHOI++ hand-contact LSTM",
          "imuhoi_hc": "probe: IMUHOI contact LSTM", "imuhoi_hc_h10": "probe: IMUHOI contact LSTM at t+10",
          "ours_in": "probe: HP pose input", "ours_rand_b": "probe: untrained b_t"}
COLORS = {"ours_ant": "#2a78d6", "ours_b": "#eb6834", "imuhoi_p_h10": "#1baf7a", "imuhoi_hc_h10": "#eda100"}
GRAY = {"ours_p": (0, (8, 3)), "imuhoi_p": (0, (1, 1.5)), "ours_in": (0, (5, 1.5, 1, 1.5)), "ours_rand_b": (0, (1, 1.5))}
PLOT_KEYS = ("ours_ant", "ours_b", "imuhoi_p_h10", "imuhoi_hc_h10", "ours_p", "ours_in", "ours_rand_b")
MOVING_MM_S = 300  # root horizontal speed separating walking from standing (11-frame average)
STRATA_KEYS = ("ours_ant", "ours_b", "ours_in", "ours_rand_b", "imuhoi_hc_h10", "imuhoi_p_h10")
PAIRS = (("ours_b", "ours_in"), ("ours_b", "ours_rand_b"), ("ours_b", "imuhoi_hc_h10"), ("ours_ant", "imuhoi_p_h10"))


def subject_of(seq, ds):
    return re.match(r"(sub\d+)" if ds == "omomo" else r"Date\d+_(Sub\d+)", seq).group(1)


def shifted(arr, offset):
    return arr if not offset else np.concatenate([arr[offset:], np.repeat(arr[-1:], offset, axis=0)])


def collect(ds, rng, src):
    """Candidate frames per hand: (file index, frame, hand, lead, ...) plus their scores and probe features.

    ``src``: data / feature directories and ``pad`` (frames < pad are a synthetic static prefix and never used)."""
    first = max(A.WARMUP, src["pad"])
    files = sorted(p.stem for p in src["ours"].glob("*.pt"))
    rows, seq_names, onsets, traces, feature_arrays = [], [], [], {}, []
    for i, f in enumerate(files):
        data = torch.load(src["data"] / f"{f}.pt", map_location="cpu", weights_only=False)
        seq_names.append(str(data["seq_name"]))
        ours = torch.load(src["ours"] / f"{f}.pt", map_location="cpu", weights_only=False)
        imu = torch.load(src["imuhoi"] / f"{f}.pt", map_location="cpu", weights_only=False)
        T_pub = ours["feat_b"].shape[0]
        p_ours = ours["pred_hand_contact_prob"].numpy()
        if "pred_contact_anticipation_prob" in ours:
            ant = ours["pred_contact_anticipation_prob"].numpy()
        else:  # pilot dumps predate saving the heads: use the A1 dump of the same model and protocol
            a1 = torch.load(A.EVENT_ROOT / f"{ds}_v5ar" / "probs" / f"{f}.pt", map_location="cpu", weights_only=False)
            if not np.allclose(p_ours, a1["pred_hand_contact_prob"].numpy(), atol=1e-4):
                raise ValueError(f"{f}: pilot and A1 dumps of v5ar disagree")
            ant = a1["pred_contact_anticipation_prob"].numpy()
        p_imu = imu["pred_hand_contact_prob"].numpy()
        feature_arrays.append({k: shifted((ours if src == "ours" else imu)[key].numpy(), off)
                               for k, (src, key, off) in PROBES.items()})
        p_imu_h10 = shifted(p_imu, A.LOOKAHEAD)
        root = data["position_global_full_gt_world"][:, 0].numpy().astype(np.float64)
        step = np.linalg.norm(np.diff(root[:, [0, 2]], axis=0), axis=1) * 30.0  # 30 fps
        speed = np.convolve(np.concatenate([step[:1], step]), np.ones(11) / 11, mode="same")
        for hand, key in enumerate(("lhand_contact", "rhand_contact")):
            gt = data[key].numpy().astype(bool)
            T = gt.shape[0]
            starts = np.flatnonzero(np.diff(gt.astype(np.int8)) == 1) + 1  # first contact frame of each segment
            idx = np.searchsorted(starts, np.arange(T), side="right")
            lead = np.where(idx < len(starts), starts[np.minimum(idx, max(len(starts) - 1, 0))] - np.arange(T), 10 ** 6) \
                if len(starts) else np.full(T, 10 ** 6)
            other = data[("rhand_contact", "lhand_contact")[hand]].numpy().astype(bool)
            anyc = gt | other
            # Frames since the last contact of either hand (fresh approach = none in the previous 30 frames).
            last = np.full(T, -10 ** 6)
            for t in range(T):
                last[t] = t if anyc[t] else (last[t - 1] if t else -10 ** 6)
            frames = np.arange(first, min(T_pub, T - MAX_LEAD))
            cand = frames[~gt[frames]]
            for t in cand:
                fresh = int((not other[t]) and t - last[t] > MAX_LEAD)
                rows.append((i, int(t), hand, int(min(lead[t], 10 ** 6)), int(not other[t]), fresh,
                             int(speed[t] * 1000)))
            for e in starts:
                if first + MAX_LEAD <= e < T_pub - A.LOOKAHEAD and not gt[e - MAX_LEAD:e].any():
                    onsets.append((i, hand, int(e)))
            traces[(i, hand)] = {"ours_ant": ant[:, 2 * hand], "ours_p": p_ours[:, hand], "imuhoi_p": p_imu[:, hand],
                                 "imuhoi_p_h10": p_imu_h10[:, hand]}
    rows = np.array(rows, dtype=np.int64)
    # Subsample idle frames per hand (seeded) to keep the probes tractable; every positive frame is kept.
    keep = np.ones(len(rows), dtype=bool)
    for hand in (0, 1):
        idle_idx = np.flatnonzero((rows[:, 2] == hand) & (rows[:, 3] > MAX_LEAD))  # file, t, hand, lead, free, fresh
        if len(idle_idx) > IDLE_PER_HAND:
            keep[rng.choice(idle_idx, len(idle_idx) - IDLE_PER_HAND, replace=False)] = False
    rows = rows[keep]
    direct = {k: np.array([traces[(r[0], r[2])][k][r[1]] for r in rows], dtype=np.float64) for k in DIRECT}
    feats = {k: np.stack([feature_arrays[r[0]][k][r[1]] for r in rows]).astype(np.float32) for k in PROBES}
    return files, seq_names, rows, direct, feats, onsets, traces


def oof_scores(X, y, train_mask, groups):
    scores = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        tr = tr[train_mask[tr]]
        if np.unique(y[tr]).size < 2:
            continue
        model = make_pipeline(StandardScaler(), PCA(n_components=min(32, X.shape[1]), random_state=0),
                              LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
        model.fit(X[tr], y[tr])
        scores[te] = model.decision_function(X[te])
    return scores


def auc_with_ci(y, s, seq, W):
    point = roc_auc_score(y, s)
    draws = []
    for b in range(N_BOOT):
        w = W[b, seq]
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        draws.append(roc_auc_score(y, s, sample_weight=w))
    return float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_auc_diff(y, s1, s2, seq, W):
    point = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    draws = []
    for b in range(N_BOOT):
        w = W[b, seq]
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        draws.append(roc_auc_score(y, s1, sample_weight=w) - roc_auc_score(y, s2, sample_weight=w))
    return [float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def detection_leads(onsets, traces, idle_scores, key):
    """Lead (frames before the onset) from which the score stays above the 5 % false-alarm threshold."""
    thr = float(np.quantile(idle_scores, 1 - FAR))
    leads = []
    for i, hand, e in onsets:
        s = traces[(i, hand)][key]
        if s[e] < thr:
            after = np.flatnonzero(s[e:e + A.LOOKAHEAD + 1] >= thr)
            leads.append(-int(after[0]) if after.size else -99)
            continue
        t = e
        while t - 1 >= e - MAX_LEAD and s[t - 1] >= thr:
            t -= 1
        leads.append(e - t)
    return thr, np.asarray(leads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("omomo", "behave"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="", help="feature dumps PILOT/{ds}{tag}_{ours,imuhoi}, e.g. _pad90")
    ap.add_argument("--data_dir", type=Path, default=None, help="sequence files (default: the dataset test split)")
    ap.add_argument("--pad", type=int, default=0, help="synthetic static prefix length to exclude")
    args = ap.parse_args()
    ds = args.dataset
    out_dir = args.out or PILOT / f"analysis_{ds}{args.tag}_onset"
    src = {"data": args.data_dir or A.DATA[ds], "pad": args.pad,
           "ours": PILOT / f"{ds}{args.tag}_ours" / "features", "imuhoi": PILOT / f"{ds}{args.tag}_imuhoi" / "features"}
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    files, seq_names, rows, direct, feats, onsets, traces = collect(ds, rng, src)
    src_pad = src["pad"]
    seq = rows[:, 0]
    lead = rows[:, 3]
    free = rows[:, 4].astype(bool)
    fresh = rows[:, 5].astype(bool)
    moving = rows[:, 6] >= MOVING_MM_S
    groups = np.array([subject_of(seq_names[i], ds) for i in seq])
    W = A.boot_weights(len(files))
    idle = lead > MAX_LEAD
    y_probe = (lead <= PROBE_LEAD).astype(int)
    train_mask = idle | (lead <= PROBE_LEAD)
    scores = dict(direct)
    for k in PROBES:
        scores[k] = oof_scores(feats[k], y_probe, train_mask, groups)
        print(f"[probe] {k} done", flush=True)

    results = {"dataset": ds, "pad": src["pad"], "sequences": len(files), "frames": int(len(rows)), "idle": int(idle.sum()),
               "onsets_for_detection": len(onsets), "curve": {}, "tasks": {}, "detection": {},
               "fresh_counts": {"positives_1_10": int((fresh & (lead <= 10)).sum()),
                                "positives_11_20": int((fresh & (lead > 10) & (lead <= 20)).sum()),
                                "idle": int((fresh & idle).sum())}}
    leads = np.arange(1, MAX_LEAD + 1)
    results["curve_fresh"] = {}
    for k in DIRECT + tuple(PROBES):
        curve, curve_fresh = [], []
        for L in leads:
            sel = idle | (np.abs(lead - L) <= 1)
            y = (~idle[sel]).astype(int)
            curve.append(auc_with_ci(y, scores[k][sel], seq[sel], W))
            if k in PLOT_KEYS:
                sub = sel & fresh
                y_sub = (~idle[sub]).astype(int)
                curve_fresh.append(auc_with_ci(y_sub, scores[k][sub], seq[sub], W))
        results["curve"][k] = curve
        if curve_fresh:
            results["curve_fresh"][k] = curve_fresh
        for name, lo, hi in (("visible_1_10", 1, 10), ("beyond_11_20", 11, 20), ("beyond_21_30", 21, 30)):
            sel = idle | ((lead >= lo) & (lead <= hi))
            y = (~idle[sel]).astype(int)
            results["tasks"].setdefault(name, {})[k] = auc_with_ci(y, scores[k][sel], seq[sel], W)
            # Both hands free: no contextual cue from the other hand already holding the object.
            for suffix, mask in (("_both_free", free), ("_fresh", fresh)):
                sub = sel & mask
                y_sub = (~idle[sub]).astype(int)
                results["tasks"].setdefault(name + suffix, {})[k] = auc_with_ci(y_sub, scores[k][sub], seq[sub], W)
        if k in DIRECT:
            thr, det = detection_leads(onsets, traces, scores[k][idle], k)
            results["detection"][k] = {"threshold": thr, "n": int(len(det)),
                                       "at_or_before_onset": float(np.mean(det >= 0)), "lead_ge1": float(np.mean(det >= 1)),
                                       "lead_ge5": float(np.mean(det >= 5)), "lead_ge10": float(np.mean(det >= 10)),
                                       "lead_ge11": float(np.mean(det >= 11)), "median_lead": float(np.median(det))}
        print(f"[auc] {k}: visible {results['tasks']['visible_1_10'][k][0]:.3f} beyond {results['tasks']['beyond_11_20'][k][0]:.3f}", flush=True)
    # Locomotion confound: approach frames are often walking, idle frames often standing still.
    results["moving_share"] = {"fresh_positives_11_30": float(moving[fresh & ~idle & (lead > 10)].mean()),
                               "fresh_idle": float(moving[fresh & idle].mean())}
    results["strata"], results["pairs"] = {}, {}
    for lo, hi in ((11, 20), (21, 30)):
        band = fresh & (idle | ((lead >= lo) & (lead <= hi)))
        strata = [("moving", moving), ("still", ~moving)]
        if src_pad:
            # The 60-frame window of real frame < 60 still contains static prefix frames.
            near = (rows[:, 1] - src_pad) < A.WARMUP
            strata += [("window_with_prefix", near), ("window_without_prefix", ~near)]
        for stratum, mask in strata:
            sub = band & mask
            y = (~idle[sub]).astype(int)
            if y.sum() < 20 or (1 - y).sum() < 20:
                continue
            results["strata"][f"fresh_{lo}_{hi}_{stratum}"] = {
                "n_pos": int(y.sum()), "n_idle": int((1 - y).sum()),
                **{k: auc_with_ci(y, scores[k][sub], seq[sub], W) for k in STRATA_KEYS}}
        y = (~idle[band]).astype(int)
        for a, b in PAIRS:
            results["pairs"][f"fresh_{lo}_{hi}|{a}-{b}"] = paired_auc_diff(y, scores[a][band], scores[b][band], seq[band], W)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(out_dir / "scores.npz", rows=rows, **{f"score_{k}": v for k, v in scores.items()})
    plot_curve(results, out_dir / "onset_imminence_auc.png", ds.upper())
    plot_curve(results, out_dir / "onset_imminence_auc_fresh.png", ds.upper(), key="curve_fresh",
               note="fresh approaches only")
    write_summary(results, out_dir / "summary.md", ds.upper())


def plot_curve(results, path, label, key="curve", note="all non-contact frames"):
    fig, ax = plt.subplots(figsize=(6.8, 4.6), facecolor=A.SURFACE)
    A.style_axes(ax)
    leads = np.arange(1, MAX_LEAD + 1)
    ax.axvspan(0.5, A.LOOKAHEAD + 0.5, color=A.GRID, alpha=0.6, linewidth=0)
    ax.text(A.LOOKAHEAD / 2 + 0.5, 0.985, "onset inside the\npublished window", ha="center", va="top", fontsize=6.5,
            color=A.INK2, transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color=A.INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    for k in PLOT_KEYS:
        c = results[key][k]
        mean = np.array([v[0] for v in c])
        if k in COLORS:
            lo = np.array([v[1] for v in c])
            hi = np.array([v[2] for v in c])
            ax.fill_between(leads, lo, hi, color=COLORS[k], alpha=0.12, linewidth=0)
            ax.plot(leads, mean, color=COLORS[k], linewidth=2, label=LABELS[k])
        else:
            ax.plot(leads, mean, color="#8a8984", linewidth=1.3, linestyle=GRAY[k], label=LABELS[k])
    ax.set_xlim(0.5, MAX_LEAD + 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.invert_xaxis()
    ax.set_xlabel("frames until the GT hand-contact onset (lead); right = closer to the onset", fontsize=7.5, color=A.INK)
    ax.set_ylabel("AUC: frames at this lead vs idle frames", fontsize=7.5, color=A.INK)
    ax.set_title(f"{label}: is an upcoming contact onset visible in the VC? ({note})", fontsize=8.5, color=A.INK, loc="left")
    ax.legend(fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.33)
    fig.savefig(path, dpi=180, facecolor=A.SURFACE)
    plt.close(fig)


def write_summary(results, path, label):
    pad_note = ([f"Sequences carry a {results['pad']}-frame static prefix (copies of their first frame); only real frames "
                 "are scored: candidates, idle frames and the 30 frames before a detection onset all lie after the prefix.", ""]
                if results.get("pad") else [])
    lines = [f"# Onset imminence: {label} test", ""] + pad_note + [
             f"{results['sequences']} sequences; {results['frames']} non-contact hand-frames on sliding-window frames "
             f"({results['idle']} idle = no onset of that hand within {MAX_LEAD} frames, subsampled). Published frame t "
             f"has seen inputs up to t+{A.LOOKAHEAD}: leads 1-{A.LOOKAHEAD} are inside the lookahead.", "",
             "## AUC against idle frames (95% CI, sequence bootstrap)", "",
             "| score | lead 1-10 (onset visible) | lead 11-20 | lead 21-30 |", "|---|---|---|---|"]
    for k in DIRECT + tuple(PROBES):
        cells = [results["tasks"][t][k] for t in ("visible_1_10", "beyond_11_20", "beyond_21_30")]
        lines.append(f"| {LABELS[k]} | " + " | ".join(f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for v in cells) + " |")
    lines += ["", "Same, restricted to frames where the other hand is not in contact either (no cue from a hand that "
              "already holds the object):", "",
              "| score | lead 1-10 | lead 11-20 | lead 21-30 |", "|---|---|---|---|"]
    for k in DIRECT + tuple(PROBES):
        cells = [results["tasks"][t + "_both_free"][k] for t in ("visible_1_10", "beyond_11_20", "beyond_21_30")]
        lines.append(f"| {LABELS[k]} | " + " | ".join(f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for v in cells) + " |")
    lines += ["", f"Fresh approach: both hands free and no contact of either hand in the previous {MAX_LEAD} frames "
              f"(positives and idle alike; {results['fresh_counts']}):", "",
              "| score | lead 1-10 | lead 11-20 | lead 21-30 |", "|---|---|---|---|"]
    for k in DIRECT + tuple(PROBES):
        cells = [results["tasks"][t + "_fresh"][k] for t in ("visible_1_10", "beyond_11_20", "beyond_21_30")]
        lines.append(f"| {LABELS[k]} | " + " | ".join(f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for v in cells) + " |")
    ms = results["moving_share"]
    lines += ["", f"Locomotion: share of walking frames (root speed >= {MOVING_MM_S / 1000:.1f} m/s) among fresh positives at "
              f"lead 11-30 {ms['fresh_positives_11_30']:.2f}, among fresh idle frames {ms['fresh_idle']:.2f}.", "",
              "Fresh approach within locomotion strata (AUC [95% CI]):", "",
              "| task | n positives / idle | " + " | ".join(LABELS[k] for k in STRATA_KEYS) + " |",
              "|---|---|" + "---|" * len(STRATA_KEYS)]
    for name, r in results["strata"].items():
        lines.append(f"| {name} | {r['n_pos']} / {r['n_idle']} | "
                     + " | ".join(f"{r[k][0]:.3f} [{r[k][1]:.3f}, {r[k][2]:.3f}]" for k in STRATA_KEYS) + " |")
    lines += ["", "Paired AUC differences on fresh approaches (sequence bootstrap):", "", "| task | difference | AUC difference [95% CI] |",
              "|---|---|---|"]
    for name, d in results["pairs"].items():
        task, pair = name.split("|")
        a, b = pair.split("-", 1)
        lines.append(f"| {task} | {LABELS[a]} minus {LABELS[b]} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |")
    lines += ["", f"## Detection lead at {FAR:.0%} false alarms on idle frames ({results['onsets_for_detection']} onsets "
              f"with {MAX_LEAD} idle frames before)", "",
              "| score | threshold | detected by the onset | >= 1 f early | >= 5 f early | >= 10 f early | >= 11 f early | median lead |",
              "|---|---|---|---|---|---|---|---|"]
    for k, d in results["detection"].items():
        lines.append(f"| {LABELS[k]} | {d['threshold']:.3f} | {d['at_or_before_onset']:.2f} | {d['lead_ge1']:.2f} | "
                     f"{d['lead_ge5']:.2f} | {d['lead_ge10']:.2f} | {d['lead_ge11']:.2f} | {d['median_lead']:+.0f} |")
    lines += ["", "Lead >= 11 frames = the score was above threshold before the published window could contain the onset.",
              "Negative median lead = detected after the onset (frames)."]
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
