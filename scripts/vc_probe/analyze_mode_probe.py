#!/usr/bin/env python3
"""Pilot: is the interaction mode of a contact segment (lift vs push/drag) readable from VC internals?

Inputs are the per-sequence dumps of ``dump_vc_features.py`` for IMUHOI++ (``ours``) and the three-branch
IMUHOI (``imuhoi``) on the same test split, and the GT segment labels of ``mode_labels.py``.

Every probe is linear (standardize -> PCA(32) -> class-balanced logistic regression), cross-validated
leave-one-subject-out unless stated, scored by balanced accuracy (chance 0.5 for two classes); 95 % CIs
bootstrap over sequences, and differences between representations use the same resamples (paired).

  1. segment level: window means before the onset (BEHAVE only), right after it and during the hold;
     variants: onsets published by sliding windows only (frames < 60 come from the warmup pass, which sees
     inputs up to frame 69), bimanual segments only, leave-one-object-out on objects with both modes;
  2. time resolved: frame features at fixed offsets from the GT contact onset;
  3. unsupervised: onset-window embeddings, kNN purity across subjects, k-means AMI, RSA.

    python scripts/vc_probe/analyze_mode_probe.py --dataset behave --root OUT
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

LOOKAHEAD = 10      # published frame t has seen inputs up to t + LOOKAHEAD
WARMUP = 60         # frames [0, 60) are published together from one pass over [0, 70)
PCA_DIM = 32
N_BOOT = 1000
CLASSES = ("lift", "ground")
GROUPS = ("left_arm", "right_arm", "torso", "left_leg", "right_leg")

# name -> (source, key, column slice, label, family, frame offset).  b_t and the object-motion BiLSTM see
# the 10 lookahead frames of their window; the IMUHOI contact LSTM is causal, so ``imuhoi_hc_h10`` reads it
# 10 frames later (same input horizon as b_t, valid on sliding-window frames).
REPS = {
    "ours_b": ("ours", "feat_b", None, "IMUHOI++ pose prior b_t", "ours", 0),
    "ours_hc": ("ours", "feat_hc", None, "IMUHOI++ hand-contact LSTM", "ours", 0),
    "ours_om": ("ours", "feat_om", None, "IMUHOI++ object-motion LSTM", "ours", 0),
    "imuhoi_hc": ("imuhoi", "feat_hc", None, "IMUHOI contact LSTM", "imuhoi", 0),
    "imuhoi_hc_h10": ("imuhoi", "feat_hc", None, "IMUHOI contact LSTM, +10 f horizon", "imuhoi", LOOKAHEAD),
    "ours_in": ("ours", "feat_in", None, "input: HP pose stream", "input", 0),
    "ours_cpl": ("ours", "feat_cpl", None, "input: coupling features", "input", 0),
    "imuhoi_cin": ("imuhoi", "feat_cin", None, "input: forearm/object IMU + vel.", "input", 0),
    "ours_rand_b": ("ours", "rand_b", None, "control: untrained b_t", "control", 0),
}
GROUP_REPS = {
    f"grp_{g}": ("ours", "feat_grp", slice(64 * i, 64 * (i + 1)), f"group encoder: {g}", "ours", 0)
    for i, g in enumerate(GROUPS)
}
REFS = ("imuhoi_hc", "imuhoi_hc_h10", "ours_rand_b", "ours_in")
ALL_REPS = {**REPS, **GROUP_REPS}
MAIN = ("ours_b", "ours_hc", "ours_om", "imuhoi_hc", "imuhoi_hc_h10", "ours_in", "ours_cpl", "imuhoi_cin",
        "ours_rand_b")
TIME_REPS = ("ours_b", "ours_hc", "imuhoi_hc", "imuhoi_hc_h10", "ours_in", "imuhoi_cin", "ours_rand_b")

# Reference palette (dataviz skill), validated: 4 slots adjacent / 3 slots all-pairs on #fcfcfb.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"ours_b": "#2a78d6", "ours_hc": "#eb6834", "imuhoi_hc": "#1baf7a", "imuhoi_hc_h10": "#eda100"}
SHORT = {"ours_b": "ours b_t", "ours_hc": "ours hand LSTM", "imuhoi_hc": "IMUHOI LSTM",
         "imuhoi_hc_h10": "IMUHOI LSTM +10 f"}
CONTROL_STYLE = {"ours_in": ("#8a8984", (0, (5, 2))), "imuhoi_cin": ("#8a8984", (0, (5, 1.5, 1, 1.5))),
                 "ours_rand_b": ("#8a8984", (0, (1, 1.5)))}
MODE_COLORS = {"lift": "#2a78d6", "ground": "#eb6834", "tilt": "#1baf7a"}
FAMILY_COLORS = {"ours": "#2a78d6", "imuhoi": "#eb6834", "input": "#8a8984", "control": "#8a8984"}


def subject_of(seq: str, dataset: str) -> str:
    pattern = r"(sub\d+)" if dataset == "omomo" else r"Date\d+_(Sub\d+)"
    return re.match(pattern, seq).group(1)


def load_segments(path: Path, dataset: str):
    segs = json.loads(path.read_text())
    by_file = collections.defaultdict(list)
    for s in segs:
        by_file[s["file"]].append(s)
    out = []
    for file, rows in by_file.items():
        rows.sort(key=lambda s: s["a"])
        prev_end = -10 ** 9
        for s in rows:
            s = dict(s, prev_end=prev_end, T_pub=s["T"] - LOOKAHEAD, subject=subject_of(s["seq"], dataset))
            prev_end = s["b"]
            # BEHAVE "sit" sequences: the chair slides while sat on, not a hand-driven mode.
            if dataset == "behave" and "_sit" in s["seq"]:
                continue
            out.append(s)
    return out


def load_features(root: Path, dataset: str, files):
    keys = {src: sorted({v[1] for v in REPS.values() if v[0] == src} | ({"feat_grp"} if src == "ours" else set()))
            for src in ("ours", "imuhoi")}
    feats = {}
    for src in ("ours", "imuhoi"):
        directory = root / f"{dataset}_{src}" / "features"
        feats[src] = {}
        for f in files:
            record = torch.load(directory / f"{f}.pt", map_location="cpu", weights_only=False)
            feats[src][f] = {k: record[k].numpy() for k in keys[src]}
    return feats


def rep_array(feats, rep, file):
    src, key, cols, _, _, offset = ALL_REPS[rep]
    arr = feats[src][file][key]
    if cols is not None:
        arr = arr[:, cols]
    if offset:
        arr = np.concatenate([arr[offset:], np.repeat(arr[-1:], offset, axis=0)])
    return arr


def window_mean(arr, lo, hi, min_frames=10):
    lo, hi = max(lo, 0), min(hi, arr.shape[0])
    if hi - lo < min_frames:
        return None
    return arr[lo:hi].astype(np.float32).mean(0)


def probe(n_features, n_train):
    k = int(min(PCA_DIM, n_features, n_train - 1))
    return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))


def oof_predict(X, y, groups):
    pred = np.empty_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if np.unique(y[tr]).size < 2:
            pred[te] = np.bincount(y[tr]).argmax()
            continue
        pred[te] = probe(X.shape[1], len(tr)).fit(X[tr], y[tr]).predict(X[te])
    return pred


def bal_acc(y, pred):
    """Balanced accuracy over the classes present in ``y`` (same as sklearn's)."""
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def bootstrap(y, preds, seqs, seed=0):
    """Balanced accuracy of every prediction vector on the same sequence resamples."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(seqs)
    members = [np.flatnonzero(seqs == s) for s in uniq]
    draws = {name: [] for name in preds}
    for _ in range(N_BOOT):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        if np.unique(y[idx]).size < 2:
            continue
        for name, pred in preds.items():
            draws[name].append(bal_acc(y[idx], pred[idx]))
    return {name: np.asarray(v) for name, v in draws.items()}


def summarize(y, preds, seqs):
    boots = bootstrap(y, preds, seqs)
    out = {}
    for name, pred in preds.items():
        row = {"ba": bal_acc(y, pred),
               "ci": [float(np.percentile(boots[name], 2.5)), float(np.percentile(boots[name], 97.5))]}
        for ref in REFS:
            if ref in preds and name != ref:
                diff = boots[name] - boots[ref]
                row[f"minus_{ref}"] = [float(np.mean(diff)), float(np.percentile(diff, 2.5)),
                                       float(np.percentile(diff, 97.5))]
        out[name] = row
    return out


def segment_decoding(segs, feats, window, reps, variant):
    rows, X = [], {r: [] for r in reps}
    for s in segs:
        if s["mode"] not in CLASSES:
            continue
        if variant == "bimanual" and s["hands"] != "LR":
            continue
        if variant == "sliding" and s["a"] < WARMUP:
            continue  # onset published by a sliding window, not by the warmup pass
        if window == "pre":
            lo, hi = s["a"] - 40, s["a"] - LOOKAHEAD
            if lo < WARMUP or s["prev_end"] > lo:
                continue
        elif window == "onset":
            lo, hi = s["a"], s["a"] + 30
        else:
            lo, hi = s["a"] + 30, min(s["b"], s["a"] + 90)
        vecs = {r: window_mean(rep_array(feats, r, s["file"]), lo, hi) for r in reps}
        if any(v is None for v in vecs.values()):
            continue
        rows.append(s)
        for r in reps:
            X[r].append(vecs[r])
    if not rows:
        return None
    y = np.array([CLASSES.index(s["mode"]) for s in rows])
    seqs = np.array([s["file"] for s in rows])
    if variant == "loo_object":
        counts = collections.Counter((s["obj"], s["mode"]) for s in rows)
        keep_obj = {o for o in {s["obj"] for s in rows} if min(counts[(o, c)] for c in CLASSES) >= 5}
        keep = np.array([s["obj"] in keep_obj for s in rows])
        rows = [s for s, k in zip(rows, keep) if k]
        y, seqs = y[keep], seqs[keep]
        X = {r: [v for v, k in zip(X[r], keep) if k] for r in reps}
        groups = np.array([s["obj"] for s in rows])
    else:
        groups = np.array([s["subject"] for s in rows])
    if len(rows) < 20 or np.unique(y).size < 2 or np.unique(groups).size < 3:
        return None
    preds = {r: oof_predict(np.stack(X[r]), y, groups) for r in reps}
    result = summarize(y, preds, seqs)
    result["_n"] = {"segments": len(rows), "lift": int((y == 0).sum()), "ground": int((y == 1).sum()),
                    "sequences": int(np.unique(seqs).size), "groups": int(np.unique(groups).size)}
    if variant == "loo_object":
        result["_n"]["objects"] = sorted({s["obj"] for s in rows})
    return result


def time_resolved(segs, feats, taus, reps, dataset):
    lo_tau, hi_tau = min(taus), max(taus)
    rows = []
    for s in segs:
        if s["mode"] not in CLASSES or s["a"] + hi_tau + 3 > s["T_pub"]:
            continue
        if lo_tau < 0 and (s["a"] + lo_tau - 2 < WARMUP or s["prev_end"] > s["a"] + lo_tau):
            continue
        rows.append(s)
    y = np.array([CLASSES.index(s["mode"]) for s in rows])
    seqs = np.array([s["file"] for s in rows])
    groups = np.array([s["subject"] for s in rows])
    curves = {r: [] for r in reps}
    for tau in taus:
        preds = {}
        for r in reps:
            X = np.stack([rep_array(feats, r, s["file"])[max(0, s["a"] + tau - 2):s["a"] + tau + 3]
                          .astype(np.float32).mean(0) for s in rows])
            preds[r] = oof_predict(X, y, groups)
        summary = summarize(y, preds, seqs)
        for r in reps:
            curves[r].append(summary[r])
    in_warmup = [int(sum(s["a"] + tau < WARMUP for s in rows)) for tau in taus]
    return {"taus": list(taus), "curves": curves, "in_warmup": in_warmup,
            "_n": {"segments": len(rows), "lift": int((y == 0).sum()), "ground": int((y == 1).sum()),
                   "sequences": int(np.unique(seqs).size), "subjects": int(np.unique(groups).size)}}


def structure(segs, feats, reps, modes=("lift", "ground", "tilt")):
    rows, X = [], {r: [] for r in reps}
    for s in segs:
        if s["mode"] not in modes:
            continue
        vecs = {r: window_mean(rep_array(feats, r, s["file"]), s["a"], s["a"] + 30) for r in reps}
        if any(v is None for v in vecs.values()):
            continue
        rows.append(s)
        for r in reps:
            X[r].append(vecs[r])
    mode = np.array([modes.index(s["mode"]) for s in rows])
    obj = np.unique([s["obj"] for s in rows], return_inverse=True)[1]
    subj = np.unique([s["subject"] for s in rows], return_inverse=True)[1]
    iu = np.triu_indices(len(rows), 1)
    same = {name: (lab[:, None] != lab[None, :])[iu].astype(float) for name, lab in
            (("mode", mode), ("object", obj), ("subject", subj))}
    ranks = {k: rankdata(v) for k, v in same.items()}
    design = np.column_stack([np.ones(len(iu[0])), ranks["object"], ranks["subject"]])
    resid_mode = ranks["mode"] - design @ np.linalg.lstsq(design, ranks["mode"], rcond=None)[0]
    out, embeds = {}, {}
    for r in reps:
        Z = PCA(n_components=PCA_DIM, random_state=0).fit_transform(StandardScaler().fit_transform(np.stack(X[r])))
        embeds[r] = Z
        dist = pdist(Z)
        rd = rankdata(dist)
        resid_d = rd - design @ np.linalg.lstsq(design, rd, rcond=None)[0]
        rsa = {k: float(np.corrcoef(rd, ranks[k])[0, 1]) for k in ranks}
        rsa["mode_partial"] = float(np.corrcoef(resid_d, resid_mode)[0, 1])
        # kNN purity among segments of other subjects, class-balanced, vs the same-class share of candidates
        full = np.zeros((len(rows), len(rows)))
        full[iu] = dist
        full = full + full.T
        purity, chance = [], []
        for i in range(len(rows)):
            cand = np.flatnonzero(subj != subj[i])
            nn = cand[np.argsort(full[i, cand])[:10]]
            purity.append(np.mean(mode[nn] == mode[i]))
            chance.append(np.mean(mode[cand] == mode[i]))
        purity, chance = np.asarray(purity), np.asarray(chance)
        ami = {}
        for name, lab in (("mode", mode), ("object", obj), ("subject", subj)):
            k = int(np.unique(lab).size)
            clusters = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)
            ami[name] = float(adjusted_mutual_info_score(lab, clusters))
        out[r] = {
            "rsa_spearman": rsa,
            "knn_purity": float(np.mean([purity[mode == c].mean() for c in np.unique(mode)])),
            "knn_chance": float(np.mean([chance[mode == c].mean() for c in np.unique(mode)])),
            "kmeans_ami": ami,
        }
    counts = collections.Counter(s["mode"] for s in rows)
    out["_n"] = {"segments": len(rows), **counts, "objects": int(obj.max() + 1), "subjects": int(subj.max() + 1)}
    return out, embeds, np.array([s["mode"] for s in rows]), np.array([s["obj"] for s in rows])


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_time(result, path, title):
    fig, ax = plt.subplots(figsize=(6.6, 4.6), facecolor=SURFACE)
    style_axes(ax)
    taus = np.asarray(result["taus"])
    if taus.min() < 0:
        ax.axvspan(-LOOKAHEAD, 0, color=GRID, alpha=0.6, linewidth=0)
        ax.text(-LOOKAHEAD / 2, 0.985, "window\nsees onset", ha="center", va="top", fontsize=7, color=INK2,
                transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.axvline(0, color=INK2, linewidth=0.8)
    ends = []
    for r in TIME_REPS:
        ba = np.array([c["ba"] for c in result["curves"][r]])
        lo = np.array([c["ci"][0] for c in result["curves"][r]])
        hi = np.array([c["ci"][1] for c in result["curves"][r]])
        label = REPS[r][3]
        if r in SERIES:
            ax.fill_between(taus, lo, hi, color=SERIES[r], alpha=0.08, linewidth=0)
            ax.plot(taus, ba, color=SERIES[r], linewidth=2, marker="o", markersize=3.5, label=label)
            ends.append([ba[-1], ba[-1], r])
        else:
            color, dash = CONTROL_STYLE[r]
            ax.plot(taus, ba, color=color, linewidth=1.4, linestyle=dash, label=label)
    ax.set_xlim(taus.min(), taus.max())
    ax.set_ylim(0.3, 1.0)
    # Direct labels at the line ends, pushed apart so they never overlap.
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.035)
    for value, y_text, r in ends:
        ax.annotate(SHORT[r], (taus[-1], value), xytext=(taus[-1] + 0.02 * np.ptp(taus), y_text),
                    textcoords="data", fontsize=7, color=INK, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=SERIES[r], linewidth=0.8, shrinkA=0, shrinkB=0))
    ax.set_xlabel("frames from GT hand-contact onset (30 fps)", fontsize=8, color=INK)
    ax.set_ylabel("balanced accuracy, lift vs push/drag", fontsize=8, color=INK)
    n = result["_n"]
    ax.set_title(f"{title}  (n = {n['lift']} lift / {n['ground']} push-drag segments, "
                 f"{n['sequences']} seqs, leave-one-subject-out)", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80, bottom=0.27)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_embeddings(embeds, modes, path, title):
    reps = ("ours_b", "imuhoi_hc")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), facecolor=SURFACE)
    for ax, r in zip(axes, reps):
        style_axes(ax)
        ax.grid(False)
        xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(embeds[r])
        for m in ("lift", "ground", "tilt"):
            sel = modes == m
            if sel.any():
                ax.scatter(xy[sel, 0], xy[sel, 1], s=14, color=MODE_COLORS[m], edgecolor=SURFACE, linewidth=0.4,
                           label={"ground": "push/drag"}.get(m, m), alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(REPS[r][3], fontsize=8.5, color=INK, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="upper right", ncol=3, bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fmt_diff(row, ref):
    diff = row.get(f"minus_{ref}")
    return f"{diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}]" if diff else "-"


def fmt_row(name, row):
    cell = f"{row['ba']:.3f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}]"
    return f"| {name} | {cell} | {fmt_diff(row, 'imuhoi_hc')} | {fmt_diff(row, 'imuhoi_hc_h10')} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("omomo", "behave"), required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--plot_only", action="store_true", help="redraw the time-resolved figure from results.json")
    args = ap.parse_args()
    ds, root = args.dataset, args.root
    out_dir = root / f"analysis_{ds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_time(json.loads((out_dir / "results.json").read_text())["time"], out_dir / "time_resolved.png", ds.upper())
        return

    segs = load_segments(root / f"labels_{ds}_test.json", ds)
    feats = load_features(root, ds, sorted({s["file"] for s in segs}))
    results = {"dataset": ds, "segments_by_mode": dict(collections.Counter(s["mode"] for s in segs))}

    windows = ("pre", "onset", "hold") if ds == "behave" else ("onset", "hold")
    results["segment"] = {}
    for window in windows:
        for variant in ("all", "sliding", "bimanual", "loo_object"):
            reps = MAIN + (tuple(GROUP_REPS) if variant == "all" else ())
            res = segment_decoding(segs, feats, window, reps, variant)
            results["segment"][f"{window}/{variant}"] = res
            print(f"[segment] {window}/{variant}: " + ("too few segments" if res is None else
                  ", ".join(f"{r} {res[r]['ba']:.3f}" for r in MAIN) + f"  n={res['_n']}"), flush=True)

    taus = range(-45, 46, 5) if ds == "behave" else range(0, 61, 5)
    results["time"] = time_resolved(segs, feats, taus, TIME_REPS, ds)
    print(f"[time] n={results['time']['_n']}", flush=True)
    plot_time(results["time"], out_dir / "time_resolved.png", ds.upper())

    struct, embeds, modes, _ = structure(segs, feats, MAIN)
    results["structure"] = struct
    plot_embeddings(embeds, modes, out_dir / "tsne_onset.png",
                    f"{ds.upper()} onset-window embeddings (t-SNE of PCA-32), colored by GT mode")
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    lines = [f"# VC mode probe pilot: {ds.upper()} test", "",
             f"Segments by mode (sit sequences excluded): {results['segments_by_mode']}", ""]
    for key, res in results["segment"].items():
        if res is None:
            lines += [f"## Segment decoding `{key}`: too few segments", ""]
            continue
        lines += [f"## Segment decoding `{key}`", "", f"n = {res['_n']}", "",
                  "| representation | balanced acc. [95% CI] | minus IMUHOI LSTM | minus IMUHOI LSTM (+10 f) |",
                  "|---|---|---|---|"]
        lines += [fmt_row(ALL_REPS[r][3], res[r]) for r in res if not r.startswith("_")]
        lines += ["", f"b_t minus untrained b_t: {fmt_diff(res['ours_b'], 'ours_rand_b')}; "
                      f"b_t minus HP pose input: {fmt_diff(res['ours_b'], 'ours_in')}", ""]
    t = results["time"]
    lines += ["## Time-resolved decoding (balanced accuracy)", "", f"n = {t['_n']}", "",
              "| tau | " + " | ".join(REPS[r][3] for r in TIME_REPS)
              + " | b_t minus IMUHOI LSTM (+10 f) | b_t minus untrained b_t | segs in warmup |",
              "|---|" + "---|" * (len(TIME_REPS) + 3)]
    for i, tau in enumerate(t["taus"]):
        b = t["curves"]["ours_b"][i]
        lines.append(f"| {tau:+d} | " + " | ".join(f"{t['curves'][r][i]['ba']:.3f}" for r in TIME_REPS)
                     + f" | {fmt_diff(b, 'imuhoi_hc_h10')} | {fmt_diff(b, 'ours_rand_b')} | {t['in_warmup'][i]} |")
    lines += ["", "## Unsupervised structure of onset-window embeddings", "", f"n = {struct['_n']}", "",
              "| representation | kNN purity (chance) | k-means AMI mode / object / subject | RSA mode / object / subject | RSA mode, partial |",
              "|---|---|---|---|---|"]
    for r in MAIN:
        s = struct[r]
        lines.append(f"| {REPS[r][3]} | {s['knn_purity']:.3f} ({s['knn_chance']:.3f}) | "
                     f"{s['kmeans_ami']['mode']:.3f} / {s['kmeans_ami']['object']:.3f} / {s['kmeans_ami']['subject']:.3f} | "
                     f"{s['rsa_spearman']['mode']:.3f} / {s['rsa_spearman']['object']:.3f} / {s['rsa_spearman']['subject']:.3f} | "
                     f"{s['rsa_spearman']['mode_partial']:.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
