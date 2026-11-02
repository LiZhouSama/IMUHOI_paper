#!/usr/bin/env python3
"""VC module ablation on OMOMO (2026-10-30) and BEHAVE (2026-10-31): contact and downstream metrics per variant.

Rows: full (= the published v5ar pipeline: training is deterministic and the OMOMO re-run reproduced it bit for
bit), w/o HP prior (the pose prior together with every other v5ar addition except the conditional contact
factorization: coupling features, bidirectional object-motion head, initializer perturbation), w/o conditional
contact probability (direct P(contact), IMU-HOI form), the pose prior without one body group (left arm, right
arm, torso, legs) and, supplementary, w/o pose prior only.  Every ablated row has its own VC, OT and joint stage
(outputs/IMUHOI++/RNN_cano/vc_ablation_omomo_20261030/run_ablation.sh).  Metrics on the online published frames
of the test sequences:

* frame F1 of left / right hand contact (threshold 0.15) and object motion (0.5): per-sequence means over the
  sequences whose GT has both classes, as eval_batch_online reports them (checked against its JSON);
* AP (area under the precision-recall curve, frames pooled): threshold-free, so it does not depend on how each
  variant's probabilities are calibrated (the direct variant is calibrated differently by construction);
* event F1 within +-10 frames for contact onsets / releases (both hands pooled) and object-motion starts / stops,
  from the same thresholded frames, in the A1 region (frames >= 90 and < T - 30);
* over-segmentation: predicted / GT segments in that region (flicker), hand channels pooled;
* hard-negative false-positive rate (object moving, hand not in contact), object and HOI error.

Differences to the full row carry 95 % paired sequence-bootstrap intervals (same resamples for every row).

    python scripts/vc_probe/summarize_vc_ablation.py --dataset {omomo,behave}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_contact_events as A1  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/IMUHOI++/RNN_cano"
SETUPS = {
    "omomo": {"abl": RUNS / "vc_ablation_omomo_20261030",
              "data": ROOT / "process/IMUHOI++cano/processed_split_data_OMOMO_full/test",
              "published": RUNS / "vc_v5_20260930_ot_omomo_v5ar_unified/v2_armrot_biclose/eval_omomo_best.json"},
    "behave": {"abl": RUNS / "vc_ablation_behave_20261031",
               "data": ROOT / "process/IMUHOI++cano/processed_split_data_BEHAVE/test",
               "published": RUNS / "vc_v5_20260930_ot_behave_v5ar/v2_armrot_biclose/eval_behave_best.json"},
}
for _ds, _test in (("imhd", "processed_split_data_IMHD"), ("pahoi", "processed_split_data_PAHOI_full"),
                   ("hodome", "processed_split_data_HODOME")):
    SETUPS[_ds] = {"abl": RUNS / f"vc_ablation_{_ds}_20261101",
                   "data": ROOT / f"process/IMUHOI++cano/{_test}/test",
                   "published": RUNS / f"remaining3_v5ar_vs_imuhoi3b_20261010/{_ds}/proposed/v2_armrot_biclose"
                                       f"/eval_{_ds}_best.json"}
DS = "omomo"  # set by main()
VARIANTS = (
    ("full", "Full model (IMUHOI++ VC)"),
    ("no_prior_bundle", "w/o HP prior"),
    ("direct", "w/o conditional contact prob."),
    ("no_larm", "w/o left arm"),
    ("no_rarm", "w/o right arm"),
    ("no_torso", "w/o torso"),
    ("no_legs", "w/o legs"),
    ("no_prior", "w/o pose prior only (suppl.)"),
)
REFERENCE = "full"
SUPPLEMENTARY = ("no_prior",)  # drawn in gray below a separator
THRESHOLDS = (0.15, 0.15, 0.5)  # hand_contact_threshold, object_contact_threshold of the pipeline config
CHANNELS = ("lhand", "rhand", "obj")
EVENT_TOL = 10  # A1's main tolerance
ENDPOINT_KEYS = ("hand_onset", "hand_offset", "obj_onset", "obj_offset")

# Reference palette (dataviz skill): one series per panel, so one hue; text stays in ink tokens.
SURFACE, INK, INK2, GRID, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def load_gt(files):
    gt = {}
    for f in files:
        d = torch.load(SETUPS[DS]["data"] / f"{f}.pt", map_location="cpu", weights_only=False)
        T_pub = int(d["obj_trans"].shape[0]) - A1.LOOKAHEAD
        gt[f] = np.stack([d[f"{c}_contact"].numpy().astype(bool)[:T_pub] for c in CHANNELS], axis=1)
    return gt


def load_probs(directory, files):
    probs = {}
    for f in files:
        p = torch.load(directory / f"{f}.pt", map_location="cpu", weights_only=False)["pred_hand_contact_prob"]
        probs[f] = p.float().numpy()
    return probs


def frame_stats(files, gt, probs):
    """Per sequence and channel: F1 / precision / recall (NaN when GT has one class), hard-negative FP rate."""
    n = len(files)
    out = {k: np.full((n, 3), np.nan) for k in ("f1", "precision", "recall")}
    hard_neg = np.full(n, np.nan)
    for i, f in enumerate(files):
        g, p = gt[f], probs[f]
        pred = p > np.asarray(THRESHOLDS)
        for c in range(3):
            if g[:, c].all() or not g[:, c].any():
                continue
            tp = float((pred[:, c] & g[:, c]).sum())
            fp = float((pred[:, c] & ~g[:, c]).sum())
            fn = float((~pred[:, c] & g[:, c]).sum())
            out["f1"][i, c] = 2 * tp / max(2 * tp + fp + fn, 1.0)
            out["precision"][i, c] = tp / (tp + fp) if tp + fp > 0 else 0.0
            out["recall"][i, c] = tp / (tp + fn)
        mask = g[:, 2] & ~g[:, 0] & ~g[:, 1]
        if mask.any():
            hard_neg[i] = float((pred[mask, 0] | pred[mask, 1]).mean())
    return out, hard_neg


def event_counts(files, gt, probs):
    """Per sequence: [tp, fp, fn] at +-EVENT_TOL for each ENDPOINT_KEYS entry, and [pred, gt] hand segments."""
    counts = np.zeros((len(files), len(ENDPOINT_KEYS), 3))
    segs = np.zeros((len(files), 2))
    for i, f in enumerate(files):
        T_pub = gt[f].shape[0]
        pred = probs[f] > np.asarray(THRESHOLDS)
        for k, key in enumerate(ENDPOINT_KEYS):
            group, kind = key.split("_")
            for c in A1.channel_ids(group):
                g_ev = A1.in_region(A1.events(gt[f][:, c], kind), T_pub)
                p_ev = A1.in_region(A1.events(pred[:, c], kind), T_pub)
                tp, _ = A1.match(g_ev, p_ev, EVENT_TOL)
                counts[i, k] += (tp, len(p_ev) - tp, len(g_ev) - tp)
                if group == "hand" and kind == "onset":
                    segs[i] += (len(p_ev), len(g_ev))
    return counts, segs


def pooled_ap(files, gt, probs, channel, W=None, chunk=50):
    """AP of all frames pooled (sklearn's step definition, ties grouped); with W, one value per resample."""
    seq_idx = np.concatenate([np.full(gt[f].shape[0], i) for i, f in enumerate(files)])
    y = np.concatenate([gt[f][:, channel] for f in files]).astype(np.float64)
    s = np.concatenate([probs[f][:, channel] for f in files]).astype(np.float64)
    order = np.argsort(-s, kind="mergesort")
    y, s, seq_idx = y[order], s[order], seq_idx[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]  # last index of each distinct score

    def ap(weights):
        tp = np.cumsum(weights * y, axis=-1)[..., last]
        fp = np.cumsum(weights * (1.0 - y), axis=-1)[..., last]
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / tp[..., -1:]
        d_recall = np.diff(np.concatenate([np.zeros_like(recall[..., :1]), recall], axis=-1), axis=-1)
        return (d_recall * precision).sum(-1)

    point = float(ap(np.ones_like(y)))
    if W is None:
        return point, None
    draws = np.concatenate([ap(W[b:b + chunk][:, seq_idx]) for b in range(0, W.shape[0], chunk)])
    return point, draws


def seq_mean(values, W):
    """Point estimate and bootstrap draws of the mean over finite entries of a per-sequence vector."""
    valid = np.isfinite(values)
    v = np.where(valid, values, 0.0)
    point = float(v.sum() / valid.sum())
    return point, (W @ v) / np.maximum(W @ valid, 1e-12)


def f1_counts(counts):
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)


def records(path, files):
    """Per-sequence object / HOI error (cm) from an eval_batch_online --sequence_records_json file."""
    rows = {Path(r["seq_path"]).stem: r for r in json.loads(path.read_text())["datasets"][DS]}
    out = {}
    for key, name in (("obj", "obj_trans_err_fusion_published_cm"), ("hoi", "hoi_err_fusion_cm")):
        vals = []
        for f in files:
            v = rows[f].get(name)
            vals.append(float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else np.nan)
        out[key] = np.asarray(vals)
    return out


def metric_table(files, gt, probs, rec, W):
    """Point estimates and bootstrap draws of every reported metric for one variant."""
    m, d = {}, {}
    stats, hard_neg = frame_stats(files, gt, probs)
    for c, name in enumerate(CHANNELS):
        for k in ("f1", "precision", "recall"):
            m[f"{k}_{name}"], d[f"{k}_{name}"] = seq_mean(stats[k][:, c], W)
        m[f"ap_{name}"], d[f"ap_{name}"] = pooled_ap(files, gt, probs, c, W)
    m["hard_neg_fp"], d["hard_neg_fp"] = seq_mean(hard_neg, W)
    counts, segs = event_counts(files, gt, probs)
    for k, key in enumerate(ENDPOINT_KEYS):
        m[f"ev_{key}"] = float(f1_counts(counts[:, k].sum(0)))
        d[f"ev_{key}"] = f1_counts(np.einsum("bs,sk->bk", W, counts[:, k]))
    m["ev_hand"] = float(np.mean([m["ev_hand_onset"], m["ev_hand_offset"]]))
    d["ev_hand"] = (d["ev_hand_onset"] + d["ev_hand_offset"]) / 2
    m["ev_obj"] = float(np.mean([m["ev_obj_onset"], m["ev_obj_offset"]]))
    d["ev_obj"] = (d["ev_obj_onset"] + d["ev_obj_offset"]) / 2
    m["overseg_hand"] = float(segs[:, 0].sum() / max(segs[:, 1].sum(), 1))
    bs = W @ segs
    d["overseg_hand"] = bs[:, 0] / np.maximum(bs[:, 1], 1)
    if rec is not None:
        for key in ("obj", "hoi"):
            m[key], d[key] = seq_mean(rec[key], W)
    return m, d


def check_against_eval(name, m, eval_json):
    """The recomputed frame F1 must reproduce the evaluator's numbers."""
    res = json.loads(eval_json.read_text())["results"][DS]
    pairs = (("f1_lhand", "contact_f1_lhand"), ("f1_rhand", "contact_f1_rhand"), ("f1_obj", "contact_f1_obj"),
             ("hard_neg_fp", "contact_hard_neg_fp_rate_any"), ("obj", "obj_trans_err_fusion_refine"),
             ("hoi", "hoi_err_fusion"))
    for ours, theirs in pairs:
        if ours in m and not math.isclose(m[ours], res[theirs], rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"{name}: recomputed {ours}={m[ours]:.6f} != eval {theirs}={res[theirs]:.6f}")
    return res


def ci(draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def fmt_delta(value, draws, scale=1.0, digits=1):
    lo, hi = ci(draws * scale)
    star = "" if lo <= 0 <= hi else "*"
    return f"{value * scale:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]{star}"


MAIN_COLUMNS = (  # key, header, scale, digits, lower is better
    ("f1_lhand", "L-hand F1", 100, 1, False),
    ("f1_rhand", "R-hand F1", 100, 1, False),
    ("f1_obj", "Obj-motion F1", 100, 1, False),
    ("ev_hand", "Hand event F1 (±10 f)", 100, 1, False),
    ("ev_obj", "Obj event F1 (±10 f)", 100, 1, False),
    ("obj", "Object err (cm)", 1, 2, True),
    ("hoi", "HOI err (cm)", 1, 2, True),
)
EXTRA_COLUMNS = (
    ("ap_lhand", "L-hand AP", 100, 1, False),
    ("ap_rhand", "R-hand AP", 100, 1, False),
    ("ap_obj", "Obj-motion AP", 100, 1, False),
    ("precision_lhand", "L-hand P", 100, 1, False),
    ("recall_lhand", "L-hand R", 100, 1, False),
    ("precision_rhand", "R-hand P", 100, 1, False),
    ("recall_rhand", "R-hand R", 100, 1, False),
    ("hard_neg_fp", "Hard-neg FP", 100, 1, True),
    ("ev_hand_onset", "Hand onset F1", 100, 1, False),
    ("ev_hand_offset", "Hand release F1", 100, 1, False),
    ("ev_obj_onset", "Obj start F1", 100, 1, False),
    ("ev_obj_offset", "Obj stop F1", 100, 1, False),
    ("overseg_hand", "Hand over-seg.", 1, 3, None),
)


def table(rows, columns, deltas):
    head = "| Variant | " + " | ".join(c[1] for c in columns) + " |"
    lines = [head, "|---" * (len(columns) + 1) + "|"]
    for name, label, m in rows:
        cells = []
        for key, _, scale, digits, _ in columns:
            cells.append(f"{m[key] * scale:.{digits}f}" if key in m else "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if deltas:
        lines += ["", "Difference to the full row, 95 % paired sequence-bootstrap CI (* = excludes 0):", "", head,
                  "|---" * (len(columns) + 1) + "|"]
        for name, label, dm in deltas:
            cells = [fmt_delta(dm[key][0], dm[key][1], scale, digits) if key in dm else "–"
                     for key, _, scale, digits, _ in columns]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def plot(deltas, path, title):
    panels = (("f1_lhand", "L-hand F1 (pp, ↑)", 100), ("f1_rhand", "R-hand F1 (pp, ↑)", 100),
              ("f1_obj", "Obj-motion F1 (pp, ↑)", 100), ("ev_hand", "Hand event F1 ±10 f (pp, ↑)", 100),
              ("obj", "Object error (cm, ↓)", 1), ("hoi", "HOI error (cm, ↓)", 1))
    labels = [label for _, label, _ in deltas]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.55 * len(panels), 0.42 * len(labels) + 1.4),
                             sharey=True, facecolor=SURFACE)
    y = np.arange(len(labels))[::-1]
    for ax, (key, panel_title, scale) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=INK2, lw=1)
        for yi, (name, _, dm) in zip(y, deltas):
            if key not in dm:
                continue
            value, draws = dm[key]
            lo, hi = ci(draws * scale)
            color = INK2 if name in SUPPLEMENTARY else ACCENT
            ax.plot([lo, hi], [yi, yi], color=color, lw=2, solid_capstyle="round")
            ax.plot(value * scale, yi, "o", ms=6, color=color, mec=SURFACE, mew=1.5)
        if any(name in SUPPLEMENTARY for name, _, _ in deltas):
            ax.axhline(len(deltas) - sum(name not in SUPPLEMENTARY for name, _, _ in deltas) - 0.5,
                       color=GRID, lw=1, ls="--")
        ax.set_title(panel_title, fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9, color=INK)
    fig.suptitle(f"Change vs. the full model, {title.split(' (')[0]} (retrained variants; 95 % paired bootstrap CI)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    global DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SETUPS), default="omomo")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    DS = args.dataset
    setup = SETUPS[DS]
    abl, out_dir = setup["abl"], args.out or setup["abl"]

    files = sorted(p.stem for p in setup["data"].glob("*.pt"))
    gt = load_gt(files)
    W = A1.boot_weights(len(files), seed=0)
    rows, results, draws, available = [], {}, {}, []
    for name, label in VARIANTS:
        vdir = abl / name
        eval_json = vdir / "eval_dump.json"
        if not ((vdir / "probs").is_dir() and (vdir / "records.json").is_file() and eval_json.is_file()):
            rows.append((name, label + " (pending)", {}))
            continue
        probs = load_probs(vdir / "probs", files)
        m, d = metric_table(files, gt, probs, records(vdir / "records.json", files), W)
        res = check_against_eval(name, m, eval_json)
        m["obj_jitter"], m["mpjpe"] = res["obj_jitter"], res["mpjpe"]
        rows.append((name, label, m))
        results[name], draws[name] = m, d
        available.append((name, label))
        print(f"{name}: F1 {m['f1_lhand']:.4f}/{m['f1_rhand']:.4f}/{m['f1_obj']:.4f}  obj {m['obj']:.2f}  "
              f"hoi {m['hoi']:.2f}")
    if REFERENCE in results:  # the full row is the published pipeline
        check_against_eval("full vs published", results[REFERENCE], setup["published"])

    deltas = []
    if REFERENCE in draws:
        for name, label in available:
            if name == REFERENCE:
                continue
            deltas.append((name, label, {k: (results[name][k] - results[REFERENCE][k],
                                             draws[name][k] - draws[REFERENCE][k]) for k in draws[name]}))

    title = {"omomo": "OMOMO (2026-10-30)", "behave": "BEHAVE (2026-10-31)", "imhd": "IMHD (2026-11-01)",
             "pahoi": "PAHOI (2026-11-01)", "hodome": "HODome (2026-11-01)"}[DS]
    lines = [
        f"# VC module ablation on {title}", "",
        "Full = the published IMUHOI++ pipeline (VC v5ar + OT v2_armrot_biclose; training is deterministic, a",
        "re-run on OMOMO reproduced its checkpoints bit for bit). Every other row retrains its own VC (the recipe of",
        "the published VC, seed 42), OT and joint stage with the published HP frozen; only the `vc_ablation_*` keys",
        "differ (`model/rnn/velocity_contact_v5.py`).",
        "w/o HP prior removes the pose prior together with every other v5ar addition except the conditional",
        "contact factorization (coupling features, bidirectional object-motion head, initializer perturbation),",
        "leaving IMU-HOI's contact evidence plus p_s = p_m * p_s|m; w/o pose prior only (supplementary) removes the",
        "pose prior alone. Body-group rows remove one group from the pose prior (palm dynamics kept, as in A2).",
        f"Scored online with the unified protocol (HP smoothing 7, diff-3 velocity init) on the {len(files)} test",
        "sequences, best joint checkpoint, seed-42 Noitom test noise. F1 at the pipeline thresholds (hand 0.15,",
        "object motion 0.5), per-sequence means; AP pools all published frames; event F1 matches GT and predicted",
        "onsets / releases within ±10 frames in frames [90, T-30). Values in percent except errors (cm) and",
        "over-segmentation. The sequence bootstrap does not cover training-seed variation (one seed per row).", "",
        "## Main table", "", *table(rows, MAIN_COLUMNS, deltas), "",
        "## Threshold-free, precision / recall, event details", "", *table(rows, EXTRA_COLUMNS, deltas), "",
        "## Sources", "",
        f"- runs: `{abl.relative_to(ROOT)}/<variant>/` (VC `vc/`, pipeline `pipeline/`, per-frame `probs/`, "
        "`records.json`, `eval_dump.json`)",
        f"- published pipeline (full row): `{setup['published'].relative_to(ROOT)}`",
        "- script: `scripts/vc_probe/summarize_vc_ablation.py`",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"rows": {name: m for name, _, m in rows if m},
         "deltas_vs_full": {name: {k: {"value": v[0], "ci95": ci(v[1])} for k, v in dm.items()}
                            for name, _, dm in deltas}}, indent=1))
    if deltas:
        plot(deltas, out_dir / "ablation_deltas.png", title)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
