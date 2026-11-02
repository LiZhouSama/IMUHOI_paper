#!/usr/bin/env python3
"""Does the VC's pose prior hold a few pose prototypes for "interacting" and "about to interact"?

Frames on sliding-window frames get a GT interaction state (any hand):
  idle (no contact within 30 frames ahead, none in the 15 frames behind), approach (onset in 11-30 frames),
  pre-contact (onset in 1-10 frames), contact-start (first 15 frames of contact), holding (later contact frames),
  post-release (contact ended within the last 15 frames).
An equal number of frames per state is clustered (standardize -> PCA-32 -> k-means, k=8) in four representations:
b_t of the IMUHOI++ VC, its HP pose input, an untrained b_t and the IMUHOI contact LSTM. For b_t the clusters are
drawn as mean GT body poses (heading-aligned joint positions) with their state make-up and the network's published
contact / anticipation probability. Separately, the non-contact frames on which the anticipation head fires most
(top 1 %) are grouped by GT pose (k=4) and drawn.

    python scripts/vc_probe/analyze_pose_prototypes.py --dataset behave
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
import torch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_contact_events as A  # noqa: E402

PILOT = A.PILOT_ROOT
STATES = ("idle", "approach", "pre-contact", "contact-start", "holding", "post-release")
PER_STATE = 3000
K = 8
TOP_FRAC = 0.01
REPS = {"ours_b": ("ours", "feat_b", "IMUHOI++ pose prior b_t"), "ours_in": ("ours", "feat_in", "HP pose input"),
        "ours_rand_b": ("ours", "rand_b", "untrained b_t"), "ours_hc": ("ours", "feat_hc", "IMUHOI++ hand-contact LSTM"),
        "ours_om": ("ours", "feat_om", "IMUHOI++ object-motion LSTM"), "imuhoi_hc": ("imuhoi", "feat_hc", "IMUHOI contact LSTM")}
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 23}
SIDE_COLORS = {"left": "#2a78d6", "right": "#eb6834", "mid": "#52514e"}
STATE_COLORS = ("#8a8984", "#c3c2b7", "#2a78d6", "#1baf7a", "#eda100", "#eb6834")


def heading_poses(data):
    """Joint positions in the pelvis heading frame: pelvis on the vertical axis, facing +z, height above the floor."""
    pos = data["position_global_full_gt_world"].numpy().astype(np.float64)
    root = data["rotation_global"][:, 0].numpy().astype(np.float64)
    forward = root[:, :, 2]
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    c, s = np.cos(-yaw), np.sin(-yaw)
    rel = pos - np.stack([pos[:, 0, 0], np.zeros(len(pos)), pos[:, 0, 2]], axis=-1)[:, None, :]
    x = c[:, None] * rel[..., 0] + s[:, None] * rel[..., 2]
    z = -s[:, None] * rel[..., 0] + c[:, None] * rel[..., 2]
    floor = np.percentile(pos[:, [10, 11], 1], 2)  # toes; keeps squatting visible
    return np.stack([x, rel[..., 1] - floor, z], axis=-1)


def frame_states(contact, T_pub):
    """GT interaction state of every frame (index into STATES, -1 = not assigned)."""
    T = contact.shape[0]
    states = np.full(T, -1)
    starts = np.flatnonzero(np.diff(contact.astype(np.int8)) == 1) + 1
    ends = np.flatnonzero(np.diff(contact.astype(np.int8)) == -1)  # last contact frame
    t = np.arange(T)
    nxt = np.searchsorted(starts, t, side="right")
    lead = np.where(nxt < len(starts), starts[np.minimum(nxt, max(len(starts) - 1, 0))] - t, 10 ** 6) if len(starts) \
        else np.full(T, 10 ** 6)
    prv = np.searchsorted(ends, t, side="left") - 1
    lag = np.where(prv >= 0, t - ends[np.maximum(prv, 0)], 10 ** 6) if len(ends) else np.full(T, 10 ** 6)
    seg_start = np.zeros(T, dtype=int)
    for i in range(1, T):
        seg_start[i] = seg_start[i - 1] if contact[i] and contact[i - 1] else i
    off = ~contact
    states[off & (lead > 30) & (lag > 15)] = 0
    states[off & (lead >= 11) & (lead <= 30)] = 1
    states[off & (lead >= 1) & (lead <= 10)] = 2
    states[off & (lag >= 1) & (lag <= 15) & (lead > 10)] = 5
    since = t - seg_start
    states[contact & (since < 15)] = 3
    states[contact & (since >= 15)] = 4
    valid = (t >= A.WARMUP) & (t < T_pub) & (t + 30 < T)
    states[~valid] = -1
    return states


def load(ds):
    files = sorted(p.stem for p in (PILOT / f"{ds}_ours" / "features").glob("*.pt"))
    per_file = []
    for f in files:
        data = torch.load(A.DATA[ds] / f"{f}.pt", map_location="cpu", weights_only=False)
        ours = torch.load(PILOT / f"{ds}_ours" / "features" / f"{f}.pt", map_location="cpu", weights_only=False)
        imu = torch.load(PILOT / f"{ds}_imuhoi" / "features" / f"{f}.pt", map_location="cpu", weights_only=False)
        a1 = torch.load(A.EVENT_ROOT / f"{ds}_v5ar" / "probs" / f"{f}.pt", map_location="cpu", weights_only=False)
        T_pub = ours["feat_b"].shape[0]
        contact = (data["lhand_contact"].numpy() | data["rhand_contact"].numpy()).astype(bool)
        lead_any = np.full(contact.shape[0], 10 ** 6)
        starts = np.flatnonzero(np.diff(contact.astype(np.int8)) == 1) + 1
        if len(starts):
            nxt = np.searchsorted(starts, np.arange(contact.shape[0]), side="right")
            lead_any = np.where(nxt < len(starts), starts[np.minimum(nxt, len(starts) - 1)] - np.arange(contact.shape[0]), 10 ** 6)
        per_file.append({
            "file": f, "states": frame_states(contact, T_pub), "contact": contact, "lead": lead_any,
            "pose": heading_poses(data),
            "rep": {k: (ours if src == "ours" else imu)[key].numpy() for k, (src, key, _) in REPS.items()},
            "p": ours["pred_hand_contact_prob"].numpy()[:, :2].max(1),
            "ant": a1["pred_contact_anticipation_prob"].numpy()[:, [0, 2]].max(1),
            "T_pub": T_pub})
    return per_file


def stratified_sample(per_file, rng):
    pool = [[] for _ in STATES]
    for i, d in enumerate(per_file):
        for s in range(len(STATES)):
            pool[s].extend((i, int(t)) for t in np.flatnonzero(d["states"] == s))
    sample = []
    for s, frames in enumerate(pool):
        take = min(PER_STATE, len(frames))
        for j in rng.choice(len(frames), take, replace=False):
            sample.append((*frames[j], s))
    return np.array(sample), {STATES[s]: len(frames) for s, frames in enumerate(pool)}


def cluster(per_file, sample, key):
    X = np.stack([per_file[i]["rep"][key][t].astype(np.float32) for i, t, _ in sample])
    Z = PCA(n_components=32, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    return KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Z)


def cluster_table(per_file, sample, labels):
    states = sample[:, 2]
    rows = []
    for c in range(K):
        m = labels == c
        comp = np.bincount(states[m], minlength=len(STATES)) / max(m.sum(), 1)
        rows.append({"cluster": c, "n": int(m.sum()), "states": comp.tolist(),
                     "p_contact": float(np.mean([per_file[i]["p"][t] for i, t, _ in sample[m]])),
                     "anticipation": float(np.mean([per_file[i]["ant"][t] for i, t, _ in sample[m]])),
                     "pose": np.mean([per_file[i]["pose"][t] for i, t, _ in sample[m]], axis=0).tolist()})
    return rows


def draw_skeleton(ax, pose, view):
    pose = np.asarray(pose)
    h = pose[:, 0] if view == "front" else pose[:, 2]
    v = pose[:, 1]
    for j, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        side = "left" if j in LEFT else "right" if j in RIGHT else "mid"
        ax.plot([h[parent], h[j]], [v[parent], v[j]], color=SIDE_COLORS[side], linewidth=2, solid_capstyle="round")
    ax.scatter(h, v, s=6, color=A.INK, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.05, 1.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.axhline(pose[:, 1].min(), color=A.GRID, linewidth=0.8, zorder=0)


def plot_prototypes(rows, path, title):
    order = sorted(range(K), key=lambda c: -(rows[c]["states"][2] + rows[c]["states"][3]))
    fig, axes = plt.subplots(2, K, figsize=(1.55 * K, 5.2), facecolor=A.SURFACE)
    for col, c in enumerate(order):
        r = rows[c]
        for row, view in enumerate(("front", "side")):
            ax = axes[row, col]
            ax.set_facecolor(A.SURFACE)
            draw_skeleton(ax, np.asarray(r["pose"]), view)
        top = np.argsort(r["states"])[::-1][:2]
        axes[0, col].set_title(f"cluster {c} (n={r['n']})\n" + "\n".join(f"{STATES[s]} {r['states'][s]:.0%}" for s in top)
                               + f"\np(contact) {r['p_contact']:.2f}", fontsize=6.5, color=A.INK)
    axes[0, 0].set_ylabel("front", fontsize=7, color=A.INK)
    axes[1, 0].set_ylabel("side (facing right)", fontsize=7, color=A.INK)
    fig.suptitle(title, fontsize=9, color=A.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, facecolor=A.SURFACE)
    plt.close(fig)


def plot_composition(tables, path, title):
    fig, axes = plt.subplots(1, len(tables), figsize=(2.4 * len(tables), 3.3), facecolor=A.SURFACE, sharey=True)
    for ax, (key, rows) in zip(axes, tables.items()):
        A.style_axes(ax)
        order = sorted(range(K), key=lambda c: -(rows[c]["states"][2] + rows[c]["states"][3]))
        bottom = np.zeros(K)
        for s in range(len(STATES)):
            vals = np.array([rows[c]["states"][s] for c in order])
            ax.bar(np.arange(K), vals, bottom=bottom, color=STATE_COLORS[s], edgecolor=A.SURFACE, linewidth=1,
                   label=STATES[s])
            bottom += vals
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([str(c) for c in order], fontsize=7)
        ax.set_title(REPS[key][2], fontsize=8, color=A.INK, loc="left")
        ax.set_xlabel("cluster", fontsize=7, color=A.INK)
    axes[0].set_ylabel("share of frames in the cluster", fontsize=7.5, color=A.INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=len(STATES), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=9, color=A.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(path, dpi=180, facecolor=A.SURFACE)
    plt.close(fig)


def anticipation_prototypes(per_file, rng):
    frames = [(i, int(t)) for i, d in enumerate(per_file) for t in range(A.WARMUP, d["T_pub"])
              if not d["contact"][t] and t + 30 < d["contact"].shape[0]]
    ant = np.array([per_file[i]["ant"][t] for i, t in frames])
    thr = float(np.quantile(ant, 1 - TOP_FRAC))
    top = [fr for fr, a in zip(frames, ant) if a >= thr]
    lead = np.array([per_file[i]["lead"][t] for i, t in top])
    poses = np.stack([per_file[i]["pose"][t] for i, t in top])
    labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(poses.reshape(len(poses), -1))
    idle = [fr for fr in frames if per_file[fr[0]]["states"][fr[1]] == 0]
    idle_pick = [idle[j] for j in rng.choice(len(idle), min(5000, len(idle)), replace=False)]
    return {"threshold": thr, "n_top": len(top), "precision_onset_15": float(np.mean(lead <= 15)),
            "base_rate_onset_15": float(np.mean([per_file[i]["lead"][t] <= 15 for i, t in frames])),
            "clusters": [{"n": int((labels == c).sum()), "onset_within_15": float(np.mean(lead[labels == c] <= 15)),
                          "pose": poses[labels == c].mean(0).tolist()} for c in range(4)],
            "idle_pose": np.mean([per_file[i]["pose"][t] for i, t in idle_pick], axis=0).tolist()}


def plot_anticipation(res, path, title):
    panels = [("idle (reference)", res["idle_pose"], None)] + [
        (f"group {c} (n={g['n']})\nonset <= 15 f: {g['onset_within_15']:.0%}", g["pose"], None)
        for c, g in enumerate(res["clusters"])]
    fig, axes = plt.subplots(2, len(panels), figsize=(1.7 * len(panels), 5.0), facecolor=A.SURFACE)
    for col, (name, pose, _) in enumerate(panels):
        for row, view in enumerate(("front", "side")):
            axes[row, col].set_facecolor(A.SURFACE)
            draw_skeleton(axes[row, col], np.asarray(pose), view)
        axes[0, col].set_title(name, fontsize=6.8, color=A.INK)
    fig.suptitle(title, fontsize=9, color=A.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=180, facecolor=A.SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("omomo", "behave"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ds = args.dataset
    out_dir = args.out or PILOT / f"analysis_{ds}_prototypes"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    per_file = load(ds)
    check = per_file[0]["pose"]
    lr = float(np.mean(check[:, 16, 0] - check[:, 17, 0]))  # left minus right shoulder along x
    if lr <= 0:
        raise ValueError(f"heading frame check failed: left-right shoulder x {lr:.3f} (expected > 0)")
    sample, available = stratified_sample(per_file, rng)
    tables, ami = {}, {}
    for key in REPS:
        labels = cluster(per_file, sample, key)
        tables[key] = cluster_table(per_file, sample, labels)
        ami[key] = float(adjusted_mutual_info_score(sample[:, 2], labels))
        print(f"[cluster] {key}: AMI with states {ami[key]:.3f}", flush=True)
    ant = anticipation_prototypes(per_file, rng)

    label = ds.upper()
    plot_prototypes(tables["ours_b"], out_dir / "b_t_cluster_poses.png",
                    f"{label}: k-means clusters of the IMUHOI++ pose prior b_t, drawn as mean GT poses "
                    "(sorted by pre-contact + contact-start share)")
    plot_composition(tables, out_dir / "cluster_state_composition.png",
                     f"{label}: interaction-state make-up of k-means clusters ({PER_STATE} frames per state at most; "
                     "clusters sorted by pre-contact + contact-start share)")
    plot_anticipation(ant, out_dir / "anticipation_top_poses.png",
                      f"{label}: poses on which the anticipation head fires most (top {TOP_FRAC:.0%} of non-contact frames)")
    results = {"dataset": ds, "available_frames": available, "sampled_per_state": PER_STATE, "k": K, "ami": ami,
               "clusters": {k: [{kk: vv for kk, vv in r.items() if kk != "pose"} for r in rows] for k, rows in tables.items()},
               "anticipation": {**{k: v for k, v in ant.items() if k not in ("clusters", "idle_pose")},
                                "clusters": [{k: v for k, v in g.items() if k != "pose"} for g in ant["clusters"]]}}
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    write_summary(results, out_dir / "summary.md", label)


def write_summary(results, path, label):
    lines = [f"# Pose prototypes of interaction states: {label} test", "",
             f"Frames per state available on sliding-window frames: {results['available_frames']}; at most "
             f"{results['sampled_per_state']} sampled per state; k-means k = {results['k']} on PCA-32.", "",
             "## Alignment of the clusters with the interaction states", "",
             "| representation | AMI with the six states |", "|---|---|"]
    for key, v in results["ami"].items():
        lines.append(f"| {REPS[key][2]} | {v:.3f} |")
    for key in ("ours_b", "ours_in"):
        lines += ["", f"## Clusters of {REPS[key][2]} (state shares, network outputs)", "",
                  "| cluster | n | " + " | ".join(STATES) + " | published p(contact) | anticipation |",
                  "|---|---|" + "---|" * len(STATES) + "---|---|"]
        rows = sorted(results["clusters"][key], key=lambda r: -(r["states"][2] + r["states"][3]))
        for r in rows:
            lines.append(f"| {r['cluster']} | {r['n']} | " + " | ".join(f"{v:.2f}" for v in r["states"])
                         + f" | {r['p_contact']:.2f} | {r['anticipation']:.3f} |")
    a = results["anticipation"]
    lines += ["", f"## Top {TOP_FRAC:.0%} anticipation frames (non-contact)", "",
              f"threshold {a['threshold']:.3f}, {a['n_top']} frames; followed by an onset within 15 frames: "
              f"{a['precision_onset_15']:.2f} (base rate over all non-contact frames {a['base_rate_onset_15']:.2f}).", "",
              "| pose group | n | onset within 15 f |", "|---|---|---|"]
    for c, g in enumerate(a["clusters"]):
        lines.append(f"| {c} | {g['n']} | {g['onset_within_15']:.2f} |")
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
