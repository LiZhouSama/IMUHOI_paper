#!/usr/bin/env python3
"""Per-hand breakdown of FK-proposal / published HOI error from two OT dumps (baseline vs grasp-latent).

    python scripts/diag/eag_dump_breakdown.py p800=dump.pt eag=dump.pt [--on 0.5 --off 0.25]

On GT-contact frames of each hand: HOI error of the hand's FK proposal, of the published object and (eag) of the
state-free prior, split by the reconstructed grasp-latent activity (VC contact-prob hysteresis) and by frames since
the GT onset; mean gate weights; FK-proposal jitter.  All errors in cm, pooled over frames.
"""
import argparse
import numpy as np
import torch


def hysteresis(prob, on, off):
    active = np.zeros(len(prob), dtype=bool)
    prev = False
    for t, p in enumerate(prob):
        prev = (p > off) if prev else (p > on)
        active[t] = prev
    return active


def frames_since_onset(contact):
    out = np.full(len(contact), -1)
    run = -1
    for t, c in enumerate(contact):
        run = run + 1 if (c and run >= 0) else (0 if c else -1)
        out[t] = run
    return out


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)["records"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--on", type=float, default=0.5)
    ap.add_argument("--off", type=float, default=0.25)
    args = ap.parse_args()
    bins = [(0, 0), (1, 15), (16, 30), (31, 60), (61, 120), (121, 10**9)]
    for spec in args.dumps:
        label, _, path = spec.partition("=")
        recs = load(path)
        rows = []  # per frame-hand: [active, since_onset, e_fk, e_pub, e_prior, w_fk, w_imu, w_static, jitter_fk]
        for r in recs:
            if not r.get("has_object", True):
                continue
            gt_p = r["gt_obj_trans"].numpy(); pub = r["pred_obj_trans_published"].numpy()
            hands_in = r["ot_hand_positions"].numpy(); gt_h = r["gt_palm_pos"].numpy()
            prob = r["pred_hand_contact_prob"].numpy(); w = r["gating_weights"].numpy()
            for h, (ck, fk, pk) in enumerate((("gt_lhand_contact", "pred_lhand_obj_trans", "prior_lhand_obj_trans"),
                                              ("gt_rhand_contact", "pred_rhand_obj_trans", "prior_rhand_obj_trans"))):
                c = r[ck].numpy().astype(bool)
                if not c.any():
                    continue
                fkp = r[fk].numpy()
                prior = r[pk].numpy() if pk in r else None
                rel_gt = gt_p - gt_h[:, h]
                e_fk = np.linalg.norm((fkp - hands_in[:, h]) - rel_gt, axis=-1) * 100
                e_pub = np.linalg.norm((pub - hands_in[:, h]) - rel_gt, axis=-1) * 100
                e_pr = np.linalg.norm((prior - hands_in[:, h]) - rel_gt, axis=-1) * 100 if prior is not None else np.full(len(c), np.nan)
                jit = np.full(len(c), np.nan)
                if len(c) > 2:
                    jit[1:-1] = np.linalg.norm(fkp[2:] - 2 * fkp[1:-1] + fkp[:-2], axis=-1) * 1000
                act = hysteresis(prob[:, h], args.on, args.off)
                since = frames_since_onset(c)
                idx = np.where(c)[0]
                rows.append(np.stack((act[idx], since[idx], e_fk[idx], e_pub[idx], e_pr[idx], w[idx, h], w[idx, 2], w[idx, 3], jit[idx]), axis=1))
        m = np.concatenate(rows, 0)
        act = m[:, 0] > 0.5
        print(f"===== {label}: {len(m)} GT-contact hand-frames; latent-active on {act.mean()*100:.1f}% of them")
        print("  frames          |   n     | FK HOI | pub HOI | prior HOI | w_fk  w_imu  w_static | FK jitter")
        def line(name, sel):
            s = m[sel]
            if len(s) == 0:
                return
            print(f"  {name:16s}| {len(s):7d} | {s[:,2].mean():6.2f} | {s[:,3].mean():7.2f} | {np.nanmean(s[:,4]):9.2f} | {s[:,5].mean():.2f}  {s[:,6].mean():.2f}   {s[:,7].mean():.2f}    | {np.nanmean(s[:,8]):6.2f}")
        line("all", np.ones(len(m), bool)); line("active", act); line("inactive", ~act)
        for lo, hi in bins:
            sel = (m[:, 1] >= lo) & (m[:, 1] <= hi)
            line(f"onset+{lo}-{min(hi, 999)}", sel)
        for lo, hi in bins:
            sel = (m[:, 1] >= lo) & (m[:, 1] <= hi) & act
            line(f" act {lo}-{min(hi, 999)}", sel)


if __name__ == "__main__":
    main()
