#!/usr/bin/env python3
"""Diagnostic open-loop v2 smoothing replay; not a closed-loop evaluation or tuning set.

Keeps total FK mass exactly unchanged. Filters object-frame bones causally,
resetting on predicted contact onset; optionally filters the left/right split
only while both predicted contacts persist. Ground truth is used only in metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix


def filter_candidates(bones, prob, weights, alpha=1., adaptive=False, split_alpha=1.):
    out = bones.copy()
    w = weights.copy()
    active = np.zeros(2, dtype=bool)
    previous_active = active.copy()
    velocity = np.zeros((2, 3))
    ratio_prev = .5
    for t in range(len(bones)):
        previous_active = active.copy()
        active = np.where(active, prob[t, :2] >= .3, prob[t, :2] >= .7)
        if t:
            velocity = .25 * ((bones[t] - bones[t-1]) * 30) + .75 * velocity
            gain = np.full(2, alpha)
            if adaptive:
                cutoff = 1. + 3. * np.linalg.norm(velocity, axis=-1)
                gain = 1. / (1. + 30. / (2 * np.pi * cutoff))
            keep = active & previous_active
            out[t] = np.where(keep[:, None], gain[:, None] * bones[t] + (1-gain[:, None]) * out[t-1], bones[t])
        mass = weights[t, :2].sum()
        ratio = weights[t, 0] / max(mass, 1e-12)
        if t and active.all() and previous_active.all():
            ratio = split_alpha * ratio + (1-split_alpha) * ratio_prev
        ratio_prev = ratio
        w[t, 0], w[t, 1] = mass * ratio, mass * (1-ratio)
    assert np.allclose(w[:, :2].sum(-1), weights[:, :2].sum(-1), atol=1e-12)
    return out, w


def filter_world_candidates(fk, prob, alpha, beta=0.):
    """Causal alpha-beta position filter. Reset on contact-state transitions.

    beta=0 uses plain EMA (no velocity extrapolation). Positive beta estimates
    candidate velocity from innovations, without consuming VC velocity or GT.
    """
    result = fk.copy()
    velocity = np.zeros_like(fk[0])
    active = np.zeros(2, dtype=bool)
    for t in range(len(fk)):
        before = active.copy()
        active = np.where(active, prob[t, :2] >= .3, prob[t, :2] >= .7)
        if not t:
            continue
        keep = active & before
        predicted = result[t-1] + (velocity / 30 if beta else 0.)
        residual = fk[t] - predicted
        result[t] = np.where(keep[:, None], predicted + alpha * residual, fk[t])
        velocity = np.where(keep[:, None], velocity + beta * 30 * residual, 0.)
    return result


def replay(fk, increment, w, initial):
    out = np.empty_like(increment)
    previous = initial.copy()
    for t in range(len(w)):
        previous = (w[t, :2, None] * fk[t]).sum(0) + w[t, 2] * (previous + increment[t]) + w[t, 3] * previous
        out[t] = previous
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dump')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    recipes = {'raw': {}, 'bone_a05': {'alpha': .5}, 'bone_a025': {'alpha': .25},
               'bone_adaptive': {'adaptive': True}, 'split_a025': {'split_alpha': .25},
               'adaptive_split': {'adaptive': True, 'split_alpha': .25},
               'rotation_a05': {'rotation_alpha': .5},
               'bone_rotation_a05': {'alpha': .5, 'rotation_alpha': .5},
               'world_ema05': {'world_alpha': .5},
               'world_ema025': {'world_alpha': .25},
               'world_ab05': {'world_alpha': .5, 'world_beta': .1},
               'world_ab035': {'world_alpha': .35, 'world_beta': .05}}
    acc = {k: {m: [] for m in ('obj', 'hoi', 'jitter', 'contact_jitter', 'boundary_hoi', 'fk_mass', 'candidate_shift_cm', 'moving_hoi')} for k in recipes}
    max_deviation = 0.
    for r in torch.load(args.dump, map_location='cpu')['records']:
        n = int(r['publish_len'])
        if not r.get('has_object', True) or n < 3:
            continue
        arr = lambda k: np.asarray(r[k][:n], dtype=np.float64)
        rotation = rotation_6d_to_matrix(torch.as_tensor(arr('gt_obj_imu')[:, 3:9])).numpy()
        raw_hand = arr('stage1_pred_palm_glb_pos') if 'stage1_pred_palm_glb_pos' in r else arr('pred_palm_glb_pos')
        hand = arr('pred_palm_glb_pos')
        world_bones = np.stack([arr('pred_lhand_obj_trans'), arr('pred_rhand_obj_trans')], 1) - raw_hand
        bones = np.einsum('tji,thj->thi', rotation, world_bones)
        gt, gt_hand = arr('gt_obj_trans'), arr('gt_palm_pos_eval') if 'gt_palm_pos_eval' in r else arr('gt_palm_pos')
        contact = np.stack([arr('gt_lhand_contact'), arr('gt_rhand_contact')], -1).astype(bool)
        boundary = np.zeros_like(contact)
        for h in range(2):
            for t in np.flatnonzero(contact[1:, h] != contact[:-1, h]) + 1:
                boundary[max(0, t-3):min(n, t+4), h] = True
        stable = (contact[2:] & contact[1:-1] & contact[:-2]).any(-1)
        w = arr('gating_weights')
        for name, spec in recipes.items():
            filter_spec = dict(spec)
            rotation_alpha = filter_spec.pop('rotation_alpha', 1.)
            world_alpha = filter_spec.pop('world_alpha', 1.)
            world_beta = filter_spec.pop('world_beta', 0.)
            used_rotation = rotation
            if rotation_alpha < 1:
                # Diagnostic causal 6D orientation filter, no GT and no extra lookahead.
                rot6d = np.concatenate((rotation[:, 0], rotation[:, 1]), axis=-1)
                for t in range(1, n):
                    rot6d[t] = rotation_alpha * rot6d[t] + (1-rotation_alpha) * rot6d[t-1]
                used_rotation = rotation_6d_to_matrix(torch.as_tensor(rot6d)).numpy()
            filtered, ww = filter_candidates(bones, arr('pred_hand_contact_prob'), w, **filter_spec)
            fk = hand + np.einsum('tij,thj->thi', used_rotation, filtered)
            fk = filter_world_candidates(fk, arr('pred_hand_contact_prob'), world_alpha, world_beta) if world_alpha < 1 else fk
            p = replay(fk, arr('pred_imu_obj_trans')-arr('pred_static_obj_trans'), ww, arr('pred_static_obj_trans')[0])
            if name == 'raw':
                max_deviation = max(max_deviation, float(np.abs(p-arr('pred_obj_trans_published')).max()))
            a = acc[name]
            a['obj'].append(np.linalg.norm(p-gt, axis=-1).mean()*100)
            err = np.linalg.norm((p[:, None]-hand)-(gt[:, None]-gt_hand), axis=-1)*100
            if contact.any():
                a['candidate_shift_cm'].append(np.linalg.norm(fk-(hand+world_bones), axis=-1)[contact].mean()*100)
                a['hoi'].append(err[contact].mean())
                speed = np.concatenate(([0.], np.linalg.norm(np.diff(gt, axis=0), axis=-1)*30))
                moving = contact & (speed[:, None] > .6)
                if moving.any():
                    a['moving_hoi'].append(err[moving].mean())
                a['fk_mass'].append(np.broadcast_to(ww[:, :2].sum(-1)[:, None], contact.shape)[contact].mean())
            if boundary.any():
                a['boundary_hoi'].append(err[boundary].mean())
            jitter = np.linalg.norm(p[2:]-2*p[1:-1]+p[:-2], axis=-1)*1000
            a['jitter'].append(jitter.mean())
            if stable.any():
                a['contact_jitter'].append(jitter[stable].mean())
    assert max_deviation < 1e-4, f'Baseline publication mismatch: {max_deviation} m'
    result = {'note': 'Open-loop, sequence-equal diagnostic metrics; cached feedback/VC/refine unchanged.',
              'baseline_max_deviation_m': max_deviation,
              'results': {k: {m: float(np.mean(v)) for m, v in a.items()} for k, a in acc.items()}}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
