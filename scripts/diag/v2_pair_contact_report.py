#!/usr/bin/env python3
"""Compare paired v2 dumps on GT single-/dual-contact frames and verify publication.

Published replay uses the filtered effective FK candidates; raw fifth-branch
weights diagnose pair use but are not directly comparable to filtered outputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _np(x):
    return np.asarray(x, dtype=np.float64)


def replay_published(record, n):
    w = _np(record['gating_weights'][:n])
    stage1 = _np(record.get('stage1_pred_palm_glb_pos', record['pred_palm_glb_pos'])[:n])
    refined = _np(record['pred_palm_glb_pos'][:n])
    fk = np.stack((_np(record['pred_lhand_obj_trans'][:n]),
                   _np(record['pred_rhand_obj_trans'][:n])), axis=1)
    corrected = fk + (refined - stage1)
    imu_step = _np(record['pred_imu_obj_trans'][:n]) - _np(record['pred_static_obj_trans'][:n])
    previous = _np(record['pred_static_obj_trans'][0]).copy()
    positions = np.empty((n, 3))
    for t in range(n):
        previous = (w[t, :2, None] * corrected[t]).sum(0) + w[t, 2] * (previous + imu_step[t]) + w[t, 3] * previous
        positions[t] = previous
    return positions


def analyze(path):
    records = torch.load(path, map_location='cpu')['records']
    acc = {k: {'obj': [], 'hoi': [], 'jitter': [], 'pair_mass': []} for k in ('dual', 'single', 'none')}
    counts = {k: 0 for k in acc}
    max_replay_cm = 0.
    max_weight_delta = 0.
    pair_dual_pred = []
    for r in records:
        n = int(r['publish_len'])
        if n < 3 or not r.get('has_object', True):
            continue
        published = _np(r['pred_obj_trans_published'][:n])
        replay = replay_published(r, n)
        max_replay_cm = max(max_replay_cm, np.abs(replay - published).max() * 100)
        mask = np.stack((_np(r['gt_lhand_contact'][:n]), _np(r['gt_rhand_contact'][:n])), axis=-1).astype(bool)
        dual = mask.all(axis=-1)
        single = mask.sum(axis=-1) == 1
        none = ~mask.any(axis=-1)
        gt = _np(r['gt_obj_trans'][:n])
        hand = _np(r['pred_palm_glb_pos'][:n])
        gt_hand = _np(r.get('gt_palm_pos_eval', r['gt_palm_pos'])[:n])
        err = np.linalg.norm((published[:, None] - hand) - (gt[:, None] - gt_hand), axis=-1) * 100
        obj = np.linalg.norm(published - gt, axis=-1) * 100
        jitter = np.linalg.norm(published[2:] - 2 * published[1:-1] + published[:-2], axis=-1) * 1000
        w = _np(r['gating_weights'][:n])
        w5 = _np(r['gating_weights_5'][:n]) if 'gating_weights_5' in r else None
        if w5 is not None:
            folded = np.concatenate((w5[:, :2] + .5*w5[:, 4:5], w5[:, 2:4]), axis=-1)
            max_weight_delta = max(max_weight_delta, np.abs(folded - w).max())
            predicted_contact = _np(r['pred_hand_contact_prob'][:n, :2])
            pair_dual_pred.append((float((w5[:, 4] > .01).sum()), float(((predicted_contact >= .2).all(-1)).sum())))
        for key, select in (('dual', dual), ('single', single), ('none', none)):
            if not select.any():
                continue
            counts[key] += int(select.sum())
            a = acc[key]
            a['obj'].append(float(obj[select].mean()))
            if key != 'none':
                a['hoi'].append(float(err[select[:, None] & mask].mean()))
            smooth_select = select[1:-1] & select[:-2] & select[2:]
            if smooth_select.any():
                a['jitter'].append(float(jitter[smooth_select].mean()))
            if w5 is not None:
                a['pair_mass'].append(float(w5[select, 4].mean()))
    return {'path': str(path), 'sequences': len(records), 'frames': counts,
            'sequence_equal': {key: {metric: (float(np.mean(values)) if values else None)
                                     for metric, values in parts.items()} for key, parts in acc.items()},
            'max_publication_replay_cm': max_replay_cm,
            'max_folded_weight_delta': max_weight_delta,
            'pair_active_frames': sum(x for x, _ in pair_dual_pred),
            'predicted_dual_support_frames': sum(x for _, x in pair_dual_pred)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dumps', nargs='+', help='label=path.pt')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = {label: analyze(Path(path)) for label, path in (item.split('=', 1) for item in args.dumps)}
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload)
    print(payload)


if __name__ == '__main__':
    main()
