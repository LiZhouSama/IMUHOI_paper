#!/usr/bin/env python3
"""Post-lock diagnostics only: never fit a model or choose a hyperparameter."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.rnn.contact_residual_runtime import ContactResidualPredictor


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    payload = torch.load(args.dump, map_location='cpu', weights_only=False)
    predictor = ContactResidualPredictor(args.checkpoint, device='cpu')
    calibration = json.loads(Path(args.calibration).read_text())
    coefficient = float(calibration['lambda'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for record in payload['records']:
        predictor.reset()
        pred = predictor.predict(record)
        base = record['pred_obj_trans_published']
        palm = record['pred_palm_glb_pos']
        gt_obj = record['gt_obj_trans']
        gt_palm = record['gt_palm_pos_eval']
        contact = torch.stack((record['gt_lhand_contact'], record['gt_rhand_contact']), -1).bool()
        base_errors = torch.linalg.vector_norm((base[:, None] - palm) - (gt_obj[:, None] - gt_palm), dim=-1) * 100
        new_errors = torch.linalg.vector_norm((pred[:, None] - palm) - (gt_obj[:, None] - gt_palm), dim=-1) * 100
        age = torch.zeros_like(contact, dtype=torch.long)
        counter = torch.zeros(2, dtype=torch.long)
        for frame in range(len(contact)):
            counter = torch.where(contact[frame], counter + 1, 0)
            age[frame] = (counter - 1).clamp_min(0)
        frames = torch.arange(len(contact))[:, None]
        masks = dict(all=contact, single=contact & (contact.sum(-1) == 1)[:, None],
                     bimanual=contact & contact.all(-1)[:, None],
                     startup=contact & (frames < 60), steady=contact & (frames >= 60),
                     age_0_14=contact & (age < 15), age_15_59=contact & (age >= 15) & (age < 60),
                     age_60_119=contact & (age >= 60) & (age < 120), age_120_plus=contact & (age >= 120))
        phases = {name: dict(hand_frames=int(mask.sum()),
                    baseline=mean(base_errors[mask].tolist()), corrected=mean(new_errors[mask].tolist()))
                  for name, mask in masks.items()}
        physical = record['gt_palm_pos']
        calibrated = base + coefficient * (record['pred_obj_trans'] - base)
        physical_errors = {}
        for name, positions in [('baseline', base), ('calibrated', calibrated), ('corrected', pred)]:
            error = torch.linalg.vector_norm((positions[:, None] - palm) - (gt_obj[:, None] - physical), dim=-1) * 100
            physical_errors[name] = mean(error[contact].tolist())
        source = Path(payload['data_path']) / Path(record['seq_file']).name
        original = torch.load(source, map_location='cpu', weights_only=False)
        original_name = str(original.get('seq_name', source.stem))
        rows.append(dict(file=source.name, original_name=original_name, group=original_name.rsplit('_', 1)[0],
                         object=record['obj_name'], phases=phases, physical_hoi_cm=physical_errors))
    phases = {}
    for name in rows[0]['phases']:
        phases[name] = dict(sequences=sum(r['phases'][name]['baseline'] is not None for r in rows),
                           contact_hand_frames=sum(r['phases'][name]['hand_frames'] for r in rows),
                           baseline_hoi_cm=mean(r['phases'][name]['baseline'] for r in rows),
                           corrected_hoi_cm=mean(r['phases'][name]['corrected'] for r in rows))
        phases[name]['delta_cm'] = phases[name]['corrected_hoi_cm'] - phases[name]['baseline_hoi_cm']
    groups = {}
    for row in rows:
        phase = row['phases']['all']
        if phase['baseline'] is not None:
            groups.setdefault(row['group'], []).append(phase['corrected'] - phase['baseline'])
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(42)
    bootstrap = []
    for _ in range(5000):
        selected = rng.integers(0, len(groups), size=len(groups))
        bootstrap.append(sums[selected].sum() / counts[selected].sum())
    report = dict(sequences=len(rows), subject_object_groups=len(groups), phases=phases,
                  physical_hoi_cm={k: mean(r['physical_hoi_cm'][k] for r in rows) for k in ('baseline', 'calibrated', 'corrected')},
                  cluster_bootstrap_95_ci_delta_cm=[float(v) for v in np.quantile(bootstrap, [0.025, 0.975])],
                  cluster_bootstrap_unit='subject_object; all trials retained within each resampled cluster',
                  elapsed_sec=time.perf_counter() - started, per_sequence=rows)
    (output / 'phase_and_geometry_diagnostics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    # Benchmark ONLY this extra module, CPU, contiguous one-frame streaming.
    record = payload['records'][0]
    n = min(180, len(record['pred_obj_trans_published']))
    timings = []
    for repeat in range(3):
        predictor.reset()
        start = time.perf_counter()
        for i in range(n):
            chunk = {k: v[i:i+1] if isinstance(v, torch.Tensor) and v.ndim else v for k,v in record.items()}
            predictor.predict(chunk)
        timings.append((time.perf_counter() - start) * 1000 / n)
    latency = dict(device='cpu', cpu_threads=4, frames_per_repeat=n, repeats=3,
                   median_ms_per_frame=float(np.median(timings)), all_ms_per_frame=timings,
                   parameters=sum(p.numel() for p in predictor.model.parameters()),
                   note='Additional postprocessor only; not total HP/VC/OT latency, measured while official evaluation may run.')
    (output / 'runtime_benchmark.json').write_text(json.dumps(latency, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    official_base = phases['all']['baseline_hoi_cm']
    official_new = phases['all']['corrected_hoi_cm']
    physical = report['physical_hoi_cm']
    for ax, values, title in [(axes[0], [official_base, official_new], 'Official canonical-GT HOI'),
                              (axes[1], [physical['baseline'], physical['corrected']], 'Preprocessed-GT geometry diagnostic')]:
        bars = ax.bar(['OT-v3 baseline', 'Causal residual'], values, color=['#72869b', '#257e78'])
        ax.bar_label(bars, fmt='%.3f cm', padding=4)
        ax.set_ylim(0, max(values) * 1.19)
        ax.set_ylabel('HOI error (cm); lower is better')
        ax.set_title(title)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Locked 1,015-sequence test: metric improvement has a geometry trade-off', fontsize=12)
    fig.savefig(output / 'metric_geometry_tradeoff.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_sequence',)}, indent=2))
    print(json.dumps(latency, indent=2))


if __name__ == '__main__':
    main()
