#!/usr/bin/env python3
"""Fit causal contact residuals on train-derived, genuine online predictions.

prepare: group TRAIN sequences before caching. train: select weights/epoch/gain
using only that development validation. evaluate: apply the locked checkpoint.
HOI uses published objects relative to final predicted palms, not a bone proxy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def prepare(args):
    """Keep all subject/object trials together, not random or adjacent frames."""
    source = Path(args.data_path).resolve()
    if source.name != 'train':
        raise ValueError('Development data must come from an explicit train directory.')
    rows = []
    for path in sorted(source.glob('*.pt')):
        sample = load_pt(path)
        length = int(sample['human_imu'].shape[0])
        if length < args.min_frames:
            continue
        original_name = str(sample.get('seq_name', path.stem))
        group = original_name.rsplit('_', 1)[0]
        partition = 'validation' if int(digest(f'{args.seed}:{group}')[:8], 16) % 5 == 0 else 'train'
        rows.append(dict(file=path.name, path=str(path), seq_name=original_name,
                         group=group, frames=length, partition=partition,
                         object=str(sample.get('obj_name', 'unknown'))))
    selected = []
    for partition, limit in [('train', args.train_sequences), ('validation', args.validation_sequences)]:
        candidates = [r for r in rows if r['partition'] == partition]
        candidates.sort(key=lambda r: digest(f"{args.seed}:{r['file']}"))
        selected.extend(candidates[:limit] if limit else candidates)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for partition in ('train', 'validation', 'all'):
        chosen = [r for r in selected if partition == 'all' or r['partition'] == partition]
        (out / f'{partition}.txt').write_text(''.join(r['path'] + '\n' for r in chosen))
    train_groups = {r['group'] for r in selected if r['partition'] == 'train'}
    valid_groups = {r['group'] for r in selected if r['partition'] == 'validation'}
    assert not train_groups & valid_groups
    payload = dict(schema_version=1, seed=args.seed, source=str(source),
                   rule='sha256(seed:subject_object) % 5 == 0 -> validation',
                   min_frames=args.min_frames, eligible_sequences=len(rows),
                   selected_sequences=len(selected), train_groups=len(train_groups),
                   validation_groups=len(valid_groups), rows=selected)
    write_json(out / 'split_manifest.json', payload)
    print(json.dumps({k: v for k, v in payload.items() if k != 'rows'}, indent=2), flush=True)


def model_from_settings(settings):
    from model.rnn.contact_residual import CausalContactResidual, PointwiseContactResidual, build_contact_residual_feature_mask
    common = dict(feature_dim=settings['feature_dim'], max_correction=settings['max_correction'])
    if settings['architecture'] == 'pointwise':
        model = PointwiseContactResidual(**common)
    else:
        model = CausalContactResidual(**common, hidden_dim=settings['hidden_dim'], num_layers=1, dropout=0.0)
    model.feature_mask = build_contact_residual_feature_mask(settings['feature_dim'], settings.get('feature_ablation', 'full'))
    return model


def prepare_record(record):
    from model.rnn.contact_residual import build_contact_residual_features
    features = build_contact_residual_features(record)
    n = features['features'].shape[0]
    result = {key: features[key].detach().cpu().float() for key in
              ('features', 'obj_rotation', 'contact_prob', 'base_obj', 'pred_palm')}
    if 'gt_palm_pos_eval' not in record or 'pred_obj_trans_published' not in record:
        raise ValueError('Cache lacks canonical evaluation palms or published objects; regenerate it.')
    result.update(target_obj=record['gt_obj_trans'][:n].float(),
                  target_palm=record['gt_palm_pos_eval'][:n].float(),
                  target_palm_preprocessed=record.get('gt_palm_pos', record['gt_palm_pos_eval'])[:n].float(),
                  contact=torch.stack([record['gt_lhand_contact'][:n], record['gt_rhand_contact'][:n]], -1).bool(),
                  file=Path(str(record.get('seq_file', record.get('seq_name', '')))).name,
                  source_file=str(record.get('seq_file', record.get('seq_name', ''))),
                  object=str(record.get('obj_name', 'unknown')))
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite {key} in {result['file']}")
    return result


def load_records(path):
    payload = load_pt(path)
    print(f"Preparing {len(payload['records'])} cached sequences from {path}", flush=True)
    return [prepare_record(r) for r in payload['records']], {k: v for k, v in payload.items() if k != 'records'}


def collate(records, device, mean, std):
    names = ('features', 'obj_rotation', 'contact_prob', 'base_obj', 'pred_palm',
             'target_obj', 'target_palm', 'contact')
    batch = {key: pad_sequence([r[key] for r in records], batch_first=True).to(device) for key in names}
    batch['target_palm_preprocessed'] = pad_sequence(
        [r.get('target_palm_preprocessed', r['target_palm']) for r in records], batch_first=True).to(device)
    lengths = torch.tensor([len(r['features']) for r in records], device=device)
    batch['valid'] = torch.arange(batch['features'].shape[1], device=device)[None] < lengths[:, None]
    batch['features'] = ((batch['features'] - mean) / std).clamp(-10.0, 10.0)
    return batch


def corrected_object(model, batch, gain=1.0):
    features = batch['features']
    feature_mask = getattr(model, 'feature_mask', None)
    if feature_mask is not None:
        features = features * feature_mask.to(features)
    delta, _ = model(features)
    contact = batch['contact_prob'][..., :2].clamp(0, 1)
    probability = 1.0 - (1.0 - contact[..., 0]) * (1.0 - contact[..., 1])
    time_mask = torch.ones_like(probability)
    time_mask[:, 0] = 0
    delta = delta * (probability * time_mask * gain).unsqueeze(-1)
    world_delta = torch.matmul(batch['obj_rotation'], delta.unsqueeze(-1)).squeeze(-1)
    return batch['base_obj'] + world_delta, world_delta


def sequence_losses(pred, batch, target_palm_key='target_palm'):
    valid = batch['valid']
    mask = batch['contact'] & valid.unsqueeze(-1)
    relative = (pred.unsqueeze(-2) - batch['pred_palm']) - (
        batch['target_obj'].unsqueeze(-2) - batch[target_palm_key])
    distance = torch.linalg.vector_norm(relative, dim=-1)
    counts = mask.sum((1, 2))
    hoi = (distance * mask).sum((1, 2)) / counts.clamp_min(1)
    obj = (torch.linalg.vector_norm(pred - batch['target_obj'], dim=-1) * valid).sum(1) / valid.sum(1)
    return hoi, obj, counts > 0


@torch.no_grad()
def score(model, records, device, mean, std, batch_size=32, gain=1.0, per_sequence=False):
    model.eval()
    rows = []
    for start in range(0, len(records), batch_size):
        samples = records[start:start + batch_size]
        batch = collate(samples, device, mean, std)
        prediction, delta = corrected_object(model, batch, gain)
        hoi, obj, has_contact = sequence_losses(prediction, batch)
        base_hoi, base_obj, _ = sequence_losses(batch['base_obj'], batch)
        physical_hoi, _, _ = sequence_losses(prediction, batch, 'target_palm_preprocessed')
        physical_base, _, _ = sequence_losses(batch['base_obj'], batch, 'target_palm_preprocessed')
        for i, sample in enumerate(samples):
            n = len(sample['features'])
            def jitter(trajectory):
                if n < 3:
                    return 0.0
                return float(torch.linalg.vector_norm(trajectory[2:n] - 2 * trajectory[1:n-1] + trajectory[:n-2], dim=-1).mean()) * 1000
            rows.append(dict(file=sample['file'], object=sample['object'], frames=n,
                             hoi_cm=float(hoi[i]) * 100 if has_contact[i] else None,
                             baseline_hoi_cm=float(base_hoi[i]) * 100 if has_contact[i] else None,
                             preprocessed_hoi_cm=float(physical_hoi[i]) * 100 if has_contact[i] else None,
                             baseline_preprocessed_hoi_cm=float(physical_base[i]) * 100 if has_contact[i] else None,
                             obj_cm=float(obj[i]) * 100, baseline_obj_cm=float(base_obj[i]) * 100,
                             jitter=jitter(prediction[i]), baseline_jitter=jitter(batch['base_obj'][i]),
                             correction_cm=float(torch.linalg.vector_norm(delta[i, :n], dim=-1).mean()) * 100))
    def average(key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    result = dict(sequences=len(rows), contact_sequences=sum(r['hoi_cm'] is not None for r in rows),
                  **{k: average(k) for k in ('hoi_cm', 'baseline_hoi_cm', 'obj_cm', 'baseline_obj_cm',
                                           'jitter', 'baseline_jitter', 'correction_cm',
                                           'preprocessed_hoi_cm', 'baseline_preprocessed_hoi_cm')})
    if per_sequence:
        result['per_sequence'] = rows
    return result


def save_checkpoint(path, model, settings, mean, std, metadata):
    torch.save(dict(schema_version=1, model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    settings=settings, feature_mean=mean.detach().cpu(), feature_std=std.detach().cpu(),
                    **metadata), path)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    records, provenance = load_records(args.dump)
    if provenance.get('mode') != 'online':
        raise ValueError('Training requires genuine online, not offline, frozen predictions.')
    split = json.loads(Path(args.split_manifest).read_text())
    source = Path(split['source']).resolve()
    if source.name != 'train' or Path(provenance['data_path']).resolve() != source:
        raise ValueError('The cache must come from exactly the train directory in the split manifest.')
    membership = {r['file']: r['partition'] for r in split['rows']}
    if len(membership) != len(split['rows']):
        raise ValueError('Split manifest contains duplicate source identities.')
    if set(membership.values()) - {'train', 'validation'}:
        raise ValueError('Split manifest contains an unsupported partition.')
    training_groups = {r['group'] for r in split['rows'] if r['partition'] == 'train'}
    validation_groups = {r['group'] for r in split['rows'] if r['partition'] == 'validation'}
    if training_groups & validation_groups:
        raise ValueError('Development train and validation source groups overlap.')
    for record in records:
        original = Path(record['source_file'])
        resolved = original.resolve() if original.is_absolute() else (source / original).resolve()
        if resolved != (source / record['file']).resolve():
            raise ValueError(f"Per-record source path escapes the locked train source: {original}")
    if len({r['file'] for r in records}) != len(records):
        raise ValueError('Cache contains duplicate sequence identities.')
    missing = set(membership) - {r['file'] for r in records}
    if missing:
        raise ValueError(f'Cache is missing locked development sequences: {sorted(missing)[:5]}')
    unexpected = [r['file'] for r in records if r['file'] not in membership]
    if unexpected:
        raise ValueError(f'Cache includes files outside locked development split: {unexpected[:5]}')
    train_records = [r for r in records if membership[r['file']] == 'train']
    validation = [r for r in records if membership[r['file']] == 'validation']
    if not train_records or not validation:
        raise ValueError('Need nonempty, disjoint train and validation sequences.')
    combined_features = torch.cat([r['features'] for r in train_records], 0)
    mean = combined_features.mean(0).to(device)
    std = combined_features.std(0).clamp_min(0.03).to(device)
    settings = dict(architecture=args.architecture, feature_dim=int(mean.numel()),
                    hidden_dim=args.hidden_dim, max_correction=args.max_correction,
                    feature_ablation=getattr(args, 'feature_ablation', 'full'))
    model = model_from_settings(settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    baseline = score(model, validation, device, mean, std, args.batch_size, gain=0.0)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(seed=args.seed, epoch=0, gain=0.0, validation=baseline,
                    train_sequences=len(train_records), validation_sequences=len(validation),
                    split_manifest=str(Path(args.split_manifest).resolve()),
                    split_sha256=digest(Path(args.split_manifest).read_bytes()),
                    data_provenance=provenance, train_args={k: v for k, v in vars(args).items() if k != 'function'})
    save_checkpoint(out / 'best.pt', model, settings, mean, std, metadata)
    best_hoi, best_epoch = baseline['hoi_cm'], 0
    history = [dict(epoch=0, loss=None, **baseline)]
    started = time.time()
    print(f'Train={len(train_records)}, validation={len(validation)}, F={mean.numel()}, baseline={baseline}', flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_records)))
        random.shuffle(order)
        epoch_loss = []
        for start in range(0, len(order), args.batch_size):
            batch = collate([train_records[i] for i in order[start:start + args.batch_size]], device, mean, std)
            predicted, delta = corrected_object(model, batch)
            hoi, obj, has_contact = sequence_losses(predicted, batch)
            hoi_loss = hoi[has_contact].mean() if has_contact.any() else hoi.sum() * 0
            continuity = batch['valid'][:, 1:] & batch['valid'][:, :-1]
            delta_velocity = (delta[:, 1:] - delta[:, :-1]).square().sum(-1)
            smooth = (delta_velocity * continuity).sum() / continuity.sum().clamp_min(1)
            regularizer = (delta.square().sum(-1) * batch['valid']).sum() / batch['valid'].sum()
            physical_weight = getattr(args, 'preprocessed_hoi_weight', 0.0)
            physical_loss = hoi_loss * 0
            if physical_weight:
                physical_hoi, _, _ = sequence_losses(predicted, batch, 'target_palm_preprocessed')
                physical_loss = physical_hoi[has_contact].mean() if has_contact.any() else physical_hoi.sum() * 0
            loss = hoi_loss + physical_weight * physical_loss + args.object_weight * obj.mean() + args.smooth_weight * smooth + args.residual_weight * regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        scores = score(model, validation, device, mean, std, args.batch_size)
        item = dict(epoch=epoch, loss=float(np.mean(epoch_loss)), elapsed_sec=time.time() - started, **scores)
        history.append(item)
        eligible = scores['obj_cm'] <= baseline['obj_cm'] + args.object_guard_cm and scores['jitter'] <= baseline['jitter'] + args.jitter_guard
        physical_guard = getattr(args, 'preprocessed_guard_cm', None)
        if physical_guard is not None:
            eligible = eligible and scores['preprocessed_hoi_cm'] <= baseline['preprocessed_hoi_cm'] + physical_guard
        if eligible and scores['hoi_cm'] < best_hoi - 1e-5:
            best_hoi, best_epoch = scores['hoi_cm'], epoch
            metadata.update(epoch=epoch, gain=1.0, validation=scores)
            save_checkpoint(out / 'best.pt', model, settings, mean, std, metadata)
        write_json(out / 'history.json', history)
        if epoch == 1 or epoch % 5 == 0 or epoch == best_epoch:
            print(f"epoch={epoch:03} loss={item['loss']:.5f} val_hoi={scores['hoi_cm']:.4f} val_obj={scores['obj_cm']:.4f} "
                  f"jitter={scores['jitter']:.4f} best_epoch={best_epoch} seconds={item['elapsed_sec']:.1f}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    checkpoint = load_pt(out / 'best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    gain_scores = [dict(gain=gain, **score(model, validation, device, mean, std, args.batch_size, gain=gain))
                   for gain in (0.0, 0.25, 0.5, 0.75, 1.0)]
    allowed = [r for r in gain_scores if r['obj_cm'] <= baseline['obj_cm'] + args.object_guard_cm
               and r['jitter'] <= baseline['jitter'] + args.jitter_guard]
    physical_guard = getattr(args, 'preprocessed_guard_cm', None)
    if physical_guard is not None:
        allowed = [r for r in allowed if r['preprocessed_hoi_cm'] <= baseline['preprocessed_hoi_cm'] + physical_guard]
    winner = min(allowed, key=lambda r: r['hoi_cm'])
    checkpoint.update(gain=winner['gain'], validation=winner)
    torch.save(checkpoint, out / 'best.pt')
    report = dict(settings=settings, seed=args.seed, train_sequences=len(train_records), validation_sequences=len(validation),
                  baseline=baseline, selected=winner, selected_epoch=checkpoint['epoch'], gain_validation=gain_scores,
                  checkpoint=str((out / 'best.pt').resolve()), elapsed_sec=time.time() - started, test_used=False, history=history)
    write_json(out / 'training_report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('history', 'gain_validation')}, indent=2), flush=True)


def evaluate(args):
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    checkpoint = load_pt(args.checkpoint)
    model = model_from_settings(checkpoint['settings']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    records, provenance = load_records(args.dump)
    if provenance.get('mode') != 'online':
        raise ValueError('Final evaluation must use genuine online frozen predictions.')
    mean, std = checkpoint['feature_mean'].to(device), checkpoint['feature_std'].to(device)
    report = score(model, records, device, mean, std, args.batch_size, gain=checkpoint['gain'], per_sequence=True)
    deltas = np.array([r['hoi_cm'] - r['baseline_hoi_cm'] for r in report['per_sequence'] if r['hoi_cm'] is not None])
    if not len(deltas):
        raise ValueError('No evaluable contact sequences.')
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(5000)])
    report.update(checkpoint=str(Path(args.checkpoint).resolve()), gain=checkpoint['gain'],
                  checkpoint_sha256=digest(Path(args.checkpoint).read_bytes()), evaluation_provenance=provenance,
                  paired_hoi_delta_cm=float(deltas.mean()),
                  paired_bootstrap_95_ci_cm=[float(v) for v in np.quantile(boot, [0.025, 0.975])],
                  fraction_improved=float((deltas < 0).mean()),
                  selection='checkpoint, epoch and gain locked using train-derived validation only')
    report['per_object'] = {}
    for name in sorted({r['object'] for r in report['per_sequence']}):
        rows = [r for r in report['per_sequence'] if r['object'] == name and r['hoi_cm'] is not None]
        if rows:
            report['per_object'][name] = dict(sequences=len(rows), hoi_cm=float(np.mean([r['hoi_cm'] for r in rows])),
                baseline_hoi_cm=float(np.mean([r['baseline_hoi_cm'] for r in rows])))
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('per_sequence', 'evaluation_provenance')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data_path', default='process/IMUHOI++/processed_split_data_OMOMO_full/train')
    p.add_argument('--output', required=True)
    p.add_argument('--train_sequences', type=int, default=384)
    p.add_argument('--validation_sequences', type=int, default=96)
    p.add_argument('--min_frames', type=int, default=71)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(function=prepare)
    for command in ('train', 'evaluate'):
        p = sub.add_parser(command)
        p.add_argument('--dump', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cuda:0')
        p.add_argument('--batch_size', type=int, default=32)
        p.add_argument('--cpu_threads', type=int, default=4)
        if command == 'train':
            p.add_argument('--split_manifest', required=True)
            p.add_argument('--architecture', choices=('gru', 'pointwise'), default='gru')
            p.add_argument('--hidden_dim', type=int, default=64)
            p.add_argument('--feature_ablation', choices=('full', 'no_publication', 'no_history', 'no_extra_imu'), default='full')
            p.add_argument('--max_correction', type=float, default=0.20)
            p.add_argument('--epochs', type=int, default=120)
            p.add_argument('--patience', type=int, default=25)
            p.add_argument('--lr', type=float, default=0.001)
            p.add_argument('--weight_decay', type=float, default=0.01)
            p.add_argument('--object_weight', type=float, default=0.5)
            p.add_argument('--preprocessed_hoi_weight', type=float, default=0.0)
            p.add_argument('--preprocessed_guard_cm', type=float, default=None)
            p.add_argument('--smooth_weight', type=float, default=10.0)
            p.add_argument('--residual_weight', type=float, default=0.1)
            p.add_argument('--object_guard_cm', type=float, default=0.0)
            p.add_argument('--jitter_guard', type=float, default=0.2)
            p.add_argument('--seed', type=int, default=42)
            p.set_defaults(function=train)
        else:
            p.add_argument('--checkpoint', required=True)
            p.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
