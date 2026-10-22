"""Read-only fixed-HP/OT VC interventions using production evaluation."""
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu, PYTHON, sha, write_json, now
from scripts.run_omomo_original_recipe_20261018 import OUT as RUN, original_config
OUT = ROOT / 'outputs/analysis/egoimu_vc_joint_probe_20261019'
ARMS = {'prejoint': RUN / 'proposed/vc/best.pt',
        'joint_best': RUN / 'proposed/joint/best_velocity_contact.pt',
        'joint_final': RUN / 'proposed/joint/final_velocity_contact.pt'}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def reuse_online_best(mode='online'):
    """Reuse the already completed identical reference, preserving provenance."""
    from scripts.run_omomo_original_recipe_20261018 import verify
    check_gpu(0)
    verify(json.loads((RUN / 'protocol.json').read_text()))
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert all(sha(Path(p)) == s for p, s in manifest['sha256'].items())
    source = RUN / f'proposed/eval_{mode}_best.json'
    coverage = json.loads(source.with_suffix('.coverage.json').read_text())
    rows = coverage['sequences']
    assert len(rows) == len({r['seq_path'] for r in rows}) == 1177
    result = json.loads(source.read_text())
    assert result['config'] == manifest['config']
    state = json.loads((RUN / 'status.json').read_text())
    command = state['stages'][f'proposed_{mode}_best']['command']
    for flag, path in (('--hp_ckpt', manifest['fixed_hp']), ('--ot_ckpt', manifest['fixed_ot']),
                       ('--vc_ckpt', str(ARMS['joint_best']))):
        assert command[command.index(flag) + 1] == path
    flags = [('--seed', '10')]
    flags += [('--online_window', '60'), ('--online_lookahead', '10'), ('--online_chunk', '16')] if mode == 'online' else [('--inference_mode', 'offline')]
    for flag, value in flags:
        assert command[command.index(flag) + 1] == value
    output = OUT / f'{mode}_joint_best.json'
    diagnostic_path = output.with_suffix('.diagnostics.json')
    if mode == 'offline' and diagnostic_path.exists():
        diagnostic_path.replace(OUT / 'offline_joint_best.batched_diagnostics.json')
    output.write_bytes(source.read_bytes())
    write_json(output.with_suffix('.coverage.json'), {'status': 'passed', 'unique_sequences': 1177, 'reused_from': str(source)})
    velocity_rows = json.loads((OUT / 'velocity_head_diagnostics.json').read_text())['arms']['joint_best'] if mode == 'offline' else []
    write_json(diagnostic_path, {'sequences': [
        {'seq_path': r['seq_path'], 'obj_trans_err_fusion_published_cm': r['obj_cm'], 'hoi_err_fusion_cm': r['hoi_cm']} for r in rows],
        'velocity': velocity_rows, 'note': 'Metrics and coverage reused from identical official reference. Offline velocity diagnostics use independently validated velocity-only forward; online per-sequence velocity unavailable.'})
    write_json(OUT / f'{mode}_best_reuse_receipt.json', {'source': str(source), 'sha256': sha(source),
               'command': command, 'checkpoint_hashes_verified': True, 'frozen_protocol_verified': True})


def worker():
    check_gpu(6000)
    import torch
    from configs import FRAME_RATE
    mode = sys.argv.pop(2)
    sys.argv.pop(1)
    entry = importlib.import_module('eval_IMUHOI' if mode == 'offline' else 'eval_batch_online')
    original_eval = entry.evaluate_model
    output = Path(sys.argv[sys.argv.index('--output_json') + 1])
    import fcntl
    lock = output.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if output.exists() and output.with_suffix('.coverage.json').exists():
        return

    def checked(model, loader, *args, **kwargs):
        assert len(loader.dataset) == 1177
        if mode == 'offline' and os.environ.get('PROBE_OFFLINE_BATCH4') == '1':
            from collections import defaultdict
            groups = defaultdict(list)
            for index, info in enumerate(loader.dataset.sequence_info):
                groups[info['seq_len']].append(index)
            batches = [indices[start:start + 4] for indices in groups.values()
                       for start in range(0, len(indices), 4)]
            loader = torch.utils.data.DataLoader(loader.dataset, batch_sampler=batches, num_workers=0)
            print('Exact-length offline batches (no padding):', len(batches), flush=True)
        records, velocity = [], []
        original_inference = model.inference

        def inference(data, *pos, **kw):
            pred = original_inference(data, *pos, **kw)
            gt = kw['gt_targets']
            v = pred['pred_obj_vel'].detach()
            t = v.shape[1]
            target = gt['obj_vel'][:, :t].to(v)
            residual = v - target
            trajectory = data['obj_trans_init'].to(v)[:, None] + torch.cat(
                (torch.zeros_like(v[:, :1]), v[:, :-1].cumsum(1) / float(FRAME_RATE)), dim=1)
            all_err = (trajectory - gt['obj_trans'][:, :t].to(v)).norm(dim=-1) * 100
            for index in range(v.shape[0]):
                err, res = all_err[index], residual[index]
                bias = res.mean(0)
                row = {'seq_path': str(gt['seq_path'][index]), 'frames': t,
                       'velocity_mse_m2ps2': res.square().mean().item(),
                       'velocity_vector_error_mps': res.norm(dim=-1).mean().item(),
                       'velocity_bias_mps': bias.tolist(), 'velocity_bias_norm_mps': bias.norm().item(),
                       'integral_obj_cm': err.mean().item(), 'integral_end_cm': err[-1].item(),
                       'integral_first30_cm': err[:30].mean().item(),
                       'integral_first60_cm': err[:60].mean().item()}
                velocity.append(row)
            if len(velocity) % 100 < v.shape[0]:
                write_json(output.with_suffix('.progress.json'), {'sequences': len(velocity), 'time': now()})
            return pred

        model.inference = inference
        kwargs['sequence_records'] = records
        result = original_eval(model, loader, *args, **kwargs)
        assert len(records) == len(velocity) == 1177
        assert len({r['seq_path'] for r in records}) == 1177
        assert all(math.isfinite(r['obj_trans_err_fusion_published_cm']) and math.isfinite(r['hoi_err_fusion_cm']) for r in records)
        write_json(output.with_suffix('.diagnostics.json'), clean({'sequences': records, 'velocity': velocity}))
        if output.stem == 'offline_joint_best':
            reference = json.loads((RUN / 'proposed/eval_offline_best.coverage.json').read_text())['sequences']
            previous = {r['seq_path']: r for r in reference}
            differences = [abs(r['obj_trans_err_fusion_published_cm'] - previous[r['seq_path']]['obj_cm']) for r in records]
            receipt = {'max_per_sequence_obj_cm_delta': max(differences), 'mean_absolute_delta_cm': sum(differences) / len(differences)}
            write_json(OUT / 'offline_batch_regression.json', receipt)
            assert max(differences) < 0.002, receipt
        write_json(output.with_suffix('.coverage.json'), {'status': 'passed', 'unique_sequences': 1177})
        return result

    entry.evaluate_model = checked
    entry.main()


def run():
    check_gpu(6000)
    import torch
    import yaml
    OUT.mkdir(parents=True, exist_ok=True)
    config = RUN / 'proposed/joint/recipe.yaml'
    hp, ot = RUN / 'shared/hp/best.pt', RUN / 'proposed/joint/best_object_trans.pt'
    cfg = yaml.safe_load(config.read_text())
    native = original_config()['loss_weights']
    audit = {'native_loss_weights': native, 'proposed_loss_weights': cfg['loss_weights'],
             'common_mismatches': {k: [v, cfg['loss_weights'].get(k)] for k, v in native.items() if cfg['loss_weights'].get(k) != v},
             'arch_overrides': cfg.get('object_trans_arch_loss_weights'),
             'joint_selection_keys': cfg.get('joint_checkpoint_selection_keys')}
    write_json(OUT / 'loss_audit.json', audit)
    manifest = {'fixed_hp': str(hp), 'fixed_ot': str(ot), 'config': str(config), 'seed': 10,
                'dataset_sequences': 1177, 'purpose': 'fixed-component intervention, not retraining',
                'sha256': {str(p): sha(p) for p in [config, hp, ot] + list(ARMS.values())}, 'arms': {}}
    for arm, path in ARMS.items():
        p = torch.load(path, map_location='cpu')
        manifest['arms'][arm] = {'checkpoint': str(path), 'metadata': {k: v for k, v in p.items() if isinstance(v, (int, float, str, bool))}, 'keys': list(p.keys())}
    write_json(OUT / 'manifest.json', clean(manifest))
    for mode in ('offline', 'online'):
        for arm, path in ARMS.items():
            output = OUT / f'{mode}_{arm}.json'
            if output.exists() and output.with_suffix('.coverage.json').exists():
                continue
            check_gpu(6000)
            cmd = [str(PYTHON), __file__, 'worker', mode, '--config', str(config), '--dataset', 'omomo',
                   '--hp_ckpt', str(hp), '--vc_ckpt', str(path), '--ot_ckpt', str(ot),
                   '--velocity_contact_arch', 'v5', '--object_trans_arch', 'v2', '--seed', '10',
                   '--num_workers', '0', '--output_json', str(output)]
            cmd += ['--inference_mode', 'offline'] if mode == 'offline' else [
                '--online_batch_size', '24', '--online_window', '60', '--online_lookahead', '10', '--online_chunk', '16']
            write_json(OUT / 'status.json', {'status': 'running', 'arm': arm, 'mode': mode, 'started_at': now(), 'command': cmd})
            with output.with_suffix('.log').open('w') as log:
                p = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '10'})
            if p.returncode:
                write_json(OUT / 'status.json', {'status': 'failed', 'arm': arm, 'mode': mode, 'returncode': p.returncode})
                raise RuntimeError(str(output.with_suffix('.log')))
    assert all(sha(Path(p)) == s for p, s in manifest['sha256'].items())
    write_json(OUT / 'status.json', {'status': 'complete', 'finished_at': now()})


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    worker() if len(sys.argv) > 1 and sys.argv[1] == 'worker' else run()
