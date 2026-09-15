"""Matched current/bigrasp continuation experiments on frozen batch-800 VC-v4."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['IMUHOI_MODEL_ARCH'] = 'rnn'
SAGE = '/home/l/anaconda3/envs/SAGE/bin/python'
EXPERIMENTS = ROOT / 'outputs/IMUHOI++/RNN_cano'
SOURCE_ROOTS = {
    'omomo': EXPERIMENTS / 'omomo_cano_ot_trust_20260910_c',
    'behave': EXPERIMENTS / 'behave_full_trust_20260911',
}

DATA_LOADING_ENV = {'IMUHOI_COMPACT_PRELOAD': '1', 'IMUHOI_STRICT_LOADING': '1'}
DATA_SOURCE_FILES = ('dataset/dataset_IMUHOI.py', 'train/rnn/train_utils.py')


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def provenance(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def prepare(dataset, tag, epochs=None):
    import yaml
    source_root = SOURCE_ROOTS[dataset]
    source = json.loads((source_root / 'experiment_manifest.json').read_text())
    pipeline = source['pipelines']['cur_hoi_p800']
    stage_dir = Path(pipeline['joint_ot_ckpt']).parent
    config = yaml.safe_load(Path(pipeline['config']).read_text())
    if config['batch_size'] != 800 or config['velocity_contact_arch'] != 'v4':
        raise ValueError('Source must be the current batch-800 VC-v4 experiment')
    hp_path = source['reused_checkpoints']['hp_ckpt'] if dataset == 'omomo' else source['stages']['human_pose']['checkpoint']
    sources = {'human_pose': hp_path, 'velocity_contact': str(stage_dir / 'final_velocity_contact.pt'),
               'object_trans': str(stage_dir / 'final_object_trans.pt')}
    root = EXPERIMENTS / ('codex-bigrasp-' + dataset + '-' + tag)
    if (root / 'manifest.json').is_file():
        return root
    source_hashes = {name: provenance(path) for name, path in sources.items()}
    epochs = int(epochs or (100 if dataset == 'omomo' else 329))
    config.update({
        'model_arch': 'rnn', 'object_trans_arch': 'current', 'velocity_contact_arch': 'v4',
        'object_trans_state_trust': False, 'object_trans_static_gate': 'none',
        'object_trans_bigrasp': False, 'enable_ot_lb_prior': False,
        'object_trans_init_noise_std': 0.0, 'object_trans_rest_perturb_std': 0.0,
        'batch_size': 800, 'windows_per_sequence': 4, 'num_workers': 0,
        'dataloader_pin_memory': False, 'train_datasets': [dataset],
        'seed': 42, 'debug': False, 'no_trans': False, 'use_multi_gpu': False,
        'amp_enabled': False, 'device': 'cuda:0', 'gpus': [0],
        'epochs': epochs, 'lr': 0.0002, 'milestones': [round(epochs * fraction) for fraction in (0.3, 0.6, 0.9)],
        'gamma': 0.5, 'early_stopping_patience_epochs': 0, 'joint_train': False,
        'train_data_window': 90, 'ot_history_training_window': 60,
        'input_mix_start_epoch': 0, 'input_mix_end_epoch': 0,
        'resume_dir': None, 'pretrained_modules': sources, 'use_tensorboard': True,
        'object_trans_arch_loss_weights': {},
    })
    config['loss_weights'].update({'obj_trans': 1.0, 'hoi_fused_l': 10.0, 'hoi_fused_r': 10.0,
                                   'prior_hoi_l': 0.0, 'prior_hoi_r': 0.0, 'state_conf': 0.0})
    configs = {}
    for variant in ('current', 'bigrasp'):
        variant_config = copy.deepcopy(config)
        variant_dir = root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_config['save_dir'] = str(variant_dir)
        variant_config['object_trans_bigrasp'] = variant == 'bigrasp'
        if variant == 'bigrasp':
            variant_config.update({'bigrasp_hidden_dim': 96})
            variant_config['loss_weights'].update({'bigrasp_common': 5.0, 'bigrasp_proposal': 5.0,
                                                   'bigrasp_delta': 0.1, 'bigrasp_quality': 0.1})
        config_path = variant_dir / 'config.yaml'
        config_path.write_text(yaml.safe_dump(variant_config, sort_keys=False), encoding='utf-8')
        configs[variant] = str(config_path)
    manifest = {'dataset': dataset, 'status': 'prepared', 'sources': source_hashes, 'configs': configs,
                'source_manifest': provenance(source_root / 'experiment_manifest.json'),
                'protocol': {'batch_size': 800, 'windows_per_sequence': 4, 'epochs': epochs,
                             'frozen': ['human_pose', 'velocity_contact'], 'train_sequence_frames': 90,
                             'warmup_frames': 60, 'overlap_frames': 30, 'supervised_frames': 30,
                             'checkpoint_selection': 'final fixed budget; no test-based selection',
                             'online_window': 60, 'online_lookahead': 10, 'seed': 42,
                             'amp_enabled': False, 'training_inputs': 'predicted HP and VC; offline upstream windows'},
                'source_files': {name: provenance(ROOT / name) for name in (
                    'model/rnn/bimanual_grasp.py', 'model/rnn/object_trans.py',
                    'train/rnn/ot_history_training.py', 'train/rnn/loss/object_trans_loss.py',
                    'train/rnn/train_object_trans.py', 'scripts/run_codex_bigrasp.py')},
                'runs': {}}
    write_json(root / 'manifest.json', manifest)
    return root


def train(config_path, device_name, smoke=False):
    os.environ.update(DATA_LOADING_ENV)
    import torch
    from model.rnn.human_pose import HumanPoseModule
    from model.rnn.object_trans import ObjectTransModule
    from model.rnn.velocity_contact import build_velocity_contact_module
    from train.rnn.train_object_trans import Stage3JointTrainer
    from train.rnn.train_utils import create_dataloaders, setup_seed, load_checkpoint, build_model_input_dict
    from utils.utils import load_config

    torch.set_num_threads(4 if device_name == 'cpu' else 1)
    cfg = load_config(str(config_path))
    cfg.device = 'cpu' if device_name == 'cpu' else 'cuda:0'
    setup_seed(cfg.seed)
    loader_cfg = copy.deepcopy(cfg)
    for dataset in cfg.train_datasets:
        loader_cfg.datasets[dataset].pop('test_path', None)
    train_loader, _ = create_dataloaders(loader_cfg)
    if train_loader is None or not len(train_loader):
        raise ValueError('No complete training batches')
    cfg.body_model_path = loader_cfg.body_model_path
    if hasattr(loader_cfg, 'body_model_type'):
        cfg.body_model_type = loader_cfg.body_model_type
    cfg.resume_dir = cfg.save_dir if (Path(cfg.save_dir) / 'last.pt').is_file() and not smoke else None
    device = torch.device(cfg.device)
    hp = HumanPoseModule(cfg, device, no_trans=False)
    vc = build_velocity_contact_module(cfg)
    ot = ObjectTransModule(cfg)
    load_checkpoint(hp, cfg.pretrained_modules.human_pose, device)
    load_checkpoint(vc, cfg.pretrained_modules.velocity_contact, device, strict=False)
    checkpoint = torch.load(cfg.pretrained_modules.object_trans, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint.get('module_state_dict'))
    missing, unexpected = ot.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith('bigrasp.') for key in missing):
        raise ValueError((missing, unexpected))
    if not cfg.object_trans_bigrasp and missing:
        raise ValueError(missing)
    setup_seed(cfg.seed)
    if smoke:
        cfg.use_tensorboard = False
    trainer = Stage3JointTrainer(cfg, vc, hp, ot, train_loader, None, joint_train=False)
    if any(parameter.requires_grad for module in (trainer.hp_model, trainer.vc_model) for parameter in module.parameters()):
        raise AssertionError('Upstream modules were not frozen')
    before = {name: hashlib.sha256(b''.join(tensor.detach().cpu().numpy().tobytes() for tensor in module.state_dict().values())).hexdigest()
              for name, module in (('hp', trainer.hp_model), ('vc', trainer.vc_model))}
    if smoke:
        batch = next(iter(train_loader))
        data = build_model_input_dict(batch, cfg, device, add_noise=True)
        trainer.ot_model.train()
        started = time.time()
        outputs, _, _, _ = trainer.forward_all(data, batch=batch, epoch=0, training=True)
        loss, components, _ = trainer.ot_loss_fn.compute_loss(outputs, batch, device)
        loss.backward()
        if not torch.isfinite(loss) or any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                                          for parameter in trainer.ot_model.parameters()):
            raise AssertionError('Nonfinite loss or gradient')
        trainer.optimizer.step()
        result = {'batch_size': int(batch['human_imu'].shape[0]), 'sequence_frames': int(batch['human_imu'].shape[1]),
                  'train_sequences': len(train_loader.dataset), 'discovered_files': len(train_loader.dataset.sequence_files),
                  'data_loading': dict(DATA_LOADING_ENV),
                  'loss': float(loss.detach()), 'seconds': time.time() - started,
                  'bigrasp_losses': {key: float(value.detach()) for key, value in components.items() if key.startswith('bigrasp')},
                  'max_gpu_allocated_mib': torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == 'cuda' else None}
        assert result['batch_size'] == 800
        print('SMOKE_RESULT ' + json.dumps(result), flush=True)
        write_json(Path(cfg.save_dir) / ('smoke_' + device_name + '.json'), result)
    else:
        trainer.train()
    after = {name: hashlib.sha256(b''.join(tensor.detach().cpu().numpy().tobytes() for tensor in module.state_dict().values())).hexdigest()
             for name, module in (('hp', trainer.hp_model), ('vc', trainer.vc_model))}
    assert before == after, 'Frozen HP/VC state changed'
    if not smoke:
        final = torch.load(Path(cfg.save_dir) / 'final.pt', map_location='cpu', weights_only=False)
        assert final['epoch'] == cfg.epochs - 1
        write_json(Path(cfg.save_dir) / 'training_complete.json', {'epoch': final['epoch'], 'steps': final['n_iter'],
                   'batches_per_epoch': len(train_loader), 'train_sequences': len(train_loader.dataset),
                   'upstream_sha256_before': before, 'upstream_sha256_after': after, 'device': cfg.device})


def run_stage(command, log_path, device_name):
    subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv'], check=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device_name == 'cpu' else '0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    environment.update(DATA_LOADING_ENV)
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            if (line.startswith('Epoch ') and 'Train Loss:' in line) or line.startswith(('SMOKE_RESULT', 'Traceback', 'RuntimeError', '训练完成', '保存模型:', 'Online window:', 'Online lookahead:')):
                print(line.rstrip(), flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError('Stage failed with code %d; see %s' % (returncode, log_path))


def write_comparison(root, manifest):
    names = ('source_current', 'current', 'bigrasp')
    metrics = ('hoi_err_fusion', 'obj_trans_err_fusion', 'obj_jitter', 'mpjpe', 'root_trans_err')
    rows = {name: manifest['runs'][name]['metrics'] for name in names}
    comparison = {}
    for reference in ('current', 'source_current'):
        comparison['bigrasp_minus_' + reference] = {
            metric: rows['bigrasp'][metric] - rows[reference][metric]
            if rows['bigrasp'].get(metric) is not None and rows[reference].get(metric) is not None else None
            for metric in metrics
        }
    manifest['comparison'] = comparison
    lines = ['# ' + manifest['dataset'].upper() + ' current vs bigrasp', '',
             'Fixed-budget final checkpoints; shared frozen HP/VC; negative deltas are improvements.', '',
             '| Variant | HOI (cm) | Object (cm) | Object jitter | MPJPE (cm) | Root (cm) |',
             '|---|---:|---:|---:|---:|---:|']
    for name, values in tuple(rows.items()) + tuple(comparison.items()):
        cells = [format(values[metric], '.4f') if values.get(metric) is not None else 'N/A' for metric in metrics]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines.extend(('', 'See manifest.json for checkpoint hashes, runtime device, training counts, and frozen-weight checks.', ''))
    (root / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def run_experiment(root, device_name):
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_names = tuple(dict.fromkeys(tuple(manifest['source_files']) + DATA_SOURCE_FILES))
    current_sources = {name: provenance(ROOT / name) for name in source_names}
    changed = [name for name in current_sources if current_sources[name]['sha256'] != manifest['source_files'].get(name, {}).get('sha256')]
    if changed:
        if any((root / variant / 'last.pt').is_file() for variant in ('current', 'bigrasp')):
            raise RuntimeError('Training source changed after checkpoint creation: ' + ', '.join(changed))
        manifest['source_files'] = current_sources
    manifest['execution_device'] = device_name
    manifest['data_loading'] = dict(DATA_LOADING_ENV)
    manifest['status'] = 'running'
    write_json(manifest_path, manifest)
    sources = {key: value['path'] for key, value in manifest['sources'].items()}
    for variant in ('bigrasp', 'current'):
        variant_dir = root / variant
        if not (variant_dir / 'training_complete.json').is_file():
            print('TRAIN ' + str(variant_dir), flush=True)
            run_stage([SAGE, str(Path(__file__).resolve()), 'train', '--config', manifest['configs'][variant], '--device', device_name], variant_dir / 'train.log', device_name)
        manifest['runs'][variant] = {'training': json.loads((variant_dir / 'training_complete.json').read_text())}
        write_json(manifest_path, manifest)
    for variant in ('source_current', 'current', 'bigrasp'):
        config = manifest['configs']['bigrasp' if variant == 'bigrasp' else 'current']
        checkpoint = sources['object_trans'] if variant == 'source_current' else str(root / variant / 'final.pt')
        output = root / (variant + '_eval.json')
        if not output.is_file():
            print('EVALUATE ' + variant + ' on ' + manifest['dataset'], flush=True)
            run_stage([SAGE, 'eval_batch_online.py', '--config', config, '--dataset', manifest['dataset'],
                       '--hp_ckpt', sources['human_pose'], '--vc_ckpt', sources['velocity_contact'], '--ot_ckpt', checkpoint,
                       '--object_trans_arch', 'current', '--velocity_contact_arch', 'v4', '--online_window', '60',
                       '--online_lookahead', '10', '--online_batch_size', '48', '--simulate_imu_noise', '--seed', '42',
                       '--output_json', str(output)], root / (variant + '_eval.log'), device_name)
        payload = json.loads(output.read_text())
        metrics = payload['results'][manifest['dataset']]
        selected = {key: metrics.get(key) for key in ('hoi_err_fusion', 'obj_trans_err_fusion', 'obj_jitter', 'mpjpe',
                                                     'root_trans_err', 'contact_f1_lhand', 'contact_f1_rhand')}
        manifest['runs'].setdefault(variant, {})['metrics'] = selected
        write_json(manifest_path, manifest)
    write_comparison(root, manifest)
    manifest['status'] = 'complete'
    write_json(manifest_path, manifest)
    print(json.dumps(manifest['runs'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'smoke', 'train', 'run'))
    parser.add_argument('--dataset', choices=('omomo', 'behave'), default='omomo')
    parser.add_argument('--tag', default='20260914')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path(SAGE).resolve():
        raise RuntimeError('Use the WSL native SAGE interpreter')
    if args.action in ('train', 'smoke'):
        if args.config is None:
            parser.error('--config is required')
        train(args.config, args.device, smoke=args.action == 'smoke')
    else:
        root = prepare(args.dataset, args.tag, args.epochs)
        print(str(root), flush=True)
        if args.action == 'run':
            try:
                run_experiment(root, args.device)
            except Exception as error:
                path = root / 'manifest.json'
                manifest = json.loads(path.read_text())
                manifest.update({'status': 'failed', 'error': str(error)})
                write_json(path, manifest)
                raise


if __name__ == '__main__':
    main()