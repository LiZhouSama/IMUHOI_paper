"""Swap trained velocity heads, then conditionally run paired augmentation ablations."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import PYTHON, check_gpu, sha, write_json, now
from scripts.run_omomo_original_recipe_20261018 import OUT as RUN, verify, read
from scripts.probe_vc_joint_20261019 import clean
OUT = ROOT / 'outputs/analysis/velocity_swap_aug_ablation_20261019'
PREFIXES = ('hand_vel_net.', 'obj_vel_net.')


def state(payload):
    return payload.get('module_state_dict', payload.get('model_state_dict', payload))


def heads_only(module_state):
    return {k: v.detach().cpu().clone() for k, v in module_state.items() if k.startswith(PREFIXES)}


def compose(heads, destination):
    import torch
    source = RUN / 'proposed/joint/best_velocity_contact.pt'
    fixed = state(torch.load(source, map_location='cpu'))
    assert set(heads) == {k for k in fixed if k.startswith(PREFIXES)}
    mixed = {k: v.clone() for k, v in fixed.items()}
    for k, v in heads.items():
        assert mixed[k].shape == v.shape
        mixed[k] = v.clone()
    assert all(torch.equal(mixed[k], fixed[k]) for k in fixed if not k.startswith(PREFIXES))
    torch.save({'module_state_dict': mixed, 'model_state_dict': mixed,
                'purpose': 'fixed joint-Best contact parameters; replaced hand/object velocity heads'}, destination)
    return {'source_contact': str(source), 'head_tensor_count': len(heads),
            'unchanged_other_tensors': sum(not k.startswith(PREFIXES) for k in fixed),
            'unchanged_other_tensors_exact': True, 'sha256': sha(destination)}


def prepare():
    check_gpu(0)
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    verify(read(RUN / 'protocol.json'))
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        assert all(sha(Path(p)) == s for p, s in manifest['fixed_sha256'].items())
        return
    paths = [RUN / 'shared/hp/best.pt', RUN / 'proposed/joint/best_object_trans.pt',
             RUN / 'proposed/joint/best_velocity_contact.pt', RUN / 'baseline/vc/best.pt',
             RUN / 'proposed/joint/recipe.yaml', RUN / 'proposed/vc/recipe.yaml']
    baseline = state(torch.load(paths[3], map_location='cpu'))
    receipt = compose(heads_only(baseline), OUT / 'baseline_velocity_in_proposed.pt')
    manifest = {'created_at': now(), 'seed': 10, 'test_sequences': 1177,
                'fixed_sha256': {str(p): sha(p) for p in paths}, 'swap_receipt': receipt,
                'fixed_components': 'proposed joint-Best contact/HP/OT; only both velocity subnetworks replaced',
                'gate': 'complete offline coverage and fusion Obj at least 0.1 cm lower than proposed Best; HOI/online separately reported',
                'conditional_training': {'arms': ['augmentation_off', 'augmentation_on'], 'epochs': 300,
                    'samples': 4115, 'batch_size': 100, 'window': 120, 'K': 1, 'steps_per_epoch': 41,
                    'optimizer': 'AdamW', 'lr': .002, 'weight_decay': .0001, 'milestones': [50,150], 'gamma': .2,
                    'precision': 'FP32', 'gradient_clipping': False, 'workers': 12, 'persistent_workers': False,
                    'velocity_loss_weights': {'obj_vel': 10, 'hand_vel': 10}, 'initialization': 'fresh common seed10 baseline velocity initializer',
                    'augmentation_on': {'probability': .5, 'object_init_std_mps': .2, 'hand_init_std_mps': .2},
                    'pairing': 'same batch/crop/IMU noise/dropout; augmentation uses isolated RNG; each epoch seed10+epoch',
                    'validation': '1016 random W120 with sigma0.1 IMU noise every10epochs; same windows/noise across arms; no init noise',
                    'selection': '10*object_velocity_MSE + 10*hand_velocity_MSE; keep best within first120 and all300 plus final120/final300',
                    'not_exact_old_replay': 'paired epoch/RNG control intentionally differs from prior sequential RNG consumption',
                    'full_eval': 'both modes for best120 and best300, keeping joint-Best contact/HP/OT fixed'}}
    write_json(manifest_path, manifest)


def eval_command(tag, checkpoint, mode):
    output = OUT / f'{tag}_{mode}.json'
    cmd = [str(PYTHON), str(ROOT / 'scripts/probe_vc_joint_20261019.py'), 'worker', mode,
           '--config', str(RUN / 'proposed/joint/recipe.yaml'), '--dataset', 'omomo',
           '--hp_ckpt', str(RUN / 'shared/hp/best.pt'), '--vc_ckpt', str(checkpoint),
           '--ot_ckpt', str(RUN / 'proposed/joint/best_object_trans.pt'), '--velocity_contact_arch', 'v5',
           '--object_trans_arch', 'v2', '--seed', '10', '--num_workers', '0', '--output_json', str(output)]
    cmd += ['--inference_mode', 'offline'] if mode == 'offline' else [
        '--online_batch_size', '24', '--online_window', '60', '--online_lookahead', '10', '--online_chunk', '16']
    return cmd, output


def job(name, cmd, marker, own_status=False):
    if marker.exists():
        return
    check_gpu(6000)
    status_path = OUT / (f'{name}.job.json' if own_status else 'status.json')
    write_json(status_path, {'status': 'running', 'stage': name, 'started_at': now(), 'command': cmd})
    with (OUT / f'{name}.log').open('a') as log:
        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '10'})
    if result.returncode or not marker.exists():
        raise RuntimeError(f'{name} failed: return={result.returncode}, marker={marker}')
    if own_status:
        write_json(status_path, {'status': 'complete', 'stage': name, 'finished_at': now(), 'command': cmd})


def paired_train():
    # cuDNN keeps recurrent dropout state outside torch's restorable RNG state.
    # Use isolated matched processes, verified by complete input/RNG digests.
    from scripts.train_paired_velocity_20261019 import run as train_in_processes
    return train_in_processes()


def run():
    prepare()
    lock = (OUT / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        for mode in ('offline','online'):
            cmd, output = eval_command('swap', OUT / 'baseline_velocity_in_proposed.pt', mode)
            job(f'swap_{mode}', cmd, output)
        original = read(RUN / 'proposed/eval_offline_best.json')['results']['omomo']
        swapped = read(OUT / 'swap_offline.json')['results']['omomo']
        improvement = original['obj_trans_err_fusion'] - swapped['obj_trans_err_fusion']
        gate = {'passed':improvement >= .1, 'offline_fusion_obj_improvement_cm':improvement,
                'original':clean(original), 'swapped':clean(swapped), 'evaluated_at':now()}
        if not (OUT / 'gate.json').exists():
            write_json(OUT / 'gate.json', gate)
        authorization = OUT / 'training_authorization.json'
        authorized = authorization.exists() and read(authorization)['authorized']
        if not gate['passed'] and not authorized:
            write_json(OUT / 'status.json', {'status':'complete_no_training', 'reason':'predeclared swap benefit gate not met', 'gate':gate, 'finished_at':now()})
            return
        job('paired_training', [str(PYTHON), __file__, 'train'], OUT / 'paired_training/completed.json')
        jobs = []
        for length in (120,300):
            for a in ('off','on'):
                tag = f'aug_{a}_{length}'
                for mode in ('offline','online'):
                    cmd, output = eval_command(tag, OUT / f'{tag}.pt', mode)
                    jobs.append((f'{tag}_{mode}', cmd, output))
        write_json(OUT / 'status.json', {'status': 'running', 'stage': 'evaluating_paired_training',
                   'jobs': [name for name, _, _ in jobs], 'concurrent_evaluations': 2, 'started_at': now()})
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(job, name, cmd, output, True) for name, cmd, output in jobs]
            for future in futures:
                future.result()
        manifest = read(OUT / 'manifest.json')
        assert all(sha(Path(p)) == s for p,s in manifest['fixed_sha256'].items())
        verify(read(RUN / 'protocol.json'))
        for name, _, output in jobs:
            coverage = read(output.with_suffix('.coverage.json'))
            assert coverage['status'] == 'passed' and coverage['unique_sequences'] == 1177, name
        write_json(OUT / 'status.json', {'status':'complete', 'finished_at':now()})
        check_gpu(0)
        subprocess.run([str(PYTHON), str(ROOT / 'scripts/summarize_velocity_swap_ablation_20261019.py')], cwd=ROOT, check=True)
    except Exception:
        write_json(OUT / 'status.json', {'status':'failed', 'error':traceback.format_exc(), 'time':now()})
        raise


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    paired_train() if len(sys.argv)>1 and sys.argv[1]=='train' else run()
