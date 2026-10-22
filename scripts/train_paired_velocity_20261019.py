"""Native CUDA velocity training in isolated, matched augmentation processes."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_velocity_swap_ablation_20261019 import (
    OUT, RUN, PREFIXES, state, heads_only, compose,
)
from scripts.validate_egoimu_vc_recipe import PYTHON, check_gpu, sha, write_json, now
from scripts.run_omomo_original_recipe_20261018 import read, verify

DIRECTORY = OUT / 'paired_training'


def digest_tensors(items):
    digest = hashlib.sha256()
    for key, value in sorted(items.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_arm(arm):
    check_gpu(3000)
    import torch
    import yaml
    from easydict import EasyDict
    from types import MethodType
    from scripts.run_omomo_original_recipe_20261018 import make_loaders
    from model.rnn.velocity_contact import build_velocity_contact_module
    from model.rnn.velocity_contact_IMUHOI import IMUHOIVelocityContactModule
    from utils.utils import setup_seed, build_model_input_dict
    from train.rnn.loss.velocity_contact_loss import VelocityContactLoss

    directory = DIRECTORY / arm
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (directory / 'completed.json').exists():
        return
    if (directory / 'history.json').exists():
        raise RuntimeError('Partial training exists. Native cuDNN dropout state cannot be exactly resumed; preserve it and restart both arms in a new run.')
    protocol = read(RUN / 'protocol.json')
    verify(protocol)
    cfg = EasyDict(yaml.safe_load((RUN / 'proposed/vc/recipe.yaml').read_text()))
    cfg.update(epochs=300, batch_size=100, num_workers=12, save_dir=str(directory),
               vc_staged_training=False, vc_velocity_stage_epochs=300,
               v5_init_noise_prob=.5 if arm == 'on' else 0.)
    assert cfg.loss_weights.obj_vel == cfg.loss_weights.hand_vel == 10
    assert cfg.velocity_contact_arch == 'v5' and cfg.seed == 10
    assert cfg.v5_obj_vel_init_noise_std == cfg.v5_hand_vel_init_noise_std == .2
    assert cfg.v5_contact_init_flip_prob == 0.
    assert cfg.milestones == [50, 150] and cfg.lr == .002 and cfg.weight_decay == .0001
    assert not cfg.amp_enabled and cfg.gradient_clip == 0.
    (directory / 'recipe.json').write_text(json.dumps(dict(cfg), indent=2))
    device = torch.device(cfg.device)
    setup_seed(10)
    train_loader, val_loader = make_loaders(cfg, protocol)
    assert len(train_loader) == 41 and len(train_loader.dataset) == 4115 and len(val_loader.dataset) == 1016
    setup_seed(10)
    base = IMUHOIVelocityContactModule(cfg).to(device)
    initial = heads_only(base.state_dict())
    del base
    fixed = state(torch.load(RUN / 'proposed/joint/best_velocity_contact.pt', map_location='cpu'))
    model = build_velocity_contact_module(cfg).to(device)
    model.load_state_dict(fixed, strict=True)
    assert set(initial) == {k for k in model.state_dict() if k.startswith(PREFIXES)}
    model.load_state_dict(initial, strict=False)
    for key, param in model.named_parameters():
        param.requires_grad_(key.startswith(PREFIXES))
    original_perturb = model._perturb_inits
    step_seed = [0]

    def perturb(self, h, o, c):
        # Initializer corruption must not advance input-noise or dropout RNGs.
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(1000000 + step_seed[0])
            torch.cuda.manual_seed_all(1000000 + step_seed[0])
            return original_perturb(h, o, c)

    model._perturb_inits = MethodType(perturb, model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=.002, weight_decay=.0001, foreach=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 150], gamma=.2)
    loss_fn = VelocityContactLoss(weights=cfg.loss_weights)
    write_json(directory / 'initialization.json', {
        'velocity_state_sha256': digest_tensors(initial),
        'trainable_tensors': [k for k, p in model.named_parameters() if p.requires_grad],
        'frozen_other_tensors': len(fixed) - len(initial),
        'active_losses': {'obj_vel': 10, 'hand_vel': 10},
        'native_cuda_lstm': True, 'process_id': os.getpid(),
    })
    torch.save({'module_state_dict': initial}, directory / 'initial_heads.pt')
    best, best_epoch, history = float('inf'), None, []
    for epoch in range(300):
        epoch_start = time.monotonic()
        setup_seed(10 + epoch)
        train_loader.generator.manual_seed(10 + epoch)
        expected_lr = .002 * .2 ** sum(epoch >= m for m in (50, 150))
        assert math.isclose(optimizer.param_groups[0]['lr'], expected_lr)
        model.train()
        sums = {'obj_vel': 0., 'hand_vel': 0.}
        input_digest, rng_digest = hashlib.sha256(), hashlib.sha256()
        for step, batch in enumerate(train_loader):
            step_seed[0] = 10 + epoch * 41 + step
            data = build_model_input_dict(batch, cfg, device, add_noise=True)
            selected = {k: data[k] for k in ('human_imu', 'obj_imu', 'hand_vel_glb_init', 'obj_vel_init')}
            selected.update(target_obj_vel=batch['obj_vel'], target_sensor_vel=batch['sensor_vel_glb'])
            input_digest.update(digest_tensors(selected).encode())
            rng_digest.update(digest_tensors({'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}).encode())
            optimizer.zero_grad(set_to_none=True)
            pred = model(data, velocity_only=True)
            loss, _, weighted = loss_fn(pred, batch, device, active_loss_keys={'obj_vel', 'hand_vel'})
            assert torch.isfinite(loss), (arm, epoch, step)
            loss.backward()
            optimizer.step()
            for key in sums:
                sums[key] += weighted[key].item() / 41
        record = {'arm': arm, 'epoch': epoch, 'epochs_completed': epoch + 1,
                  'updates': (epoch + 1) * 41, 'lr': expected_lr,
                  'train_weighted_components': sums, 'train_loss': sum(sums.values()),
                  'train_input_sha256': input_digest.hexdigest(), 'train_rng_sha256': rng_digest.hexdigest()}
        if epoch % 10 == 0:
            setup_seed(200010 + epoch)
            val_loader.generator.manual_seed(10 + epoch)
            model.eval()
            vals = {'obj_vel': 0., 'hand_vel': 0.}
            val_digest = hashlib.sha256()
            with torch.no_grad():
                for batch in val_loader:
                    data = build_model_input_dict(batch, cfg, device, add_noise=True)
                    selected = {k: data[k] for k in ('human_imu', 'obj_imu', 'hand_vel_glb_init', 'obj_vel_init')}
                    selected.update(target_obj_vel=batch['obj_vel'], target_sensor_vel=batch['sensor_vel_glb'])
                    val_digest.update(digest_tensors(selected).encode())
                    pred = model(data, velocity_only=True)
                    _, _, weighted = loss_fn(pred, batch, device, active_loss_keys={'obj_vel', 'hand_vel'})
                    for key in vals:
                        vals[key] += weighted[key].item() / len(val_loader)
            score = sum(vals.values())
            assert math.isfinite(score), (arm, epoch, vals)
            record.update(validation_weighted_components=vals, validation_loss=score,
                          validation_input_sha256=val_digest.hexdigest())
            if score < best:
                best, best_epoch = score, epoch
                torch.save({'module_state_dict': heads_only(model.state_dict()), 'epoch': epoch,
                            'validation_loss': score, 'weighted_components': vals}, directory / 'best300.pt')
        record.update(best_validation_loss=best, best_epoch=best_epoch,
                      elapsed_seconds=time.monotonic() - epoch_start, updated_at=now())
        history.append(record)
        write_json(directory / 'history.json', history)
        write_json(directory / 'progress.json', record)
        print(json.dumps(record), flush=True)
        if epoch == 119:
            (directory / 'best120.pt').write_bytes((directory / 'best300.pt').read_bytes())
            torch.save({'module_state_dict': heads_only(model.state_dict()), 'epoch': epoch}, directory / 'final120.pt')
        scheduler.step()
        if epoch % 10 == 0 or epoch in (119, 299):
            temporary = directory / 'last.pt.tmp'
            torch.save({'epoch': epoch, 'best': best, 'best_epoch': best_epoch,
                        'module_state_dict': heads_only(model.state_dict()),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, temporary)
            temporary.replace(directory / 'last.pt')
    torch.save({'module_state_dict': heads_only(model.state_dict()), 'epoch': 299}, directory / 'final300.pt')
    assert all(torch.equal(model.state_dict()[k].cpu(), fixed[k]) for k in fixed if not k.startswith(PREFIXES))
    verify(protocol)
    write_json(directory / 'completed.json', {'epochs': 300, 'updates': 12300, 'finished_at': now(),
               'frozen_other_tensors_exact': True, 'best_epoch': best_epoch, 'best_validation_loss': best})


def run():
    check_gpu(7000)
    assert read(OUT / 'dropout_process_pairing_check.json')['passed']
    for mode in ('offline', 'online'):
        coverage = read(OUT / f'swap_{mode}.coverage.json')
        assert coverage['status'] == 'passed' and coverage['unique_sequences'] == 1177
    authorization = OUT / 'training_authorization.json'
    assert read(OUT / 'gate.json')['passed'] or (authorization.exists() and read(authorization)['authorized'])
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    lock = (DIRECTORY / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(DIRECTORY / 'pairing_protocol.json', {
        'method': 'separate native CUDA processes, matching calls and seeds; initializer RNG isolated',
        'reason': 'Torch 1.13 cuDNN recurrent dropout cannot be paired by restoring torch RNG inside one process',
        'inputs': 'all training/validation inputs and velocity targets hashed each batch, compared each epoch',
        'rng': 'CPU/CUDA generator states hashed before every training forward; compared each epoch',
        'seed': 10, 'epoch_seed': '10+epoch', 'validation_seed': '200010+epoch',
        'selection': '10*object velocity MSE + 10*hand velocity MSE',
        'source_sha256': sha(Path(__file__)), 'created_at': now(),
    })
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '10'}
    children, logs = {}, []
    try:
        for arm in ('off', 'on'):
            if (DIRECTORY / arm / 'completed.json').exists():
                continue
            check_gpu(3000)
            log = (DIRECTORY / f'{arm}.log').open('a')
            logs.append(log)
            children[arm] = subprocess.Popen([str(PYTHON), __file__, arm], cwd=ROOT,
                                            stdout=log, stderr=subprocess.STDOUT, env=env)
        while any(p.poll() is None for p in children.values()):
            for arm, process in children.items():
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'{arm} training failed, code {process.returncode}; see {arm}.log')
            histories = {a: read(DIRECTORY / a / 'history.json') if (DIRECTORY / a / 'history.json').exists() else []
                         for a in ('off', 'on')}
            matched = min(len(h) for h in histories.values())
            for index in range(matched):
                left, right = histories['off'][index], histories['on'][index]
                for key in ('epoch', 'train_input_sha256', 'train_rng_sha256', 'validation_input_sha256'):
                    assert left.get(key) == right.get(key), (index, key, left.get(key), right.get(key))
            write_json(DIRECTORY / 'progress.json', {
                'arms': {a: h[-1] if h else None for a, h in histories.items()},
                'matched_epochs_inputs_and_rng': matched, 'updated_at': now(),
            })
            time.sleep(10)
        assert all(p.returncode == 0 for p in children.values())
    finally:
        for process in children.values():
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    check_gpu(3000)
    import torch
    receipts = {}
    histories = {a: read(DIRECTORY / a / 'history.json') for a in ('off', 'on')}
    assert all(len(h) == 300 for h in histories.values())
    initial = [read(DIRECTORY / a / 'initialization.json')['velocity_state_sha256'] for a in ('off', 'on')]
    assert initial[0] == initial[1]
    for index in range(300):
        for key in ('epoch', 'train_input_sha256', 'train_rng_sha256', 'validation_input_sha256'):
            assert histories['off'][index].get(key) == histories['on'][index].get(key), (index, key)
    for arm in ('off', 'on'):
        assert read(DIRECTORY / arm / 'completed.json')['frozen_other_tensors_exact']
        for length in (120, 300):
            checkpoint = torch.load(DIRECTORY / arm / f'best{length}.pt', map_location='cpu')
            tag = f'aug_{arm}_{length}'
            receipts[tag] = compose(state(checkpoint), OUT / f'{tag}.pt')
            receipts[tag]['velocity_best_epoch'] = checkpoint['epoch']
    verify(read(RUN / 'protocol.json'))
    write_json(DIRECTORY / 'completed.json', {
        'status': 'complete', 'epochs': 300, 'updates_per_arm': 12300,
        'matched_epochs_inputs_and_rng': 300, 'velocity_initial_state_sha256': initial[0],
        'compositions': receipts, 'finished_at': now(),
    })


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    train_arm(sys.argv[1]) if len(sys.argv) > 1 else run()
