"""Null-control for native recurrent dropout in separate training processes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu, PYTHON, write_json
OUT = ROOT / 'outputs/analysis/velocity_swap_aug_ablation_20261019'


def worker(arm):
    check_gpu(2000)
    import torch
    from model.rnn.base import RNNWithInit
    from utils.utils import setup_seed
    setup_seed(10)
    model = torch.nn.ModuleList([RNNWithInit(54, 6, 128, 6, 2, dropout=.2),
                                 RNNWithInit(9, 3, 128, 3, 2, dropout=.2)]).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001, foreach=False)
    outputs, gradients, rngs = [], [], []
    for step in range(3):
        setup_seed(10 + step)
        xs = [torch.randn(4, 120, n, device='cuda') for n in (54, 9)]
        inits = [torch.randn(4, n, device='cuda') for n in (6, 3)]
        targets = [torch.randn(4, 120, n, device='cuda') for n in (6, 3)]
        optimizer.zero_grad(set_to_none=True)
        rngs.append(hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest())
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1000010 + step)
            torch.cuda.manual_seed_all(1000010 + step)
            if arm == 'on_zero_noise':
                chosen = torch.rand(4, 1, device='cuda') < .5
                inits = [v + torch.randn_like(v) * chosen * 0. for v in inits]
        predictions = [m((x, i)) for m, x, i in zip(model, xs, inits)]
        digest = hashlib.sha256()
        for p in predictions:
            digest.update(p.detach().cpu().numpy().tobytes())
        outputs.append(digest.hexdigest())
        loss = sum((p-t).square().mean() * 10 for p, t in zip(predictions, targets))
        loss.backward()
        digest = hashlib.sha256()
        for p in model.parameters():
            digest.update(p.grad.detach().cpu().numpy().tobytes())
        gradients.append(digest.hexdigest())
        optimizer.step()
    write_json(OUT / f'dropout_process_{arm}.json', {'outputs': outputs, 'gradients': gradients, 'rngs': rngs})


def run():
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '10'}
    for arm in ('off', 'on_zero_noise'):
        check_gpu(2000)
        subprocess.run([str(PYTHON), __file__, arm], cwd=ROOT, env=env, check=True)
    left, right = [json.loads((OUT / f'dropout_process_{a}.json').read_text()) for a in ('off', 'on_zero_noise')]
    assert left == right, (left, right)
    write_json(OUT / 'dropout_process_pairing_check.json', {
        'passed': True, 'steps': 3, 'two_velocity_heads': True,
        'output_gradient_rng_hashes_exact': True,
        'null_control': 'augmentation draws occur in one arm, noise amplitude zero; native CUDA LSTM remains unchanged',
    })
    print('PASS: isolated-process outputs, gradients and RNG states match exactly for 3 updates.')


if __name__ == '__main__':
    worker(sys.argv[1]) if len(sys.argv) > 1 else run()
