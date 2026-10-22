"""Check CUDA dropout pairing before the velocity augmentation experiment."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu, write_json

check_gpu(2000)
import torch
from model.rnn.base import RNNWithInit
from utils.utils import setup_seed

setup_seed(10)
a = RNNWithInit(9, 3, 128, 3, 2, dropout=.2).cuda().train()
b = copy.deepcopy(a)
x = torch.randn(4, 120, 9, device='cuda')
init = torch.randn(4, 3, device='cuda')
results = {}
for method in ('restore_state', 'reseed'):
    differences = []
    for step in range(3):
        cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        if method == 'reseed':
            torch.manual_seed(10 + step)
            torch.cuda.manual_seed_all(10 + step)
        p = a((x, init))
        if method == 'restore_state':
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state_all(cuda)
        else:
            torch.manual_seed(10 + step)
            torch.cuda.manual_seed_all(10 + step)
        q = b((x, init))
        differences.append((p-q).abs().max().item())
    results[method] = differences
results['torch_version'] = torch.__version__
print(json.dumps(results))
write_json(ROOT / 'outputs/analysis/velocity_swap_aug_ablation_20261019/dropout_pairing_check.json', results)
