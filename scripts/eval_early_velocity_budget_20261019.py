"""Evaluate immutable 120-epoch snapshots while the 300-epoch runs continue."""
import fcntl
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import PYTHON, check_gpu, write_json, sha, now
from scripts.run_velocity_swap_ablation_20261019 import OUT, compose, state, eval_command, job
from scripts.run_omomo_original_recipe_20261018 import read


def run():
    check_gpu(0)
    lock = (OUT / 'early_120.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = OUT / 'early_120_status.json'
    write_json(status_path, {'stage': 'waiting_for_120_snapshots', 'started_at': now()})
    directory = OUT / 'paired_training'
    while True:
        counts = {}
        for arm in ('off', 'on'):
            path = directory / arm / 'progress.json'
            counts[arm] = read(path).get('epochs_completed', 0) if path.exists() else 0
        if min(counts.values()) >= 121:
            break
        if read(OUT / 'status.json')['status'] == 'failed':
            raise RuntimeError('Main experiment failed before early snapshots were ready')
        time.sleep(15)
    check_gpu(3000)
    import torch
    manifest, jobs = {}, []
    for arm in ('off', 'on'):
        source = directory / arm / 'best120.pt'
        checkpoint = torch.load(source, map_location='cpu')
        destination = OUT / f'early_aug_{arm}_120.pt'
        receipt = compose(state(checkpoint), destination)
        manifest[arm] = {'head_source': str(source), 'head_source_sha256': sha(source),
                         'selected_epoch': checkpoint['epoch'], 'composition': receipt}
        for mode in ('offline', 'online'):
            tag = f'aug_{arm}_120'
            cmd, output = eval_command(tag, destination, mode)
            jobs.append((f'{tag}_{mode}', cmd, output))
    write_json(OUT / 'early_120_manifest.json', manifest)
    write_json(status_path, {'stage': 'evaluating_120', 'jobs': [j[0] for j in jobs], 'started_at': now()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(job, name, cmd, output, True) for name, cmd, output in jobs]
        for future in futures:
            future.result()
    for arm, receipt in manifest.items():
        assert sha(Path(receipt['head_source'])) == receipt['head_source_sha256'], arm
    for name, _, output in jobs:
        coverage = read(output.with_suffix('.coverage.json'))
        assert coverage['status'] == 'passed' and coverage['unique_sequences'] == 1177, name
    write_json(status_path, {'stage': 'complete', 'finished_at': now()})


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    run()
