"""Evaluate fixed final300 velocity checkpoints after the selected-best jobs."""
import fcntl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_velocity_swap_ablation_20261019 import OUT, RUN, compose, state, eval_command, job
from scripts.run_omomo_original_recipe_20261018 import read, verify
from scripts.validate_egoimu_vc_recipe import PYTHON, check_gpu, write_json, sha, now


def run():
    check_gpu(0)
    status = OUT / 'final300_status.json'
    own_lock = (OUT / 'final300.lock').open('w')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(status, {'stage': 'queued_after_best_evaluations', 'created_at': now()})
    try:
        # Acquire in the same order as the existing postprocessor.
        post_lock = (OUT / 'postprocess.lock').open('r')
        fcntl.flock(post_lock, fcntl.LOCK_EX)
        main_lock = (OUT / 'run.lock').open('r')
        fcntl.flock(main_lock, fcntl.LOCK_EX)
        assert read(OUT / 'status.json')['status'] == 'complete'
        check_gpu(6000)
        import torch
        receipts, jobs = {}, []
        for arm in ('off', 'on'):
            source = OUT / 'paired_training' / arm / 'final300.pt'
            tag = f'final_{arm}_300'
            checkpoint = OUT / f'{tag}.pt'
            if not checkpoint.exists():
                receipts[arm] = compose(state(torch.load(source, map_location='cpu')), checkpoint)
            else:
                receipts[arm] = {'sha256': sha(checkpoint)}
            receipts[arm].update({'source': str(source), 'source_sha256': sha(source)})
            for mode in ('offline', 'online'):
                cmd, marker = eval_command(tag, checkpoint, mode)
                jobs.append((f'{tag}_{mode}', cmd, marker))
        write_json(OUT / 'final300_manifest.json', {
            'purpose': 'Test fixed-300-epoch results without window-validation checkpoint selection',
            'fixed_components': 'Same contact parameters, HP and OT as prior paired evaluations',
            'receipts': receipts, 'created_at': now()})
        write_json(status, {'stage': 'evaluating', 'jobs': [j[0] for j in jobs], 'started_at': now()})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(job, name, cmd, marker, True) for name, cmd, marker in jobs]
            for future in futures:
                future.result()
        for name, _, marker in jobs:
            coverage = read(marker.with_suffix('.coverage.json'))
            assert coverage['status'] == 'passed' and coverage['unique_sequences'] == 1177, name
        fixed = read(OUT / 'manifest.json')['fixed_sha256']
        assert all(sha(Path(path)) == digest for path, digest in fixed.items())
        verify(read(RUN / 'protocol.json'))
        check_gpu(0)
        subprocess.run([str(PYTHON), str(ROOT / 'scripts/summarize_velocity_swap_ablation_20261019.py')], cwd=ROOT, check=True)
        write_json(status, {'stage': 'complete', 'finished_at': now()})
    except Exception:
        write_json(status, {'stage': 'failed', 'error': traceback.format_exc(), 'time': now()})
        raise


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    run()
