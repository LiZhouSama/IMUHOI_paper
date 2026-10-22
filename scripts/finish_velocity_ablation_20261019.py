"""Complete fixed-budget diagnostics and reporting after the paired training."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu, PYTHON, write_json, now
from scripts.run_velocity_swap_ablation_20261019 import OUT
from scripts.run_omomo_original_recipe_20261018 import read


def run():
    check_gpu(0)
    lock = (OUT / 'postprocess.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = OUT / 'postprocess_status.json'
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': '10'}

    def execute(stage, script):
        check_gpu(0)
        write_json(status, {'stage': stage, 'started_at': now()})
        with (OUT / f'{stage}.log').open('a') as log:
            subprocess.run([str(PYTHON), str(ROOT / 'scripts' / script)], cwd=ROOT,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        write_json(status, {'stage': 'waiting_for_training', 'started_at': now()})
        while not (OUT / 'paired_training/completed.json').exists():
            if read(OUT / 'status.json')['status'] == 'failed':
                raise RuntimeError('Main experiment failed before training completed')
            time.sleep(15)
        if not (OUT / 'fixed_budget_velocity.json').exists():
            execute('fixed_budget_velocity', 'probe_paired_velocity_budget_20261019.py')
        write_json(status, {'stage': 'waiting_for_full_evaluation', 'updated_at': now()})
        # The main process holds this lock through all evaluations and its report.
        main_lock = (OUT / 'run.lock').open('r')
        fcntl.flock(main_lock, fcntl.LOCK_EX)
        assert read(OUT / 'status.json')['status'] == 'complete'
        execute('plots', 'plot_velocity_ablation_20261019.py')
        execute('final_report', 'summarize_velocity_swap_ablation_20261019.py')
        write_json(status, {'stage': 'complete', 'finished_at': now()})
    except Exception:
        write_json(status, {'stage': 'failed', 'error': traceback.format_exc(), 'time': now()})
        raise


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    run()
