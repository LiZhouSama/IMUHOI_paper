"""Reuse full evaluation only when best120 and best300 checkpoints are byte-identical."""
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import PYTHON, check_gpu, sha, write_json, now
from scripts.run_velocity_swap_ablation_20261019 import OUT, RUN
from scripts.run_omomo_original_recipe_20261018 import read, verify


def run():
    check_gpu(0)
    assert read(OUT / 'paired_training/completed.json')['matched_epochs_inputs_and_rng'] == 300
    verify(read(RUN / 'protocol.json'))
    manifest = read(OUT / 'manifest.json')
    assert all(sha(Path(p)) == digest for p, digest in manifest['fixed_sha256'].items())
    reused = []
    for arm in ('off', 'on'):
        directory = OUT / 'paired_training' / arm
        before_hash, after_hash = sha(directory / 'best120.pt'), sha(directory / 'best300.pt')
        if before_hash != after_hash:
            continue
        for mode in ('offline', 'online'):
            source = OUT / f'aug_{arm}_120_{mode}.json'
            target = OUT / f'aug_{arm}_300_{mode}.json'
            if not source.exists() or not source.with_suffix('.coverage.json').exists():
                continue
            lock = target.with_suffix('.lock').open('w')
            with lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if target.exists() and target.with_suffix('.coverage.json').exists():
                    continue
                coverage = read(source.with_suffix('.coverage.json'))
                assert coverage['status'] == 'passed' and coverage['unique_sequences'] == 1177
                receipt = {
                    'source': str(source), 'target': str(target), 'source_result_sha256': sha(source),
                    'head_checkpoint_sha256_equal': before_hash, 'verified_at': now(),
                    'reason': 'best120 and best300 checkpoint files are byte-identical; same fixed contact/HP/OT/config/data',
                    'training_completed_epochs': 300, 'reused_evaluation_only': True,
                }
                for suffix in ('.diagnostics.json', '.coverage.json'):
                    target.with_suffix(suffix).write_bytes(source.with_suffix(suffix).read_bytes())
                write_json(target.with_suffix('.reuse.json'), receipt)
                target.write_bytes(source.read_bytes())
                reused.append(receipt)
    write_json(OUT / 'identical_checkpoint_reuse.json', {'reused': reused, 'checked_at': now()})
    print(f'Reused {len(reused)} identical-checkpoint evaluations.')


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    run()
