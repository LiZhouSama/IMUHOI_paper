"""Fixed-update and validation-best velocity diagnostics for both 300-epoch arms."""
import json
import math
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu, PYTHON, write_json, sha, now
from scripts.run_velocity_swap_ablation_20261019 import OUT, RUN, PREFIXES, state
from scripts.run_omomo_original_recipe_20261018 import EXP, read, verify


def run():
    check_gpu(3000)
    import torch
    import yaml
    from collections import defaultdict
    from easydict import EasyDict
    from dataset.dataset_IMUHOI import IMUDataset
    from model.rnn.velocity_contact import build_velocity_contact_module
    from utils.utils import build_model_input_dict, setup_seed
    from configs import FRAME_RATE
    protocol = read(RUN / 'protocol.json')
    verify(protocol)
    training = OUT / 'paired_training'
    assert read(training / 'completed.json')['matched_epochs_inputs_and_rng'] == 300
    cfg = EasyDict(yaml.safe_load((RUN / 'proposed/joint/recipe.yaml').read_text()))
    setup_seed(10)
    dataset = IMUDataset(data_dir=str(EXP / 'data/test'), window_size=120, debug=False,
                         simulate_imu_noise=False, min_obj_contact_frames=0, full_sequence=True,
                         resolve_bimanual_contact_conflicts=False, require_canonical_skeleton=True)
    assert len(dataset) == 1177
    groups = defaultdict(list)
    for index, item in enumerate(dataset.sequence_info):
        groups[item['seq_len']].append(index)
    batches = [indices[start:start + 8] for indices in groups.values() for start in range(0, len(indices), 8)]
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches, num_workers=0)
    fixed_path = RUN / 'proposed/joint/best_velocity_contact.pt'
    fixed = state(torch.load(fixed_path, map_location='cpu'))
    paths = {'baseline': RUN / 'baseline/vc/best.pt', 'joint_best': fixed_path}
    paths.update({f'{arm}_{choice}{budget}': training / arm / f'{choice}{budget}.pt'
                  for arm in ('off', 'on') for choice in ('best', 'final') for budget in (120, 300)})
    models, receipts = {}, {}
    for tag, path in paths.items():
        model_cfg = EasyDict(dict(cfg))
        model_cfg.velocity_contact_arch = 'imuhoi' if tag == 'baseline' else 'v5'
        model = build_velocity_contact_module(model_cfg).to(cfg.device).eval()
        payload = torch.load(path, map_location='cpu')
        checkpoint = state(payload)
        if tag not in ('baseline', 'joint_best'):
            assert set(checkpoint) == {key for key in fixed if key.startswith(PREFIXES)}
            model.load_state_dict(fixed, strict=True)
            model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=True)
        models[tag] = model
        receipts[tag] = {'checkpoint': str(path), 'sha256': sha(path), 'epoch': payload.get('epoch')}
    results = {tag: [] for tag in models}
    with torch.no_grad():
        for batch in loader:
            data = build_model_input_dict(batch, cfg, torch.device(cfg.device), add_noise=False)
            target = batch['obj_vel'].to(cfg.device)
            target_hand = batch['sensor_vel_glb'][:, :, -2:, :].to(cfg.device)
            position = batch['obj_trans'].to(cfg.device)
            for tag, model in models.items():
                prediction = model(data, velocity_only=True)
                velocity = prediction['pred_obj_vel']
                residual = velocity - target
                hand_residual = prediction['pred_hand_glb_vel'] - target_hand
                integral = data['obj_trans_init'][:, None] + torch.cat(
                    (torch.zeros_like(velocity[:, :1]), velocity[:, :-1].cumsum(1) / FRAME_RATE), 1)
                error = (integral - position).norm(dim=-1) * 100
                for i in range(velocity.shape[0]):
                    r, h, e = residual[i], hand_residual[i], error[i]
                    results[tag].append({
                        'seq_path': batch['seq_path'][i], 'frames': r.shape[0],
                        'velocity_mse_m2ps2': r.square().mean().item(),
                        'velocity_vector_error_mps': r.norm(dim=-1).mean().item(),
                        'velocity_bias_norm_mps': r.mean(0).norm().item(),
                        'velocity_first30_bias_norm_mps': r[:30].mean(0).norm().item(),
                        'hand_velocity_mse_m2ps2': h.square().mean().item(),
                        'hand_velocity_bias_norm_mps': h.mean(0).norm(dim=-1).mean().item(),
                        'integral_obj_cm': e.mean().item(), 'integral_end_cm': e[-1].item(),
                        'integral_first30_cm': e[:30].mean().item(), 'integral_first60_cm': e[:60].mean().item(),
                    })
            if len(results['baseline']) % 100 < velocity.shape[0]:
                write_json(OUT / 'fixed_budget_velocity.progress.json', {
                    'sequences': len(results['baseline']), 'updated_at': now()})
    means = {}
    for tag, rows in results.items():
        assert len(rows) == len({r['seq_path'] for r in rows}) == 1177
        keys = [k for k in rows[0] if k not in ('seq_path', 'frames')]
        assert all(math.isfinite(r[k]) for r in rows for k in keys)
        means[tag] = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    for tag, method in (('baseline', 'baseline'), ('joint_best', 'proposed')):
        expected = read(RUN / method / 'eval_offline_best.json')['results']['omomo']['obj_trans_err_imu']
        assert abs(means[tag]['integral_obj_cm'] - expected) < .002, (tag, means[tag], expected)
    effects = {}
    for choice in ('best', 'final'):
        effects[choice] = {}
        for key in means['baseline']:
            effects[choice][key] = {
                'augmentation_on_minus_off_120': means[f'on_{choice}120'][key] - means[f'off_{choice}120'][key],
                'augmentation_on_minus_off_300': means[f'on_{choice}300'][key] - means[f'off_{choice}300'][key],
                '300_minus_120_off': means[f'off_{choice}300'][key] - means[f'off_{choice}120'][key],
                '300_minus_120_on': means[f'on_{choice}300'][key] - means[f'on_{choice}120'][key],
            }
    paired_changes = {}
    for first, second, label in (
        ('off_final120', 'on_final120', 'augmentation_at_120'),
        ('off_final300', 'on_final300', 'augmentation_at_300'),
        ('off_final120', 'off_final300', 'longer_training_without_aug'),
        ('on_final120', 'on_final300', 'longer_training_with_aug'),
    ):
        before = {row['seq_path']: row for row in results[first]}
        joined = [(before[row['seq_path']], row) for row in results[second]]
        paired_changes[label] = {}
        for subset, members in (
            ('all', joined),
            ('up_to_120_frames', [(a, b) for a, b in joined if a['frames'] <= 120]),
            ('121_to_300_frames', [(a, b) for a, b in joined if 120 < a['frames'] <= 300]),
            ('over_300_frames', [(a, b) for a, b in joined if a['frames'] > 300]),
        ):
            if not members:
                continue
            paired_changes[label][subset] = {}
            for metric in ('integral_obj_cm', 'velocity_bias_norm_mps', 'velocity_mse_m2ps2'):
                deltas = [b[metric] - a[metric] for a, b in members]
                paired_changes[label][subset][metric] = {
                    'count': len(deltas), 'mean_delta': statistics.mean(deltas),
                    'median_delta': statistics.median(deltas),
                    'improved_count': sum(delta < 0 for delta in deltas),
                    'worsened_count': sum(delta > 0 for delta in deltas),
                }
    verify(protocol)
    write_json(OUT / 'fixed_budget_velocity.json', {
        'mode': 'offline', 'unique_sequences': 1177, 'aggregation': 'equal per-sequence means',
        'purpose': 'final120/final300 isolate fixed update budgets without best-checkpoint selection',
        'means': means, 'factorial_effects': effects, 'checkpoints': receipts, 'arms': results,
        'paired_changes_second_minus_first': paired_changes,
        'finished_at': now(),
    })
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    run()
