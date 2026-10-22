"""Full-sequence velocity-only diagnostics, including the baseline reference."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_vc_joint_20261019 import OUT, RUN, ARMS
from scripts.validate_egoimu_vc_recipe import check_gpu, PYTHON, write_json


def main():
    check_gpu(2500)
    import torch
    import yaml
    from collections import defaultdict
    from easydict import EasyDict
    from dataset.dataset_IMUHOI import IMUDataset
    from model.rnn.velocity_contact import build_velocity_contact_module
    from scripts.run_omomo_original_recipe_20261018 import load_weights, EXP
    from utils.utils import build_model_input_dict, setup_seed
    from configs import FRAME_RATE
    cfg = EasyDict(yaml.safe_load((RUN / 'proposed/joint/recipe.yaml').read_text()))
    setup_seed(10)
    ds = IMUDataset(data_dir=str(EXP / 'data/test'), window_size=120, debug=False,
                    simulate_imu_noise=False, min_obj_contact_frames=0, full_sequence=True,
                    resolve_bimanual_contact_conflicts=False, require_canonical_skeleton=True)
    assert len(ds) == 1177
    groups = defaultdict(list)
    for i, info in enumerate(ds.sequence_info):
        groups[info['seq_len']].append(i)
    batches = [ids[s:s+8] for ids in groups.values() for s in range(0, len(ids), 8)]
    loader = torch.utils.data.DataLoader(ds, batch_sampler=batches, num_workers=0)
    models = {}
    for arm, path in {'baseline': RUN / 'baseline/vc/best.pt', **ARMS}.items():
        model_cfg = EasyDict(dict(cfg))
        model_cfg.velocity_contact_arch = 'imuhoi' if arm == 'baseline' else 'v5'
        model = build_velocity_contact_module(model_cfg).to(cfg.device).eval()
        load_weights(model, path, torch.device(cfg.device))
        models[arm] = model
    results = {arm: [] for arm in models}
    with torch.no_grad():
        for batch in loader:
            data = build_model_input_dict(batch, cfg, torch.device(cfg.device), add_noise=False)
            target = batch['obj_vel'].to(cfg.device)
            position = batch['obj_trans'].to(cfg.device)
            for arm, model in models.items():
                velocity = model(data, velocity_only=True)['pred_obj_vel']
                residual = velocity - target
                integral = data['obj_trans_init'][:, None] + torch.cat(
                    (torch.zeros_like(velocity[:, :1]), velocity[:, :-1].cumsum(1) / FRAME_RATE), 1)
                errors = (integral - position).norm(dim=-1) * 100
                for i in range(velocity.shape[0]):
                    r, e = residual[i], errors[i]
                    results[arm].append({'seq_path': batch['seq_path'][i], 'frames': r.shape[0],
                        'velocity_mse_m2ps2': r.square().mean().item(),
                        'velocity_vector_error_mps': r.norm(dim=-1).mean().item(),
                        'velocity_bias_norm_mps': r.mean(0).norm().item(),
                        'velocity_first30_bias_norm_mps': r[:30].mean(0).norm().item(),
                        'integral_obj_cm': e.mean().item(), 'integral_end_cm': e[-1].item(),
                        'integral_first30_cm': e[:30].mean().item(), 'integral_first60_cm': e[:60].mean().item()})
    means = {}
    for arm, rows in results.items():
        assert len(rows) == len({r['seq_path'] for r in rows}) == 1177
        means[arm] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k not in ('seq_path', 'frames')}
    for arm, ref in [('baseline', 'baseline'), ('joint_best', 'proposed')]:
        expected = json.loads((RUN / ref / 'eval_offline_best.json').read_text())['results']['omomo']['obj_trans_err_imu']
        assert abs(means[arm]['integral_obj_cm'] - expected) < .002, (arm, means[arm], expected)
    write_json(OUT / 'velocity_head_diagnostics.json', {'mode': 'offline', 'unique_sequences': 1177,
               'aggregation': 'equal per-sequence means', 'means': means, 'arms': results})
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    main()
