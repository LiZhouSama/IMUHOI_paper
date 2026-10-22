"""Inspect effective losses and local gradient directions without optimizer updates."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_vc_joint_20261019 import OUT, RUN, ARMS, clean
from scripts.validate_egoimu_vc_recipe import check_gpu, PYTHON, write_json


def main():
    check_gpu(3500)
    import torch
    import yaml
    from easydict import EasyDict
    from torch.utils.data import DataLoader
    from dataset.dataset_IMUHOI import IMUDataset
    from scripts.run_omomo_original_recipe_20261018 import init_models, load_weights, EXP
    from train.rnn.train_object_trans import Stage3JointTrainer
    from utils.utils import build_model_input_dict, setup_seed

    cfg = EasyDict(yaml.safe_load((RUN / 'proposed/joint/recipe.yaml').read_text()))
    cfg.save_dir = str(OUT / 'unused_trainer_directory')
    cfg.use_tensorboard = False
    protocol = json.loads((RUN / 'protocol.json').read_text())
    # Fixed dispersed training members; this is a local gradient diagnostic, not an evaluation metric.
    names = protocol['members']['train'][::170][:24]
    ds = IMUDataset(data_dir=[str(EXP / 'data/train')], window_size=120,
                    sequence_paths=[str(EXP / 'data/train' / n) for n in names], debug=False,
                    random_window=True, simulate_imu_noise=False, require_canonical_skeleton=True,
                    resolve_bimanual_contact_conflicts=False, obj_points_sample_count=int(cfg.mesh_downsample_points))
    for sequence in ds.loaded_data.values():
        for key in ('human_imu', 'obj_imu', 'ori_root_reduced', 'root_vel', 'obj_vel', 'sensor_vel_root', 'sensor_vel_glb', 'obj_rot_6d'):
            sequence.pop(key, None)
    loader = DataLoader(ds, batch_size=8, num_workers=0, shuffle=False)
    hp, vc, ot = init_models(cfg, 'proposed', 'joint', OUT)
    load_weights(ot, RUN / 'proposed/joint/best_object_trans.pt', torch.device(cfg.device))
    trainer = Stage3JointTrainer(cfg, vc, hp, ot, loader, None, joint_train=True)
    results = {'purpose': 'local gradient diagnostic; 24 train members, no optimizer step', 'members': names,
               'effective_selection': 'six original OT test terms plus VC compute_test_loss; runner override',
               'vc_test_keys': sorted(trainer.vc_loss_fn.TEST_LOSS_KEYS),
               'ot_loss_keys': sorted(trainer.ot_loss_fn.LOSS_KEYS), 'arms': {}}
    for arm in ('prejoint', 'joint_best'):
        load_weights(vc, ARMS[arm], torch.device(cfg.device))
        setup_seed(10)
        hp.eval(); vc.train(); ot.train()
        # Match the original-recipe frozen-HP dropout restoration.
        for layer in hp.modules():
            if isinstance(layer, (torch.nn.Dropout, torch.nn.RNNBase)):
                layer.train()
        rows = []
        params = list(vc.obj_vel_net.parameters())
        for batch in loader:
            data = build_model_input_dict(batch, cfg, trainer.device, add_noise=True)
            _, vc_out, _, ot_out = trainer.forward_all(data, batch=batch, epoch=100, training=True)
            vc_total, _, vl = trainer.vc_loss_fn(vc_out, batch, trainer.device)
            ot_total, _, ol = trainer.ot_loss_fn(ot_out, batch, trainer.device)
            terms = {'velocity_mse_x10': vl['obj_vel'],
                     'contact_total': sum(v for k, v in vl.items() if k not in ('obj_vel', 'hand_vel')),
                     'ot_total': ot_total}
            gradients = {}
            for key, loss in terms.items():
                g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                gradients[key] = torch.cat([(torch.zeros_like(p) if x is None else x).reshape(-1) for p, x in zip(params, g)])
            base = gradients['velocity_mse_x10']
            directions = {}
            for key, g in gradients.items():
                directions[key] = {'norm': g.norm().item(), 'cosine_vs_velocity_mse':
                    ((g @ base) / (g.norm() * base.norm()).clamp_min(1e-20)).item()}
            rows.append({'weighted_vc_losses': {k: v.item() for k, v in vl.items()},
                         'weighted_ot_losses': {k: v.item() for k, v in ol.items()},
                         'object_velocity_head_gradients': directions})
        results['arms'][arm] = rows
        write_json(OUT / 'gradient_audit.json', clean(results))
    metadata = {}
    for arm, path in ARMS.items():
        payload = torch.load(path, map_location='cpu')
        metadata[arm] = {k: payload.get(k) for k in ('epoch', 'loss', 'test_components', 'checkpoint_selection_keys')}
    write_json(OUT / 'checkpoint_metadata.json', clean(metadata))


if __name__ == '__main__':
    assert Path(sys.executable).resolve() == PYTHON.resolve()
    main()
