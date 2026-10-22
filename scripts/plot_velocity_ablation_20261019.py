"""Export validation curves and fixed-update integration results."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_egoimu_vc_recipe import check_gpu
from scripts.run_omomo_original_recipe_20261018 import read
from scripts.run_velocity_swap_ablation_20261019 import OUT


def run():
    check_gpu(0)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 160, 'savefig.dpi': 240})
    history = {a: read(OUT / 'paired_training' / a / 'history.json') for a in ('off', 'on')}
    colors = {'off': '#2365a8', 'on': '#d36c28'}
    labels = {'off': 'No initializer augmentation', 'on': 'Initializer augmentation'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis, component, title in zip(axes, ('obj_vel', 'hand_vel'), ('Object velocity', 'Hand velocity')):
        for arm in ('off', 'on'):
            rows = [r for r in history[arm] if 'validation_weighted_components' in r]
            axis.plot([r['epochs_completed'] for r in rows],
                      [r['validation_weighted_components'][component] / 10 for r in rows],
                      color=colors[arm], label=labels[arm], marker='o', markersize=3, linewidth=1.4)
        axis.axvline(120, color='#555555', linestyle='--', linewidth=.9)
        axis.axvline(50, color='#aaaaaa', linestyle=':', linewidth=.7)
        axis.axvline(150, color='#aaaaaa', linestyle=':', linewidth=.7)
        axis.set(title=title, xlabel='Completed velocity-training epochs', ylabel='Window validation MSE (m²/s²)', yscale='log')
        axis.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Paired seed-10 runs: K=1, W120, identical windows/noise/RNG; fixed contact/HP/OT', fontsize=10)
    fig.savefig(OUT / 'validation_curves.png')
    fig.savefig(OUT / 'validation_curves.pdf')
    plt.close(fig)
    results = OUT / 'fixed_budget_velocity.json'
    if not results.exists():
        return
    means = read(results)['means']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), constrained_layout=True)
    for axis, metric, factor, title, ylabel in (
        (axes[0], 'integral_obj_cm', 1, 'Full-sequence pure integration', 'Object position error (cm)'),
        (axes[1], 'velocity_bias_norm_mps', 100, 'Temporal mean velocity bias', 'Mean per-sequence bias norm (cm/s)'),
    ):
        x = np.arange(2)
        for i, arm in enumerate(('off', 'on')):
            values = [means[f'{arm}_final{n}'][metric] * factor for n in (120, 300)]
            bars = axis.bar(x + (i - .5) * .34, values, .34, color=colors[arm], label=labels[arm])
            axis.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        axis.set_xticks(x, ['120 epochs', '300 epochs'])
        axis.set(title=title, ylabel=ylabel)
        axis.set_ylim(0, axis.get_ylim()[1] * 1.15)
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Fixed final checkpoints, 1177 OMOMO sequences; lower is better', fontsize=10)
    fig.savefig(OUT / 'fixed_budget_effects.png')
    fig.savefig(OUT / 'fixed_budget_effects.pdf')
    plt.close(fig)
    print('Saved validation_curves and fixed_budget_effects as PNG/PDF.')


if __name__ == '__main__':
    run()
