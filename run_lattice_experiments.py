"""Run all lattice experiment patterns sequentially and produce comparison summary."""

import os
import json
import importlib

from train.trainer import train_and_evaluate
from eval.analysis import run_full_evaluation


EXPERIMENTS = [
    ('lattice_standard', 'configs.lattice_standard'),
    ('lattice_halfplane', 'configs.lattice_halfplane'),
    ('lattice_standard_wide', 'configs.lattice_standard_wide'),
]


def run_all(base_dir: str = 'runs') -> dict:
    """Run all lattice experiment patterns.

    Args:
        base_dir: Base directory for experiment outputs.

    Returns:
        Dictionary mapping experiment names to their summary metrics.
    """
    all_summaries = {}

    for name, config_module in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  Experiment: {name}")
        print(f"{'='*60}\n")

        module = importlib.import_module(config_module)
        config = module.get_config()
        workdir = os.path.join(base_dir, name)

        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, workdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, workdir)
        all_summaries[name] = summary

    # Save combined summary
    summary_path = os.path.join(base_dir, 'lattice_summaries.json')
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*60}")
    print("  All lattice experiments complete!")
    print(f"  Combined summary: {summary_path}")
    print(f"{'='*60}")

    # Print comparison table
    print(f"\n{'Experiment':<25} {'MSE':>10} {'MAE':>10} {'j-corr':>10}")
    print('-' * 57)
    for name, summary in all_summaries.items():
        recon = summary.get('reconstruction', {})
        mse = recon.get('mse', float('nan'))
        mae = recon.get('mae', float('nan'))
        j_corr = summary.get('j_correlation', {}).get('max_abs_correlation', float('nan'))
        print(f"{name:<25} {mse:>10.6f} {mae:>10.6f} {j_corr:>10.4f}")

    return all_summaries


if __name__ == '__main__':
    run_all()
