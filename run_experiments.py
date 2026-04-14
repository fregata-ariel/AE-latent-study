"""Run all experiment patterns sequentially and produce comparison summary."""

import os
import json
import importlib

from train.trainer import train_and_evaluate
from eval.analysis import run_full_evaluation


EXPERIMENTS = [
    ('t1_standard', 'configs.t1_standard'),
    ('t1_torus', 'configs.t1_torus'),
    ('t2_standard', 'configs.t2_standard'),
    ('t2_torus', 'configs.t2_torus'),
]


def run_all(base_dir: str = 'runs') -> dict:
    """Run all experiment patterns.

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
    summary_path = os.path.join(base_dir, 'all_summaries.json')
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*60}")
    print("  All experiments complete!")
    print(f"  Combined summary: {summary_path}")
    print(f"{'='*60}")

    # Print comparison table
    print(f"\n{'Experiment':<20} {'MSE':>10} {'MAE':>10}")
    print('-' * 42)
    for name, summary in all_summaries.items():
        recon = summary.get('reconstruction', {})
        mse = recon.get('mse', float('nan'))
        mae = recon.get('mae', float('nan'))
        print(f"{name:<20} {mse:>10.6f} {mae:>10.6f}")

    return all_summaries


if __name__ == '__main__':
    run_all()
