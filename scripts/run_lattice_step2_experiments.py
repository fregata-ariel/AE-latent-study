"""Run Step 2 lattice experiments and generate a comparison report."""

import importlib
import json
import os
from collections.abc import Callable

from train.trainer import train_and_evaluate
from eval.analysis import run_full_evaluation


EXPERIMENTS = [
    ('lattice_standard_norm', 'configs.lattice_standard_norm'),
    ('lattice_halfplane_norm', 'configs.lattice_halfplane_norm'),
    ('lattice_standard_wide_norm', 'configs.lattice_standard_wide_norm'),
    ('lattice_standard_norm_inv', 'configs.lattice_standard_norm_inv'),
    ('lattice_standard_wide_norm_inv', 'configs.lattice_standard_wide_norm_inv'),
    ('lattice_standard_norm_latent4', 'configs.lattice_standard_norm_latent4'),
    ('lattice_standard_norm_latent8', 'configs.lattice_standard_norm_latent8'),
    ('lattice_vae_norm_beta001', 'configs.lattice_vae_norm_beta001'),
]

PHASES = {
    'Phase 1': [
        'lattice_standard_norm',
        'lattice_halfplane_norm',
        'lattice_standard_wide_norm',
    ],
    'Phase 2': [
        'lattice_standard_norm_inv',
        'lattice_standard_wide_norm_inv',
    ],
    'Phase 3': [
        'lattice_standard_norm_latent4',
        'lattice_standard_norm_latent8',
        'lattice_vae_norm_beta001',
    ],
}

STEP1_BASELINE_FALLBACK = {
    'lattice_standard': {
        'reconstruction': {'mse': 4.255687817931175e-08},
        'j_correlation': {'max_abs_correlation': 0.008139114111514476},
        'modular_invariance': {'mean_latent_distance': 1.0194441080093384},
    },
    'lattice_halfplane': {
        'reconstruction': {'mse': 2.0063703879714012e-07},
        'j_correlation': {'max_abs_correlation': 0.007602388523639556},
        'modular_invariance': {'mean_latent_distance': 2.0525264739990234},
    },
    'lattice_standard_wide': {
        'reconstruction': {'mse': 5.338775217533111e-06},
        'j_correlation': {'max_abs_correlation': 0.016996238112451878},
        'modular_invariance': {'mean_latent_distance': 0.9578040838241577},
    },
}


def _load_config(config_source: str | Callable[[], object]):
    if callable(config_source):
        return config_source()

    module = importlib.import_module(config_source)
    return module.get_config()


def _selection_key(summary: dict) -> tuple[float, float, float]:
    modular = summary.get('modular_invariance', {}).get(
        'mean_latent_distance', float('inf'),
    )
    spearman = summary.get('j_correlation', {}).get(
        'max_abs_logabsj_spearman', float('-inf'),
    )
    mse = summary.get('reconstruction', {}).get('mse', float('inf'))
    return (modular, -spearman, mse)


def _select_best_run(run_names: list[str], summaries: dict[str, dict]) -> tuple[str, dict] | tuple[None, None]:
    candidates = [(name, summaries[name]) for name in run_names if name in summaries]
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: _selection_key(item[1]))


def write_step2_report(
    step2_summaries: dict[str, dict],
    output_path: str = 'walkthrough-lattice-step2.md',
    baseline_path: str = 'runs/lattice_summaries.json',
) -> None:
    """Write a Step 2 markdown report from summary JSON data."""
    baseline = {}
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
    if not baseline:
        baseline = STEP1_BASELINE_FALLBACK

    lines = [
        '# Lattice Step 2 — Normalization, Invariance, and Capacity Sweep',
        '',
        '## Summary',
        '',
        '- Step 2 compares normalized lattice signals, modular-invariance regularization, and latent-capacity expansions.',
        '- Ranking rule: smallest modular mean distance, then largest `log10|j|` Spearman correlation, then smallest reconstruction MSE.',
        '- Quotient-chart metrics (`trust`, `overlap`, `eff_dim`) are supplementary and track how natural the learned 2D chart looks.',
        '- Normalized runs should be compared against one another, not directly against raw Step 1 MSE values.',
        '',
        '## Step 1 Baseline',
        '',
        '| Experiment | MSE | max abs corr vs Re/Im(j) | SL₂(Z) mean dist |',
        '|---|---|---|---|',
    ]

    baseline_order = [
        'lattice_standard',
        'lattice_halfplane',
        'lattice_standard_wide',
    ]
    for name in baseline_order:
        summary = baseline.get(name)
        if summary is None:
            continue
        recon = summary.get('reconstruction', {})
        j_corr = summary.get('j_correlation', {})
        mod = summary.get('modular_invariance', {})
        lines.append(
            f"| {name} | {recon.get('mse', float('nan')):.6g} | "
            f"{j_corr.get('max_abs_correlation', float('nan')):.4f} | "
            f"{mod.get('mean_latent_distance', float('nan')):.4f} |"
        )

    for phase_name, run_names in PHASES.items():
        lines.extend([
            '',
            f'## {phase_name}',
            '',
            '| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | trust | overlap | eff_dim | PCA EVR |',
            '|---|---|---|---|---|---|---|---|---|',
        ])
        for name in run_names:
            summary = step2_summaries.get(name)
            if summary is None:
                lines.append(
                    f'| {name} | pending | pending | pending | pending | pending | pending | pending | pending |'
                )
                continue

            recon = summary.get('reconstruction', {})
            j_corr = summary.get('j_correlation', {})
            mod = summary.get('modular_invariance', {})
            chart = summary.get('chart_quality', {})
            pca = summary.get('pca_explained_variance', [])
            pca_text = (
                f"{pca[0]:.3f}, {pca[1]:.3f}" if len(pca) >= 2 else '-'
            )
            lines.append(
                f"| {name} | {recon.get('mse', float('nan')):.6g} | "
                f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
                f"{j_corr.get('max_logabsj_mutual_info', float('nan')):.4f} | "
                f"{mod.get('mean_latent_distance', float('nan')):.4f} | "
                f"{chart.get('trustworthiness', float('nan')):.4f} | "
                f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
                f"{chart.get('effective_dimension', float('nan')):.4f} | "
                f"{pca_text} |"
            )

        best_name, best_summary = _select_best_run(run_names, step2_summaries)
        if best_name is None:
            lines.extend(['', '- Best run: pending'])
        else:
            best_chart = best_summary.get('chart_quality', {})
            lines.extend([
                '',
                f"- Best run: `{best_name}`",
                f"- Reason: modular mean distance = "
                f"{best_summary['modular_invariance']['mean_latent_distance']:.4f}, "
                f"log10|j| Spearman = "
                f"{best_summary['j_correlation'].get('max_abs_logabsj_spearman', float('nan')):.4f}, "
                f"MSE = {best_summary['reconstruction']['mse']:.6g}",
                f"- Orbit-gluing view: smaller modular mean distance is better.",
                f"- 2D chart view: higher trust/overlap and effective dimension closer to 2 are better. "
                f"(best run here: trust = {best_chart.get('trustworthiness', float('nan')):.4f}, "
                f"overlap = {best_chart.get('knn_jaccard_mean', float('nan')):.4f}, "
                f"eff_dim = {best_chart.get('effective_dimension', float('nan')):.4f})",
            ])

    overall_best, overall_summary = _select_best_run(
        list(step2_summaries.keys()), step2_summaries,
    )
    lines.extend([
        '',
        '## Adopted Run',
        '',
    ])
    if overall_best is None:
        lines.append('- Pending Step 2 execution.')
    else:
        overall_chart = overall_summary.get('chart_quality', {})
        lines.extend([
            f"- Selected run: `{overall_best}`",
            f"- Modular mean distance: "
            f"{overall_summary['modular_invariance']['mean_latent_distance']:.4f}",
            f"- max Spearman vs log10|j|: "
            f"{overall_summary['j_correlation'].get('max_abs_logabsj_spearman', float('nan')):.4f}",
            f"- Quotient-chart trust / overlap / eff_dim: "
            f"{overall_chart.get('trustworthiness', float('nan')):.4f} / "
            f"{overall_chart.get('knn_jaccard_mean', float('nan')):.4f} / "
            f"{overall_chart.get('effective_dimension', float('nan')):.4f}",
            f"- Reconstruction MSE: {overall_summary['reconstruction']['mse']:.6g}",
        ])

    lines.extend([
        '',
        '## Open Issues',
        '',
        '- Differentiable projection to the fundamental domain is still out of scope for Step 2.',
        '- Half-plane latent is only retained as a Phase 1 comparison point.',
        '- Normalized runs should not be compared directly to raw Step 1 MSE values.',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    experiments: list[tuple[str, str | Callable[[], object]]] | None = None,
    summary_filename: str = 'lattice_step2_summaries.json',
    report_path: str = 'walkthrough-lattice-step2.md',
) -> dict:
    """Run the Step 2 lattice experiment matrix."""
    experiments = experiments or EXPERIMENTS
    all_summaries = {}

    for name, config_source in experiments:
        print(f"\n{'=' * 60}")
        print(f"  Step 2 Experiment: {name}")
        print(f"{'=' * 60}\n")

        config = _load_config(config_source)
        workdir = os.path.join(base_dir, name)

        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, workdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, workdir)
        all_summaries[name] = summary

    summary_path = os.path.join(base_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    write_step2_report(all_summaries, output_path=report_path)

    print(f"\n{'=' * 60}")
    print('  All Step 2 lattice experiments complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f"{'=' * 60}")

    print(
        f"\n{'Experiment':<32} {'MSE':>10} {'log|j|ρ':>10} "
        f"{'ModDist':>10} {'Trust':>8} {'Overlap':>8} {'EffDim':>8}"
    )
    print('-' * 96)
    for name, summary in all_summaries.items():
        recon = summary.get('reconstruction', {})
        j_corr = summary.get('j_correlation', {})
        mod = summary.get('modular_invariance', {})
        chart = summary.get('chart_quality', {})
        print(
            f"{name:<32} "
            f"{recon.get('mse', float('nan')):>10.6f} "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):>10.4f} "
            f"{mod.get('mean_latent_distance', float('nan')):>10.4f} "
            f"{chart.get('trustworthiness', float('nan')):>8.4f} "
            f"{chart.get('knn_jaccard_mean', float('nan')):>8.4f} "
            f"{chart.get('effective_dimension', float('nan')):>8.4f}"
        )

    return all_summaries


if __name__ == '__main__':
    run_all()
