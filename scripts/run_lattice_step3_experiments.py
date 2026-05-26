"""Run Step 3 lattice VAE+invariance experiments and generate a report."""

import json
import os
from collections.abc import Callable

from configs.lattice_default import get_config as get_lattice_config
from eval.analysis import run_full_evaluation
from train.trainer import train_and_evaluate


ANCHOR_RUNS = [
    'lattice_standard_norm_inv',
    'lattice_standard_wide_norm_inv',
    'lattice_vae_norm_beta001',
]

GROUPS = {
    'Fundamental-Domain VAE + Invariance': 'fundamental_domain',
    'Wide Half-Plane VAE + Invariance': 'halfplane',
}

VAE_BETAS = [0.003, 0.01, 0.03]
INVARIANCE_WEIGHTS = [0.03, 0.1]

SUCCESS_GATE = {
    'mean_latent_distance_max': 0.01,
    'effective_dimension_min': 1.4,
    'knn_jaccard_mean_min': 0.05,
    'trustworthiness_min': 0.85,
    'max_abs_logabsj_spearman_min': 0.80,
    'mse_max': 5e-06,
}


def _format_milli_tag(value: float) -> str:
    """Format a small float as a 3-digit milli-scale tag."""
    return f"{int(round(value * 1000.0)):03d}"


def _make_step3_config(
    sample_region: str,
    vae_beta: float,
    modular_invariance_weight: float,
):
    """Create a Step 3 lattice VAE config."""
    config = get_lattice_config()
    config.data.lattice_signal_normalization = 'max'
    config.data.lattice_sample_region = sample_region
    if sample_region == 'halfplane':
        config.data.lattice_x_range = (-2.0, 2.0)
        config.data.lattice_y_range = (0.3, 4.0)
    config.model.latent_type = 'vae'
    config.model.latent_dim = 4
    config.model.vae_beta = vae_beta
    config.train.modular_invariance_weight = modular_invariance_weight
    return config


def _build_experiments() -> list[tuple[str, Callable[[], object]]]:
    """Build the default Step 3 experiment matrix."""
    experiments = []
    for sample_region in ('fundamental_domain', 'halfplane'):
        for beta in VAE_BETAS:
            for weight in INVARIANCE_WEIGHTS:
                beta_tag = _format_milli_tag(beta)
                weight_tag = _format_milli_tag(weight)
                prefix = 'lattice_vae_norm_inv' if sample_region == 'fundamental_domain' else 'lattice_vae_wide_norm_inv'
                name = f'{prefix}_b{beta_tag}_l{weight_tag}'

                def factory(
                    sample_region=sample_region,
                    beta=beta,
                    weight=weight,
                ):
                    return _make_step3_config(sample_region, beta, weight)

                experiments.append((name, factory))

    return experiments


EXPERIMENTS = _build_experiments()


def _selection_key(summary: dict) -> tuple[float, float, float, float, float, float]:
    """Lexicographic run ranking for Step 3."""
    modular = summary.get('modular_invariance', {}).get(
        'mean_latent_distance', float('inf'),
    )
    chart = summary.get('chart_quality', {})
    eff_dim = chart.get('effective_dimension', float('nan'))
    eff_penalty = abs(eff_dim - 2.0) if eff_dim == eff_dim else float('inf')
    overlap = chart.get('knn_jaccard_mean', float('-inf'))
    trust = chart.get('trustworthiness', float('-inf'))
    spearman = summary.get('j_correlation', {}).get(
        'max_abs_logabsj_spearman', float('-inf'),
    )
    mse = summary.get('reconstruction', {}).get('mse', float('inf'))
    return (modular, eff_penalty, -overlap, -trust, -spearman, mse)


def _select_best_run(run_names: list[str], summaries: dict[str, dict]) -> tuple[str, dict] | tuple[None, None]:
    candidates = [(name, summaries[name]) for name in run_names if name in summaries]
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: _selection_key(item[1]))


def _passes_success_gate(summary: dict) -> bool:
    """Whether a Step 3 run satisfies the target success criteria."""
    chart = summary.get('chart_quality', {})
    mod = summary.get('modular_invariance', {})
    j_corr = summary.get('j_correlation', {})
    recon = summary.get('reconstruction', {})
    return (
        mod.get('mean_latent_distance', float('inf')) <= SUCCESS_GATE['mean_latent_distance_max']
        and chart.get('effective_dimension', float('-inf')) >= SUCCESS_GATE['effective_dimension_min']
        and chart.get('knn_jaccard_mean', float('-inf')) >= SUCCESS_GATE['knn_jaccard_mean_min']
        and chart.get('trustworthiness', float('-inf')) >= SUCCESS_GATE['trustworthiness_min']
        and j_corr.get('max_abs_logabsj_spearman', float('-inf')) >= SUCCESS_GATE['max_abs_logabsj_spearman_min']
        and recon.get('mse', float('inf')) <= SUCCESS_GATE['mse_max']
    )


def _load_anchor_summaries(
    anchor_summary_path: str,
) -> dict[str, dict]:
    if not os.path.exists(anchor_summary_path):
        return {}
    with open(anchor_summary_path) as f:
        return json.load(f)


def _run_group_names(sample_region: str, summaries: dict[str, dict]) -> list[str]:
    prefix = 'lattice_vae_norm_inv_' if sample_region == 'fundamental_domain' else 'lattice_vae_wide_norm_inv_'
    return [name for name in summaries if name.startswith(prefix)]


def write_step3_report(
    step3_summaries: dict[str, dict],
    output_path: str = 'walkthrough-lattice-step3.md',
    anchor_summary_path: str = 'runs/lattice_step2_summaries.json',
) -> None:
    """Write the Step 3 markdown report."""
    anchors = _load_anchor_summaries(anchor_summary_path)

    lines = [
        '# Lattice Step 3 — VAE + Invariance Sweep',
        '',
        '## Summary',
        '',
        '- Step 3 tests whether VAE KL pressure and modular invariance can jointly improve orbit gluing and 2D chart quality.',
        '- Ranking rule: modular distance, then distance of effective dimension from 2, then chart overlap, trustworthiness, `log10|j|` Spearman, and reconstruction MSE.',
        '- Success gate requires simultaneously strong orbit gluing, strong chart quality, strong `j` retention, and low reconstruction error.',
        '',
        '## Step 2 Anchors',
        '',
        '| Run | MSE | log10_abs_j Spearman | ModDist | trust | overlap | eff_dim |',
        '|---|---|---|---|---|---|---|',
    ]

    for name in ANCHOR_RUNS:
        summary = anchors.get(name)
        if summary is None:
            lines.append(f'| `{name}` | missing | missing | missing | missing | missing | missing |')
            continue
        recon = summary.get('reconstruction', {})
        chart = summary.get('chart_quality', {})
        j_corr = summary.get('j_correlation', {})
        mod = summary.get('modular_invariance', {})
        lines.append(
            f"| `{name}` | {recon.get('mse', float('nan')):.6g} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{mod.get('mean_latent_distance', float('nan')):.4f} | "
            f"{chart.get('trustworthiness', float('nan')):.4f} | "
            f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{chart.get('effective_dimension', float('nan')):.4f} |"
        )

    for title, sample_region in GROUPS.items():
        run_names = _run_group_names(sample_region, step3_summaries)
        lines.extend([
            '',
            f'## {title}',
            '',
            '| Run | MSE | log10_abs_j Spearman | ModDist | trust | overlap | eff_dim | gate |',
            '|---|---|---|---|---|---|---|---|',
        ])
        for name in sorted(run_names):
            summary = step3_summaries[name]
            recon = summary.get('reconstruction', {})
            chart = summary.get('chart_quality', {})
            j_corr = summary.get('j_correlation', {})
            mod = summary.get('modular_invariance', {})
            gate = 'PASS' if _passes_success_gate(summary) else 'FAIL'
            lines.append(
                f"| `{name}` | {recon.get('mse', float('nan')):.6g} | "
                f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
                f"{mod.get('mean_latent_distance', float('nan')):.4f} | "
                f"{chart.get('trustworthiness', float('nan')):.4f} | "
                f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
                f"{chart.get('effective_dimension', float('nan')):.4f} | "
                f"{gate} |"
            )

        best_name, best_summary = _select_best_run(run_names, step3_summaries)
        if best_name is None:
            lines.extend(['', '- Best run: pending'])
        else:
            best_chart = best_summary.get('chart_quality', {})
            best_j = best_summary.get('j_correlation', {})
            best_mod = best_summary.get('modular_invariance', {})
            best_recon = best_summary.get('reconstruction', {})
            lines.extend([
                '',
                f"- Best run: `{best_name}`",
                f"- Orbit gluing: mean modular distance = {best_mod.get('mean_latent_distance', float('nan')):.4f}",
                f"- 2D chart: eff_dim = {best_chart.get('effective_dimension', float('nan')):.4f}, "
                f"overlap = {best_chart.get('knn_jaccard_mean', float('nan')):.4f}, "
                f"trust = {best_chart.get('trustworthiness', float('nan')):.4f}",
                f"- `j` retention: Spearman = {best_j.get('max_abs_logabsj_spearman', float('nan')):.4f}",
                f"- Reconstruction: MSE = {best_recon.get('mse', float('nan')):.6g}",
                f"- Success gate: {'PASS' if _passes_success_gate(best_summary) else 'FAIL'}",
            ])

    overall_best, overall_summary = _select_best_run(
        list(step3_summaries.keys()), step3_summaries,
    )
    lines.extend([
        '',
        '## Selected Run',
        '',
    ])
    if overall_best is None:
        lines.append('- Pending Step 3 execution.')
    else:
        chart = overall_summary.get('chart_quality', {})
        j_corr = overall_summary.get('j_correlation', {})
        mod = overall_summary.get('modular_invariance', {})
        recon = overall_summary.get('reconstruction', {})
        lines.extend([
            f"- Selected run: `{overall_best}`",
            f"- Orbit gluing: mean modular distance = {mod.get('mean_latent_distance', float('nan')):.4f}",
            f"- 2D chart: eff_dim = {chart.get('effective_dimension', float('nan')):.4f}, "
            f"overlap = {chart.get('knn_jaccard_mean', float('nan')):.4f}, "
            f"trust = {chart.get('trustworthiness', float('nan')):.4f}",
            f"- `j` retention: Spearman = {j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f}",
            f"- Reconstruction: MSE = {recon.get('mse', float('nan')):.6g}",
            f"- Success gate: {'PASS' if _passes_success_gate(overall_summary) else 'FAIL'}",
        ])

    any_pass = any(_passes_success_gate(summary) for summary in step3_summaries.values())
    lines.extend([
        '',
        '## Conclusion',
        '',
    ])
    if any_pass:
        lines.append(
            '- At least one Step 3 run passed the success gate, so VAE + invariance is a viable path for jointly improving orbit gluing and 2D chart quality.'
        )
    else:
        lines.append(
            '- No Step 3 run passed the success gate. Treat this as evidence that a Gaussian VAE prior alone is insufficient, and move next to equivariant latent design.'
        )

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    experiments: list[tuple[str, Callable[[], object]]] | None = None,
    summary_filename: str = 'lattice_step3_summaries.json',
    report_path: str = 'walkthrough-lattice-step3.md',
    anchor_summary_path: str = 'runs/lattice_step2_summaries.json',
) -> dict[str, dict]:
    """Run the Step 3 VAE+invariance sweep."""
    experiments = experiments or EXPERIMENTS
    all_summaries = {}

    for name, config_factory in experiments:
        print(f"\n{'=' * 60}")
        print(f"  Step 3 Experiment: {name}")
        print(f"{'=' * 60}\n")

        config = config_factory()
        workdir = os.path.join(base_dir, name)
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, workdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, workdir)
        all_summaries[name] = summary

    summary_path = os.path.join(base_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    write_step3_report(
        all_summaries,
        output_path=report_path,
        anchor_summary_path=anchor_summary_path,
    )

    print(f"\n{'=' * 60}")
    print('  All Step 3 lattice experiments complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f"{'=' * 60}")

    print(
        f"\n{'Experiment':<36} {'MSE':>10} {'ModDist':>10} "
        f"{'EffΔ':>8} {'Overlap':>8} {'Trust':>8} {'log|j|ρ':>10} {'Gate':>6}"
    )
    print('-' * 112)
    for name, summary in all_summaries.items():
        chart = summary.get('chart_quality', {})
        mod = summary.get('modular_invariance', {})
        recon = summary.get('reconstruction', {})
        j_corr = summary.get('j_correlation', {})
        eff_dim = chart.get('effective_dimension', float('nan'))
        eff_penalty = abs(eff_dim - 2.0) if eff_dim == eff_dim else float('nan')
        print(
            f"{name:<36} "
            f"{recon.get('mse', float('nan')):>10.6f} "
            f"{mod.get('mean_latent_distance', float('nan')):>10.4f} "
            f"{eff_penalty:>8.4f} "
            f"{chart.get('knn_jaccard_mean', float('nan')):>8.4f} "
            f"{chart.get('trustworthiness', float('nan')):>8.4f} "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):>10.4f} "
            f"{('PASS' if _passes_success_gate(summary) else 'FAIL'):>6}"
        )

    return all_summaries


if __name__ == '__main__':
    run_all()
