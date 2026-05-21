"""Run Step 4 factorized lattice VAE experiments and generate a report."""

import json
import os
from collections.abc import Callable

from configs.lattice_default import get_config as get_lattice_config
from eval.analysis import run_full_evaluation
from train.trainer import train_and_evaluate


ANCHOR_RUNS = [
    'lattice_standard_norm_inv',
    'lattice_vae_norm_inv_b010_l100',
    'lattice_vae_norm_inv_b030_l100',
    'lattice_vae_wide_norm_inv_b003_l030',
]


def _make_step4_config(
    sample_region: str,
    vae_beta: float,
):
    """Create a Step 4 factorized lattice config."""
    config = get_lattice_config()
    config.data.lattice_signal_normalization = 'max'
    config.data.lattice_sample_region = sample_region
    if sample_region == 'halfplane':
        config.data.lattice_x_range = (-2.0, 2.0)
        config.data.lattice_y_range = (0.3, 4.0)
    config.model.latent_type = 'factorized_vae'
    config.model.latent_dim = 6
    config.model.quotient_dim = 2
    config.model.gauge_dim = 4
    config.model.gauge_action_type = 'affine'
    config.model.vae_beta = vae_beta
    config.train.modular_invariance_weight = 0.1
    config.train.gauge_equivariance_weight = 0.03
    config.train.decoder_equivariance_weight = 0.03
    config.train.gauge_action_reg_weight = 1e-4
    return config


EXPERIMENTS = [
    (
        'lattice_factorized_vae_fd_b010_q100_g030_d030',
        lambda: _make_step4_config('fundamental_domain', 0.01),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030',
        lambda: _make_step4_config('fundamental_domain', 0.03),
    ),
    (
        'lattice_factorized_vae_wide_b030_q100_g030_d030',
        lambda: _make_step4_config('halfplane', 0.03),
    ),
]


def _load_anchor_summaries(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _selection_key(summary: dict) -> tuple[float, float, float, float, float, float, float, float]:
    chart = summary.get('chart_quality', {})
    factorized = summary.get('factorized_consistency', {})
    j_corr = summary.get('j_correlation', {})
    recon = summary.get('reconstruction', {})

    eff_dim = chart.get('effective_dimension', float('nan'))
    eff_penalty = abs(eff_dim - 2.0) if eff_dim == eff_dim else float('inf')
    return (
        factorized.get('quotient_partner_rank_percentile_mean', float('inf')),
        eff_penalty,
        -chart.get('knn_jaccard_mean', float('-inf')),
        -chart.get('trustworthiness', float('-inf')),
        factorized.get('gauge_equivariance_mse', float('inf')),
        factorized.get('decoder_equivariance_mse', float('inf')),
        -j_corr.get('max_abs_logabsj_spearman', float('-inf')),
        recon.get('mse', float('inf')),
    )


def _select_best_run(summaries: dict[str, dict]) -> tuple[str, dict] | tuple[None, None]:
    candidates = list(summaries.items())
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: _selection_key(item[1]))


def write_step4_report(
    step4_summaries: dict[str, dict],
    output_path: str = 'walkthrough-lattice-step4.md',
    anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
) -> None:
    """Write the Step 4 markdown report."""
    anchors = _load_anchor_summaries(anchor_summary_path)
    lines = [
        '# Lattice Step 4 — Factorized VAE',
        '',
        '## Summary',
        '',
        '- Step 4 makes the quotient chart explicit by splitting the latent into `quotient(2) + gauge(4)` parts.',
        '- Ranking rule: quotient partner rank, quotient effective-dimension penalty, quotient overlap, quotient trust, gauge equivariance, decoder equivariance, `log10|j|` Spearman, then reconstruction MSE.',
        '',
        '## Step 3 Anchors',
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

    lines.extend([
        '',
        '## Step 4 Runs',
        '',
        '| Run | q rank | q hit | q trust | q overlap | q eff_dim | gauge mse | decode mse | log10_abs_j Spearman | MSE |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ])

    for name in step4_summaries:
        summary = step4_summaries[name]
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        recon = summary.get('reconstruction', {})
        lines.append(
            f"| `{name}` | "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f} | "
            f"{factorized.get('quotient_partner_knn_hit_rate', float('nan')):.4f} | "
            f"{chart.get('trustworthiness', float('nan')):.4f} | "
            f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{chart.get('effective_dimension', float('nan')):.4f} | "
            f"{factorized.get('gauge_equivariance_mse', float('nan')):.6f} | "
            f"{factorized.get('decoder_equivariance_mse', float('nan')):.6f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    best_name, best_summary = _select_best_run(step4_summaries)
    lines.extend([
        '',
        '## Selected Run',
        '',
    ])
    if best_name is None:
        lines.append('- Pending Step 4 execution.')
    else:
        chart = best_summary.get('chart_quality', {})
        factorized = best_summary.get('factorized_consistency', {})
        j_corr = best_summary.get('j_correlation', {})
        recon = best_summary.get('reconstruction', {})
        lines.extend([
            f"- Selected run: `{best_name}`",
            f"- Quotient chart: eff_dim = {chart.get('effective_dimension', float('nan')):.4f}, overlap = {chart.get('knn_jaccard_mean', float('nan')):.4f}, trust = {chart.get('trustworthiness', float('nan')):.4f}",
            f"- Quotient partner preservation: rank = {factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f}, hit = {factorized.get('quotient_partner_knn_hit_rate', float('nan')):.4f}",
            f"- Gauge / decoder consistency: gauge mse = {factorized.get('gauge_equivariance_mse', float('nan')):.6f}, decoder mse = {factorized.get('decoder_equivariance_mse', float('nan')):.6f}",
            f"- `j` retention: Spearman = {j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f}",
            f"- Reconstruction: MSE = {recon.get('mse', float('nan')):.6g}",
        ])

    lines.extend([
        '',
        '## Interpretation',
        '',
        '- Read Step 4 as a direct test of whether an explicit quotient/gauge split is easier to compare than the Step 3 monolithic VAE latent.',
        '- The next topology pass should compare the selected factorized run against the Step 3 anchors using the quotient view only.',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    experiments: list[tuple[str, Callable[[], object]]] | None = None,
    summary_filename: str = 'lattice_step4_summaries.json',
    report_path: str = 'walkthrough-lattice-step4.md',
    anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
) -> dict[str, dict]:
    """Run the Step 4 factorized lattice sweep."""
    experiments = experiments or EXPERIMENTS
    all_summaries = {}

    for name, config_factory in experiments:
        print(f"\n{'=' * 60}")
        print(f"  Step 4 Experiment: {name}")
        print(f"{'=' * 60}\n")
        config = config_factory()
        workdir = os.path.join(base_dir, name)
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, workdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, workdir)
        all_summaries[name] = summary

    summary_path = os.path.join(base_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    write_step4_report(
        all_summaries,
        output_path=report_path,
        anchor_summary_path=anchor_summary_path,
    )

    print(f"\n{'=' * 60}")
    print('  All Step 4 lattice experiments complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f"{'=' * 60}")

    print(
        f"\n{'Experiment':<46} {'q-rank':>8} {'effΔ':>8} {'overlap':>8} "
        f"{'trust':>8} {'gauge':>10} {'decode':>10} {'log|j|ρ':>10} {'MSE':>10}"
    )
    print('-' * 126)
    for name, summary in all_summaries.items():
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        recon = summary.get('reconstruction', {})
        eff_dim = chart.get('effective_dimension', float('nan'))
        eff_penalty = abs(eff_dim - 2.0) if eff_dim == eff_dim else float('nan')
        print(
            f"{name:<46} "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):>8.4f} "
            f"{eff_penalty:>8.4f} "
            f"{chart.get('knn_jaccard_mean', float('nan')):>8.4f} "
            f"{chart.get('trustworthiness', float('nan')):>8.4f} "
            f"{factorized.get('gauge_equivariance_mse', float('nan')):>10.6f} "
            f"{factorized.get('decoder_equivariance_mse', float('nan')):>10.6f} "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):>10.4f} "
            f"{recon.get('mse', float('nan')):>10.6f}"
        )

    return all_summaries


if __name__ == '__main__':
    run_all()
