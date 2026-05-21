"""Run Step 5 factorized lattice experiments with quotient chart regularizers."""

import json
import os
from collections.abc import Callable

from configs.lattice_default import get_config as get_lattice_config
from eval.analysis import run_full_evaluation
from train.trainer import train_and_evaluate


STEP4_ANCHOR = 'lattice_factorized_vae_fd_b030_q100_g030_d030'


def _make_step5_config(
    chart_weight: float,
    variance_weight: float,
):
    """Create a Step 5 config anchored on the selected Step 4 scaffold."""
    config = get_lattice_config()
    config.data.lattice_signal_normalization = 'max'
    config.data.lattice_sample_region = 'fundamental_domain'
    config.model.latent_type = 'factorized_vae'
    config.model.latent_dim = 6
    config.model.quotient_dim = 2
    config.model.gauge_dim = 4
    config.model.gauge_action_type = 'affine'
    config.model.vae_beta = 0.03
    config.train.modular_invariance_weight = 0.1
    config.train.gauge_equivariance_weight = 0.03
    config.train.decoder_equivariance_weight = 0.03
    config.train.gauge_action_reg_weight = 1e-4
    config.train.chart_preserving_weight = chart_weight
    config.train.chart_preserving_n_neighbors = 8
    config.train.quotient_variance_floor_weight = variance_weight
    config.train.quotient_variance_floor_target = 0.15
    return config


EXPERIMENTS = [
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_c010_v000',
        lambda: _make_step5_config(0.01, 0.00),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_c030_v000',
        lambda: _make_step5_config(0.03, 0.00),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_c030_v010',
        lambda: _make_step5_config(0.03, 0.01),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_c050_v010',
        lambda: _make_step5_config(0.05, 0.01),
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
        eff_penalty,
        -chart.get('knn_jaccard_mean', float('-inf')),
        factorized.get('quotient_partner_rank_percentile_mean', float('inf')),
        -chart.get('trustworthiness', float('-inf')),
        -j_corr.get('max_abs_logabsj_spearman', float('-inf')),
        factorized.get('gauge_equivariance_mse', float('inf')),
        factorized.get('decoder_equivariance_mse', float('inf')),
        recon.get('mse', float('inf')),
    )


def _select_best_run(summaries: dict[str, dict]) -> tuple[str, dict] | tuple[None, None]:
    candidates = list(summaries.items())
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: _selection_key(item[1]))


def write_step5_report(
    step5_summaries: dict[str, dict],
    output_path: str = 'walkthrough-lattice-step5.md',
    anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
) -> None:
    """Write the Step 5 markdown report."""
    anchors = _load_anchor_summaries(anchor_summary_path)
    anchor_summary = anchors.get(STEP4_ANCHOR)

    lines = [
        '# Lattice Step 5 — Quotient Chart-Preserving Regularizer',
        '',
        '## Summary',
        '',
        '- Step 5 keeps the Step 4 factorized scaffold fixed and adds quotient-only chart regularization.',
        '- Selection rule: quotient effective-dimension penalty, quotient overlap, quotient partner rank, quotient trust, `log10|j|` Spearman, gauge equivariance, decoder equivariance, then reconstruction MSE.',
        '',
        '## Step 4 Anchor',
        '',
        '| Run | q rank | q trust | q overlap | q eff_dim | q var0 | q var1 | log10_abs_j Spearman | MSE |',
        '|---|---|---|---|---|---|---|---|---|',
    ]

    if anchor_summary is None:
        lines.append(f'| `{STEP4_ANCHOR}` | missing | missing | missing | missing | missing | missing | missing | missing |')
    else:
        chart = anchor_summary.get('chart_quality', {})
        factorized = anchor_summary.get('factorized_consistency', {})
        j_corr = anchor_summary.get('j_correlation', {})
        recon = anchor_summary.get('reconstruction', {})
        lines.append(
            f"| `{STEP4_ANCHOR}` | "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f} | "
            f"{chart.get('trustworthiness', float('nan')):.4f} | "
            f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{chart.get('effective_dimension', float('nan')):.4f} | "
            f"{factorized.get('quotient_var_dim0', float('nan')):.4f} | "
            f"{factorized.get('quotient_var_dim1', float('nan')):.4f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    lines.extend([
        '',
        '## Step 5 Runs',
        '',
        '| Run | q rank | q trust | q overlap | q eff_dim | q var0 | q var1 | q chart loss | log10_abs_j Spearman | gauge mse | decode mse | MSE |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ])

    for name, summary in step5_summaries.items():
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        recon = summary.get('reconstruction', {})
        lines.append(
            f"| `{name}` | "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f} | "
            f"{chart.get('trustworthiness', float('nan')):.4f} | "
            f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{chart.get('effective_dimension', float('nan')):.4f} | "
            f"{factorized.get('quotient_var_dim0', float('nan')):.4f} | "
            f"{factorized.get('quotient_var_dim1', float('nan')):.4f} | "
            f"{factorized.get('quotient_chart_loss', float('nan')):.6f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{factorized.get('gauge_equivariance_mse', float('nan')):.6f} | "
            f"{factorized.get('decoder_equivariance_mse', float('nan')):.6f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    best_name, best_summary = _select_best_run(step5_summaries)
    lines.extend([
        '',
        '## Selected Run',
        '',
    ])
    if best_name is None:
        lines.append('- Pending Step 5 execution.')
    else:
        chart = best_summary.get('chart_quality', {})
        factorized = best_summary.get('factorized_consistency', {})
        j_corr = best_summary.get('j_correlation', {})
        recon = best_summary.get('reconstruction', {})
        lines.extend([
            f"- Selected run: `{best_name}`",
            f"- Quotient chart: eff_dim = {chart.get('effective_dimension', float('nan')):.4f}, overlap = {chart.get('knn_jaccard_mean', float('nan')):.4f}, trust = {chart.get('trustworthiness', float('nan')):.4f}",
            f"- Quotient spread: var0 = {factorized.get('quotient_var_dim0', float('nan')):.4f}, var1 = {factorized.get('quotient_var_dim1', float('nan')):.4f}, chart loss = {factorized.get('quotient_chart_loss', float('nan')):.6f}, variance-floor loss = {factorized.get('quotient_variance_floor_loss', float('nan')):.6f}",
            f"- Partner / `j` retention: rank = {factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f}, Spearman = {j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f}",
            f"- Gauge / decoder consistency: gauge mse = {factorized.get('gauge_equivariance_mse', float('nan')):.6f}, decoder mse = {factorized.get('decoder_equivariance_mse', float('nan')):.6f}",
            f"- Reconstruction: MSE = {recon.get('mse', float('nan')):.6g}",
        ])

    lines.extend([
        '',
        '## Interpretation',
        '',
        '- Read Step 5 as a focused test of whether quotient-only geometry regularization can widen the chart without breaking partner alignment or gauge consistency.',
        '- Topology Phase A/B should be run only for the Step 5 selected run against the fixed Step 3/4 anchors.',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    experiments: list[tuple[str, Callable[[], object]]] | None = None,
    summary_filename: str = 'lattice_step5_summaries.json',
    report_path: str = 'walkthrough-lattice-step5.md',
    anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
) -> dict[str, dict]:
    """Run the Step 5 lattice sweep."""
    experiments = experiments or EXPERIMENTS
    all_summaries = {}

    for name, config_factory in experiments:
        print(f"\n{'=' * 60}")
        print(f"  Step 5 Experiment: {name}")
        print(f"{'=' * 60}\n")
        config = config_factory()
        workdir = os.path.join(base_dir, name)
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, workdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, workdir)
        all_summaries[name] = summary

    summary_path = os.path.join(base_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    write_step5_report(
        all_summaries,
        output_path=report_path,
        anchor_summary_path=anchor_summary_path,
    )

    print(f"\n{'=' * 60}")
    print('  All Step 5 lattice experiments complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f"{'=' * 60}")

    print(
        f"\n{'Experiment':<56} {'q-rank':>8} {'effΔ':>8} {'overlap':>8} "
        f"{'trust':>8} {'var0':>8} {'var1':>8} {'chart':>10} "
        f"{'log|j|ρ':>10} {'MSE':>10}"
    )
    print('-' * 146)
    for name, summary in all_summaries.items():
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        recon = summary.get('reconstruction', {})
        eff_dim = chart.get('effective_dimension', float('nan'))
        eff_penalty = abs(eff_dim - 2.0) if eff_dim == eff_dim else float('nan')
        print(
            f"{name:<56} "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):>8.4f} "
            f"{eff_penalty:>8.4f} "
            f"{chart.get('knn_jaccard_mean', float('nan')):>8.4f} "
            f"{chart.get('trustworthiness', float('nan')):>8.4f} "
            f"{factorized.get('quotient_var_dim0', float('nan')):>8.4f} "
            f"{factorized.get('quotient_var_dim1', float('nan')):>8.4f} "
            f"{factorized.get('quotient_chart_loss', float('nan')):>10.6f} "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):>10.4f} "
            f"{recon.get('mse', float('nan')):>10.6f}"
        )

    return all_summaries


if __name__ == '__main__':
    run_all()
