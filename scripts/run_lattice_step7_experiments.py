"""Run Step 7 factorized lattice experiments with Jacobian-like quotient regularization."""

import json
import os
from collections.abc import Callable

from configs.lattice_default import get_config as get_lattice_config
from eval.analysis import run_full_evaluation
from train.trainer import train_and_evaluate


STEP4_ANCHOR = 'lattice_factorized_vae_fd_b030_q100_g030_d030'
STEP6_FALLBACK = 'lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030'
STEP3_ANCHORS = [
    'lattice_standard_norm_inv',
    'lattice_vae_norm_inv_b010_l100',
    'lattice_vae_norm_inv_b030_l100',
]


def _make_step7_config(
    jacobian_weight: float,
    logdet_weight: float,
):
    """Create a Step 7 config anchored on the selected Step 4 scaffold."""
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
    config.train.chart_preserving_weight = 0.0
    config.train.chart_preserving_n_neighbors = 8
    config.train.quotient_variance_floor_weight = 0.0
    config.train.quotient_spread_weight = 0.0
    config.train.quotient_min_eig_ratio_target = 0.20
    config.train.quotient_trace_cap_ratio = 1.50
    config.train.jacobian_gram_weight = jacobian_weight
    config.train.jacobian_n_neighbors = 8
    config.train.quotient_logdet_weight = logdet_weight
    config.train.quotient_logdet_ratio_target = 0.10
    return config


EXPERIMENTS = [
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010',
        lambda: _make_step7_config(0.01, 0.01),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld030',
        lambda: _make_step7_config(0.01, 0.03),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j030_ld010',
        lambda: _make_step7_config(0.03, 0.01),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j030_ld030',
        lambda: _make_step7_config(0.03, 0.03),
    ),
]


def _load_anchor_summaries(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _passes_gate(summary: dict) -> bool:
    factorized = summary.get('factorized_consistency', {})
    j_corr = summary.get('j_correlation', {})
    return (
        factorized.get('quotient_partner_rank_percentile_mean', float('inf')) <= 0.35
        and j_corr.get('max_abs_logabsj_spearman', float('-inf')) >= 0.88
    )


def _selection_key(summary: dict) -> tuple[float, float, float, float, float, float]:
    chart = summary.get('chart_quality', {})
    factorized = summary.get('factorized_consistency', {})
    recon = summary.get('reconstruction', {})

    eff_dim = chart.get('effective_dimension', float('nan'))
    eff_penalty = abs(eff_dim - 2.0) if eff_dim == eff_dim else float('inf')
    return (
        factorized.get('quotient_partner_rank_percentile_mean', float('inf')),
        eff_penalty,
        -chart.get('knn_jaccard_mean', float('-inf')),
        -chart.get('trustworthiness', float('-inf')),
        factorized.get('decoder_equivariance_mse', float('inf')),
        recon.get('mse', float('inf')),
    )


def _select_best_run(
    summaries: dict[str, dict],
) -> tuple[str, dict, bool] | tuple[None, None, bool]:
    candidates = list(summaries.items())
    if not candidates:
        return None, None, False

    gated = [(name, summary) for name, summary in candidates if _passes_gate(summary)]
    pool = gated if gated else candidates
    best_name, best_summary = min(pool, key=lambda item: _selection_key(item[1]))
    return best_name, best_summary, bool(gated)


def write_step7_report(
    step7_summaries: dict[str, dict],
    output_path: str = 'walkthrough-lattice-step7.md',
    step3_anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
    step4_anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
    step6_anchor_summary_path: str = 'runs/lattice_step6_summaries.json',
) -> None:
    """Write the Step 7 markdown report."""
    step3_anchors = _load_anchor_summaries(step3_anchor_summary_path)
    step4_anchors = _load_anchor_summaries(step4_anchor_summary_path)
    step6_anchors = _load_anchor_summaries(step6_anchor_summary_path)

    lines = [
        '# Lattice Step 7 — Jacobian-like Quotient Metric Regularizer',
        '',
        '## Summary',
        '',
        '- Step 7 keeps the Step 4 factorized scaffold fixed and replaces the Step 5/6 local term with a Jacobian-like local Gram match.',
        '- A weak global logdet floor + trace cap is added only as a collapse guard.',
        '- Selection first applies the Step 6 gate: quotient partner rank <= 0.35 and `log10|j|` Spearman >= 0.88.',
        '',
        '## Anchors',
        '',
        '| Run | Source | q rank | q trust | q overlap | q eff_dim | log10_abs_j Spearman | MSE |',
        '|---|---|---|---|---|---|---|---|',
    ]

    for name in STEP3_ANCHORS:
        summary = step3_anchors.get(name)
        if summary is None:
            lines.append(f'| `{name}` | Step 3 | missing | missing | missing | missing | missing | missing |')
            continue
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        recon = summary.get('reconstruction', {})
        lines.append(
            f"| `{name}` | Step 3 | "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f} | "
            f"{chart.get('trustworthiness', float('nan')):.4f} | "
            f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{chart.get('effective_dimension', float('nan')):.4f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    for source, run_name, anchor_map in (
        ('Step 4', STEP4_ANCHOR, step4_anchors),
        ('Step 6', STEP6_FALLBACK, step6_anchors),
    ):
        summary = anchor_map.get(run_name)
        if summary is None:
            lines.append(f'| `{run_name}` | {source} | missing | missing | missing | missing | missing | missing |')
            continue
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        recon = summary.get('reconstruction', {})
        lines.append(
            f"| `{run_name}` | {source} | "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f} | "
            f"{chart.get('trustworthiness', float('nan')):.4f} | "
            f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{chart.get('effective_dimension', float('nan')):.4f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    lines.extend([
        '',
        '## Step 7 Runs',
        '',
        '| Run | Gate | q rank | q trust | q overlap | q eff_dim | jacobian loss | logdet loss | q eig_min | q eig_max | log10_abs_j Spearman | decode mse | MSE |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ])

    for name, summary in step7_summaries.items():
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        recon = summary.get('reconstruction', {})
        lines.append(
            f"| `{name}` | "
            f"{'PASS' if _passes_gate(summary) else 'FAIL'} | "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f} | "
            f"{chart.get('trustworthiness', float('nan')):.4f} | "
            f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{chart.get('effective_dimension', float('nan')):.4f} | "
            f"{factorized.get('quotient_jacobian_gram_loss', float('nan')):.6f} | "
            f"{factorized.get('quotient_logdet_loss', float('nan')):.6f} | "
            f"{factorized.get('quotient_cov_eig_min', float('nan')):.6f} | "
            f"{factorized.get('quotient_cov_eig_max', float('nan')):.6f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{factorized.get('decoder_equivariance_mse', float('nan')):.6f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    best_name, best_summary, used_gate = _select_best_run(step7_summaries)
    lines.extend([
        '',
        '## Selected Run',
        '',
    ])
    if best_name is None:
        lines.append('- Pending Step 7 execution.')
    else:
        chart = best_summary.get('chart_quality', {})
        factorized = best_summary.get('factorized_consistency', {})
        j_corr = best_summary.get('j_correlation', {})
        recon = best_summary.get('reconstruction', {})
        lines.extend([
            f"- Selected run: `{best_name}`",
            f"- Gate status: {'passed' if used_gate else 'no run passed the gate; selected best fallback while keeping A2 active'}",
            f"- Quotient chart: eff_dim = {chart.get('effective_dimension', float('nan')):.4f}, overlap = {chart.get('knn_jaccard_mean', float('nan')):.4f}, trust = {chart.get('trustworthiness', float('nan')):.4f}",
            f"- Partner / `j` retention: rank = {factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f}, Spearman = {j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f}",
            f"- Jacobian / logdet: jacobian loss = {factorized.get('quotient_jacobian_gram_loss', float('nan')):.6f}, logdet loss = {factorized.get('quotient_logdet_loss', float('nan')):.6f}",
            f"- Decoder / reconstruction: decoder mse = {factorized.get('decoder_equivariance_mse', float('nan')):.6f}, reconstruction MSE = {recon.get('mse', float('nan')):.6g}",
        ])

    lines.extend([
        '',
        '## Interpretation',
        '',
        '- Read Step 7 as an A2 test of whether a Jacobian-like local metric term can improve quotient spread without losing the Step 4 partner-rank and `j` retention gains.',
        '- After selecting the Step 7 winner, run the full topology anchor rerun and update the roadmap snapshot in the same phase.',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    experiments: list[tuple[str, Callable[[], object]]] | None = None,
    summary_filename: str = 'lattice_step7_summaries.json',
    report_path: str = 'walkthrough-lattice-step7.md',
    step3_anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
    step4_anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
    step6_anchor_summary_path: str = 'runs/lattice_step6_summaries.json',
) -> dict[str, dict]:
    """Run the Step 7 lattice sweep."""
    experiments = experiments or EXPERIMENTS
    all_summaries = {}

    for name, config_factory in experiments:
        print(f"\n{'=' * 60}")
        print(f"  Step 7 Experiment: {name}")
        print(f"{'=' * 60}\n")
        config = config_factory()
        workdir = os.path.join(base_dir, name)
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, workdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, workdir)
        all_summaries[name] = summary

    summary_path = os.path.join(base_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    write_step7_report(
        all_summaries,
        output_path=report_path,
        step3_anchor_summary_path=step3_anchor_summary_path,
        step4_anchor_summary_path=step4_anchor_summary_path,
        step6_anchor_summary_path=step6_anchor_summary_path,
    )

    print(f"\n{'=' * 60}")
    print('  All Step 7 lattice experiments complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f"{'=' * 60}")

    print(
        f"\n{'Experiment':<58} {'Gate':>5} {'q-rank':>8} {'effΔ':>8} "
        f"{'overlap':>8} {'trust':>8} {'jac':>10} {'logdet':>10} "
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
            f"{name:<58} "
            f"{('PASS' if _passes_gate(summary) else 'FAIL'):>5} "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):>8.4f} "
            f"{eff_penalty:>8.4f} "
            f"{chart.get('knn_jaccard_mean', float('nan')):>8.4f} "
            f"{chart.get('trustworthiness', float('nan')):>8.4f} "
            f"{factorized.get('quotient_jacobian_gram_loss', float('nan')):>10.6f} "
            f"{factorized.get('quotient_logdet_loss', float('nan')):>10.6f} "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):>10.4f} "
            f"{recon.get('mse', float('nan')):>10.6f}"
        )

    return all_summaries


if __name__ == '__main__':
    run_all()
