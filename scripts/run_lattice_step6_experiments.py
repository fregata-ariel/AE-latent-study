"""Run Step 6 factorized lattice experiments with rotation-aware spread loss."""

import json
import os
from collections.abc import Callable

from configs.lattice_default import get_config as get_lattice_config
from eval.analysis import run_full_evaluation
from train.trainer import train_and_evaluate


STEP4_ANCHOR = 'lattice_factorized_vae_fd_b030_q100_g030_d030'
STEP3_ANCHORS = [
    'lattice_standard_norm_inv',
    'lattice_vae_norm_inv_b010_l100',
    'lattice_vae_norm_inv_b030_l100',
]


def _make_step6_config(
    chart_weight: float,
    spread_weight: float,
):
    """Create a Step 6 config anchored on the selected Step 4 scaffold."""
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
    config.train.quotient_variance_floor_weight = 0.0
    config.train.quotient_variance_floor_target = 0.15
    config.train.quotient_spread_weight = spread_weight
    config.train.quotient_min_eig_ratio_target = 0.20
    config.train.quotient_trace_cap_ratio = 1.50
    return config


EXPERIMENTS = [
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_l010_s010',
        lambda: _make_step6_config(0.01, 0.01),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_l010_s030',
        lambda: _make_step6_config(0.01, 0.03),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030',
        lambda: _make_step6_config(0.02, 0.03),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s050',
        lambda: _make_step6_config(0.02, 0.05),
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
        eff_penalty,
        factorized.get('quotient_partner_rank_percentile_mean', float('inf')),
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


def write_step6_report(
    step6_summaries: dict[str, dict],
    output_path: str = 'walkthrough-lattice-step6.md',
    step3_anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
    step4_anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
) -> None:
    """Write the Step 6 markdown report."""
    step3_anchors = _load_anchor_summaries(step3_anchor_summary_path)
    step4_anchors = _load_anchor_summaries(step4_anchor_summary_path)

    lines = [
        '# Lattice Step 6 — Rotation-Aware Quotient Spread Regularizer',
        '',
        '## Summary',
        '',
        '- Step 6 keeps the Step 4 factorized scaffold fixed and replaces the Step 5 axis-wise spread term with a covariance-eigenvalue spread loss.',
        '- Selection rule first applies a gate: quotient partner rank <= 0.35 and `log10|j|` Spearman >= 0.88.',
        '- Among gated runs, ranking uses quotient effective-dimension penalty, quotient partner rank, quotient overlap, quotient trust, decoder equivariance, then reconstruction MSE.',
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
        rank = factorized.get('quotient_partner_rank_percentile_mean', float("nan"))
        lines.append(
            f"| `{name}` | Step 3 | "
            f"{rank:.4f} | "
            f"{chart.get('trustworthiness', float('nan')):.4f} | "
            f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{chart.get('effective_dimension', float('nan')):.4f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    step4_summary = step4_anchors.get(STEP4_ANCHOR)
    if step4_summary is None:
        lines.append(f'| `{STEP4_ANCHOR}` | Step 4 | missing | missing | missing | missing | missing | missing |')
    else:
        chart = step4_summary.get('chart_quality', {})
        factorized = step4_summary.get('factorized_consistency', {})
        j_corr = step4_summary.get('j_correlation', {})
        recon = step4_summary.get('reconstruction', {})
        lines.append(
            f"| `{STEP4_ANCHOR}` | Step 4 | "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f} | "
            f"{chart.get('trustworthiness', float('nan')):.4f} | "
            f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{chart.get('effective_dimension', float('nan')):.4f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    lines.extend([
        '',
        '## Step 6 Runs',
        '',
        '| Run | Gate | q rank | q trust | q overlap | q eff_dim | q eig_min | q eig_max | q spread loss | log10_abs_j Spearman | decode mse | MSE |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ])

    for name, summary in step6_summaries.items():
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
            f"{factorized.get('quotient_cov_eig_min', float('nan')):.6f} | "
            f"{factorized.get('quotient_cov_eig_max', float('nan')):.6f} | "
            f"{factorized.get('quotient_spread_loss', float('nan')):.6f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{factorized.get('decoder_equivariance_mse', float('nan')):.6f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    best_name, best_summary, used_gate = _select_best_run(step6_summaries)
    lines.extend([
        '',
        '## Selected Run',
        '',
    ])
    if best_name is None:
        lines.append('- Pending Step 6 execution.')
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
            f"- Spread metrics: eig_min = {factorized.get('quotient_cov_eig_min', float('nan')):.6f}, eig_max = {factorized.get('quotient_cov_eig_max', float('nan')):.6f}, spread loss = {factorized.get('quotient_spread_loss', float('nan')):.6f}",
            f"- Decoder / reconstruction: decoder mse = {factorized.get('decoder_equivariance_mse', float('nan')):.6f}, reconstruction MSE = {recon.get('mse', float('nan')):.6g}",
        ])

    lines.extend([
        '',
        '## Interpretation',
        '',
        '- Read Step 6 as an A2 test of whether rotation-aware spread can stop quotient collapse without giving up the Step 4 partner-rank and `j` retention gains.',
        '- Only the Step 6 selected run should be sent to topology Phase A/B against the fixed Step 3/4 anchors.',
        '- If no run passes the gate, keep `A2` active and redesign the local chart term itself before revisiting topology.',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    experiments: list[tuple[str, Callable[[], object]]] | None = None,
    summary_filename: str = 'lattice_step6_summaries.json',
    report_path: str = 'walkthrough-lattice-step6.md',
    step3_anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
    step4_anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
) -> dict[str, dict]:
    """Run the Step 6 lattice sweep."""
    experiments = experiments or EXPERIMENTS
    all_summaries = {}

    for name, config_factory in experiments:
        print(f"\n{'=' * 60}")
        print(f"  Step 6 Experiment: {name}")
        print(f"{'=' * 60}\n")
        config = config_factory()
        workdir = os.path.join(base_dir, name)
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, workdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, workdir)
        all_summaries[name] = summary

    summary_path = os.path.join(base_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    write_step6_report(
        all_summaries,
        output_path=report_path,
        step3_anchor_summary_path=step3_anchor_summary_path,
        step4_anchor_summary_path=step4_anchor_summary_path,
    )

    print(f"\n{'=' * 60}")
    print('  All Step 6 lattice experiments complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f"{'=' * 60}")

    print(
        f"\n{'Experiment':<56} {'Gate':>6} {'q-rank':>8} {'effΔ':>8} "
        f"{'overlap':>8} {'trust':>8} {'eigmin':>10} {'eigmax':>10} "
        f"{'spread':>10} {'log|j|ρ':>10} {'MSE':>10}"
    )
    print('-' * 152)
    for name, summary in all_summaries.items():
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        recon = summary.get('reconstruction', {})
        eff_dim = chart.get('effective_dimension', float('nan'))
        eff_penalty = abs(eff_dim - 2.0) if eff_dim == eff_dim else float('nan')
        print(
            f"{name:<56} "
            f"{('PASS' if _passes_gate(summary) else 'FAIL'):>6} "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):>8.4f} "
            f"{eff_penalty:>8.4f} "
            f"{chart.get('knn_jaccard_mean', float('nan')):>8.4f} "
            f"{chart.get('trustworthiness', float('nan')):>8.4f} "
            f"{factorized.get('quotient_cov_eig_min', float('nan')):>10.6f} "
            f"{factorized.get('quotient_cov_eig_max', float('nan')):>10.6f} "
            f"{factorized.get('quotient_spread_loss', float('nan')):>10.6f} "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):>10.4f} "
            f"{recon.get('mse', float('nan')):>10.6f}"
        )

    return all_summaries


if __name__ == '__main__':
    run_all()
