"""Run Step 8 factorized lattice experiments with pairwise log|j| rank retention."""

import json
import os
from collections.abc import Callable

from configs.lattice_default import get_config as get_lattice_config
from eval.analysis import run_full_evaluation
from train.trainer import train_and_evaluate


STEP4_ANCHOR = 'lattice_factorized_vae_fd_b030_q100_g030_d030'
STEP6_FALLBACK = 'lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030'
STEP7_WINNER = 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010'
STEP3_ANCHORS = [
    'lattice_standard_norm_inv',
    'lattice_vae_norm_inv_b010_l100',
    'lattice_vae_norm_inv_b030_l100',
]


def _make_step8_config(
    jacobian_weight: float,
    logdet_weight: float,
    rank_weight: float,
    jacobian_n_neighbors: int = 8,
):
    """Create a Step 8 config anchored on the selected Step 4 scaffold."""
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
    config.train.jacobian_n_neighbors = jacobian_n_neighbors
    config.train.quotient_logdet_weight = logdet_weight
    config.train.quotient_logdet_ratio_target = 0.10
    config.train.j_rank_preserving_weight = rank_weight
    config.train.j_rank_temperature = 0.10
    config.train.j_rank_min_delta = 0.10
    config.train.j_rank_n_terms = 50
    return config


EXPERIMENTS = [
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r030',
        lambda: _make_step8_config(0.01, 0.01, 0.03, 8),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100',
        lambda: _make_step8_config(0.01, 0.01, 0.10, 8),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld000_r030',
        lambda: _make_step8_config(0.01, 0.00, 0.03, 8),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j005_ld010_r030',
        lambda: _make_step8_config(0.005, 0.01, 0.03, 8),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j005_ld000_r100',
        lambda: _make_step8_config(0.005, 0.00, 0.10, 8),
    ),
    (
        'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r030_k4',
        lambda: _make_step8_config(0.01, 0.01, 0.03, 4),
    ),
]


def _load_anchor_summaries(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _passes_gate(summary: dict) -> bool:
    chart = summary.get('chart_quality', {})
    factorized = summary.get('factorized_consistency', {})
    j_corr = summary.get('j_correlation', {})
    return (
        factorized.get('quotient_partner_rank_percentile_mean', float('inf')) <= 0.25
        and j_corr.get('max_abs_logabsj_spearman', float('-inf')) >= 0.88
        and chart.get('effective_dimension', float('-inf')) >= 1.50
    )


def _selection_key(summary: dict) -> tuple[float, float, float, float, float, float, float, float]:
    chart = summary.get('chart_quality', {})
    factorized = summary.get('factorized_consistency', {})
    j_corr = summary.get('j_correlation', {})
    mod_inv = summary.get('modular_invariance', {})
    recon = summary.get('reconstruction', {})

    eff_dim = chart.get('effective_dimension', float('nan'))
    eff_penalty = abs(eff_dim - 2.0) if eff_dim == eff_dim else float('inf')
    return (
        -j_corr.get('max_abs_logabsj_spearman', float('-inf')),
        factorized.get('quotient_partner_rank_percentile_mean', float('inf')),
        eff_penalty,
        -chart.get('knn_jaccard_mean', float('-inf')),
        -chart.get('trustworthiness', float('-inf')),
        mod_inv.get('mean_latent_distance', float('inf')),
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


def _append_anchor_row(lines: list[str], name: str, source: str, summary: dict | None) -> None:
    if summary is None:
        lines.append(
            f'| `{name}` | {source} | missing | missing | missing | missing | missing | missing | missing |',
        )
        return

    chart = summary.get('chart_quality', {})
    factorized = summary.get('factorized_consistency', {})
    j_corr = summary.get('j_correlation', {})
    recon = summary.get('reconstruction', {})
    lines.append(
        f"| `{name}` | {source} | "
        f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f} | "
        f"{chart.get('trustworthiness', float('nan')):.4f} | "
        f"{chart.get('knn_jaccard_mean', float('nan')):.4f} | "
        f"{chart.get('effective_dimension', float('nan')):.4f} | "
        f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
        f"{factorized.get('quotient_j_rank_loss', float('nan')):.6f} | "
        f"{recon.get('mse', float('nan')):.6g} |"
    )


def write_step8_report(
    step8_summaries: dict[str, dict],
    output_path: str = 'walkthrough-lattice-step8.md',
    step3_anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
    step4_anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
    step6_anchor_summary_path: str = 'runs/lattice_step6_summaries.json',
    step7_anchor_summary_path: str = 'runs/lattice_step7_summaries.json',
) -> None:
    """Write the Step 8 markdown report."""
    step3_anchors = _load_anchor_summaries(step3_anchor_summary_path)
    step4_anchors = _load_anchor_summaries(step4_anchor_summary_path)
    step6_anchors = _load_anchor_summaries(step6_anchor_summary_path)
    step7_anchors = _load_anchor_summaries(step7_anchor_summary_path)

    lines = [
        '# Lattice Step 8 — Step 7b Pairwise log|j| Rank Retention',
        '',
        '## Summary',
        '',
        '- Step 8 keeps the Step 7 factorized scaffold and `Local Gram match + logdet guard` structure.',
        '- The new diagnostic term preserves pairwise `log10|j|` ordering along a batch-adaptive quotient direction.',
        '- The goal is to test whether Step 7 lost global quotient semantics rather than proving the whole A2 regularizer family failed.',
        '',
        '## Anchors',
        '',
        '| Run | Source | q rank | q trust | q overlap | q eff_dim | log10_abs_j Spearman | j-rank loss | MSE |',
        '|---|---|---|---|---|---|---|---|---|',
    ]

    for name in STEP3_ANCHORS:
        _append_anchor_row(lines, name, 'Step 3', step3_anchors.get(name))

    for source, run_name, anchor_map in (
        ('Step 4', STEP4_ANCHOR, step4_anchors),
        ('Step 6', STEP6_FALLBACK, step6_anchors),
        ('Step 7', STEP7_WINNER, step7_anchors),
    ):
        _append_anchor_row(lines, run_name, source, anchor_map.get(run_name))

    lines.extend([
        '',
        '## Step 8 Runs',
        '',
        '| Run | Gate | q rank | q trust | q overlap | q eff_dim | jacobian loss | logdet loss | j-rank loss | j target std | log10_abs_j Spearman | mod dist | decode mse | MSE |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ])

    for name, summary in step8_summaries.items():
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        mod_inv = summary.get('modular_invariance', {})
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
            f"{factorized.get('quotient_j_rank_loss', float('nan')):.6f} | "
            f"{factorized.get('quotient_j_rank_target_std', float('nan')):.4f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{mod_inv.get('mean_latent_distance', float('nan')):.6f} | "
            f"{factorized.get('decoder_equivariance_mse', float('nan')):.6f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    best_name, best_summary, used_gate = _select_best_run(step8_summaries)
    lines.extend([
        '',
        '## Selected Run',
        '',
    ])
    if best_name is None:
        lines.append('- Pending Step 8 execution.')
    else:
        chart = best_summary.get('chart_quality', {})
        factorized = best_summary.get('factorized_consistency', {})
        j_corr = best_summary.get('j_correlation', {})
        mod_inv = best_summary.get('modular_invariance', {})
        recon = best_summary.get('reconstruction', {})
        lines.extend([
            f"- Selected run: `{best_name}`",
            f"- Gate status: {'passed' if used_gate else 'no run passed the gate; selected best fallback while keeping A2 active'}",
            f"- `j` retention: Spearman = {j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f}, j-rank loss = {factorized.get('quotient_j_rank_loss', float('nan')):.6f}",
            f"- Quotient chart: eff_dim = {chart.get('effective_dimension', float('nan')):.4f}, overlap = {chart.get('knn_jaccard_mean', float('nan')):.4f}, trust = {chart.get('trustworthiness', float('nan')):.4f}",
            f"- Partner / modular: rank = {factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f}, modular distance = {mod_inv.get('mean_latent_distance', float('nan')):.6f}",
            f"- Local / global losses: jacobian = {factorized.get('quotient_jacobian_gram_loss', float('nan')):.6f}, logdet = {factorized.get('quotient_logdet_loss', float('nan')):.6f}",
            f"- Decoder / reconstruction: decoder mse = {factorized.get('decoder_equivariance_mse', float('nan')):.6f}, reconstruction MSE = {recon.get('mse', float('nan')):.6g}",
        ])

    lines.extend([
        '',
        '## Interpretation',
        '',
        '- If the selected run restores `log10|j|` Spearman while keeping Step 7 chart spread, Step 7 failure was likely semantic erosion rather than the whole Jacobian-like family failing.',
        '- If no run passes the gate, treat this as evidence that small Step 7 variants are insufficient and keep A2 active toward teacher distillation or contrastive local geometry.',
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    experiments: list[tuple[str, Callable[[], object]]] | None = None,
    summary_filename: str = 'lattice_step8_summaries.json',
    report_path: str = 'walkthrough-lattice-step8.md',
    step3_anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
    step4_anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
    step6_anchor_summary_path: str = 'runs/lattice_step6_summaries.json',
    step7_anchor_summary_path: str = 'runs/lattice_step7_summaries.json',
) -> dict[str, dict]:
    """Run the Step 8 lattice sweep."""
    experiments = experiments or EXPERIMENTS
    all_summaries = {}

    for name, config_factory in experiments:
        print(f"\n{'=' * 60}")
        print(f"  Step 8 Experiment: {name}")
        print(f"{'=' * 60}\n")
        config = config_factory()
        workdir = os.path.join(base_dir, name)
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, workdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, workdir)
        all_summaries[name] = summary

    summary_path = os.path.join(base_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    write_step8_report(
        all_summaries,
        output_path=report_path,
        step3_anchor_summary_path=step3_anchor_summary_path,
        step4_anchor_summary_path=step4_anchor_summary_path,
        step6_anchor_summary_path=step6_anchor_summary_path,
        step7_anchor_summary_path=step7_anchor_summary_path,
    )

    print(f"\n{'=' * 60}")
    print('  All Step 8 lattice experiments complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f"{'=' * 60}")

    print(
        f"\n{'Experiment':<63} {'Gate':>5} {'q-rank':>8} {'effΔ':>8} "
        f"{'overlap':>8} {'trust':>8} {'j-rank':>10} {'log|j|ρ':>10} "
        f"{'moddist':>9} {'MSE':>10}"
    )
    print('-' * 156)
    for name, summary in all_summaries.items():
        chart = summary.get('chart_quality', {})
        factorized = summary.get('factorized_consistency', {})
        j_corr = summary.get('j_correlation', {})
        mod_inv = summary.get('modular_invariance', {})
        recon = summary.get('reconstruction', {})
        eff_dim = chart.get('effective_dimension', float('nan'))
        eff_penalty = abs(eff_dim - 2.0) if eff_dim == eff_dim else float('nan')
        print(
            f"{name:<63} "
            f"{('PASS' if _passes_gate(summary) else 'FAIL'):>5} "
            f"{factorized.get('quotient_partner_rank_percentile_mean', float('nan')):>8.4f} "
            f"{eff_penalty:>8.4f} "
            f"{chart.get('knn_jaccard_mean', float('nan')):>8.4f} "
            f"{chart.get('trustworthiness', float('nan')):>8.4f} "
            f"{factorized.get('quotient_j_rank_loss', float('nan')):>10.6f} "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):>10.4f} "
            f"{mod_inv.get('mean_latent_distance', float('nan')):>9.6f} "
            f"{recon.get('mse', float('nan')):>10.6f}"
        )

    return all_summaries


if __name__ == '__main__':
    run_all()
