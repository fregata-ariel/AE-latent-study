"""Run Step 9 factorized lattice experiments with Step 7 teacher distillation."""

import json
import os
from collections.abc import Callable

from configs.lattice_default import get_config as get_lattice_config
from eval.analysis import run_full_evaluation
from train.trainer import train_and_evaluate


STEP4_ANCHOR = 'lattice_factorized_vae_fd_b030_q100_g030_d030'
STEP6_FALLBACK = 'lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030'
STEP7_TEACHER = 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010'
STEP8_WINNER = 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100'
STEP3_ANCHORS = [
    'lattice_standard_norm_inv',
    'lattice_vae_norm_inv_b010_l100',
    'lattice_vae_norm_inv_b030_l100',
]


def _make_step9_config(
    teacher_run_dir: str,
    teacher_distill_weight: float,
    rank_weight: float,
    logdet_weight: float,
):
    """Create a Step 9 config anchored on the selected Step 4 scaffold."""
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
    config.train.jacobian_gram_weight = 0.0
    config.train.jacobian_n_neighbors = 8
    config.train.quotient_logdet_weight = logdet_weight
    config.train.quotient_logdet_ratio_target = 0.10
    config.train.j_rank_preserving_weight = rank_weight
    config.train.j_rank_temperature = 0.10
    config.train.j_rank_min_delta = 0.10
    config.train.j_rank_n_terms = 50
    config.train.teacher_distill_weight = teacher_distill_weight
    config.train.teacher_run_dir = os.path.abspath(teacher_run_dir)
    config.train.teacher_distill_n_neighbors = 8
    config.train.teacher_distill_view = 'quotient'
    config.train.teacher_distill_loss_type = 'local_distance'
    return config


def _default_experiments(base_dir: str) -> list[tuple[str, Callable[[], object]]]:
    teacher_run_dir = os.path.join(base_dir, STEP7_TEACHER)
    return [
        (
            'lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld010',
            lambda: _make_step9_config(teacher_run_dir, 0.03, 0.03, 0.01),
        ),
        (
            'lattice_factorized_vae_fd_b030_q100_g030_d030_td100_r030_ld010',
            lambda: _make_step9_config(teacher_run_dir, 0.10, 0.03, 0.01),
        ),
        (
            'lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r100_ld010',
            lambda: _make_step9_config(teacher_run_dir, 0.03, 0.10, 0.01),
        ),
        (
            'lattice_factorized_vae_fd_b030_q100_g030_d030_td100_r100_ld010',
            lambda: _make_step9_config(teacher_run_dir, 0.10, 0.10, 0.01),
        ),
        (
            'lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld000',
            lambda: _make_step9_config(teacher_run_dir, 0.03, 0.03, 0.00),
        ),
        (
            'lattice_factorized_vae_fd_b030_q100_g030_d030_td100_r030_ld000',
            lambda: _make_step9_config(teacher_run_dir, 0.10, 0.03, 0.00),
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
    mod_inv = summary.get('modular_invariance', {})
    recon = summary.get('reconstruction', {})
    return (
        factorized.get('quotient_partner_rank_percentile_mean', float('inf')) <= 0.20
        and j_corr.get('max_abs_logabsj_spearman', float('-inf')) >= 0.88
        and chart.get('effective_dimension', float('-inf')) >= 1.55
        and chart.get('knn_jaccard_mean', float('-inf')) >= 0.055
        and chart.get('trustworthiness', float('-inf')) >= 0.845
        and mod_inv.get('mean_latent_distance', float('inf')) <= 0.02
        and factorized.get('decoder_equivariance_mse', float('inf')) <= 5e-4
        and recon.get('mse', float('inf')) <= 5e-5
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
        factorized.get('quotient_partner_rank_percentile_mean', float('inf')),
        -j_corr.get('max_abs_logabsj_spearman', float('-inf')),
        eff_penalty,
        -chart.get('knn_jaccard_mean', float('-inf')),
        mod_inv.get('mean_latent_distance', float('inf')),
        factorized.get('quotient_teacher_distill_loss', float('inf')),
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
        f"{factorized.get('quotient_teacher_distill_loss', float('nan')):.6f} | "
        f"{recon.get('mse', float('nan')):.6g} |"
    )


def write_step9_report(
    step9_summaries: dict[str, dict],
    output_path: str = 'walkthrough-lattice-step9.md',
    step3_anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
    step4_anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
    step6_anchor_summary_path: str = 'runs/lattice_step6_summaries.json',
    step7_anchor_summary_path: str = 'runs/lattice_step7_summaries.json',
    step8_anchor_summary_path: str = 'runs/lattice_step8_summaries.json',
) -> None:
    """Write the Step 9 markdown report."""
    step3_anchors = _load_anchor_summaries(step3_anchor_summary_path)
    step4_anchors = _load_anchor_summaries(step4_anchor_summary_path)
    step6_anchors = _load_anchor_summaries(step6_anchor_summary_path)
    step7_anchors = _load_anchor_summaries(step7_anchor_summary_path)
    step8_anchors = _load_anchor_summaries(step8_anchor_summary_path)

    lines = [
        '# Lattice Step 9 — Step 7 Teacher Quotient-Structure Distillation',
        '',
        '## Summary',
        '',
        '- Step 9 keeps the factorized VAE scaffold fixed and uses the Step 7 winner as a frozen teacher.',
        '- The teacher target is local quotient distance structure, not raw quotient coordinates.',
        '- Step 9 keeps `j` rank retention and optional logdet guard, while disabling Step 5/6/7 local/spread terms.',
        '',
        '## Anchors',
        '',
        '| Run | Source | q rank | q trust | q overlap | q eff_dim | log10_abs_j Spearman | teacher loss | MSE |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for name in STEP3_ANCHORS:
        _append_anchor_row(lines, name, 'Step 3', step3_anchors.get(name))
    for source, run_name, anchor_map in (
        ('Step 4', STEP4_ANCHOR, step4_anchors),
        ('Step 6', STEP6_FALLBACK, step6_anchors),
        ('Step 7 teacher', STEP7_TEACHER, step7_anchors),
        ('Step 8 winner', STEP8_WINNER, step8_anchors),
    ):
        _append_anchor_row(lines, run_name, source, anchor_map.get(run_name))

    lines.extend([
        '',
        '## Step 9 Runs',
        '',
        '| Run | Gate | q rank | q trust | q overlap | q eff_dim | teacher loss | pairwise corr | j-rank loss | logdet loss | log10_abs_j Spearman | mod dist | decode mse | MSE |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ])
    for name, summary in step9_summaries.items():
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
            f"{factorized.get('quotient_teacher_distill_loss', float('nan')):.6f} | "
            f"{factorized.get('student_teacher_pairwise_distance_corr', float('nan')):.4f} | "
            f"{factorized.get('quotient_j_rank_loss', float('nan')):.6f} | "
            f"{factorized.get('quotient_logdet_loss', float('nan')):.6f} | "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f} | "
            f"{mod_inv.get('mean_latent_distance', float('nan')):.6f} | "
            f"{factorized.get('decoder_equivariance_mse', float('nan')):.6f} | "
            f"{recon.get('mse', float('nan')):.6g} |"
        )

    best_name, best_summary, used_gate = _select_best_run(step9_summaries)
    lines.extend(['', '## Selected Run', ''])
    if best_name is None:
        lines.append('- Pending Step 9 execution.')
    else:
        chart = best_summary.get('chart_quality', {})
        factorized = best_summary.get('factorized_consistency', {})
        j_corr = best_summary.get('j_correlation', {})
        mod_inv = best_summary.get('modular_invariance', {})
        recon = best_summary.get('reconstruction', {})
        lines.extend([
            f"- Selected run: `{best_name}`",
            f"- Gate status: {'passed' if used_gate else 'no run passed the gate; selected best fallback and keep A2 active'}",
            f"- Teacher distillation: loss = {factorized.get('quotient_teacher_distill_loss', float('nan')):.6f}, pairwise corr = {factorized.get('student_teacher_pairwise_distance_corr', float('nan')):.4f}",
            f"- Quotient chart: eff_dim = {chart.get('effective_dimension', float('nan')):.4f}, overlap = {chart.get('knn_jaccard_mean', float('nan')):.4f}, trust = {chart.get('trustworthiness', float('nan')):.4f}",
            f"- Partner / `j`: rank = {factorized.get('quotient_partner_rank_percentile_mean', float('nan')):.4f}, Spearman = {j_corr.get('max_abs_logabsj_spearman', float('nan')):.4f}",
            f"- Modular / decoder / reconstruction: mod dist = {mod_inv.get('mean_latent_distance', float('nan')):.6f}, decoder mse = {factorized.get('decoder_equivariance_mse', float('nan')):.6f}, MSE = {recon.get('mse', float('nan')):.6g}",
        ])

    lines.extend([
        '',
        '## Interpretation',
        '',
        '- If a Step 9 run passes the gate, run the topology follow-up and check whether it improves Step 8 topology-side `j` retention while preserving Step 7 partner/chart quality.',
        '- If no run passes the gate, treat teacher distillation as insufficient and move A2 to contrastive local geometry.',
    ])
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    experiments: list[tuple[str, Callable[[], object]]] | None = None,
    summary_filename: str = 'lattice_step9_summaries.json',
    report_path: str = 'walkthrough-lattice-step9.md',
    step3_anchor_summary_path: str = 'runs/lattice_step3_summaries.json',
    step4_anchor_summary_path: str = 'runs/lattice_step4_summaries.json',
    step6_anchor_summary_path: str = 'runs/lattice_step6_summaries.json',
    step7_anchor_summary_path: str = 'runs/lattice_step7_summaries.json',
    step8_anchor_summary_path: str = 'runs/lattice_step8_summaries.json',
) -> dict[str, dict]:
    """Run the Step 9 lattice sweep."""
    experiments = experiments or _default_experiments(base_dir)
    all_summaries = {}
    for name, config_factory in experiments:
        print(f"\n{'=' * 60}")
        print(f"  Step 9 Experiment: {name}")
        print(f"{'=' * 60}\n")
        config = config_factory()
        workdir = os.path.join(base_dir, name)
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, workdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, workdir)
        all_summaries[name] = summary

    summary_path = os.path.join(base_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    write_step9_report(
        all_summaries,
        output_path=report_path,
        step3_anchor_summary_path=step3_anchor_summary_path,
        step4_anchor_summary_path=step4_anchor_summary_path,
        step6_anchor_summary_path=step6_anchor_summary_path,
        step7_anchor_summary_path=step7_anchor_summary_path,
        step8_anchor_summary_path=step8_anchor_summary_path,
    )

    print(f"\n{'=' * 60}")
    print('  All Step 9 lattice experiments complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f"{'=' * 60}")
    print(
        f"\n{'Experiment':<63} {'Gate':>5} {'q-rank':>8} {'effΔ':>8} "
        f"{'overlap':>8} {'trust':>8} {'teacher':>10} {'log|j|ρ':>10} "
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
            f"{factorized.get('quotient_teacher_distill_loss', float('nan')):>10.6f} "
            f"{j_corr.get('max_abs_logabsj_spearman', float('nan')):>10.4f} "
            f"{mod_inv.get('mean_latent_distance', float('nan')):>9.6f} "
            f"{recon.get('mse', float('nan')):>10.6f}"
        )
    return all_summaries


if __name__ == '__main__':
    run_all()
