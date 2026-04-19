"""Post-hoc Phase B comparison for topology-diagnostics payloads."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np

from eval.topology import (
    load_diagram_payload,
    plot_phaseb_diagram_distance,
    plot_phaseb_h1_diagram_grid,
    plot_phaseb_h1_trajectory,
    tda_dependencies_available,
)
from run_latent_topology_diagnostics import (
    DEFAULT_EXPERIMENTS,
    _collapse_to_one,
    _dim_metrics,
    _run_topology_for_experiment,
    _stable_to_two,
    classify_branch,
)


RUN_ORDER = [
    't2_standard',
    't2_torus',
    'lattice_standard_norm',
    'lattice_standard_norm_inv',
    'lattice_vae_norm_beta001',
    'lattice_vae_norm_inv_b010_l100',
    'lattice_vae_norm_inv_b030_l100',
    'lattice_vae_wide_norm_inv_b003_l030',
]
FUNDAMENTAL_RUNS = [
    'lattice_vae_norm_inv_b010_l100',
    'lattice_vae_norm_inv_b030_l100',
]
WIDE_RUN = 'lattice_vae_wide_norm_inv_b003_l030'


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _fmt(value, precision: int = 4) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, (float, int, np.floating, np.integer)):
        return f'{float(value):.{precision}f}'
    return str(value)


def _ensure_phasea_run_outputs(
    spec: dict,
    base_dir: str,
    diagnostics_dir: str,
) -> dict:
    """Load one Phase A diagnostic run, regenerating payloads if needed."""
    run_name = spec['name']
    run_output_dir = os.path.join(diagnostics_dir, run_name)
    summary_path = os.path.join(run_output_dir, 'summary.json')
    payload_path = os.path.join(run_output_dir, 'diagram_payload.npz')

    if os.path.exists(summary_path) and os.path.exists(payload_path):
        return _load_json(summary_path)

    if not tda_dependencies_available():
        raise RuntimeError(
            f'Missing payload for {run_name} and optional TDA dependencies are not available. '
            'Install with: pip install ".[tda]" or pip install ripser persim'
        )

    return _run_topology_for_experiment(spec, base_dir, diagnostics_dir)


def _payload_path(run_summary: dict, diagnostics_dir: str) -> str:
    run_dir = os.path.join(diagnostics_dir, run_summary['name'])
    payload_name = run_summary.get('diagram_payload', 'diagram_payload.npz')
    return os.path.join(run_dir, payload_name)


def _trajectory_drop(summary: dict, metric_key: str) -> tuple[float | None, float | None]:
    dim3 = _dim_metrics(summary, 3)
    dim2 = _dim_metrics(summary, 2)
    dim1 = _dim_metrics(summary, 1)
    drop_32 = None
    drop_21 = None
    if dim3 and dim2:
        value3 = dim3.get(metric_key)
        value2 = dim2.get(metric_key)
        if value3 is not None and value2 is not None:
            drop_32 = float(value2) - float(value3)
    if dim2 and dim1:
        value2 = dim2.get(metric_key)
        value1 = dim1.get(metric_key)
        if value2 is not None and value1 is not None:
            drop_21 = float(value1) - float(value2)
    return drop_32, drop_21


def _diagram_shift(summary: dict, dim: int, key: str) -> float:
    distance = _dim_metrics(summary, dim).get('diagram_distance_to_prev')
    if not distance:
        return 0.0
    return float(distance.get(key, 0.0))


def _supports_equivariant_transition(summary: dict) -> bool:
    """Whether one representative run supports moving to equivariant/factorized latent."""
    if not (_stable_to_two(summary) and _collapse_to_one(summary)):
        return False

    dim2 = _dim_metrics(summary, 2)
    dim1 = _dim_metrics(summary, 1)
    h1_ratio = dim2.get('h1_total_persistence', 0.0) / max(
        dim1.get('h1_total_persistence', 0.0), 1e-8,
    )
    shift_32 = _diagram_shift(summary, 2, 'h1_bottleneck')
    shift_21 = _diagram_shift(summary, 1, 'h1_bottleneck')
    return h1_ratio >= 2.0 and (shift_21 >= shift_32 or shift_32 == 0.0)


def _wide_dominates(
    wide_summary: dict | None,
    fundamental_summaries: list[dict],
) -> bool:
    """Whether the wide-sampling run clearly dominates the fundamental-domain runs."""
    if wide_summary is None or not _stable_to_two(wide_summary):
        return False

    dim2_wide = _dim_metrics(wide_summary, 2)
    dim2_fundamental = [_dim_metrics(summary, 2) for summary in fundamental_summaries if summary]
    if not dim2_fundamental:
        return False

    score = 0
    if dim2_wide.get('trustworthiness', 0.0) >= max(
        dim.get('trustworthiness', 0.0) for dim in dim2_fundamental
    ) + 0.02:
        score += 1
    if dim2_wide.get('knn_jaccard_mean', 0.0) >= max(
        dim.get('knn_jaccard_mean', 0.0) for dim in dim2_fundamental
    ) + 0.005:
        score += 1

    rank_candidates = [
        dim.get('partner_rank_percentile_mean')
        for dim in dim2_fundamental
        if dim.get('partner_rank_percentile_mean') is not None
    ]
    if rank_candidates and dim2_wide.get('partner_rank_percentile_mean') is not None:
        if dim2_wide['partner_rank_percentile_mean'] <= min(rank_candidates) - 0.03:
            score += 1

    if dim2_wide.get('h1_total_persistence', 0.0) >= max(
        dim.get('h1_total_persistence', 0.0) for dim in dim2_fundamental
    ) + 1.0:
        score += 1

    return score >= 3


def choose_next_branch(
    topology_runs: dict[str, dict],
    phasea_branch: dict,
) -> dict:
    """Choose the most plausible immediate next research branch."""
    source_branch = phasea_branch.get('branch', 'A')
    fundamental_summaries = [
        topology_runs.get(name) for name in FUNDAMENTAL_RUNS
        if topology_runs.get(name) is not None
    ]
    supported_runs = [
        name for name in FUNDAMENTAL_RUNS
        if topology_runs.get(name) is not None and _supports_equivariant_transition(topology_runs[name])
    ]
    wide_summary = topology_runs.get(WIDE_RUN)
    wide_dominates = _wide_dominates(wide_summary, fundamental_summaries)

    evidence = [
        f"Phase A branch: `{source_branch}`.",
    ]
    if supported_runs:
        evidence.append(
            'Fundamental-domain runs supporting a 2D quotient chart transition: '
            + ', '.join(f'`{name}`' for name in supported_runs) + '.'
        )
    if wide_summary is not None:
        wide_dim2 = _dim_metrics(wide_summary, 2)
        evidence.append(
            f"`{WIDE_RUN}` at k=2: trust={_fmt(wide_dim2.get('trustworthiness'))}, "
            f"overlap={_fmt(wide_dim2.get('knn_jaccard_mean'))}, "
            f"rank={_fmt(wide_dim2.get('partner_rank_percentile_mean'))}, "
            f"H1={_fmt(wide_dim2.get('h1_total_persistence'))}."
        )

    if source_branch == 'E' or wide_dominates:
        return {
            'primary_branch': 'A3',
            'summary': 'Wide sampling now looks like the most actionable bottleneck, so sampling redesign should come before new latent symmetries.',
            'recommended_next_step': 'Design a lattice sampling/coverage experiment before implementing a new latent action.',
            'evidence': evidence,
        }

    if source_branch == 'A' and supported_runs:
        return {
            'primary_branch': 'A1',
            'summary': 'The best fundamental-domain VAE+invariance runs still support a stable 2D quotient chart through k=2, so the next model step should encode that chart explicitly.',
            'recommended_next_step': 'Prototype an equivariant or factorized latent model that preserves a 2D quotient chart.',
            'evidence': evidence,
        }

    return {
        'primary_branch': 'A2',
        'summary': 'PH comparisons still suggest chart fragility in the fundamental-domain runs, so geometry-preserving regularization should come before a structured latent action.',
        'recommended_next_step': 'Add a chart-preserving regularizer and rerun the representative lattice comparison.',
        'evidence': evidence,
    }


def write_phaseb_report(
    topology_runs: dict[str, dict],
    decision: dict,
    phasea_branch: dict,
    output_path: str,
) -> None:
    """Write the Phase B markdown report."""
    lines = [
        '# Topology Diagnostics Phase B',
        '',
        '## Summary',
        '',
        '- Phase B focuses on PH trajectories themselves, with `4→3→2→1` used as the main comparison axis.',
        f"- Source Phase A branch: `{phasea_branch['branch']}`",
        f"- Phase B primary branch: `{decision['primary_branch']}`",
        f"- Interpretation: {decision['summary']}",
        f"- Recommended next step: {decision['recommended_next_step']}",
        '',
        '## Control Anchors',
        '',
        '| Run | k=2 trust | k=1 trust | k=2 H1 total | k=1 H1 total | 2→1 H1 bottleneck |',
        '|---|---|---|---|---|---|',
    ]

    for run_name in ('t2_standard', 't2_torus'):
        summary = topology_runs.get(run_name)
        if summary is None:
            lines.append(f'| `{run_name}` | missing | missing | missing | missing | missing |')
            continue
        dim2 = _dim_metrics(summary, 2)
        dim1 = _dim_metrics(summary, 1)
        lines.append(
            f"| `{run_name}` | {_fmt(dim2.get('trustworthiness'))} | "
            f"{_fmt(dim1.get('trustworthiness'))} | "
            f"{_fmt(dim2.get('h1_total_persistence'))} | "
            f"{_fmt(dim1.get('h1_total_persistence'))} | "
            f"{_fmt(_diagram_shift(summary, 1, 'h1_bottleneck'))} |"
        )

    lines.extend([
        '',
        '## PH Trajectory Comparison',
        '',
        '| Run | k=2 rank | k=2 hit | k=2 trust | k=2 overlap | k=2 eff_dim | k=2 H1 total | k=2 H1 longest | 3→2 H1 bottleneck | 2→1 H1 bottleneck | k=1 Spearman |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ])

    phaseb_run_order = list(RUN_ORDER)
    phaseb_run_order.extend(
        name for name in topology_runs
        if name not in phaseb_run_order
    )

    for run_name in phaseb_run_order:
        if run_name.startswith('t2_'):
            continue
        summary = topology_runs.get(run_name)
        if summary is None:
            lines.append(f'| `{run_name}` | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |')
            continue
        dim2 = _dim_metrics(summary, 2)
        dim1 = _dim_metrics(summary, 1)
        lines.append(
            f"| `{run_name}` | {_fmt(dim2.get('partner_rank_percentile_mean'))} | "
            f"{_fmt(dim2.get('partner_knn_hit_rate'))} | "
            f"{_fmt(dim2.get('trustworthiness'))} | "
            f"{_fmt(dim2.get('knn_jaccard_mean'))} | "
            f"{_fmt(dim2.get('effective_dimension'))} | "
            f"{_fmt(dim2.get('h1_total_persistence'))} | "
            f"{_fmt(dim2.get('h1_longest_bar'))} | "
            f"{_fmt(_diagram_shift(summary, 2, 'h1_bottleneck'))} | "
            f"{_fmt(_diagram_shift(summary, 1, 'h1_bottleneck'))} | "
            f"{_fmt(dim1.get('max_abs_logabsj_spearman'))} |"
        )

    lines.extend([
        '',
        '## Key Reading',
        '',
        '- `lattice_standard_norm_inv` remains the reference 1D ribbon baseline: orbit gluing is strong, but the PH trajectory is already thin at `k=2`.',
        '- `lattice_vae_norm_beta001` remains the VAE-only control: it opens a 2D chart more than the invariant AE baseline, but does not carry the strongest quotient signal.',
        '- `lattice_vae_norm_inv_b010_l100` and `lattice_vae_norm_inv_b030_l100` are the main Phase B runs. They are the ones to read first when deciding whether the quotient chart is genuinely 2D down to `k=2`.',
        '- `lattice_vae_wide_norm_inv_b003_l030` is the coverage control. Read it as a sampling probe, not as the default successor model.',
        '',
        '## Decision',
        '',
        f"- Primary branch: `{decision['primary_branch']}`",
        f"- Summary: {decision['summary']}",
        f"- Next step: {decision['recommended_next_step']}",
        '',
        '### Active Branches',
        '',
        '- `A1`: Move to an equivariant or factorized latent model if the best fundamental-domain VAE+invariance runs stay stable through `k=2` and collapse only at `k=1`.',
        '- `A2`: Add a chart-preserving regularizer first if the fundamental-domain PH trajectory is still fragile before `k=2`.',
        '- `A3`: Prefer sampling redesign if the wide run clearly outperforms the fundamental-domain runs at `k=2`.',
        '',
        '### Evidence',
        '',
    ])
    lines.extend([f'- {item}' for item in decision['evidence']])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_roadmap(
    decision: dict,
    phasea_branch: dict,
    output_path: str,
) -> None:
    """Write a short living roadmap focused on the immediate next branches."""
    lines = [
        '# AE Latent Study Roadmap',
        '',
        '## Current Node',
        '',
        f"- Confirmed state: `Branch {phasea_branch['branch']}`",
        '- Current diagnostic basis: `Topology Phase B`',
        f"- Current recommendation: `{decision['primary_branch']}`",
        f"- Why now: {decision['summary']}",
        '',
        '## Active Branches',
        '',
        '### A1. Equivariant / Factorized Latent',
        '',
        '- Trigger: the best fundamental-domain VAE+invariance runs remain stable through `k=2` and collapse mainly at `k=1`.',
        '- Immediate implementation idea: encode a 2D quotient chart explicitly, then separate invariant and group-action parts in latent space.',
        '',
        '### A2. Chart-Preserving Regularizer',
        '',
        '- Trigger: PH still shows fragility before `k=2` in the fundamental-domain runs.',
        '- Immediate implementation idea: add a regularizer that preserves local chart geometry before introducing a richer latent action.',
        '',
        '### A3. Sampling Redesign',
        '',
        '- Trigger: the wide-sampling representative run clearly outperforms the fundamental-domain runs at `k=2`.',
        '- Immediate implementation idea: rebalance lattice sampling and density control, then rerun the representative PH comparison.',
        '',
        '## Parked Branches',
        '',
        '- `non-Euclidean latent`: parked until the Euclidean quotient-chart path is exhausted.',
        '- `latent function space / spectral basis`: parked until the quotient chart itself is more stable.',
        '- `T^2 × R_+ control`: parked unless the current torus controls stop being sufficient.',
        '',
        '## Update Trigger',
        '',
        '- Update this roadmap whenever `walkthrough-topology-phaseB.md` changes its primary branch recommendation.',
        '- Promote `A1` if the fundamental-domain PH trajectories stay stable through `k=2` and the wide run does not clearly dominate.',
        '- Promote `A2` if the PH comparison still looks fragile before `k=2`.',
        '- Promote `A3` if the wide run becomes the clearest `k=2` winner.',
    ]

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    diagnostics_dir: str = 'runs/topology_diagnostics',
    summary_filename: str = 'phaseB_comparison_summary.json',
    report_path: str = 'walkthrough-topology-phaseB.md',
    roadmap_path: str = 'ae-latent-study-roadmap.md',
    experiments: list[dict] | None = None,
) -> dict:
    """Run the post-hoc Phase B comparison on existing topology diagnostics."""
    os.makedirs(diagnostics_dir, exist_ok=True)
    experiments = experiments or DEFAULT_EXPERIMENTS

    topology_runs = {}
    payloads = {}
    for spec in experiments:
        summary = _ensure_phasea_run_outputs(spec, base_dir, diagnostics_dir)
        topology_runs[spec['name']] = summary
        payloads[spec['name']] = load_diagram_payload(
            _payload_path(summary, diagnostics_dir),
        )

    phasea_branch = classify_branch(topology_runs)
    decision = choose_next_branch(topology_runs, phasea_branch)

    ordered_runs = [name for name in RUN_ORDER if name in topology_runs]
    ordered_runs.extend(
        spec['name']
        for spec in experiments
        if spec['name'] in topology_runs and spec['name'] not in ordered_runs
    )

    fig = plot_phaseb_h1_trajectory(
        topology_runs,
        run_order=ordered_runs,
        save_path=os.path.join(diagnostics_dir, 'phaseB_h1_trajectory.png'),
    )
    plt.close(fig)
    fig = plot_phaseb_diagram_distance(
        topology_runs,
        run_order=ordered_runs,
        save_path=os.path.join(diagnostics_dir, 'phaseB_diagram_distance.png'),
    )
    plt.close(fig)
    fig = plot_phaseb_h1_diagram_grid(
        payloads,
        dim=2,
        run_order=ordered_runs,
        save_path=os.path.join(diagnostics_dir, 'phaseB_diagram_grid_k2.png'),
    )
    plt.close(fig)
    fig = plot_phaseb_h1_diagram_grid(
        payloads,
        dim=1,
        run_order=ordered_runs,
        save_path=os.path.join(diagnostics_dir, 'phaseB_diagram_grid_k1.png'),
    )
    plt.close(fig)

    combined = {
        'source_phaseA_branch': phasea_branch,
        'run_order': ordered_runs,
        'topology_diagnostics': topology_runs,
        'phaseB_decision': decision,
    }

    summary_path = os.path.join(diagnostics_dir, summary_filename)
    with open(summary_path, 'w') as f:
        json.dump(combined, f, indent=2)

    write_phaseb_report(topology_runs, decision, phasea_branch, output_path=report_path)
    write_roadmap(decision, phasea_branch, output_path=roadmap_path)

    print(f"\n{'=' * 60}")
    print('  Topology Phase B comparison complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f'  Roadmap: {roadmap_path}')
    print(f"{'=' * 60}")

    print(
        f"\n{'Run':<36} {'k=2 rank':>10} {'k=2 hit':>10} "
        f"{'k=2 H1':>10} {'2->1 bneck':>12}"
    )
    print('-' * 84)
    for name in ordered_runs:
        summary = topology_runs[name]
        dim2 = _dim_metrics(summary, 2)
        print(
            f"{name:<36} "
            f"{(float('nan') if dim2.get('partner_rank_percentile_mean') is None else dim2.get('partner_rank_percentile_mean')):>10.4f} "
            f"{(float('nan') if dim2.get('partner_knn_hit_rate') is None else dim2.get('partner_knn_hit_rate')):>10.4f} "
            f"{dim2.get('h1_total_persistence', float('nan')):>10.4f} "
            f"{_diagram_shift(summary, 1, 'h1_bottleneck'):>12.4f}"
        )

    print(f"\nPrimary next branch: {decision['primary_branch']}")
    print(decision['summary'])
    return combined


if __name__ == '__main__':
    run_all()
