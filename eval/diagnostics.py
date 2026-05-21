"""Pure-logic library extracted from topology diagnostics scripts.

This module hosts the experiment specs, accessors, predicates, and
classification functions that were previously embedded in
``run_latent_topology_diagnostics.py`` and
``run_topology_phaseB_comparison.py``.

The scripts retain their CLI entry points (``run_all``) and side-effectful
orchestration helpers (``_run_topology_for_experiment``, ``write_*_report``).
Everything in this module is side-effect free: callers pass dict-shaped
summaries in, get values out.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Experiment registries (constants)
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENTS = [
    {
        'name': 't2_standard',
        'kind': 'control',
        'config_source': 'configs.t2_standard',
    },
    {
        'name': 't2_torus',
        'kind': 'control',
        'config_source': 'configs.t2_torus',
    },
    {
        'name': 'lattice_standard_norm',
        'kind': 'lattice',
        'config_source': 'configs.lattice_standard_norm',
    },
    {
        'name': 'lattice_standard_norm_inv',
        'kind': 'lattice',
        'config_source': 'configs.lattice_standard_norm_inv',
    },
    {
        'name': 'lattice_vae_norm_beta001',
        'kind': 'lattice',
        'config_source': 'configs.lattice_vae_norm_beta001',
    },
    {
        'name': 'lattice_vae_norm_inv_b010_l100',
        'kind': 'lattice',
    },
    {
        'name': 'lattice_vae_norm_inv_b030_l100',
        'kind': 'lattice',
    },
    {
        'name': 'lattice_vae_wide_norm_inv_b003_l030',
        'kind': 'lattice',
    },
]


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


# ---------------------------------------------------------------------------
# Low-level accessors
# ---------------------------------------------------------------------------

def dim_metrics(run_summary: dict, dim: int) -> dict:
    """Convenience accessor for one projection-dimension summary."""
    return run_summary['topology_diagnostics']['dims'].get(str(dim), {})


def fmt_metric(value, precision: int = 4) -> str:
    """Format a scalar metric for reports."""
    if value is None:
        return 'n/a'
    if isinstance(value, (float, int, np.floating, np.integer)):
        return f'{float(value):.{precision}f}'
    return str(value)


# ---------------------------------------------------------------------------
# Phase A predicates (consume topology_diagnostics output)
# ---------------------------------------------------------------------------

def control_calibrated(run_name: str, control_summary: dict) -> tuple[bool, str]:
    """Whether one control run shows the expected 2D-stable / 1D-collapse pattern."""
    dim2 = dim_metrics(control_summary, 2)
    dim1 = dim_metrics(control_summary, 1)
    if not dim2 or not dim1:
        return False, f'`{run_name}` is missing k=2 or k=1 diagnostics.'

    conditions = {
        'eff_dim': 1.7 <= dim2.get('effective_dimension', 0.0) <= 2.3,
        'trust_k2': dim2.get('trustworthiness', 0.0) >= 0.85,
        'trust_gap': (
            dim2.get('trustworthiness', 0.0) - dim1.get('trustworthiness', 0.0)
        ) >= 0.15,
        'overlap_gap': (
            dim2.get('knn_jaccard_mean', 0.0) - dim1.get('knn_jaccard_mean', 0.0)
        ) >= 0.08,
        'h1_collapse': dim1.get('h1_total_persistence', 0.0) <= 0.1 * max(
            dim2.get('h1_total_persistence', 0.0), 1e-8,
        ),
    }
    evidence = (
        f"`{run_name}`: eff2={fmt_metric(dim2.get('effective_dimension'))}, "
        f"trust2={fmt_metric(dim2.get('trustworthiness'))}, "
        f"trust drop={fmt_metric(dim2.get('trustworthiness', 0.0) - dim1.get('trustworthiness', 0.0))}, "
        f"overlap drop={fmt_metric(dim2.get('knn_jaccard_mean', 0.0) - dim1.get('knn_jaccard_mean', 0.0))}, "
        f"H1(1)/H1(2)={fmt_metric(dim1.get('h1_total_persistence', 0.0) / max(dim2.get('h1_total_persistence', 0.0), 1e-8))}"
    )
    return all(conditions.values()), evidence


def stable_to_two(run_summary: dict) -> bool:
    """Whether a run looks stable down to k=2 under PCA projection."""
    dims = run_summary['topology_diagnostics']['dims']
    full_dim = max(int(key) for key in dims)
    dim_full = dims[str(full_dim)]
    dim2 = dims.get('2')
    if dim2 is None:
        return False

    eff_ok = 1.4 <= dim2.get('effective_dimension', 0.0) <= 2.4
    trust_ok = dim2.get('trustworthiness', 0.0) >= dim_full.get('trustworthiness', 0.0) - 0.03
    overlap_ok = dim2.get('knn_jaccard_mean', 0.0) >= 0.8 * dim_full.get('knn_jaccard_mean', 0.0)

    spearman_full = dim_full.get('max_abs_logabsj_spearman', None)
    spearman_2 = dim2.get('max_abs_logabsj_spearman', None)
    if spearman_full is None or spearman_2 is None:
        spearman_ok = True
    else:
        spearman_ok = spearman_2 >= 0.8 * spearman_full

    rank_full = dim_full.get('partner_rank_percentile_mean', None)
    rank_2 = dim2.get('partner_rank_percentile_mean', None)
    if rank_full is None or rank_2 is None:
        rank_ok = True
    else:
        rank_ok = rank_2 <= rank_full + 0.05

    hit_full = dim_full.get('partner_knn_hit_rate', None)
    hit_2 = dim2.get('partner_knn_hit_rate', None)
    if hit_full is None or hit_2 is None:
        hit_ok = True
    else:
        hit_ok = hit_2 >= 0.8 * hit_full

    return eff_ok and trust_ok and overlap_ok and spearman_ok and rank_ok and hit_ok


def collapse_to_one(run_summary: dict) -> bool:
    """Whether a run shows a clear degradation from k=2 to k=1."""
    dim2 = dim_metrics(run_summary, 2)
    dim1 = dim_metrics(run_summary, 1)
    if not dim2 or not dim1:
        return False

    conditions = [
        dim1.get('trustworthiness', 0.0) <= dim2.get('trustworthiness', 0.0) - 0.05,
        dim1.get('knn_jaccard_mean', 0.0) <= 0.75 * dim2.get('knn_jaccard_mean', 0.0),
        dim1.get('h1_total_persistence', 0.0) <= 0.1 * max(dim2.get('h1_total_persistence', 0.0), 1e-8),
    ]

    spearman_2 = dim2.get('max_abs_logabsj_spearman', None)
    spearman_1 = dim1.get('max_abs_logabsj_spearman', None)
    if spearman_2 is not None and spearman_1 is not None:
        conditions.append(spearman_1 <= 0.5 * spearman_2)

    rank_2 = dim2.get('partner_rank_percentile_mean', None)
    rank_1 = dim1.get('partner_rank_percentile_mean', None)
    if rank_2 is not None and rank_1 is not None:
        conditions.append(rank_1 >= rank_2 + 0.10)

    return sum(bool(condition) for condition in conditions) >= 2


def projection_artifact(run_summary: dict) -> bool:
    """Whether H1 gains appear to be projection artifacts."""
    dims = run_summary['topology_diagnostics']['dims']
    full_dim = max(int(key) for key in dims)
    dim_full = dims[str(full_dim)]
    dim2 = dims.get('2')
    if dim2 is None:
        return False

    pca_h1 = dim2.get('h1_total_persistence', 0.0)
    full_h1 = dim_full.get('h1_total_persistence', 0.0)
    if pca_h1 < 1.5 * max(full_h1, 1e-8):
        return False

    baseline = dim2.get('random_projection_baseline', {})
    overlap_baseline = baseline.get('knn_jaccard_mean', {'mean': -np.inf, 'std': 0.0})
    hit_baseline = baseline.get('partner_knn_hit_rate', {'mean': -np.inf, 'std': 0.0})
    spearman_baseline = baseline.get('max_abs_logabsj_spearman', {'mean': -np.inf, 'std': 0.0})

    overlap_not_better = dim2.get('knn_jaccard_mean', 0.0) <= (
        overlap_baseline['mean'] + overlap_baseline['std']
    )
    hit_not_better = dim2.get('partner_knn_hit_rate', 0.0) <= (
        hit_baseline['mean'] + hit_baseline['std']
    )
    spearman_not_better = dim2.get('max_abs_logabsj_spearman', 0.0) <= (
        spearman_baseline['mean'] + spearman_baseline['std']
    )

    return overlap_not_better and hit_not_better and spearman_not_better


# ---------------------------------------------------------------------------
# Phase B analysis
# ---------------------------------------------------------------------------

def trajectory_drop(summary: dict, metric_key: str) -> tuple[float | None, float | None]:
    """Differences in one metric across k=3 → k=2 and k=2 → k=1 transitions."""
    dim3 = dim_metrics(summary, 3)
    dim2 = dim_metrics(summary, 2)
    dim1 = dim_metrics(summary, 1)
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


def diagram_shift(summary: dict, dim: int, key: str) -> float:
    """Bottleneck/Wasserstein shift between consecutive projection diagrams."""
    distance = dim_metrics(summary, dim).get('diagram_distance_to_prev')
    if not distance:
        return 0.0
    return float(distance.get(key, 0.0))


def supports_equivariant_transition(summary: dict) -> bool:
    """Whether one representative run supports moving to equivariant/factorized latent."""
    if not (stable_to_two(summary) and collapse_to_one(summary)):
        return False

    dim2 = dim_metrics(summary, 2)
    dim1 = dim_metrics(summary, 1)
    h1_ratio = dim2.get('h1_total_persistence', 0.0) / max(
        dim1.get('h1_total_persistence', 0.0), 1e-8,
    )
    shift_32 = diagram_shift(summary, 2, 'h1_bottleneck')
    shift_21 = diagram_shift(summary, 1, 'h1_bottleneck')
    return h1_ratio >= 2.0 and (shift_21 >= shift_32 or shift_32 == 0.0)


def wide_dominates(
    wide_summary: dict | None,
    fundamental_summaries: list[dict],
) -> bool:
    """Whether the wide-sampling run clearly dominates the fundamental-domain runs."""
    if wide_summary is None or not stable_to_two(wide_summary):
        return False

    dim2_wide = dim_metrics(wide_summary, 2)
    dim2_fundamental = [dim_metrics(summary, 2) for summary in fundamental_summaries if summary]
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


def focus_collapse_visible(summary: dict) -> tuple[bool, list[str]]:
    """Return whether k=1 collapse is visible for a focus run."""
    dim2 = dim_metrics(summary, 2)
    dim1 = dim_metrics(summary, 1)
    conditions = []
    if dim1.get('trustworthiness', 0.0) <= dim2.get('trustworthiness', 0.0) - 0.05:
        conditions.append('trust drops from k=2 to k=1')
    if dim1.get('knn_jaccard_mean', 0.0) <= 0.75 * dim2.get('knn_jaccard_mean', 0.0):
        conditions.append('overlap drops from k=2 to k=1')
    if dim1.get('h1_total_persistence', 0.0) <= 0.1 * dim2.get('h1_total_persistence', 0.0):
        conditions.append('H1 collapses from k=2 to k=1')
    if dim1.get('max_abs_logabsj_spearman', 0.0) <= 0.5 * dim2.get('max_abs_logabsj_spearman', 0.0):
        conditions.append('j Spearman drops from k=2 to k=1')
    return len(conditions) >= 2, conditions


# ---------------------------------------------------------------------------
# High-level classifiers (public API)
# ---------------------------------------------------------------------------

def classify_branch(topology_runs: dict[str, dict]) -> dict:
    """Classify the current research branch from control and lattice diagnostics."""
    control_standard = topology_runs.get('t2_standard')
    control_torus = topology_runs.get('t2_torus')
    lattice_b010 = topology_runs.get('lattice_vae_norm_inv_b010_l100')
    lattice_b030 = topology_runs.get('lattice_vae_norm_inv_b030_l100')
    wide = topology_runs.get('lattice_vae_wide_norm_inv_b003_l030')

    evidence = []
    if control_standard is None or control_torus is None:
        return {
            'branch': 'C',
            'summary': 'Missing one of the pure-torus control diagnostics, so calibration is incomplete.',
            'recommended_next_step': 'Run the control diagnostics before interpreting lattice geometry.',
            'evidence': evidence,
        }

    control_standard_ok, standard_evidence = control_calibrated('t2_standard', control_standard)
    control_torus_ok, torus_evidence = control_calibrated('t2_torus', control_torus)
    evidence.extend([standard_evidence, torus_evidence])
    if not (control_standard_ok and control_torus_ok):
        return {
            'branch': 'C',
            'summary': 'The pure-torus control does not yet show the expected 2D-stable / 1D-collapse pattern.',
            'recommended_next_step': 'Debug the PH pipeline, projection ladder, and noise-floor choices before using lattice conclusions.',
            'evidence': evidence,
        }

    key_runs = [
        (name, summary)
        for name, summary in (
            ('lattice_vae_norm_inv_b010_l100', lattice_b010),
            ('lattice_vae_norm_inv_b030_l100', lattice_b030),
        )
        if summary is not None
    ]
    stable_names = [name for name, summary in key_runs if stable_to_two(summary)]
    collapse_names = [name for name, summary in key_runs if collapse_to_one(summary)]
    stable_fundamental = bool(stable_names)
    collapse_fundamental = any(name in collapse_names for name in stable_names)
    wide_stable = wide is not None and stable_to_two(wide)

    if not stable_fundamental and wide_stable:
        evidence.append('`lattice_vae_wide_norm_inv_b003_l030` stays stable to k=2 while the fundamental-domain VAE+inv runs do not.')
        return {
            'branch': 'E',
            'summary': 'Wide lattice coverage remains stable to k=2 while fundamental-domain runs degrade earlier.',
            'recommended_next_step': 'Redesign lattice sampling and density control before changing the model class.',
            'evidence': evidence,
        }

    artifact_names = [name for name, summary in key_runs if projection_artifact(summary)]
    if artifact_names:
        evidence.append(f'Projection-artifact warning triggered for: {", ".join(f"`{name}`" for name in artifact_names)}.')
        return {
            'branch': 'D',
            'summary': 'The strongest low-dimensional H1 signal looks comparable to random-projection artifacts and degrades local geometry.',
            'recommended_next_step': 'Treat those loops as projection artifacts and keep using geometry-preservation metrics as the main guide.',
            'evidence': evidence,
        }

    if stable_fundamental and collapse_fundamental:
        evidence.append(
            f'Stable-to-k=2 runs: {", ".join(f"`{name}`" for name in stable_names)}; '
            f'2->1 collapse is visible in: {", ".join(f"`{name}`" for name in collapse_names)}.'
        )
        return {
            'branch': 'A',
            'summary': 'The best VAE+invariance runs remain comparatively stable down to k=2, then collapse at k=1.',
            'recommended_next_step': 'Move to equivariant or factorized latent models that explicitly preserve a 2D quotient chart.',
            'evidence': evidence,
        }

    if not stable_fundamental:
        evidence.append('Neither fundamental-domain VAE+inv representative run satisfies the stable-to-k=2 criteria.')
        return {
            'branch': 'B',
            'summary': 'Orbit gluing is present, but lattice quotient geometry already weakens at the 3->2 transition.',
            'recommended_next_step': 'Add chart-preserving regularization before moving to more structured latent actions.',
            'evidence': evidence,
        }

    evidence.append(
        f'Stable-to-k=2 runs: {", ".join(f"`{name}`" for name in stable_names)}; '
        'the calibrated controls support a genuinely 2D quotient interpretation.'
    )
    return {
        'branch': 'A',
        'summary': 'The best lattice runs look stable to k=2, which supports a genuinely 2D quotient geometry.',
        'recommended_next_step': 'Use this as the basis for equivariant or factorized latent model design.',
        'evidence': evidence,
    }


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
        if topology_runs.get(name) is not None and supports_equivariant_transition(topology_runs[name])
    ]
    wide_summary = topology_runs.get(WIDE_RUN)
    wide_is_dominant = wide_dominates(wide_summary, fundamental_summaries)

    evidence = [
        f"Phase A branch: `{source_branch}`.",
    ]
    if supported_runs:
        evidence.append(
            'Fundamental-domain runs supporting a 2D quotient chart transition: '
            + ', '.join(f'`{name}`' for name in supported_runs) + '.'
        )
    if wide_summary is not None:
        wide_dim2 = dim_metrics(wide_summary, 2)
        evidence.append(
            f"`{WIDE_RUN}` at k=2: trust={fmt_metric(wide_dim2.get('trustworthiness'))}, "
            f"overlap={fmt_metric(wide_dim2.get('knn_jaccard_mean'))}, "
            f"rank={fmt_metric(wide_dim2.get('partner_rank_percentile_mean'))}, "
            f"H1={fmt_metric(wide_dim2.get('h1_total_persistence'))}."
        )

    if source_branch == 'E' or wide_is_dominant:
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


def choose_focus_branch(
    topology_runs: dict[str, dict],
    focus_run_name: str | None,
) -> dict | None:
    """Classify a latest focus run alongside the global Phase B decision."""
    if not focus_run_name:
        return None

    summary = topology_runs.get(focus_run_name)
    if summary is None:
        return {
            'focus_run_name': focus_run_name,
            'primary_branch': 'A2 continues',
            'summary': 'Focus run was not available in the topology comparison, so the factorized branch cannot be promoted.',
            'recommended_next_step': 'Regenerate topology diagnostics for the selected focus run.',
            'accepted': False,
            'evidence': [f'Focus run `{focus_run_name}` is missing.'],
        }

    dim2 = dim_metrics(summary, 2)
    collapse_visible, collapse_evidence = focus_collapse_visible(summary)
    criteria = {
        'k=2 rank <= 0.15': dim2.get('partner_rank_percentile_mean', float('inf')) <= 0.15,
        'k=2 overlap >= 0.058': dim2.get('knn_jaccard_mean', float('-inf')) >= 0.058,
        'k=2 eff_dim >= 1.55': dim2.get('effective_dimension', float('-inf')) >= 1.55,
        'k=2 j Spearman >= 0.85': dim2.get('max_abs_logabsj_spearman', float('-inf')) >= 0.85,
        'k=1 collapse visible': collapse_visible,
    }
    accepted = all(criteria.values())
    evidence = [
        f"Focus run: `{focus_run_name}`.",
        f"k=2 rank={fmt_metric(dim2.get('partner_rank_percentile_mean'))}, "
        f"overlap={fmt_metric(dim2.get('knn_jaccard_mean'))}, "
        f"eff_dim={fmt_metric(dim2.get('effective_dimension'))}, "
        f"j={fmt_metric(dim2.get('max_abs_logabsj_spearman'))}.",
        'Passed criteria: '
        + (', '.join(name for name, passed in criteria.items() if passed) or 'none')
        + '.',
        'Failed criteria: '
        + (', '.join(name for name, passed in criteria.items() if not passed) or 'none')
        + '.',
    ]
    if collapse_evidence:
        evidence.append('Collapse evidence: ' + '; '.join(collapse_evidence) + '.')

    if accepted:
        return {
            'focus_run_name': focus_run_name,
            'primary_branch': 'A1-return candidate',
            'summary': 'The focus factorized run satisfies the k=2 chart / partner / j criteria and still collapses at k=1.',
            'recommended_next_step': 'Treat this run as an A1 return candidate and compare it against the next model-family options.',
            'accepted': True,
            'criteria': criteria,
            'evidence': evidence,
        }

    return {
        'focus_run_name': focus_run_name,
        'primary_branch': 'A2 continues',
        'summary': 'The global VAE-anchor decision may remain A1, but the latest factorized focus run has not met the balanced k=2 criteria.',
        'recommended_next_step': 'Continue A2 with stronger contrastive / semantic geometry for the factorized branch.',
        'accepted': False,
        'criteria': criteria,
        'evidence': evidence,
    }
