"""Run topology diagnostics for control and lattice representative experiments."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable

import jax
import ml_collections
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import create_splits
from eval.metrics import encode_dataset
from eval.topology import (
    diagnose_projection_ladder,
    encode_lattice_partner_latent,
    make_reference_coords,
    plot_persistence_panels,
    plot_projection_comparison,
    plot_topology_metrics_vs_k,
    tda_dependencies_available,
)
from models import create_model
from train.checkpointing import create_checkpoint_manager, restore_checkpoint
from train.train_state import create_train_state
from train.trainer import train_and_evaluate


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


def _deterministic_subsample_indices(n_total: int, max_samples: int) -> np.ndarray:
    """Return deterministic sorted subsample indices."""
    if max_samples <= 0 or n_total <= max_samples:
        return np.arange(n_total)

    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n_total, size=max_samples, replace=False))


def _step3_factory_map() -> dict[str, Callable[[], object]]:
    """Map generated Step 3 run names to their config factories."""
    try:
        from run_lattice_step3_experiments import EXPERIMENTS as step3_experiments
    except Exception:
        return {}

    return {name: factory for name, factory in step3_experiments}


def _load_config(config_source: str | Callable[[], object]):
    """Load a config from a module path or factory."""
    if callable(config_source):
        return config_source()

    module = importlib.import_module(config_source)
    return module.get_config()


def _load_config_from_json(config_path: str):
    """Load a ConfigDict from a saved config.json file."""
    with open(config_path) as f:
        return ml_collections.ConfigDict(json.load(f))


def _ensure_topology_defaults(config) -> None:
    """Backfill topology-eval defaults for older saved configs."""
    if not hasattr(config, 'eval'):
        config.eval = ml_collections.ConfigDict()

    defaults = {
        'ph_enabled': False,
        'ph_max_samples': 2000,
        'ph_proj_dims': (),
        'ph_maxdim': 1,
        'ph_random_projection_trials': 8,
        'ph_knn_for_lid': 10,
        'ph_noise_floor': 0.05,
        'chart_n_neighbors': 8,
        'output_dir': 'results/',
    }
    for key, value in defaults.items():
        if not hasattr(config.eval, key):
            setattr(config.eval, key, value)


def _data_generation_key(seed: int):
    """Reproduce the trainer's data-generation PRNG split."""
    key = jax.random.PRNGKey(seed)
    _, data_key = jax.random.split(key)
    return data_key


def _best_checkpoint_step(workdir: str, history: dict, checkpoint_dir: str) -> int:
    """Pick the best checkpoint step, with a fallback to the latest available."""
    best_step = int(np.argmin(history['val_loss']))
    checkpoint_path = os.path.join(workdir, checkpoint_dir)
    checkpoint_files = sorted(
        [
            fname for fname in os.listdir(checkpoint_path)
            if fname.endswith('.msgpack')
        ],
        key=lambda name: int(name.split('.')[0]),
    )
    available_steps = [int(name.split('.')[0]) for name in checkpoint_files]
    if best_step in available_steps:
        return best_step
    if not available_steps:
        raise FileNotFoundError(f'No checkpoints found in {checkpoint_path}')
    return available_steps[-1]


def _load_state_from_run(workdir: str, config):
    """Restore the best checkpointed state for an existing run directory."""
    history_path = os.path.join(workdir, 'history.json')
    if not os.path.exists(history_path):
        raise FileNotFoundError(f'Missing history.json in {workdir}')

    with open(history_path) as f:
        history = json.load(f)

    model = create_model(config)
    state_template = create_train_state(config, model, jax.random.PRNGKey(config.seed))
    checkpoint_manager = create_checkpoint_manager(config, workdir)
    step = _best_checkpoint_step(workdir, history, config.checkpoint.dir)
    state = restore_checkpoint(checkpoint_manager, step, state_template)
    return state, history


def _materialize_experiment(spec: dict, base_dir: str):
    """Load an existing run or train it if no checkpointed run exists."""
    run_name = spec['name']
    workdir = os.path.join(base_dir, spec.get('run_dir_name', run_name))
    config_path = os.path.join(workdir, 'config.json')
    checkpoint_dir = os.path.join(workdir, 'checkpoints')

    if os.path.exists(config_path) and os.path.isdir(checkpoint_dir):
        config = _load_config_from_json(config_path)
        _ensure_topology_defaults(config)
        state, history = _load_state_from_run(workdir, config)
    else:
        config_source = spec.get('config_source')
        step3_factories = _step3_factory_map()
        if config_source is None and run_name in step3_factories:
            config_source = step3_factories[run_name]
        if config_source is None:
            raise FileNotFoundError(
                f'No saved run found for {run_name} and no config source is available.'
            )

        config = _load_config(config_source)
        _ensure_topology_defaults(config)
        state, history, _ = train_and_evaluate(config, workdir)

    train_ds, _, _ = create_splits(config, _data_generation_key(config.seed))
    return workdir, config, state, history, train_ds


def _run_topology_for_experiment(
    spec: dict,
    base_dir: str,
    diagnostics_dir: str,
) -> dict:
    """Run topology diagnostics for a single materialized experiment."""
    run_name = spec['name']
    kind = spec['kind']
    workdir, config, state, history, train_ds = _materialize_experiment(spec, base_dir)
    is_vae = config.model.latent_type == 'vae'

    z_all = np.asarray(encode_dataset(state, train_ds, is_vae=is_vae))
    reference_coords, reference_label = make_reference_coords(train_ds, config)

    max_samples = int(getattr(config.eval, 'ph_max_samples', 2000))
    subset_idx = _deterministic_subsample_indices(len(train_ds), max_samples)
    z_subset = z_all[subset_idx]
    reference_subset = reference_coords[subset_idx]

    j_subset = None
    partner_subset = None
    if getattr(config.data, 'data_type', 'torus') == 'lattice':
        if train_ds.j_invariant is not None:
            j_subset = np.asarray(train_ds.j_invariant)[subset_idx]
        partner_subset = encode_lattice_partner_latent(
            state, train_ds, config, subset_idx, is_vae=is_vae,
        )

    diagnostics_summary, diagnostics_artifacts = diagnose_projection_ladder(
        z_subset,
        reference_subset,
        projection_dims=getattr(config.eval, 'ph_proj_dims', ()),
        n_neighbors=getattr(config.eval, 'chart_n_neighbors', 8),
        lid_neighbors=getattr(config.eval, 'ph_knn_for_lid', 10),
        maxdim=getattr(config.eval, 'ph_maxdim', 1),
        max_samples=0,
        noise_floor=getattr(config.eval, 'ph_noise_floor', 0.05),
        random_projection_trials=getattr(config.eval, 'ph_random_projection_trials', 8),
        j_values=j_subset,
        partner_latent=partner_subset,
        reference_label=reference_label,
    )

    run_output_dir = os.path.join(diagnostics_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    fig = plot_topology_metrics_vs_k(
        diagnostics_summary,
        save_path=os.path.join(run_output_dir, 'metrics_vs_k.png'),
    )
    plt.close(fig)
    fig = plot_persistence_panels(
        diagnostics_summary,
        diagnostics_artifacts,
        save_path=os.path.join(run_output_dir, 'persistence_panels.png'),
    )
    plt.close(fig)
    fig = plot_projection_comparison(
        diagnostics_summary,
        save_path=os.path.join(run_output_dir, 'projection_comparison.png'),
    )
    plt.close(fig)

    run_summary = {
        'name': run_name,
        'kind': kind,
        'workdir': workdir,
        'reference_space': reference_label,
        'topology_diagnostics': diagnostics_summary,
    }
    with open(os.path.join(run_output_dir, 'summary.json'), 'w') as f:
        json.dump(run_summary, f, indent=2)

    print(f'Topology diagnostics saved to {run_output_dir}')
    return run_summary


def _dim_metrics(run_summary: dict, dim: int) -> dict:
    """Convenience accessor for one projection-dimension summary."""
    return run_summary['topology_diagnostics']['dims'].get(str(dim), {})


def _control_calibrated(control_summary: dict) -> bool:
    """Whether the primary T^2 control exhibits a stable 2D regime."""
    dim4 = _dim_metrics(control_summary, 4)
    dim2 = _dim_metrics(control_summary, 2)
    dim1 = _dim_metrics(control_summary, 1)
    if not dim4 or not dim2 or not dim1:
        return False

    trust_ok = dim2.get('trustworthiness', 0.0) >= 0.85 * dim4.get('trustworthiness', 0.0)
    overlap_ok = dim2.get('knn_jaccard_mean', 0.0) >= 0.75 * dim4.get('knn_jaccard_mean', 0.0)
    collapse_to_one = (
        dim1.get('trustworthiness', 0.0) <= dim2.get('trustworthiness', 0.0) - 0.05
        or dim1.get('knn_jaccard_mean', 0.0) <= 0.75 * dim2.get('knn_jaccard_mean', 0.0)
        or dim1.get('h1_total_persistence', 0.0) <= 0.8 * dim2.get('h1_total_persistence', 0.0)
    )
    return trust_ok and overlap_ok and collapse_to_one


def _stable_to_two(run_summary: dict) -> bool:
    """Whether a run looks stable down to k=2 under PCA projection."""
    dims = run_summary['topology_diagnostics']['dims']
    full_dim = max(int(key) for key in dims)
    dim_full = dims[str(full_dim)]
    dim2 = dims.get('2')
    if dim2 is None:
        return False

    trust_ok = dim2.get('trustworthiness', 0.0) >= dim_full.get('trustworthiness', 0.0) - 0.05
    overlap_ok = dim2.get('knn_jaccard_mean', 0.0) >= 0.8 * dim_full.get('knn_jaccard_mean', 0.0)

    spearman_full = dim_full.get('max_abs_logabsj_spearman', None)
    spearman_2 = dim2.get('max_abs_logabsj_spearman', None)
    if spearman_full is None or spearman_2 is None:
        spearman_ok = True
    else:
        spearman_ok = spearman_2 >= 0.8 * spearman_full

    mod_full = dim_full.get('projected_modular_distance', None)
    mod_2 = dim2.get('projected_modular_distance', None)
    if mod_full is None or mod_2 is None:
        modular_ok = True
    else:
        modular_ok = mod_2 <= max(3.0 * mod_full, 1e-8)

    return trust_ok and overlap_ok and spearman_ok and modular_ok


def _collapse_to_one(run_summary: dict) -> bool:
    """Whether a run shows a clear degradation from k=2 to k=1."""
    dim2 = _dim_metrics(run_summary, 2)
    dim1 = _dim_metrics(run_summary, 1)
    if not dim2 or not dim1:
        return False

    return (
        dim1.get('trustworthiness', 0.0) <= dim2.get('trustworthiness', 0.0) - 0.05
        or dim1.get('knn_jaccard_mean', 0.0) <= 0.7 * dim2.get('knn_jaccard_mean', 0.0)
        or dim1.get('h1_total_persistence', 0.0) <= 0.5 * dim2.get('h1_total_persistence', 0.0)
        or (
            dim1.get('projected_modular_distance') is not None
            and dim2.get('projected_modular_distance') is not None
            and dim1.get('projected_modular_distance', 0.0) >= 3.0 * dim2.get('projected_modular_distance', 0.0)
        )
    )


def _projection_artifact(run_summary: dict) -> bool:
    """Whether H1 gains appear to be projection artifacts."""
    dims = run_summary['topology_diagnostics']['dims']
    full_dim = max(int(key) for key in dims)
    dim_full = dims[str(full_dim)]
    dim2 = dims.get('2')
    if dim2 is None:
        return False

    pca_h1 = dim2.get('h1_total_persistence', 0.0)
    full_h1 = dim_full.get('h1_total_persistence', 0.0)
    baseline = dim2.get('random_projection_baseline', {}).get('h1_total_persistence', {})
    random_h1 = baseline.get('mean', 0.0)
    trust_bad = dim2.get('trustworthiness', 0.0) <= dim_full.get('trustworthiness', 0.0) - 0.05
    overlap_bad = dim2.get('knn_jaccard_mean', 0.0) <= 0.8 * dim_full.get('knn_jaccard_mean', 0.0)

    return (
        pca_h1 >= 1.5 * max(full_h1, 1e-8)
        and random_h1 >= 0.8 * pca_h1
        and (trust_bad or overlap_bad)
    )


def classify_branch(topology_runs: dict[str, dict]) -> dict:
    """Classify the current research branch from control and lattice diagnostics."""
    control = topology_runs.get('t2_standard')
    lattice_b010 = topology_runs.get('lattice_vae_norm_inv_b010_l100')
    lattice_b030 = topology_runs.get('lattice_vae_norm_inv_b030_l100')
    wide = topology_runs.get('lattice_vae_wide_norm_inv_b003_l030')

    evidence = []
    if control is None:
        return {
            'branch': 'C',
            'summary': 'Missing T^2 control diagnostics, so calibration is incomplete.',
            'recommended_next_step': 'Run the control diagnostics before interpreting lattice geometry.',
            'evidence': evidence,
        }

    control_ok = _control_calibrated(control)
    if not control_ok:
        return {
            'branch': 'C',
            'summary': 'The pure-torus control does not yet show the expected 2D-stable / 1D-collapse pattern.',
            'recommended_next_step': 'Debug the PH pipeline, projection ladder, and noise-floor choices before using lattice conclusions.',
            'evidence': evidence,
        }

    key_runs = [summary for summary in (lattice_b010, lattice_b030) if summary is not None]
    stable_fundamental = any(_stable_to_two(summary) for summary in key_runs)
    collapse_fundamental = any(_collapse_to_one(summary) for summary in key_runs)
    wide_stable = wide is not None and _stable_to_two(wide)

    if not stable_fundamental and wide_stable:
        return {
            'branch': 'E',
            'summary': 'Wide lattice coverage remains stable to k=2 while fundamental-domain runs degrade earlier.',
            'recommended_next_step': 'Redesign lattice sampling and density control before changing the model class.',
            'evidence': evidence,
        }

    if any(_projection_artifact(summary) for summary in key_runs if summary is not None):
        return {
            'branch': 'D',
            'summary': 'The strongest low-dimensional H1 signal looks comparable to random-projection artifacts and degrades local geometry.',
            'recommended_next_step': 'Treat those loops as projection artifacts and keep using geometry-preservation metrics as the main guide.',
            'evidence': evidence,
        }

    if stable_fundamental and collapse_fundamental:
        return {
            'branch': 'A',
            'summary': 'The best VAE+invariance runs remain comparatively stable down to k=2, then collapse at k=1.',
            'recommended_next_step': 'Move to equivariant or factorized latent models that explicitly preserve a 2D quotient chart.',
            'evidence': evidence,
        }

    if not stable_fundamental:
        return {
            'branch': 'B',
            'summary': 'Orbit gluing is present, but lattice quotient geometry already weakens at the 3->2 transition.',
            'recommended_next_step': 'Add chart-preserving regularization before moving to more structured latent actions.',
            'evidence': evidence,
        }

    return {
        'branch': 'A',
        'summary': 'The best lattice runs look stable to k=2, which supports a genuinely 2D quotient geometry.',
        'recommended_next_step': 'Use this as the basis for equivariant or factorized latent model design.',
        'evidence': evidence,
    }


def write_report(
    topology_runs: dict[str, dict],
    branch_assessment: dict,
    output_path: str = 'walkthrough-topology-phaseA.md',
) -> None:
    """Write a markdown summary of the topology diagnostics."""
    lines = [
        '# Topology Diagnostics Phase A',
        '',
        '## Summary',
        '',
        '- Persistent Homology is used here as a diagnostic for projection stability, not as a proof of the true quotient topology.',
        f"- Branch classification: **{branch_assessment['branch']}**",
        f"- Interpretation: {branch_assessment['summary']}",
        f"- Recommended next step: {branch_assessment['recommended_next_step']}",
        '',
        '## Control Calibration',
        '',
        '| Run | k=4 trust | k=2 trust | k=2 overlap | k=1 trust | k=1 overlap |',
        '|---|---|---|---|---|---|',
    ]

    for run_name in ('t2_standard', 't2_torus'):
        summary = topology_runs.get(run_name)
        if summary is None:
            lines.append(f'| `{run_name}` | missing | missing | missing | missing | missing |')
            continue

        dim4 = _dim_metrics(summary, 4)
        dim2 = _dim_metrics(summary, 2)
        dim1 = _dim_metrics(summary, 1)
        lines.append(
            f"| `{run_name}` | {dim4.get('trustworthiness', float('nan')):.4f} | "
            f"{dim2.get('trustworthiness', float('nan')):.4f} | "
            f"{dim2.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{dim1.get('trustworthiness', float('nan')):.4f} | "
            f"{dim1.get('knn_jaccard_mean', float('nan')):.4f} |"
        )

    lines.extend([
        '',
        '## Lattice Representatives',
        '',
        '| Run | full-k moddist | k=2 trust | k=2 overlap | k=2 eff_dim | k=2 H1 total | k=1 moddist | k=1 Spearman |',
        '|---|---|---|---|---|---|---|---|',
    ])

    lattice_run_names = [
        'lattice_standard_norm',
        'lattice_standard_norm_inv',
        'lattice_vae_norm_beta001',
        'lattice_vae_norm_inv_b010_l100',
        'lattice_vae_norm_inv_b030_l100',
        'lattice_vae_wide_norm_inv_b003_l030',
    ]
    for run_name in lattice_run_names:
        summary = topology_runs.get(run_name)
        if summary is None:
            lines.append(f'| `{run_name}` | missing | missing | missing | missing | missing | missing | missing |')
            continue

        dims = summary['topology_diagnostics']['dims']
        full_dim = max(int(key) for key in dims)
        dim_full = dims[str(full_dim)]
        dim2 = dims.get('2', {})
        dim1 = dims.get('1', {})
        lines.append(
            f"| `{run_name}` | {0.0 if dim_full.get('projected_modular_distance') is None else dim_full.get('projected_modular_distance'):.4g} | "
            f"{dim2.get('trustworthiness', float('nan')):.4f} | "
            f"{dim2.get('knn_jaccard_mean', float('nan')):.4f} | "
            f"{dim2.get('effective_dimension', float('nan')):.4f} | "
            f"{dim2.get('h1_total_persistence', float('nan')):.4f} | "
            f"{0.0 if dim1.get('projected_modular_distance') is None else dim1.get('projected_modular_distance'):.4g} | "
            f"{0.0 if dim1.get('max_abs_logabsj_spearman') is None else dim1.get('max_abs_logabsj_spearman'):.4f} |"
        )

    lines.extend([
        '',
        '## Branch Outcome',
        '',
        f"- Branch: `{branch_assessment['branch']}`",
        f"- Summary: {branch_assessment['summary']}",
        f"- Next step: {branch_assessment['recommended_next_step']}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run_all(
    base_dir: str = 'runs',
    diagnostics_dir: str = 'runs/topology_diagnostics',
    report_path: str = 'walkthrough-topology-phaseA.md',
    experiments: list[dict] | None = None,
) -> dict:
    """Run the topology diagnostics phase."""
    if not tda_dependencies_available():
        raise RuntimeError(
            'Topology diagnostics require optional TDA dependencies. '
            'Install with: pip install ".[tda]" or pip install ripser persim'
        )

    os.makedirs(diagnostics_dir, exist_ok=True)
    experiments = experiments or DEFAULT_EXPERIMENTS

    topology_runs = {}
    for spec in experiments:
        print(f"\n{'=' * 60}")
        print(f"  Topology Diagnostics: {spec['name']}")
        print(f"{'=' * 60}\n")
        run_summary = _run_topology_for_experiment(spec, base_dir, diagnostics_dir)
        topology_runs[spec['name']] = run_summary

    branch_assessment = classify_branch(topology_runs)
    combined = {
        'topology_diagnostics': topology_runs,
        'branch_assessment': branch_assessment,
    }

    summary_path = os.path.join(diagnostics_dir, 'topology_diagnostics_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(combined, f, indent=2)

    write_report(topology_runs, branch_assessment, output_path=report_path)

    print(f"\n{'=' * 60}")
    print('  Topology diagnostics complete!')
    print(f'  Combined summary: {summary_path}')
    print(f'  Report: {report_path}')
    print(f"{'=' * 60}")

    print(
        f"\n{'Run':<36} {'Kind':<10} {'k=2 trust':>10} "
        f"{'k=2 overlap':>12} {'k=2 H1':>10} {'Branch':>8}"
    )
    print('-' * 92)
    for name, summary in topology_runs.items():
        dim2 = _dim_metrics(summary, 2)
        print(
            f"{name:<36} {summary['kind']:<10} "
            f"{dim2.get('trustworthiness', float('nan')):>10.4f} "
            f"{dim2.get('knn_jaccard_mean', float('nan')):>12.4f} "
            f"{dim2.get('h1_total_persistence', float('nan')):>10.4f} "
            f"{branch_assessment['branch']:>8}"
        )

    return combined


if __name__ == '__main__':
    run_all()
