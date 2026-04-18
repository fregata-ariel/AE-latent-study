"""Topology diagnostics for latent-space projection ladders."""

from __future__ import annotations

import importlib
from typing import Any

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from data.dataset import Dataset
from data.generation import (
    generate_lattice_theta,
    make_cyclic_modular_partners,
    normalize_lattice_signals,
    reduce_to_fundamental_domain,
)
from eval.metrics import compute_j_correlation, encode_dataset


def tda_dependencies_available() -> bool:
    """Whether optional TDA dependencies are importable."""
    return all(importlib.util.find_spec(name) is not None for name in ('ripser', 'persim'))


def _require_tda_dependencies():
    """Import optional TDA dependencies or raise a clear error."""
    if not tda_dependencies_available():
        raise RuntimeError(
            'Persistent-homology diagnostics require optional dependencies. '
            'Install with: pip install ".[tda]" or pip install ripser persim'
        )

    ripser_mod = importlib.import_module('ripser')
    persim_mod = importlib.import_module('persim')
    return ripser_mod, persim_mod


def _deterministic_subsample_indices(n_total: int, max_samples: int) -> np.ndarray:
    """Return sorted deterministic subsample indices."""
    if max_samples <= 0 or n_total <= max_samples:
        return np.arange(n_total)

    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n_total, size=max_samples, replace=False))


def _resolve_projection_dims(latent_dim: int, configured_dims) -> list[int]:
    """Resolve a descending projection ladder from config or defaults."""
    dims = [
        int(dim) for dim in configured_dims
        if int(dim) > 0 and int(dim) <= latent_dim
    ]
    if not dims:
        dims = list(range(latent_dim, 0, -1))

    dims = sorted(set(dims), reverse=True)
    return dims


def _standardize_latent(z_latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize latent coordinates with an epsilon floor."""
    z = np.asarray(z_latent, dtype=float)
    mean = np.mean(z, axis=0, keepdims=True)
    scale = np.std(z, axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-8)
    return (z - mean) / scale, mean, scale


def _torus_reference_coords(thetas: np.ndarray) -> np.ndarray:
    """Embed torus angles into a periodic Euclidean reference space."""
    thetas_np = np.asarray(thetas, dtype=float)
    if thetas_np.ndim == 1:
        return np.stack([np.cos(thetas_np), np.sin(thetas_np)], axis=-1)

    parts = []
    for axis in range(thetas_np.shape[1]):
        parts.append(np.cos(thetas_np[:, axis]))
        parts.append(np.sin(thetas_np[:, axis]))
    return np.stack(parts, axis=-1)


def _lattice_reference_coords(tau_values) -> np.ndarray:
    """Reduce lattice samples to the fundamental-domain Euclidean chart."""
    tau_arr = np.asarray(tau_values)
    tau_reduced = reduce_to_fundamental_domain(tau_arr)
    return np.stack([tau_reduced.real, tau_reduced.imag], axis=-1)


def make_reference_coords(dataset: Dataset, config) -> tuple[np.ndarray, str]:
    """Build the reference metric space for topology diagnostics."""
    data_type = getattr(config.data, 'data_type', 'torus')
    if data_type == 'lattice':
        tau = dataset.tau
        if tau is None:
            tau = np.asarray(dataset.thetas[:, 0]) + 1j * np.asarray(dataset.thetas[:, 1])
        return _lattice_reference_coords(tau), 'euclidean_fd'

    return _torus_reference_coords(np.asarray(dataset.thetas)), 'torus_cos_sin'


def _neighbor_indices(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Return kNN indices excluding the point itself."""
    points_arr = np.asarray(points, dtype=float)
    n_samples = points_arr.shape[0]
    if n_samples <= 1:
        return np.zeros((n_samples, 0), dtype=int)

    k = max(1, min(n_neighbors, n_samples - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(points_arr)
    return nbrs.kneighbors(return_distance=False)[:, 1:]


def _local_knn_jaccard(
    reference_coords: np.ndarray,
    latent_coords: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, float, float]:
    """Compute local Jaccard overlap between reference-space and latent kNNs."""
    ref_knn = _neighbor_indices(reference_coords, n_neighbors)
    lat_knn = _neighbor_indices(latent_coords, n_neighbors)
    overlaps = np.zeros(reference_coords.shape[0], dtype=float)

    for idx in range(reference_coords.shape[0]):
        ref_set = set(ref_knn[idx].tolist())
        lat_set = set(lat_knn[idx].tolist())
        union = ref_set | lat_set
        overlaps[idx] = len(ref_set & lat_set) / len(union) if union else 0.0

    return overlaps, float(np.mean(overlaps)), float(np.std(overlaps))


def _participation_ratio(latent_coords: np.ndarray) -> float:
    """Estimate effective dimension from covariance eigenvalues."""
    z = np.asarray(latent_coords, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2:
        return 0.0

    centered = z - np.mean(z, axis=0, keepdims=True)
    cov = np.atleast_2d(np.cov(centered, rowvar=False))
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    denom = float(np.sum(eigvals ** 2))
    if denom <= 1e-30:
        return 0.0
    numer = float(np.sum(eigvals)) ** 2
    return numer / denom


def compute_local_intrinsic_dimension(
    points: np.ndarray,
    n_neighbors: int = 10,
) -> dict[str, float]:
    """Estimate local intrinsic dimension via the Levina-Bickel estimator."""
    pts = np.asarray(points, dtype=float)
    n_samples = pts.shape[0]
    if n_samples <= 2:
        return {'median': 0.0, 'iqr': 0.0, 'mean': 0.0, 'valid_fraction': 0.0}

    k = max(3, min(n_neighbors, n_samples - 1))
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(pts)
    distances = nbrs.kneighbors(return_distance=True)[0][:, 1:]

    lid_values = []
    for row in distances:
        positive = np.asarray(row[row > 1e-8], dtype=float)
        if positive.size < 4:
            continue

        tail = positive[-1]
        body = positive[:-1]
        if body.size < 3:
            continue

        log_ratios = np.log(tail / body)
        denom = float(np.mean(log_ratios))
        if denom <= 1e-12 or not np.isfinite(denom):
            continue

        lid = 1.0 / denom
        if np.isfinite(lid):
            lid_values.append(lid)

    lid = np.asarray(lid_values, dtype=float)
    valid_fraction = float(lid.size / n_samples)
    if lid.size == 0:
        return {'median': 0.0, 'iqr': 0.0, 'mean': 0.0, 'valid_fraction': 0.0}

    q25, q75 = np.percentile(lid, [25, 75])
    return {
        'median': float(np.median(lid)),
        'iqr': float(q75 - q25),
        'mean': float(np.mean(lid)),
        'valid_fraction': valid_fraction,
    }


def _canonical_diagram(diagram: np.ndarray) -> np.ndarray:
    """Return a finite diagram array with shape (n_bars, 2)."""
    diag = np.asarray(diagram, dtype=float)
    if diag.size == 0:
        return np.zeros((0, 2), dtype=float)
    diag = np.atleast_2d(diag)
    keep = np.isfinite(diag[:, 0]) & np.isfinite(diag[:, 1])
    return diag[keep]


def _finite_bar_lengths(diagram: np.ndarray) -> np.ndarray:
    """Return finite bar lengths for a persistence diagram."""
    diag = _canonical_diagram(diagram)
    if diag.size == 0:
        return np.zeros(0, dtype=float)
    lengths = diag[:, 1] - diag[:, 0]
    return lengths[np.isfinite(lengths) & (lengths >= 0.0)]


def _compute_persistence_diagrams(points: np.ndarray, maxdim: int) -> list[np.ndarray]:
    """Compute persistence diagrams with ripser."""
    ripser_mod, _ = _require_tda_dependencies()
    result = ripser_mod.ripser(np.asarray(points, dtype=float), maxdim=maxdim)
    return [_canonical_diagram(diagram) for diagram in result['dgms']]


def _compute_diagram_distance_metrics(
    previous_diagrams: list[np.ndarray] | None,
    current_diagrams: list[np.ndarray],
) -> dict[str, float] | None:
    """Compute bottleneck / Wasserstein distances to the previous projection."""
    if previous_diagrams is None:
        return None

    _, persim_mod = _require_tda_dependencies()

    metrics = {}
    for dim in (0, 1):
        prev = _canonical_diagram(previous_diagrams[dim]) if dim < len(previous_diagrams) else np.zeros((0, 2))
        curr = _canonical_diagram(current_diagrams[dim]) if dim < len(current_diagrams) else np.zeros((0, 2))
        metrics[f'h{dim}_bottleneck'] = float(persim_mod.bottleneck(prev, curr))
        metrics[f'h{dim}_wasserstein'] = float(persim_mod.wasserstein(prev, curr))

    metrics['max_bottleneck'] = max(metrics['h0_bottleneck'], metrics['h1_bottleneck'])
    return metrics


def _compute_partner_preservation_metrics(
    latent_coords: np.ndarray,
    partner_latent: np.ndarray,
    n_neighbors: int,
) -> dict[str, float]:
    """Measure how well each point keeps its modular partner nearby."""
    z = np.asarray(latent_coords, dtype=float)
    partner = np.asarray(partner_latent, dtype=float)
    n_samples = z.shape[0]
    if n_samples <= 1:
        return {
            'partner_rank_percentile_mean': 1.0,
            'partner_rank_percentile_std': 0.0,
            'partner_knn_hit_rate': 0.0,
        }

    dists = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    partner_dists = np.linalg.norm(z - partner, axis=-1)

    closer_counts = np.sum(dists < partner_dists[:, None], axis=1)
    partner_rank_percentiles = closer_counts / max(n_samples - 1, 1)

    k_neighbors = max(1, min(n_neighbors, n_samples - 1))
    kth_neighbor = np.partition(dists, kth=k_neighbors - 1, axis=1)[:, k_neighbors - 1]
    partner_hits = partner_dists <= kth_neighbor

    return {
        'partner_rank_percentile_mean': float(np.mean(partner_rank_percentiles)),
        'partner_rank_percentile_std': float(np.std(partner_rank_percentiles)),
        'partner_knn_hit_rate': float(np.mean(partner_hits)),
    }


def _summarize_diagrams(
    diagrams: list[np.ndarray],
    noise_floor: float,
    noise_floor_mode: str = 'absolute',
) -> dict[str, float]:
    """Summarize H0/H1 persistence diagrams into scalar diagnostics."""
    summary = {}
    for dim in (0, 1):
        lengths = _finite_bar_lengths(diagrams[dim]) if dim < len(diagrams) else np.zeros(0)
        longest = float(np.max(lengths)) if lengths.size else 0.0
        if noise_floor_mode == 'relative':
            noise_floor_value = noise_floor * longest
        else:
            noise_floor_value = noise_floor
        summary[f'h{dim}_longest_bar'] = longest
        summary[f'noise_floor_value_h{dim}'] = float(noise_floor_value)
        summary[f'h{dim}_bar_count'] = int(np.sum(lengths > noise_floor_value))
        if dim == 1:
            summary['h1_total_persistence'] = float(np.sum(lengths))
    if 'h1_total_persistence' not in summary:
        summary['h1_total_persistence'] = 0.0
    return summary


def _random_orthonormal_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    """Draw a random orthonormal projection matrix."""
    rng = np.random.default_rng(seed)
    gaussian = rng.normal(size=(input_dim, output_dim))
    q, _ = np.linalg.qr(gaussian)
    return q[:, :output_dim]


def _evaluate_projection_metrics(
    projected_latent: np.ndarray,
    reference_coords: np.ndarray,
    n_neighbors: int,
    lid_neighbors: int,
    noise_floor: float,
    noise_floor_mode: str,
    maxdim: int,
    j_values=None,
    projected_partner_latent: np.ndarray | None = None,
    previous_diagrams: list[np.ndarray] | None = None,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray]:
    """Compute scalar topology diagnostics for one projected point cloud."""
    z_proj = np.asarray(projected_latent, dtype=float)
    ref = np.asarray(reference_coords, dtype=float)
    n_samples = z_proj.shape[0]
    k_neighbors = max(1, min(n_neighbors, n_samples - 1))

    metrics = {
        'effective_dimension': float(_participation_ratio(z_proj)),
    }
    lid_stats = compute_local_intrinsic_dimension(z_proj, n_neighbors=lid_neighbors)
    metrics['lid_median'] = lid_stats['median']
    metrics['lid_iqr'] = lid_stats['iqr']
    metrics['lid_mean'] = lid_stats['mean']
    metrics['lid_valid_fraction'] = lid_stats['valid_fraction']

    if n_samples > k_neighbors + 1:
        metrics['trustworthiness'] = float(
            trustworthiness(ref, z_proj, n_neighbors=k_neighbors),
        )
        local_overlap, overlap_mean, overlap_std = _local_knn_jaccard(
            ref, z_proj, k_neighbors,
        )
        metrics['knn_jaccard_mean'] = overlap_mean
        metrics['knn_jaccard_std'] = overlap_std
    else:
        local_overlap = np.zeros(n_samples, dtype=float)
        metrics['trustworthiness'] = 0.0
        metrics['knn_jaccard_mean'] = 0.0
        metrics['knn_jaccard_std'] = 0.0

    if j_values is not None:
        metrics['max_abs_logabsj_spearman'] = float(
            compute_j_correlation(z_proj, j_values).get('max_abs_logabsj_spearman', 0.0)
        )
    else:
        metrics['max_abs_logabsj_spearman'] = None

    if projected_partner_latent is not None:
        partner = np.asarray(projected_partner_latent, dtype=float)
        dists = np.sqrt(np.sum((z_proj - partner) ** 2, axis=-1))
        metrics['projected_modular_distance'] = float(np.mean(dists))
        metrics.update(_compute_partner_preservation_metrics(z_proj, partner, k_neighbors))
    else:
        metrics['projected_modular_distance'] = None
        metrics['partner_rank_percentile_mean'] = None
        metrics['partner_rank_percentile_std'] = None
        metrics['partner_knn_hit_rate'] = None

    diagrams = _compute_persistence_diagrams(z_proj, maxdim=maxdim)
    metrics.update(_summarize_diagrams(diagrams, noise_floor, noise_floor_mode=noise_floor_mode))
    metrics['diagram_distance_to_prev'] = _compute_diagram_distance_metrics(
        previous_diagrams, diagrams,
    )
    return metrics, diagrams, local_overlap


def _aggregate_random_metrics(random_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate scalar metrics from repeated random projections."""
    if not random_metrics:
        return {}

    aggregates = {}
    keys = [
        key for key, value in random_metrics[0].items()
        if isinstance(value, (int, float)) and value is not None
    ]
    for key in keys:
        values = np.asarray([metrics[key] for metrics in random_metrics], dtype=float)
        aggregates[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }
    return aggregates


def encode_lattice_partner_latent(
    state,
    dataset: Dataset,
    config,
    subset_indices: np.ndarray,
    is_vae: bool = False,
) -> np.ndarray:
    """Encode modularly transformed lattice partners for a fixed subset."""
    tau_values = dataset.tau
    if tau_values is None:
        tau_values = np.asarray(dataset.thetas[:, 0]) + 1j * np.asarray(dataset.thetas[:, 1])

    tau_subset = np.asarray(tau_values)[subset_indices]
    tau_partner, _ = make_cyclic_modular_partners(tau_subset)

    partner_signals = generate_lattice_theta(
        tau_partner,
        config.data.signal_length,
        t_min=getattr(config.data, 'lattice_t_min', 0.5),
        t_max=getattr(config.data, 'lattice_t_max', 5.0),
        K=getattr(config.data, 'lattice_K', 10),
    )
    partner_signals = normalize_lattice_signals(
        partner_signals,
        method=getattr(config.data, 'lattice_signal_normalization', 'none'),
    )

    partner_thetas = np.stack([tau_partner.real, tau_partner.imag], axis=-1)
    partner_dataset = Dataset(
        signals=jnp.asarray(partner_signals),
        thetas=jnp.asarray(partner_thetas),
        tau=tau_partner,
    )
    return np.asarray(encode_dataset(state, partner_dataset, is_vae=is_vae))


def diagnose_projection_ladder(
    z_latent: np.ndarray,
    reference_coords: np.ndarray,
    projection_dims,
    n_neighbors: int = 8,
    lid_neighbors: int = 10,
    maxdim: int = 1,
    max_samples: int = 2000,
    noise_floor: float = 0.05,
    noise_floor_mode: str = 'relative',
    random_projection_trials: int = 8,
    j_values=None,
    partner_latent: np.ndarray | None = None,
    reference_label: str = 'unknown',
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run topology diagnostics along a PCA projection ladder."""
    z_all = np.asarray(z_latent, dtype=float)
    ref_all = np.asarray(reference_coords, dtype=float)
    subset_idx = _deterministic_subsample_indices(z_all.shape[0], max_samples)

    z_subset = z_all[subset_idx]
    ref_subset = ref_all[subset_idx]
    j_subset = None if j_values is None else np.asarray(j_values)[subset_idx]
    partner_subset = None if partner_latent is None else np.asarray(partner_latent, dtype=float)[subset_idx]

    z_std, z_mean, z_scale = _standardize_latent(z_subset)
    partner_std = None if partner_subset is None else (partner_subset - z_mean) / z_scale

    projection_dims = _resolve_projection_dims(z_std.shape[1], projection_dims)
    pca = PCA(n_components=min(z_std.shape[1], z_std.shape[0]))
    pca.fit(z_std)
    pca_full = pca.transform(z_std)
    partner_pca_full = None if partner_std is None else pca.transform(partner_std)

    summary = {
        'projection_basis': 'pca',
        'reference_space': reference_label,
        'n_samples': int(z_std.shape[0]),
        'full_latent_dim': int(z_std.shape[1]),
        'dims': {},
    }
    artifacts = {
        'subsample_indices': subset_idx,
        'pca_diagrams': {},
        'local_overlap': {},
    }

    previous_diagrams = None
    for dim in projection_dims:
        z_proj = pca_full[:, :dim]
        partner_proj = None if partner_pca_full is None else partner_pca_full[:, :dim]
        metrics, diagrams, local_overlap = _evaluate_projection_metrics(
            z_proj,
            ref_subset,
            n_neighbors=n_neighbors,
            lid_neighbors=lid_neighbors,
            noise_floor=noise_floor,
            noise_floor_mode=noise_floor_mode,
            maxdim=maxdim,
            j_values=j_subset,
            projected_partner_latent=partner_proj,
            previous_diagrams=previous_diagrams,
        )

        random_metrics = []
        for trial in range(random_projection_trials):
            projection = _random_orthonormal_projection(
                z_std.shape[1], dim, seed=1000 * dim + trial,
            )
            z_rand = z_std @ projection
            partner_rand = None if partner_std is None else partner_std @ projection
            rand_summary, _, _ = _evaluate_projection_metrics(
                z_rand,
                ref_subset,
                n_neighbors=n_neighbors,
                lid_neighbors=lid_neighbors,
                noise_floor=noise_floor,
                noise_floor_mode=noise_floor_mode,
                maxdim=maxdim,
                j_values=j_subset,
                projected_partner_latent=partner_rand,
                previous_diagrams=None,
            )
            random_metrics.append(rand_summary)

        metrics['random_projection_baseline'] = _aggregate_random_metrics(random_metrics)
        summary['dims'][str(dim)] = metrics
        artifacts['pca_diagrams'][str(dim)] = diagrams
        artifacts['local_overlap'][str(dim)] = local_overlap
        previous_diagrams = diagrams

    return summary, artifacts


def plot_topology_metrics_vs_k(
    diagnostics_summary: dict[str, Any],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot key topology diagnostics across projection dimension."""
    dims = sorted((int(dim) for dim in diagnostics_summary['dims']), reverse=True)
    dim_keys = [str(dim) for dim in dims]

    trust = [diagnostics_summary['dims'][key]['trustworthiness'] for key in dim_keys]
    overlap = [diagnostics_summary['dims'][key]['knn_jaccard_mean'] for key in dim_keys]
    eff = [diagnostics_summary['dims'][key]['effective_dimension'] for key in dim_keys]
    lid = [diagnostics_summary['dims'][key]['lid_median'] for key in dim_keys]
    h1_total = [diagnostics_summary['dims'][key]['h1_total_persistence'] for key in dim_keys]
    h1_long = [diagnostics_summary['dims'][key]['h1_longest_bar'] for key in dim_keys]
    spearman = [diagnostics_summary['dims'][key]['max_abs_logabsj_spearman'] for key in dim_keys]
    partner_rank = [diagnostics_summary['dims'][key]['partner_rank_percentile_mean'] for key in dim_keys]
    partner_hit = [diagnostics_summary['dims'][key]['partner_knn_hit_rate'] for key in dim_keys]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(dims, trust, marker='o', label='trustworthiness')
    ax.plot(dims, overlap, marker='o', label='kNN overlap')
    ax.set_title('Neighborhood preservation')
    ax.set_xlabel('Projection dimension k')
    ax.grid(True, alpha=0.2)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(dims, eff, marker='o', label='effective dim')
    ax.plot(dims, lid, marker='o', label='local ID median')
    ax.set_title('Intrinsic dimension diagnostics')
    ax.set_xlabel('Projection dimension k')
    ax.grid(True, alpha=0.2)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(dims, h1_total, marker='o', label='H1 total persistence')
    ax.plot(dims, h1_long, marker='o', label='H1 longest bar')
    ax.set_title('Persistence summary')
    ax.set_xlabel('Projection dimension k')
    ax.grid(True, alpha=0.2)
    ax.legend()

    ax = axes[1, 1]
    if any(value is not None for value in spearman):
        ax.plot(dims, [0.0 if value is None else value for value in spearman],
                marker='o', label='max log|j| Spearman')
    if any(value is not None for value in partner_hit):
        ax.plot(dims, [0.0 if value is None else value for value in partner_hit],
                marker='o', label='partner kNN hit rate')
    if any(value is not None for value in partner_rank):
        ax.plot(dims, [0.0 if value is None else 1.0 - value for value in partner_rank],
                marker='o', label='1 - partner rank percentile')
    if not ax.lines:
        ax.plot(dims, eff, marker='o', label='effective dim')
    ax.set_title('Target observables')
    ax.set_xlabel('Projection dimension k')
    ax.grid(True, alpha=0.2)
    ax.legend()

    fig.suptitle('Topology diagnostics vs projection dimension', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_persistence_panels(
    diagnostics_summary: dict[str, Any],
    diagnostics_artifacts: dict[str, Any],
    save_path: str | None = None,
) -> plt.Figure:
    """Plot H0/H1 persistence diagrams for each projection dimension."""
    dims = sorted((int(dim) for dim in diagnostics_summary['dims']), reverse=True)
    fig, axes = plt.subplots(len(dims), 2, figsize=(10, 3.4 * len(dims)))
    axes = np.atleast_2d(axes)

    for row, dim in enumerate(dims):
        dim_key = str(dim)
        diagrams = diagnostics_artifacts['pca_diagrams'][dim_key]
        for col, hom_dim in enumerate((0, 1)):
            ax = axes[row, col]
            diagram = diagrams[hom_dim] if hom_dim < len(diagrams) else np.zeros((0, 2))
            if diagram.size:
                max_val = float(np.max(diagram)) if np.max(diagram) > 0 else 1.0
                ax.scatter(diagram[:, 0], diagram[:, 1], s=10, alpha=0.7)
                ax.plot([0.0, max_val], [0.0, max_val], 'k--', linewidth=0.8, alpha=0.5)
                ax.set_xlim(0.0, max_val * 1.05)
                ax.set_ylim(0.0, max_val * 1.05)
            else:
                ax.text(0.5, 0.5, 'no finite bars', ha='center', va='center',
                        transform=ax.transAxes)
            ax.set_xlabel('birth')
            ax.set_ylabel('death')
            ax.set_title(f'k={dim}, H{hom_dim}')
            ax.grid(True, alpha=0.2)

    fig.suptitle('Persistence diagrams along the PCA ladder', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_projection_comparison(
    diagnostics_summary: dict[str, Any],
    save_path: str | None = None,
) -> plt.Figure:
    """Compare PCA diagnostics against random-projection baselines."""
    dims = sorted((int(dim) for dim in diagnostics_summary['dims']), reverse=True)
    dim_keys = [str(dim) for dim in dims]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panel_specs = [
        ('trustworthiness', 'Trustworthiness'),
        ('knn_jaccard_mean', 'kNN overlap'),
        ('h1_total_persistence', 'H1 total persistence'),
        ('effective_dimension', 'Effective dimension'),
    ]

    if any(
        diagnostics_summary['dims'][key]['partner_knn_hit_rate'] is not None
        for key in dim_keys
    ):
        panel_specs[-1] = ('partner_knn_hit_rate', 'Partner kNN hit rate')
    elif any(
        diagnostics_summary['dims'][key]['max_abs_logabsj_spearman'] is not None
        for key in dim_keys
    ):
        panel_specs[-1] = ('max_abs_logabsj_spearman', 'max log|j| Spearman')

    for ax, (metric_key, title) in zip(axes.flat, panel_specs, strict=True):
        pca_values = [
            diagnostics_summary['dims'][key].get(metric_key, 0.0) or 0.0
            for key in dim_keys
        ]
        baseline_means = []
        baseline_stds = []
        for key in dim_keys:
            baseline = diagnostics_summary['dims'][key].get('random_projection_baseline', {})
            metric = baseline.get(metric_key, {'mean': np.nan, 'std': np.nan})
            baseline_means.append(metric['mean'])
            baseline_stds.append(metric['std'])

        ax.plot(dims, pca_values, marker='o', label='PCA')
        ax.fill_between(
            dims,
            np.asarray(baseline_means) - np.asarray(baseline_stds),
            np.asarray(baseline_means) + np.asarray(baseline_stds),
            alpha=0.25,
            label='random mean ± std',
        )
        ax.plot(dims, baseline_means, marker='o', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Projection dimension k')
        ax.grid(True, alpha=0.2)
        ax.legend()

    fig.suptitle('PCA vs random projection baselines', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
