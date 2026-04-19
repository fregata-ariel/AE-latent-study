"""Evaluation metrics for reconstruction quality and periodicity."""

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from data.dataset import Dataset, batched_iterator
from data.generation import modular_transform_ids
from models import create_model


def _resolve_latent_type(
    latent_type: str | None = None,
    is_vae: bool = False,
) -> str:
    """Resolve legacy `is_vae` flag and explicit latent-type string."""
    if latent_type is not None:
        return latent_type
    return 'vae' if is_vae else 'standard'


def _deterministic_forward(
    state: TrainState,
    batch: jnp.ndarray,
    key: jax.Array,
    latent_type: str,
    latent_view: str = 'primary',
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one deterministic forward pass and pick a latent view."""
    if latent_type == 'factorized_vae':
        x_hat, z, q_mean, _, g_mean, _ = state.apply_fn(
            {'params': state.params}, batch, key, deterministic=True,
        )
        if latent_view in ('primary', 'quotient'):
            latent = q_mean
        elif latent_view == 'gauge':
            latent = g_mean
        elif latent_view == 'full':
            latent = jnp.concatenate([q_mean, g_mean], axis=-1)
        else:
            raise ValueError(
                f"Unknown latent_view '{latent_view}'. "
                "Choose from 'primary', 'quotient', 'gauge', 'full'."
            )
        return x_hat, latent

    if latent_type == 'vae':
        x_hat, z, _, _ = state.apply_fn(
            {'params': state.params}, batch, key, deterministic=True,
        )
        return x_hat, z

    x_hat, z = state.apply_fn({'params': state.params}, batch)
    return x_hat, z


def _iterate_all_samples(
    dataset: Dataset,
    batch_size: int,
    key: jax.Array,
):
    """Yield dataset batches with full coverage and an empty-dataset guard."""
    if len(dataset) == 0:
        raise ValueError('Empty dataset for evaluation')

    for batch_signals, _ in batched_iterator(dataset, batch_size, key, shuffle=False):
        yield batch_signals

    n_covered = (len(dataset) // batch_size) * batch_size
    if n_covered < len(dataset):
        yield dataset.signals[n_covered:]


def compute_reconstruction_error(
    state: TrainState,
    dataset: Dataset,
    batch_size: int = 512,
    is_vae: bool = False,
    latent_type: str | None = None,
) -> dict[str, float]:
    """Compute reconstruction MSE and MAE on a full dataset.

    Args:
        state: Trained model state.
        dataset: Dataset to evaluate.
        batch_size: Batch size for evaluation.
        is_vae: Whether the model is a VAE.

    Returns:
        Dictionary with 'mse' and 'mae'.
    """
    key = jax.random.PRNGKey(0)  # deterministic for eval
    latent_type = _resolve_latent_type(latent_type, is_vae=is_vae)
    sum_sq_err = 0.0
    sum_abs_err = 0.0
    n_elems = 0
    n_samples = 0

    for batch_signals in _iterate_all_samples(dataset, batch_size, key):
        x_hat, _ = _deterministic_forward(
            state, batch_signals, key, latent_type=latent_type,
        )

        diff = batch_signals - x_hat
        sum_sq_err += float(jnp.sum(diff ** 2))
        sum_abs_err += float(jnp.sum(jnp.abs(diff)))
        n_elems += diff.size
        n_samples += batch_signals.shape[0]

    if n_elems == 0:
        raise ValueError('Empty dataset for reconstruction error')

    return {
        'mse': sum_sq_err / n_elems,
        'mae': sum_abs_err / n_elems,
        'n_samples': n_samples,
    }


def encode_dataset(
    state: TrainState,
    dataset: Dataset,
    batch_size: int = 512,
    is_vae: bool = False,
    latent_type: str | None = None,
    latent_view: str = 'primary',
) -> jnp.ndarray:
    """Encode an entire dataset to latent representations.

    Args:
        state: Trained model state.
        dataset: Dataset to encode.
        batch_size: Batch size for encoding.
        is_vae: Whether the model is a VAE.

    Returns:
        Latent codes, shape (N, latent_dim).
    """
    key = jax.random.PRNGKey(0)
    latent_type = _resolve_latent_type(latent_type, is_vae=is_vae)
    all_z = []

    for batch_signals in _iterate_all_samples(dataset, batch_size, key):
        _, z = _deterministic_forward(
            state,
            batch_signals,
            key,
            latent_type=latent_type,
            latent_view=latent_view,
        )
        all_z.append(z)

    return jnp.concatenate(all_z, axis=0)


def encode_factorized_views(
    state: TrainState,
    dataset: Dataset,
    batch_size: int = 512,
) -> dict[str, jnp.ndarray]:
    """Encode a factorized VAE dataset into full / quotient / gauge views."""
    return {
        'full': encode_dataset(
            state, dataset, batch_size=batch_size,
            latent_type='factorized_vae', latent_view='full',
        ),
        'quotient': encode_dataset(
            state, dataset, batch_size=batch_size,
            latent_type='factorized_vae', latent_view='quotient',
        ),
        'gauge': encode_dataset(
            state, dataset, batch_size=batch_size,
            latent_type='factorized_vae', latent_view='gauge',
        ),
    }


def check_periodicity(
    state: TrainState,
    config,
    n_points: int = 100,
    is_vae: bool = False,
    latent_type: str | None = None,
) -> dict[str, float]:
    """Check if theta=0 and theta~=2*pi map to similar latent representations.

    Generates signals at evenly spaced angles near the boundary and computes
    the maximum latent distance between theta~0 and theta~2*pi.

    Args:
        state: Trained model state.
        config: Experiment configuration.
        n_points: Number of test points.
        is_vae: Whether the model is a VAE.

    Returns:
        Dictionary with 'latent_distance' and 'reconstruction_distance'.
    """
    from data.generation import generate_t1_signals

    key = jax.random.PRNGKey(0)
    eps = 1e-4

    # Signals at theta near 0 and theta near 2*pi
    theta_start = jnp.array([0.0])
    theta_end = jnp.array([2.0 * jnp.pi - eps])

    sig_start = generate_t1_signals(
        theta_start, config.data.omega, config.data.signal_length, config.data.dt,
    )
    sig_end = generate_t1_signals(
        theta_end, config.data.omega, config.data.signal_length, config.data.dt,
    )

    latent_type = _resolve_latent_type(latent_type, is_vae=is_vae)
    recon_start, z_start = _deterministic_forward(
        state, sig_start, key, latent_type=latent_type,
    )
    recon_end, z_end = _deterministic_forward(
        state, sig_end, key, latent_type=latent_type,
    )

    latent_dist = float(jnp.sqrt(jnp.sum((z_start - z_end) ** 2)))
    recon_dist = float(jnp.sqrt(jnp.mean((recon_start - recon_end) ** 2)))

    return {
        'latent_distance': latent_dist,
        'reconstruction_distance': recon_dist,
    }


# ---------------------------------------------------------------------------
# Lattice / Modular metrics
# ---------------------------------------------------------------------------

def check_modular_invariance(
    state: TrainState,
    config,
    n_pairs: int = 100,
    is_vae: bool = False,
    latent_type: str | None = None,
    latent_view: str = 'primary',
) -> dict[str, float]:
    """Check if SL₂(Z)-equivalent τ values map to similar latent representations.

    Generates τ from the fundamental domain, applies random SL₂(Z) elements
    (T and S generators), computes theta functions for both, encodes them,
    and measures the latent distance.

    Args:
        state: Trained model state.
        config: Experiment configuration.
        n_pairs: Number of test pairs.
        is_vae: Whether the model is a VAE.

    Returns:
        Dictionary with 'mean_latent_distance' and 'max_latent_distance'.
    """
    from data.generation import (
        sample_fundamental_domain, generate_lattice_theta,
        make_cyclic_modular_partners, normalize_lattice_signals,
    )

    key = jax.random.PRNGKey(42)
    K = getattr(config.data, 'lattice_K', 10)
    t_min = getattr(config.data, 'lattice_t_min', 0.5)
    t_max = getattr(config.data, 'lattice_t_max', 5.0)
    signal_length = config.data.signal_length
    y_max = getattr(config.data, 'lattice_y_max', 3.0)
    normalization = getattr(config.data, 'lattice_signal_normalization', 'none')

    # Sample τ from F
    tau_orig = sample_fundamental_domain(n_pairs, y_max=y_max, key=key)
    tau_transformed, _ = make_cyclic_modular_partners(tau_orig)

    # Generate signals for both
    sig_orig = generate_lattice_theta(
        tau_orig, signal_length, t_min=t_min, t_max=t_max, K=K,
    )
    sig_trans = generate_lattice_theta(
        tau_transformed, signal_length, t_min=t_min, t_max=t_max, K=K,
    )
    sig_orig = normalize_lattice_signals(sig_orig, method=normalization)
    sig_trans = normalize_lattice_signals(sig_trans, method=normalization)

    latent_type = _resolve_latent_type(latent_type, is_vae=is_vae)
    _, z_orig = _deterministic_forward(
        state, sig_orig, key, latent_type=latent_type, latent_view=latent_view,
    )
    _, z_trans = _deterministic_forward(
        state, sig_trans, key, latent_type=latent_type, latent_view=latent_view,
    )

    # Compute pairwise latent distances
    dists = jnp.sqrt(jnp.sum((z_orig - z_trans) ** 2, axis=-1))

    return {
        'mean_latent_distance': float(jnp.mean(dists)),
        'max_latent_distance': float(jnp.max(dists)),
    }


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return a finite Pearson correlation with a constant-input guard."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.size == 0 or y_arr.size == 0:
        return 0.0
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return 0.0

    corr = np.corrcoef(x_arr, y_arr)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Rank values with average tie handling."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    order = np.argsort(values, kind='mergesort')
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=float)

    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end

    return ranks


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return a finite Spearman rank correlation."""
    return _pearson_correlation(_rankdata(x), _rankdata(y))


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate mutual information for a single latent dimension vs a target."""
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.shape[0] == 0 or np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return 0.0

    mi = mutual_info_regression(x_arr, y_arr, random_state=42)
    return float(mi[0]) if len(mi) else 0.0


def _deterministic_subsample_indices(
    n_total: int,
    max_samples: int | None,
) -> np.ndarray:
    """Return deterministic sorted indices for evaluation subsampling."""
    if max_samples is None or max_samples <= 0 or n_total <= max_samples:
        return np.arange(n_total)

    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n_total, size=max_samples, replace=False))


def _neighbor_indices(points: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Return kNN indices excluding the point itself."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nbrs.fit(points)
    return nbrs.kneighbors(return_distance=False)[:, 1:]


def _compute_partner_preservation_metrics(
    latent_coords: np.ndarray,
    partner_latent: np.ndarray,
    n_neighbors: int,
) -> dict[str, float]:
    """Measure how well a latent point cloud keeps modular partners nearby."""
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

    k_neighbors = max(1, min(int(n_neighbors), n_samples - 1))
    partner_hits = closer_counts < k_neighbors
    return {
        'partner_rank_percentile_mean': float(np.mean(partner_rank_percentiles)),
        'partner_rank_percentile_std': float(np.std(partner_rank_percentiles)),
        'partner_knn_hit_rate': float(np.mean(partner_hits)),
    }


def _compute_quotient_chart_loss_numpy(
    tau_coords: np.ndarray,
    quotient_coords: np.ndarray,
    n_neighbors: int,
) -> float:
    """Match normalized local τ_F and quotient distances using a kNN graph."""
    tau = np.asarray(tau_coords, dtype=float)
    quotient = np.asarray(quotient_coords, dtype=float)
    n_samples = tau.shape[0]
    if n_samples <= 1:
        return 0.0

    k_use = min(max(1, int(n_neighbors)), max(1, n_samples - 1))
    tau_dists = np.linalg.norm(tau[:, None, :] - tau[None, :, :], axis=-1)
    quotient_dists = np.linalg.norm(
        quotient[:, None, :] - quotient[None, :, :], axis=-1,
    )
    tau_for_knn = tau_dists.copy()
    np.fill_diagonal(tau_for_knn, np.inf)
    knn_idx = np.argsort(tau_for_knn, axis=1)[:, :k_use]

    tau_knn = np.take_along_axis(tau_dists, knn_idx, axis=1)
    quotient_knn = np.take_along_axis(quotient_dists, knn_idx, axis=1)
    tau_scale = np.maximum(np.mean(tau_knn, axis=1, keepdims=True), 1e-6)
    quotient_scale = np.maximum(np.mean(quotient_knn, axis=1, keepdims=True), 1e-6)
    return float(np.mean(((tau_knn / tau_scale) - (quotient_knn / quotient_scale)) ** 2))


def _compute_quotient_variance_floor_metrics(
    quotient_coords: np.ndarray,
    target: float,
) -> tuple[float, np.ndarray]:
    """Return variance-floor loss and per-dimension quotient variances."""
    quotient = np.asarray(quotient_coords, dtype=float)
    if quotient.ndim != 2 or quotient.shape[0] == 0:
        variances = np.zeros(0, dtype=float)
        return 0.0, variances

    variances = np.var(quotient, axis=0)
    loss = float(np.mean(np.maximum(target - variances, 0.0) ** 2))
    return loss, variances


def _compute_covariance(points: np.ndarray) -> np.ndarray:
    """Return a stable sample covariance matrix."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] <= 1:
        return np.zeros((pts.shape[-1], pts.shape[-1]), dtype=float)

    centered = pts - np.mean(pts, axis=0, keepdims=True)
    denom = max(pts.shape[0] - 1, 1)
    return (centered.T @ centered) / denom


def _compute_quotient_spread_metrics(
    tau_coords: np.ndarray,
    quotient_coords: np.ndarray,
    min_eig_ratio_target: float,
    trace_cap_ratio: float,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Return rotation-aware quotient spread metrics."""
    tau_cov = _compute_covariance(tau_coords)
    quotient_cov = _compute_covariance(quotient_coords)
    tau_eigs = np.clip(np.linalg.eigvalsh(tau_cov), a_min=0.0, a_max=None)
    quotient_eigs = np.clip(np.linalg.eigvalsh(quotient_cov), a_min=0.0, a_max=None)

    tau_min = float(tau_eigs[0]) if tau_eigs.size else 0.0
    tau_trace = float(np.sum(tau_eigs))
    quotient_min = float(quotient_eigs[0]) if quotient_eigs.size else 0.0
    quotient_trace = float(np.sum(quotient_eigs))

    min_eig_floor = max(min_eig_ratio_target * tau_min - quotient_min, 0.0) ** 2
    trace_cap = max(quotient_trace - trace_cap_ratio * tau_trace, 0.0) ** 2
    loss = float(min_eig_floor + 0.1 * trace_cap)
    return loss, quotient_eigs, tau_eigs, tau_trace


def _local_knn_jaccard(
    tau_coords: np.ndarray,
    latent_coords: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-point Jaccard overlap between τ-space and latent-space kNN."""
    tau_knn = _neighbor_indices(tau_coords, n_neighbors)
    latent_knn = _neighbor_indices(latent_coords, n_neighbors)

    overlaps = np.zeros(tau_coords.shape[0], dtype=float)
    for idx in range(tau_coords.shape[0]):
        tau_set = set(tau_knn[idx].tolist())
        latent_set = set(latent_knn[idx].tolist())
        union = tau_set | latent_set
        overlaps[idx] = (
            len(tau_set & latent_set) / len(union)
            if union else 0.0
        )

    return overlaps, tau_knn, latent_knn


def _participation_ratio(latent_coords: np.ndarray) -> float:
    """Estimate effective latent dimension from covariance eigenvalues."""
    z = np.asarray(latent_coords, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2:
        return 0.0

    centered = z - np.mean(z, axis=0, keepdims=True)
    cov = np.atleast_2d(np.cov(centered, rowvar=False))
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

    denom = float(np.sum(eigvals ** 2))
    numer = float(np.sum(eigvals)) ** 2
    if denom <= 1e-30:
        return 0.0
    return numer / denom


def compute_quotient_chart_quality(
    z_latent: jnp.ndarray,
    tau_values,
    n_neighbors: int = 8,
    max_samples: int = 2000,
    return_aux: bool = False,
):
    """Evaluate whether latent space behaves like a 2D quotient chart.

    Args:
        z_latent: Latent representations, shape (N, d).
        tau_values: Complex τ values, shape (N,).
        n_neighbors: Neighborhood size for local chart metrics.
        max_samples: Deterministic subsample cap for expensive metrics.
        return_aux: Whether to return per-point data for visualization.

    Returns:
        Summary dictionary, or `(summary, aux)` when `return_aux=True`.
    """
    from data.generation import reduce_to_fundamental_domain

    z_np = np.asarray(z_latent, dtype=float)
    tau_arr = np.asarray(tau_values, dtype=complex).reshape(-1)

    if z_np.ndim != 2:
        raise ValueError('z_latent must have shape (N, d)')
    if z_np.shape[0] != tau_arr.shape[0]:
        raise ValueError('z_latent and tau_values must have the same length')
    if n_neighbors < 1:
        raise ValueError('n_neighbors must be >= 1')

    sample_indices = _deterministic_subsample_indices(z_np.shape[0], max_samples)
    z_eval = z_np[sample_indices]
    tau_eval = tau_arr[sample_indices]
    tau_fd = np.asarray(reduce_to_fundamental_domain(tau_eval), dtype=complex).reshape(-1)
    tau_coords = np.stack([tau_fd.real, tau_fd.imag], axis=-1)

    n_eval = z_eval.shape[0]
    if n_eval < 3:
        summary = {
            'tau_geometry': 'euclidean_fd',
            'latent_metric_space': 'raw',
            'n_neighbors': 0,
            'n_samples': int(n_eval),
            'trustworthiness': 0.0,
            'knn_jaccard_mean': 0.0,
            'knn_jaccard_std': 0.0,
            'effective_dimension': 0.0,
            'pc1_explained_variance': 0.0,
            'pc2_explained_variance': 0.0,
            'pc2_pc1_ratio': 0.0,
        }
        aux = {
            'sample_indices': sample_indices,
            'tau_coords': tau_coords,
            'latent_coords': z_eval,
            'local_knn_jaccard': np.zeros(n_eval, dtype=float),
        }
        return (summary, aux) if return_aux else summary

    k_eval = min(n_neighbors, max(1, (n_eval - 1) // 2))
    local_overlap, _, _ = _local_knn_jaccard(tau_coords, z_eval, k_eval)
    trust = float(trustworthiness(tau_coords, z_eval, n_neighbors=k_eval))

    n_components = min(2, z_eval.shape[1], n_eval)
    pca = PCA(n_components=n_components)
    pca.fit(z_eval)
    explained = pca.explained_variance_ratio_
    pc1 = float(explained[0]) if explained.size >= 1 else 0.0
    pc2 = float(explained[1]) if explained.size >= 2 else 0.0

    summary = {
        'tau_geometry': 'euclidean_fd',
        'latent_metric_space': 'raw',
        'n_neighbors': int(k_eval),
        'n_samples': int(n_eval),
        'trustworthiness': trust,
        'knn_jaccard_mean': float(np.mean(local_overlap)),
        'knn_jaccard_std': float(np.std(local_overlap)),
        'effective_dimension': float(_participation_ratio(z_eval)),
        'pc1_explained_variance': pc1,
        'pc2_explained_variance': pc2,
        'pc2_pc1_ratio': float(pc2 / pc1) if pc1 > 1e-12 else 0.0,
    }

    if not return_aux:
        return summary

    aux = {
        'sample_indices': sample_indices,
        'tau_coords': tau_coords,
        'latent_coords': z_eval,
        'local_knn_jaccard': local_overlap,
    }
    return summary, aux


def compute_j_correlation(
    z_latent: jnp.ndarray,
    j_values,
) -> dict[str, float]:
    """Compute correlation between learned latent coordinates and j-invariant.

    Args:
        z_latent: Latent representations, shape (N, 2).
        j_values: Complex j-invariant values, shape (N,).

    Returns:
        Dictionary with correlation coefficients for each latent dim vs
        Re(j) and Im(j).
    """
    z_np = np.array(z_latent)
    j_arr = np.array(j_values)
    j_real = j_arr.real
    j_imag = j_arr.imag
    logabs_j = np.log10(np.maximum(np.abs(j_arr), 1e-30))

    results = {}
    for d in range(z_np.shape[1]):
        results[f'z{d}_vs_Re_j'] = _pearson_correlation(z_np[:, d], j_real)
        results[f'z{d}_vs_Im_j'] = _pearson_correlation(z_np[:, d], j_imag)
        results[f'z{d}_vs_logabsj_pearson'] = _pearson_correlation(z_np[:, d], logabs_j)
        results[f'z{d}_vs_logabsj_spearman'] = _spearman_correlation(z_np[:, d], logabs_j)
        results[f'z{d}_vs_logabsj_mutual_info'] = _mutual_information(z_np[:, d], logabs_j)

    # Best absolute correlation across any combination
    abs_corrs = [
        abs(value) for key, value in results.items()
        if key.endswith('_vs_Re_j') or key.endswith('_vs_Im_j')
    ]
    logabs_pearsons = [
        abs(value) for key, value in results.items()
        if key.endswith('_vs_logabsj_pearson')
    ]
    logabs_spearmans = [
        abs(value) for key, value in results.items()
        if key.endswith('_vs_logabsj_spearman')
    ]
    logabs_mis = [
        value for key, value in results.items()
        if key.endswith('_vs_logabsj_mutual_info')
    ]

    results['max_abs_correlation'] = max(abs_corrs) if abs_corrs else 0.0
    results['max_abs_logabsj_pearson'] = max(logabs_pearsons) if logabs_pearsons else 0.0
    results['max_abs_logabsj_spearman'] = max(logabs_spearmans) if logabs_spearmans else 0.0
    results['max_logabsj_mutual_info'] = max(logabs_mis) if logabs_mis else 0.0

    return results


def compute_factorized_consistency(
    state: TrainState,
    dataset: Dataset,
    config,
    batch_size: int = 512,
) -> dict[str, float]:
    """Compute quotient/gauge consistency metrics for factorized lattice VAEs."""
    from data.generation import (
        generate_lattice_theta,
        make_cyclic_modular_partners,
        normalize_lattice_signals,
        reduce_to_fundamental_domain,
    )

    if config.model.latent_type != 'factorized_vae':
        raise ValueError('compute_factorized_consistency requires latent_type="factorized_vae".')

    tau_values = dataset.tau
    if tau_values is None:
        tau_values = np.asarray(dataset.thetas[:, 0]) + 1j * np.asarray(dataset.thetas[:, 1])

    tau_partner, transform_names = make_cyclic_modular_partners(tau_values)
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
        j_invariant=None,
        tau=tau_partner,
    )

    views = encode_factorized_views(state, dataset, batch_size=batch_size)
    partner_views = encode_factorized_views(state, partner_dataset, batch_size=batch_size)

    model = create_model(config)
    transform_ids = jnp.asarray(modular_transform_ids(transform_names), dtype=jnp.int32)
    acted_gauge = model.apply(
        {'params': state.params},
        jnp.asarray(views['gauge']),
        transform_ids,
        method=model.apply_gauge_action,
    )
    partner_recon = model.apply(
        {'params': state.params},
        jnp.asarray(views['quotient']),
        acted_gauge,
        method=model.decode_parts,
    )
    action_reg = model.apply(
        {'params': state.params},
        method=model.action_regularizer,
    )

    quotient = np.asarray(views['quotient'], dtype=float)
    quotient_partner = np.asarray(partner_views['quotient'], dtype=float)
    gauge_partner = np.asarray(partner_views['gauge'], dtype=float)
    partner_preservation = _compute_partner_preservation_metrics(
        quotient,
        quotient_partner,
        n_neighbors=getattr(config.eval, 'chart_n_neighbors', 8),
    )
    tau_fd = np.asarray(reduce_to_fundamental_domain(tau_values), dtype=complex).reshape(-1)
    tau_coords = np.stack([tau_fd.real, tau_fd.imag], axis=-1)
    quotient_chart_loss = _compute_quotient_chart_loss_numpy(
        tau_coords,
        quotient,
        n_neighbors=getattr(config.train, 'chart_preserving_n_neighbors', 8),
    )
    quotient_variance_floor_loss, quotient_variances = _compute_quotient_variance_floor_metrics(
        quotient,
        target=getattr(config.train, 'quotient_variance_floor_target', 0.15),
    )
    quotient_spread_loss, quotient_cov_eigs, tau_cov_eigs, tau_cov_trace = (
        _compute_quotient_spread_metrics(
            tau_coords,
            quotient,
            min_eig_ratio_target=getattr(
                config.train, 'quotient_min_eig_ratio_target', 0.20,
            ),
            trace_cap_ratio=getattr(config.train, 'quotient_trace_cap_ratio', 1.50),
        )
    )

    return {
        'quotient_pair_distance_mean': float(np.mean(np.linalg.norm(
            quotient - quotient_partner, axis=-1,
        ))),
        'quotient_partner_rank_percentile_mean': partner_preservation['partner_rank_percentile_mean'],
        'quotient_partner_rank_percentile_std': partner_preservation['partner_rank_percentile_std'],
        'quotient_partner_knn_hit_rate': partner_preservation['partner_knn_hit_rate'],
        'gauge_equivariance_mse': float(np.mean(np.sum(
            (np.asarray(acted_gauge) - gauge_partner) ** 2, axis=-1,
        ))),
        'decoder_equivariance_mse': float(np.mean(
            (np.asarray(partner_recon) - np.asarray(partner_signals)) ** 2,
        )),
        'gauge_action_reg': float(action_reg),
        'quotient_chart_loss': quotient_chart_loss,
        'quotient_variance_floor_loss': quotient_variance_floor_loss,
        'quotient_var_dim0': float(quotient_variances[0]) if quotient_variances.size >= 1 else 0.0,
        'quotient_var_dim1': float(quotient_variances[1]) if quotient_variances.size >= 2 else 0.0,
        'quotient_spread_loss': quotient_spread_loss,
        'quotient_cov_eig_min': float(quotient_cov_eigs[0]) if quotient_cov_eigs.size >= 1 else 0.0,
        'quotient_cov_eig_max': float(quotient_cov_eigs[-1]) if quotient_cov_eigs.size >= 1 else 0.0,
        'tau_cov_eig_min': float(tau_cov_eigs[0]) if tau_cov_eigs.size >= 1 else 0.0,
        'tau_cov_trace': float(tau_cov_trace),
    }
