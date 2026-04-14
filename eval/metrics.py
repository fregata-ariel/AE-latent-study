"""Evaluation metrics for reconstruction quality and periodicity."""

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from sklearn.feature_selection import mutual_info_regression

from data.dataset import Dataset, batched_iterator


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
    sum_sq_err = 0.0
    sum_abs_err = 0.0
    n_elems = 0
    n_samples = 0

    for batch_signals in _iterate_all_samples(dataset, batch_size, key):
        if is_vae:
            x_hat, _, _, _ = state.apply_fn(
                {'params': state.params}, batch_signals, key, deterministic=True,
            )
        else:
            x_hat, _ = state.apply_fn({'params': state.params}, batch_signals)

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
    all_z = []

    for batch_signals in _iterate_all_samples(dataset, batch_size, key):
        if is_vae:
            _, z, _, _ = state.apply_fn(
                {'params': state.params}, batch_signals, key, deterministic=True,
            )
        else:
            _, z = state.apply_fn({'params': state.params}, batch_signals)
        all_z.append(z)

    return jnp.concatenate(all_z, axis=0)


def check_periodicity(
    state: TrainState,
    config,
    n_points: int = 100,
    is_vae: bool = False,
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

    if is_vae:
        _, z_start, _, _ = state.apply_fn(
            {'params': state.params}, sig_start, key, deterministic=True,
        )
        _, z_end, _, _ = state.apply_fn(
            {'params': state.params}, sig_end, key, deterministic=True,
        )
        recon_start, _, _, _ = state.apply_fn(
            {'params': state.params}, sig_start, key, deterministic=True,
        )
        recon_end, _, _, _ = state.apply_fn(
            {'params': state.params}, sig_end, key, deterministic=True,
        )
    else:
        _, z_start = state.apply_fn({'params': state.params}, sig_start)
        _, z_end = state.apply_fn({'params': state.params}, sig_end)
        recon_start, _ = state.apply_fn({'params': state.params}, sig_start)
        recon_end, _ = state.apply_fn({'params': state.params}, sig_end)

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

    # Encode both
    if is_vae:
        _, z_orig, _, _ = state.apply_fn(
            {'params': state.params}, sig_orig, key, deterministic=True,
        )
        _, z_trans, _, _ = state.apply_fn(
            {'params': state.params}, sig_trans, key, deterministic=True,
        )
    else:
        _, z_orig = state.apply_fn({'params': state.params}, sig_orig)
        _, z_trans = state.apply_fn({'params': state.params}, sig_trans)

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
