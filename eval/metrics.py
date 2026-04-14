"""Evaluation metrics for reconstruction quality and periodicity."""

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from data.dataset import Dataset, batched_iterator


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
    all_mse = []
    all_mae = []

    for batch_signals, _ in batched_iterator(dataset, batch_size, key, shuffle=False):
        if is_vae:
            x_hat, _, _, _ = state.apply_fn(
                {'params': state.params}, batch_signals, key, deterministic=True,
            )
        else:
            x_hat, _ = state.apply_fn({'params': state.params}, batch_signals)

        diff = batch_signals - x_hat
        all_mse.append(float(jnp.mean(diff ** 2)))
        all_mae.append(float(jnp.mean(jnp.abs(diff))))

    return {
        'mse': float(np.mean(all_mse)),
        'mae': float(np.mean(all_mae)),
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

    for batch_signals, _ in batched_iterator(dataset, batch_size, key, shuffle=False):
        if is_vae:
            _, z, _, _ = state.apply_fn(
                {'params': state.params}, batch_signals, key, deterministic=True,
            )
        else:
            _, z = state.apply_fn({'params': state.params}, batch_signals)
        all_z.append(z)

    # Handle remainder samples not covered by batched_iterator
    n_covered = (len(dataset) // batch_size) * batch_size
    if n_covered < len(dataset):
        remainder = dataset.signals[n_covered:]
        if is_vae:
            _, z, _, _ = state.apply_fn(
                {'params': state.params}, remainder, key, deterministic=True,
            )
        else:
            _, z = state.apply_fn({'params': state.params}, remainder)
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
