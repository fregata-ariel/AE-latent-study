"""Visualization functions for latent space analysis."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from flax.training.train_state import TrainState

from data.dataset import Dataset
from data.generation import generate_t1_signals, generate_t2_signals
from eval.metrics import encode_dataset


def _angle_colormap(thetas: np.ndarray) -> np.ndarray:
    """Map angles in [0, 2*pi) to HSV-based colors."""
    hue = (thetas % (2 * np.pi)) / (2 * np.pi)
    hsv = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
    return hsv_to_rgb(hsv)


def plot_training_curves(
    history: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], label='Train Loss')
    ax.plot(epochs, history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_reconstructions(
    state: TrainState,
    dataset: Dataset,
    n_examples: int = 8,
    is_vae: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot input signals alongside their reconstructions.

    Args:
        state: Trained model state.
        dataset: Dataset to draw examples from.
        n_examples: Number of examples to plot.
        is_vae: Whether the model is a VAE.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    key = jax.random.PRNGKey(0)
    signals = dataset.signals[:n_examples]

    if is_vae:
        x_hat, _, _, _ = state.apply_fn(
            {'params': state.params}, signals, key, deterministic=True,
        )
    else:
        x_hat, _ = state.apply_fn({'params': state.params}, signals)

    signals_np = np.array(signals)
    x_hat_np = np.array(x_hat)

    n_cols = min(4, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for i in range(n_examples):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        ax.plot(signals_np[i], label='Input', alpha=0.8)
        ax.plot(x_hat_np[i], '--', label='Recon', alpha=0.8)
        ax.set_title(f'Sample {i}')
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for i in range(n_examples, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('Reconstruction Examples', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_latent_scatter(
    state: TrainState,
    dataset: Dataset,
    config,
    is_vae: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """Scatter plot of latent codes colored by true theta.

    Args:
        state: Trained model state.
        dataset: Dataset with ground-truth thetas.
        config: Experiment configuration.
        is_vae: Whether the model is a VAE.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    z = np.array(encode_dataset(state, dataset, is_vae=is_vae))
    thetas_np = np.array(dataset.thetas)
    torus_dim = config.data.torus_dim
    latent_type = config.model.latent_type
    latent_dim = z.shape[-1]

    if torus_dim == 1:
        colors = _angle_colormap(thetas_np)

        if latent_dim == 2:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.scatter(z[:, 0], z[:, 1], c=colors, s=10, alpha=0.7)
            ax.set_xlabel('z_0')
            ax.set_ylabel('z_1')
            ax.set_title(f'Latent Space (T^1, {latent_type})')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
        else:
            # For >2D, show first two components
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.scatter(z[:, 0], z[:, 1], c=colors, s=10, alpha=0.7)
            ax.set_xlabel('z_0')
            ax.set_ylabel('z_1')
            ax.set_title(f'Latent (first 2 dims, {latent_type})')
            ax.grid(True, alpha=0.2)

    elif torus_dim == 2:
        colors1 = _angle_colormap(thetas_np[:, 0])
        colors2 = _angle_colormap(thetas_np[:, 1])

        n_plots = min(latent_dim, 4)
        fig, axes = plt.subplots(2, n_plots, figsize=(5 * n_plots, 10))

        for j in range(n_plots):
            for k in range(j + 1, min(j + 2, latent_dim)):
                # Color by theta1
                axes[0, j].scatter(z[:, j], z[:, k], c=colors1, s=5, alpha=0.5)
                axes[0, j].set_xlabel(f'z_{j}')
                axes[0, j].set_ylabel(f'z_{k}')
                axes[0, j].set_title(f'color=theta1')
                axes[0, j].grid(True, alpha=0.2)

                # Color by theta2
                axes[1, j].scatter(z[:, j], z[:, k], c=colors2, s=5, alpha=0.5)
                axes[1, j].set_xlabel(f'z_{j}')
                axes[1, j].set_ylabel(f'z_{k}')
                axes[1, j].set_title(f'color=theta2')
                axes[1, j].grid(True, alpha=0.2)

        fig.suptitle(f'Latent Space (T^2, {latent_type})', fontsize=14)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        ax.text(0.5, 0.5, f'Unsupported torus_dim={torus_dim}',
                ha='center', va='center', transform=ax.transAxes)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_latent_interpolation(
    state: TrainState,
    config,
    n_points: int = 50,
    is_vae: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """Interpolate in latent space and visualize decoded signals.

    For torus-aware models: sweep angle from 0 to 2*pi.
    For standard models: linear interpolation between two encoded points.

    Args:
        state: Trained model state.
        config: Experiment configuration.
        n_points: Number of interpolation points.
        is_vae: Whether the model is a VAE.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    from models.layers import TorusLatent

    key = jax.random.PRNGKey(0)
    latent_type = config.model.latent_type

    if latent_type == 'torus':
        # Sweep angle(s) from 0 to 2*pi
        angles = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
        n_angles = config.model.latent_dim

        if n_angles == 1:
            z_interp = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        else:
            # Fix theta2=0, sweep theta1
            cos1 = jnp.cos(angles)
            sin1 = jnp.sin(angles)
            cos2 = jnp.ones_like(angles)  # theta2 = 0
            sin2 = jnp.zeros_like(angles)
            z_interp = jnp.stack([cos1, cos2, sin1, sin2], axis=-1)
    else:
        # Linear interpolation between two endpoints
        theta_a = jnp.array([0.0])
        theta_b = jnp.array([jnp.pi])
        sig_a = generate_t1_signals(
            theta_a, config.data.omega, config.data.signal_length, config.data.dt,
        )
        sig_b = generate_t1_signals(
            theta_b, config.data.omega, config.data.signal_length, config.data.dt,
        )

        if is_vae:
            _, z_a, _, _ = state.apply_fn(
                {'params': state.params}, sig_a, key, deterministic=True,
            )
            _, z_b, _, _ = state.apply_fn(
                {'params': state.params}, sig_b, key, deterministic=True,
            )
        else:
            _, z_a = state.apply_fn({'params': state.params}, sig_a)
            _, z_b = state.apply_fn({'params': state.params}, sig_b)

        alphas = jnp.linspace(0, 1, n_points)
        z_interp = z_a + alphas[:, None] * (z_b - z_a)

    # Decode interpolated latents
    # Access the decoder through the model's decode method
    # We need to call the model's decoder directly
    from models.ae import AutoEncoder
    from models.torus_ae import TorusAutoEncoder
    from models.vae import VAE as VAEModel

    model = create_model_from_config(config)
    # Use apply_fn with method if available, otherwise reconstruct
    # For simplicity, feed latent through the decoder part
    # We achieve this by using a helper that calls decode
    decoded_signals = _decode_latents(state, z_interp, config)
    decoded_np = np.array(decoded_signals)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Heatmap of signals over interpolation
    im = ax1.imshow(
        decoded_np, aspect='auto', cmap='RdBu_r',
        extent=[0, decoded_np.shape[1], n_points, 0],
    )
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Interpolation index')
    ax1.set_title('Decoded signals along interpolation path')
    plt.colorbar(im, ax=ax1)

    # Overlay a few decoded signals
    for i in range(0, n_points, max(1, n_points // 8)):
        alpha = i / n_points
        ax2.plot(decoded_np[i], alpha=0.7, label=f'step {i}',
                 color=plt.cm.viridis(alpha))
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Selected decoded signals')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(f'Latent Interpolation ({latent_type})', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def _decode_latents(state, z, config):
    """Decode latent vectors using the model's decoder.

    This works by constructing a signal from the decoder sub-module.
    We use a trick: apply the full model is not possible for just decode,
    so we use flax's module access pattern.
    """
    from models import create_model
    import flax.linen as nn

    model = create_model(config)
    latent_type = config.model.latent_type

    # Use bound module approach
    if latent_type == 'vae':
        # For VAE, decoder is a sub-module
        @jax.jit
        def decode_fn(params, z):
            return model.apply({'params': params}, z, method=model.decode)
    else:
        @jax.jit
        def decode_fn(params, z):
            return model.apply({'params': params}, z, method=model.decode)

    return decode_fn(state.params, z)


def plot_periodicity_check(
    state: TrainState,
    config,
    n_points: int = 200,
    is_vae: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """Visualize latent continuity around the periodic boundary.

    Encodes signals at theta from 0 to 2*pi and checks that the latent
    path forms a closed loop.

    Args:
        state: Trained model state.
        config: Experiment configuration.
        n_points: Number of points around the circle.
        is_vae: Whether the model is a VAE.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    key = jax.random.PRNGKey(0)
    thetas = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=True)

    signals = generate_t1_signals(
        thetas, config.data.omega, config.data.signal_length, config.data.dt,
    )

    if is_vae:
        _, z, _, _ = state.apply_fn(
            {'params': state.params}, signals, key, deterministic=True,
        )
    else:
        _, z = state.apply_fn({'params': state.params}, signals)

    z_np = np.array(z)
    thetas_np = np.array(thetas)
    colors = _angle_colormap(thetas_np)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Latent trajectory
    ax = axes[0]
    if z_np.shape[1] >= 2:
        ax.scatter(z_np[:, 0], z_np[:, 1], c=colors, s=15, alpha=0.8)
        ax.plot(z_np[:, 0], z_np[:, 1], 'k-', alpha=0.2, linewidth=0.5)
        # Mark start and end
        ax.scatter([z_np[0, 0]], [z_np[0, 1]], c='black', s=100,
                   marker='*', zorder=5, label='theta=0')
        ax.scatter([z_np[-1, 0]], [z_np[-1, 1]], c='red', s=100,
                   marker='X', zorder=5, label='theta=2pi')
    ax.set_xlabel('z_0')
    ax.set_ylabel('z_1')
    ax.set_title('Latent trajectory (theta: 0 -> 2pi)')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Latent components vs theta
    ax = axes[1]
    for d in range(min(z_np.shape[1], 4)):
        ax.plot(thetas_np, z_np[:, d], label=f'z_{d}')
    ax.set_xlabel('theta')
    ax.set_ylabel('Latent value')
    ax.set_title('Latent components vs theta')
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.suptitle('Periodicity Check', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
