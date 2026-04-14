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
from sklearn.decomposition import PCA

from data.dataset import Dataset
from data.generation import generate_t1_signals, generate_t2_signals
from eval.metrics import (
    encode_dataset, compute_j_correlation,
)


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

    # Decode interpolated latents via _decode_latents().
    # _decode_latents() is responsible for constructing the model with
    # models.create_model(config) and calling model.decode.
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

    This helper builds the model via models.create_model(config) and
    runs model.decode through flax's method dispatch.
    """
    from models import create_model

    model = create_model(config)

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


# ---------------------------------------------------------------------------
# Lattice / Modular visualization
# ---------------------------------------------------------------------------

def _draw_fundamental_domain(ax, color='black', linewidth=1.5, alpha=0.5):
    """Draw the fundamental domain boundary on an axes."""
    # Left edge: Re(τ) = -0.5
    ax.axvline(-0.5, color=color, linewidth=linewidth, alpha=alpha,
               linestyle='--')
    # Right edge: Re(τ) = 0.5
    ax.axvline(0.5, color=color, linewidth=linewidth, alpha=alpha,
               linestyle='--')
    # Bottom arc: |τ| = 1
    theta_arc = np.linspace(-np.pi / 3, -2 * np.pi / 3, 100)
    arc_x = np.cos(theta_arc)
    arc_y = np.sin(theta_arc)
    ax.plot(arc_x, arc_y, color=color, linewidth=linewidth, alpha=alpha,
            linestyle='--')


def plot_lattice_latent_scatter(
    state: TrainState,
    dataset: Dataset,
    config,
    is_vae: bool = False,
    save_path: str | None = None,
) -> plt.Figure:
    """Scatter plot of latent codes for lattice experiments.

    Creates two panels:
    1. Latent space colored by Re(τ) and Im(τ)
    2. Fundamental domain view with τ coordinates

    Args:
        state: Trained model state.
        dataset: Dataset with ground-truth τ parameters.
        config: Experiment configuration.
        is_vae: Whether the model is a VAE.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    z = np.array(encode_dataset(state, dataset, is_vae=is_vae))
    thetas_np = np.array(dataset.thetas)  # (N, 2): [Re(τ), Im(τ)]
    latent_type = config.model.latent_type

    tau_real = thetas_np[:, 0]
    tau_imag = thetas_np[:, 1]
    axis_labels = ('z_0', 'z_1')

    if z.shape[1] > 2:
        pca = PCA(n_components=2)
        z_plot = pca.fit_transform(z)
        axis_labels = ('PC1', 'PC2')
    else:
        z_plot = z

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Latent scatter colored by Re(τ)
    ax = axes[0, 0]
    sc = ax.scatter(z_plot[:, 0], z_plot[:, 1], c=tau_real, cmap='coolwarm',
                    s=8, alpha=0.7)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title('Latent space (color=Re(τ))')
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='Re(τ)')

    # Panel 2: Latent scatter colored by Im(τ)
    ax = axes[0, 1]
    sc = ax.scatter(z_plot[:, 0], z_plot[:, 1], c=tau_imag, cmap='viridis',
                    s=8, alpha=0.7)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title('Latent space (color=Im(τ))')
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='Im(τ)')

    # Panel 3: True τ in fundamental domain
    ax = axes[1, 0]
    sc = ax.scatter(tau_real, tau_imag, c=tau_imag, cmap='viridis',
                    s=8, alpha=0.7)
    _draw_fundamental_domain(ax)
    ax.set_xlabel('Re(τ)')
    ax.set_ylabel('Im(τ)')
    ax.set_title('Ground truth τ (fundamental domain)')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(0.6, 3.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='Im(τ)')

    # Panel 4: Latent with fundamental domain boundary overlay
    # (only meaningful if latent ≈ fundamental domain coordinates)
    ax = axes[1, 1]
    sc = ax.scatter(z_plot[:, 0], z_plot[:, 1],
                    c=np.sqrt(tau_real**2 + tau_imag**2),
                    cmap='plasma', s=8, alpha=0.7)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title('Latent space (color=|τ|)')
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='|τ|')

    fig.suptitle(f'Lattice Latent Space ({latent_type})', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_j_invariant_correlation(
    z_latent: np.ndarray,
    j_values,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot correlation between learned latent coordinates and j-invariant.

    Args:
        z_latent: Latent representations, shape (N, 2).
        j_values: Complex j-invariant values, shape (N,).
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    z_np = np.array(z_latent)
    j_arr = np.array(j_values)
    logabs_j = np.log10(np.maximum(np.abs(j_arr), 1e-30))
    corr_metrics = compute_j_correlation(z_np, j_arr)

    n_rows = z_np.shape[1] + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.2 * n_rows))
    axes = np.atleast_2d(axes)

    for d in range(z_np.shape[1]):
        ax = axes[d, 0]
        pearson = corr_metrics.get(f'z{d}_vs_logabsj_pearson', 0.0)
        spearman = corr_metrics.get(f'z{d}_vs_logabsj_spearman', 0.0)
        mi = corr_metrics.get(f'z{d}_vs_logabsj_mutual_info', 0.0)
        corr_re = corr_metrics.get(f'z{d}_vs_Re_j', 0.0)
        corr_im = corr_metrics.get(f'z{d}_vs_Im_j', 0.0)
        ax.scatter(logabs_j, z_np[:, d], s=5, alpha=0.5)
        ax.set_xlabel('log10|j(τ)|')
        ax.set_ylabel(f'z_{d}')
        ax.set_title(
            f'z_{d} vs log10|j|, Pearson={pearson:.3f}, '
            f'Spearman={spearman:.3f}, MI={mi:.3f}'
        )
        ax.grid(True, alpha=0.2)

        ax = axes[d, 1]
        ax.axis('off')
        ax.text(
            0.0, 1.0,
            '\n'.join([
                f'z_{d} vs Re(j) Pearson: {corr_re:.3f}',
                f'z_{d} vs Im(j) Pearson: {corr_im:.3f}',
            ]),
            ha='left', va='top', transform=ax.transAxes,
        )

    summary_left = axes[-1, 0]
    summary_left.axis('off')
    summary_left.text(
        0.0, 1.0,
        '\n'.join([
            'Summary',
            f"max |corr| vs Re/Im(j): {corr_metrics.get('max_abs_correlation', 0.0):.3f}",
            f"max |corr| vs log10|j| (Pearson): "
            f"{corr_metrics.get('max_abs_logabsj_pearson', 0.0):.3f}",
            f"max |corr| vs log10|j| (Spearman): "
            f"{corr_metrics.get('max_abs_logabsj_spearman', 0.0):.3f}",
            f"max MI vs log10|j|: {corr_metrics.get('max_logabsj_mutual_info', 0.0):.3f}",
        ]),
        ha='left', va='top', transform=summary_left.transAxes,
    )

    summary_right = axes[-1, 1]
    summary_right.axis('off')
    summary_right.text(
        0.0, 1.0,
        '\n'.join([
            'Notes',
            '- Main comparison uses log10|j| to control cusp-driven scale.',
            '- Re(j), Im(j) Pearson values are retained as reference.',
        ]),
        ha='left', va='top', transform=summary_right.transAxes,
    )

    fig.suptitle('Latent vs j-invariant correlation', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_quotient_chart_quality(
    chart_summary: dict,
    chart_aux: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """Visualize quotient-chart quality diagnostics for lattice experiments."""
    tau_coords = np.asarray(chart_aux['tau_coords'], dtype=float)
    latent_coords = np.asarray(chart_aux['latent_coords'], dtype=float)
    local_overlap = np.asarray(chart_aux['local_knn_jaccard'], dtype=float)

    latent_labels = ('z_0', 'z_1')
    latent_plot = latent_coords
    latent_projection_note = 'raw latent'

    if latent_coords.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_plot = pca.fit_transform(latent_coords)
        latent_labels = ('PC1', 'PC2')
        latent_projection_note = 'PCA projection for plotting only'
    elif latent_coords.shape[1] == 1:
        latent_plot = np.column_stack([latent_coords[:, 0], np.zeros(latent_coords.shape[0])])
        latent_labels = ('z_0', '0')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: τ in the fundamental domain, colored by local overlap.
    ax = axes[0, 0]
    sc = ax.scatter(
        tau_coords[:, 0], tau_coords[:, 1],
        c=local_overlap, cmap='viridis', vmin=0.0, vmax=1.0,
        s=10, alpha=0.8,
    )
    _draw_fundamental_domain(ax)
    ax.set_xlabel('Re(τ_F)')
    ax.set_ylabel('Im(τ_F)')
    ax.set_title('Fundamental-domain chart (color=local kNN overlap)')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(0.6, max(3.2, float(np.max(tau_coords[:, 1]) + 0.1)))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='kNN Jaccard')

    # Panel 2: latent scatter with the same color signal.
    ax = axes[0, 1]
    sc = ax.scatter(
        latent_plot[:, 0], latent_plot[:, 1],
        c=local_overlap, cmap='viridis', vmin=0.0, vmax=1.0,
        s=10, alpha=0.8,
    )
    ax.set_xlabel(latent_labels[0])
    ax.set_ylabel(latent_labels[1])
    ax.set_title('Latent chart (same local overlap coloring)')
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='kNN Jaccard')

    # Panel 3: overlap distribution.
    ax = axes[1, 0]
    ax.hist(local_overlap, bins=20, range=(0.0, 1.0), color='tab:green', alpha=0.8)
    ax.axvline(chart_summary.get('knn_jaccard_mean', 0.0), color='black', linestyle='--')
    ax.set_xlabel('Local kNN Jaccard overlap')
    ax.set_ylabel('Count')
    ax.set_title('Local neighborhood overlap distribution')
    ax.grid(True, alpha=0.2)

    # Panel 4: text summary.
    ax = axes[1, 1]
    ax.axis('off')
    ax.text(
        0.0, 1.0,
        '\n'.join([
            'Chart Summary',
            f"tau geometry: {chart_summary.get('tau_geometry', 'unknown')}",
            f"latent metric: {chart_summary.get('latent_metric_space', 'unknown')}",
            f"n samples: {chart_summary.get('n_samples', 0)}",
            f"n neighbors: {chart_summary.get('n_neighbors', 0)}",
            '',
            f"trustworthiness: {chart_summary.get('trustworthiness', 0.0):.4f}",
            f"kNN overlap mean: {chart_summary.get('knn_jaccard_mean', 0.0):.4f}",
            f"kNN overlap std: {chart_summary.get('knn_jaccard_std', 0.0):.4f}",
            f"effective dim: {chart_summary.get('effective_dimension', 0.0):.4f}",
            '',
            f"PC1 EVR: {chart_summary.get('pc1_explained_variance', 0.0):.4f}",
            f"PC2 EVR: {chart_summary.get('pc2_explained_variance', 0.0):.4f}",
            f"PC2/PC1: {chart_summary.get('pc2_pc1_ratio', 0.0):.4f}",
            '',
            f"plot note: {latent_projection_note}",
        ]),
        ha='left', va='top', transform=ax.transAxes,
    )

    fig.suptitle('Quotient Chart Quality', fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
