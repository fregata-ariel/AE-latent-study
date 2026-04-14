"""Analysis utilities: PCA, UMAP, and orchestrated evaluation."""

import os
import json

import jax
import numpy as np
from sklearn.decomposition import PCA
from flax.training.train_state import TrainState

from data.dataset import Dataset
from eval.metrics import compute_reconstruction_error, encode_dataset, check_periodicity
from eval.visualization import (
    plot_training_curves,
    plot_reconstructions,
    plot_latent_scatter,
    plot_latent_interpolation,
    plot_periodicity_check,
)


def pca_latent(
    state: TrainState,
    dataset: Dataset,
    n_components: int = 2,
    is_vae: bool = False,
) -> tuple[np.ndarray, PCA]:
    """Apply PCA to latent representations.

    Args:
        state: Trained model state.
        dataset: Dataset to encode.
        n_components: Number of PCA components.
        is_vae: Whether the model is a VAE.

    Returns:
        Tuple of (projected_latents, fitted_pca_object).
    """
    z = np.array(encode_dataset(state, dataset, is_vae=is_vae))
    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(z)
    return z_pca, pca


def umap_latent(
    state: TrainState,
    dataset: Dataset,
    n_components: int = 2,
    is_vae: bool = False,
) -> np.ndarray:
    """Apply UMAP to latent representations.

    Requires umap-learn to be installed.

    Args:
        state: Trained model state.
        dataset: Dataset to encode.
        n_components: Number of UMAP dimensions.
        is_vae: Whether the model is a VAE.

    Returns:
        UMAP-projected latent representations.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    z = np.array(encode_dataset(state, dataset, is_vae=is_vae))
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(z)


def run_full_evaluation(
    state: TrainState,
    config,
    train_ds: Dataset,
    test_ds: Dataset,
    history: dict,
    workdir: str,
) -> dict:
    """Run all evaluation steps and save results.

    Args:
        state: Trained model state.
        config: Experiment configuration.
        train_ds: Training dataset (for latent visualization).
        test_ds: Test dataset (for reconstruction metrics).
        history: Training history dictionary.
        workdir: Directory to save results.

    Returns:
        Summary metrics dictionary.
    """
    results_dir = os.path.join(workdir, config.eval.output_dir)
    os.makedirs(results_dir, exist_ok=True)

    is_vae = config.model.latent_type == 'vae'
    summary = {}

    # 1. Reconstruction error
    print("Computing reconstruction error...")
    print(f"  Planned test samples (len(test_ds)): {len(test_ds)}")
    recon_batch_size = min(512, len(test_ds)) if len(test_ds) > 0 else 512
    recon_metrics = compute_reconstruction_error(
        state, test_ds, batch_size=recon_batch_size, is_vae=is_vae,
    )
    summary['reconstruction'] = recon_metrics
    print(f"  Evaluated test samples (actual): {recon_metrics['n_samples']}")
    print(f"  Test MSE: {recon_metrics['mse']:.6f}, MAE: {recon_metrics['mae']:.6f}")

    # 2. Periodicity check (T^1 only)
    if config.data.torus_dim == 1:
        print("Checking periodicity...")
        period_metrics = check_periodicity(state, config, is_vae=is_vae)
        summary['periodicity'] = period_metrics
        print(f"  Latent distance (0 vs 2pi): {period_metrics['latent_distance']:.6f}")

    # 3. Training curves
    print("Generating plots...")
    plot_training_curves(
        history,
        save_path=os.path.join(results_dir, 'training_curves.png'),
    )

    # 4. Reconstruction examples
    plot_reconstructions(
        state, test_ds, n_examples=8, is_vae=is_vae,
        save_path=os.path.join(results_dir, 'reconstructions.png'),
    )

    # 5. Latent scatter
    plot_latent_scatter(
        state, train_ds, config, is_vae=is_vae,
        save_path=os.path.join(results_dir, 'latent_scatter.png'),
    )

    # 6. Latent interpolation
    plot_latent_interpolation(
        state, config, n_points=config.eval.n_interpolation, is_vae=is_vae,
        save_path=os.path.join(results_dir, 'interpolation.png'),
    )

    # 7. Periodicity check plot (T^1)
    if config.data.torus_dim == 1:
        plot_periodicity_check(
            state, config, is_vae=is_vae,
            save_path=os.path.join(results_dir, 'periodicity_check.png'),
        )

    # 8. PCA analysis (useful for higher-dim latent)
    z_np = np.array(encode_dataset(state, train_ds, is_vae=is_vae))
    if z_np.shape[1] > 2:
        z_pca, pca = pca_latent(state, train_ds, n_components=2, is_vae=is_vae)
        summary['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
        print(f"  PCA explained variance: {pca.explained_variance_ratio_}")

    # Save summary
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {results_dir}")
    return summary
