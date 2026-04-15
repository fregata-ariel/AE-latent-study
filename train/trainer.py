"""Training loop orchestration."""

import os
import json

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
from flax.training.train_state import TrainState

from data.dataset import create_splits, batched_iterator, Dataset
from data.generation import (
    generate_lattice_theta, make_cyclic_modular_partners, normalize_lattice_signals,
)
from models import create_model
from train.train_state import create_train_state
from train.train_step import (
    train_step_ae, eval_step_ae, eval_step_vae, _make_train_step_vae,
    _make_train_step_lattice_invariant, _make_eval_step_lattice_invariant,
    _make_train_step_lattice_invariant_vae, _make_eval_step_lattice_invariant_vae,
)
from train.checkpointing import create_checkpoint_manager, save_checkpoint


def _average_metrics(
    metrics_list: list[dict[str, jnp.ndarray]],
) -> dict[str, float]:
    """Average a list of metrics dicts."""
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {
        k: float(jnp.mean(jnp.array([m[k] for m in metrics_list])))
        for k in keys
    }


def _evaluate(
    state: TrainState,
    dataset: Dataset,
    batch_size: int,
    key: jax.Array,
    is_vae: bool,
    config: ml_collections.ConfigDict | None = None,
) -> dict[str, float]:
    """Run evaluation on a full dataset."""
    metrics_list = []
    use_invariance = _should_use_lattice_invariance(config)
    if use_invariance and is_vae:
        eval_fn = _make_eval_step_lattice_invariant_vae(
            config.model.vae_beta,
            config.train.modular_invariance_weight,
        )
    elif use_invariance:
        eval_fn = _make_eval_step_lattice_invariant(
            config.train.modular_invariance_weight,
        )
    else:
        eval_fn = eval_step_vae if is_vae else eval_step_ae

    for batch_signals, batch_thetas in _iter_eval_batches(dataset, batch_size, key):
        if use_invariance:
            paired_batch = _make_lattice_partner_signals(batch_thetas, config)
            metrics, _ = eval_fn(state, batch_signals, paired_batch)
        else:
            metrics, _ = eval_fn(state, batch_signals)
        metrics_list.append(metrics)
    return _average_metrics(metrics_list)


def _should_use_lattice_invariance(
    config: ml_collections.ConfigDict | None,
) -> bool:
    """Whether the current experiment should optimize modular invariance."""
    if config is None:
        return False

    return (
        getattr(config.data, 'data_type', 'torus') == 'lattice'
        and getattr(config.train, 'modular_invariance_weight', 0.0) > 0.0
        and config.model.latent_type in ('standard', 'vae')
    )


def _iter_eval_batches(
    dataset: Dataset,
    batch_size: int,
    key: jax.Array,
):
    """Yield evaluation batches with full coverage, including the remainder."""
    for batch_signals, batch_thetas in batched_iterator(
        dataset, batch_size, key, shuffle=False,
    ):
        yield batch_signals, batch_thetas

    n_covered = (len(dataset) // batch_size) * batch_size
    if n_covered < len(dataset):
        yield dataset.signals[n_covered:], dataset.thetas[n_covered:]


def _make_lattice_partner_signals(
    batch_thetas: jnp.ndarray,
    config: ml_collections.ConfigDict,
) -> jnp.ndarray:
    """Generate cyclic modular partners for a batch of lattice parameters."""
    tau = np.array(batch_thetas[:, 0]) + 1j * np.array(batch_thetas[:, 1])
    tau_partner, _ = make_cyclic_modular_partners(tau)

    signals = generate_lattice_theta(
        tau_partner,
        config.data.signal_length,
        t_min=getattr(config.data, 'lattice_t_min', 0.5),
        t_max=getattr(config.data, 'lattice_t_max', 5.0),
        K=getattr(config.data, 'lattice_K', 10),
    )
    return normalize_lattice_signals(
        signals,
        method=getattr(config.data, 'lattice_signal_normalization', 'none'),
    )


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str,
) -> tuple[TrainState, dict, tuple[Dataset, Dataset, Dataset]]:
    """Full training pipeline.

    Args:
        config: Experiment configuration.
        workdir: Directory for checkpoints and results.

    Returns:
        Tuple of (final_state, history_dict, (train_ds, val_ds, test_ds)).
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    # Save config
    with open(os.path.join(workdir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    key = jax.random.PRNGKey(config.seed)

    # Data
    key, data_key = jax.random.split(key)
    train_ds, val_ds, test_ds = create_splits(config, data_key)
    print(f"Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Model
    model = create_model(config)
    key, init_key = jax.random.split(key)
    state = create_train_state(config, model, init_key)
    print(f"Model: {config.model.latent_type}, latent_dim={config.model.latent_dim}")
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"Parameters: {param_count:,}")

    # Checkpointing
    ckpt_mngr = create_checkpoint_manager(config, workdir)

    # Training setup
    is_vae = config.model.latent_type == 'vae'
    use_invariance = _should_use_lattice_invariance(config)
    if (
        getattr(config.train, 'modular_invariance_weight', 0.0) > 0.0
        and getattr(config.data, 'data_type', 'torus') != 'lattice'
    ):
        raise ValueError('modular_invariance_weight is only supported for lattice data.')
    if (
        getattr(config.train, 'modular_invariance_weight', 0.0) > 0.0
        and config.model.latent_type not in ('standard', 'vae')
    ):
        raise ValueError(
            'modular_invariance_weight is only supported for standard lattice AEs and lattice VAEs.'
        )

    if is_vae and use_invariance:
        train_step_fn = _make_train_step_lattice_invariant_vae(
            config.model.vae_beta,
            config.train.modular_invariance_weight,
        )
    elif is_vae:
        train_step_fn = _make_train_step_vae(config.model.vae_beta)
    elif use_invariance:
        train_step_fn = _make_train_step_lattice_invariant(
            config.train.modular_invariance_weight,
        )
    else:
        train_step_fn = train_step_ae

    batch_size = config.train.batch_size
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'val_mse': [],
    }
    if is_vae:
        history['train_kl'] = []
    if use_invariance:
        history['train_inv_loss'] = []

    best_val_loss = float('inf')
    best_epoch = -1
    best_state = None
    patience_counter = 0

    # Training loop
    for epoch in range(config.train.num_epochs):
        key, epoch_key = jax.random.split(key)

        # Train
        train_metrics_list = []
        for batch_signals, batch_thetas in batched_iterator(
            train_ds, batch_size, epoch_key, shuffle=True,
        ):
            if use_invariance:
                paired_batch = _make_lattice_partner_signals(batch_thetas, config)
                state, metrics = train_step_fn(state, batch_signals, paired_batch)
            else:
                state, metrics = train_step_fn(state, batch_signals)
            train_metrics_list.append(metrics)

        train_metrics = _average_metrics(train_metrics_list)

        # Validate
        key, val_key = jax.random.split(key)
        val_metrics = _evaluate(
            state, val_ds, batch_size, val_key, is_vae, config=config,
        )

        # Record history
        history['train_loss'].append(train_metrics.get('loss', 0.0))
        history['val_loss'].append(val_metrics.get('loss', 0.0))
        history['train_mse'].append(train_metrics.get('mse', 0.0))
        history['val_mse'].append(val_metrics.get('mse', 0.0))
        if is_vae and 'kl' in train_metrics:
            history['train_kl'].append(train_metrics['kl'])
        if use_invariance and 'inv_loss' in train_metrics:
            history['train_inv_loss'].append(train_metrics['inv_loss'])

        # Logging
        if (epoch + 1) % config.train.log_every == 0 or epoch == 0:
            msg = (
                f"Epoch {epoch + 1:4d}/{config.train.num_epochs} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f}"
            )
            print(msg)

        # Checkpoint best model
        val_loss = val_metrics.get('loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = state
            patience_counter = 0
            save_checkpoint(ckpt_mngr, epoch, state)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.train.patience:
            print(f"Early stopping at epoch {epoch + 1} (best={best_epoch + 1})")
            break

    # Restore best model
    if best_epoch >= 0 and best_state is not None:
        state = best_state
        print(f"Selected best model from epoch {best_epoch + 1} "
              f"(val_loss={best_val_loss:.6f})")

    # Save history
    with open(os.path.join(workdir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return state, history, (train_ds, val_ds, test_ds)
