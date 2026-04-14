"""Training loop orchestration."""

import os
import json

import jax
import jax.numpy as jnp
import ml_collections
from flax.training.train_state import TrainState

from data.dataset import create_splits, batched_iterator, Dataset
from models import create_model
from train.train_state import create_train_state
from train.train_step import train_step_ae, eval_step_ae, eval_step_vae, _make_train_step_vae
from train.checkpointing import create_checkpoint_manager, save_checkpoint, restore_checkpoint


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
) -> dict[str, float]:
    """Run evaluation on a full dataset."""
    eval_fn = eval_step_vae if is_vae else eval_step_ae
    metrics_list = []
    for batch_signals, _ in batched_iterator(dataset, batch_size, key, shuffle=False):
        metrics, _ = eval_fn(state, batch_signals)
        metrics_list.append(metrics)
    return _average_metrics(metrics_list)


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
    if is_vae:
        train_step_fn = _make_train_step_vae(config.model.vae_beta)
    else:
        train_step_fn = train_step_ae

    batch_size = config.train.batch_size
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'val_mse': [],
    }
    if is_vae:
        history['train_kl'] = []

    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0

    # Training loop
    for epoch in range(config.train.num_epochs):
        key, epoch_key = jax.random.split(key)

        # Train
        train_metrics_list = []
        for batch_signals, _ in batched_iterator(
            train_ds, batch_size, epoch_key, shuffle=True,
        ):
            state, metrics = train_step_fn(state, batch_signals)
            train_metrics_list.append(metrics)

        train_metrics = _average_metrics(train_metrics_list)

        # Validate
        key, val_key = jax.random.split(key)
        val_metrics = _evaluate(state, val_ds, batch_size, val_key, is_vae)

        # Record history
        history['train_loss'].append(train_metrics.get('loss', 0.0))
        history['val_loss'].append(val_metrics.get('loss', 0.0))
        history['train_mse'].append(train_metrics.get('mse', 0.0))
        history['val_mse'].append(val_metrics.get('mse', 0.0))
        if is_vae and 'kl' in train_metrics:
            history['train_kl'].append(train_metrics['kl'])

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
            patience_counter = 0
            save_checkpoint(ckpt_mngr, epoch, state)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.train.patience:
            print(f"Early stopping at epoch {epoch + 1} (best={best_epoch + 1})")
            break

    # Restore best model
    if best_epoch >= 0:
        state = restore_checkpoint(ckpt_mngr, best_epoch, state)
        print(f"Restored best model from epoch {best_epoch + 1} "
              f"(val_loss={best_val_loss:.6f})")

    # Save history
    with open(os.path.join(workdir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return state, history, (train_ds, val_ds, test_ds)
