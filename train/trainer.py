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
    compute_j_invariant,
    generate_lattice_theta,
    make_cyclic_modular_partners,
    modular_transform_ids,
    normalize_lattice_signals,
    reduce_to_fundamental_domain,
)
from models import create_model
from train.train_state import create_train_state
from train.train_step import (
    train_step_ae, eval_step_ae, eval_step_vae, _make_train_step_vae,
    _make_train_step_lattice_invariant, _make_eval_step_lattice_invariant,
    _make_train_step_lattice_invariant_vae, _make_eval_step_lattice_invariant_vae,
    _make_train_step_factorized_lattice_vae, _make_eval_step_factorized_lattice_vae,
)
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
    latent_type: str,
    config: ml_collections.ConfigDict | None = None,
    teacher_state: TrainState | None = None,
) -> dict[str, float]:
    """Run evaluation on a full dataset."""
    metrics_list = []
    use_invariance = _should_use_lattice_invariance(config)
    is_vae = latent_type == 'vae'
    is_factorized = latent_type == 'factorized_vae'

    if is_factorized:
        model = create_model(config)
        eval_fn = _make_eval_step_factorized_lattice_vae(
            model,
            config.model.vae_beta,
            config.train.modular_invariance_weight,
            config.train.gauge_equivariance_weight,
            config.train.decoder_equivariance_weight,
            config.train.gauge_action_reg_weight,
            getattr(config.train, 'chart_preserving_weight', 0.0),
            getattr(config.train, 'chart_preserving_n_neighbors', 8),
            getattr(config.train, 'quotient_variance_floor_weight', 0.0),
            getattr(config.train, 'quotient_variance_floor_target', 0.15),
            getattr(config.train, 'quotient_spread_weight', 0.0),
            getattr(config.train, 'quotient_min_eig_ratio_target', 0.20),
            getattr(config.train, 'quotient_trace_cap_ratio', 1.50),
            getattr(config.train, 'jacobian_gram_weight', 0.0),
            getattr(config.train, 'jacobian_n_neighbors', 8),
            getattr(config.train, 'quotient_logdet_weight', 0.0),
            getattr(config.train, 'quotient_logdet_ratio_target', 0.10),
            getattr(config.train, 'contrastive_local_weight', 0.0),
            getattr(config.train, 'contrastive_n_neighbors', 8),
            getattr(config.train, 'contrastive_temperature', 0.10),
            getattr(config.train, 'j_rank_preserving_weight', 0.0),
            getattr(config.train, 'j_rank_temperature', 0.10),
            getattr(config.train, 'j_rank_min_delta', 0.10),
            getattr(config.train, 'teacher_distill_weight', 0.0),
            getattr(config.train, 'teacher_distill_n_neighbors', 8),
        )
    elif use_invariance and is_vae:
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
            paired_batch, transform_ids = _make_lattice_partner_batch(batch_thetas, config)
            if is_factorized:
                tau_fd_coords = _reduce_tau_batch_to_fd_coords(batch_thetas)
                j_rank_targets = _make_j_rank_targets(batch_thetas, config)
                teacher_quotient = _make_teacher_quotient_batch(
                    batch_signals,
                    teacher_state,
                    config,
                )
                metrics, _ = eval_fn(
                    state,
                    batch_signals,
                    paired_batch,
                    transform_ids,
                    tau_fd_coords,
                    j_rank_targets,
                    teacher_quotient,
                )
            else:
                metrics, _ = eval_fn(state, batch_signals, paired_batch)
        else:
            metrics, _ = eval_fn(state, batch_signals)
        metrics_list.append(metrics)
    return _average_metrics(metrics_list)


def _should_use_lattice_invariance(
    config: ml_collections.ConfigDict | None,
) -> bool:
    """Whether the current experiment should use lattice partner batches."""
    if config is None:
        return False

    latent_type = config.model.latent_type
    weight = getattr(config.train, 'modular_invariance_weight', 0.0)
    return (
        getattr(config.data, 'data_type', 'torus') == 'lattice'
        and (
            (latent_type in ('standard', 'vae') and weight > 0.0)
            or latent_type == 'factorized_vae'
        )
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


def _make_lattice_partner_batch(
    batch_thetas: jnp.ndarray,
    config: ml_collections.ConfigDict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate cyclic modular partners and transform ids for a batch."""
    tau = np.array(batch_thetas[:, 0]) + 1j * np.array(batch_thetas[:, 1])
    tau_partner, transform_names = make_cyclic_modular_partners(tau)

    signals = generate_lattice_theta(
        tau_partner,
        config.data.signal_length,
        t_min=getattr(config.data, 'lattice_t_min', 0.5),
        t_max=getattr(config.data, 'lattice_t_max', 5.0),
        K=getattr(config.data, 'lattice_K', 10),
    )
    normalized = normalize_lattice_signals(
        signals,
        method=getattr(config.data, 'lattice_signal_normalization', 'none'),
    )
    return normalized, jnp.asarray(modular_transform_ids(transform_names))


def _reduce_tau_batch_to_fd_coords(
    batch_thetas: jnp.ndarray,
) -> jnp.ndarray:
    """Reduce a batch of lattice τ values to fundamental-domain coordinates."""
    tau = np.array(batch_thetas[:, 0]) + 1j * np.array(batch_thetas[:, 1])
    tau_fd = reduce_to_fundamental_domain(tau)
    coords = np.stack([tau_fd.real, tau_fd.imag], axis=-1)
    return jnp.asarray(coords)


def _make_j_rank_targets(
    batch_thetas: jnp.ndarray,
    config: ml_collections.ConfigDict,
) -> jnp.ndarray:
    """Compute standardized batch log10|j| targets for pairwise rank retention."""
    batch_size = batch_thetas.shape[0]
    if getattr(config.train, 'j_rank_preserving_weight', 0.0) <= 0.0:
        return jnp.zeros((batch_size,), dtype=jnp.float32)

    tau = np.array(batch_thetas[:, 0]) + 1j * np.array(batch_thetas[:, 1])
    j_values = compute_j_invariant(
        tau,
        n_terms=getattr(config.train, 'j_rank_n_terms', 50),
    )
    targets = np.log10(np.maximum(np.abs(j_values), 1e-30))
    targets = targets - np.mean(targets)
    target_std = np.std(targets)
    if target_std > 1e-6:
        targets = targets / target_std
    else:
        targets = np.zeros_like(targets)
    return jnp.asarray(targets, dtype=jnp.float32)


def _teacher_distillation_enabled(config: ml_collections.ConfigDict | None) -> bool:
    """Whether the current factorized run should use a restored teacher."""
    if config is None:
        return False
    return (
        getattr(config.model, 'latent_type', '') == 'factorized_vae'
        and getattr(config.train, 'teacher_distill_weight', 0.0) > 0.0
        and bool(getattr(config.train, 'teacher_run_dir', ''))
    )


def _best_checkpoint_step(workdir: str, history: dict, checkpoint_dir: str) -> int:
    """Pick the validation-best checkpoint step, falling back to latest available."""
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


def _load_config_from_json(config_path: str) -> ml_collections.ConfigDict:
    """Load a saved experiment config as ConfigDict."""
    with open(config_path) as f:
        return ml_collections.ConfigDict(json.load(f))


def _load_teacher_state(
    teacher_run_dir: str,
) -> tuple[TrainState, ml_collections.ConfigDict]:
    """Restore a fixed factorized teacher from a saved run directory."""
    config_path = os.path.join(teacher_run_dir, 'config.json')
    history_path = os.path.join(teacher_run_dir, 'history.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Missing teacher config: {config_path}')
    if not os.path.exists(history_path):
        raise FileNotFoundError(f'Missing teacher history: {history_path}')

    teacher_config = _load_config_from_json(config_path)
    if teacher_config.model.latent_type != 'factorized_vae':
        raise ValueError('teacher_run_dir must point to a factorized_vae run.')

    with open(history_path) as f:
        history = json.load(f)
    teacher_model = create_model(teacher_config)
    state_template = create_train_state(
        teacher_config,
        teacher_model,
        jax.random.PRNGKey(teacher_config.seed),
    )
    checkpoint_manager = create_checkpoint_manager(teacher_config, teacher_run_dir)
    step = _best_checkpoint_step(
        teacher_run_dir,
        history,
        teacher_config.checkpoint.dir,
    )
    return restore_checkpoint(checkpoint_manager, step, state_template), teacher_config


def _make_teacher_quotient_batch(
    batch_signals: jnp.ndarray,
    teacher_state: TrainState | None,
    config: ml_collections.ConfigDict,
) -> jnp.ndarray:
    """Encode a batch with the fixed teacher or return zeros when disabled."""
    quotient_dim = int(getattr(config.model, 'quotient_dim', 2))
    if not _teacher_distillation_enabled(config) or teacher_state is None:
        return jnp.zeros((batch_signals.shape[0], quotient_dim), dtype=batch_signals.dtype)

    _, _, q_mean, _, _, _ = teacher_state.apply_fn(
        {'params': teacher_state.params},
        batch_signals,
        teacher_state.rng,
        deterministic=True,
    )
    return jax.lax.stop_gradient(q_mean)


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
    teacher_state = None
    if _teacher_distillation_enabled(config):
        teacher_run_dir = os.path.abspath(config.train.teacher_run_dir)
        teacher_state, _ = _load_teacher_state(teacher_run_dir)
        print(f"Teacher: {teacher_run_dir}")

    # Training setup
    latent_type = config.model.latent_type
    is_vae = latent_type in ('vae', 'factorized_vae')
    is_factorized = latent_type == 'factorized_vae'
    use_invariance = _should_use_lattice_invariance(config)
    if (
        latent_type == 'factorized_vae'
        and getattr(config.data, 'data_type', 'torus') != 'lattice'
    ):
        raise ValueError('factorized_vae is currently only supported for lattice data.')
    if (
        getattr(config.train, 'modular_invariance_weight', 0.0) > 0.0
        and getattr(config.data, 'data_type', 'torus') != 'lattice'
    ):
        raise ValueError('modular_invariance_weight is only supported for lattice data.')
    if (
        getattr(config.train, 'modular_invariance_weight', 0.0) > 0.0
        and config.model.latent_type not in ('standard', 'vae', 'factorized_vae')
    ):
        raise ValueError(
            'modular_invariance_weight is only supported for standard lattice AEs, lattice VAEs, and factorized lattice VAEs.'
        )

    if is_factorized:
        train_step_fn = _make_train_step_factorized_lattice_vae(
            model,
            config.model.vae_beta,
            config.train.modular_invariance_weight,
            config.train.gauge_equivariance_weight,
            config.train.decoder_equivariance_weight,
            config.train.gauge_action_reg_weight,
            getattr(config.train, 'chart_preserving_weight', 0.0),
            getattr(config.train, 'chart_preserving_n_neighbors', 8),
            getattr(config.train, 'quotient_variance_floor_weight', 0.0),
            getattr(config.train, 'quotient_variance_floor_target', 0.15),
            getattr(config.train, 'quotient_spread_weight', 0.0),
            getattr(config.train, 'quotient_min_eig_ratio_target', 0.20),
            getattr(config.train, 'quotient_trace_cap_ratio', 1.50),
            getattr(config.train, 'jacobian_gram_weight', 0.0),
            getattr(config.train, 'jacobian_n_neighbors', 8),
            getattr(config.train, 'quotient_logdet_weight', 0.0),
            getattr(config.train, 'quotient_logdet_ratio_target', 0.10),
            getattr(config.train, 'contrastive_local_weight', 0.0),
            getattr(config.train, 'contrastive_n_neighbors', 8),
            getattr(config.train, 'contrastive_temperature', 0.10),
            getattr(config.train, 'j_rank_preserving_weight', 0.0),
            getattr(config.train, 'j_rank_temperature', 0.10),
            getattr(config.train, 'j_rank_min_delta', 0.10),
            getattr(config.train, 'teacher_distill_weight', 0.0),
            getattr(config.train, 'teacher_distill_n_neighbors', 8),
        )
    elif is_vae and use_invariance:
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
    if use_invariance and not is_factorized:
        history['train_inv_loss'] = []
    if is_factorized:
        history['train_quotient_invariance'] = []
        history['train_gauge_equivariance'] = []
        history['train_decoder_equivariance'] = []
        history['train_action_regularizer'] = []
        history['train_quotient_chart_loss'] = []
        history['train_quotient_variance_floor_loss'] = []
        history['train_quotient_spread_loss'] = []
        history['train_quotient_jacobian_gram_loss'] = []
        history['train_quotient_logdet_loss'] = []
        history['train_quotient_contrastive_local_loss'] = []
        history['train_quotient_j_rank_loss'] = []
        history['train_quotient_teacher_distill_loss'] = []

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
                paired_batch, transform_ids = _make_lattice_partner_batch(
                    batch_thetas, config,
                )
                if is_factorized:
                    tau_fd_coords = _reduce_tau_batch_to_fd_coords(batch_thetas)
                    j_rank_targets = _make_j_rank_targets(batch_thetas, config)
                    teacher_quotient = _make_teacher_quotient_batch(
                        batch_signals,
                        teacher_state,
                        config,
                    )
                    state, metrics = train_step_fn(
                        state,
                        batch_signals,
                        paired_batch,
                        transform_ids,
                        tau_fd_coords,
                        j_rank_targets,
                        teacher_quotient,
                    )
                else:
                    state, metrics = train_step_fn(state, batch_signals, paired_batch)
            else:
                state, metrics = train_step_fn(state, batch_signals)
            train_metrics_list.append(metrics)

        train_metrics = _average_metrics(train_metrics_list)

        # Validate
        key, val_key = jax.random.split(key)
        val_metrics = _evaluate(
            state,
            val_ds,
            batch_size,
            val_key,
            latent_type,
            config=config,
            teacher_state=teacher_state,
        )

        # Record history
        history['train_loss'].append(train_metrics.get('loss', 0.0))
        history['val_loss'].append(val_metrics.get('loss', 0.0))
        history['train_mse'].append(train_metrics.get('mse', 0.0))
        history['val_mse'].append(val_metrics.get('mse', 0.0))
        if is_vae and 'kl' in train_metrics:
            history['train_kl'].append(train_metrics['kl'])
        if use_invariance and not is_factorized and 'inv_loss' in train_metrics:
            history['train_inv_loss'].append(train_metrics['inv_loss'])
        if is_factorized:
            history['train_quotient_invariance'].append(
                train_metrics.get('quotient_invariance', 0.0),
            )
            history['train_gauge_equivariance'].append(
                train_metrics.get('gauge_equivariance', 0.0),
            )
            history['train_decoder_equivariance'].append(
                train_metrics.get('decoder_equivariance', 0.0),
            )
            history['train_action_regularizer'].append(
                train_metrics.get('action_regularizer', 0.0),
            )
            history['train_quotient_chart_loss'].append(
                train_metrics.get('quotient_chart_loss', 0.0),
            )
            history['train_quotient_variance_floor_loss'].append(
                train_metrics.get('quotient_variance_floor_loss', 0.0),
            )
            history['train_quotient_spread_loss'].append(
                train_metrics.get('quotient_spread_loss', 0.0),
            )
            history['train_quotient_jacobian_gram_loss'].append(
                train_metrics.get('quotient_jacobian_gram_loss', 0.0),
            )
            history['train_quotient_logdet_loss'].append(
                train_metrics.get('quotient_logdet_loss', 0.0),
            )
            history['train_quotient_contrastive_local_loss'].append(
                train_metrics.get('quotient_contrastive_local_loss', 0.0),
            )
            history['train_quotient_j_rank_loss'].append(
                train_metrics.get('quotient_j_rank_loss', 0.0),
            )
            history['train_quotient_teacher_distill_loss'].append(
                train_metrics.get('quotient_teacher_distill_loss', 0.0),
            )

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
