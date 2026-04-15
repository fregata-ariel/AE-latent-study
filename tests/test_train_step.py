"""Unit tests for training step builders."""

import jax
import jax.numpy as jnp
import numpy as np

from configs.lattice_default import get_config as get_lattice_config
from models import create_model
from train.train_state import create_train_state
from train.train_step import (
    _make_train_step_lattice_invariant_vae,
    _make_train_step_vae,
)


def _tiny_lattice_vae_config():
    config = get_lattice_config()
    config.seed = 0
    config.data.signal_length = 32
    config.data.n_train = 16
    config.model.latent_type = 'vae'
    config.model.latent_dim = 4
    config.model.encoder_hidden = (16, 8)
    config.model.decoder_hidden = (8, 16)
    config.model.vae_beta = 0.01
    config.train.batch_size = 8
    config.train.num_epochs = 1
    config.train.modular_invariance_weight = 0.1
    return config


def _make_state_and_batches():
    config = _tiny_lattice_vae_config()
    key = jax.random.PRNGKey(config.seed)
    model = create_model(config)
    state = create_train_state(config, model, key)

    batch_key, pair_key = jax.random.split(jax.random.PRNGKey(123))
    batch = jax.random.normal(batch_key, (config.train.batch_size, config.data.signal_length))
    paired_batch = jax.random.normal(pair_key, (config.train.batch_size, config.data.signal_length))
    return config, state, batch, paired_batch


def test_train_step_lattice_invariant_vae_reports_all_terms():
    config, state, batch, paired_batch = _make_state_and_batches()
    step_fn = _make_train_step_lattice_invariant_vae(
        config.model.vae_beta,
        config.train.modular_invariance_weight,
    )

    next_state, metrics = step_fn(state, batch, paired_batch)

    assert next_state.rng.shape == state.rng.shape
    for key in ('loss', 'mse', 'kl', 'inv_loss'):
        assert key in metrics
        assert np.isfinite(float(metrics[key]))
    assert float(metrics['inv_loss']) >= 0.0


def test_train_step_lattice_invariant_vae_inv_loss_independent_of_rng():
    config, state, batch, paired_batch = _make_state_and_batches()
    step_fn = _make_train_step_lattice_invariant_vae(
        config.model.vae_beta,
        config.train.modular_invariance_weight,
    )

    state_a = state.replace(rng=jax.random.PRNGKey(7))
    state_b = state.replace(rng=jax.random.PRNGKey(99))

    _, metrics_a = step_fn(state_a, batch, paired_batch)
    _, metrics_b = step_fn(state_b, batch, paired_batch)

    np.testing.assert_allclose(
        np.array(metrics_a['inv_loss']),
        np.array(metrics_b['inv_loss']),
        atol=1e-7,
    )


def test_train_step_lattice_invariant_vae_weight_zero_matches_vae_step():
    config, state, batch, paired_batch = _make_state_and_batches()
    state_ref = state.replace(rng=jax.random.PRNGKey(11))
    state_inv = state.replace(rng=jax.random.PRNGKey(11))

    vae_step = _make_train_step_vae(config.model.vae_beta)
    inv_step = _make_train_step_lattice_invariant_vae(config.model.vae_beta, 0.0)

    next_ref, metrics_ref = vae_step(state_ref, batch)
    next_inv, metrics_inv = inv_step(state_inv, batch, paired_batch)

    for key in ('loss', 'mse', 'kl'):
        np.testing.assert_allclose(
            np.array(metrics_ref[key]),
            np.array(metrics_inv[key]),
            atol=1e-7,
        )

    ref_leaves = jax.tree_util.tree_leaves(next_ref.params)
    inv_leaves = jax.tree_util.tree_leaves(next_inv.params)
    assert len(ref_leaves) == len(inv_leaves)
    for ref_leaf, inv_leaf in zip(ref_leaves, inv_leaves, strict=True):
        np.testing.assert_allclose(np.array(ref_leaf), np.array(inv_leaf), atol=1e-7)

    np.testing.assert_array_equal(np.array(next_ref.rng), np.array(next_inv.rng))
