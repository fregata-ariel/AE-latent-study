"""Unit tests for training step builders."""

import jax
import jax.numpy as jnp
import numpy as np

from configs.lattice_default import get_config as get_lattice_config
from models import create_model
from train.train_state import create_train_state
from train.train_step import (
    _make_train_step_factorized_lattice_vae,
    _make_train_step_lattice_invariant_vae,
    _quotient_chart_loss,
    _quotient_spread_loss,
    _quotient_variance_floor_loss,
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


def _tiny_factorized_lattice_config():
    config = get_lattice_config()
    config.seed = 0
    config.data.signal_length = 32
    config.data.n_train = 16
    config.model.latent_type = 'factorized_vae'
    config.model.latent_dim = 6
    config.model.quotient_dim = 2
    config.model.gauge_dim = 4
    config.model.encoder_hidden = (16, 8)
    config.model.decoder_hidden = (8, 16)
    config.model.vae_beta = 0.01
    config.train.batch_size = 8
    config.train.num_epochs = 1
    config.train.modular_invariance_weight = 0.1
    config.train.gauge_equivariance_weight = 0.03
    config.train.decoder_equivariance_weight = 0.03
    config.train.gauge_action_reg_weight = 1e-4
    config.train.chart_preserving_weight = 0.03
    config.train.chart_preserving_n_neighbors = 4
    config.train.quotient_variance_floor_weight = 0.01
    config.train.quotient_variance_floor_target = 0.15
    config.train.quotient_spread_weight = 0.03
    config.train.quotient_min_eig_ratio_target = 0.20
    config.train.quotient_trace_cap_ratio = 1.50
    return config


def test_quotient_chart_loss_is_scale_invariant():
    tau = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    quotient = tau * 2.5

    loss_a = _quotient_chart_loss(tau, quotient, n_neighbors=2)
    loss_b = _quotient_chart_loss(tau * 3.0, quotient * 7.0, n_neighbors=2)

    np.testing.assert_allclose(np.array(loss_a), np.array(loss_b), atol=1e-7)


def test_quotient_chart_loss_worsens_when_local_geometry_is_distorted():
    tau = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.4],
    ])
    quotient_good = tau * 1.7
    quotient_bad = jnp.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [0.0, 0.1],
        [5.0, 0.2],
        [0.2, 3.0],
    ])

    loss_good = _quotient_chart_loss(tau, quotient_good, n_neighbors=3)
    loss_bad = _quotient_chart_loss(tau, quotient_bad, n_neighbors=3)

    assert float(loss_bad) > float(loss_good)


def test_quotient_variance_floor_loss_detects_collapse():
    collapsed = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])
    spread = jnp.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [3.0, 0.5],
    ])

    collapsed_loss, _ = _quotient_variance_floor_loss(collapsed, target=0.15)
    spread_loss, _ = _quotient_variance_floor_loss(spread, target=0.15)

    assert float(collapsed_loss) > 0.0
    assert float(spread_loss) >= 0.0
    assert float(spread_loss) < float(collapsed_loss)


def test_quotient_spread_loss_is_rotation_invariant():
    tau = jnp.array([
        [0.0, 0.0],
        [1.0, 0.2],
        [0.3, 1.2],
        [1.1, 1.0],
        [0.5, 0.7],
    ])
    quotient = jnp.array([
        [0.1, 0.0],
        [1.3, 0.2],
        [0.5, 1.0],
        [1.2, 0.9],
        [0.6, 0.5],
    ])
    theta = np.pi / 3.0
    rotation = jnp.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ], dtype=jnp.float32)
    rotated = quotient @ rotation.T

    loss_a, eigs_a, _, _ = _quotient_spread_loss(tau, quotient, 0.2, 1.5)
    loss_b, eigs_b, _, _ = _quotient_spread_loss(tau, rotated, 0.2, 1.5)

    np.testing.assert_allclose(np.array(loss_a), np.array(loss_b), atol=1e-7)
    np.testing.assert_allclose(np.array(eigs_a), np.array(eigs_b), atol=1e-7)


def test_quotient_spread_loss_detects_collapse():
    tau = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.2, 1.2],
        [1.1, 1.0],
        [0.6, 0.7],
    ])
    collapsed = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ])
    spread = jnp.array([
        [0.0, 0.0],
        [1.0, 0.1],
        [0.2, 1.1],
        [1.2, 1.0],
        [0.6, 0.7],
    ])

    loss_collapsed, _, _, _ = _quotient_spread_loss(tau, collapsed, 0.2, 1.5)
    loss_spread, _, _, _ = _quotient_spread_loss(tau, spread, 0.2, 1.5)

    assert float(loss_collapsed) > float(loss_spread)


def test_quotient_spread_loss_penalizes_over_expansion():
    tau = jnp.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.4, 0.6],
    ])
    balanced = tau
    overexpanded = tau * jnp.array([4.0, 0.2])

    loss_balanced, _, _, _ = _quotient_spread_loss(tau, balanced, 0.2, 1.5)
    loss_overexpanded, _, _, _ = _quotient_spread_loss(tau, overexpanded, 0.2, 1.5)

    assert float(loss_overexpanded) > float(loss_balanced)


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


def test_train_step_factorized_lattice_vae_reports_all_terms():
    config = _tiny_factorized_lattice_config()
    key = jax.random.PRNGKey(config.seed)
    model = create_model(config)
    state = create_train_state(config, model, key)

    batch_key, pair_key = jax.random.split(jax.random.PRNGKey(321))
    batch = jax.random.normal(
        batch_key, (config.train.batch_size, config.data.signal_length),
    )
    paired_batch = jax.random.normal(
        pair_key, (config.train.batch_size, config.data.signal_length),
    )
    transform_ids = jnp.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=jnp.int32)
    tau_fd_coords = jnp.array([
        [-0.4, 1.1],
        [-0.2, 1.2],
        [0.1, 1.4],
        [0.3, 1.6],
        [-0.1, 1.8],
        [0.2, 2.0],
        [-0.3, 2.2],
        [0.0, 2.4],
    ], dtype=jnp.float32)

    step_fn = _make_train_step_factorized_lattice_vae(
        model,
        config.model.vae_beta,
        config.train.modular_invariance_weight,
        config.train.gauge_equivariance_weight,
        config.train.decoder_equivariance_weight,
        config.train.gauge_action_reg_weight,
        config.train.chart_preserving_weight,
        config.train.chart_preserving_n_neighbors,
        config.train.quotient_variance_floor_weight,
        config.train.quotient_variance_floor_target,
        config.train.quotient_spread_weight,
        config.train.quotient_min_eig_ratio_target,
        config.train.quotient_trace_cap_ratio,
    )
    next_state, metrics = step_fn(
        state, batch, paired_batch, transform_ids, tau_fd_coords,
    )

    assert next_state.rng.shape == state.rng.shape
    for key in (
        'loss',
        'mse',
        'kl',
        'quotient_invariance',
        'gauge_equivariance',
        'decoder_equivariance',
        'action_regularizer',
        'quotient_chart_loss',
        'quotient_variance_floor_loss',
        'quotient_spread_loss',
    ):
        assert key in metrics
        assert np.isfinite(float(metrics[key]))
    assert float(metrics['decoder_equivariance']) >= 0.0
