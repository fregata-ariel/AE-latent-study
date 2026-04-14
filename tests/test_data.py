"""Tests for data generation and dataset utilities."""

import jax
import jax.numpy as jnp
import numpy as np

from configs.default import get_config
from data.generation import (
    apply_modular_transform,
    generate_t1_signals,
    generate_t2_signals,
    generate_dataset,
    make_cyclic_modular_partners,
    normalize_lattice_signals,
)
from data.dataset import Dataset, create_splits, batched_iterator
from eval.metrics import compute_j_correlation


def test_t1_signal_shape():
    thetas = jnp.array([0.0, 1.0, 2.0])
    signals = generate_t1_signals(thetas, omega=2.0, signal_length=100, dt=0.1)
    assert signals.shape == (3, 100)


def test_t2_signal_shape():
    theta1 = jnp.array([0.0, 1.0])
    theta2 = jnp.array([0.5, 1.5])
    signals = generate_t2_signals(
        theta1, theta2, omega1=2.0, omega2=3.0,
        a1=1.0, a2=0.5, signal_length=100, dt=0.1,
    )
    assert signals.shape == (2, 100)


def test_t1_periodicity():
    """Signals at theta=0 and theta=2*pi should be identical."""
    theta_0 = jnp.array([0.0])
    theta_2pi = jnp.array([2.0 * jnp.pi])
    sig_0 = generate_t1_signals(theta_0, omega=2.0, signal_length=100, dt=0.1)
    sig_2pi = generate_t1_signals(theta_2pi, omega=2.0, signal_length=100, dt=0.1)
    np.testing.assert_allclose(np.array(sig_0), np.array(sig_2pi), atol=1e-5)


def test_t1_noise():
    key = jax.random.PRNGKey(0)
    thetas = jnp.array([0.0])
    signals_clean = generate_t1_signals(
        thetas, omega=2.0, signal_length=100, dt=0.1, noise_std=0.0,
    )
    signals_noisy = generate_t1_signals(
        thetas, omega=2.0, signal_length=100, dt=0.1, noise_std=0.1, key=key,
    )
    # Noisy signal should differ from clean
    assert not np.allclose(np.array(signals_clean), np.array(signals_noisy))


def test_generate_dataset_t1():
    config = get_config()
    config.data.torus_dim = 1
    config.data.n_train = 10
    config.data.n_val = 5
    config.data.n_test = 5
    key = jax.random.PRNGKey(42)
    data = generate_dataset(config, key)
    assert data['signals'].shape == (20, 100)
    assert data['thetas'].shape == (20,)


def test_generate_dataset_t2():
    config = get_config()
    config.data.torus_dim = 2
    config.data.n_train = 10
    config.data.n_val = 5
    config.data.n_test = 5
    key = jax.random.PRNGKey(42)
    data = generate_dataset(config, key)
    assert data['signals'].shape == (20, 100)
    assert data['thetas'].shape == (20, 2)


def test_create_splits():
    config = get_config()
    config.data.n_train = 20
    config.data.n_val = 5
    config.data.n_test = 5
    key = jax.random.PRNGKey(42)
    train_ds, val_ds, test_ds = create_splits(config, key)
    assert len(train_ds) == 20
    assert len(val_ds) == 5
    assert len(test_ds) == 5


def test_batched_iterator():
    signals = jnp.ones((10, 50))
    thetas = jnp.zeros(10)
    ds = Dataset(signals=signals, thetas=thetas)
    key = jax.random.PRNGKey(0)

    batches = list(batched_iterator(ds, batch_size=3, key=key, shuffle=False))
    assert len(batches) == 3  # 10 // 3 = 3 (drops remainder)
    for sig_batch, theta_batch in batches:
        assert sig_batch.shape == (3, 50)
        assert theta_batch.shape == (3,)


def test_seed_reproducibility():
    config = get_config()
    config.data.n_train = 10
    config.data.n_val = 5
    config.data.n_test = 5
    key = jax.random.PRNGKey(42)
    data1 = generate_dataset(config, key)
    data2 = generate_dataset(config, key)
    np.testing.assert_array_equal(np.array(data1['signals']), np.array(data2['signals']))


def test_normalize_lattice_signals_max():
    signals = jnp.array([[1.0, 2.0, 4.0], [3.0, 1.5, 0.75]])
    normalized = normalize_lattice_signals(signals, method='max')
    np.testing.assert_allclose(np.max(np.array(normalized), axis=1), 1.0, atol=1e-6)


def test_normalize_lattice_signals_none():
    signals = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    unchanged = normalize_lattice_signals(signals, method='none')
    np.testing.assert_array_equal(np.array(signals), np.array(unchanged))


def test_modular_transforms_preserve_upper_halfplane():
    tau = np.array([0.1 + 1.2j, -0.3 + 0.9j, 0.4 + 2.1j])
    for name in ('T', 'S', 'ST'):
        transformed = apply_modular_transform(tau, name)
        assert np.all(transformed.imag > 0.0)

    partners, names = make_cyclic_modular_partners(tau)
    assert list(names) == ['T', 'S', 'ST']
    assert np.all(partners.imag > 0.0)


def test_compute_j_correlation_extended_metrics():
    z = jnp.stack([
        jnp.linspace(-1.0, 1.0, 32),
        jnp.linspace(1.0, -1.0, 32),
    ], axis=-1)
    logabs = np.linspace(0.0, 3.0, 32)
    j_values = np.exp(logabs).astype(np.complex128)

    metrics = compute_j_correlation(z, j_values)

    for key in (
        'z0_vs_logabsj_pearson',
        'z0_vs_logabsj_spearman',
        'z0_vs_logabsj_mutual_info',
        'max_abs_logabsj_pearson',
        'max_abs_logabsj_spearman',
        'max_logabsj_mutual_info',
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])

    assert metrics['z0_vs_logabsj_spearman'] > 0.99
