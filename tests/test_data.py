"""Tests for data generation and dataset utilities."""

import jax
import jax.numpy as jnp
import numpy as np

from configs.default import get_config
from data.generation import generate_t1_signals, generate_t2_signals, generate_dataset
from data.dataset import Dataset, create_splits, batched_iterator


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
