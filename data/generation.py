"""Signal generation from torus-structured latent parameters."""

from typing import Optional

import jax
import jax.numpy as jnp
import ml_collections


def generate_t1_signals(
    thetas: jnp.ndarray,
    omega: float,
    signal_length: int,
    dt: float,
    noise_std: float = 0.0,
    key: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """Generate signals from T^1 (single angle) parameters.

    x(t; theta) = sin(omega * t + theta) + noise

    Args:
        thetas: Angle parameters, shape (N,) in [0, 2*pi).
        omega: Angular frequency.
        signal_length: Number of time steps.
        dt: Time step spacing.
        noise_std: Standard deviation of additive Gaussian noise.
        key: PRNG key for noise generation.

    Returns:
        Signals of shape (N, signal_length).
    """
    t = jnp.arange(signal_length) * dt  # (T,)
    signals = jnp.sin(omega * t[None, :] + thetas[:, None])  # (N, T)

    if noise_std > 0.0 and key is not None:
        noise = jax.random.normal(key, signals.shape) * noise_std
        signals = signals + noise

    return signals


def generate_t2_signals(
    theta1: jnp.ndarray,
    theta2: jnp.ndarray,
    omega1: float,
    omega2: float,
    a1: float,
    a2: float,
    signal_length: int,
    dt: float,
    noise_std: float = 0.0,
    key: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """Generate signals from T^2 (two angles) parameters.

    x(t; theta1, theta2) = a1*sin(omega1*t + theta1) + a2*sin(omega2*t + theta2)

    Args:
        theta1: First angle parameters, shape (N,) in [0, 2*pi).
        theta2: Second angle parameters, shape (N,) in [0, 2*pi).
        omega1: First angular frequency.
        omega2: Second angular frequency.
        a1: Amplitude for first component.
        a2: Amplitude for second component.
        signal_length: Number of time steps.
        dt: Time step spacing.
        noise_std: Standard deviation of additive Gaussian noise.
        key: PRNG key for noise generation.

    Returns:
        Signals of shape (N, signal_length).
    """
    t = jnp.arange(signal_length) * dt  # (T,)
    comp1 = a1 * jnp.sin(omega1 * t[None, :] + theta1[:, None])
    comp2 = a2 * jnp.sin(omega2 * t[None, :] + theta2[:, None])
    signals = comp1 + comp2  # (N, T)

    if noise_std > 0.0 and key is not None:
        noise = jax.random.normal(key, signals.shape) * noise_std
        signals = signals + noise

    return signals


def generate_dataset(
    config: ml_collections.ConfigDict,
    key: jax.Array,
) -> dict:
    """Generate synthetic dataset based on config.

    Args:
        config: Experiment configuration.
        key: PRNG key.

    Returns:
        Dictionary with 'signals' (N, T) and 'thetas' (N,) or (N, 2).
    """
    n_total = config.data.n_train + config.data.n_val + config.data.n_test
    torus_dim = config.data.torus_dim

    if torus_dim == 1:
        key, subkey = jax.random.split(key)
        thetas = jax.random.uniform(
            subkey, (n_total,), minval=0.0, maxval=2.0 * jnp.pi
        )
        key, subkey = jax.random.split(key)
        signals = generate_t1_signals(
            thetas=thetas,
            omega=config.data.omega,
            signal_length=config.data.signal_length,
            dt=config.data.dt,
            noise_std=config.data.noise_std,
            key=subkey,
        )
    elif torus_dim == 2:
        key, k1, k2 = jax.random.split(key, 3)
        theta1 = jax.random.uniform(
            k1, (n_total,), minval=0.0, maxval=2.0 * jnp.pi
        )
        theta2 = jax.random.uniform(
            k2, (n_total,), minval=0.0, maxval=2.0 * jnp.pi
        )
        thetas = jnp.stack([theta1, theta2], axis=-1)  # (N, 2)
        key, subkey = jax.random.split(key)
        signals = generate_t2_signals(
            theta1=theta1,
            theta2=theta2,
            omega1=config.data.omega1,
            omega2=config.data.omega2,
            a1=config.data.a1,
            a2=config.data.a2,
            signal_length=config.data.signal_length,
            dt=config.data.dt,
            noise_std=config.data.noise_std,
            key=subkey,
        )
    else:
        raise ValueError(f"Unsupported torus_dim={torus_dim}. Use 1 or 2.")

    return {'signals': signals, 'thetas': thetas}
