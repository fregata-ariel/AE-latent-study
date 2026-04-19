"""Signal generation from torus-structured and lattice latent parameters."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections


_MODULAR_TRANSFORMS = ('T', 'S', 'ST')
_MODULAR_TRANSFORM_TO_ID = {
    name: idx for idx, name in enumerate(_MODULAR_TRANSFORMS)
}


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


# ---------------------------------------------------------------------------
# Lattice / Modular data generation
# ---------------------------------------------------------------------------

def normalize_lattice_signals(
    signals: jnp.ndarray,
    method: str = 'none',
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Normalize lattice signals sample-wise.

    Args:
        signals: Signal array of shape (N, T).
        method: Either 'none' or 'max'.
        eps: Small value to avoid division by zero.

    Returns:
        Normalized signals with the same shape as the input.
    """
    if method == 'none':
        return signals
    if method != 'max':
        raise ValueError(
            f"Unknown lattice normalization '{method}'. Choose from 'none', 'max'."
        )

    row_max = jnp.max(signals, axis=1, keepdims=True)
    return signals / jnp.maximum(row_max, eps)


def sample_fundamental_domain(
    n_samples: int,
    y_max: float = 3.0,
    key: Optional[jax.Array] = None,
) -> np.ndarray:
    """Sample τ uniformly from the fundamental domain F of SL₂(Z).

    F = {τ ∈ H : |Re(τ)| ≤ 1/2, |τ| ≥ 1}

    Uses rejection sampling: draw (x, y) uniformly from the bounding box
    [-0.5, 0.5] × [sqrt(3)/2, y_max] and reject points with |τ| < 1.

    Args:
        n_samples: Number of τ values to generate.
        y_max: Upper bound on Im(τ) (controls cusp truncation).
        key: JAX PRNG key.

    Returns:
        Complex array of shape (n_samples,) with τ values in F.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    y_min = np.sqrt(3.0) / 2.0  # minimum Im(τ) on |τ|=1 boundary
    collected = []

    while len(collected) < n_samples:
        key, k1, k2 = jax.random.split(key, 3)
        # Over-sample to reduce iterations
        batch = max(n_samples * 2, 1000)
        x = np.array(jax.random.uniform(k1, (batch,), minval=-0.5, maxval=0.5))
        y = np.array(jax.random.uniform(k2, (batch,), minval=y_min, maxval=y_max))

        # Reject points outside the fundamental domain (|τ| < 1)
        r_sq = x ** 2 + y ** 2
        mask = r_sq >= 1.0
        valid_x = x[mask]
        valid_y = y[mask]

        for i in range(len(valid_x)):
            if len(collected) >= n_samples:
                break
            collected.append(complex(valid_x[i], valid_y[i]))

    return np.array(collected)


def sample_upper_halfplane(
    n_samples: int,
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (0.3, 4.0),
    key: Optional[jax.Array] = None,
) -> np.ndarray:
    """Sample τ uniformly from a rectangular region of the upper half-plane.

    This samples outside the fundamental domain as well, useful for testing
    SL₂(Z) equivalence.

    Args:
        n_samples: Number of τ values to generate.
        x_range: Range for Re(τ).
        y_range: Range for Im(τ).
        key: JAX PRNG key.

    Returns:
        Complex array of shape (n_samples,).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    k1, k2 = jax.random.split(key)
    x = np.array(jax.random.uniform(k1, (n_samples,),
                                     minval=x_range[0], maxval=x_range[1]))
    y = np.array(jax.random.uniform(k2, (n_samples,),
                                     minval=y_range[0], maxval=y_range[1]))
    return x + 1j * y


def reduce_to_fundamental_domain(tau: np.ndarray) -> np.ndarray:
    """Reduce τ values to the fundamental domain F via SL₂(Z) transformations.

    Applies alternating T (τ → τ+1) and S (τ → -1/τ) transformations until
    τ lands in F = {|Re(τ)| ≤ 1/2, |τ| ≥ 1}.

    Args:
        tau: Complex array of τ values, any shape.

    Returns:
        Complex array of same shape with each τ in F.
    """
    tau_flat = np.array(tau, dtype=complex).ravel()
    result = tau_flat.copy()

    for i in range(len(result)):
        t = result[i]
        max_iter = 200
        for _ in range(max_iter):
            # T-translation: shift Re(τ) into [-0.5, 0.5]
            t = t - np.round(t.real) * 1.0 + 0j  # keep complex
            t = complex(t.real - np.round(t.real), t.imag)

            # S-inversion: if |τ| < 1, apply τ → -1/τ
            if abs(t) < 1.0 - 1e-10:
                t = -1.0 / t
            else:
                break

        result[i] = t

    return result.reshape(tau.shape)


def apply_modular_transform(
    tau: np.ndarray | complex,
    transform: str,
) -> np.ndarray:
    """Apply a basic SL₂(Z) generator composition to τ in the upper half-plane."""
    tau_arr = np.asarray(tau, dtype=complex)

    if transform == 'T':
        transformed = tau_arr + 1.0
    elif transform == 'S':
        transformed = -1.0 / tau_arr
    elif transform == 'ST':
        transformed = -1.0 / (tau_arr + 1.0)
    else:
        raise ValueError(
            f"Unknown modular transform '{transform}'. Choose from {_MODULAR_TRANSFORMS}."
        )

    return transformed


def make_cyclic_modular_partners(
    tau: np.ndarray | complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Create cyclic T/S/ST modular partners for a batch of τ values.

    The transform assignment follows the fixed pattern [T, S, ST, T, S, ST, ...].

    Args:
        tau: Complex array of τ values, shape (N,) or scalar.

    Returns:
        Tuple of (transformed_tau, transform_names).
    """
    tau_arr = np.asarray(tau, dtype=complex).reshape(-1)
    transform_ids = np.arange(len(tau_arr)) % len(_MODULAR_TRANSFORMS)
    transformed = tau_arr.copy()

    for idx, name in enumerate(_MODULAR_TRANSFORMS):
        mask = transform_ids == idx
        if np.any(mask):
            transformed[mask] = apply_modular_transform(tau_arr[mask], name)

    transform_names = np.array([_MODULAR_TRANSFORMS[i] for i in transform_ids])
    return transformed.reshape(np.asarray(tau).shape), transform_names


def modular_transform_ids(
    transform_names: np.ndarray | list[str],
) -> np.ndarray:
    """Map transform names in {'T','S','ST'} to integer ids."""
    transform_arr = np.asarray(transform_names)
    if transform_arr.size == 0:
        return np.zeros(0, dtype=np.int32)

    flat = transform_arr.reshape(-1)
    ids = np.array([_MODULAR_TRANSFORM_TO_ID[str(name)] for name in flat], dtype=np.int32)
    return ids.reshape(transform_arr.shape)


def generate_lattice_theta(
    tau_values: np.ndarray,
    signal_length: int,
    t_min: float = 0.5,
    t_max: float = 5.0,
    K: int = 10,
    noise_std: float = 0.0,
    key: Optional[jax.Array] = None,
) -> jnp.ndarray:
    """Compute lattice theta function θ_Λ(t; τ) for an array of τ values.

    θ(t; τ) = Σ_{|m|,|n|≤K} exp(-π |m + nτ|² · t)

    The (0,0) term (always 1) is subtracted so the signal reflects only
    the non-trivial lattice structure.

    Args:
        tau_values: Complex array of shape (N,) with τ ∈ H.
        signal_length: Number of t sample points.
        t_min: Minimum t value.
        t_max: Maximum t value.
        K: Range for lattice index summation (|m|, |n| ≤ K).
        noise_std: Standard deviation of additive Gaussian noise.
        key: PRNG key for noise generation.

    Returns:
        Signals of shape (N, signal_length).
    """
    t_vals = np.linspace(t_min, t_max, signal_length)  # (T,)

    # Lattice indices (exclude (0,0))
    m_range = np.arange(-K, K + 1)
    n_range = np.arange(-K, K + 1)
    mm, nn = np.meshgrid(m_range, n_range, indexing='ij')  # (2K+1, 2K+1)
    mm = mm.ravel()  # ((2K+1)^2,)
    nn = nn.ravel()

    # Remove the (0,0) term
    nonzero_mask = ~((mm == 0) & (nn == 0))
    mm = mm[nonzero_mask]
    nn = nn[nonzero_mask]

    N = len(tau_values)
    tau_arr = np.asarray(tau_values)
    tau_x = tau_arr.real  # (N,)
    tau_y = tau_arr.imag  # (N,)

    # |m + n*τ|² = m² + 2*m*n*Re(τ) + n²*(Re(τ)² + Im(τ)²)
    # Shape: (N, n_lattice)
    mm_f = mm.astype(np.float64)
    nn_f = nn.astype(np.float64)

    dist_sq = (
        mm_f[None, :] ** 2
        + 2.0 * mm_f[None, :] * nn_f[None, :] * tau_x[:, None]
        + nn_f[None, :] ** 2 * (tau_x[:, None] ** 2 + tau_y[:, None] ** 2)
    )  # (N, n_lattice)

    # θ(t; τ) = Σ exp(-π · dist_sq · t)  for each (N, T) combination
    # Use numpy to compute the sum over lattice points
    # dist_sq: (N, L), t_vals: (T,) → need (N, T)
    # Efficient: for each τ, sum over lattice points for each t
    signals = np.zeros((N, signal_length), dtype=np.float64)
    for i in range(signal_length):
        # (N, L) * scalar → sum over L → (N,)
        signals[:, i] = np.sum(
            np.exp(-np.pi * dist_sq * t_vals[i]),
            axis=1,
        )

    signals = jnp.array(signals, dtype=jnp.float32)

    if noise_std > 0.0 and key is not None:
        noise = jax.random.normal(key, signals.shape) * noise_std
        signals = signals + noise

    return signals


def compute_j_invariant(tau_values: np.ndarray, n_terms: int = 50) -> np.ndarray:
    """Compute the j-invariant j(τ) using Eisenstein series q-expansion.

    j(τ) = 1728 · E₄(τ)³ / (E₄(τ)³ - E₆(τ)²)

    where E₄ = 1 + 240 Σ σ₃(n) qⁿ, E₆ = 1 - 504 Σ σ₅(n) qⁿ, q = e^{2πiτ}.

    Args:
        tau_values: Complex array of shape (N,) with τ ∈ H.
        n_terms: Number of terms in the q-expansion.

    Returns:
        Complex array of shape (N,) with j(τ) values.
    """
    # Precompute divisor sum functions σ_k(n)
    def sigma_k(n: int, k: int) -> int:
        return sum(d ** k for d in range(1, n + 1) if n % d == 0)

    sigma3 = np.array([sigma_k(n, 3) for n in range(1, n_terms + 1)])
    sigma5 = np.array([sigma_k(n, 5) for n in range(1, n_terms + 1)])

    tau_arr = np.asarray(tau_values, dtype=complex)
    q = np.exp(2j * np.pi * tau_arr)  # (N,)

    # q^n for n = 1..n_terms, shape (N, n_terms)
    ns = np.arange(1, n_terms + 1)
    q_powers = q[:, None] ** ns[None, :]  # (N, n_terms)

    # Eisenstein series
    E4 = 1.0 + 240.0 * np.sum(sigma3[None, :] * q_powers, axis=1)  # (N,)
    E6 = 1.0 - 504.0 * np.sum(sigma5[None, :] * q_powers, axis=1)  # (N,)

    # j-invariant
    E4_cubed = E4 ** 3
    delta = E4_cubed - E6 ** 2
    # Avoid division by zero (shouldn't happen for τ ∈ H)
    j_vals = 1728.0 * E4_cubed / np.where(np.abs(delta) > 1e-30, delta, 1e-30)

    return j_vals


def generate_dataset(
    config: ml_collections.ConfigDict,
    key: jax.Array,
) -> dict:
    """Generate synthetic dataset based on config.

    Args:
        config: Experiment configuration.
        key: PRNG key.

    Returns:
        Dictionary with 'signals' and parameter arrays.
        For torus: 'thetas' (N,) or (N, 2).
        For lattice: 'thetas' (N, 2) [Re(τ), Im(τ)], 'tau' (N,) complex,
                     'j_invariant' (N,) complex.
    """
    n_total = config.data.n_train + config.data.n_val + config.data.n_test

    data_type = getattr(config.data, 'data_type', 'torus')

    if data_type == 'lattice':
        return _generate_lattice_dataset(config, key, n_total)

    # --- Torus data (original logic) ---
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


def _generate_lattice_dataset(
    config: ml_collections.ConfigDict,
    key: jax.Array,
    n_total: int,
) -> dict:
    """Generate lattice theta function dataset.

    Internal helper called by generate_dataset when data_type='lattice'.
    """
    sample_region = getattr(config.data, 'lattice_sample_region',
                            'fundamental_domain')
    y_max = getattr(config.data, 'lattice_y_max', 3.0)
    K = getattr(config.data, 'lattice_K', 10)
    t_min = getattr(config.data, 'lattice_t_min', 0.5)
    t_max = getattr(config.data, 'lattice_t_max', 5.0)
    signal_length = config.data.signal_length
    noise_std = getattr(config.data, 'noise_std', 0.0)
    normalization = getattr(config.data, 'lattice_signal_normalization', 'none')

    key, sample_key = jax.random.split(key)

    if sample_region == 'fundamental_domain':
        tau = sample_fundamental_domain(n_total, y_max=y_max, key=sample_key)
    elif sample_region == 'halfplane':
        x_range = getattr(config.data, 'lattice_x_range', (-2.0, 2.0))
        y_range = getattr(config.data, 'lattice_y_range', (0.3, 4.0))
        tau = sample_upper_halfplane(n_total, x_range=x_range,
                                     y_range=y_range, key=sample_key)
    else:
        raise ValueError(f"Unknown sample_region='{sample_region}'")

    key, noise_key = jax.random.split(key)
    signals = generate_lattice_theta(
        tau, signal_length, t_min=t_min, t_max=t_max, K=K,
        noise_std=noise_std, key=noise_key,
    )
    signals = normalize_lattice_signals(signals, method=normalization)

    # Compute j-invariant for post-hoc analysis
    j_vals = compute_j_invariant(tau)

    # Store τ as (Re, Im) pair for compatibility with Dataset.thetas
    thetas = jnp.stack([jnp.array(tau.real, dtype=jnp.float32),
                        jnp.array(tau.imag, dtype=jnp.float32)], axis=-1)

    return {
        'signals': signals,
        'thetas': thetas,        # (N, 2): [Re(τ), Im(τ)]
        'tau': tau,              # (N,) complex
        'j_invariant': j_vals,   # (N,) complex
    }
