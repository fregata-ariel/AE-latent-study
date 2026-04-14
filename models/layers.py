"""Reusable building blocks: MLP and TorusLatent modules."""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn


_ACTIVATIONS = {
    'relu': nn.relu,
    'tanh': jnp.tanh,
    'gelu': nn.gelu,
}


def get_activation(name: str):
    """Look up an activation function by name."""
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation.

    Attributes:
        hidden_dims: Sequence of hidden layer widths.
        activation: Name of the activation function.
    """
    hidden_dims: Sequence[int]
    activation: str = 'relu'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act_fn = get_activation(self.activation)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = act_fn(x)
        return x


class TorusLatent(nn.Module):
    """Project hidden representation to (cos, sin) pairs for torus structure.

    Maps the encoder's output through a Dense layer to produce raw angles,
    then applies cos/sin to create a latent representation that inherently
    lives on S^1 x ... x S^1.

    Attributes:
        n_angles: Number of angle components (1 for T^1, 2 for T^2).
    """
    n_angles: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map hidden representation to torus-structured latent.

        Args:
            x: Encoder output, shape (batch, hidden_dim).

        Returns:
            Latent of shape (batch, 2 * n_angles) laid out as
            [cos_1, ..., cos_n, sin_1, ..., sin_n].
        """
        raw_angles = nn.Dense(self.n_angles)(x)  # (batch, n_angles)
        cos_vals = jnp.cos(raw_angles)
        sin_vals = jnp.sin(raw_angles)
        return jnp.concatenate([cos_vals, sin_vals], axis=-1)

    @staticmethod
    def recover_angles(latent: jnp.ndarray, n_angles: int) -> jnp.ndarray:
        """Extract angles from (cos, sin) latent representation.

        Args:
            latent: Shape (..., 2 * n_angles).
            n_angles: Number of angle components.

        Returns:
            Angles in (-pi, pi], shape (..., n_angles).
        """
        cos_vals = latent[..., :n_angles]
        sin_vals = latent[..., n_angles:]
        return jnp.arctan2(sin_vals, cos_vals)


class HalfPlaneLatent(nn.Module):
    """Project hidden representation to the upper half-plane H = {(x,y) : y > 0}.

    Maps the encoder's output through a Dense layer to produce a 2D latent
    where the second coordinate (imaginary part of τ) is constrained to be
    positive via softplus. This is the minimal geometric constraint for
    learning moduli space structure.

    Attributes:
        y_min: Minimum value for the imaginary part (default: 0.01).
    """
    y_min: float = 0.01

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Map hidden representation to upper half-plane point.

        Args:
            x: Encoder output, shape (batch, hidden_dim).

        Returns:
            Latent of shape (batch, 2) as [Re(τ), Im(τ)] with Im(τ) > y_min.
        """
        raw = nn.Dense(2)(x)  # (batch, 2)
        real_part = raw[:, 0:1]                           # Re(τ), unconstrained
        imag_part = nn.softplus(raw[:, 1:2]) + self.y_min  # Im(τ) > y_min
        return jnp.concatenate([real_part, imag_part], axis=-1)
