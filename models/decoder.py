"""MLP Decoder module."""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from models.layers import MLP


class Decoder(nn.Module):
    """MLP decoder mapping latent representation back to signal space.

    Attributes:
        hidden_dims: Sequence of hidden layer widths.
        output_dim: Dimension of the output (signal_length).
        activation: Activation function name.
    """
    hidden_dims: Sequence[int]
    output_dim: int
    activation: str = 'relu'

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Decode latent representation to reconstructed signal.

        Args:
            z: Latent representation, shape (batch, latent_dim).

        Returns:
            Reconstructed signal, shape (batch, output_dim).
        """
        h = MLP(self.hidden_dims, self.activation)(z)
        return nn.Dense(self.output_dim)(h)
