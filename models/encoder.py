"""MLP Encoder module."""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from models.layers import MLP


class Encoder(nn.Module):
    """MLP encoder mapping signals to a hidden representation.

    Does NOT include the final latent projection; the wrapping AE/VAE
    module applies the appropriate latent layer (Dense or TorusLatent).

    Attributes:
        hidden_dims: Sequence of hidden layer widths.
        activation: Activation function name.
    """
    hidden_dims: Sequence[int]
    activation: str = 'relu'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode input signals to hidden representation.

        Args:
            x: Input signals, shape (batch, signal_length).

        Returns:
            Hidden representation, shape (batch, hidden_dims[-1]).
        """
        return MLP(self.hidden_dims, self.activation)(x)
