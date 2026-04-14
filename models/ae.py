"""Standard Autoencoder with unconstrained R^d latent space."""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from models.encoder import Encoder
from models.decoder import Decoder


class AutoEncoder(nn.Module):
    """Standard autoencoder with R^d latent space.

    Attributes:
        encoder_hidden: Hidden layer widths for the encoder.
        decoder_hidden: Hidden layer widths for the decoder.
        latent_dim: Dimension of the latent space.
        output_dim: Dimension of the output (signal_length).
        activation: Activation function name.
    """
    encoder_hidden: Sequence[int]
    decoder_hidden: Sequence[int]
    latent_dim: int
    output_dim: int
    activation: str = 'relu'

    def setup(self):
        self.encoder = Encoder(self.encoder_hidden, self.activation)
        self.latent_proj = nn.Dense(self.latent_dim)
        self.decoder = Decoder(self.decoder_hidden, self.output_dim, self.activation)

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Input signals, shape (batch, signal_length).

        Returns:
            Tuple of (reconstruction, latent):
                - reconstruction: shape (batch, signal_length)
                - latent: shape (batch, latent_dim)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.encoder(x)
        return self.latent_proj(h)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)
