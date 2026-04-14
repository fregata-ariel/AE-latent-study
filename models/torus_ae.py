"""Torus-aware Autoencoder with (cos, sin) structured latent space."""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from models.encoder import Encoder
from models.decoder import Decoder
from models.layers import TorusLatent


class TorusAutoEncoder(nn.Module):
    """Autoencoder with torus-structured (cos, sin) latent space.

    The latent space is embedded as (cos(theta), sin(theta)) pairs,
    so it inherently lies on S^1 x ... x S^1.

    Attributes:
        encoder_hidden: Hidden layer widths for the encoder.
        decoder_hidden: Hidden layer widths for the decoder.
        n_angles: Number of angle components (1 for T^1, 2 for T^2).
        output_dim: Dimension of the output (signal_length).
        activation: Activation function name.
    """
    encoder_hidden: Sequence[int]
    decoder_hidden: Sequence[int]
    n_angles: int
    output_dim: int
    activation: str = 'relu'

    def setup(self):
        self.encoder = Encoder(self.encoder_hidden, self.activation)
        self.torus_latent = TorusLatent(self.n_angles)
        self.decoder = Decoder(
            self.decoder_hidden, self.output_dim, self.activation,
        )

    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Input signals, shape (batch, signal_length).

        Returns:
            Tuple of (reconstruction, latent):
                - reconstruction: shape (batch, signal_length)
                - latent: shape (batch, 2 * n_angles)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.encoder(x)
        return self.torus_latent(h)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)

    def recover_angles(self, z: jnp.ndarray) -> jnp.ndarray:
        """Extract angles from (cos, sin) latent for evaluation."""
        return TorusLatent.recover_angles(z, self.n_angles)
