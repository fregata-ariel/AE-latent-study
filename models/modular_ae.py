"""Modular Autoencoder with upper half-plane structured latent space."""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn

from models.encoder import Encoder
from models.decoder import Decoder
from models.layers import HalfPlaneLatent


class ModularAutoEncoder(nn.Module):
    """Autoencoder with upper half-plane latent structure.

    The latent space is constrained to H = {(x, y) : y > 0} via
    HalfPlaneLatent, reflecting the natural geometry of the moduli
    space of 2D lattices.

    Attributes:
        encoder_hidden: Hidden layer widths for the encoder.
        decoder_hidden: Hidden layer widths for the decoder.
        output_dim: Dimension of the output (signal_length).
        activation: Activation function name.
        y_min: Minimum imaginary part for the half-plane constraint.
    """
    encoder_hidden: Sequence[int]
    decoder_hidden: Sequence[int]
    output_dim: int
    activation: str = 'relu'
    y_min: float = 0.01

    def setup(self):
        self.encoder = Encoder(self.encoder_hidden, self.activation)
        self.halfplane_latent = HalfPlaneLatent(y_min=self.y_min)
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
                - latent: shape (batch, 2) as [Re(τ), Im(τ)]
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.encoder(x)
        return self.halfplane_latent(h)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)
