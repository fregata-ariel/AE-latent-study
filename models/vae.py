"""Variational Autoencoder with Gaussian prior on R^d latent."""

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from models.encoder import Encoder
from models.decoder import Decoder


class VAE(nn.Module):
    """Variational Autoencoder with diagonal Gaussian posterior and prior.

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
        self.mean_proj = nn.Dense(self.latent_dim)
        self.logvar_proj = nn.Dense(self.latent_dim)
        self.decoder = Decoder(self.decoder_hidden, self.output_dim, self.activation)

    def __call__(
        self,
        x: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Input signals, shape (batch, signal_length).
            key: PRNG key for reparameterization sampling.
            deterministic: If True, use mean as latent (no sampling).

        Returns:
            Tuple of (reconstruction, z, mean, logvar).
        """
        h = self.encoder(x)
        mean = self.mean_proj(h)
        logvar = self.logvar_proj(h)

        if deterministic:
            z = mean
        else:
            z = self._reparameterize(key, mean, logvar)

        x_hat = self.decoder(z)
        return x_hat, z, mean, logvar

    @staticmethod
    def _reparameterize(
        key: jax.Array,
        mean: jnp.ndarray,
        logvar: jnp.ndarray,
    ) -> jnp.ndarray:
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mean.shape)
        return mean + eps * std

    @staticmethod
    def kl_divergence(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        """KL(q(z|x) || N(0, I)), per-sample.

        Args:
            mean: Posterior mean, shape (batch, latent_dim).
            logvar: Log variance, shape (batch, latent_dim).

        Returns:
            KL divergence per sample, shape (batch,).
        """
        return -0.5 * jnp.sum(
            1.0 + logvar - mean ** 2 - jnp.exp(logvar), axis=-1
        )

    def encode(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Encode to posterior parameters (mean, logvar)."""
        h = self.encoder(x)
        return self.mean_proj(h), self.logvar_proj(h)

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)
