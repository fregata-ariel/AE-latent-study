"""Factorized VAE with quotient and gauge latent parts for lattice data."""

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from models.decoder import Decoder
from models.encoder import Encoder


_TRANSFORM_T = 0
_TRANSFORM_S = 1
_TRANSFORM_ST = 2


class AffineGaugeAction(nn.Module):
    """Learn affine gauge actions for T and S generators."""
    gauge_dim: int
    init_scale: float = 1e-2

    def setup(self):
        eye = jnp.eye(self.gauge_dim)

        def matrix_init(key, shape):
            noise = jax.random.normal(key, shape) * self.init_scale
            return eye + noise

        self.A_T = self.param('A_T', matrix_init, (self.gauge_dim, self.gauge_dim))
        self.b_T = self.param('b_T', nn.initializers.zeros_init(), (self.gauge_dim,))
        self.A_S = self.param('A_S', matrix_init, (self.gauge_dim, self.gauge_dim))
        self.b_S = self.param('b_S', nn.initializers.zeros_init(), (self.gauge_dim,))

    def _apply_affine(
        self,
        gauge: jnp.ndarray,
        matrix: jnp.ndarray,
        bias: jnp.ndarray,
    ) -> jnp.ndarray:
        return gauge @ matrix.T + bias

    def _apply_T(self, gauge: jnp.ndarray) -> jnp.ndarray:
        return self._apply_affine(gauge, self.A_T, self.b_T)

    def _apply_S(self, gauge: jnp.ndarray) -> jnp.ndarray:
        return self._apply_affine(gauge, self.A_S, self.b_S)

    def __call__(
        self,
        gauge: jnp.ndarray,
        transform_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply T/S/ST gauge action to a batch of latent gauge vectors."""
        transform_ids = jnp.asarray(transform_ids, dtype=jnp.int32).reshape(-1)
        gauge_T = self._apply_T(gauge)
        gauge_S = self._apply_S(gauge)
        gauge_ST = self._apply_S(gauge_T)

        transformed = gauge
        transformed = jnp.where(
            (transform_ids == _TRANSFORM_T)[:, None], gauge_T, transformed,
        )
        transformed = jnp.where(
            (transform_ids == _TRANSFORM_S)[:, None], gauge_S, transformed,
        )
        transformed = jnp.where(
            (transform_ids == _TRANSFORM_ST)[:, None], gauge_ST, transformed,
        )
        return transformed

    def regularizer(self) -> jnp.ndarray:
        """Penalize deviation from the identity / zero initialization."""
        eye = jnp.eye(self.gauge_dim)
        reg_T = jnp.sum((self.A_T - eye) ** 2) + jnp.sum(self.b_T ** 2)
        reg_S = jnp.sum((self.A_S - eye) ** 2) + jnp.sum(self.b_S ** 2)
        return reg_T + reg_S


class FactorizedVAE(nn.Module):
    """Variational autoencoder with factorized quotient and gauge latent parts."""
    encoder_hidden: Sequence[int]
    decoder_hidden: Sequence[int]
    quotient_dim: int
    gauge_dim: int
    output_dim: int
    activation: str = 'relu'
    gauge_action_type: str = 'affine'

    def setup(self):
        if self.gauge_action_type != 'affine':
            raise ValueError(
                f"Unsupported gauge_action_type '{self.gauge_action_type}'. "
                "Only 'affine' is implemented."
            )

        self.encoder = Encoder(self.encoder_hidden, self.activation)
        self.q_mean_proj = nn.Dense(self.quotient_dim)
        self.q_logvar_proj = nn.Dense(self.quotient_dim)
        self.g_mean_proj = nn.Dense(self.gauge_dim)
        self.g_logvar_proj = nn.Dense(self.gauge_dim)
        self.decoder = Decoder(self.decoder_hidden, self.output_dim, self.activation)
        self.gauge_action = AffineGaugeAction(self.gauge_dim)

    def encode_parts(
        self,
        x: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Encode input signals into quotient/gauge posterior parameters."""
        h = self.encoder(x)
        q_mean = self.q_mean_proj(h)
        q_logvar = self.q_logvar_proj(h)
        g_mean = self.g_mean_proj(h)
        g_logvar = self.g_logvar_proj(h)
        return q_mean, q_logvar, g_mean, g_logvar

    @staticmethod
    def _reparameterize(
        key: jax.Array,
        mean: jnp.ndarray,
        logvar: jnp.ndarray,
    ) -> jnp.ndarray:
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mean.shape)
        return mean + eps * std

    def sample_parts(
        self,
        q_mean: jnp.ndarray,
        q_logvar: jnp.ndarray,
        g_mean: jnp.ndarray,
        g_logvar: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample or deterministically select quotient/gauge latent parts."""
        if deterministic:
            return q_mean, g_mean

        q_key, g_key = jax.random.split(key)
        q = self._reparameterize(q_key, q_mean, q_logvar)
        g = self._reparameterize(g_key, g_mean, g_logvar)
        return q, g

    def decode_parts(
        self,
        quotient: jnp.ndarray,
        gauge: jnp.ndarray,
    ) -> jnp.ndarray:
        """Decode from explicit quotient and gauge parts."""
        return self.decoder(jnp.concatenate([quotient, gauge], axis=-1))

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)

    def apply_gauge_action(
        self,
        gauge: jnp.ndarray,
        transform_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        """Apply the learned gauge action associated with T, S, or ST."""
        return self.gauge_action(gauge, transform_ids)

    def action_regularizer(self) -> jnp.ndarray:
        """Regularization for the learned affine gauge action."""
        return self.gauge_action.regularizer()

    def __call__(
        self,
        x: jnp.ndarray,
        key: jax.Array,
        deterministic: bool = False,
    ) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        """Forward pass returning reconstruction and factorized posterior pieces."""
        # Touch the gauge-action parameters during the standard forward path so
        # they are initialized alongside the encoder/decoder weights.
        _ = self.gauge_action.regularizer()
        q_mean, q_logvar, g_mean, g_logvar = self.encode_parts(x)
        q, g = self.sample_parts(
            q_mean, q_logvar, g_mean, g_logvar, key, deterministic=deterministic,
        )
        z = jnp.concatenate([q, g], axis=-1)
        x_hat = self.decode(z)
        return x_hat, z, q_mean, q_logvar, g_mean, g_logvar

    @staticmethod
    def kl_divergence(
        mean: jnp.ndarray,
        logvar: jnp.ndarray,
    ) -> jnp.ndarray:
        """KL(q(z|x) || N(0, I)), per-sample."""
        return -0.5 * jnp.sum(
            1.0 + logvar - mean ** 2 - jnp.exp(logvar), axis=-1,
        )
