"""Tests for model definitions and shapes."""

import jax
import jax.numpy as jnp
import numpy as np

from configs.default import get_config
from models import create_model
from models.layers import MLP, TorusLatent
from models.ae import AutoEncoder
from models.factorized_vae import FactorizedVAE
from models.torus_ae import TorusAutoEncoder
from models.vae import VAE


def test_mlp_shape():
    model = MLP(hidden_dims=(32, 16), activation='relu')
    key = jax.random.PRNGKey(0)
    x = jnp.ones((4, 100))
    variables = model.init(key, x)
    out = model.apply(variables, x)
    assert out.shape == (4, 16)


def test_torus_latent_shape():
    model = TorusLatent(n_angles=2)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((4, 64))
    variables = model.init(key, x)
    out = model.apply(variables, x)
    assert out.shape == (4, 4)  # 2 angles -> 4D (cos1, cos2, sin1, sin2)


def test_torus_latent_unit_circle():
    """Each (cos, sin) pair should have norm 1."""
    model = TorusLatent(n_angles=2)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 64))
    variables = model.init(key, x)
    out = model.apply(variables, x)

    cos_vals = out[:, :2]
    sin_vals = out[:, 2:]
    norms = jnp.sqrt(cos_vals ** 2 + sin_vals ** 2)
    np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-6)


def test_torus_recover_angles():
    angles = jnp.array([[0.5, 1.5], [-1.0, 2.0]])
    cos_vals = jnp.cos(angles)
    sin_vals = jnp.sin(angles)
    latent = jnp.concatenate([cos_vals, sin_vals], axis=-1)
    recovered = TorusLatent.recover_angles(latent, n_angles=2)
    np.testing.assert_allclose(np.array(recovered), np.array(angles), atol=1e-6)


def test_ae_shapes():
    config = get_config()
    config.model.latent_type = 'standard'
    config.model.latent_dim = 2
    config.model.encoder_hidden = (32, 16)
    config.model.decoder_hidden = (16, 32)

    model = create_model(config)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((4, config.data.signal_length))
    variables = model.init(key, x)
    x_hat, z = model.apply(variables, x)

    assert x_hat.shape == (4, config.data.signal_length)
    assert z.shape == (4, 2)


def test_torus_ae_shapes():
    config = get_config()
    config.model.latent_type = 'torus'
    config.model.latent_dim = 1  # 1 angle
    config.model.encoder_hidden = (32, 16)
    config.model.decoder_hidden = (16, 32)

    model = create_model(config)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((4, config.data.signal_length))
    variables = model.init(key, x)
    x_hat, z = model.apply(variables, x)

    assert x_hat.shape == (4, config.data.signal_length)
    assert z.shape == (4, 2)  # 1 angle -> (cos, sin)


def test_torus_ae_t2_shapes():
    config = get_config()
    config.model.latent_type = 'torus'
    config.model.latent_dim = 2  # 2 angles
    config.model.encoder_hidden = (32, 16)
    config.model.decoder_hidden = (16, 32)

    model = create_model(config)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((4, config.data.signal_length))
    variables = model.init(key, x)
    x_hat, z = model.apply(variables, x)

    assert x_hat.shape == (4, config.data.signal_length)
    assert z.shape == (4, 4)  # 2 angles -> 4D


def test_vae_shapes():
    config = get_config()
    config.model.latent_type = 'vae'
    config.model.latent_dim = 2
    config.model.encoder_hidden = (32, 16)
    config.model.decoder_hidden = (16, 32)

    model = create_model(config)
    key = jax.random.PRNGKey(0)
    init_key, sample_key = jax.random.split(key)
    x = jnp.ones((4, config.data.signal_length))
    variables = model.init(init_key, x, sample_key)
    x_hat, z, mean, logvar = model.apply(variables, x, sample_key)

    assert x_hat.shape == (4, config.data.signal_length)
    assert z.shape == (4, 2)
    assert mean.shape == (4, 2)
    assert logvar.shape == (4, 2)


def test_vae_deterministic():
    model = VAE(
        encoder_hidden=(32, 16),
        decoder_hidden=(16, 32),
        latent_dim=2,
        output_dim=100,
    )
    key = jax.random.PRNGKey(0)
    init_key, sample_key = jax.random.split(key)
    x = jnp.ones((4, 100))
    variables = model.init(init_key, x, sample_key)

    x_hat1, z1, m1, _ = model.apply(variables, x, sample_key, deterministic=True)
    x_hat2, z2, m2, _ = model.apply(variables, x, sample_key, deterministic=True)
    np.testing.assert_array_equal(np.array(z1), np.array(m1))
    np.testing.assert_array_equal(np.array(x_hat1), np.array(x_hat2))


def test_factorized_vae_shapes():
    config = get_config()
    config.model.latent_type = 'factorized_vae'
    config.model.latent_dim = 6
    config.model.quotient_dim = 2
    config.model.gauge_dim = 4
    config.model.encoder_hidden = (32, 16)
    config.model.decoder_hidden = (16, 32)

    model = create_model(config)
    key = jax.random.PRNGKey(0)
    init_key, sample_key = jax.random.split(key)
    x = jnp.ones((4, config.data.signal_length))
    variables = model.init(init_key, x, sample_key)
    x_hat, z, q_mean, q_logvar, g_mean, g_logvar = model.apply(
        variables, x, sample_key,
    )

    assert x_hat.shape == (4, config.data.signal_length)
    assert z.shape == (4, 6)
    assert q_mean.shape == (4, 2)
    assert q_logvar.shape == (4, 2)
    assert g_mean.shape == (4, 4)
    assert g_logvar.shape == (4, 4)


def test_factorized_vae_gauge_action_composes_st():
    model = FactorizedVAE(
        encoder_hidden=(32, 16),
        decoder_hidden=(16, 32),
        quotient_dim=2,
        gauge_dim=4,
        output_dim=100,
    )
    key = jax.random.PRNGKey(0)
    init_key, sample_key = jax.random.split(key)
    x = jnp.ones((2, 100))
    variables = model.init(init_key, x, sample_key)
    gauge = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

    acted_t = model.apply(
        variables, gauge, jnp.array([0, 0]), method=model.apply_gauge_action,
    )
    acted_st = model.apply(
        variables, gauge, jnp.array([2, 2]), method=model.apply_gauge_action,
    )
    manual_st = model.apply(
        variables, acted_t, jnp.array([1, 1]), method=model.apply_gauge_action,
    )

    np.testing.assert_allclose(np.array(acted_st), np.array(manual_st), atol=1e-6)


def test_vae_kl_divergence():
    mean = jnp.zeros((4, 2))
    logvar = jnp.zeros((4, 2))
    kl = VAE.kl_divergence(mean, logvar)
    # KL(N(0,I) || N(0,I)) = 0
    np.testing.assert_allclose(np.array(kl), 0.0, atol=1e-6)


def test_ae_encode_decode_roundtrip():
    model = AutoEncoder(
        encoder_hidden=(32, 16),
        decoder_hidden=(16, 32),
        latent_dim=2,
        output_dim=100,
    )
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 100))
    variables = model.init(key, x)

    z = model.apply(variables, x, method=model.encode)
    x_hat = model.apply(variables, z, method=model.decode)
    assert z.shape == (4, 2)
    assert x_hat.shape == (4, 100)
