"""T^1 data with standard R^2 latent autoencoder."""

from configs.default import get_config as _default


def get_config():
    config = _default()
    config.data.torus_dim = 1
    config.model.latent_type = 'standard'
    config.model.latent_dim = 2
    return config
