"""T^2 data with standard R^d latent autoencoder."""

from configs.default import get_config as _default


def get_config():
    config = _default()
    config.data.torus_dim = 2
    config.model.latent_type = 'standard'
    config.model.latent_dim = 4
    return config
