"""T^1 data with VAE (Gaussian prior)."""

from configs.default import get_config as _default


def get_config():
    config = _default()
    config.data.torus_dim = 1
    config.model.latent_type = 'vae'
    config.model.latent_dim = 2
    config.model.vae_beta = 0.1  # Small beta to avoid posterior collapse
    return config
