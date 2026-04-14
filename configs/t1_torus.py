"""T^1 data with torus-aware (cos, sin) latent autoencoder."""

from configs.default import get_config as _default


def get_config():
    config = _default()
    config.data.torus_dim = 1
    config.model.latent_type = 'torus'
    config.model.latent_dim = 1  # 1 angle -> 2D (cos, sin) embedding
    return config
