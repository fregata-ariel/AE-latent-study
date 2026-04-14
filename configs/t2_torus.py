"""T^2 data with torus-aware (cos, sin) latent autoencoder."""

from configs.default import get_config as _default


def get_config():
    config = _default()
    config.data.torus_dim = 2
    config.model.latent_type = 'torus'
    config.model.latent_dim = 2  # 2 angles -> 4D (cos1, cos2, sin1, sin2)
    return config
