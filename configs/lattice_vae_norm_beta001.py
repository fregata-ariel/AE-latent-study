"""Step 2 capacity sweep: normalized lattice data with a small-beta VAE."""

from configs.lattice_standard_norm import get_config as _base


def get_config():
    config = _base()
    config.model.latent_type = 'vae'
    config.model.latent_dim = 4
    config.model.vae_beta = 0.01
    return config
