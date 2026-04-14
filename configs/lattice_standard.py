"""Lattice data with standard R^2 latent autoencoder (fundamental domain sampling)."""

from configs.lattice_default import get_config as _default


def get_config():
    config = _default()
    config.model.latent_type = 'standard'
    config.model.latent_dim = 2
    config.data.lattice_sample_region = 'fundamental_domain'
    return config
