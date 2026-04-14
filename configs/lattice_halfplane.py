"""Lattice data with upper half-plane constrained latent (fundamental domain sampling)."""

from configs.lattice_default import get_config as _default


def get_config():
    config = _default()
    config.model.latent_type = 'halfplane'
    config.model.latent_dim = 2  # always 2 for halfplane
    config.data.lattice_sample_region = 'fundamental_domain'
    return config
