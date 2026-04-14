"""Step 2 capacity sweep: normalized lattice data with 4D standard latent."""

from configs.lattice_standard_norm import get_config as _base


def get_config():
    config = _base()
    config.model.latent_dim = 4
    return config
