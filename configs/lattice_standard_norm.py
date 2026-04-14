"""Step 2 baseline: normalized lattice data with standard R^2 latent."""

from configs.lattice_standard import get_config as _base


def get_config():
    config = _base()
    config.data.lattice_signal_normalization = 'max'
    return config
