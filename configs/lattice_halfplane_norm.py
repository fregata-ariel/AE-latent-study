"""Step 2 baseline: normalized lattice data with half-plane latent."""

from configs.lattice_halfplane import get_config as _base


def get_config():
    config = _base()
    config.data.lattice_signal_normalization = 'max'
    return config
