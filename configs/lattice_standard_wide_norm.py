"""Step 2 baseline: normalized wide-half-plane lattice data with standard latent."""

from configs.lattice_standard_wide import get_config as _base


def get_config():
    config = _base()
    config.data.lattice_signal_normalization = 'max'
    return config
