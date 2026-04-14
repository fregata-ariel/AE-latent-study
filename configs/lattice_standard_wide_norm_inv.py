"""Step 2 regularized: normalized wide-half-plane lattice data with invariance loss."""

from configs.lattice_standard_wide_norm import get_config as _base


def get_config():
    config = _base()
    config.train.modular_invariance_weight = 0.1
    return config
