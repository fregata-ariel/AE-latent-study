"""Lattice data with standard R^2 latent (wide upper half-plane sampling).

This experiment samples τ from a wide region of H (not just F),
testing whether the AE naturally folds SL₂(Z)-equivalent τ values
into the same latent representation.
"""

from configs.lattice_default import get_config as _default


def get_config():
    config = _default()
    config.model.latent_type = 'standard'
    config.model.latent_dim = 2
    config.data.lattice_sample_region = 'halfplane'
    config.data.lattice_x_range = (-2.0, 2.0)
    config.data.lattice_y_range = (0.3, 4.0)
    return config
