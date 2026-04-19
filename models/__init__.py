"""Model factory."""

import ml_collections
from flax import linen as nn

from models.ae import AutoEncoder
from models.factorized_vae import FactorizedVAE
from models.torus_ae import TorusAutoEncoder
from models.modular_ae import ModularAutoEncoder
from models.vae import VAE


def create_model(config: ml_collections.ConfigDict) -> nn.Module:
    """Create a model based on configuration.

    Args:
        config: Experiment configuration.

    Returns:
        A Flax nn.Module instance.
    """
    latent_type = config.model.latent_type
    signal_length = config.data.signal_length
    encoder_hidden = tuple(config.model.encoder_hidden)
    decoder_hidden = tuple(config.model.decoder_hidden)
    activation = config.model.activation

    if latent_type == 'standard':
        return AutoEncoder(
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            latent_dim=config.model.latent_dim,
            output_dim=signal_length,
            activation=activation,
        )
    elif latent_type == 'torus':
        return TorusAutoEncoder(
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            n_angles=config.model.latent_dim,
            output_dim=signal_length,
            activation=activation,
        )
    elif latent_type == 'halfplane':
        return ModularAutoEncoder(
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            output_dim=signal_length,
            activation=activation,
        )
    elif latent_type == 'vae':
        return VAE(
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            latent_dim=config.model.latent_dim,
            output_dim=signal_length,
            activation=activation,
        )
    elif latent_type == 'factorized_vae':
        return FactorizedVAE(
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            quotient_dim=config.model.quotient_dim,
            gauge_dim=config.model.gauge_dim,
            output_dim=signal_length,
            activation=activation,
            gauge_action_type=config.model.gauge_action_type,
        )
    else:
        raise ValueError(
            f"Unknown latent_type '{latent_type}'. "
            "Choose from 'standard', 'torus', 'halfplane', 'vae', 'factorized_vae'."
        )
