"""TrainState creation and helpers."""

import jax
import jax.numpy as jnp
import optax
import ml_collections
from flax import linen as nn
from flax.training.train_state import TrainState


class VAETrainState(TrainState):
    """TrainState extended with a PRNG key for VAE sampling."""
    rng: jax.Array


def create_train_state(
    config: ml_collections.ConfigDict,
    model: nn.Module,
    key: jax.Array,
) -> TrainState:
    """Initialize model parameters, optimizer, and wrap in TrainState.

    Args:
        config: Experiment configuration.
        model: Flax model instance.
        key: PRNG key for parameter initialization.

    Returns:
        Initialized TrainState (or VAETrainState for VAE).
    """
    init_key, rng_key = jax.random.split(key)

    # Initialize with dummy input
    dummy_input = jnp.ones((1, config.data.signal_length))

    if config.model.latent_type in ('vae', 'factorized_vae'):
        variables = model.init(init_key, dummy_input, rng_key)
    else:
        variables = model.init(init_key, dummy_input)

    params = variables['params']

    # Build optimizer
    lr = config.train.learning_rate
    if config.train.lr_schedule == 'cosine':
        steps_per_epoch = config.data.n_train // config.train.batch_size
        total_steps = steps_per_epoch * config.train.num_epochs
        schedule = optax.cosine_decay_schedule(lr, total_steps)
        lr_or_schedule = schedule
    else:
        lr_or_schedule = lr

    if config.train.weight_decay > 0.0:
        tx = optax.adamw(lr_or_schedule, weight_decay=config.train.weight_decay)
    else:
        tx = optax.adam(lr_or_schedule)

    if config.model.latent_type in ('vae', 'factorized_vae'):
        return VAETrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            rng=rng_key,
        )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
