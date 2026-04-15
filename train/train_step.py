"""JIT-compiled training and evaluation step functions."""

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from train.train_state import VAETrainState
from models.vae import VAE


@jax.jit
def train_step_ae(
    state: TrainState,
    batch: jnp.ndarray,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """One training step for standard AE or torus AE.

    Args:
        state: Current training state.
        batch: Batch of input signals, shape (B, signal_length).

    Returns:
        Tuple of (updated_state, metrics_dict).
    """
    def loss_fn(params):
        x_hat, z = state.apply_fn({'params': params}, batch)
        mse = jnp.mean((batch - x_hat) ** 2)
        return mse, {'loss': mse, 'mse': mse}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


def _make_train_step_lattice_invariant(weight: float):
    """Create a JIT-compiled lattice AE training step with invariance loss."""

    @jax.jit
    def train_step_lattice_invariant(
        state: TrainState,
        batch: jnp.ndarray,
        paired_batch: jnp.ndarray,
    ) -> tuple[TrainState, dict[str, jnp.ndarray]]:
        def loss_fn(params):
            x_hat, z = state.apply_fn({'params': params}, batch)
            _, z_pair = state.apply_fn({'params': params}, paired_batch)
            mse = jnp.mean((batch - x_hat) ** 2)
            inv_loss = jnp.mean(jnp.sum((z - z_pair) ** 2, axis=-1))
            loss = mse + weight * inv_loss
            return loss, {'loss': loss, 'mse': mse, 'inv_loss': inv_loss}

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    return train_step_lattice_invariant


def _make_train_step_vae(beta: float):
    """Create a JIT-compiled VAE training step with fixed beta."""

    @jax.jit
    def train_step_vae(
        state: VAETrainState,
        batch: jnp.ndarray,
    ) -> tuple[VAETrainState, dict[str, jnp.ndarray]]:
        """One training step for VAE.

        Args:
            state: Current VAE training state (includes rng).
            batch: Batch of input signals, shape (B, signal_length).

        Returns:
            Tuple of (updated_state, metrics_dict).
        """
        rng, step_rng = jax.random.split(state.rng)

        def loss_fn(params):
            x_hat, z, mean, logvar = state.apply_fn(
                {'params': params}, batch, step_rng,
            )
            mse = jnp.mean((batch - x_hat) ** 2)
            kl = jnp.mean(VAE.kl_divergence(mean, logvar))
            loss = mse + beta * kl
            return loss, {'loss': loss, 'mse': mse, 'kl': kl}

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(rng=rng)
        return state, metrics

    return train_step_vae


def _make_train_step_lattice_invariant_vae(beta: float, weight: float):
    """Create a JIT-compiled lattice VAE training step with invariance loss."""

    @jax.jit
    def train_step_vae_lattice_invariant(
        state: VAETrainState,
        batch: jnp.ndarray,
        paired_batch: jnp.ndarray,
    ) -> tuple[VAETrainState, dict[str, jnp.ndarray]]:
        rng, step_rng = jax.random.split(state.rng)

        def loss_fn(params):
            x_hat, _, mean, logvar = state.apply_fn(
                {'params': params}, batch, step_rng,
            )
            _, _, mean_pair, _ = state.apply_fn(
                {'params': params}, paired_batch, step_rng, deterministic=True,
            )
            mse = jnp.mean((batch - x_hat) ** 2)
            kl = jnp.mean(VAE.kl_divergence(mean, logvar))
            inv_loss = jnp.mean(jnp.sum((mean - mean_pair) ** 2, axis=-1))
            loss = mse + beta * kl + weight * inv_loss
            return loss, {'loss': loss, 'mse': mse, 'kl': kl, 'inv_loss': inv_loss}

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(rng=rng)
        return state, metrics

    return train_step_vae_lattice_invariant


def _make_eval_step_lattice_invariant(weight: float):
    """Create a JIT-compiled lattice AE eval step with invariance loss."""

    @jax.jit
    def eval_step_lattice_invariant(
        state: TrainState,
        batch: jnp.ndarray,
        paired_batch: jnp.ndarray,
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        x_hat, z = state.apply_fn({'params': state.params}, batch)
        _, z_pair = state.apply_fn({'params': state.params}, paired_batch)
        mse = jnp.mean((batch - x_hat) ** 2)
        inv_loss = jnp.mean(jnp.sum((z - z_pair) ** 2, axis=-1))
        loss = mse + weight * inv_loss
        return {'mse': mse, 'loss': loss, 'inv_loss': inv_loss}, z

    return eval_step_lattice_invariant


def _make_eval_step_lattice_invariant_vae(beta: float, weight: float):
    """Create a JIT-compiled lattice VAE eval step with invariance loss."""

    @jax.jit
    def eval_step_lattice_invariant_vae(
        state: VAETrainState,
        batch: jnp.ndarray,
        paired_batch: jnp.ndarray,
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        x_hat, z, mean, logvar = state.apply_fn(
            {'params': state.params}, batch, state.rng, deterministic=True,
        )
        _, _, mean_pair, _ = state.apply_fn(
            {'params': state.params}, paired_batch, state.rng, deterministic=True,
        )
        mse = jnp.mean((batch - x_hat) ** 2)
        kl = jnp.mean(VAE.kl_divergence(mean, logvar))
        inv_loss = jnp.mean(jnp.sum((mean - mean_pair) ** 2, axis=-1))
        loss = mse + beta * kl + weight * inv_loss
        return {'mse': mse, 'loss': loss, 'kl': kl, 'inv_loss': inv_loss}, z

    return eval_step_lattice_invariant_vae


@jax.jit
def eval_step_ae(
    state: TrainState,
    batch: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Evaluate a batch for standard AE or torus AE (no gradients).

    Args:
        state: Training state.
        batch: Batch of input signals.

    Returns:
        Tuple of (metrics_dict, latent_codes).
    """
    x_hat, z = state.apply_fn({'params': state.params}, batch)
    mse = jnp.mean((batch - x_hat) ** 2)
    return {'mse': mse, 'loss': mse}, z


@jax.jit
def eval_step_vae(
    state: VAETrainState,
    batch: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Evaluate a batch for VAE (deterministic, no sampling).

    Args:
        state: VAE training state.
        batch: Batch of input signals.

    Returns:
        Tuple of (metrics_dict, latent_codes).
    """
    x_hat, z, mean, logvar = state.apply_fn(
        {'params': state.params}, batch, state.rng, deterministic=True,
    )
    mse = jnp.mean((batch - x_hat) ** 2)
    return {'mse': mse, 'loss': mse}, z
