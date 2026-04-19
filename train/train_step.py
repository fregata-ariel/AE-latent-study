"""JIT-compiled training and evaluation step functions."""

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from models.factorized_vae import FactorizedVAE
from train.train_state import VAETrainState
from models.vae import VAE


def _pairwise_l2(points: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise Euclidean distances for a point cloud."""
    diffs = points[:, None, :] - points[None, :, :]
    return jnp.sqrt(jnp.sum(diffs ** 2, axis=-1) + 1e-12)


def _covariance_matrix(points: jnp.ndarray) -> jnp.ndarray:
    """Compute a stable sample covariance for a point cloud."""
    n_points = points.shape[0]
    n_dims = points.shape[-1]
    if n_points <= 1:
        return jnp.zeros((n_dims, n_dims), dtype=points.dtype)

    centered = points - jnp.mean(points, axis=0, keepdims=True)
    denom = jnp.maximum(n_points - 1, 1)
    return (centered.T @ centered) / denom


def _quotient_chart_loss(
    tau_fd_coords: jnp.ndarray,
    quotient_mean: jnp.ndarray,
    n_neighbors: int,
) -> jnp.ndarray:
    """Match normalized local τ_F distances and quotient distances."""
    n_points = tau_fd_coords.shape[0]
    if n_points <= 1:
        return jnp.asarray(0.0, dtype=quotient_mean.dtype)

    k_use = min(max(1, int(n_neighbors)), max(1, n_points - 1))
    tau_dists = _pairwise_l2(tau_fd_coords)
    quotient_dists = _pairwise_l2(quotient_mean)
    tau_dists_for_knn = jnp.where(
        jnp.eye(n_points, dtype=bool),
        jnp.inf,
        tau_dists,
    )
    knn_idx = jnp.argsort(tau_dists_for_knn, axis=1)[:, :k_use]

    tau_knn = jnp.take_along_axis(tau_dists, knn_idx, axis=1)
    quotient_knn = jnp.take_along_axis(quotient_dists, knn_idx, axis=1)

    tau_scale = jnp.maximum(jnp.mean(tau_knn, axis=1, keepdims=True), 1e-6)
    quotient_scale = jnp.maximum(
        jnp.mean(quotient_knn, axis=1, keepdims=True), 1e-6,
    )
    return jnp.mean(((tau_knn / tau_scale) - (quotient_knn / quotient_scale)) ** 2)


def _quotient_variance_floor_loss(
    quotient_mean: jnp.ndarray,
    target: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Encourage both quotient dimensions to maintain a minimum variance."""
    variances = jnp.var(quotient_mean, axis=0)
    loss = jnp.mean(jax.nn.relu(target - variances) ** 2)
    return loss, variances


def _quotient_spread_loss(
    tau_fd_coords: jnp.ndarray,
    quotient_mean: jnp.ndarray,
    min_eig_ratio_target: float,
    trace_cap_ratio: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Encourage quotient spread using rotation-aware covariance metrics."""
    if tau_fd_coords.shape[0] <= 1 or quotient_mean.shape[0] <= 1:
        zero = jnp.asarray(0.0, dtype=quotient_mean.dtype)
        eigs = jnp.zeros((2,), dtype=quotient_mean.dtype)
        return zero, eigs, eigs, zero

    tau_cov = _covariance_matrix(tau_fd_coords)
    quotient_cov = _covariance_matrix(quotient_mean)
    tau_eigs = jnp.maximum(jnp.linalg.eigvalsh(tau_cov), 0.0)
    quotient_eigs = jnp.maximum(jnp.linalg.eigvalsh(quotient_cov), 0.0)

    tau_min = tau_eigs[0]
    tau_trace = jnp.sum(tau_eigs)
    quotient_min = quotient_eigs[0]
    quotient_trace = jnp.sum(quotient_eigs)

    min_eig_floor = jax.nn.relu(min_eig_ratio_target * tau_min - quotient_min) ** 2
    trace_cap = jax.nn.relu(quotient_trace - trace_cap_ratio * tau_trace) ** 2
    loss = min_eig_floor + 0.1 * trace_cap
    return loss, quotient_eigs, tau_eigs, tau_trace


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


def _make_train_step_factorized_lattice_vae(
    model,
    beta: float,
    quotient_weight: float,
    gauge_weight: float,
    decoder_weight: float,
    action_reg_weight: float,
    chart_preserving_weight: float,
    chart_preserving_n_neighbors: int,
    quotient_variance_floor_weight: float,
    quotient_variance_floor_target: float,
    quotient_spread_weight: float = 0.0,
    quotient_min_eig_ratio_target: float = 0.20,
    quotient_trace_cap_ratio: float = 1.50,
):
    """Create a JIT-compiled factorized lattice VAE training step."""

    @jax.jit
    def train_step_factorized_lattice_vae(
        state: VAETrainState,
        batch: jnp.ndarray,
        paired_batch: jnp.ndarray,
        transform_ids: jnp.ndarray,
        tau_fd_coords: jnp.ndarray,
    ) -> tuple[VAETrainState, dict[str, jnp.ndarray]]:
        rng, step_rng = jax.random.split(state.rng)

        def loss_fn(params):
            x_hat, _, q_mean, q_logvar, g_mean, g_logvar = state.apply_fn(
                {'params': params}, batch, step_rng,
            )
            _, _, q_pair_mean, _, g_pair_mean, _ = state.apply_fn(
                {'params': params}, paired_batch, step_rng, deterministic=True,
            )
            acted_gauge = state.apply_fn(
                {'params': params},
                g_mean,
                transform_ids,
                method=model.apply_gauge_action,
            )
            paired_recon = state.apply_fn(
                {'params': params},
                q_mean,
                acted_gauge,
                method=model.decode_parts,
            )
            action_reg = state.apply_fn(
                {'params': params},
                method=model.action_regularizer,
            )

            mse = jnp.mean((batch - x_hat) ** 2)
            kl_q = jnp.mean(FactorizedVAE.kl_divergence(q_mean, q_logvar))
            kl_g = jnp.mean(FactorizedVAE.kl_divergence(g_mean, g_logvar))
            kl = kl_q + kl_g
            quotient_invariance = jnp.mean(
                jnp.sum((q_mean - q_pair_mean) ** 2, axis=-1),
            )
            gauge_equivariance = jnp.mean(
                jnp.sum((g_pair_mean - acted_gauge) ** 2, axis=-1),
            )
            decoder_equivariance = jnp.mean((paired_recon - paired_batch) ** 2)
            quotient_chart = _quotient_chart_loss(
                tau_fd_coords,
                q_mean,
                chart_preserving_n_neighbors,
            )
            quotient_variance_floor, _ = _quotient_variance_floor_loss(
                q_mean,
                quotient_variance_floor_target,
            )
            quotient_spread, _, _, _ = _quotient_spread_loss(
                tau_fd_coords,
                q_mean,
                quotient_min_eig_ratio_target,
                quotient_trace_cap_ratio,
            )
            loss = (
                mse
                + beta * kl
                + quotient_weight * quotient_invariance
                + gauge_weight * gauge_equivariance
                + decoder_weight * decoder_equivariance
                + action_reg_weight * action_reg
                + chart_preserving_weight * quotient_chart
                + quotient_variance_floor_weight * quotient_variance_floor
                + quotient_spread_weight * quotient_spread
            )
            return loss, {
                'loss': loss,
                'mse': mse,
                'kl': kl,
                'quotient_invariance': quotient_invariance,
                'gauge_equivariance': gauge_equivariance,
                'decoder_equivariance': decoder_equivariance,
                'action_regularizer': action_reg,
                'quotient_chart_loss': quotient_chart,
                'quotient_variance_floor_loss': quotient_variance_floor,
                'quotient_spread_loss': quotient_spread,
            }

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(rng=rng)
        return state, metrics

    return train_step_factorized_lattice_vae


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


def _make_eval_step_factorized_lattice_vae(
    model,
    beta: float,
    quotient_weight: float,
    gauge_weight: float,
    decoder_weight: float,
    action_reg_weight: float,
    chart_preserving_weight: float,
    chart_preserving_n_neighbors: int,
    quotient_variance_floor_weight: float,
    quotient_variance_floor_target: float,
    quotient_spread_weight: float = 0.0,
    quotient_min_eig_ratio_target: float = 0.20,
    quotient_trace_cap_ratio: float = 1.50,
):
    """Create a JIT-compiled factorized lattice VAE eval step."""

    @jax.jit
    def eval_step_factorized_lattice_vae(
        state: VAETrainState,
        batch: jnp.ndarray,
        paired_batch: jnp.ndarray,
        transform_ids: jnp.ndarray,
        tau_fd_coords: jnp.ndarray,
    ) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
        x_hat, z, q_mean, q_logvar, g_mean, g_logvar = state.apply_fn(
            {'params': state.params}, batch, state.rng, deterministic=True,
        )
        _, _, q_pair_mean, _, g_pair_mean, _ = state.apply_fn(
            {'params': state.params}, paired_batch, state.rng, deterministic=True,
        )
        acted_gauge = state.apply_fn(
            {'params': state.params},
            g_mean,
            transform_ids,
            method=model.apply_gauge_action,
        )
        paired_recon = state.apply_fn(
            {'params': state.params},
            q_mean,
            acted_gauge,
            method=model.decode_parts,
        )
        action_reg = state.apply_fn(
            {'params': state.params},
            method=model.action_regularizer,
        )

        mse = jnp.mean((batch - x_hat) ** 2)
        kl_q = jnp.mean(FactorizedVAE.kl_divergence(q_mean, q_logvar))
        kl_g = jnp.mean(FactorizedVAE.kl_divergence(g_mean, g_logvar))
        kl = kl_q + kl_g
        quotient_invariance = jnp.mean(
            jnp.sum((q_mean - q_pair_mean) ** 2, axis=-1),
        )
        gauge_equivariance = jnp.mean(
            jnp.sum((g_pair_mean - acted_gauge) ** 2, axis=-1),
        )
        decoder_equivariance = jnp.mean((paired_recon - paired_batch) ** 2)
        quotient_chart = _quotient_chart_loss(
            tau_fd_coords,
            q_mean,
            chart_preserving_n_neighbors,
        )
        quotient_variance_floor, _ = _quotient_variance_floor_loss(
            q_mean,
            quotient_variance_floor_target,
        )
        quotient_spread, _, _, _ = _quotient_spread_loss(
            tau_fd_coords,
            q_mean,
            quotient_min_eig_ratio_target,
            quotient_trace_cap_ratio,
        )
        loss = (
            mse
            + beta * kl
            + quotient_weight * quotient_invariance
            + gauge_weight * gauge_equivariance
            + decoder_weight * decoder_equivariance
            + action_reg_weight * action_reg
            + chart_preserving_weight * quotient_chart
            + quotient_variance_floor_weight * quotient_variance_floor
            + quotient_spread_weight * quotient_spread
        )
        return {
            'loss': loss,
            'mse': mse,
            'kl': kl,
            'quotient_invariance': quotient_invariance,
            'gauge_equivariance': gauge_equivariance,
            'decoder_equivariance': decoder_equivariance,
            'action_regularizer': action_reg,
            'quotient_chart_loss': quotient_chart,
            'quotient_variance_floor_loss': quotient_variance_floor,
            'quotient_spread_loss': quotient_spread,
        }, z

    return eval_step_factorized_lattice_vae


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
