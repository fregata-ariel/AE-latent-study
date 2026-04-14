"""Dataset container, splitting, and batching utilities."""

import dataclasses
from typing import Iterator

import jax
import jax.numpy as jnp
import ml_collections

from data.generation import generate_dataset


@dataclasses.dataclass
class Dataset:
    """Simple dataset holding signals and their ground-truth theta parameters.

    Attributes:
        signals: Observed time-series, shape (N, signal_length).
        thetas: Ground-truth latent angles, shape (N,) for T^1 or (N, 2) for T^2.
    """
    signals: jnp.ndarray
    thetas: jnp.ndarray

    def __len__(self) -> int:
        return self.signals.shape[0]


def create_splits(
    config: ml_collections.ConfigDict,
    key: jax.Array,
) -> tuple[Dataset, Dataset, Dataset]:
    """Generate data and split into train/val/test datasets.

    Args:
        config: Experiment configuration.
        key: PRNG key for data generation.

    Returns:
        Tuple of (train_ds, val_ds, test_ds).
    """
    data = generate_dataset(config, key)
    signals = data['signals']
    thetas = data['thetas']

    n_train = config.data.n_train
    n_val = config.data.n_val

    train_ds = Dataset(
        signals=signals[:n_train],
        thetas=thetas[:n_train],
    )
    val_ds = Dataset(
        signals=signals[n_train:n_train + n_val],
        thetas=thetas[n_train:n_train + n_val],
    )
    test_ds = Dataset(
        signals=signals[n_train + n_val:],
        thetas=thetas[n_train + n_val:],
    )

    return train_ds, val_ds, test_ds


def batched_iterator(
    dataset: Dataset,
    batch_size: int,
    key: jax.Array,
    shuffle: bool = True,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield mini-batches of (signals, thetas) from the dataset.

    Drops the last incomplete batch to maintain fixed batch sizes
    (avoids JIT recompilation).

    Args:
        dataset: Source dataset.
        batch_size: Number of samples per batch.
        key: PRNG key for shuffling.
        shuffle: Whether to shuffle indices before batching.

    Yields:
        Tuples of (signal_batch, theta_batch).
    """
    n = len(dataset)
    if shuffle:
        indices = jax.random.permutation(key, n)
    else:
        indices = jnp.arange(n)

    n_batches = n // batch_size
    for i in range(n_batches):
        batch_idx = indices[i * batch_size:(i + 1) * batch_size]
        yield dataset.signals[batch_idx], dataset.thetas[batch_idx]
