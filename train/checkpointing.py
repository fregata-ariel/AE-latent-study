"""Lightweight checkpoint save/restore helpers."""

from __future__ import annotations

import dataclasses
import os

from flax import serialization
import ml_collections


@dataclasses.dataclass
class CheckpointManager:
    """Minimal checkpoint manager used by the training loop."""

    directory: str
    max_to_keep: int


def _checkpoint_path(mngr: CheckpointManager, step: int) -> str:
    return os.path.join(mngr.directory, f'{step}.msgpack')


def create_checkpoint_manager(
    config: ml_collections.ConfigDict,
    workdir: str,
) -> CheckpointManager:
    """Create a simple checkpoint manager rooted under the workdir."""
    ckpt_dir = os.path.join(workdir, config.checkpoint.dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    return CheckpointManager(
        directory=ckpt_dir,
        max_to_keep=config.checkpoint.max_to_keep,
    )


def save_checkpoint(mngr: CheckpointManager, step: int, state) -> None:
    """Save a TrainState-like object to a msgpack checkpoint."""
    ckpt_path = _checkpoint_path(mngr, step)
    with open(ckpt_path, 'wb') as f:
        f.write(serialization.to_bytes(state))

    checkpoint_files = sorted(
        [
            fname for fname in os.listdir(mngr.directory)
            if fname.endswith('.msgpack')
        ],
        key=lambda name: int(name.split('.')[0]),
    )
    stale = checkpoint_files[:-mngr.max_to_keep]
    for fname in stale:
        os.remove(os.path.join(mngr.directory, fname))


def restore_checkpoint(mngr: CheckpointManager, step: int, state):
    """Restore a TrainState-like object from a msgpack checkpoint."""
    ckpt_path = _checkpoint_path(mngr, step)
    with open(ckpt_path, 'rb') as f:
        payload = f.read()
    return serialization.from_bytes(state, payload)
