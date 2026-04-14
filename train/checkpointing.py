"""Checkpoint save/restore utilities using orbax."""

import os

import orbax.checkpoint as ocp
import ml_collections


def create_checkpoint_manager(
    config: ml_collections.ConfigDict,
    workdir: str,
) -> ocp.CheckpointManager:
    """Create an orbax CheckpointManager.

    Args:
        config: Experiment configuration.
        workdir: Working directory for outputs.

    Returns:
        A CheckpointManager instance.
    """
    ckpt_dir = os.path.join(workdir, config.checkpoint.dir)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.checkpoint.max_to_keep,
    )
    return ocp.CheckpointManager(ckpt_dir, options=options)


def save_checkpoint(mngr: ocp.CheckpointManager, step: int, state) -> None:
    """Save a checkpoint.

    Args:
        mngr: CheckpointManager instance.
        step: Current step/epoch number.
        state: TrainState to save.
    """
    mngr.save(step, args=ocp.args.StandardSave(state))
    mngr.wait_until_finished()


def restore_checkpoint(mngr: ocp.CheckpointManager, step: int, state):
    """Restore a checkpoint.

    Args:
        mngr: CheckpointManager instance.
        step: Step/epoch number to restore.
        state: Template TrainState for shape/dtype inference.

    Returns:
        Restored TrainState.
    """
    return mngr.restore(step, args=ocp.args.StandardRestore(state))
