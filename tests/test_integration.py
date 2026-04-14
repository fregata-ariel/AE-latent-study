"""Integration smoke tests: full pipeline with tiny configs."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from configs.default import get_config
from train.trainer import train_and_evaluate


def _tiny_config(latent_type='standard', torus_dim=1, latent_dim=2):
    """Create a tiny config for fast smoke tests."""
    config = get_config()
    config.seed = 0
    config.data.torus_dim = torus_dim
    config.data.signal_length = 50
    config.data.n_train = 32
    config.data.n_val = 16
    config.data.n_test = 16
    config.model.latent_type = latent_type
    config.model.latent_dim = latent_dim
    config.model.encoder_hidden = (16, 8)
    config.model.decoder_hidden = (8, 16)
    config.train.batch_size = 16
    config.train.num_epochs = 3
    config.train.patience = 10
    config.train.log_every = 1
    config.checkpoint.max_to_keep = 1
    return config


def test_standard_ae_pipeline():
    config = _tiny_config(latent_type='standard', torus_dim=1, latent_dim=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, (train_ds, val_ds, test_ds) = train_and_evaluate(
            config, tmpdir,
        )
        assert len(history['train_loss']) == 3
        assert len(history['val_loss']) == 3
        # Loss should have been computed
        assert history['train_loss'][0] > 0


def test_torus_ae_pipeline():
    config = _tiny_config(latent_type='torus', torus_dim=1, latent_dim=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, _ = train_and_evaluate(config, tmpdir)
        assert len(history['train_loss']) == 3


def test_t2_standard_pipeline():
    config = _tiny_config(latent_type='standard', torus_dim=2, latent_dim=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, _ = train_and_evaluate(config, tmpdir)
        assert len(history['train_loss']) == 3


def test_t2_torus_pipeline():
    config = _tiny_config(latent_type='torus', torus_dim=2, latent_dim=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, _ = train_and_evaluate(config, tmpdir)
        assert len(history['train_loss']) == 3


def test_vae_pipeline():
    config = _tiny_config(latent_type='vae', torus_dim=1, latent_dim=2)
    config.model.vae_beta = 0.01
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, _ = train_and_evaluate(config, tmpdir)
        assert len(history['train_loss']) == 3
        assert 'train_kl' in history


def test_loss_decreases():
    """Train for a bit longer and check that loss decreases."""
    config = _tiny_config(latent_type='standard', torus_dim=1, latent_dim=2)
    config.train.num_epochs = 20
    config.train.patience = 25
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, _ = train_and_evaluate(config, tmpdir)
        # Loss should decrease over training
        assert history['train_loss'][-1] < history['train_loss'][0]


def test_checkpoint_saved():
    """Check that checkpoint files are created."""
    config = _tiny_config(latent_type='standard', torus_dim=1, latent_dim=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, _ = train_and_evaluate(config, tmpdir)
        ckpt_dir = os.path.join(tmpdir, config.checkpoint.dir)
        assert os.path.exists(ckpt_dir)
