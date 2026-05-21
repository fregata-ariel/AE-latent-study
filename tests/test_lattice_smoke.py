"""Smoke tests for the lattice experiment pipeline.

These verify that data generation, dataset creation, model construction,
training, and evaluation modules wire together correctly under a tiny
lattice config. Each test stays small enough to run on CPU in a few
seconds.

Migrated from the legacy root-level ``test_lattice_smoke.py`` script.
"""

from __future__ import annotations

import os
import tempfile

import jax
import numpy as np
import pytest

from configs.lattice_standard import get_config
from data.dataset import create_splits
from data.generation import (
    compute_j_invariant,
    generate_lattice_theta,
    reduce_to_fundamental_domain,
    sample_fundamental_domain,
    sample_upper_halfplane,
)
from eval.analysis import run_full_evaluation
from models import create_model
from train.trainer import train_and_evaluate


@pytest.fixture
def tiny_lattice_config():
    config = get_config()
    config.data.n_train = 50
    config.data.n_val = 20
    config.data.n_test = 20
    config.data.lattice_K = 5
    config.train.num_epochs = 5
    config.train.batch_size = 16
    config.train.patience = 100
    config.train.log_every = 1
    return config


def test_sample_fundamental_domain_within_F():
    key = jax.random.PRNGKey(0)
    tau = sample_fundamental_domain(100, y_max=3.0, key=key)
    assert len(tau) == 100
    assert all(abs(t.real) <= 0.5 + 1e-6 for t in tau), 'Re(τ) out of range'
    assert all(abs(t) >= 1.0 - 1e-6 for t in tau), '|τ| < 1 found'
    assert all(t.imag > 0 for t in tau), 'Im(τ) ≤ 0 found'


def test_sample_upper_halfplane_positive_imag():
    key = jax.random.PRNGKey(0)
    tau = sample_upper_halfplane(50, key=key)
    assert len(tau) == 50
    assert all(t.imag > 0 for t in tau)


def test_reduce_to_fundamental_domain_maps_back_to_F():
    key = jax.random.PRNGKey(0)
    tau_h = sample_upper_halfplane(50, key=key)
    tau_reduced = reduce_to_fundamental_domain(tau_h)
    for t in tau_reduced:
        assert abs(t.real) <= 0.5 + 1e-4, f'Re(τ)={t.real:.4f} out of range'
        assert abs(t) >= 1.0 - 1e-4, f'|τ|={abs(t):.4f} < 1'


def test_generate_lattice_theta_shape_and_finiteness():
    key = jax.random.PRNGKey(0)
    tau = sample_fundamental_domain(10, y_max=3.0, key=key)
    signals = generate_lattice_theta(tau, signal_length=50, t_min=0.5, t_max=3.0, K=5)
    assert signals.shape == (10, 50)
    assert np.all(np.isfinite(signals))


def test_compute_j_invariant_at_i_is_about_1728():
    j_i = compute_j_invariant(np.array([1j]))
    assert len(j_i) == 1
    # j(i) = 1728 exactly for τ = i (square lattice). Allow small numerical slack.
    assert abs(np.real(j_i[0]) - 1728.0) < 5.0


def test_create_splits_lattice(tiny_lattice_config):
    key = jax.random.PRNGKey(42)
    train_ds, val_ds, test_ds = create_splits(tiny_lattice_config, key)
    assert len(train_ds) == 50
    assert len(val_ds) == 20
    assert len(test_ds) == 20
    assert train_ds.signals.shape[0] == 50
    assert train_ds.thetas.shape[0] == 50
    assert train_ds.j_invariant is not None
    assert train_ds.tau is not None


def test_create_model_standard_lattice(tiny_lattice_config):
    model = create_model(tiny_lattice_config)
    dummy = jax.numpy.ones((1, tiny_lattice_config.data.signal_length))
    params = model.init(jax.random.PRNGKey(0), dummy)
    x_hat, z = model.apply(params, dummy)
    assert x_hat.shape == dummy.shape
    assert z.ndim == 2 and z.shape[0] == 1


def test_create_model_halfplane_keeps_imag_positive():
    config = get_config()
    config.model.latent_type = 'halfplane'
    config.data.n_train = 50
    config.data.n_val = 10
    config.data.n_test = 10
    model = create_model(config)
    dummy = jax.numpy.ones((1, config.data.signal_length))
    params = model.init(jax.random.PRNGKey(0), dummy)
    _, z = model.apply(params, dummy)
    assert z.shape == (1, 2)
    assert float(z[0, 1]) > 0, f'Im(τ) = {float(z[0, 1]):.4f} ≤ 0 — halfplane parametrization broken'


def test_train_and_evaluate_full_pipeline(tiny_lattice_config):
    with tempfile.TemporaryDirectory() as workdir:
        state, history, (train_ds, _val_ds, test_ds) = train_and_evaluate(
            tiny_lattice_config, workdir,
        )
        assert state is not None
        assert 'train_loss' in history and 'val_loss' in history
        assert len(history['train_loss']) >= 1

        summary = run_full_evaluation(
            state, tiny_lattice_config, train_ds, test_ds, history, workdir,
        )
        assert 'reconstruction' in summary
        assert np.isfinite(summary['reconstruction']['mse'])

        results_dir = os.path.join(workdir, 'results')
        for fname in (
            'training_curves.png',
            'reconstructions.png',
            'lattice_latent_scatter.png',
            'j_invariant_correlation.png',
            'summary.json',
        ):
            assert os.path.exists(os.path.join(results_dir, fname)), (
                f'Expected evaluation artifact missing: {fname}'
            )
