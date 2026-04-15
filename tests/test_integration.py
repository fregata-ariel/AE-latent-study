"""Integration smoke tests: full pipeline with tiny configs."""

import os
import tempfile
import json

import jax
import jax.numpy as jnp
import numpy as np

from configs.default import get_config
from configs.lattice_default import get_config as get_lattice_config
from run_lattice_step2_experiments import run_all as run_step2_all
from run_lattice_step3_experiments import run_all as run_step3_all
from eval.analysis import run_full_evaluation
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


def _tiny_lattice_config(
    latent_type='standard',
    latent_dim=2,
    normalization='max',
    modular_invariance_weight=0.0,
    sample_region='fundamental_domain',
):
    config = get_lattice_config()
    config.seed = 0
    config.data.signal_length = 40
    config.data.n_train = 32
    config.data.n_val = 16
    config.data.n_test = 16
    config.data.lattice_K = 4
    config.data.lattice_y_max = 2.0
    config.data.lattice_sample_region = sample_region
    config.data.lattice_signal_normalization = normalization
    if sample_region == 'halfplane':
        config.data.lattice_x_range = (-1.0, 1.0)
        config.data.lattice_y_range = (0.5, 2.0)
    config.model.latent_type = latent_type
    config.model.latent_dim = latent_dim
    config.model.encoder_hidden = (16, 8)
    config.model.decoder_hidden = (8, 16)
    if latent_type == 'vae':
        config.model.vae_beta = 0.01
    config.train.batch_size = 16
    config.train.num_epochs = 3
    config.train.patience = 10
    config.train.log_every = 1
    config.train.modular_invariance_weight = modular_invariance_weight
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


def test_lattice_pipeline_without_invariance():
    config = _tiny_lattice_config(modular_invariance_weight=0.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, tmpdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, tmpdir)
        assert 'train_inv_loss' not in history
        assert 'j_correlation' in summary
        assert 'max_abs_logabsj_spearman' in summary['j_correlation']
        assert 'chart_quality' in summary
        assert 'trustworthiness' in summary['chart_quality']
        assert 'modular_invariance' in summary
        chart_plot = os.path.join(
            tmpdir, config.eval.output_dir, 'quotient_chart_quality.png',
        )
        assert os.path.exists(chart_plot)


def test_lattice_pipeline_with_invariance():
    config = _tiny_lattice_config(modular_invariance_weight=0.1)
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, _ = train_and_evaluate(config, tmpdir)
        assert 'train_inv_loss' in history
        assert len(history['train_inv_loss']) == len(history['train_loss'])
        assert history['train_inv_loss'][0] >= 0.0


def test_lattice_vae_pipeline_with_invariance():
    config = _tiny_lattice_config(
        latent_type='vae',
        latent_dim=4,
        modular_invariance_weight=0.1,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, tmpdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, tmpdir)

        assert 'train_kl' in history
        assert 'train_inv_loss' in history
        assert len(history['train_kl']) == len(history['train_loss'])
        assert len(history['train_inv_loss']) == len(history['train_loss'])
        assert history['train_inv_loss'][0] >= 0.0

        history_path = os.path.join(tmpdir, 'history.json')
        with open(history_path) as f:
            saved_history = json.load(f)
        assert 'train_kl' in saved_history
        assert 'train_inv_loss' in saved_history

        assert 'chart_quality' in summary
        assert 'j_correlation' in summary
        assert 'modular_invariance' in summary


def test_step2_runner_with_tiny_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-lattice-step2.md')

        def tiny_config_factory():
            return _tiny_lattice_config(modular_invariance_weight=0.1)

        summaries = run_step2_all(
            base_dir=tmpdir,
            experiments=[('tiny_step2', tiny_config_factory)],
            summary_filename='tiny_step2_summaries.json',
            report_path=report_path,
        )

        assert 'tiny_step2' in summaries
        summary = summaries['tiny_step2']
        assert 'j_correlation' in summary
        assert 'max_abs_logabsj_spearman' in summary['j_correlation']
        assert 'chart_quality' in summary
        assert 'trustworthiness' in summary['chart_quality']
        assert 'modular_invariance' in summary

        summary_path = os.path.join(tmpdir, 'tiny_step2_summaries.json')
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            saved = json.load(f)
        assert 'tiny_step2' in saved
        assert 'chart_quality' in saved['tiny_step2']

        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_text = f.read()
        assert 'trust' in report_text
        assert 'eff_dim' in report_text


def test_step3_runner_with_tiny_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-lattice-step3.md')
        anchor_path = os.path.join(tmpdir, 'step2_anchors.json')

        anchor_summaries = {
            'lattice_standard_norm_inv': {
                'reconstruction': {'mse': 1e-7},
                'j_correlation': {'max_abs_logabsj_spearman': 0.98},
                'modular_invariance': {'mean_latent_distance': 0.001},
                'chart_quality': {
                    'trustworthiness': 0.86,
                    'knn_jaccard_mean': 0.05,
                    'effective_dimension': 1.0,
                },
            },
            'lattice_standard_wide_norm_inv': {
                'reconstruction': {'mse': 2e-7},
                'j_correlation': {'max_abs_logabsj_spearman': 0.90},
                'modular_invariance': {'mean_latent_distance': 0.004},
                'chart_quality': {
                    'trustworthiness': 0.88,
                    'knn_jaccard_mean': 0.06,
                    'effective_dimension': 1.3,
                },
            },
            'lattice_vae_norm_beta001': {
                'reconstruction': {'mse': 8e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.63},
                'modular_invariance': {'mean_latent_distance': 0.007},
                'chart_quality': {
                    'trustworthiness': 0.90,
                    'knn_jaccard_mean': 0.06,
                    'effective_dimension': 1.8,
                },
            },
        }
        with open(anchor_path, 'w') as f:
            json.dump(anchor_summaries, f, indent=2)

        def tiny_step3_factory():
            config = _tiny_lattice_config(
                latent_type='vae',
                latent_dim=4,
                modular_invariance_weight=0.03,
            )
            config.model.vae_beta = 0.003
            return config

        summaries = run_step3_all(
            base_dir=tmpdir,
            experiments=[('lattice_vae_norm_inv_b003_l030', tiny_step3_factory)],
            summary_filename='tiny_step3_summaries.json',
            report_path=report_path,
            anchor_summary_path=anchor_path,
        )

        assert 'lattice_vae_norm_inv_b003_l030' in summaries
        summary = summaries['lattice_vae_norm_inv_b003_l030']
        assert 'chart_quality' in summary
        assert 'j_correlation' in summary
        assert 'modular_invariance' in summary

        summary_path = os.path.join(tmpdir, 'tiny_step3_summaries.json')
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            saved = json.load(f)
        assert 'lattice_vae_norm_inv_b003_l030' in saved

        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_text = f.read()
        assert 'lattice_standard_norm_inv' in report_text
        assert 'Success gate' in report_text
        assert 'Selected run' in report_text
