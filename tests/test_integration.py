"""Integration smoke tests: full pipeline with tiny configs."""

import os
import tempfile
import json
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from configs.default import get_config
from configs.lattice_default import get_config as get_lattice_config
from run_lattice_step2_experiments import run_all as run_step2_all
from run_lattice_step3_experiments import run_all as run_step3_all
from run_lattice_step4_experiments import run_all as run_step4_all
from run_lattice_step5_experiments import run_all as run_step5_all
from run_lattice_step6_experiments import run_all as run_step6_all
from run_lattice_step7_experiments import run_all as run_step7_all
from run_lattice_step8_experiments import run_all as run_step8_all
from run_lattice_step9_experiments import run_all as run_step9_all
from run_latent_topology_diagnostics import run_all as run_topology_all
from run_topology_phaseB_comparison import run_all as run_topology_phaseb_all
from run_topology_step7_followup import run_all as run_topology_step7_followup_all
from run_topology_step8_followup import run_all as run_topology_step8_followup_all
from run_topology_step9_followup import run_all as run_topology_step9_followup_all
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
    if latent_type == 'factorized_vae':
        config.model.latent_dim = 6
        config.model.quotient_dim = 2
        config.model.gauge_dim = 4
        config.model.gauge_action_type = 'affine'
        config.model.vae_beta = 0.01
        config.train.gauge_equivariance_weight = 0.03
        config.train.decoder_equivariance_weight = 0.03
        config.train.gauge_action_reg_weight = 1e-4
        config.train.chart_preserving_weight = 0.03
        config.train.chart_preserving_n_neighbors = 4
        config.train.quotient_variance_floor_weight = 0.01
        config.train.quotient_variance_floor_target = 0.05
        config.train.quotient_spread_weight = 0.0
        config.train.quotient_min_eig_ratio_target = 0.20
        config.train.quotient_trace_cap_ratio = 1.50
        config.train.jacobian_gram_weight = 0.0
        config.train.jacobian_n_neighbors = 4
        config.train.quotient_logdet_weight = 0.0
        config.train.quotient_logdet_ratio_target = 0.10
        config.train.j_rank_preserving_weight = 0.0
        config.train.j_rank_temperature = 0.10
        config.train.j_rank_min_delta = 0.10
        config.train.j_rank_n_terms = 20
        config.train.teacher_distill_weight = 0.0
        config.train.teacher_run_dir = ''
        config.train.teacher_distill_n_neighbors = 4
        config.train.teacher_distill_view = 'quotient'
        config.train.teacher_distill_loss_type = 'local_distance'
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


def test_factorized_lattice_pipeline():
    config = _tiny_lattice_config(
        latent_type='factorized_vae',
        latent_dim=6,
        modular_invariance_weight=0.1,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        state, history, (train_ds, _, test_ds) = train_and_evaluate(config, tmpdir)
        summary = run_full_evaluation(state, config, train_ds, test_ds, history, tmpdir)

        assert 'train_kl' in history
        assert 'train_quotient_invariance' in history
        assert 'train_gauge_equivariance' in history
        assert 'train_decoder_equivariance' in history
        assert 'train_action_regularizer' in history
        assert 'train_quotient_chart_loss' in history
        assert 'train_quotient_variance_floor_loss' in history
        assert 'train_quotient_spread_loss' in history
        assert 'train_quotient_j_rank_loss' in history
        assert 'train_quotient_teacher_distill_loss' in history
        assert 'factorized_consistency' in summary
        assert 'chart_quality' in summary
        assert 'j_correlation' in summary
        assert 'modular_invariance' in summary
        assert 'quotient_partner_rank_percentile_mean' in summary['factorized_consistency']
        assert 'quotient_chart_loss' in summary['factorized_consistency']
        assert 'quotient_var_dim0' in summary['factorized_consistency']
        assert 'quotient_spread_loss' in summary['factorized_consistency']
        assert 'quotient_cov_eig_min' in summary['factorized_consistency']
        assert 'quotient_j_rank_loss' in summary['factorized_consistency']
        assert 'quotient_teacher_distill_loss' in summary['factorized_consistency']


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


def test_step4_runner_with_tiny_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-lattice-step4.md')
        anchor_path = os.path.join(tmpdir, 'step3_anchors.json')

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
            'lattice_vae_norm_inv_b010_l100': {
                'reconstruction': {'mse': 8e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.96},
                'modular_invariance': {'mean_latent_distance': 0.0001},
                'chart_quality': {
                    'trustworthiness': 0.85,
                    'knn_jaccard_mean': 0.06,
                    'effective_dimension': 2.0,
                },
            },
            'lattice_vae_norm_inv_b030_l100': {
                'reconstruction': {'mse': 8e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.81},
                'modular_invariance': {'mean_latent_distance': 0.0001},
                'chart_quality': {
                    'trustworthiness': 0.86,
                    'knn_jaccard_mean': 0.067,
                    'effective_dimension': 1.85,
                },
            },
            'lattice_vae_wide_norm_inv_b003_l030': {
                'reconstruction': {'mse': 8.7e-4},
                'j_correlation': {'max_abs_logabsj_spearman': 0.85},
                'modular_invariance': {'mean_latent_distance': 0.0004},
                'chart_quality': {
                    'trustworthiness': 0.88,
                    'knn_jaccard_mean': 0.053,
                    'effective_dimension': 1.96,
                },
            },
        }
        with open(anchor_path, 'w') as f:
            json.dump(anchor_summaries, f, indent=2)

        def tiny_step4_factory():
            return _tiny_lattice_config(
                latent_type='factorized_vae',
                latent_dim=6,
                modular_invariance_weight=0.1,
            )

        summaries = run_step4_all(
            base_dir=tmpdir,
            experiments=[('lattice_factorized_vae_fd_b010_q100_g030_d030', tiny_step4_factory)],
            summary_filename='tiny_step4_summaries.json',
            report_path=report_path,
            anchor_summary_path=anchor_path,
        )

        assert 'lattice_factorized_vae_fd_b010_q100_g030_d030' in summaries
        summary = summaries['lattice_factorized_vae_fd_b010_q100_g030_d030']
        assert 'factorized_consistency' in summary
        assert 'chart_quality' in summary
        assert 'j_correlation' in summary

        summary_path = os.path.join(tmpdir, 'tiny_step4_summaries.json')
        assert os.path.exists(summary_path)
        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_text = f.read()
        assert 'Step 3 Anchors' in report_text
        assert 'Selected run' in report_text


def test_step5_runner_with_tiny_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-lattice-step5.md')
        anchor_path = os.path.join(tmpdir, 'step4_anchors.json')

        anchor_summaries = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030': {
                'reconstruction': {'mse': 1.7e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.92},
                'chart_quality': {
                    'trustworthiness': 0.848,
                    'knn_jaccard_mean': 0.049,
                    'effective_dimension': 1.38,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.246,
                    'quotient_var_dim0': 0.12,
                    'quotient_var_dim1': 0.04,
                },
            },
        }
        with open(anchor_path, 'w') as f:
            json.dump(anchor_summaries, f, indent=2)

        def tiny_step5_factory():
            config = _tiny_lattice_config(
                latent_type='factorized_vae',
                latent_dim=6,
                modular_invariance_weight=0.1,
            )
            config.model.vae_beta = 0.03
            config.train.chart_preserving_weight = 0.03
            config.train.quotient_variance_floor_weight = 0.01
            config.train.quotient_variance_floor_target = 0.05
            return config

        summaries = run_step5_all(
            base_dir=tmpdir,
            experiments=[('lattice_factorized_vae_fd_b030_q100_g030_d030_c030_v010', tiny_step5_factory)],
            summary_filename='tiny_step5_summaries.json',
            report_path=report_path,
            anchor_summary_path=anchor_path,
        )

        assert 'lattice_factorized_vae_fd_b030_q100_g030_d030_c030_v010' in summaries
        summary = summaries['lattice_factorized_vae_fd_b030_q100_g030_d030_c030_v010']
        assert 'factorized_consistency' in summary
        assert 'quotient_chart_loss' in summary['factorized_consistency']
        assert 'quotient_variance_floor_loss' in summary['factorized_consistency']
        assert 'quotient_var_dim0' in summary['factorized_consistency']
        assert 'quotient_var_dim1' in summary['factorized_consistency']

        summary_path = os.path.join(tmpdir, 'tiny_step5_summaries.json')
        assert os.path.exists(summary_path)
        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_text = f.read()
        assert 'Step 4 Anchor' in report_text
        assert 'q chart loss' in report_text


def test_step6_runner_with_tiny_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-lattice-step6.md')
        step3_anchor_path = os.path.join(tmpdir, 'step3_anchors.json')
        step4_anchor_path = os.path.join(tmpdir, 'step4_anchors.json')

        step3_anchors = {
            'lattice_standard_norm_inv': {
                'reconstruction': {'mse': 1e-7},
                'j_correlation': {'max_abs_logabsj_spearman': 0.98},
                'chart_quality': {
                    'trustworthiness': 0.85,
                    'knn_jaccard_mean': 0.05,
                    'effective_dimension': 1.0,
                },
            },
            'lattice_vae_norm_inv_b010_l100': {
                'reconstruction': {'mse': 8e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.96},
                'chart_quality': {
                    'trustworthiness': 0.85,
                    'knn_jaccard_mean': 0.058,
                    'effective_dimension': 2.0,
                },
            },
            'lattice_vae_norm_inv_b030_l100': {
                'reconstruction': {'mse': 8e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.81},
                'chart_quality': {
                    'trustworthiness': 0.86,
                    'knn_jaccard_mean': 0.067,
                    'effective_dimension': 1.85,
                },
            },
        }
        step4_anchors = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030': {
                'reconstruction': {'mse': 1.7e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.92},
                'chart_quality': {
                    'trustworthiness': 0.848,
                    'knn_jaccard_mean': 0.049,
                    'effective_dimension': 1.38,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.246,
                    'quotient_cov_eig_min': 0.03,
                    'quotient_cov_eig_max': 0.12,
                    'quotient_spread_loss': 0.002,
                },
            },
        }
        with open(step3_anchor_path, 'w') as f:
            json.dump(step3_anchors, f, indent=2)
        with open(step4_anchor_path, 'w') as f:
            json.dump(step4_anchors, f, indent=2)

        def tiny_step6_factory():
            config = _tiny_lattice_config(
                latent_type='factorized_vae',
                latent_dim=6,
                modular_invariance_weight=0.1,
            )
            config.model.vae_beta = 0.03
            config.train.chart_preserving_weight = 0.01
            config.train.quotient_variance_floor_weight = 0.0
            config.train.quotient_spread_weight = 0.03
            config.train.quotient_min_eig_ratio_target = 0.20
            config.train.quotient_trace_cap_ratio = 1.50
            return config

        summaries = run_step6_all(
            base_dir=tmpdir,
            experiments=[('lattice_factorized_vae_fd_b030_q100_g030_d030_l010_s030', tiny_step6_factory)],
            summary_filename='tiny_step6_summaries.json',
            report_path=report_path,
            step3_anchor_summary_path=step3_anchor_path,
            step4_anchor_summary_path=step4_anchor_path,
        )

        assert 'lattice_factorized_vae_fd_b030_q100_g030_d030_l010_s030' in summaries
        summary = summaries['lattice_factorized_vae_fd_b030_q100_g030_d030_l010_s030']
        assert 'factorized_consistency' in summary
        assert 'quotient_spread_loss' in summary['factorized_consistency']
        assert 'quotient_cov_eig_min' in summary['factorized_consistency']
        assert 'quotient_cov_eig_max' in summary['factorized_consistency']
        assert 'tau_cov_eig_min' in summary['factorized_consistency']
        assert 'tau_cov_trace' in summary['factorized_consistency']

        summary_path = os.path.join(tmpdir, 'tiny_step6_summaries.json')
        assert os.path.exists(summary_path)
        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_text = f.read()
        assert 'Step 6 Runs' in report_text
        assert 'q spread loss' in report_text or 'spread loss' in report_text


def test_step7_runner_with_tiny_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-lattice-step7.md')
        step3_anchor_path = os.path.join(tmpdir, 'step3_anchors.json')
        step4_anchor_path = os.path.join(tmpdir, 'step4_anchors.json')
        step6_anchor_path = os.path.join(tmpdir, 'step6_anchors.json')

        step3_anchors = {
            'lattice_standard_norm_inv': {
                'reconstruction': {'mse': 1e-7},
                'j_correlation': {'max_abs_logabsj_spearman': 0.98},
                'chart_quality': {
                    'trustworthiness': 0.85,
                    'knn_jaccard_mean': 0.05,
                    'effective_dimension': 1.0,
                },
            },
            'lattice_vae_norm_inv_b010_l100': {
                'reconstruction': {'mse': 8e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.96},
                'chart_quality': {
                    'trustworthiness': 0.85,
                    'knn_jaccard_mean': 0.058,
                    'effective_dimension': 2.0,
                },
            },
            'lattice_vae_norm_inv_b030_l100': {
                'reconstruction': {'mse': 8e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.81},
                'chart_quality': {
                    'trustworthiness': 0.86,
                    'knn_jaccard_mean': 0.067,
                    'effective_dimension': 1.85,
                },
            },
        }
        step4_anchors = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030': {
                'reconstruction': {'mse': 1.7e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.92},
                'chart_quality': {
                    'trustworthiness': 0.848,
                    'knn_jaccard_mean': 0.049,
                    'effective_dimension': 1.38,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.246,
                },
            },
        }
        step6_anchors = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030': {
                'reconstruction': {'mse': 7.7e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.39},
                'chart_quality': {
                    'trustworthiness': 0.855,
                    'knn_jaccard_mean': 0.067,
                    'effective_dimension': 1.76,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.662,
                },
            },
        }
        for path, payload in (
            (step3_anchor_path, step3_anchors),
            (step4_anchor_path, step4_anchors),
            (step6_anchor_path, step6_anchors),
        ):
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2)

        def tiny_step7_factory():
            config = _tiny_lattice_config(
                latent_type='factorized_vae',
                latent_dim=6,
                modular_invariance_weight=0.1,
            )
            config.model.vae_beta = 0.03
            config.train.chart_preserving_weight = 0.0
            config.train.quotient_spread_weight = 0.0
            config.train.quotient_variance_floor_weight = 0.0
            config.train.jacobian_gram_weight = 0.01
            config.train.jacobian_n_neighbors = 4
            config.train.quotient_logdet_weight = 0.01
            config.train.quotient_logdet_ratio_target = 0.10
            config.train.quotient_trace_cap_ratio = 1.50
            return config

        summaries = run_step7_all(
            base_dir=tmpdir,
            experiments=[('lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010', tiny_step7_factory)],
            summary_filename='tiny_step7_summaries.json',
            report_path=report_path,
            step3_anchor_summary_path=step3_anchor_path,
            step4_anchor_summary_path=step4_anchor_path,
            step6_anchor_summary_path=step6_anchor_path,
        )

        assert 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010' in summaries
        summary = summaries['lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010']
        assert 'factorized_consistency' in summary
        assert 'quotient_jacobian_gram_loss' in summary['factorized_consistency']
        assert 'quotient_logdet_loss' in summary['factorized_consistency']
        assert 'quotient_cov_logdet' in summary['factorized_consistency']
        assert 'tau_cov_logdet' in summary['factorized_consistency']

        assert os.path.exists(os.path.join(tmpdir, 'tiny_step7_summaries.json'))
        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_text = f.read()
        assert 'Step 7 Runs' in report_text
        assert 'jacobian loss' in report_text


def test_step8_runner_with_tiny_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-lattice-step8.md')
        step3_anchor_path = os.path.join(tmpdir, 'step3_anchors.json')
        step4_anchor_path = os.path.join(tmpdir, 'step4_anchors.json')
        step6_anchor_path = os.path.join(tmpdir, 'step6_anchors.json')
        step7_anchor_path = os.path.join(tmpdir, 'step7_anchors.json')

        step3_anchors = {
            'lattice_standard_norm_inv': {
                'reconstruction': {'mse': 1e-7},
                'j_correlation': {'max_abs_logabsj_spearman': 0.98},
                'chart_quality': {
                    'trustworthiness': 0.85,
                    'knn_jaccard_mean': 0.05,
                    'effective_dimension': 1.0,
                },
            },
        }
        step4_anchors = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030': {
                'reconstruction': {'mse': 1.7e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.92},
                'chart_quality': {
                    'trustworthiness': 0.848,
                    'knn_jaccard_mean': 0.049,
                    'effective_dimension': 1.38,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.246,
                    'quotient_j_rank_loss': 0.7,
                },
            },
        }
        step6_anchors = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030': {
                'reconstruction': {'mse': 7.7e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.39},
                'chart_quality': {
                    'trustworthiness': 0.855,
                    'knn_jaccard_mean': 0.067,
                    'effective_dimension': 1.76,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.662,
                    'quotient_j_rank_loss': 1.2,
                },
            },
        }
        step7_anchors = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010': {
                'reconstruction': {'mse': 4e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.70},
                'chart_quality': {
                    'trustworthiness': 0.855,
                    'knn_jaccard_mean': 0.066,
                    'effective_dimension': 1.70,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.30,
                    'quotient_j_rank_loss': 0.9,
                },
            },
        }
        for path, payload in (
            (step3_anchor_path, step3_anchors),
            (step4_anchor_path, step4_anchors),
            (step6_anchor_path, step6_anchors),
            (step7_anchor_path, step7_anchors),
        ):
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2)

        def tiny_step8_factory():
            config = _tiny_lattice_config(
                latent_type='factorized_vae',
                latent_dim=6,
                modular_invariance_weight=0.1,
            )
            config.model.vae_beta = 0.03
            config.train.chart_preserving_weight = 0.0
            config.train.quotient_spread_weight = 0.0
            config.train.quotient_variance_floor_weight = 0.0
            config.train.jacobian_gram_weight = 0.005
            config.train.jacobian_n_neighbors = 4
            config.train.quotient_logdet_weight = 0.0
            config.train.quotient_logdet_ratio_target = 0.10
            config.train.quotient_trace_cap_ratio = 1.50
            config.train.j_rank_preserving_weight = 0.03
            config.train.j_rank_temperature = 0.10
            config.train.j_rank_min_delta = 0.10
            config.train.j_rank_n_terms = 10
            return config

        summaries = run_step8_all(
            base_dir=tmpdir,
            experiments=[('lattice_factorized_vae_fd_b030_q100_g030_d030_j005_ld000_r030', tiny_step8_factory)],
            summary_filename='tiny_step8_summaries.json',
            report_path=report_path,
            step3_anchor_summary_path=step3_anchor_path,
            step4_anchor_summary_path=step4_anchor_path,
            step6_anchor_summary_path=step6_anchor_path,
            step7_anchor_summary_path=step7_anchor_path,
        )

        assert 'lattice_factorized_vae_fd_b030_q100_g030_d030_j005_ld000_r030' in summaries
        summary = summaries['lattice_factorized_vae_fd_b030_q100_g030_d030_j005_ld000_r030']
        assert 'factorized_consistency' in summary
        assert 'quotient_j_rank_loss' in summary['factorized_consistency']
        assert 'quotient_j_rank_target_std' in summary['factorized_consistency']

        assert os.path.exists(os.path.join(tmpdir, 'tiny_step8_summaries.json'))
        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_text = f.read()
        assert 'Step 8 Runs' in report_text
        assert 'j-rank loss' in report_text


def test_step9_runner_with_tiny_teacher_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_name = 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010'
        teacher_dir = os.path.join(tmpdir, teacher_name)
        teacher_config = _tiny_lattice_config(
            latent_type='factorized_vae',
            latent_dim=6,
            modular_invariance_weight=0.1,
        )
        teacher_config.train.num_epochs = 2
        teacher_config.train.chart_preserving_weight = 0.0
        teacher_config.train.quotient_variance_floor_weight = 0.0
        teacher_config.train.quotient_spread_weight = 0.0
        teacher_config.train.jacobian_gram_weight = 0.01
        teacher_config.train.quotient_logdet_weight = 0.0
        teacher_config.train.j_rank_preserving_weight = 0.0
        train_and_evaluate(teacher_config, teacher_dir)

        report_path = os.path.join(tmpdir, 'walkthrough-lattice-step9.md')
        step3_anchor_path = os.path.join(tmpdir, 'step3_anchors.json')
        step4_anchor_path = os.path.join(tmpdir, 'step4_anchors.json')
        step6_anchor_path = os.path.join(tmpdir, 'step6_anchors.json')
        step7_anchor_path = os.path.join(tmpdir, 'step7_anchors.json')
        step8_anchor_path = os.path.join(tmpdir, 'step8_anchors.json')

        step3_anchors = {
            'lattice_standard_norm_inv': {
                'reconstruction': {'mse': 1e-7},
                'j_correlation': {'max_abs_logabsj_spearman': 0.98},
                'chart_quality': {
                    'trustworthiness': 0.85,
                    'knn_jaccard_mean': 0.05,
                    'effective_dimension': 1.0,
                },
            },
        }
        step4_anchors = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030': {
                'reconstruction': {'mse': 1.7e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.92},
                'chart_quality': {
                    'trustworthiness': 0.848,
                    'knn_jaccard_mean': 0.049,
                    'effective_dimension': 1.38,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.246,
                    'quotient_teacher_distill_loss': 0.0,
                },
            },
        }
        step6_anchors = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030': {
                'reconstruction': {'mse': 7.7e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.39},
                'chart_quality': {
                    'trustworthiness': 0.855,
                    'knn_jaccard_mean': 0.067,
                    'effective_dimension': 1.76,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.662,
                    'quotient_teacher_distill_loss': 0.4,
                },
            },
        }
        step7_anchors = {
            teacher_name: {
                'reconstruction': {'mse': 4e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.70},
                'chart_quality': {
                    'trustworthiness': 0.855,
                    'knn_jaccard_mean': 0.066,
                    'effective_dimension': 1.70,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.30,
                    'quotient_teacher_distill_loss': 0.0,
                },
            },
        }
        step8_anchors = {
            'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100': {
                'reconstruction': {'mse': 4e-5},
                'j_correlation': {'max_abs_logabsj_spearman': 0.90},
                'chart_quality': {
                    'trustworthiness': 0.85,
                    'knn_jaccard_mean': 0.058,
                    'effective_dimension': 1.55,
                },
                'factorized_consistency': {
                    'quotient_partner_rank_percentile_mean': 0.24,
                    'quotient_teacher_distill_loss': 0.2,
                },
            },
        }
        for path, payload in (
            (step3_anchor_path, step3_anchors),
            (step4_anchor_path, step4_anchors),
            (step6_anchor_path, step6_anchors),
            (step7_anchor_path, step7_anchors),
            (step8_anchor_path, step8_anchors),
        ):
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2)

        def tiny_step9_factory():
            config = _tiny_lattice_config(
                latent_type='factorized_vae',
                latent_dim=6,
                modular_invariance_weight=0.1,
            )
            config.model.vae_beta = 0.03
            config.train.chart_preserving_weight = 0.0
            config.train.quotient_spread_weight = 0.0
            config.train.quotient_variance_floor_weight = 0.0
            config.train.jacobian_gram_weight = 0.0
            config.train.quotient_logdet_weight = 0.0
            config.train.j_rank_preserving_weight = 0.03
            config.train.j_rank_n_terms = 10
            config.train.teacher_distill_weight = 0.03
            config.train.teacher_run_dir = teacher_dir
            config.train.teacher_distill_n_neighbors = 4
            return config

        summaries = run_step9_all(
            base_dir=tmpdir,
            experiments=[('lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld000', tiny_step9_factory)],
            summary_filename='tiny_step9_summaries.json',
            report_path=report_path,
            step3_anchor_summary_path=step3_anchor_path,
            step4_anchor_summary_path=step4_anchor_path,
            step6_anchor_summary_path=step6_anchor_path,
            step7_anchor_summary_path=step7_anchor_path,
            step8_anchor_summary_path=step8_anchor_path,
        )

        assert 'lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld000' in summaries
        summary = summaries['lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld000']
        assert 'factorized_consistency' in summary
        assert 'quotient_teacher_distill_loss' in summary['factorized_consistency']
        assert 'student_teacher_pairwise_distance_corr' in summary['factorized_consistency']

        with open(os.path.join(tmpdir, 'tiny_step9_summaries.json')) as f:
            saved = json.load(f)
        assert 'lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld000' in saved
        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_text = f.read()
        assert 'Step 9 Runs' in report_text
        assert 'teacher loss' in report_text


def test_topology_runner_requires_tda_dependencies():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('run_latent_topology_diagnostics.tda_dependencies_available', return_value=False):
            try:
                run_topology_all(
                    base_dir=tmpdir,
                    diagnostics_dir=os.path.join(tmpdir, 'topology_diagnostics'),
                    report_path=os.path.join(tmpdir, 'walkthrough-topology-phaseA.md'),
                    experiments=[],
                )
            except RuntimeError as exc:
                assert 'Topology diagnostics require optional TDA dependencies' in str(exc)
            else:
                raise AssertionError('Expected a missing-dependency RuntimeError')


def test_topology_runner_with_fake_backend():
    def fake_persistence_diagrams(points, maxdim):
        spread = max(float(np.std(points)), 1e-3)
        h0 = np.array([[0.0, spread]], dtype=float)
        h1 = (
            np.array([[0.1 * spread, 0.7 * spread]], dtype=float)
            if maxdim >= 1 and points.shape[1] >= 2 else
            np.zeros((0, 2), dtype=float)
        )
        return [h0, h1]

    def fake_diagram_distances(previous, current):
        if previous is None:
            return None
        return {
            'h0_bottleneck': 0.05,
            'h1_bottleneck': 0.07,
            'h0_wasserstein': 0.06,
            'h1_wasserstein': 0.08,
            'max_bottleneck': 0.07,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-topology-phaseA.md')

        def tiny_t2_factory():
            config = _tiny_config(latent_type='standard', torus_dim=2, latent_dim=4)
            config.train.num_epochs = 3
            return config

        def tiny_t2_torus_factory():
            config = _tiny_config(latent_type='torus', torus_dim=2, latent_dim=2)
            config.train.num_epochs = 3
            return config

        def tiny_lattice_factory():
            config = _tiny_lattice_config(
                latent_type='vae',
                latent_dim=4,
                modular_invariance_weight=0.1,
            )
            config.model.vae_beta = 0.01
            return config

        experiments = [
            {
                'name': 't2_standard',
                'kind': 'control',
                'config_source': tiny_t2_factory,
            },
            {
                'name': 't2_torus',
                'kind': 'control',
                'config_source': tiny_t2_torus_factory,
            },
            {
                'name': 'lattice_vae_norm_inv_b030_l100',
                'kind': 'lattice',
                'config_source': tiny_lattice_factory,
            },
        ]

        with patch('run_latent_topology_diagnostics.tda_dependencies_available', return_value=True):
            with patch('eval.topology._compute_persistence_diagrams', side_effect=fake_persistence_diagrams):
                with patch('eval.topology._compute_diagram_distance_metrics', side_effect=fake_diagram_distances):
                    combined = run_topology_all(
                        base_dir=tmpdir,
                        diagnostics_dir=os.path.join(tmpdir, 'topology_diagnostics'),
                        report_path=report_path,
                        experiments=experiments,
                    )

        assert 'topology_diagnostics' in combined
        assert 'branch_assessment' in combined
        assert 't2_standard' in combined['topology_diagnostics']
        assert 't2_torus' in combined['topology_diagnostics']
        assert 'lattice_vae_norm_inv_b030_l100' in combined['topology_diagnostics']
        dim2 = combined['topology_diagnostics']['lattice_vae_norm_inv_b030_l100']['topology_diagnostics']['dims']['2']
        assert 'partner_rank_percentile_mean' in dim2
        assert 'partner_knn_hit_rate' in dim2
        assert 'lid_valid_fraction' in dim2
        assert combined['topology_diagnostics']['lattice_vae_norm_inv_b030_l100']['diagram_payload'] == 'diagram_payload.npz'

        summary_path = os.path.join(
            tmpdir, 'topology_diagnostics', 'topology_diagnostics_summary.json',
        )
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            saved = json.load(f)
        assert 'topology_diagnostics' in saved

        run_plot = os.path.join(
            tmpdir, 'topology_diagnostics', 'lattice_vae_norm_inv_b030_l100',
            'metrics_vs_k.png',
        )
        assert os.path.exists(run_plot)
        assert os.path.exists(report_path)
        with open(report_path) as f:
            report_text = f.read()
        assert 'Evidence' in report_text
        assert 'k=2 rank' in report_text
        payload_path = os.path.join(
            tmpdir, 'topology_diagnostics', 'lattice_vae_norm_inv_b030_l100',
            'diagram_payload.npz',
        )
        assert os.path.exists(payload_path)


def test_topology_phaseb_runner_with_saved_phasea_outputs():
    def fake_persistence_diagrams(points, maxdim):
        spread = max(float(np.std(points)), 1e-3)
        h0 = np.array([[0.0, spread]], dtype=float)
        h1 = (
            np.array([[0.1 * spread, 0.7 * spread]], dtype=float)
            if maxdim >= 1 and points.shape[1] >= 2 else
            np.zeros((0, 2), dtype=float)
        )
        return [h0, h1]

    def fake_diagram_distances(previous, current):
        if previous is None:
            return None
        return {
            'h0_bottleneck': 0.05,
            'h1_bottleneck': 0.15,
            'h0_wasserstein': 0.06,
            'h1_wasserstein': 0.18,
            'max_bottleneck': 0.15,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        phasea_report = os.path.join(tmpdir, 'walkthrough-topology-phaseA.md')
        phaseb_report = os.path.join(tmpdir, 'walkthrough-topology-phaseB.md')
        roadmap_path = os.path.join(tmpdir, 'ae-latent-study-roadmap.md')

        def tiny_t2_factory():
            config = _tiny_config(latent_type='standard', torus_dim=2, latent_dim=4)
            config.train.num_epochs = 3
            return config

        def tiny_t2_torus_factory():
            config = _tiny_config(latent_type='torus', torus_dim=2, latent_dim=2)
            config.train.num_epochs = 3
            return config

        def tiny_lattice_factory():
            config = _tiny_lattice_config(
                latent_type='vae',
                latent_dim=4,
                modular_invariance_weight=0.1,
            )
            config.model.vae_beta = 0.01
            return config

        experiments = [
            {'name': 't2_standard', 'kind': 'control', 'config_source': tiny_t2_factory},
            {'name': 't2_torus', 'kind': 'control', 'config_source': tiny_t2_torus_factory},
            {'name': 'lattice_vae_norm_inv_b010_l100', 'kind': 'lattice', 'config_source': tiny_lattice_factory},
            {'name': 'lattice_vae_norm_inv_b030_l100', 'kind': 'lattice', 'config_source': tiny_lattice_factory},
        ]

        with patch('run_latent_topology_diagnostics.tda_dependencies_available', return_value=True):
            with patch('eval.topology._compute_persistence_diagrams', side_effect=fake_persistence_diagrams):
                with patch('eval.topology._compute_diagram_distance_metrics', side_effect=fake_diagram_distances):
                    run_topology_all(
                        base_dir=tmpdir,
                        diagnostics_dir=os.path.join(tmpdir, 'topology_diagnostics'),
                        report_path=phasea_report,
                        experiments=experiments,
                    )

        combined = run_topology_phaseb_all(
            base_dir=tmpdir,
            diagnostics_dir=os.path.join(tmpdir, 'topology_diagnostics'),
            summary_filename='phaseB_comparison_summary.json',
            report_path=phaseb_report,
            roadmap_path=roadmap_path,
            experiments=experiments,
        )

        assert combined['source_phaseA_branch']['branch'] in {'A', 'B', 'C', 'D', 'E'}
        assert 'phaseB_decision' in combined
        assert os.path.exists(os.path.join(tmpdir, 'topology_diagnostics', 'phaseB_comparison_summary.json'))
        assert os.path.exists(os.path.join(tmpdir, 'topology_diagnostics', 'phaseB_h1_trajectory.png'))
        assert os.path.exists(os.path.join(tmpdir, 'topology_diagnostics', 'phaseB_diagram_distance.png'))
        assert os.path.exists(os.path.join(tmpdir, 'topology_diagnostics', 'phaseB_diagram_grid_k2.png'))
        assert os.path.exists(os.path.join(tmpdir, 'topology_diagnostics', 'phaseB_diagram_grid_k1.png'))
        assert os.path.exists(phaseb_report)
        assert os.path.exists(roadmap_path)

        with open(phaseb_report) as f:
            phaseb_text = f.read()
        assert 'Primary branch' in phaseb_text
        assert 'PH Trajectory Comparison' in phaseb_text

        with open(roadmap_path) as f:
            roadmap_text = f.read()
        assert 'Current recommendation' in roadmap_text
        assert 'Active Branches' in roadmap_text


def test_topology_runner_factorized_uses_quotient_view():
    def fake_persistence_diagrams(points, maxdim):
        spread = max(float(np.std(points)), 1e-3)
        h0 = np.array([[0.0, spread]], dtype=float)
        h1 = (
            np.array([[0.1 * spread, 0.7 * spread]], dtype=float)
            if maxdim >= 1 and points.shape[1] >= 2 else
            np.zeros((0, 2), dtype=float)
        )
        return [h0, h1]

    def fake_diagram_distances(previous, current):
        if previous is None:
            return None
        return {
            'h0_bottleneck': 0.05,
            'h1_bottleneck': 0.07,
            'h0_wasserstein': 0.06,
            'h1_wasserstein': 0.08,
            'max_bottleneck': 0.07,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-topology-phaseA.md')

        def tiny_factorized_factory():
            return _tiny_lattice_config(
                latent_type='factorized_vae',
                latent_dim=6,
                modular_invariance_weight=0.1,
            )

        experiments = [
            {
                'name': 'lattice_factorized_vae_fd_b010_q100_g030_d030',
                'kind': 'lattice',
                'config_source': tiny_factorized_factory,
            },
        ]

        with patch('run_latent_topology_diagnostics.tda_dependencies_available', return_value=True):
            with patch('eval.topology._compute_persistence_diagrams', side_effect=fake_persistence_diagrams):
                with patch('eval.topology._compute_diagram_distance_metrics', side_effect=fake_diagram_distances):
                    combined = run_topology_all(
                        base_dir=tmpdir,
                        diagnostics_dir=os.path.join(tmpdir, 'topology_diagnostics'),
                        report_path=report_path,
                        experiments=experiments,
                    )

        summary = combined['topology_diagnostics']['lattice_factorized_vae_fd_b010_q100_g030_d030']
        assert summary['topology_diagnostics']['full_latent_dim'] == 2


def test_topology_runner_applies_config_overrides():
    def fake_persistence_diagrams(points, maxdim):
        spread = max(float(np.std(points)), 1e-3)
        h0 = np.array([[0.0, spread]], dtype=float)
        h1 = (
            np.array([[0.1 * spread, 0.7 * spread]], dtype=float)
            if maxdim >= 1 and points.shape[1] >= 2 else
            np.zeros((0, 2), dtype=float)
        )
        return [h0, h1]

    def fake_diagram_distances(previous, current):
        if previous is None:
            return None
        return {
            'h0_bottleneck': 0.05,
            'h1_bottleneck': 0.07,
            'h0_wasserstein': 0.06,
            'h1_wasserstein': 0.08,
            'max_bottleneck': 0.07,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'walkthrough-topology-phaseA.md')

        def tiny_factorized_factory():
            return _tiny_lattice_config(
                latent_type='factorized_vae',
                latent_dim=6,
                modular_invariance_weight=0.1,
            )

        experiments = [
            {
                'name': 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010',
                'kind': 'lattice',
                'config_source': tiny_factorized_factory,
                'config_overrides': {'eval': {'ph_proj_dims': (2, 1), 'ph_random_projection_trials': 4}},
            },
        ]

        with patch('run_latent_topology_diagnostics.tda_dependencies_available', return_value=True):
            with patch('eval.topology._compute_persistence_diagrams', side_effect=fake_persistence_diagrams):
                with patch('eval.topology._compute_diagram_distance_metrics', side_effect=fake_diagram_distances):
                    combined = run_topology_all(
                        base_dir=tmpdir,
                        diagnostics_dir=os.path.join(tmpdir, 'topology_diagnostics'),
                        report_path=report_path,
                        experiments=experiments,
                    )

        dims = combined['topology_diagnostics'][
            'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010'
        ]['topology_diagnostics']['dims']
        assert set(dims.keys()) == {'2', '1'}


def test_topology_step7_followup_runner():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = os.path.join(tmpdir, 'lattice_step7_summaries.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010': {
                    'chart_quality': {
                        'effective_dimension': 1.8,
                        'knn_jaccard_mean': 0.06,
                        'trustworthiness': 0.85,
                    },
                    'j_correlation': {'max_abs_logabsj_spearman': 0.91},
                    'reconstruction': {'mse': 4e-5},
                    'factorized_consistency': {
                        'quotient_partner_rank_percentile_mean': 0.31,
                        'decoder_equivariance_mse': 8e-4,
                    },
                },
            }, f, indent=2)

        phasea_payload = {'branch_assessment': {'branch': 'A'}}
        phaseb_payload = {'phaseB_decision': {'primary_branch': 'A1'}}

        with patch('run_topology_step7_followup.run_topology_all', return_value=phasea_payload) as mock_phasea:
            with patch('run_topology_step7_followup.run_topology_phaseb_all', return_value=phaseb_payload) as mock_phaseb:
                combined = run_topology_step7_followup_all(
                    base_dir=tmpdir,
                    step7_summary_path=summary_path,
                    diagnostics_dir=os.path.join(tmpdir, 'topology_diagnostics_step7'),
                    phasea_report_path=os.path.join(tmpdir, 'walkthrough-topology-step7-phaseA.md'),
                    phaseb_report_path=os.path.join(tmpdir, 'walkthrough-topology-step7-phaseB.md'),
                    roadmap_path=os.path.join(tmpdir, 'ae-latent-study-roadmap-step7.md'),
                )

        assert combined['winner_name'] == 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010'
        assert combined['winner_used_gate'] is True
        phasea_experiments = mock_phasea.call_args.kwargs['experiments']
        assert phasea_experiments[-1]['name'] == combined['winner_name']
        assert phasea_experiments[0]['config_overrides']['eval']['ph_max_samples'] == 1500
        assert phasea_experiments[0]['config_overrides']['eval']['ph_random_projection_trials'] == 4
        assert mock_phaseb.call_args.kwargs['experiments'][-1]['name'] == combined['winner_name']
        assert os.path.exists(os.path.join(tmpdir, 'topology_diagnostics_step7', 'step7_followup_summary.json'))


def test_topology_step8_followup_runner():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = os.path.join(tmpdir, 'lattice_step8_summaries.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r030': {
                    'chart_quality': {
                        'effective_dimension': 1.8,
                        'knn_jaccard_mean': 0.06,
                        'trustworthiness': 0.85,
                    },
                    'j_correlation': {'max_abs_logabsj_spearman': 0.91},
                    'modular_invariance': {'mean_latent_distance': 0.01},
                    'reconstruction': {'mse': 4e-5},
                    'factorized_consistency': {
                        'quotient_partner_rank_percentile_mean': 0.22,
                        'decoder_equivariance_mse': 4e-4,
                    },
                },
            }, f, indent=2)

        phasea_payload = {'branch_assessment': {'branch': 'A'}}
        phaseb_payload = {'phaseB_decision': {'primary_branch': 'A1'}}

        with patch('run_topology_step8_followup.run_topology_all', return_value=phasea_payload) as mock_phasea:
            with patch('run_topology_step8_followup.run_topology_phaseb_all', return_value=phaseb_payload) as mock_phaseb:
                combined = run_topology_step8_followup_all(
                    base_dir=tmpdir,
                    step8_summary_path=summary_path,
                    diagnostics_dir=os.path.join(tmpdir, 'topology_diagnostics_step8'),
                    phasea_report_path=os.path.join(tmpdir, 'walkthrough-topology-step8-phaseA.md'),
                    phaseb_report_path=os.path.join(tmpdir, 'walkthrough-topology-step8-phaseB.md'),
                    roadmap_path=os.path.join(tmpdir, 'ae-latent-study-roadmap-step8.md'),
                )

        assert combined['winner_name'] == 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r030'
        assert combined['winner_used_gate'] is True
        phasea_experiments = mock_phasea.call_args.kwargs['experiments']
        assert phasea_experiments[-1]['name'] == combined['winner_name']
        assert any(
            exp['name'] == 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010'
            for exp in phasea_experiments
        )
        assert phasea_experiments[0]['config_overrides']['eval']['ph_max_samples'] == 1500
        assert phasea_experiments[0]['config_overrides']['eval']['ph_random_projection_trials'] == 4
        assert mock_phaseb.call_args.kwargs['experiments'][-1]['name'] == combined['winner_name']
        assert os.path.exists(os.path.join(tmpdir, 'topology_diagnostics_step8', 'step8_followup_summary.json'))


def test_topology_step9_followup_runner():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = os.path.join(tmpdir, 'lattice_step9_summaries.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld010': {
                    'chart_quality': {
                        'effective_dimension': 1.8,
                        'knn_jaccard_mean': 0.06,
                        'trustworthiness': 0.85,
                    },
                    'j_correlation': {'max_abs_logabsj_spearman': 0.91},
                    'modular_invariance': {'mean_latent_distance': 0.01},
                    'reconstruction': {'mse': 4e-5},
                    'factorized_consistency': {
                        'quotient_partner_rank_percentile_mean': 0.18,
                        'quotient_teacher_distill_loss': 0.04,
                        'decoder_equivariance_mse': 4e-4,
                    },
                },
            }, f, indent=2)

        phasea_payload = {'branch_assessment': {'branch': 'A'}}
        phaseb_payload = {'phaseB_decision': {'primary_branch': 'A1'}}

        with patch('run_topology_step9_followup.run_topology_all', return_value=phasea_payload) as mock_phasea:
            with patch('run_topology_step9_followup.run_topology_phaseb_all', return_value=phaseb_payload) as mock_phaseb:
                combined = run_topology_step9_followup_all(
                    base_dir=tmpdir,
                    step9_summary_path=summary_path,
                    diagnostics_dir=os.path.join(tmpdir, 'topology_diagnostics_step9'),
                    phasea_report_path=os.path.join(tmpdir, 'walkthrough-topology-step9-phaseA.md'),
                    phaseb_report_path=os.path.join(tmpdir, 'walkthrough-topology-step9-phaseB.md'),
                    roadmap_path=os.path.join(tmpdir, 'ae-latent-study-roadmap-step9.md'),
                )

        assert combined['winner_name'] == 'lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld010'
        assert combined['winner_used_gate'] is True
        phasea_experiments = mock_phasea.call_args.kwargs['experiments']
        assert phasea_experiments[-1]['name'] == combined['winner_name']
        assert any(
            exp['name'] == 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100'
            for exp in phasea_experiments
        )
        assert phasea_experiments[0]['config_overrides']['eval']['ph_max_samples'] == 1500
        assert phasea_experiments[0]['config_overrides']['eval']['ph_random_projection_trials'] == 4
        assert mock_phaseb.call_args.kwargs['experiments'][-1]['name'] == combined['winner_name']
        assert os.path.exists(os.path.join(tmpdir, 'topology_diagnostics_step9', 'step9_followup_summary.json'))
