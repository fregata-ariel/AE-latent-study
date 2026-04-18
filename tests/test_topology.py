"""Unit tests for latent topology diagnostics."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from configs.default import get_config as get_torus_config
from configs.lattice_default import get_config as get_lattice_config
from data.dataset import Dataset
from eval.topology import (
    _compute_partner_preservation_metrics,
    _summarize_diagrams,
    compute_local_intrinsic_dimension,
    diagnose_projection_ladder,
    make_reference_coords,
)
from run_latent_topology_diagnostics import classify_branch


def _fake_persistence_diagrams(points: np.ndarray, maxdim: int):
    spread = max(float(np.std(points)), 1e-3)
    h0 = np.array([[0.0, spread]], dtype=float)
    h1 = (
        np.array([[0.2 * spread, 0.8 * spread]], dtype=float)
        if maxdim >= 1 and points.shape[1] >= 2 else
        np.zeros((0, 2), dtype=float)
    )
    return [h0, h1]


def _fake_diagram_distances(previous, current):
    if previous is None:
        return None
    return {
        'h0_bottleneck': 0.05,
        'h1_bottleneck': 0.07,
        'h0_wasserstein': 0.06,
        'h1_wasserstein': 0.08,
        'max_bottleneck': 0.07,
    }


def test_local_intrinsic_dimension_line_vs_plane():
    x = np.linspace(-1.0, 1.0, 128)
    line = np.stack([x, np.zeros_like(x)], axis=-1)
    plane = np.stack(np.meshgrid(
        np.linspace(-1.0, 1.0, 16),
        np.linspace(-1.0, 1.0, 16),
        indexing='xy',
    ), axis=-1).reshape(-1, 2)

    line_stats = compute_local_intrinsic_dimension(line, n_neighbors=10)
    plane_stats = compute_local_intrinsic_dimension(plane, n_neighbors=10)

    assert 1.0 < line_stats['median'] < 1.6
    assert 2.0 < plane_stats['median'] < 3.2
    assert plane_stats['median'] > line_stats['median']
    assert line_stats['valid_fraction'] > 0.9
    assert plane_stats['valid_fraction'] > 0.9


def test_local_intrinsic_dimension_handles_duplicate_points():
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(24, 2))
    cloud[::3] = cloud[0]

    stats = compute_local_intrinsic_dimension(cloud, n_neighbors=8)

    assert stats['valid_fraction'] < 1.0
    assert stats['valid_fraction'] > 0.0
    assert np.isfinite(stats['median'])


def test_partner_preservation_metrics_are_scale_and_rotation_invariant():
    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    latent = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    partner = latent + 0.02 * np.stack([np.cos(angles + 0.2), np.sin(angles + 0.2)], axis=-1)

    theta = np.pi / 3.0
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    rotated = latent @ rotation.T
    rotated_partner = partner @ rotation.T
    scaled = 3.5 * latent
    scaled_partner = 3.5 * partner

    base = _compute_partner_preservation_metrics(latent, partner, n_neighbors=4)
    rotated_metrics = _compute_partner_preservation_metrics(rotated, rotated_partner, n_neighbors=4)
    scaled_metrics = _compute_partner_preservation_metrics(scaled, scaled_partner, n_neighbors=4)

    assert np.isclose(base['partner_rank_percentile_mean'], rotated_metrics['partner_rank_percentile_mean'])
    assert np.isclose(base['partner_rank_percentile_mean'], scaled_metrics['partner_rank_percentile_mean'])
    assert np.isclose(base['partner_knn_hit_rate'], rotated_metrics['partner_knn_hit_rate'])
    assert np.isclose(base['partner_knn_hit_rate'], scaled_metrics['partner_knn_hit_rate'])


def test_partner_preservation_metrics_improve_when_partners_are_closer():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(32, 3))
    close_partner = latent + 0.01 * rng.normal(size=(32, 3))
    far_partner = latent + 1.0 * rng.normal(size=(32, 3))

    close_metrics = _compute_partner_preservation_metrics(latent, close_partner, n_neighbors=5)
    far_metrics = _compute_partner_preservation_metrics(latent, far_partner, n_neighbors=5)

    assert close_metrics['partner_rank_percentile_mean'] < far_metrics['partner_rank_percentile_mean']
    assert close_metrics['partner_knn_hit_rate'] > far_metrics['partner_knn_hit_rate']


def test_relative_noise_floor_is_scale_invariant():
    diagrams = [
        np.array([[0.0, 0.4], [0.0, 0.2]], dtype=float),
        np.array([[0.1, 0.6], [0.2, 0.3]], dtype=float),
    ]
    scaled = [2.5 * diagrams[0], 2.5 * diagrams[1]]

    base = _summarize_diagrams(diagrams, noise_floor=0.5, noise_floor_mode='relative')
    scaled_summary = _summarize_diagrams(scaled, noise_floor=0.5, noise_floor_mode='relative')

    assert base['h0_bar_count'] == scaled_summary['h0_bar_count']
    assert base['h1_bar_count'] == scaled_summary['h1_bar_count']
    assert np.isclose(
        scaled_summary['noise_floor_value_h1'],
        2.5 * base['noise_floor_value_h1'],
    )


def test_make_reference_coords_torus_embedding():
    config = get_torus_config()
    config.data.torus_dim = 2
    dataset = Dataset(
        signals=jnp.ones((4, 10)),
        thetas=jnp.asarray([
            [0.0, 0.0],
            [np.pi / 2.0, np.pi],
            [np.pi, np.pi / 3.0],
            [3.0 * np.pi / 2.0, np.pi / 4.0],
        ]),
    )

    reference_coords, label = make_reference_coords(dataset, config)

    assert label == 'torus_cos_sin'
    assert reference_coords.shape == (4, 4)


def test_make_reference_coords_lattice_reduces_to_fundamental_domain():
    config = get_lattice_config()
    tau = np.array([0.8 + 1.0j, -1.1 + 1.2j, 0.3 + 2.0j], dtype=np.complex128)
    dataset = Dataset(
        signals=jnp.ones((3, 10)),
        thetas=jnp.asarray(np.stack([tau.real, tau.imag], axis=-1)),
        tau=tau,
    )

    reference_coords, label = make_reference_coords(dataset, config)

    assert label == 'euclidean_fd'
    assert reference_coords.shape == (3, 2)
    assert np.all(np.abs(reference_coords[:, 0]) <= 0.5 + 1e-6)
    assert np.all(reference_coords[:, 1] > 0.0)


def test_diagnose_projection_ladder_requires_tda_dependencies():
    z = np.random.default_rng(0).normal(size=(32, 4))
    ref = np.random.default_rng(1).normal(size=(32, 2))

    with patch('eval.topology.tda_dependencies_available', return_value=False):
        try:
            diagnose_projection_ladder(
                z, ref, projection_dims=(4, 2, 1), max_samples=0,
            )
        except RuntimeError as exc:
            assert 'Persistent-homology diagnostics require optional dependencies' in str(exc)
        else:
            raise AssertionError('Expected a missing-dependency RuntimeError')


def test_diagnose_projection_ladder_schema_with_fake_backend():
    z = np.random.default_rng(0).normal(size=(48, 4))
    ref = np.random.default_rng(1).normal(size=(48, 2))
    j_values = np.exp(np.linspace(0.0, 4.0, 48)).astype(np.complex128)
    partner = z + 0.01 * np.random.default_rng(2).normal(size=(48, 4))

    with patch('eval.topology._compute_persistence_diagrams', side_effect=_fake_persistence_diagrams):
        with patch('eval.topology._compute_diagram_distance_metrics', side_effect=_fake_diagram_distances):
            summary, artifacts = diagnose_projection_ladder(
                z,
                ref,
                projection_dims=(4, 2, 1),
                max_samples=0,
                j_values=j_values,
                partner_latent=partner,
                noise_floor_mode='relative',
                random_projection_trials=2,
            )

    assert summary['projection_basis'] == 'pca'
    assert summary['n_samples'] == 48
    assert set(summary['dims'].keys()) == {'4', '2', '1'}
    assert 'random_projection_baseline' in summary['dims']['2']
    assert 'trustworthiness' in summary['dims']['2']
    assert 'diagram_distance_to_prev' in summary['dims']['2']
    assert 'max_abs_logabsj_spearman' in summary['dims']['2']
    assert 'projected_modular_distance' in summary['dims']['2']
    assert 'partner_rank_percentile_mean' in summary['dims']['2']
    assert 'partner_knn_hit_rate' in summary['dims']['2']
    assert 'lid_valid_fraction' in summary['dims']['2']
    assert 'noise_floor_value_h0' in summary['dims']['2']
    assert 'pca_diagrams' in artifacts


def _make_dim(
    *,
    effective_dimension,
    trustworthiness,
    overlap,
    h1_total,
    spearman=None,
    partner_rank=None,
    partner_hit=None,
):
    return {
        'effective_dimension': effective_dimension,
        'trustworthiness': trustworthiness,
        'knn_jaccard_mean': overlap,
        'h1_total_persistence': h1_total,
        'max_abs_logabsj_spearman': spearman,
        'partner_rank_percentile_mean': partner_rank,
        'partner_knn_hit_rate': partner_hit,
        'random_projection_baseline': {
            'knn_jaccard_mean': {'mean': overlap - 0.02, 'std': 0.01},
            'partner_knn_hit_rate': {'mean': (partner_hit or 0.0) - 0.05, 'std': 0.01},
            'max_abs_logabsj_spearman': {'mean': (spearman or 0.0) - 0.05, 'std': 0.01},
        },
    }


def _make_run(name, dims):
    return {
        'name': name,
        'kind': 'control' if name.startswith('t2_') else 'lattice',
        'topology_diagnostics': {'dims': {str(k): v for k, v in dims.items()}},
    }


def test_classify_branch_can_return_c_a_and_e():
    control_dims = {
        4: _make_dim(effective_dimension=3.0, trustworthiness=0.99, overlap=0.60, h1_total=10.0),
        2: _make_dim(effective_dimension=2.0, trustworthiness=0.88, overlap=0.18, h1_total=4.0),
        1: _make_dim(effective_dimension=1.0, trustworthiness=0.68, overlap=0.05, h1_total=0.1),
    }
    stable_dims = {
        4: _make_dim(
            effective_dimension=2.1, trustworthiness=0.86, overlap=0.07, h1_total=5.0,
            spearman=0.90, partner_rank=0.06, partner_hit=0.42,
        ),
        2: _make_dim(
            effective_dimension=2.0, trustworthiness=0.84, overlap=0.06, h1_total=4.2,
            spearman=0.85, partner_rank=0.08, partner_hit=0.38,
        ),
        1: _make_dim(
            effective_dimension=1.0, trustworthiness=0.75, overlap=0.03, h1_total=0.0,
            spearman=0.30, partner_rank=0.25, partner_hit=0.10,
        ),
    }
    unstable_dims = {
        4: _make_dim(
            effective_dimension=2.2, trustworthiness=0.86, overlap=0.07, h1_total=5.0,
            spearman=0.90, partner_rank=0.07, partner_hit=0.40,
        ),
        2: _make_dim(
            effective_dimension=1.2, trustworthiness=0.78, overlap=0.03, h1_total=1.0,
            spearman=0.40, partner_rank=0.20, partner_hit=0.05,
        ),
        1: _make_dim(
            effective_dimension=1.0, trustworthiness=0.74, overlap=0.02, h1_total=0.0,
            spearman=0.20, partner_rank=0.35, partner_hit=0.01,
        ),
    }

    branch_c = classify_branch({
        't2_standard': _make_run('t2_standard', {
            4: _make_dim(effective_dimension=3.0, trustworthiness=0.99, overlap=0.60, h1_total=10.0),
            2: _make_dim(effective_dimension=1.3, trustworthiness=0.80, overlap=0.10, h1_total=4.0),
            1: _make_dim(effective_dimension=1.0, trustworthiness=0.74, overlap=0.05, h1_total=0.2),
        }),
        't2_torus': _make_run('t2_torus', control_dims),
    })
    assert branch_c['branch'] == 'C'

    branch_a = classify_branch({
        't2_standard': _make_run('t2_standard', control_dims),
        't2_torus': _make_run('t2_torus', control_dims),
        'lattice_vae_norm_inv_b010_l100': _make_run('lattice_vae_norm_inv_b010_l100', stable_dims),
        'lattice_vae_norm_inv_b030_l100': _make_run('lattice_vae_norm_inv_b030_l100', stable_dims),
        'lattice_vae_wide_norm_inv_b003_l030': _make_run('lattice_vae_wide_norm_inv_b003_l030', stable_dims),
    })
    assert branch_a['branch'] == 'A'

    branch_e = classify_branch({
        't2_standard': _make_run('t2_standard', control_dims),
        't2_torus': _make_run('t2_torus', control_dims),
        'lattice_vae_norm_inv_b010_l100': _make_run('lattice_vae_norm_inv_b010_l100', unstable_dims),
        'lattice_vae_norm_inv_b030_l100': _make_run('lattice_vae_norm_inv_b030_l100', unstable_dims),
        'lattice_vae_wide_norm_inv_b003_l030': _make_run('lattice_vae_wide_norm_inv_b003_l030', stable_dims),
    })
    assert branch_e['branch'] == 'E'
