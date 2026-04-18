"""Unit tests for latent topology diagnostics."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from configs.default import get_config as get_torus_config
from configs.lattice_default import get_config as get_lattice_config
from data.dataset import Dataset
from eval.topology import (
    compute_local_intrinsic_dimension,
    diagnose_projection_ladder,
    make_reference_coords,
)


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
    assert 'pca_diagrams' in artifacts
