"""Smoke test for lattice experiment pipeline.

Quick validation that all components work together:
- Data generation (fundamental domain sampling + theta function)
- Model creation (standard + halfplane)
- Training (5 epochs)
- Evaluation (lattice-specific metrics + plots)
"""

import os
import sys
import numpy as np

# Minimal test of data generation
print("=" * 60)
print("  SMOKE TEST: Lattice Pipeline")
print("=" * 60)

# 1. Test data generation functions
print("\n[1/6] Testing data generation functions...")
from data.generation import (
    sample_fundamental_domain,
    sample_upper_halfplane,
    reduce_to_fundamental_domain,
    generate_lattice_theta,
    compute_j_invariant,
)
import jax

key = jax.random.PRNGKey(0)

# Fundamental domain sampling
tau_f = sample_fundamental_domain(100, y_max=3.0, key=key)
assert len(tau_f) == 100, f"Expected 100 samples, got {len(tau_f)}"
assert all(abs(t.real) <= 0.5 + 1e-6 for t in tau_f), "Re(τ) out of range"
assert all(abs(t) >= 1.0 - 1e-6 for t in tau_f), "|τ| < 1 found"
assert all(t.imag > 0 for t in tau_f), "Im(τ) ≤ 0 found"
print(f"  ✓ sample_fundamental_domain: {len(tau_f)} samples, all in F")

# Upper half-plane sampling
tau_h = sample_upper_halfplane(50, key=key)
assert len(tau_h) == 50
assert all(t.imag > 0 for t in tau_h)
print(f"  ✓ sample_upper_halfplane: {len(tau_h)} samples")

# Reduce to fundamental domain
tau_reduced = reduce_to_fundamental_domain(tau_h)
for t in tau_reduced:
    assert abs(t.real) <= 0.5 + 1e-4, f"Re(τ)={t.real:.4f} out of range"
    assert abs(t) >= 1.0 - 1e-4, f"|τ|={abs(t):.4f} < 1"
print(f"  ✓ reduce_to_fundamental_domain: all {len(tau_reduced)} in F")

# Theta function
signals = generate_lattice_theta(tau_f[:10], signal_length=50, t_min=0.5, t_max=3.0, K=5)
assert signals.shape == (10, 50), f"Shape mismatch: {signals.shape}"
assert np.all(np.isfinite(signals)), "Non-finite values in theta function"
print(f"  ✓ generate_lattice_theta: shape={signals.shape}")

# j-invariant
j_vals = compute_j_invariant(tau_f[:10])
assert len(j_vals) == 10
# Check j(i) ≈ 1728 (τ = i is the square lattice)
j_i = compute_j_invariant(np.array([1j]))
print(f"  ✓ compute_j_invariant: j(i) = {j_i[0]:.1f} (expected ≈ 1728)")


# 2. Test dataset creation
print("\n[2/6] Testing dataset creation...")
from configs.lattice_standard import get_config
config = get_config()
# Make it tiny
config.data.n_train = 50
config.data.n_val = 20
config.data.n_test = 20
config.data.lattice_K = 5  # faster computation
config.train.num_epochs = 5
config.train.batch_size = 16  # must be < n_train to produce batches
config.train.patience = 100  # disable early stopping for smoke test
config.train.log_every = 1

from data.dataset import create_splits
key = jax.random.PRNGKey(42)
train_ds, val_ds, test_ds = create_splits(config, key)
print(f"  ✓ create_splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
print(f"    signals shape: {train_ds.signals.shape}")
print(f"    thetas shape:  {train_ds.thetas.shape}")
print(f"    j_invariant:   {'present' if train_ds.j_invariant is not None else 'None'}")
print(f"    tau:           {'present' if train_ds.tau is not None else 'None'}")


# 3. Test model creation (standard)
print("\n[3/6] Testing model creation (standard)...")
from models import create_model
model_std = create_model(config)
dummy = jax.numpy.ones((1, config.data.signal_length))
params = model_std.init(jax.random.PRNGKey(0), dummy)
x_hat, z = model_std.apply(params, dummy)
print(f"  ✓ Standard AE: input={dummy.shape} → latent={z.shape} → output={x_hat.shape}")


# 4. Test model creation (halfplane)
print("\n[4/6] Testing model creation (halfplane)...")
config_hp = get_config()
config_hp.model.latent_type = 'halfplane'
config_hp.data.n_train = 50
config_hp.data.n_val = 10
config_hp.data.n_test = 10
model_hp = create_model(config_hp)
params_hp = model_hp.init(jax.random.PRNGKey(0), dummy)
x_hat_hp, z_hp = model_hp.apply(params_hp, dummy)
assert z_hp.shape == (1, 2), f"Half-plane latent shape: {z_hp.shape}"
assert float(z_hp[0, 1]) > 0, f"Im(τ) = {float(z_hp[0, 1]):.4f} ≤ 0!"
print(f"  ✓ HalfPlane AE: latent={z_hp.shape}, Im(τ)={float(z_hp[0, 1]):.4f} > 0")


# 5. Test full training pipeline (tiny)
print("\n[5/6] Testing full training pipeline (5 epochs)...")
workdir = 'runs/_smoke_test_lattice'
os.makedirs(workdir, exist_ok=True)

from train.trainer import train_and_evaluate
state, history, (tr, _, te) = train_and_evaluate(config, workdir)
print(f"  ✓ Training complete: {len(history['train_loss'])} epochs")
print(f"    Final train_loss: {history['train_loss'][-1]:.6f}")
print(f"    Final val_loss:   {history['val_loss'][-1]:.6f}")


# 6. Test full evaluation pipeline
print("\n[6/6] Testing full evaluation pipeline...")
from eval.analysis import run_full_evaluation
summary = run_full_evaluation(state, config, tr, te, history, workdir)
print(f"  ✓ Evaluation complete")
print(f"    MSE: {summary['reconstruction']['mse']:.6f}")
if 'j_correlation' in summary:
    print(f"    j-corr max: {summary['j_correlation']['max_abs_correlation']:.4f}")
if 'modular_invariance' in summary:
    print(f"    SL₂(Z) mean dist: {summary['modular_invariance']['mean_latent_distance']:.6f}")

# Check output files exist
results_dir = os.path.join(workdir, 'results')
expected_files = [
    'training_curves.png',
    'reconstructions.png',
    'lattice_latent_scatter.png',
    'j_invariant_correlation.png',
    'summary.json',
]
for fname in expected_files:
    fpath = os.path.join(results_dir, fname)
    exists = os.path.exists(fpath)
    status = '✓' if exists else '✗'
    print(f"  {status} {fname}")

print(f"\n{'='*60}")
print("  SMOKE TEST PASSED ✓")
print(f"{'='*60}")
