# Lattice Step 2 — Normalization, Invariance, and Capacity Sweep

## Summary

- Step 2 compares normalized lattice signals, modular-invariance regularization, and latent-capacity expansions.
- Ranking rule: smallest modular mean distance, then largest `log10|j|` Spearman correlation, then smallest reconstruction MSE.
- Normalized runs should be compared against one another, not directly against raw Step 1 MSE values.
- This file is overwritten by `run_lattice_step2_experiments.py` after the full Step 2 sweep completes.

## Step 1 Baseline

| Experiment | MSE | max abs corr vs Re/Im(j) | SL₂(Z) mean dist |
|---|---|---|---|
| `lattice_standard` | 4.25569e-08 | 0.0081 | 1.0194 |
| `lattice_halfplane` | 2.00637e-07 | 0.0076 | 2.0525 |
| `lattice_standard_wide` | 5.33878e-06 | 0.0170 | 0.9578 |

## Phase 1

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | PCA EVR |
|---|---|---|---|---|---|
| `lattice_standard_norm` | pending | pending | pending | pending | pending |
| `lattice_halfplane_norm` | pending | pending | pending | pending | pending |
| `lattice_standard_wide_norm` | pending | pending | pending | pending | pending |

- Best run: pending full Step 2 execution.

## Phase 2

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | PCA EVR |
|---|---|---|---|---|---|
| `lattice_standard_norm_inv` | pending | pending | pending | pending | pending |
| `lattice_standard_wide_norm_inv` | pending | pending | pending | pending | pending |

- Best run: pending full Step 2 execution.

## Phase 3

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | PCA EVR |
|---|---|---|---|---|---|
| `lattice_standard_norm_latent4` | pending | pending | pending | pending | pending |
| `lattice_standard_norm_latent8` | pending | pending | pending | pending | pending |
| `lattice_vae_norm_beta001` | pending | pending | pending | pending | pending |

- Best run: pending full Step 2 execution.

## Adopted Run

- Pending Step 2 execution.

## Open Issues

- Differentiable projection to the fundamental domain is still out of scope for Step 2.
- Half-plane latent is only retained as a Phase 1 comparison point.
- Normalized runs should not be compared directly to raw Step 1 MSE values.
