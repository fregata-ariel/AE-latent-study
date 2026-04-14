# Lattice Step 2 — Normalization, Invariance, and Capacity Sweep

## Summary

- Step 2 compares normalized lattice signals, modular-invariance regularization, and latent-capacity expansions.
- Ranking rule: smallest modular mean distance, then largest `log10|j|` Spearman correlation, then smallest reconstruction MSE.
- Normalized runs should be compared against one another, not directly against raw Step 1 MSE values.

## Step 1 Baseline

| Experiment | MSE | max abs corr vs Re/Im(j) | SL₂(Z) mean dist |
|---|---|---|---|

## Phase 1

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | PCA EVR |
|---|---|---|---|---|---|
| lattice_standard_norm | 2.89401e-07 | 0.9834 | 3.9925 | 0.0944 | - |
| lattice_halfplane_norm | 6.10116e-07 | 0.9830 | 4.0311 | 0.0930 | - |
| lattice_standard_wide_norm | 1.2288e-06 | 0.8869 | 2.7423 | 0.1537 | - |

- Best run: `lattice_halfplane_norm`
- Reason: modular mean distance = 0.0930, log10|j| Spearman = 0.9830, MSE = 6.10116e-07

## Phase 2

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | PCA EVR |
|---|---|---|---|---|---|
| lattice_standard_norm_inv | 4.1832e-07 | 0.9825 | 4.1880 | 0.0010 | - |
| lattice_standard_wide_norm_inv | 1.30117e-06 | 0.7548 | 2.6745 | 0.0042 | - |

- Best run: `lattice_standard_norm_inv`
- Reason: modular mean distance = 0.0010, log10|j| Spearman = 0.9825, MSE = 4.1832e-07

## Phase 3

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | PCA EVR |
|---|---|---|---|---|---|
| lattice_standard_norm_latent4 | 4.64934e-07 | 0.9871 | 4.0687 | 0.2500 | 0.999, 0.001 |
| lattice_standard_norm_latent8 | 2.88902e-08 | 0.9905 | 4.3985 | 0.1925 | 0.992, 0.008 |
| lattice_vae_norm_beta001 | 8.09567e-05 | 0.6280 | 4.4373 | 0.0074 | 0.668, 0.319 |

- Best run: `lattice_vae_norm_beta001`
- Reason: modular mean distance = 0.0074, log10|j| Spearman = 0.6280, MSE = 8.09567e-05

## Adopted Run

- Selected run: `lattice_standard_norm_inv`
- Modular mean distance: 0.0010
- max Spearman vs log10|j|: 0.9825
- Reconstruction MSE: 4.1832e-07

## Open Issues

- Differentiable projection to the fundamental domain is still out of scope for Step 2.
- Half-plane latent is only retained as a Phase 1 comparison point.
- Normalized runs should not be compared directly to raw Step 1 MSE values.
