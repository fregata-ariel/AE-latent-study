# Lattice Step 2 — Normalization, Invariance, and Capacity Sweep

## Summary

- Step 2 compares normalized lattice signals, modular-invariance regularization, and latent-capacity expansions.
- Ranking rule: smallest modular mean distance, then largest `log10|j|` Spearman correlation, then smallest reconstruction MSE.
- Quotient-chart metrics (`trust`, `overlap`, `eff_dim`) are supplementary and track how natural the learned 2D chart looks.
- Normalized runs should be compared against one another, not directly against raw Step 1 MSE values.

## Step 1 Baseline

| Experiment | MSE | max abs corr vs Re/Im(j) | SL₂(Z) mean dist |
|---|---|---|---|
| lattice_standard | 4.25569e-08 | 0.0081 | 1.0194 |
| lattice_halfplane | 2.00637e-07 | 0.0076 | 2.0525 |
| lattice_standard_wide | 5.33878e-06 | 0.0170 | 0.9578 |

## Phase 1

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | trust | overlap | eff_dim | PCA EVR |
|---|---|---|---|---|---|---|---|---|
| lattice_standard_norm | 2.73822e-07 | 0.9835 | 3.9800 | 0.0829 | 0.8505 | 0.0477 | 1.0013 | 1.000, 0.000 |
| lattice_halfplane_norm | 6.11513e-07 | 0.9831 | 4.0307 | 0.0969 | 0.8398 | 0.0387 | 1.0000 | 0.997, 0.003 |
| lattice_standard_wide_norm | 1.30654e-06 | 0.8881 | 2.7555 | 0.1457 | 0.8803 | 0.0480 | 1.0219 | 0.989, 0.011 |

- Best run: `lattice_standard_norm`
- Reason: modular mean distance = 0.0829, log10|j| Spearman = 0.9835, MSE = 2.73822e-07
- Orbit-gluing view: smaller modular mean distance is better.
- 2D chart view: higher trust/overlap and effective dimension closer to 2 are better. (best run here: trust = 0.8505, overlap = 0.0477, eff_dim = 1.0013)

## Phase 2

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | trust | overlap | eff_dim | PCA EVR |
|---|---|---|---|---|---|---|---|---|
| lattice_standard_norm_inv | 1.12377e-07 | 0.9879 | 4.1715 | 0.0004 | 0.8528 | 0.0570 | 1.0438 | 0.979, 0.021 |
| lattice_standard_wide_norm_inv | 1.70056e-06 | 0.7322 | 2.8229 | 0.0044 | 0.8820 | 0.0462 | 1.2332 | 0.897, 0.103 |

- Best run: `lattice_standard_norm_inv`
- Reason: modular mean distance = 0.0004, log10|j| Spearman = 0.9879, MSE = 1.12377e-07
- Orbit-gluing view: smaller modular mean distance is better.
- 2D chart view: higher trust/overlap and effective dimension closer to 2 are better. (best run here: trust = 0.8528, overlap = 0.0570, eff_dim = 1.0438)

## Phase 3

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | trust | overlap | eff_dim | PCA EVR |
|---|---|---|---|---|---|---|---|---|
| lattice_standard_norm_latent4 | 4.62227e-07 | 0.9871 | 4.0545 | 0.2500 | 0.8513 | 0.0481 | 1.0022 | 1.000, 0.000 |
| lattice_standard_norm_latent8 | 4.8303e-07 | 0.9895 | 4.2734 | 0.2489 | 0.8515 | 0.0489 | 1.0025 | 1.000, 0.000 |
| lattice_vae_norm_beta001 | 8.08377e-05 | 0.4673 | 4.4144 | 0.0072 | 0.8545 | 0.0575 | 1.3867 | 0.831, 0.153 |

- Best run: `lattice_vae_norm_beta001`
- Reason: modular mean distance = 0.0072, log10|j| Spearman = 0.4673, MSE = 8.08377e-05
- Orbit-gluing view: smaller modular mean distance is better.
- 2D chart view: higher trust/overlap and effective dimension closer to 2 are better. (best run here: trust = 0.8545, overlap = 0.0575, eff_dim = 1.3867)

## Adopted Run

- Selected run: `lattice_standard_norm_inv`
- Modular mean distance: 0.0004
- max Spearman vs log10|j|: 0.9879
- Quotient-chart trust / overlap / eff_dim: 0.8528 / 0.0570 / 1.0438
- Reconstruction MSE: 1.12377e-07

## Open Issues

- Differentiable projection to the fundamental domain is still out of scope for Step 2.
- Half-plane latent is only retained as a Phase 1 comparison point.
- Normalized runs should not be compared directly to raw Step 1 MSE values.
