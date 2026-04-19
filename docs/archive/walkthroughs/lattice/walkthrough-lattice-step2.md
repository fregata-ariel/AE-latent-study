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
| lattice_standard_norm | 2.89401e-07 | 0.9834 | 3.9925 | 0.0944 | 0.8508 | 0.0483 | 1.0015 | 1.000, 0.000 |
| lattice_halfplane_norm | 6.10116e-07 | 0.9830 | 4.0311 | 0.0930 | 0.8403 | 0.0386 | 1.0000 | 0.995, 0.005 |
| lattice_standard_wide_norm | 1.2288e-06 | 0.8869 | 2.7423 | 0.1537 | 0.8831 | 0.0489 | 1.0246 | 0.987, 0.013 |

- Best run: `lattice_halfplane_norm`
- Reason: modular mean distance = 0.0930, log10|j| Spearman = 0.9830, MSE = 6.10116e-07
- Orbit-gluing view: smaller modular mean distance is better.
- 2D chart view: higher trust/overlap and effective dimension closer to 2 are better. (best run here: trust = 0.8403, overlap = 0.0386, eff_dim = 1.0000)

## Phase 2

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | trust | overlap | eff_dim | PCA EVR |
|---|---|---|---|---|---|---|---|---|
| lattice_standard_norm_inv | 4.18319e-07 | 0.9825 | 4.1880 | 0.0010 | 0.8519 | 0.0504 | 1.0029 | 0.998, 0.002 |
| lattice_standard_wide_norm_inv | 1.30117e-06 | 0.7548 | 2.6745 | 0.0042 | 0.8792 | 0.0470 | 1.3264 | 0.858, 0.142 |

- Best run: `lattice_standard_norm_inv`
- Reason: modular mean distance = 0.0010, log10|j| Spearman = 0.9825, MSE = 4.18319e-07
- Orbit-gluing view: smaller modular mean distance is better.
- 2D chart view: higher trust/overlap and effective dimension closer to 2 are better. (best run here: trust = 0.8519, overlap = 0.0504, eff_dim = 1.0029)

## Phase 3

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | trust | overlap | eff_dim | PCA EVR |
|---|---|---|---|---|---|---|---|---|
| lattice_standard_norm_latent4 | 4.64934e-07 | 0.9871 | 4.0687 | 0.2500 | 0.8516 | 0.0482 | 1.0022 | 0.999, 0.001 |
| lattice_standard_norm_latent8 | 2.88902e-08 | 0.9905 | 4.3985 | 0.1925 | 0.8522 | 0.0550 | 1.0160 | 0.992, 0.008 |
| lattice_vae_norm_beta001 | 8.09567e-05 | 0.6280 | 4.4373 | 0.0074 | 0.8549 | 0.0577 | 1.8271 | 0.668, 0.319 |

- Best run: `lattice_vae_norm_beta001`
- Reason: modular mean distance = 0.0074, log10|j| Spearman = 0.6280, MSE = 8.09567e-05
- Orbit-gluing view: smaller modular mean distance is better.
- 2D chart view: higher trust/overlap and effective dimension closer to 2 are better. (best run here: trust = 0.8549, overlap = 0.0577, eff_dim = 1.8271)

## Adopted Run

- Selected run: `lattice_standard_norm_inv`
- Modular mean distance: 0.0010
- max Spearman vs log10|j|: 0.9825
- Quotient-chart trust / overlap / eff_dim: 0.8519 / 0.0504 / 1.0029
- Reconstruction MSE: 4.18319e-07

## Open Issues

- Differentiable projection to the fundamental domain is still out of scope for Step 2.
- Half-plane latent is only retained as a Phase 1 comparison point.
- Normalized runs should not be compared directly to raw Step 1 MSE values.
