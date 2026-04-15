# Lattice Step 3 — VAE + Invariance Sweep

## Summary

- Step 3 tests whether VAE KL pressure and modular invariance can jointly improve orbit gluing and 2D chart quality.
- Ranking rule: modular distance, then distance of effective dimension from 2, then chart overlap, trustworthiness, `log10|j|` Spearman, and reconstruction MSE.
- Success gate requires simultaneously strong orbit gluing, strong chart quality, strong `j` retention, and low reconstruction error.

## Step 2 Anchors

| Run | MSE | log10_abs_j Spearman | ModDist | trust | overlap | eff_dim |
|---|---|---|---|---|---|---|
| `lattice_standard_norm_inv` | missing | missing | missing | missing | missing | missing |
| `lattice_standard_wide_norm_inv` | missing | missing | missing | missing | missing | missing |
| `lattice_vae_norm_beta001` | missing | missing | missing | missing | missing | missing |

## Fundamental-Domain VAE + Invariance

| Run | MSE | log10_abs_j Spearman | ModDist | trust | overlap | eff_dim | gate |
|---|---|---|---|---|---|---|---|
| `lattice_vae_norm_inv_b003_l030` | 8.1109e-05 | 0.7786 | 0.0002 | 0.8544 | 0.0608 | 1.7299 | FAIL |
| `lattice_vae_norm_inv_b003_l100` | 8.09374e-05 | 0.8155 | 0.0001 | 0.8535 | 0.0598 | 1.0998 | FAIL |
| `lattice_vae_norm_inv_b010_l030` | 8.08364e-05 | 0.5285 | 0.0002 | 0.8543 | 0.0622 | 1.4708 | FAIL |
| `lattice_vae_norm_inv_b010_l100` | 8.10962e-05 | 0.9590 | 0.0001 | 0.8534 | 0.0576 | 2.0096 | FAIL |
| `lattice_vae_norm_inv_b030_l030` | 8.09336e-05 | 0.7095 | 0.0001 | 0.8531 | 0.0579 | 2.5656 | FAIL |
| `lattice_vae_norm_inv_b030_l100` | 8.09051e-05 | 0.8138 | 0.0001 | 0.8556 | 0.0673 | 1.8469 | FAIL |

- Best run: `lattice_vae_norm_inv_b030_l100`
- Orbit gluing: mean modular distance = 0.0001
- 2D chart: eff_dim = 1.8469, overlap = 0.0673, trust = 0.8556
- `j` retention: Spearman = 0.8138
- Reconstruction: MSE = 8.09051e-05
- Success gate: FAIL

## Wide Half-Plane VAE + Invariance

| Run | MSE | log10_abs_j Spearman | ModDist | trust | overlap | eff_dim | gate |
|---|---|---|---|---|---|---|---|
| `lattice_vae_wide_norm_inv_b003_l030` | 0.000874424 | 0.8511 | 0.0004 | 0.8855 | 0.0531 | 1.9625 | FAIL |
| `lattice_vae_wide_norm_inv_b003_l100` | 0.00087445 | 0.8184 | 0.0001 | 0.8848 | 0.0556 | 2.2629 | FAIL |
| `lattice_vae_wide_norm_inv_b010_l030` | 0.000874157 | 0.6043 | 0.0004 | 0.8859 | 0.0510 | 2.3417 | FAIL |
| `lattice_vae_wide_norm_inv_b010_l100` | 0.000874123 | 0.8906 | 0.0004 | 0.8835 | 0.0509 | 2.1297 | FAIL |
| `lattice_vae_wide_norm_inv_b030_l030` | 0.000874616 | 0.6068 | 0.0001 | 0.8836 | 0.0520 | 2.5646 | FAIL |
| `lattice_vae_wide_norm_inv_b030_l100` | 0.000874462 | 0.5808 | 0.0005 | 0.8853 | 0.0534 | 1.2913 | FAIL |

- Best run: `lattice_vae_wide_norm_inv_b030_l030`
- Orbit gluing: mean modular distance = 0.0001
- 2D chart: eff_dim = 2.5646, overlap = 0.0520, trust = 0.8836
- `j` retention: Spearman = 0.6068
- Reconstruction: MSE = 0.000874616
- Success gate: FAIL

## Selected Run

- Selected run: `lattice_vae_norm_inv_b030_l100`
- Orbit gluing: mean modular distance = 0.0001
- 2D chart: eff_dim = 1.8469, overlap = 0.0673, trust = 0.8556
- `j` retention: Spearman = 0.8138
- Reconstruction: MSE = 8.09051e-05
- Success gate: FAIL

## Conclusion

- No Step 3 run passed the success gate. Treat this as evidence that a Gaussian VAE prior alone is insufficient, and move next to equivariant latent design.
