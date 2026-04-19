# Lattice Step 4 — Factorized VAE

## Summary

- Step 4 makes the quotient chart explicit by splitting the latent into `quotient(2) + gauge(4)` parts.
- Ranking rule: quotient partner rank, quotient effective-dimension penalty, quotient overlap, quotient trust, gauge equivariance, decoder equivariance, `log10|j|` Spearman, then reconstruction MSE.

## Step 3 Anchors

| Run | MSE | log10_abs_j Spearman | ModDist | trust | overlap | eff_dim |
|---|---|---|---|---|---|---|
| `lattice_standard_norm_inv` | missing | missing | missing | missing | missing | missing |
| `lattice_vae_norm_inv_b010_l100` | 8.10962e-05 | 0.9590 | 0.0001 | 0.8534 | 0.0576 | 2.0096 |
| `lattice_vae_norm_inv_b030_l100` | 8.09051e-05 | 0.8138 | 0.0001 | 0.8556 | 0.0673 | 1.8469 |
| `lattice_vae_wide_norm_inv_b003_l030` | 0.000874424 | 0.8511 | 0.0004 | 0.8855 | 0.0531 | 1.9625 |

## Step 4 Runs

| Run | q rank | q hit | q trust | q overlap | q eff_dim | gauge mse | decode mse | log10_abs_j Spearman | MSE |
|---|---|---|---|---|---|---|---|---|---|
| `lattice_factorized_vae_fd_b010_q100_g030_d030` | 0.2611 | 0.3344 | 0.8462 | 0.0427 | 1.3815 | 0.000027 | 0.000273 | 0.8660 | 3.14505e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030` | 0.2460 | 0.3336 | 0.8477 | 0.0486 | 1.3854 | 0.000014 | 0.000321 | 0.9250 | 1.72509e-05 |
| `lattice_factorized_vae_wide_b030_q100_g030_d030` | 0.2540 | 0.3360 | 0.8810 | 0.0462 | 1.0594 | 0.000049 | 0.001149 | 0.9754 | 4.21955e-05 |

## Selected Run

- Selected run: `lattice_factorized_vae_fd_b030_q100_g030_d030`
- Quotient chart: eff_dim = 1.3854, overlap = 0.0486, trust = 0.8477
- Quotient partner preservation: rank = 0.2460, hit = 0.3336
- Gauge / decoder consistency: gauge mse = 0.000014, decoder mse = 0.000321
- `j` retention: Spearman = 0.9250
- Reconstruction: MSE = 1.72509e-05

## Interpretation

- Read Step 4 as a direct test of whether an explicit quotient/gauge split is easier to compare than the Step 3 monolithic VAE latent.
- The next topology pass should compare the selected factorized run against the Step 3 anchors using the quotient view only.
