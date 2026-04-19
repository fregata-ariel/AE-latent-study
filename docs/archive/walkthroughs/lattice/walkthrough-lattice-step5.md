# Lattice Step 5 — Quotient Chart-Preserving Regularizer

## Summary

- Step 5 keeps the Step 4 factorized scaffold fixed and adds quotient-only chart regularization.
- Selection rule: quotient effective-dimension penalty, quotient overlap, quotient partner rank, quotient trust, `log10|j|` Spearman, gauge equivariance, decoder equivariance, then reconstruction MSE.

## Step 4 Anchor

| Run | q rank | q trust | q overlap | q eff_dim | q var0 | q var1 | log10_abs_j Spearman | MSE |
|---|---|---|---|---|---|---|---|---|
| `lattice_factorized_vae_fd_b030_q100_g030_d030` | 0.2460 | 0.8477 | 0.0486 | 1.3854 | nan | nan | 0.9250 | 1.72509e-05 |

## Step 5 Runs

| Run | q rank | q trust | q overlap | q eff_dim | q var0 | q var1 | q chart loss | log10_abs_j Spearman | gauge mse | decode mse | MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lattice_factorized_vae_fd_b030_q100_g030_d030_c010_v000` | 0.6588 | 0.8557 | 0.0714 | 1.2509 | 0.0000 | 0.0000 | 0.926917 | 0.8480 | 0.000041 | 0.001625 | 8.12711e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_c030_v000` | 0.6578 | 0.8544 | 0.0723 | 1.3704 | 0.0000 | 0.0000 | 0.909134 | 0.7778 | 0.000077 | 0.001604 | 8.13576e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_c030_v010` | 0.6568 | 0.8531 | 0.0719 | 1.3655 | 0.0000 | 0.0000 | 0.944956 | 0.8977 | 0.000052 | 0.001706 | 8.15381e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_c050_v010` | 0.6621 | 0.8457 | 0.0561 | 1.2072 | 0.0000 | 0.0000 | 0.733878 | 0.7421 | 0.000090 | 0.001607 | 8.14209e-05 |

## Selected Run

- Selected run: `lattice_factorized_vae_fd_b030_q100_g030_d030_c030_v000`
- Quotient chart: eff_dim = 1.3704, overlap = 0.0723, trust = 0.8544
- Quotient spread: var0 = 0.0000, var1 = 0.0000, chart loss = 0.909134, variance-floor loss = 0.022500
- Partner / `j` retention: rank = 0.6578, Spearman = 0.7778
- Gauge / decoder consistency: gauge mse = 0.000077, decoder mse = 0.001604
- Reconstruction: MSE = 8.13576e-05

## Interpretation

- Read Step 5 as a focused test of whether quotient-only geometry regularization can widen the chart without breaking partner alignment or gauge consistency.
- Topology Phase A/B should be run only for the Step 5 selected run against the fixed Step 3/4 anchors.
