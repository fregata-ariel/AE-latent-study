# Lattice Step 6 — Rotation-Aware Quotient Spread Regularizer

## Summary

- Step 6 keeps the Step 4 factorized scaffold fixed and replaces the Step 5 axis-wise spread term with a covariance-eigenvalue spread loss.
- Selection rule first applies a gate: quotient partner rank <= 0.35 and `log10|j|` Spearman >= 0.88.
- Among gated runs, ranking uses quotient effective-dimension penalty, quotient partner rank, quotient overlap, quotient trust, decoder equivariance, then reconstruction MSE.

## Anchors

| Run | Source | q rank | q trust | q overlap | q eff_dim | log10_abs_j Spearman | MSE |
|---|---|---|---|---|---|---|---|
| `lattice_standard_norm_inv` | Step 3 | missing | missing | missing | missing | missing | missing |
| `lattice_vae_norm_inv_b010_l100` | Step 3 | nan | 0.8534 | 0.0576 | 2.0096 | 0.9590 | 8.10962e-05 |
| `lattice_vae_norm_inv_b030_l100` | Step 3 | nan | 0.8556 | 0.0673 | 1.8469 | 0.8138 | 8.09051e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030` | Step 4 | 0.2460 | 0.8477 | 0.0486 | 1.3854 | 0.9250 | 1.72509e-05 |

## Step 6 Runs

| Run | Gate | q rank | q trust | q overlap | q eff_dim | q eig_min | q eig_max | q spread loss | log10_abs_j Spearman | decode mse | MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l010_s010` | FAIL | 0.6553 | 0.8474 | 0.0606 | 1.3868 | 0.000000 | 0.000000 | 0.000277 | 0.8901 | 0.001696 | 8.21291e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l010_s030` | FAIL | 0.6565 | 0.8540 | 0.0623 | 1.4053 | 0.000000 | 0.000000 | 0.000277 | 0.7036 | 0.000989 | 7.61864e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030` | FAIL | 0.6616 | 0.8555 | 0.0667 | 1.7600 | 0.000000 | 0.000000 | 0.000277 | 0.3940 | 0.001440 | 7.7346e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s050` | FAIL | 0.6559 | 0.8552 | 0.0713 | 1.3428 | 0.000000 | 0.000000 | 0.000277 | 0.8316 | 0.001601 | 8.13764e-05 |

## Selected Run

- Selected run: `lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030`
- Gate status: no run passed the gate; selected best fallback while keeping A2 active
- Quotient chart: eff_dim = 1.7600, overlap = 0.0667, trust = 0.8555
- Partner / `j` retention: rank = 0.6616, Spearman = 0.3940
- Spread metrics: eig_min = 0.000000, eig_max = 0.000000, spread loss = 0.000277
- Decoder / reconstruction: decoder mse = 0.001440, reconstruction MSE = 7.7346e-05

## Interpretation

- Read Step 6 as an A2 test of whether rotation-aware spread can stop quotient collapse without giving up the Step 4 partner-rank and `j` retention gains.
- Only the Step 6 selected run should be sent to topology Phase A/B against the fixed Step 3/4 anchors.
- If no run passes the gate, keep `A2` active and redesign the local chart term itself before revisiting topology.
