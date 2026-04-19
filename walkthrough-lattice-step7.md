# Lattice Step 7 — Jacobian-like Quotient Metric Regularizer

## Summary

- Step 7 keeps the Step 4 factorized scaffold fixed and replaces the Step 5/6 local term with a Jacobian-like local Gram match.
- A weak global logdet floor + trace cap is added only as a collapse guard.
- Selection first applies the Step 6 gate: quotient partner rank <= 0.35 and `log10|j|` Spearman >= 0.88.

## Anchors

| Run | Source | q rank | q trust | q overlap | q eff_dim | log10_abs_j Spearman | MSE |
|---|---|---|---|---|---|---|---|
| `lattice_standard_norm_inv` | Step 3 | missing | missing | missing | missing | missing | missing |
| `lattice_vae_norm_inv_b010_l100` | Step 3 | nan | 0.8534 | 0.0576 | 2.0096 | 0.9590 | 8.10962e-05 |
| `lattice_vae_norm_inv_b030_l100` | Step 3 | nan | 0.8556 | 0.0673 | 1.8469 | 0.8138 | 8.09051e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030` | Step 4 | 0.2460 | 0.8477 | 0.0486 | 1.3854 | 0.9250 | 1.72509e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030` | Step 6 | 0.6616 | 0.8555 | 0.0667 | 1.7600 | 0.3940 | 7.7346e-05 |

## Step 7 Runs

| Run | Gate | q rank | q trust | q overlap | q eff_dim | jacobian loss | logdet loss | q eig_min | q eig_max | log10_abs_j Spearman | decode mse | MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010` | FAIL | 0.1036 | 0.8532 | 0.0568 | 1.6285 | 5.977479 | 0.022819 | 0.029760 | 0.084202 | 0.8155 | 0.000157 | 9.53359e-06 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld030` | FAIL | 0.1202 | 0.8540 | 0.0588 | 1.6800 | 9.773455 | 0.000000 | 0.034097 | 0.087703 | 0.8542 | 0.000274 | 8.1135e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j030_ld010` | FAIL | 0.1283 | 0.8544 | 0.0593 | 1.5848 | 4.323659 | 0.038522 | 0.027745 | 0.086325 | 0.8335 | 0.000284 | 3.92673e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j030_ld030` | FAIL | 0.1217 | 0.8548 | 0.0587 | 1.5804 | 4.855347 | 0.001933 | 0.029826 | 0.093513 | 0.8167 | 0.000232 | 1.55792e-05 |

## Selected Run

- Selected run: `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010`
- Gate status: no run passed the gate; selected best fallback while keeping A2 active
- Quotient chart: eff_dim = 1.6285, overlap = 0.0568, trust = 0.8532
- Partner / `j` retention: rank = 0.1036, Spearman = 0.8155
- Jacobian / logdet: jacobian loss = 5.977479, logdet loss = 0.022819
- Decoder / reconstruction: decoder mse = 0.000157, reconstruction MSE = 9.53359e-06

## Interpretation

- Read Step 7 as an A2 test of whether a Jacobian-like local metric term can improve quotient spread without losing the Step 4 partner-rank and `j` retention gains.
- After selecting the Step 7 winner, run the full topology anchor rerun and update the roadmap snapshot in the same phase.
