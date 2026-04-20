# Lattice Step 8 — Step 7b Pairwise log|j| Rank Retention

## Summary

- Step 8 keeps the Step 7 factorized scaffold and `Local Gram match + logdet guard` structure.
- The new diagnostic term preserves pairwise `log10|j|` ordering along a batch-adaptive quotient direction.
- The goal is to test whether Step 7 lost global quotient semantics rather than proving the whole A2 regularizer family failed.

## Anchors

| Run | Source | q rank | q trust | q overlap | q eff_dim | log10_abs_j Spearman | j-rank loss | MSE |
|---|---|---|---|---|---|---|---|---|
| `lattice_standard_norm_inv` | Step 3 | missing | missing | missing | missing | missing | missing | missing |
| `lattice_vae_norm_inv_b010_l100` | Step 3 | nan | 0.8534 | 0.0576 | 2.0096 | 0.9590 | nan | 8.10962e-05 |
| `lattice_vae_norm_inv_b030_l100` | Step 3 | nan | 0.8556 | 0.0673 | 1.8469 | 0.8138 | nan | 8.09051e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030` | Step 4 | 0.2460 | 0.8477 | 0.0486 | 1.3854 | 0.9250 | nan | 1.72509e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030` | Step 6 | 0.6616 | 0.8555 | 0.0667 | 1.7600 | 0.3940 | nan | 7.7346e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010` | Step 7 | 0.1036 | 0.8532 | 0.0568 | 1.6285 | 0.8155 | nan | 9.53359e-06 |

## Step 8 Runs

| Run | Gate | q rank | q trust | q overlap | q eff_dim | jacobian loss | logdet loss | j-rank loss | j target std | log10_abs_j Spearman | mod dist | decode mse | MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r030` | PASS | 0.1273 | 0.8491 | 0.0420 | 1.6033 | 19.030798 | 0.027567 | 0.106190 | 1.6478 | 0.8927 | 0.037377 | 0.000283 | 3.67491e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100` | PASS | 0.1077 | 0.8504 | 0.0498 | 1.5934 | 6.509302 | 0.051275 | 0.093508 | 1.6478 | 0.9990 | 0.038818 | 0.000284 | 7.71642e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld000_r030` | FAIL | 0.6222 | 0.8538 | 0.0572 | 1.5946 | 0.582799 | 390.028926 | 0.203002 | 1.6478 | 0.9942 | 0.009056 | 0.000547 | 8.7145e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j005_ld010_r030` | FAIL | 0.0980 | 0.8487 | 0.0405 | 1.7135 | 10.682091 | 0.015969 | 0.106040 | 1.6478 | 0.8321 | 0.026556 | 0.000209 | 5.35752e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j005_ld000_r100` | FAIL | 0.5755 | 0.8510 | 0.0539 | 1.0067 | 0.582588 | 334.256946 | 0.235455 | 1.6478 | 0.9943 | 0.010333 | 0.000666 | 7.09783e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r030_k4` | FAIL | 0.1603 | 0.8504 | 0.0480 | 1.3363 | 5.105306 | 0.038112 | 0.130314 | 1.6478 | 0.9222 | 0.058596 | 0.000305 | 6.12013e-05 |

## Selected Run

- Selected run: `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100`
- Gate status: passed
- `j` retention: Spearman = 0.9990, j-rank loss = 0.093508
- Quotient chart: eff_dim = 1.5934, overlap = 0.0498, trust = 0.8504
- Partner / modular: rank = 0.1077, modular distance = 0.038818
- Local / global losses: jacobian = 6.509302, logdet = 0.051275
- Decoder / reconstruction: decoder mse = 0.000284, reconstruction MSE = 7.71642e-05

## Interpretation

- If the selected run restores `log10|j|` Spearman while keeping Step 7 chart spread, Step 7 failure was likely semantic erosion rather than the whole Jacobian-like family failing.
- If no run passes the gate, treat this as evidence that small Step 7 variants are insufficient and keep A2 active toward teacher distillation or contrastive local geometry.
