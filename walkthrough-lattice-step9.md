# Lattice Step 9 — Step 7 Teacher Quotient-Structure Distillation

## Summary

- Step 9 keeps the factorized VAE scaffold fixed and uses the Step 7 winner as a frozen teacher.
- The teacher target is local quotient distance structure, not raw quotient coordinates.
- Step 9 keeps `j` rank retention and optional logdet guard, while disabling Step 5/6/7 local/spread terms.

## Anchors

| Run | Source | q rank | q trust | q overlap | q eff_dim | log10_abs_j Spearman | teacher loss | MSE |
|---|---|---|---|---|---|---|---|---|
| `lattice_standard_norm_inv` | Step 3 | missing | missing | missing | missing | missing | missing | missing |
| `lattice_vae_norm_inv_b010_l100` | Step 3 | nan | 0.8534 | 0.0576 | 2.0096 | 0.9590 | nan | 8.10962e-05 |
| `lattice_vae_norm_inv_b030_l100` | Step 3 | nan | 0.8556 | 0.0673 | 1.8469 | 0.8138 | nan | 8.09051e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030` | Step 4 | 0.2460 | 0.8477 | 0.0486 | 1.3854 | 0.9250 | nan | 1.72509e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030` | Step 6 | 0.6616 | 0.8555 | 0.0667 | 1.7600 | 0.3940 | nan | 7.7346e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010` | Step 7 teacher | 0.1036 | 0.8532 | 0.0568 | 1.6285 | 0.8155 | nan | 9.53359e-06 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100` | Step 8 winner | 0.1077 | 0.8504 | 0.0498 | 1.5934 | 0.9990 | nan | 7.71642e-05 |

## Step 9 Runs

| Run | Gate | q rank | q trust | q overlap | q eff_dim | teacher loss | pairwise corr | j-rank loss | logdet loss | log10_abs_j Spearman | mod dist | decode mse | MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld010` | FAIL | 0.1022 | 0.8508 | 0.0485 | 1.6662 | 0.059317 | 0.9747 | 0.104891 | 0.000086 | 0.9961 | 0.035499 | 0.000257 | 9.00884e-06 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_td100_r030_ld010` | FAIL | 0.1169 | 0.8539 | 0.0583 | 1.3387 | 0.043905 | 0.9798 | 0.235411 | 0.028568 | 0.9786 | 0.049462 | 0.000341 | 0.000134057 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r100_ld010` | FAIL | 0.1162 | 0.8504 | 0.0519 | 1.8091 | 0.066339 | 0.9435 | 0.086926 | 0.071685 | 0.9984 | 0.045708 | 0.000260 | 7.16748e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_td100_r100_ld010` | FAIL | 0.1058 | 0.8525 | 0.0565 | 1.5840 | 0.035273 | 0.9530 | 0.091298 | 0.008666 | 0.9987 | 0.034976 | 0.000194 | 1.87919e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld000` | FAIL | 0.1912 | 0.8538 | 0.0600 | 1.3016 | 0.111330 | 0.9406 | 0.104195 | 2.971064 | 0.9942 | 0.074820 | 0.000393 | 0.0003029 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_td100_r030_ld000` | FAIL | 0.1188 | 0.8526 | 0.0577 | 1.1796 | 0.119398 | 0.9481 | 0.278109 | 1.516095 | 0.9801 | 0.055374 | 0.000291 | 8.42548e-05 |

## Selected Run

- Selected run: `lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld010`
- Gate status: no run passed the gate; selected best fallback and keep A2 active
- Teacher distillation: loss = 0.059317, pairwise corr = 0.9747
- Quotient chart: eff_dim = 1.6662, overlap = 0.0485, trust = 0.8508
- Partner / `j`: rank = 0.1022, Spearman = 0.9961
- Modular / decoder / reconstruction: mod dist = 0.035499, decoder mse = 0.000257, MSE = 9.00884e-06

## Interpretation

- If a Step 9 run passes the gate, run the topology follow-up and check whether it improves Step 8 topology-side `j` retention while preserving Step 7 partner/chart quality.
- If no run passes the gate, treat teacher distillation as insufficient and move A2 to contrastive local geometry.
