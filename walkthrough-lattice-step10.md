# Lattice Step 10 — τ_F Supervised Contrastive Local Geometry

## Summary

- Step 10 keeps the factorized VAE scaffold fixed and replaces teacher/Jacobian local terms with τ_F kNN supervised contrastive geometry.
- `j` rank retention remains as a weak semantic guard, and logdet is used only as an optional volume guard.
- The goal is to improve quotient overlap / topology stability without losing Step 7-level partner rank.

## Anchors

| Run | Source | q rank | q trust | q overlap | q eff_dim | log10_abs_j Spearman | contrastive loss | MSE |
|---|---|---|---|---|---|---|---|---|
| `lattice_standard_norm_inv` | Step 3 | missing | missing | missing | missing | missing | missing | missing |
| `lattice_vae_norm_inv_b010_l100` | Step 3 | nan | 0.8534 | 0.0576 | 2.0096 | 0.9590 | nan | 8.10962e-05 |
| `lattice_vae_norm_inv_b030_l100` | Step 3 | nan | 0.8556 | 0.0673 | 1.8469 | 0.8138 | nan | 8.09051e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030` | Step 4 | 0.2460 | 0.8477 | 0.0486 | 1.3854 | 0.9250 | nan | 1.72509e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030` | Step 6 | 0.6616 | 0.8555 | 0.0667 | 1.7600 | 0.3940 | nan | 7.7346e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010` | Step 7 | 0.1036 | 0.8532 | 0.0568 | 1.6285 | 0.8155 | nan | 9.53359e-06 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100` | Step 8 | 0.1077 | 0.8504 | 0.0498 | 1.5934 | 0.9990 | nan | 7.71642e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld010` | Step 9 | 0.1022 | 0.8508 | 0.0485 | 1.6662 | 0.9961 | nan | 9.00884e-06 |

## Step 10 Runs

| Run | Gate | q rank | q trust | q overlap | q eff_dim | contrastive loss | j-rank loss | logdet loss | log10_abs_j Spearman | mod dist | decode mse | MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lattice_factorized_vae_fd_b030_q100_g030_d030_cl010_r030_ld010` | FAIL | 0.0774 | 0.8497 | 0.0472 | 1.7447 | 5.335452 | 0.082421 | 0.000000 | 0.8325 | 0.026982 | 0.000226 | 6.79212e-06 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_cl030_r030_ld010` | FAIL | 0.0874 | 0.8484 | 0.0453 | 1.9655 | 4.990370 | 0.086925 | 0.000000 | 0.8469 | 0.044901 | 0.000215 | 1.45912e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_cl010_r100_ld010` | FAIL | 0.1064 | 0.8506 | 0.0501 | 1.7407 | 5.380695 | 0.093542 | 0.000000 | 0.9967 | 0.041870 | 0.000333 | 0.000101145 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_cl030_r100_ld010` | FAIL | 0.1076 | 0.8484 | 0.0447 | 1.8028 | 5.205260 | 0.078024 | 0.000000 | 0.8864 | 0.052722 | 0.000267 | 6.50597e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_cl010_r030_ld000` | PASS | 0.0934 | 0.8531 | 0.0583 | 1.5518 | 5.419578 | 0.080589 | 0.147077 | 0.9796 | 0.031339 | 0.000277 | 2.13519e-05 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_cl030_r030_ld000` | FAIL | 0.0870 | 0.8507 | 0.0476 | 1.0081 | 5.265966 | 0.089735 | 4.781955 | 0.9974 | 0.036143 | 0.000255 | 3.85333e-05 |

## Selected Run

- Selected run: `lattice_factorized_vae_fd_b030_q100_g030_d030_cl010_r030_ld000`
- Gate status: passed
- Contrastive local loss = 5.419578
- Quotient chart: eff_dim = 1.5518, overlap = 0.0583, trust = 0.8531
- Partner / `j`: rank = 0.0934, Spearman = 0.9796
- Modular / decoder / reconstruction: mod dist = 0.031339, decoder mse = 0.000277, MSE = 2.13519e-05

## Interpretation

- If a Step 10 run passes the gate, run the topology follow-up and check whether focus metrics improve Step 9 overlap or topology-side `j` while keeping Step 7-level partner rank.
- If no run passes the gate, keep A2 active and move toward stronger contrastive / semantic geometry.
