# Topology Diagnostics Phase A

## Summary

- Persistent Homology is used here as a diagnostic for projection stability, not as a proof of the true quotient topology.
- Branch classification: **C**
- Interpretation: The pure-torus control does not yet show the expected 2D-stable / 1D-collapse pattern.
- Recommended next step: Debug the PH pipeline, projection ladder, and noise-floor choices before using lattice conclusions.

## Control Calibration

| Run | k=4 trust | k=2 trust | k=2 overlap | k=1 trust | k=1 overlap |
|---|---|---|---|---|---|
| `t2_standard` | 0.9996 | 0.8937 | 0.1465 | 0.6704 | 0.0256 |
| `t2_torus` | 0.9923 | 0.8624 | 0.1944 | 0.6905 | 0.0330 |

## Lattice Representatives

| Run | full-k moddist | k=2 trust | k=2 overlap | k=2 eff_dim | k=2 H1 total | k=1 moddist | k=1 Spearman |
|---|---|---|---|---|---|---|---|
| `lattice_standard_norm` | 3.565 | 0.8521 | 0.0518 | 1.0116 | 0.9989 | 1.883 | 0.9874 |
| `lattice_standard_norm_inv` | 0.1765 | 0.8521 | 0.0502 | 1.0033 | 0.6972 | 0.1705 | 0.9780 |
| `lattice_vae_norm_beta001` | 110.6 | 0.8531 | 0.0519 | 1.9875 | 1.8111 | 35.66 | 0.1021 |
| `lattice_vae_norm_inv_b010_l100` | 5.024 | 0.8475 | 0.0518 | 1.8289 | 4.9647 | 2.386 | 0.0485 |
| `lattice_vae_norm_inv_b030_l100` | 4.38 | 0.8503 | 0.0569 | 1.9993 | 6.3616 | 1.057 | 0.0478 |
| `lattice_vae_wide_norm_inv_b003_l030` | 2.655 | 0.8749 | 0.0511 | 1.9673 | 7.8128 | 0.8666 | 0.8359 |

## Branch Outcome

- Branch: `C`
- Summary: The pure-torus control does not yet show the expected 2D-stable / 1D-collapse pattern.
- Next step: Debug the PH pipeline, projection ladder, and noise-floor choices before using lattice conclusions.
