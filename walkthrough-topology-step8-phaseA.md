# Topology Diagnostics Phase A

## Summary

- Persistent Homology is used here as a diagnostic for projection stability, not as a proof of the true quotient topology.
- Branch classification: **A**
- Interpretation: The best VAE+invariance runs remain comparatively stable down to k=2, then collapse at k=1.
- Recommended next step: Move to equivariant or factorized latent models that explicitly preserve a 2D quotient chart.

## Control Calibration

| Run | k=4 trust | k=2 trust | k=1 trust | Δtrust 4→2 | Δtrust 2→1 | k=2 overlap | k=1 overlap | Δoverlap 2→1 |
|---|---|---|---|---|---|---|---|---|
| `t2_standard` | 0.9995 | 0.8923 | 0.6714 | -0.1072 | -0.2209 | 0.1479 | 0.0317 | -0.1162 |
| `t2_torus` | 0.9908 | 0.8535 | 0.6887 | -0.1373 | -0.1648 | 0.2020 | 0.0393 | -0.1628 |

## Lattice Representatives

| Run | full rank | k=2 rank | k=2 hit | k=2 trust | k=2 overlap | k=2 eff_dim | k=2 H1 total | k=1 rank | k=1 Spearman |
|---|---|---|---|---|---|---|---|---|---|
| `lattice_standard_norm` | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| `lattice_standard_norm_inv` | 0.1685 | 0.1685 | 0.3373 | 0.8479 | 0.0585 | 1.0032 | 0.4405 | 0.1663 | 0.9755 |
| `lattice_vae_norm_beta001` | 0.6552 | 0.6553 | 0.3273 | 0.8505 | 0.0602 | 1.9948 | 1.3878 | 0.6500 | 0.1281 |
| `lattice_vae_norm_inv_b010_l100` | 0.6546 | 0.5756 | 0.3180 | 0.8465 | 0.0613 | 1.8563 | 4.6643 | 0.5350 | 0.0454 |
| `lattice_vae_norm_inv_b030_l100` | 0.6155 | 0.5585 | 0.3253 | 0.8459 | 0.0604 | 1.9997 | 4.3495 | 0.3671 | 0.1418 |
| `lattice_vae_wide_norm_inv_b003_l030` | 0.5419 | 0.5351 | 0.3160 | 0.8756 | 0.0584 | 1.9710 | 6.1422 | 0.3583 | 0.8430 |

## Branch Outcome

- Branch: `A`
- Summary: The best VAE+invariance runs remain comparatively stable down to k=2, then collapse at k=1.
- Next step: Move to equivariant or factorized latent models that explicitly preserve a 2D quotient chart.

### Evidence

- `t2_standard`: eff2=1.9379, trust2=0.8923, trust drop=0.2209, overlap drop=0.1162, H1(1)/H1(2)=0.0000
- `t2_torus`: eff2=1.9963, trust2=0.8535, trust drop=0.1648, overlap drop=0.1628, H1(1)/H1(2)=0.0000
- Stable-to-k=2 runs: `lattice_vae_norm_inv_b010_l100`, `lattice_vae_norm_inv_b030_l100`; 2->1 collapse is visible in: `lattice_vae_norm_inv_b010_l100`, `lattice_vae_norm_inv_b030_l100`.
