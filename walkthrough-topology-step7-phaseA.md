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
| `lattice_standard_norm_inv` | 0.1685 | 0.1685 | 0.3373 | 0.8481 | 0.0584 | 1.0032 | 0.4405 | 0.1663 | 0.9755 |
| `lattice_vae_norm_beta001` | 0.6551 | 0.6552 | 0.3333 | 0.8504 | 0.0603 | 1.9948 | 1.3887 | 0.6499 | 0.1281 |
| `lattice_vae_norm_inv_b010_l100` | 0.6544 | 0.5754 | 0.3347 | 0.8462 | 0.0613 | 1.8563 | 4.6679 | 0.5343 | 0.0454 |
| `lattice_vae_norm_inv_b030_l100` | 0.6153 | 0.5583 | 0.3340 | 0.8460 | 0.0605 | 1.9997 | 4.3520 | 0.3671 | 0.1420 |
| `lattice_vae_wide_norm_inv_b003_l030` | 0.5390 | 0.5319 | 0.3393 | 0.8749 | 0.0577 | 1.9710 | 6.1416 | 0.3546 | 0.8420 |

## Branch Outcome

- Branch: `A`
- Summary: The best VAE+invariance runs remain comparatively stable down to k=2, then collapse at k=1.
- Next step: Move to equivariant or factorized latent models that explicitly preserve a 2D quotient chart.

### Evidence

- `t2_standard`: eff2=1.9379, trust2=0.8923, trust drop=0.2209, overlap drop=0.1162, H1(1)/H1(2)=0.0000
- `t2_torus`: eff2=1.9963, trust2=0.8535, trust drop=0.1648, overlap drop=0.1628, H1(1)/H1(2)=0.0000
- Stable-to-k=2 runs: `lattice_vae_norm_inv_b010_l100`, `lattice_vae_norm_inv_b030_l100`; 2->1 collapse is visible in: `lattice_vae_norm_inv_b010_l100`, `lattice_vae_norm_inv_b030_l100`.
