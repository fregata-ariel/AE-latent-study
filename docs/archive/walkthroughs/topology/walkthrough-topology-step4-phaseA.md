# Topology Diagnostics Phase A

## Summary

- Persistent Homology is used here as a diagnostic for projection stability, not as a proof of the true quotient topology.
- Branch classification: **A**
- Interpretation: The best VAE+invariance runs remain comparatively stable down to k=2, then collapse at k=1.
- Recommended next step: Move to equivariant or factorized latent models that explicitly preserve a 2D quotient chart.

## Control Calibration

| Run | k=4 trust | k=2 trust | k=1 trust | Δtrust 4→2 | Δtrust 2→1 | k=2 overlap | k=1 overlap | Δoverlap 2→1 |
|---|---|---|---|---|---|---|---|---|
| `t2_standard` | 0.9996 | 0.8937 | 0.6704 | -0.1059 | -0.2233 | 0.1465 | 0.0256 | -0.1209 |
| `t2_torus` | 0.9923 | 0.8624 | 0.6906 | -0.1299 | -0.1719 | 0.1944 | 0.0330 | -0.1614 |

## Lattice Representatives

| Run | full rank | k=2 rank | k=2 hit | k=2 trust | k=2 overlap | k=2 eff_dim | k=2 H1 total | k=1 rank | k=1 Spearman |
|---|---|---|---|---|---|---|---|---|---|
| `lattice_standard_norm` | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| `lattice_standard_norm_inv` | 0.1699 | 0.1699 | 0.3365 | 0.8518 | 0.0500 | 1.0033 | 0.6973 | 0.1686 | 0.9780 |
| `lattice_vae_norm_beta001` | 0.6587 | 0.6587 | 0.3335 | 0.8529 | 0.0526 | 1.9875 | 1.8081 | 0.6538 | 0.1021 |
| `lattice_vae_norm_inv_b010_l100` | 0.6569 | 0.5748 | 0.3270 | 0.8486 | 0.0521 | 1.8290 | 4.9563 | 0.5396 | 0.0494 |
| `lattice_vae_norm_inv_b030_l100` | 0.6136 | 0.5534 | 0.3295 | 0.8494 | 0.0567 | 1.9993 | 6.3422 | 0.3600 | 0.0467 |
| `lattice_vae_wide_norm_inv_b003_l030` | 0.5359 | 0.5301 | 0.3255 | 0.8736 | 0.0493 | 1.9673 | 7.8158 | 0.3589 | 0.8330 |

## Branch Outcome

- Branch: `A`
- Summary: The best VAE+invariance runs remain comparatively stable down to k=2, then collapse at k=1.
- Next step: Move to equivariant or factorized latent models that explicitly preserve a 2D quotient chart.

### Evidence

- `t2_standard`: eff2=1.9468, trust2=0.8937, trust drop=0.2233, overlap drop=0.1209, H1(1)/H1(2)=0.0000
- `t2_torus`: eff2=1.9977, trust2=0.8624, trust drop=0.1719, overlap drop=0.1614, H1(1)/H1(2)=0.0000
- Stable-to-k=2 runs: `lattice_vae_norm_inv_b010_l100`, `lattice_vae_norm_inv_b030_l100`; 2->1 collapse is visible in: `lattice_vae_norm_inv_b010_l100`, `lattice_vae_norm_inv_b030_l100`.
