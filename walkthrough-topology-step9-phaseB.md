# Topology Diagnostics Phase B

## Summary

- Phase B focuses on PH trajectories themselves, with `4→3→2→1` used as the main comparison axis.
- Source Phase A branch: `A`
- Phase B primary branch: `A1`
- Interpretation: The best fundamental-domain VAE+invariance runs still support a stable 2D quotient chart through k=2, so the next model step should encode that chart explicitly.
- Recommended next step: Prototype an equivariant or factorized latent model that preserves a 2D quotient chart.

## Control Anchors

| Run | k=2 trust | k=1 trust | k=2 H1 total | k=1 H1 total | 2→1 H1 bottleneck |
|---|---|---|---|---|---|
| `t2_standard` | 0.8923 | 0.6714 | 10.2153 | 0.0000 | 0.9885 |
| `t2_torus` | 0.8535 | 0.6887 | 10.1775 | 0.0000 | 0.2172 |

## PH Trajectory Comparison

| Run | k=2 rank | k=2 hit | k=2 trust | k=2 overlap | k=2 eff_dim | k=2 H1 total | k=2 H1 longest | 3→2 H1 bottleneck | 2→1 H1 bottleneck | k=1 Spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| `lattice_standard_norm` | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| `lattice_standard_norm_inv` | 0.1685 | 0.3373 | 0.8479 | 0.0586 | 1.0032 | 0.4405 | 0.0897 | 0.0000 | 0.0449 | 0.9755 |
| `lattice_vae_norm_beta001` | 0.6552 | 0.3347 | 0.8508 | 0.0595 | 1.9948 | 1.3872 | 0.0951 | 0.1149 | 0.0475 | 0.1281 |
| `lattice_vae_norm_inv_b010_l100` | 0.5753 | 0.3360 | 0.8440 | 0.0598 | 1.8562 | 4.6679 | 2.4917 | 1.2237 | 1.2458 | 0.0448 |
| `lattice_vae_norm_inv_b030_l100` | 0.5583 | 0.3353 | 0.8458 | 0.0611 | 1.9998 | 4.3663 | 0.7035 | 0.3172 | 0.3517 | 0.1437 |
| `lattice_vae_wide_norm_inv_b003_l030` | 0.5313 | 0.3453 | 0.8742 | 0.0570 | 1.9710 | 6.1452 | 0.8928 | 0.6025 | 0.4464 | 0.8497 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030` | 0.2450 | 0.3353 | 0.8446 | 0.0553 | 1.3852 | 0.9644 | 0.1057 | 0.0000 | 0.0528 | 0.8453 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030` | 0.6588 | 0.3353 | 0.8509 | 0.0717 | 1.9530 | 4.7442 | 1.0757 | 0.0000 | 0.5378 | 0.4583 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010` | 0.0959 | 0.3467 | 0.8495 | 0.0642 | 1.6319 | 1.1480 | 0.1890 | 0.0000 | 0.0945 | 0.9595 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100` | 0.1090 | 0.3393 | 0.8456 | 0.0545 | 1.9795 | 0.5994 | 0.0949 | 0.0000 | 0.0475 | 0.3996 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_td030_r030_ld010` | 0.1006 | 0.3367 | 0.8450 | 0.0533 | 1.9862 | 0.6269 | 0.0495 | 0.0000 | 0.0248 | 0.3828 |

## Key Reading

- `lattice_standard_norm_inv` remains the reference 1D ribbon baseline: orbit gluing is strong, but the PH trajectory is already thin at `k=2`.
- `lattice_vae_norm_beta001` remains the VAE-only control: it opens a 2D chart more than the invariant AE baseline, but does not carry the strongest quotient signal.
- `lattice_vae_norm_inv_b010_l100` and `lattice_vae_norm_inv_b030_l100` are the main Phase B runs. They are the ones to read first when deciding whether the quotient chart is genuinely 2D down to `k=2`.
- `lattice_vae_wide_norm_inv_b003_l030` is the coverage control. Read it as a sampling probe, not as the default successor model.

## Decision

- Primary branch: `A1`
- Summary: The best fundamental-domain VAE+invariance runs still support a stable 2D quotient chart through k=2, so the next model step should encode that chart explicitly.
- Next step: Prototype an equivariant or factorized latent model that preserves a 2D quotient chart.

### Active Branches

- `A1`: Move to an equivariant or factorized latent model if the best fundamental-domain VAE+invariance runs stay stable through `k=2` and collapse only at `k=1`.
- `A2`: Add a chart-preserving regularizer first if the fundamental-domain PH trajectory is still fragile before `k=2`.
- `A3`: Prefer sampling redesign if the wide run clearly outperforms the fundamental-domain runs at `k=2`.

### Evidence

- Phase A branch: `A`.
- Fundamental-domain runs supporting a 2D quotient chart transition: `lattice_vae_norm_inv_b010_l100`, `lattice_vae_norm_inv_b030_l100`.
- `lattice_vae_wide_norm_inv_b003_l030` at k=2: trust=0.8742, overlap=0.0570, rank=0.5313, H1=6.1452.
