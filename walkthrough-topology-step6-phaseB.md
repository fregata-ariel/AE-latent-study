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
| `t2_standard` | 0.8937 | 0.6704 | 11.7875 | 0.0000 | 0.9844 |
| `t2_torus` | 0.8624 | 0.6905 | 12.0160 | 0.0000 | 0.2438 |

## PH Trajectory Comparison

| Run | k=2 rank | k=2 hit | k=2 trust | k=2 overlap | k=2 eff_dim | k=2 H1 total | k=2 H1 longest | 3→2 H1 bottleneck | 2→1 H1 bottleneck | k=1 Spearman |
|---|---|---|---|---|---|---|---|---|---|---|
| `lattice_standard_norm` | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |
| `lattice_standard_norm_inv` | 0.1699 | 0.3365 | 0.8515 | 0.0498 | 1.0033 | 0.6973 | 0.0915 | 0.0000 | 0.0457 | 0.9780 |
| `lattice_vae_norm_beta001` | 0.6586 | 0.3360 | 0.8522 | 0.0518 | 1.9875 | 1.8118 | 0.0943 | 0.0764 | 0.0471 | 0.1021 |
| `lattice_vae_norm_inv_b010_l100` | 0.5746 | 0.3355 | 0.8483 | 0.0532 | 1.8289 | 4.9718 | 2.3889 | 1.1415 | 1.1944 | 0.0479 |
| `lattice_vae_norm_inv_b030_l100` | 0.5532 | 0.3360 | 0.8502 | 0.0556 | 1.9993 | 6.3612 | 0.7556 | 0.4001 | 0.3778 | 0.0479 |
| `lattice_vae_wide_norm_inv_b003_l030` | 0.5282 | 0.3385 | 0.8741 | 0.0496 | 1.9673 | 7.8123 | 0.9174 | 0.4702 | 0.4587 | 0.8411 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030` | 0.2528 | 0.3345 | 0.8477 | 0.0489 | 1.4007 | 1.2022 | 0.1102 | 0.0000 | 0.0551 | 0.8493 |
| `lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030` | 0.6621 | 0.3350 | 0.8542 | 0.0671 | 1.9796 | 5.7805 | 1.0479 | 0.0000 | 0.5240 | 0.4444 |

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
- Fundamental-domain runs supporting a 2D quotient chart transition: `lattice_vae_norm_inv_b010_l100`.
- `lattice_vae_wide_norm_inv_b003_l030` at k=2: trust=0.8741, overlap=0.0496, rank=0.5282, H1=7.8123.
