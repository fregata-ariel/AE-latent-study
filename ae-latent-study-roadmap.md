# AE Latent Study Roadmap

## Current Node

- Confirmed state: `Branch A`
- Current diagnostic basis: `Topology Phase A`
- Current recommendation: `A1`
- Why now: the calibrated controls support a 2D quotient interpretation, and the best fundamental-domain VAE+invariance runs still look stable to `k=2` before collapsing at `k=1`.

## Active Branches

### A1. Equivariant / Factorized Latent

- Trigger: the best fundamental-domain VAE+invariance runs remain stable through `k=2` and collapse mainly at `k=1`.
- Immediate implementation idea: encode a 2D quotient chart explicitly, then separate invariant and group-action parts in latent space.

### A2. Chart-Preserving Regularizer

- Trigger: PH still shows fragility before `k=2` in the fundamental-domain runs.
- Immediate implementation idea: add a regularizer that preserves local chart geometry before introducing a richer latent action.

### A3. Sampling Redesign

- Trigger: the wide-sampling representative run clearly outperforms the fundamental-domain runs at `k=2`.
- Immediate implementation idea: rebalance lattice sampling and density control, then rerun the representative PH comparison.

## Parked Branches

- `non-Euclidean latent`: parked until the Euclidean quotient-chart path is exhausted.
- `latent function space / spectral basis`: parked until the quotient chart itself is more stable.
- `T^2 × R_+ control`: parked unless the current torus controls stop being sufficient.

## Update Trigger

- Update this roadmap whenever `walkthrough-topology-phaseB.md` changes its primary branch recommendation.
- Promote `A1` if the fundamental-domain PH trajectories stay stable through `k=2` and the wide run does not clearly dominate.
- Promote `A2` if the PH comparison still looks fragile before `k=2`.
- Promote `A3` if the wide run becomes the clearest `k=2` winner.
