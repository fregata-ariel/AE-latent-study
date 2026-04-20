# AE Latent Study Roadmap

## Current Node

- Confirmed state: `Branch A`
- Current diagnostic basis: `Step 8 lattice + topology follow-up`
- Current recommendation: `A2 continues; next = teacher quotient-structure distillation`
- Why now: Step 8 partially rescued raw quotient `j` retention, but the selected run still missed balanced acceptance and did not preserve the `j` signal robustly through the PCA / PH ladder.

## Active Branches

### A1. Equivariant / Factorized Latent

- Trigger: the best fundamental-domain VAE+invariance runs remain stable through `k=2` and collapse mainly at `k=1`.
- Status: validated as the reason for introducing the factorized scaffold, but not the current next move.

### A2. Chart-Preserving Regularizer

- Trigger: PH still shows fragility before `k=2` in the fundamental-domain runs.
- Immediate implementation idea: use the Step 7 winner as a fixed teacher and distill quotient local distance structure while keeping `j` rank retention as a semantic guard.

### A3. Sampling Redesign

- Trigger: the wide-sampling representative run clearly outperforms the fundamental-domain runs at `k=2`.
- Immediate implementation idea: rebalance lattice sampling and density control, then rerun the representative PH comparison.

## Parked Branches

- `non-Euclidean latent`: parked until the Euclidean quotient-chart path is exhausted.
- `latent function space / spectral basis`: parked until the quotient chart itself is more stable.
- `T^2 × R_+ control`: parked unless the current torus controls stop being sufficient.

## Update Trigger

- Update this roadmap whenever a Step 9 follow-up report is generated.
- Return to `A1` only if teacher distillation improves topology-side `j` retention while preserving Step 7 partner/chart quality.
- Keep `A2` and switch to contrastive local geometry if Step 9 misses the balanced gate.
- Promote `A3` if the wide run becomes the clearest `k=2` winner.
