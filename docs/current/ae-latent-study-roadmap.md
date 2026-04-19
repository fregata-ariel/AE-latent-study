# AE Latent Study Roadmap

Lattice 系の current recommendation と、そこから 1-2 フェーズ先までの分岐を管理する living document。

# Current Node
::: {lang=ja}
この roadmap は、Lattice 系の canonical summary と最新実験結果に基づいて更新する current document である。
Step 5 を踏まえた現在の recommendation は `A2` である。
:::

::: {lang=en}
This roadmap is the current decision document for the lattice-centered study.
After Step 5, the active recommendation shifts to `A2`.
:::

- Confirmed state: `Branch A`
- Current diagnostic basis: `Topology Phase B` + `Lattice Step 5`
- Current recommendation: `A2`
- Why now: Step 4 validated factorized scaffolding, but Step 5 showed that the current quotient chart regularizer collapses spread instead of stabilizing a natural 2D chart.

# Active Branches
Active branches are listed in the order they should be considered from here.

## A2. Chart-Preserving Regularizer Redesign
- Trigger
  - Step 5 family failed to improve Step 4 in the way that matters most: quotient spread and partner-preserving chart quality did not improve together.
- Current reading
  - `overlap` increased, but `partner_rank` degraded from the Step 4 anchor `0.2460` to the Step 5 range `0.6568 - 0.6621`
  - `quotient_var_dim0/1` collapsed to nearly zero across the Step 5 family
  - therefore the current local-distance matching regularizer is not the right shape
- Immediate implementation idea
  - keep the factorized scaffold
  - redesign the quotient-side regularizer so that local chart preservation and global spread are both enforced

## A1. Previous Validated Branch
- Status
  - previous validated branch, not the current recommendation
- Why it mattered
  - Topology Phase B supported stable quotient geometry through `k=2`
  - Step 4 successfully encoded an explicit `quotient + gauge` split
- Why it is not current
  - Step 5 showed that merely adding the current quotient-geometry loss on top of the factorized scaffold is not enough
  - full equivariant expansion should wait until the quotient regularization itself is better posed

## A3. Sampling Redesign
- Trigger
  - revisit if wide-sampling runs start to dominate the fundamental-domain branch again after A2 is reworked
- Role for now
  - coverage probe, not the main successor path
- Immediate implementation idea
  - rebalance lattice sampling and density control, then rerun representative PH comparison

# Parked Branches
- `non-Euclidean latent`
  - parked until the Euclidean quotient-chart path is exhausted
- `latent function space / spectral basis`
  - parked until the quotient chart itself is stable enough to support a basis-level discussion
- `T^2 × R_+ control`
  - parked unless torus control is no longer sufficient for diagnostics

# Update Trigger
- Update this roadmap whenever one of the following changes substantially:
  - `docs/current/ae-latent-study-summary.md`
  - `runs/lattice_step5_summaries.json`
  - `runs/topology_diagnostics_step4/phaseB_comparison_summary.json`
- Keep `A2` as the recommendation if:
  - new quotient regularizer candidates still need to be evaluated, or
  - Step 5 remains the latest evidence on the factorized scaffold
- Re-promote `A1` only if:
  - a revised quotient regularizer preserves partner rank / `j` while pushing quotient geometry closer to a stable 2D chart
- Promote `A3` if:
  - wide-sampling evidence becomes clearly stronger than the fundamental-domain branch after the A2 redesign cycle
