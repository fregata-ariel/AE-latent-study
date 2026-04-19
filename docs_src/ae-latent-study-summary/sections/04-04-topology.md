Topology Phase A/B では、Persistent Homology を主役ではなく **projection stability の診断器**として用いた。

canonical reading は `walkthrough-topology-phaseA.md` と `walkthrough-topology-phaseB.md` にある。

- control (`t2_standard`, `t2_torus`) は `k=2` までは比較的保たれ、`k=1` で明確に崩れる
- `lattice_standard_norm_inv` は orbit gluing は強いが、`k=2` の時点で既に薄い 1D ribbon 側にある
- `lattice_vae_norm_inv_b010_l100` と `lattice_vae_norm_inv_b030_l100` は、`k=2` までは比較的安定で、`k=1` で `j` と H1 が崩れる

Phase B の primary branch は `A1` だった。

- `k=2` までの 2D quotient chart は支持される
- 次のモデルは、その chart を explicit に latent に持たせるべき

この判断が Step 4 の factorized latent につながった。
