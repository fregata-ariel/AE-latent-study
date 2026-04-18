# Topology Diagnostics Phase A

`python run_latent_topology_diagnostics.py` を実行すると、このファイルに PH / projection-ladder 診断の結果が出力されます。

- 目的: latent の ribbon-like geometry がどの次元で壊れるかを、PH・近傍保存・局所有効次元で診断する
- control: `t2_standard`, `t2_torus`
- lattice representatives:
  - `lattice_standard_norm`
  - `lattice_standard_norm_inv`
  - `lattice_vae_norm_beta001`
  - `lattice_vae_norm_inv_b010_l100`
  - `lattice_vae_norm_inv_b030_l100`
  - `lattice_vae_wide_norm_inv_b003_l030`
- branch outcomes:
  - `A`: 2D quotient geometry を支持
  - `B`: 3->2 で chart 学習が崩れる
  - `C`: control / PH pipeline の校正不足
  - `D`: projection artifact 優勢
  - `E`: sampling redesign 優勢
