# Lattice Step 3 — VAE + Invariance Sweep

`python run_lattice_step3_experiments.py` を実行すると、このファイルに Step 3 の比較結果が出力されます。

- 目的: VAE の 2D chart 性と modular invariance による orbit gluing を両立できるか検証する
- 実験行列: `fundamental_domain` / `wide halfplane` × `vae_beta` `{0.003, 0.01, 0.03}` × `modular_invariance_weight` `{0.03, 0.1}`
- anchor runs: `lattice_standard_norm_inv`, `lattice_standard_wide_norm_inv`, `lattice_vae_norm_beta001`
- selection key: `modular distance -> |effective_dimension - 2| -> overlap -> trustworthiness -> log10_abs_j Spearman -> MSE`
- success gate:
  - `mean modular distance <= 0.01`
  - `effective_dimension >= 1.4`
  - `knn_jaccard_mean >= 0.05`
  - `trustworthiness >= 0.85`
  - `max_abs_logabsj_spearman >= 0.80`
  - `mse <= 5e-06`
