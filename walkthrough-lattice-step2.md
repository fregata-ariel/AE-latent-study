# Lattice Step 2 — Normalization, Invariance, and Capacity Sweep

## Summary

Step 2 の結論は明確である。**信号正規化だけでも Step 1 の 1D 退化を大きく緩和するが、決定打は modular invariant loss だった。**  
採用 run は **`lattice_standard_norm_inv`** で、再構成精度を維持したまま SL₂(Z) 同値点の latent 距離をほぼ 0 まで圧縮した。

- Step 1 baseline の `mean modular distance` は `1.019` だった
- Phase 1 の正規化だけで `0.093 ~ 0.154` まで改善した
- Phase 2 の `lattice_standard_norm_inv` で **`0.000963`** まで改善した
- Phase 3 の latent 拡張や VAE は興味深い形状を作るが、今回の目的では採用理由にならなかった

評価順位は計画どおり、
1. `modular_invariance.mean_latent_distance`
2. `max_abs_logabsj_spearman`
3. reconstruction MSE

とした。

---

## Step 1 Baseline

| Experiment | MSE | max abs corr vs Re/Im(j) | SL₂(Z) mean dist |
|---|---:|---:|---:|
| `lattice_standard` | 4.25569e-08 | 0.0081 | 1.0194 |
| `lattice_halfplane` | 2.00637e-07 | 0.0076 | 2.0525 |
| `lattice_standard_wide` | 5.33878e-06 | 0.0170 | 0.9578 |

Step 1 では再構成は十分だったが、latent はほぼ 1 次元曲線に潰れていた。

---

## Phase 1 — Normalization

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | PCA EVR |
|---|---:|---:|---:|---:|---:|
| `lattice_standard_norm` | 2.89401e-07 | 0.9834 | 3.9925 | 0.0944 | - |
| `lattice_halfplane_norm` | 6.10116e-07 | 0.9830 | 4.0311 | 0.0930 | - |
| `lattice_standard_wide_norm` | 1.22880e-06 | 0.8869 | 2.7423 | 0.1537 | - |

- ルール上の勝者は **`lattice_halfplane_norm`**。ただし `lattice_standard_norm` との差は極小で、実質的には同等とみてよい
- Step 1 の `1.019` から `0.094` 付近まで一気に改善しており、**正規化だけで改善方針 A は妥当と確認できた**
- wide sampling は Step 1 では相対的に良かったが、正規化後は fundamental-domain sampling より悪化した

**解釈:**  
Step 1 では振幅スケールが `Im(τ)` をほぼ単独で支配していたが、max 正規化によりその優位が大きく弱まった。その結果、latent は依然として細い曲線状ではあるものの、Step 1 よりは明らかに「折りたたみ可能な」形に近づいた。

---

## Phase 2 — Modular Invariant Loss

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | PCA EVR |
|---|---:|---:|---:|---:|---:|
| `lattice_standard_norm_inv` | 4.18320e-07 | 0.9825 | 4.1880 | 0.0010 | - |
| `lattice_standard_wide_norm_inv` | 1.30117e-06 | 0.7548 | 2.6745 | 0.0042 | - |

- 勝者は **`lattice_standard_norm_inv`**
- `mean modular distance = 0.000963` は Step 1 baseline 比で約 **1000 倍改善**
- `max distance = 0.0121` も十分小さく、SL₂(Z) 同値点が実質的に同じ latent に貼り合わされている
- `log10|j|` Spearman も `0.9825` を保っており、「modularity を入れたせいで意味構造が壊れた」わけではない

**解釈:**  
今回の sweep では、改善案 D が最も効いた。正規化で作った「かなり良い初期形状」に対して、group action を loss で明示化すると latent の貼り合わせがほぼ完成する。Step 2 の主結果はここにある。

---

## Phase 3 — Capacity / VAE

| Experiment | MSE | max Spearman vs log10_abs_j | max MI vs log10_abs_j | SL₂(Z) mean dist | PCA EVR |
|---|---:|---:|---:|---:|---:|
| `lattice_standard_norm_latent4` | 4.64934e-07 | 0.9871 | 4.0687 | 0.2500 | 0.999, 0.001 |
| `lattice_standard_norm_latent8` | 2.88902e-08 | 0.9905 | 4.3985 | 0.1925 | 0.992, 0.008 |
| `lattice_vae_norm_beta001` | 8.09567e-05 | 0.6280 | 4.4373 | 0.0074 | 0.668, 0.319 |

- ルール上の勝者は **`lattice_vae_norm_beta001`**。理由は modular distance が `0.0074` と最小だから
- ただし採用候補としては弱い。再構成誤差が `8.10e-05` まで悪化し、Spearman も `0.628` に落ちる
- `latent4` / `latent8` は Spearman は非常に高いが modular distance が悪く、今回の主目的には不適
- PCA より、`latent4` と `latent8` は依然としてほぼ低次元の有効自由度しか使っていない

**解釈:**  
改善案 C は「表現を広げれば解ける」という単純な話ではないことを示した。特に `latent8` は再構成 MSE こそ最良だが、modularity の観点ではむしろ悪い。  
一方 VAE は 2D 的な分散を最も明確に持っており、**幾何の可視化**としては面白いが、今回の ranking では採用できない。

---

## Adopted Run

**採用:** `lattice_standard_norm_inv`

- Reconstruction MSE: `4.18320e-07`
- max Spearman vs `log10|j|`: `0.9825`
- mean modular distance: `0.000963`
- max modular distance: `0.012141`

**採用理由:**  
Step 2 の目的は「`Im(τ)` 優勢の 1D 退化を崩し、SL₂(Z) 同値点を整合的にまとめる」ことだった。この観点では `lattice_standard_norm_inv` が最もバランスがよい。  
`lattice_halfplane_norm` は Phase 1 では僅差で良かったが、決定打ではない。`latent8` は再構成専用モデルとしては良いが modularity が弱い。VAE は可視化上は魅力的だが再構成と Spearman が劣る。

---

## Visual Sanity Checks

代表画像を確認した。数値と見た目は概ね整合している。

- [standard_norm latent scatter](runs/lattice_standard_norm/results/lattice_latent_scatter.png)
  - Step 1 より整理されているが、依然として細い 1D 的リボンが支配的
  - `Im(τ)` の勾配が主幹方向にきれいに乗っており、正規化が効いていることは見える

- [standard_norm_inv latent scatter](runs/lattice_standard_norm_inv/results/lattice_latent_scatter.png)
  - `standard_norm` よりさらに細く整列し、枝の貼り合わせが進んでいる
  - modular distance が `0.000963` まで落ちた結果と視覚的に矛盾しない

- [standard_norm_inv j-correlation](runs/lattice_standard_norm_inv/results/j_invariant_correlation.png)
  - `log10|j|` に対して滑らかな単調関係が見える
  - `Re(j), Im(j)` Pearson が小さい一方で、`log10|j|` Spearman が高いという数値結果をよく説明している

- [standard_norm_latent8 latent scatter](runs/lattice_standard_norm_latent8/results/lattice_latent_scatter.png)
  - PCA 平面では 2D 的に見えるが、内部に疎な散布があり、同値点の貼り合わせは弱い
  - `PCA = (0.992, 0.008)` と `mean modular distance = 0.1925` の組は、この図と整合する

- [vae latent scatter](runs/lattice_vae_norm_beta001/results/lattice_latent_scatter.png)
  - PCA 平面で最も「葉」のような厚みがあり、2D 的広がりは確かに見える
  - ただし再構成誤差の悪化と `log10|j|` Spearman 低下も納得できる見た目で、採用 run にはしない判断を支持する

- [wide_norm_inv latent scatter](runs/lattice_standard_wide_norm_inv/results/lattice_latent_scatter.png)
  - wide sampling 由来の分岐構造は残るが、standard-wide-norm より同値点の折りたたみが明らかに進んでいる
  - それでも fundamental-domain 学習付き invariant loss には及ばない

また [standard_norm_inv training curves](runs/lattice_standard_norm_inv/results/training_curves.png) と [reconstructions](runs/lattice_standard_norm_inv/results/reconstructions.png) も確認した。  
学習は早期に十分低損失へ入り、再構成波形も視覚上ほぼ完全一致している。

---

## Open Issues

- `walkthrough-lattice-step2.md` 自動生成側は、Step 1 baseline JSON が手元に無い場合に表が空になることがある。これは fallback を入れて改善済み
- 長い sweep では matplotlib の figure を閉じずに warning が出やすかったため、plot 保存後に close するよう修正した
- differentiable projection to the fundamental domain は依然として未着手
- VAE の 2D 幾何は魅力的なので、将来は「再構成をあまり落とさずに modularity を保つ β 調整」を別テーマとして試す価値がある

---

## Conclusion

Step 2 は成功である。  
**Normalization は必要条件、modular invariant loss は十分条件に近い。**  
次の実験は `lattice_standard_norm_inv` を基準系として進めるのが自然であり、以後の改良はこの run を上回れるかどうかで判断すればよい。
