# AE Latent Study Summary

Lattice 中心の AE latent 研究について、Step 1-5 と topology diagnostics を通じて得られた理解を整理した canonical summary。

# 概要
::: {lang=ja}
本研究は、autoencoder 様ネットワークの latent を単なる `R^d` としてではなく、対称性で割った **商空間の座標系**として理解できるかを、格子テータ関数データで検証してきた。
この文書は、Lattice Step 1-5 と topology diagnostics を通じて何が分かり、何が次の課題として残ったのかをまとめた canonical summary である。
:::

::: {lang=en}
This document summarizes the lattice-centered AE latent study as a quotient-geometry learning problem.
It consolidates the evidence from Lattice Steps 1-5 and topology diagnostics into a single canonical reading.
:::

- 主対象は lattice 実験系である。
- torus 系は前史としてのみ扱う。
- 数値の source of truth は `runs/*.json` を優先し、必要に応じて最新 walkthrough を補う。
- Step 5 を踏まえた current recommendation は `A2` である。

# 研究の問い
この研究の中心的な問いは、`ae-latent-study-discuss1.md` で整理した次の視点を、実際のモジュライ的データでどこまで実証できるかにある。

- latent 空間そのものを保型形式と同一視するのではなく、まずは **群作用で割った商空間の座標**として読む
- 不変量や共変量は latent そのものではなく、その上の関数や構造として現れる
- 対称性が明示されていない標準 AE/VAE では、再構成が良くても商空間構造は自動的には得られない

このため、研究の評価軸は最初から再構成誤差だけでは足りなかった。Lattice 系では特に次の 3 点を追う必要があった。

- `SL₂(Z)` 同値点が latent 上で近くなるか
- latent が本当に 2D quotient chart として広がるか
- `j` や PH diagnostics を通して、その chart が 1D ribbon ではなく 2D 構造を保っているか

# 前史
前史としての torus 実験は、周期構造をもつデータで AE/VAE がどの程度自然な latent を学ぶかを確認するための基礎段階だった。

- torus 系では周期性や低次元 manifold 学習の基本的な挙動を確認できた
- ただし torus だけでは、群作用で割られたモジュライ的構造を直接検証することはできない
- そのため lattice 実験では、最初から `SL₂(Z)` と楕円曲線モジュライに対応する入力を選び、比喩ではなく本物の商空間問題として AE latent を調べる方針に進んだ

この位置づけにより、lattice 系は単なる応用例ではなく、`discuss1` の見取り図を本番の対象で試す主舞台になった。

# Lattice 実験の流れ
Lattice 実験は、再構成中心の baseline から始めて、orbit gluing、2D chart 化、PH 診断、factorization、そして正則化設計の失敗までを順に切り分ける流れで進んだ。

各段階では「何が改善したか」だけでなく、「何がまだできていないか」を次の設計に繋げることを重視した。

## Step 1: 1D 退化の確認
Step 1 の baseline は、再構成だけでは商空間構造が自動的には出ないことを明確に示した。

- `walkthrough-lattice-step1.md` では Standard / HalfPlane / wide sampling の 3 実験を比較した
- 全 run で reconstruction MSE はほぼ 0 だった
- それにもかかわらず latent は実質的に 1D 曲線へ退化し、`SL₂(Z)` 同値点の距離も大きかった

重要なのは、ここで AE が「失敗した」のではなく、**再構成だけを最適化した場合の自然な帰結**が観測されたことだ。

- 振幅情報は強く `Im(τ)` に支配される
- `j(τ)` は巨大な dynamic range を持ち、単純な線形相関ではほとんど捉えられない
- 標準 AE は最も再構成に効く 1D 的な方向へ圧縮する

この時点で、次の課題は「表現が悪い」のではなく、「目的関数と評価軸が商空間学習に向いていない」ことだと分かった。

## Step 2: Orbit Gluing の確立
Step 2 では、入力正規化と modular invariance を入れることで orbit gluing が決定的に改善した。

代表 run `lattice_standard_norm_inv` の要点:

- `mean modular distance = 0.0010`
- `max_abs_logabsj_spearman = 0.9825`
- `reconstruction.mse = 4.18e-07`
- ただし `effective_dimension = 1.0029`

ここで得られた理解はかなり重要だった。

- max 正規化で振幅スケール支配を抑えることは有効
- `log10|j|` ベースの評価は latent の単調構造をかなりよく捉える
- `E(gx) ≈ E(x)` 型の invariance loss は orbit gluing に非常に強い
- それでも latent は 2D chart にならず、ほぼ 1D ribbon のまま残る

つまり Step 2 は、**商空間の同値類をまとめる段階**としては成功したが、**自然な 2D quotient chart を作る段階**にはまだ届いていなかった。

## Step 3: VAE + Invariance で 2D chart の兆候を得る
Step 3 では、VAE の KL と modular invariance を組み合わせて、orbit gluing と 2D chart 性を同時に改善できるかを調べた。

fundamental-domain 側で特に重要だったのは次の 2 本である。

- `lattice_vae_norm_inv_b010_l100`
  - `effective_dimension = 2.0096`
  - `trust = 0.8534`
  - `overlap = 0.0576`
  - `max_abs_logabsj_spearman = 0.9590`
  - `mean modular distance = 0.0001`
- `lattice_vae_norm_inv_b030_l100`
  - `effective_dimension = 1.8469`
  - `trust = 0.8556`
  - `overlap = 0.0673`
  - `max_abs_logabsj_spearman = 0.8138`
  - `mean modular distance = 0.0001`

Step 3 で初めて、orbit gluing と 2D chart の間にある程度の両立可能性が見えた。

- Step 2 の invariant AE より明らかに 2D 寄り
- それでも reconstruction MSE は `~8e-05` までしか下がらない
- wide sampling は chart 側では面白いが、本線の successor model としては扱いにくい

この段階の結論は、「VAE + invariance で 2D quotient chart の兆候は得られるが、それを explicit に整理する latent 設計が必要」というものだった。

## Topology Phase A/B: k=2 安定と k=1 崩壊
Topology Phase A/B では、Persistent Homology を主役ではなく **projection stability の診断器**として用いた。

canonical reading は `walkthrough-topology-phaseA.md` と `walkthrough-topology-phaseB.md` にある。

- control (`t2_standard`, `t2_torus`) は `k=2` までは比較的保たれ、`k=1` で明確に崩れる
- `lattice_standard_norm_inv` は orbit gluing は強いが、`k=2` の時点で既に薄い 1D ribbon 側にある
- `lattice_vae_norm_inv_b010_l100` と `lattice_vae_norm_inv_b030_l100` は、`k=2` までは比較的安定で、`k=1` で `j` と H1 が崩れる

Phase B の primary branch は `A1` だった。

- `k=2` までの 2D quotient chart は支持される
- 次のモデルは、その chart を explicit に latent に持たせるべき

この判断が Step 4 の factorized latent につながった。

## Step 4: Factorized Latent の明示化
Step 4 は `quotient(2) + gauge(4)` の factorized latent を導入し、A1 の仮説を最小構成で試した。

selected run `lattice_factorized_vae_fd_b030_q100_g030_d030`:

- `quotient_partner_rank_percentile_mean = 0.2460`
- `effective_dimension = 1.3854`
- `trust = 0.8477`
- `overlap = 0.0486`
- `gauge_equivariance_mse = 1.36e-05`
- `decoder_equivariance_mse = 3.21e-04`
- `max_abs_logabsj_spearman = 0.9250`
- `reconstruction.mse = 1.73e-05`

Step 4 の意味はかなり明確だった。

- factorization によって quotient/gauge を分けて議論できるようになった
- quotient partner alignment と gauge consistency はかなり良くなった
- しかし quotient 自体はまだ細く、PH 的にも「整理されたがまだ薄い」状態に留まった

つまり Step 4 は scaffold としては成功だが、quotient chart を押し広げる別の仕組みが必要だった。

## Step 5: Quotient 正則化の失敗
Step 5 では、Step 4 の factorized scaffold を固定したまま、quotient の local geometry を押し広げる regularizer を足した。

しかし結果は、**有益な失敗**だった。

Step 5 family の共通的な観察:

- overlap は Step 4 より上がった
  - 例: `0.0486 -> 0.0723`
- trust もわずかに上がった
- その一方で quotient partner rank は大きく悪化した
  - Step 4 anchor: `0.2460`
  - Step 5 runs: `0.6568 - 0.6621`
- `max_abs_logabsj_spearman` も低下した
- reconstruction / decoder consistency も悪化した
- さらに `quotient_var_dim0`, `quotient_var_dim1` は全 run でほぼ `0` に落ちた

このことから、現行の chart-preserving loss は「2D chart を押し広げる」というより、
**小さい雲に潰したまま local order だけ合わせる**方向へ働いていると読める。

したがって Step 5 の結論は、

- factorized scaffold 自体は維持する価値がある
- しかし quotient spread を押す regularizer の形式は作り直しが必要
- roadmap は `A1` 維持ではなく `A2` へ正式移行する

というものになった。

# 現在の理解
現時点での理解は、次のようにまとめられる。

- **Step 2** は orbit gluing の確立点である
  - AE でも invariance loss を入れれば `X/G` 的な圧縮はかなり強く作れる
- **Step 3 + topology** は、fundamental-domain VAE+invariance が `k=2` までは比較的安定な 2D quotient chart を持つことの主証拠である
- **Step 4** は quotient/gauge を分けて議論する scaffold として成功した
- **Step 5** は、その scaffold 上で現行の chart regularizer 形式が不適切であることを示した

このため現在の研究判断は、次の 2 点に収束している。

- factorization という方向性は正しい
- ただし quotient spread を押す loss は、今の local-distance matching のままではだめ

したがって current recommendation は `A2` であり、「factorized scaffold を保ちながら chart-preserving regularizer の設計を作り直す」ことが次の本線になる。

# 今後の分岐
現在の分岐は、roadmap と対応させると次のように整理できる。

## A2. Chart-Preserving Regularizer 再設計

本線。Step 5 が示したように、現行 regularizer は scale-invariant すぎて quotient collapse を止められていない。

次に設計し直すべき論点:

- quotient spread を直接保つ loss をどう定義するか
- local geometry と global spread をどう両立させるか
- partner rank / `j` / gauge consistency を壊さない条件をどう組み込むか

## A1. Previous Validated Branch

履歴上の重要到達点として残す。

- Phase B までは `A1` が自然だった
- Step 4 で factorized latent を explicit に実装することには成功した
- ただし Step 5 を踏まえると、そのまま full equivariant latent に進む前に quotient regularization を詰める必要がある

## A3. Sampling Redesign

現時点では parked に近い active branch である。

- wide sampling は coverage probe としては有益
- ただし現時点では本線の successor model ではなく、A2 で fundamental-domain 側を詰めた後に再評価するのが妥当

# 参照マップ
詳細な履歴や数値は、次の文書と summary JSON を参照する。

## Canonical Current

- [現行総括](ae-latent-study-summary.md)
- [現行ロードマップ](ae-latent-study-roadmap.md)

## Lattice Walkthroughs

- [Step 1](../archive/walkthroughs/lattice/walkthrough-lattice-step1.md)
- [Step 2](../archive/walkthroughs/lattice/walkthrough-lattice-step2.md)
- [Step 3](../archive/walkthroughs/lattice/walkthrough-lattice-step3.md)
- [Step 4](../archive/walkthroughs/lattice/walkthrough-lattice-step4.md)
- [Step 5](../archive/walkthroughs/lattice/walkthrough-lattice-step5.md)

## Topology Diagnostics

- [Topology Phase A](../archive/walkthroughs/topology/walkthrough-topology-phaseA.md)
- [Topology Phase B](../archive/walkthroughs/topology/walkthrough-topology-phaseB.md)
- [Step 4 Topology Phase A](../archive/walkthroughs/topology/walkthrough-topology-step4-phaseA.md)
- [Step 4 Topology Phase B](../archive/walkthroughs/topology/walkthrough-topology-step4-phaseB.md)

## JSON Summaries

- [runs/lattice_step3_summaries.json](../../runs/lattice_step3_summaries.json)
- [runs/lattice_step4_summaries.json](../../runs/lattice_step4_summaries.json)
- [runs/lattice_step5_summaries.json](../../runs/lattice_step5_summaries.json)
- [runs/topology_diagnostics/topology_diagnostics_summary.json](../../runs/topology_diagnostics/topology_diagnostics_summary.json)
- [runs/topology_diagnostics/phaseB_comparison_summary.json](../../runs/topology_diagnostics/phaseB_comparison_summary.json)
- [runs/topology_diagnostics_step4/topology_diagnostics_summary.json](../../runs/topology_diagnostics_step4/topology_diagnostics_summary.json)
- [runs/topology_diagnostics_step4/phaseB_comparison_summary.json](../../runs/topology_diagnostics_step4/phaseB_comparison_summary.json)

## Discussion History

- [Discuss 1](../archive/discuss/ae-latent-study-discuss1.md)
- [Discuss 2](../archive/discuss/ae-latent-study-discuss2.md)
- [Discuss 3](../archive/discuss/ae-latent-study-discuss3.md)
