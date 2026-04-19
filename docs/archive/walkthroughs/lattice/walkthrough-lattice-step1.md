# Lattice Modular AE — Experiment Results & Analysis

## Summary

全3実験（L1, L2, L3）が正常に完了した。再構成誤差は極めて低い（MSE ≈ 0）が、**潜在空間は2次元の基本領域 $\mathcal{F}$ を再現できておらず、おおむね1次元曲線に退化している。** これはテータ関数の構造上予想可能な結果であり、次のステップへの重要な手がかりを提供する。

---

## Quantitative Results

| Experiment | MSE | MAE | j-corr | SL₂(Z) mean dist | SL₂(Z) max dist |
|---|---|---|---|---|---|
| **L1** Standard / $\mathcal{F}$ | 4.3e-8 | 1.2e-4 | 0.008 | 1.019 | 2.878 |
| **L2** HalfPlane / $\mathcal{F}$ | 2.0e-7 | 2.5e-4 | 0.008 | 2.053 | 5.805 |
| **L3** Standard / wide $\mathbb{H}$ | 5.3e-6 | 9.7e-4 | 0.017 | 0.958 | 2.865 |

> **Note:** 再構成誤差は全実験で極めて低い。AE は格子テータ関数を完璧に再構成する能力を持つが、問題は**潜在空間の構造**にある。

---

## Latent Space Analysis

### L1: Standard AE on Fundamental Domain

**潜在空間の散布図:**

Im(τ)で色づけすると滑らかな1次元曲線が見える。Re(τ)の情報は曲線に沿って混在している。

![L1 latent scatter](img/lattice_step1/L1_latent_scatter.png)

**j-不変量との相関:**

j(τ)の値域が10^8に達するため、線形相関がほぼゼロ。

![L1 j-invariant correlation](img/lattice_step1/L1_j_corr.png)

**観察:**
- 潜在空間は**1次元曲線に退化**している（右上図: Im(τ)で色づけすると完全に単調な勾配）
- Re(τ) の情報は曲線上で微小な広がりとしてのみ残存
- 原因: $\mathrm{Im}(\tau)$ がθ関数の全体的なスケール（指数減衰率）を支配しており、エンコーダはこの「振幅情報」を最優先で捕捉する

### L2: HalfPlane AE on Fundamental Domain

HalfPlane制約付きの潜在空間。L1とほぼ同じ1次元曲線構造。

![L2 latent scatter](img/lattice_step1/L2_latent_scatter.png)

**観察:**
- L1 とほぼ同じ1D曲線構造。$y > 0$ 制約は曲線の位置を変えるだけ
- SL₂(Z) 不変性はむしろ悪化（mean dist: 2.05 vs L1の1.02）
- **結論:** 上半平面制約だけでは、潜在空間の構造改善に不十分

### L3: Standard AE on Wide Half-Plane

広域H からのサンプル。2Dに展開した構造が見え、SL₂(Z)同値な折り畳みの兆候がある。

![L3 latent scatter](img/lattice_step1/L3_latent_scatter.png)

**観察:**
- L1/L2 より**2次元的な広がり**が見られ、最も興味深い結果
- Im(τ)で色付けすると、高Im(τ)の点は1箇所（右上）に集中し、低Im(τ)の点は広がっている
- **SL₂(Z) 同値性:** mean dist = 0.958 は L1 より若干良好。テータ関数が同じ格子に対応する $\tau$ は、AE により近い潜在表現にマッピングされている兆候がある
- Re(τ) の情報がある程度回復している（左上図の色分布）

**j-不変量との相関 (L3):**

![L3 j-invariant correlation](img/lattice_step1/L3_j_corr.png)

---

## Root Cause Analysis: なぜ 1D に退化するのか

### テータ関数のスケール問題

格子テータ関数 $\theta(t;\tau) = \sum \exp(-\pi|m+n\tau|^2 t)$ の主要項は最短ベクトル長 $\lambda_1(\tau)$ によって支配される:

$$\theta(t;\tau) \approx N_1 \cdot \exp(-\pi \lambda_1^2 \cdot t) + O(\exp(-\pi \lambda_2^2 \cdot t))$$

ここで $\lambda_1 = \min_{(m,n)\neq 0}|m+n\tau|$ は格子の最短ベクトル長。

- $\mathrm{Im}(\tau)$ が大きい → $\lambda_1 \approx 1$（$(1,0)$ ベクトル）→ 信号は $\approx 2e^{-\pi t}$
- $\mathrm{Im}(\tau)$ が小さい → $\lambda_1$ は $\mathrm{Re}(\tau)$ にも依存（特に $|\tau| \approx 1$ 近傍）

**結果:** テータ関数の信号は $\mathrm{Im}(\tau)$ に対して指数的に敏感だが、$\mathrm{Re}(\tau)$ に対しては（特に $\mathrm{Im}(\tau) > 1.5$ では）ほとんど変化しない。AE は最も分散の大きい方向（$\mathrm{Im}(\tau)$）を優先的にエンコードする。

### j-不変量のダイナミックレンジ問題

$j(\tau)$ は $\mathcal{F}$ のカスプ近傍（$\mathrm{Im}(\tau) \gg 1$）で $|j(\tau)| \sim e^{2\pi \cdot \mathrm{Im}(\tau)}$ と爆発的に増大する。$y_{\max} = 3.0$ では $|j| \sim 10^8$ に達し:
- 線形相関係数（Pearson）はカスプ近傍の数点に支配される
- 大多数の点は $j \approx 0$ 付近に密集し、相関が検出不能

---

## Key Findings

| # | Finding | Significance |
|---|---------|-------------|
| 1 | AE は完璧な再構成能力を持つ | 2次元潜在空間でテータ関数を完全にエンコード可能 |
| 2 | 潜在空間は Im(τ) 支配の 1D 曲線に退化 | テータ関数のスケール構造が支配的 |
| 3 | HalfPlane 制約は構造改善に不十分 | $y > 0$ は幾何的にあまりに弱い制約 |
| 4 | L3 (広域) が最も 2D 的構造を示す | $\mathrm{Re}(\tau)$ が広い範囲を持つと区別しやすい |
| 5 | j-不変量との線形相関は検出不能 | ダイナミックレンジ問題。非線形比較が必要 |

---

## Proposed Next Steps

> **Important:** 以下の改善方針のうち、どれを優先するか検討をお願いします。

### A. 信号正規化（最優先、即効性が高い）

テータ関数の「スケール」を除いて「形」だけを見せる：

```python
# 各信号を max で正規化
signals_normalized = signals / signals.max(axis=1, keepdims=True)
```

これにより $\mathrm{Im}(\tau)$ の情報が振幅ではなく**減衰曲線の形状**に移り、$\mathrm{Re}(\tau)$ の影響が相対的に増大する。

### B. j-不変量との比較改善

- **Spearman rank correlation:** スケール不変な相関
- **$\log|j|$ との比較:** ダイナミックレンジを抑制
- **Mutual Information:** 非線形依存関係の検出

### C. エンコーダアーキテクチャの強化

- **Latent dim = 4 or 8:** 2D では AE が1D 曲線に退化するプレッシャーが強い。高次元 latent + PCA で有効次元を分析
- **β-VAE:** KL 正則化でアイソトロピック（等方的）な潜在空間を促進
- **Contrastive loss:** 同じ格子（SL₂(Z)同値な $\tau$）のペアを近づける

### D. 基本領域制約の直接導入

- **Modular invariant loss:** $L_{\text{inv}} = \|z(\tau) - z(\gamma\tau)\|^2$ を明示的に追加
- **Projection layer:** 潜在空間→基本領域への微分可能な射影（ただし $T, S$ 操作の組み合わせは微分不可能なので近似が必要）

---

## Backing for Proposed Improvements

以下は、上の改善案が「自然そうだから」ではなく、**現行コードの制約**と **`runs/` に残っている結果**の両方から導けることを確認したメモである。

### A. 信号正規化が最優先である根拠

- データ生成コード `data/generation.py` では、格子信号は $\sum \exp(-\pi |m+n\tau|^2 t)$ をそのまま入力としており、サンプルごとの正規化は一切していない
- 学習側 `train/train_step.py` でも loss は純粋な再構成 MSE のみで、振幅優勢を打ち消す項が入っていない
- L1 と同じ設定でデータ生成コードを再実行すると、サンプルごとの最大値 `signals.max(axis=1)` と $\mathrm{Im}(\tau)$ の相関は **-0.748**、L2 ノルムとの相関は **-0.709** だった。一方で $\mathrm{Re}(\tau)$ との相関はそれぞれ **0.011**, **0.010** とほぼゼロ
- 同じデータで、曲線形状の粗い指標 `signal(t_max) / signal(t_min)` は $\mathrm{Im}(\tau)$ と **0.838** の相関を持つ。つまり現行入力では「振幅」と「減衰形状」の両方がほぼ同じ方向に $\mathrm{Im}(\tau)$ を強く運んでおり、AE がまずそこを使うのは自然
- 実際、L1/L2 は再構成誤差がほぼゼロなのに潜在構造だけが崩れているので、まずは学習を難しくするより、**入力側でスケール優位を弱める**のが最短手になる

### B. `j` 比較の見直しが必要な根拠

- 現行評価 `eval/metrics.py` の `compute_j_correlation` は、各 latent 次元と `Re(j), Im(j)` の **Pearson 相関だけ**を見ている
- この実装だと、latent 空間が回転していても不利であり、また `j` のダイナミックレンジにも極端に弱い
- L1 と同じ設定で `j(\tau)` を再計算すると、`|j|` は **min = 5.1e-2**, **median = 2.5e5**, **95 percentile = 8.0e7**, **max = 1.53e8** だった。Pearson がカスプ近傍の少数点に引っ張られるのはこの値域から見ても自然
- 同じ値を $\log_{10}|j|$ にすると範囲は **[-1.29, 8.18]** まで圧縮される。`log|j|` や Spearman を使う提案は、単なる一般論ではなく、この実験分布に対して直接有効

### C. 高次元 latent / β-VAE / contrastive の妥当性

- 現行 lattice 実験は 3 本とも `latent_dim = 2` 固定で、構造上 2 次元に全情報を押し込む設定になっている
- 一方で `eval/analysis.py` は latent 次元が 3 以上なら PCA を自動で保存できるようになっており、**latent_dim = 4 or 8** の検証は追加インフラなしで回せる
- 訓練履歴を見ると、L1 は **118 epoch** で `best_val_loss = 4.53e-8`、L2 は **54 epoch** で `1.99e-7`、L3 は **54 epoch** で `6.14e-6` に到達している。つまり現アーキテクチャは「再構成だけ」なら十分に強く、問題は容量不足より **表現にかかる誘導バイアス不足**と見る方が自然
- 現行 AE 学習は MSE のみ、weight decay も 0 なので、「1D に潰れた方が再構成しやすい」局所解を避ける仕組みがない。β-VAE や contrastive loss を検討する価値はここにある

### D. 基本領域制約を学習に入れる意義

- `models/layers.py` の `HalfPlaneLatent` が課す制約は **$\mathrm{Im}(z) > 0$ だけ**で、基本領域条件 $|\mathrm{Re}(z)| \le 1/2, |z| \ge 1$ はどこにも入っていない
- `eval/metrics.py` には `check_modular_invariance` があるが、これは **学習後の評価**であって、訓練 loss には一切反映されない
- Wide half-plane 実験 L3 のサンプリング領域では、再生成した $\tau$ の **78.7%** が最初から基本領域の外にあり、基本領域へ還元すると $\tau$ の移動量は **median = 1.0**, **mean = 1.06**, **95 percentile = 2.0** だった
- それでも L3 の modular invariance 指標は **0.958** まで下がっており、L1 の **1.019** よりわずかに良い。つまり AE は完全ではないが、**同値類をまとめるシグナル自体はすでに拾えている**。この方向を loss として明示化するのは筋がよい
- 逆に L2 は half-plane 制約を入れても modular invariance が **2.053** まで悪化しているので、`y > 0` という弱い幾何制約だけでは足りないことも既存 run が示している

### まとめ

今回の `runs/` と実装を踏まえると、改善案の優先順位は次のように読むのが妥当である。

1. **A. 信号正規化**
2. **B. `j` 比較の見直し**
3. **D. modular invariant loss など、群作用を直接入れる改善**
4. **C. latent 次元や β-VAE の拡張**

理由は、A/B は現行コードの評価・入力表現の弱点を直接突く「低コスト高確度」の修正であり、D/C はその次に来る **表現学習側の本格改修**だからである。
