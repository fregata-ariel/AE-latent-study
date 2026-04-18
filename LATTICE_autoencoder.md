# Lattice / Modular Autoencoder — 実装ドキュメント

## 概要

複素上半平面 $\mathbb{H}$ 上のパラメータ $\tau$ から生成される格子テータ信号 $\theta(t;\tau)$ を入力とし、$\mathrm{SL}_2(\mathbb{Z})$ 作用による商空間 $\mathbb{H}/\mathrm{SL}_2(\mathbb{Z})$（基本領域 $\mathcal{F}$）を潜在として学習することを目指す実験基盤。j-不変量 $j(\tau)$ との相関、modular invariance loss、quotient chart 品質指標で評価する。

Torus AE の $S^1/T^2$ 対称性を、モジュラー群という非可換・離散無限群に拡張した姉妹系である。理論的背景は [`ae-latent-study-discuss1.md`](./ae-latent-study-discuss1.md)、実験の進捗と Step1→Step2 の経緯は [`walkthrough-lattice-step1.md`](./walkthrough-lattice-step1.md)・[`walkthrough-lattice-step2.md`](./walkthrough-lattice-step2.md)・[`ae-latent-study-discuss2.md`](./ae-latent-study-discuss2.md) を参照。

---

## モジュール構成

### Models

| パス | クラス/関数 | 役割 |
|---|---|---|
| `models/modular_ae.py` | `ModularAutoEncoder` | `HalfPlaneLatent` を挟み、$\mathrm{Im}(\tau)>0$ を保証する上半平面潜在 AE |
| `models/ae.py` | `AutoEncoder` | invariance loss 付き標準潜在のベースライン（`lattice_standard*` で使用） |
| `models/vae.py` | `VAE` | β-VAE 変種（`lattice_vae_norm_beta001`） |
| `models/layers.py` | `HalfPlaneLatent` | `Dense(2)` の出力に対し $\mathrm{Im}$ 成分を `softplus` で正値化 |
| `models/__init__.py` | `create_model(config)` | `config.model.latent_type`（`standard` / `torus` / `halfplane` / `vae`）で dispatch |

### Data

| パス | 関数 | 役割 |
|---|---|---|
| `data/generation.py` | `sample_fundamental_domain` | 基本領域 $\mathcal{F}=\{|\tau|\ge1, |\mathrm{Re}\tau|\le1/2, \mathrm{Im}\tau>0\}$ からの一様サンプル |
| `data/generation.py` | `sample_upper_halfplane` | $\mathbb{H}$ 全体からの一様サンプル |
| `data/generation.py` | `reduce_to_fundamental_domain` | $\tau\in\mathbb{H}$ を $\mathcal{F}$ 内の代表元に引き戻す |
| `data/generation.py` | `apply_modular_transform` | 指定行列（$T$, $S$, $ST$ など）を $\tau$ に作用 |
| `data/generation.py` | `make_cyclic_modular_partners` | invariance loss 用に $\mathrm{SL}_2(\mathbb{Z})$ パートナーを生成 |
| `data/generation.py` | `generate_lattice_theta` | $\theta(t;\tau)=\sum_{(m,n)\ne(0,0),\;|m|,|n|\le K}\exp(-\pi|m+n\tau|^2 t)$ |
| `data/generation.py` | `compute_j_invariant` | Eisenstein 級数 $E_4, E_6$ の q-展開から $j(\tau)$ を計算 |
| `data/generation.py` | `normalize_lattice_signals` | `max` / `identity` 正規化 |
| `data/generation.py` | `generate_dataset` / `_generate_lattice_dataset` | config から格子データ一式を生成 |
| `data/dataset.py` | `Dataset`, `create_splits`, `batched_iterator` | Torus 系と共通 |

### Training

| パス | 関数/クラス | 役割 |
|---|---|---|
| `train/train_state.py` | `create_train_state`, `VAETrainState` | Torus 系と共通 |
| `train/train_step.py` | `_make_train_step_lattice_invariant(weight)` | $\mathrm{MSE}(x,\hat{x})+\mathrm{MSE}(x_{\text{partner}},\hat{x}_{\text{partner}})+\mathrm{weight}\cdot\lVert z - z_{\text{partner}}\rVert^2$ |
| `train/train_step.py` | `_make_eval_step_lattice_invariant(weight)` | 上記の評価モード版 |
| `train/trainer.py` | `_should_use_lattice_invariance`, `_make_lattice_partner_signals`, `_iter_eval_batches`, `train_and_evaluate` | modular invariance を有効にするか config で判定し、partner batch を組み立てる |

### Evaluation

| パス | 関数 | 役割 |
|---|---|---|
| `eval/metrics.py` | `check_modular_invariance` | SL₂(ℤ) 等価ペアでの潜在距離分布（mean / max） |
| `eval/metrics.py` | `compute_j_correlation` | 潜在各次元と $\mathrm{Re}(j), \mathrm{Im}(j), \log_{10}|j|$ の Pearson/Spearman/MI |
| `eval/metrics.py` | `compute_quotient_chart_quality` | `trust`（local KNN Jaccard）、`overlap`、`eff_dim`（participation ratio）、PCA EVR |
| `eval/metrics.py` | `compute_reconstruction_error`, `encode_dataset` | Torus 系と共通 |
| `eval/visualization.py` | `plot_lattice_latent_scatter` | $\mathcal{F}$ オーバーレイ付き潜在散布図 |
| `eval/visualization.py` | `plot_j_invariant_correlation` | $j$ との相関プロット |
| `eval/visualization.py` | `plot_quotient_chart_quality` | quotient chart 指標の可視化 |
| `eval/visualization.py` | `_draw_fundamental_domain` | $\mathcal{F}$ の境界描画ヘルパ |
| `eval/analysis.py` | `_run_lattice_evaluation`, `run_full_evaluation` | lattice 専用評価の統括 |

---

## Config 一覧

すべて `configs/lattice_default.py::get_config()` を取得して override する。

| Config | 潜在タイプ | 正規化 | invariance loss | 備考 |
|---|---|---|---|---|
| `configs/lattice_default.py` | `standard` | `identity` | off | ベース |
| `configs/lattice_standard.py` | `standard` | `identity` | off | Step1 素の標準 AE |
| `configs/lattice_standard_norm.py` | `standard` | `max` | off | Step2 Phase1 ベスト |
| `configs/lattice_standard_norm_inv.py` | `standard` | `max` | on | **Step2 採用構成** |
| `configs/lattice_standard_wide.py` / `_wide_norm.py` / `_wide_norm_inv.py` | `standard` (wide MLP) | 各種 | off/off/on | 容量スイープ |
| `configs/lattice_standard_norm_latent4.py` / `_latent8.py` | `standard` | `max` | off | 潜在次元スイープ |
| `configs/lattice_halfplane.py` / `_halfplane_norm.py` | `halfplane` | 各種 | off | $\mathbb{H}$ 直接パラメタライズ |
| `configs/lattice_vae_norm_beta001.py` | `vae` | `max` | off | $\beta=0.01$ の VAE |

主要 config キー（`lattice_default.py` 参照）: `data.{dataset_kind='lattice', tau_sampler, theta_K, signal_length, n_train/n_val/n_test, normalization}`、`data.modular_invariance.{enabled, partners, weight}`、`model.{latent_type, latent_dim, encoder_hidden, decoder_hidden}`。

---

## 実行方法

### 単一実験

```bash
python main.py --config=configs/lattice_standard_norm_inv.py --workdir=runs/lattice_snorm_inv
```

### バッチ実験

```bash
# Step1 相当（raw lattice AE の観察）
python run_lattice_experiments.py

# Step2（正規化 / invariance loss / 容量スイープ、レポート自動生成）
python run_lattice_step2_experiments.py
```

`run_lattice_step2_experiments.py` は各 phase を走らせた後 `walkthrough-lattice-step2.md` を自動生成する（`output_path` 引数で変更可）。

### 出力

`runs/<workdir>/` 以下に以下が保存される:

- `training_curves.png`
- `reconstructions.png`
- `lattice_latent_scatter.png`（基本領域オーバーレイ付き）
- `j_invariant_correlation.png`
- `quotient_chart_quality.png`
- `metrics.json`（`modular_mean_distance`、`spearman_vs_log10_abs_j`、`trust` / `overlap` / `eff_dim` など）
- orbax checkpoint

---

## テスト

| パス | 内容 |
|---|---|
| `tests/test_data.py` | 基本領域サンプラ・$\theta$ 関数 shape・$j$ 不変量の有限性と shape |
| `tests/test_integration.py` | `_tiny_lattice_config` による smoke test（全 lattice variants） |
| `test_lattice_smoke.py` | エンドツーエンド（サンプリング → データ生成 → モデル → 学習 → 指標） |

実行: `python -m pytest tests/ test_lattice_smoke.py`

---

## 採用設定と結果

Step2 での最良実験は `lattice_standard_norm_inv`（`configs/lattice_standard_norm_inv.py`）:

- Modular mean distance: **0.0004**
- Spearman 相関（潜在 vs $\log_{10}|j|$）: **0.9879**
- Reconstruction MSE: 1.12e-07
- Quotient chart: trust 0.8528 / overlap 0.0570 / eff_dim 1.0438

詳細な phase 比較と ranking は [`walkthrough-lattice-step2.md`](./walkthrough-lattice-step2.md) を参照。Step1 での1次元潜在崩壊とその原因分析は [`walkthrough-lattice-step1.md`](./walkthrough-lattice-step1.md)。

---

## 設計メモ

- **基本領域の微分可能引き戻し**: Step2 時点では未実装（open issue）。現状は invariance loss でサンプルペアを束ねる形で対応。
- **`HalfPlaneLatent`**: softplus による $\mathrm{Im}(\tau)>0$ の保証のみで、$\mathrm{SL}_2(\mathbb{Z})$ 対称性自体はネットワーク構造としては持たない。対称性は invariance loss で学習させる。
- **正規化の比較**: `max` 正規化を通した実験同士でのみ MSE を直接比較可能。Step1 の raw MSE とは比較しない。
- **j-invariant のダイナミックレンジ**: $|j(\tau)|$ は極めて広い範囲を取るため、相関は $\log_{10}|j|$ に対して評価する。

---

## 関連ドキュメント

- [`TORUS_autoencoder.md`](./TORUS_autoencoder.md) — 姉妹: $T^1/T^2$ AE
- [`walkthrough-lattice-step1.md`](./walkthrough-lattice-step1.md) — Step1 の詳細分析（1D 潜在崩壊の原因究明）
- [`walkthrough-lattice-step2.md`](./walkthrough-lattice-step2.md) — Step2 実験レポート（自動生成）
- [`ae-latent-study-discuss1.md`](./ae-latent-study-discuss1.md) — 群対称性・商空間・保型形式との関係
- [`ae-latent-study-discuss2.md`](./ae-latent-study-discuss2.md) — Step1/Step2 の理論的位置付け
