# Torus Autoencoder — 実装ドキュメント

## 概要

周期角パラメータ $\theta$ から生成される1次元時系列 $x(t;\theta)$ を入力とし、真の潜在空間が円周 $S^1$（$T^1$）または2次元トーラス $T^2$ となる Toy 問題の実験基盤。標準の $\mathbb{R}^d$ 潜在・$(\cos,\sin)$ 制約付き潜在・VAE の3系統を同一パイプラインで比較できる。

本ドキュメントは as-built な仕様書であり、実装の起点となった要件・哲学は [`toy-latent-study-torus-requirements.md`](./toy-latent-study-torus-requirements.md)、背景となる理論は [`ae-latent-study-discuss1.md`](./ae-latent-study-discuss1.md)、姉妹となる格子/モジュラー拡張は [`LATTICE_autoencoder.md`](./LATTICE_autoencoder.md) を参照。

---

## モジュール構成

### Models

| パス | クラス/関数 | 役割 |
|---|---|---|
| `models/ae.py` | `AutoEncoder` | 標準 $\mathbb{R}^d$ 潜在の AE |
| `models/torus_ae.py` | `TorusAutoEncoder` | `TorusLatent` を挟んで $(\cos,\sin)$ 制約付き潜在にする AE |
| `models/vae.py` | `VAE` | 対角 Gaussian posterior + KL 正則化 |
| `models/encoder.py` | `Encoder` | MLP エンコーダ本体（潜在射影は含まない） |
| `models/decoder.py` | `Decoder` | MLP デコーダ |
| `models/layers.py` | `MLP` / `TorusLatent` / `HalfPlaneLatent` / `get_activation` | 共通レイヤ |
| `models/__init__.py` | `create_model(config)` | `config.model.latent_type` で dispatch するファクトリ |

`TorusLatent` は `Dense(n_angles)` の出力 $\phi$ に対し $(\cos\phi, \sin\phi)$ を連結する。出力次元は $2\cdot n_{\text{angles}}$（$T^1$: 2D、$T^2$: 4D）。`TorusLatent.recover_angles` で逆変換（`arctan2`）。

### Data

| パス | 関数 | 役割 |
|---|---|---|
| `data/generation.py` | `generate_t1_signals` | $x(t;\theta)=\sin(\omega t+\theta)+\varepsilon$ |
| `data/generation.py` | `generate_t2_signals` | $x(t;\theta_1,\theta_2)=a_1\sin(\omega_1 t+\theta_1)+a_2\sin(\omega_2 t+\theta_2)+\varepsilon$ |
| `data/generation.py` | `generate_dataset(config, key)` | config から信号・真値 $\theta$ を一括生成 |
| `data/dataset.py` | `Dataset`, `create_splits`, `batched_iterator` | train/val/test 分割とバッチ取り出し |

真値 $\theta$ は学習損失では使わず、評価・可視化用のみ。

### Training

| パス | 関数/クラス | 役割 |
|---|---|---|
| `train/train_state.py` | `create_train_state`, `VAETrainState` | Flax `TrainState` の初期化、VAE 用拡張 |
| `train/train_step.py` | `train_step_ae`, `eval_step_ae` | MSE 再構成損失 |
| `train/train_step.py` | `_make_train_step_vae(beta)`, `eval_step_vae` | $\mathrm{MSE}+\beta\cdot\mathrm{KL}$、eval は posterior mean を使用 |
| `train/trainer.py` | `train_and_evaluate` | epoch ループ・early stopping・best checkpoint |
| `train/checkpointing.py` | `CheckpointManager`, `save/restore_checkpoint` | orbax ベースの保存・復元 |

不完全バッチは drop して JIT 再コンパイルを回避。

### Evaluation

| パス | 関数 | 役割 |
|---|---|---|
| `eval/metrics.py` | `compute_reconstruction_error` | テストセット MSE/MAE |
| `eval/metrics.py` | `check_periodicity` | $\theta=0$ と $\theta=2\pi$ での潜在距離 |
| `eval/metrics.py` | `encode_dataset` | 全データの潜在コードを収集 |
| `eval/visualization.py` | `plot_training_curves` / `plot_reconstructions` / `plot_latent_scatter` / `plot_latent_interpolation` / `plot_periodicity_check` | 標準プロット一式 |
| `eval/analysis.py` | `pca_latent` / `umap_latent` | 高次元潜在の次元削減 |
| `eval/analysis.py` | `run_full_evaluation` | 上記を統括する entry 関数 |

---

## Config 一覧

すべて `configs/default.py::get_config()` を取得して override する方式。

| Config | データ | 潜在タイプ | 用途 |
|---|---|---|---|
| `configs/default.py` | $T^1$ (デフォルト) | `standard` | ベース設定 |
| `configs/t1_standard.py` | $T^1$ | `standard` (2D) | 無制約潜在が円周構造を学ぶか検証 |
| `configs/t1_torus.py` | $T^1$ | `torus` (1 angle → 2D) | 明示的 $S^1$ 制約 |
| `configs/t1_vae.py` | $T^1$ | `vae` | KL 正則化下での潜在形状観察 |
| `configs/t2_standard.py` | $T^2$ | `standard` | トーラス再構成の限界 |
| `configs/t2_torus.py` | $T^2$ | `torus` (2 angles → 4D) | 明示的 $T^2$ 制約の効果 |

主要 config キー（`default.py` 参照）: `data.{torus_dim, signal_length, omega*, a*, noise_std, dt, n_train/n_val/n_test}`、`model.{latent_type, latent_dim, encoder_hidden, decoder_hidden, activation, vae_beta}`、`train.{batch_size, num_epochs, learning_rate, weight_decay, lr_schedule, patience, log_every}`、`checkpoint.{dir, max_to_keep}`、`eval.{output_dir, n_interpolation, use_umap}`。

---

## 実行方法

### 単一実験

```bash
python main.py --config=configs/t1_torus.py --workdir=runs/t1_torus
python main.py --config=configs/t2_standard.py --config.model.latent_dim=3 --workdir=runs/t2_std_d3
```

### 全実験バッチ

```bash
python run_experiments.py
```

`run_experiments.py` は `configs/t{1,2}_{standard,torus}.py`（および任意で VAE）を順次実行し、比較プロットを生成する。

### 出力

`runs/<workdir>/` 以下に以下が保存される:

- `training_curves.png`
- `reconstructions.png`
- `latent_scatter.png`（真値 $\theta$ で色付け）
- `interpolation.png`
- `periodicity_check.png`
- `metrics.json`
- orbax checkpoint（`checkpoints/` サブディレクトリ）

---

## テスト

| パス | 内容 |
|---|---|
| `tests/test_models.py` | `test_ae_shapes` / `test_torus_ae_shapes` / `test_vae_shapes`、`test_torus_latent_unit_circle`（$(\cos,\sin)$ ペアの norm = 1）、`test_torus_recover_angles` |
| `tests/test_data.py` | $T^1/T^2$ 信号 shape、`test_t1_periodicity`（$\theta=0$ と $2\pi$ で信号一致）、splits の整合性 |
| `tests/test_integration.py` | tiny config でのパイプライン smoke test、loss 減少、checkpoint round-trip |

実行: `python -m pytest tests/`

---

## 設計メモ

- **$(\cos,\sin)$ 射影 vs 正規化**: `Dense → cos/sin` のほうが正規化ベースの射影より勾配が安定し、実装も簡潔。ゼロ勾配点は実用上問題にならない（Dense 出力が特定点に集中しにくい）。
- **VAE posterior collapse**: `config.model.vae_beta` を小さく（例 $10^{-2}$）するか warmup で対処。
- **JIT 再コンパイル回避**: `batched_iterator` で不完全バッチを drop。
- **標準 AE の円周構造**: $T^1$ では自発的に円周状配置を獲得しない場合がある。これは比較実験の意図通り。

---

## 関連ドキュメント

- [`toy-latent-study-torus-requirements.md`](./toy-latent-study-torus-requirements.md) — 原初の要件定義
- [`LATTICE_autoencoder.md`](./LATTICE_autoencoder.md) — 姉妹: 格子/モジュラー AE
- [`ae-latent-study-discuss1.md`](./ae-latent-study-discuss1.md) — 群対称性と潜在空間の理論的背景
- [`ae-latent-study-discuss2.md`](./ae-latent-study-discuss2.md) — 理論 → 実装への橋渡し
