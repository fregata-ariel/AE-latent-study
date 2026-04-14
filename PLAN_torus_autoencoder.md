# Plan: Torus Autoencoder Toy Experiment (JAX/Flax Linen)

## Context

周期角パラメータ (theta) から生成される1次元時系列に対して、真の潜在空間が円周 (S^1) またはトーラス (T^2) となるToy問題を構築する。シンプルなAutoencoderで学習し、latentが元の周期・トーラス構造をどの程度反映するかを可視化・評価する実験基盤を作る。

リポジトリは空の状態（コミットなし、ファイルなし）。すべてゼロから構築する。

---

## Design Decisions

| 項目 | 選定 | 理由 |
|---|---|---|
| Config管理 | `ml_collections.ConfigDict` + `absl-py` | JAX/Flaxエコシステム標準。CLI override対応 |
| データパイプライン | 純NumPy配列 + JAXベースの手動バッチ | メモリに収まるToyデータ。tf.data不要 |
| Checkpoint | `orbax-checkpoint` (`CheckpointManager`) | Flax公式推奨。自動cleanup対応 |
| Torus latent | `nn.Dense -> cos/sin` 射影 (カスタムModule) | 正規化ベースより勾配が安定 |
| Training loop | `@jax.jit` 付き純関数 `train_step` + `TrainState` | JAXイディオマティック |
| 可視化 | matplotlib のみ | Toyには十分。PDF/PNG保存 |
| Optimizer | `optax.adam` (weight_decay時は`adamw`) | 標準的 |

---

## File Structure

```
AE-latent-study/
├── pyproject.toml
├── requirements.txt
├── main.py                    # CLIエントリポイント
├── run_experiments.py         # 全実験バッチ実行
├── configs/
│   ├── __init__.py
│   ├── default.py             # 基本Config (get_config())
│   ├── t1_standard.py         # T^1 + 通常AE
│   ├── t1_torus.py            # T^1 + torus-aware AE
│   ├── t2_standard.py         # T^2 + 通常AE
│   ├── t2_torus.py            # T^2 + torus-aware AE
│   └── t1_vae.py              # T^1 + VAE (optional)
├── data/
│   ├── __init__.py
│   ├── generation.py          # 信号生成 (T^1, T^2)
│   └── dataset.py             # Dataset, split, batching
├── models/
│   ├── __init__.py            # create_model() factory
│   ├── layers.py              # MLP, TorusLatent
│   ├── encoder.py             # MLP Encoder
│   ├── decoder.py             # MLP Decoder
│   ├── ae.py                  # Standard AutoEncoder
│   ├── torus_ae.py            # Torus-aware AutoEncoder
│   └── vae.py                 # VAE (optional)
├── train/
│   ├── __init__.py
│   ├── train_state.py         # TrainState生成, VAETrainState
│   ├── train_step.py          # JIT-compiled train/eval step
│   ├── trainer.py             # 学習ループ本体
│   └── checkpointing.py       # orbax save/restore
├── eval/
│   ├── __init__.py
│   ├── metrics.py             # 再構成誤差, 周期性チェック
│   ├── visualization.py       # 全プロット関数
│   └── analysis.py            # PCA/UMAP, 補間, run_full_evaluation
└── tests/
    ├── test_models.py         # Shape tests, unit circle test
    ├── test_data.py           # 生成・split tests
    └── test_integration.py    # Smoke test (tiny config)
```

---

## Dependencies (`requirements.txt`)

```
jax>=0.4.20
flax>=0.8.0
optax>=0.1.7
orbax-checkpoint>=0.5.0
ml-collections
numpy
matplotlib
scikit-learn
absl-py
umap-learn  # optional
```

---

## Config Schema (`configs/default.py`)

```python
config.seed = 42

# Data
config.data.torus_dim = 1          # 1: T^1, 2: T^2
config.data.signal_length = 100
config.data.n_train = 2000
config.data.n_val = 500
config.data.n_test = 500
config.data.omega = 2.0            # T^1用
config.data.omega1 = 2.0           # T^2用
config.data.omega2 = 3.0
config.data.a1 = 1.0
config.data.a2 = 0.5
config.data.noise_std = 0.0
config.data.dt = 0.1

# Model
config.model.latent_type = 'standard'  # 'standard' | 'torus' | 'vae'
config.model.latent_dim = 2            # R^d次元 or n_angles
config.model.encoder_hidden = [256, 128, 64]
config.model.decoder_hidden = [64, 128, 256]
config.model.activation = 'relu'
config.model.vae_beta = 1.0

# Training
config.train.batch_size = 128
config.train.num_epochs = 200
config.train.learning_rate = 1e-3
config.train.weight_decay = 0.0
config.train.lr_schedule = 'constant'  # 'constant' | 'cosine'
config.train.patience = 20
config.train.log_every = 10

# Checkpoint
config.checkpoint.dir = 'checkpoints/'
config.checkpoint.max_to_keep = 3

# Eval
config.eval.output_dir = 'results/'
config.eval.n_interpolation = 50
config.eval.use_umap = False
```

実験別configは `default.get_config()` を取得して必要箇所をoverrideする。

---

## Key Components Detail

### 1. Data Generation (`data/generation.py`)

```python
def generate_t1_signals(thetas, omega, signal_length, dt, noise_std=0.0, key=None):
    """x(t; theta) = sin(omega * t + theta) + noise  -> (N, signal_length)"""
    # t = jnp.arange(signal_length) * dt
    # signals = jnp.sin(omega * t[None, :] + thetas[:, None])

def generate_t2_signals(theta1, theta2, omega1, omega2, a1, a2, ...):
    """x(t) = a1*sin(omega1*t + theta1) + a2*sin(omega2*t + theta2)"""

def generate_dataset(config, key) -> dict:
    """Dispatcher. Returns {'signals': ..., 'thetas': ...}"""
```

- 角度は `jax.random.uniform(key, (n,), minval=0, maxval=2*pi)` で一様サンプリング
- ブロードキャスト計算で高速生成

### 2. Dataset / Batching (`data/dataset.py`)

```python
@dataclasses.dataclass
class Dataset:
    signals: jnp.ndarray    # (N, T)
    thetas: jnp.ndarray     # (N,) for T^1 or (N, 2) for T^2

def create_splits(config, key) -> tuple[Dataset, Dataset, Dataset]
def batched_iterator(dataset, batch_size, key, shuffle=True) -> Iterator
```

- `jax.random.permutation` でシャッフル
- 最後の不完全バッチはdrop（JIT再コンパイル回避）
- thetasはバッチに含めるが学習損失では使わない（評価用のみ）

### 3. TorusLatent Layer (`models/layers.py`)

```python
class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activation: str = 'relu'
    @nn.compact
    def __call__(self, x): ...

class TorusLatent(nn.Module):
    n_angles: int  # 1 for T^1, 2 for T^2
    @nn.compact
    def __call__(self, x):
        raw = nn.Dense(self.n_angles)(x)  # (batch, n_angles)
        return jnp.concatenate([jnp.cos(raw), jnp.sin(raw)], axis=-1)
        # Output: (batch, 2*n_angles) = [cos_1,...,cos_n, sin_1,...,sin_n]

    @staticmethod
    def recover_angles(latent, n_angles):
        return jnp.arctan2(latent[..., n_angles:], latent[..., :n_angles])
```

- Dense -> cos/sin は正規化ベースより勾配安定
- Decoder入力は `2*n_angles` 次元 (T^1: 2D, T^2: 4D)

### 4. Models

**Encoder** (`models/encoder.py`): Signal -> hidden (latent射影は含めない。再利用のため)

**Decoder** (`models/decoder.py`): Latent -> reconstructed signal

**AutoEncoder** (`models/ae.py`):
```python
class AutoEncoder(nn.Module):
    def __call__(self, x) -> tuple[x_hat, z]:
        h = self.encoder(x)
        z = self.latent_proj(h)    # nn.Dense(latent_dim)
        x_hat = self.decoder(z)
        return x_hat, z
```

**TorusAutoEncoder** (`models/torus_ae.py`):
```python
class TorusAutoEncoder(nn.Module):
    def __call__(self, x) -> tuple[x_hat, z]:
        h = self.encoder(x)
        z = self.torus_latent(h)   # TorusLatent(n_angles)
        x_hat = self.decoder(z)
        return x_hat, z
```

**VAE** (`models/vae.py`): reparameterization trick, KL divergence, beta制御

**Factory** (`models/__init__.py`): `create_model(config)` が `latent_type` に応じてdispatch

### 5. Training (`train/`)

**train_state.py**:
```python
def create_train_state(config, model, key) -> TrainState:
    # dummy input (1, signal_length) で init
    # optax.adam or adamw
    # cosine schedule: optax.cosine_decay_schedule
```

**train_step.py**:
```python
@jax.jit
def train_step_ae(state, batch) -> (state, metrics):
    # loss = MSE(x, x_hat)
    # jax.value_and_grad + apply_gradients

@jax.jit
def train_step_vae(state, batch, beta) -> (state, metrics):
    # loss = MSE + beta * KL

@jax.jit
def eval_step(state, batch) -> (metrics, z)
```

**trainer.py** (`train_and_evaluate`):
1. `create_splits` でデータ生成
2. `create_model` + `create_train_state`
3. Epoch loop: batch iterate -> train_step -> val eval -> checkpoint (best) -> early stopping
4. Best checkpoint restore -> return (state, history)

**checkpointing.py**: `ocp.CheckpointManager` + `StandardSave`/`StandardRestore`

### 6. Evaluation (`eval/`)

**metrics.py**:
- `compute_reconstruction_error(state, dataset)` -> MSE, MAE
- `check_periodicity(state, config)` -> theta=0 vs theta=2pi の latent距離
- `compute_latent_statistics(state, dataset)`

**visualization.py**:
- `plot_training_curves(history)` - 学習曲線
- `plot_reconstructions(state, dataset)` - 入力 vs 再構成
- `plot_latent_scatter(state, dataset, config)` - theta色付きlatent散布図
- `plot_latent_interpolation(state, config)` - latent補間 -> decoded信号変化
- `plot_periodicity_check(state, config)` - 周期接続確認

**analysis.py**:
- `pca_latent(state, dataset)` - PCA (高次元latent用)
- `umap_latent(state, dataset)` - UMAP (T^2用, optional)
- `run_full_evaluation(state, config, train_ds, test_ds, history, workdir)` - 全評価統括

### 7. Entry Points

**main.py**:
```bash
python main.py --config=configs/t1_standard.py --workdir=runs/t1_standard
python main.py --config=configs/t1_torus.py --config.model.latent_dim=1
```

**run_experiments.py**: 全4実験パターンを順次実行 + 比較プロット生成

---

## Data Flow

```
[Config] -> [data/generation.py] -> signals + ground-truth thetas
                                         |
                                  [data/dataset.py]
                                         |
                              +----------+----------+
                              |          |          |
                         train_ds     val_ds     test_ds
                              |          |          |
                              v          |          |
                   [models/ create_model]|          |
                              |          |          |
                              v          v          |
                   [train/trainer.py]               |
                    train_and_evaluate               |
                    +-- train_step (per batch)       |
                    +-- eval_step (on val_ds)        |
                    +-- checkpoint best              |
                    +-- return (state, history)      |
                              |                     |
                              v                     v
                   [eval/analysis.py]
                    run_full_evaluation
                    +-- reconstruction metrics
                    +-- latent scatter (theta-colored)
                    +-- periodicity check
                    +-- interpolation
                    +-- PCA/UMAP (T^2)
                    +-- training curves
                              |
                              v
                       results/ directory
                       (figures PNG/PDF + metrics JSON)
```

---

## Implementation Order

### Phase 1: Foundation
1. `pyproject.toml` + `requirements.txt`
2. `configs/default.py` - 全configスキーマ定義
3. `data/generation.py` - T^1, T^2 信号生成
4. `data/dataset.py` - Dataset, splits, batching

### Phase 2: Core Models
5. `models/layers.py` - MLP, TorusLatent
6. `models/encoder.py` + `models/decoder.py`
7. `models/ae.py` - Standard AE
8. `models/torus_ae.py` - Torus-aware AE
9. `models/__init__.py` - create_model factory

### Phase 3: Training
10. `train/train_state.py`
11. `train/train_step.py` - train_step_ae, eval_step
12. `train/checkpointing.py`
13. `train/trainer.py` - train_and_evaluate

### Phase 4: Evaluation
14. `eval/metrics.py`
15. `eval/visualization.py`
16. `eval/analysis.py` - run_full_evaluation

### Phase 5: Entry Points + Experiment Configs
17. `main.py`
18. `configs/t1_standard.py`, `t1_torus.py`, `t2_standard.py`, `t2_torus.py`
19. `run_experiments.py`

### Phase 6: Optional
20. `models/vae.py` + `train_step_vae` + `VAETrainState` + `configs/t1_vae.py`

---

## Experiment Patterns

| # | Data | Latent Type | Latent Dim | 目的 |
|---|------|-------------|------------|------|
| 1 | T^1 | standard (R^d) | 2 | 円周構造が無制約latentに現れるか |
| 2 | T^1 | torus (cos,sin) | 1 angle -> 2D | 明示的S^1制約との比較 |
| 3 | T^2 | standard (R^d) | 2, 3, 4 | 次元不足/十分でトーラス再構成がどう変わるか |
| 4 | T^2 | torus (cos,sin) | 2 angles -> 4D | 明示的T^2制約の効果 |

---

## Testing Strategy

### Shape Tests (`tests/test_models.py`)
- 各モデルの出力shape確認 (T^1, T^2両方)
- `TorusLatent` の出力が各(cos,sin)ペアで norm=1 であることを確認
- バッチイテレータが期待サイズを返すことを確認

### Data Tests (`tests/test_data.py`)
- 信号shape == (n_samples, signal_length)
- theta=0 と theta=2*pi で信号が一致（float精度内）
- splits のサイズが正しく重複しないこと

### Integration Smoke Test (`tests/test_integration.py`)
- tiny config (n_train=20, epochs=2, hidden=[8,4]) で全パイプライン実行
- lossが減少することを確認
- checkpoint round-trip (save -> restore -> params一致)

---

## Known Challenges

1. **Standard AEは円構造を自発的に学ばない可能性がある** - これは実験の意図通り。torus-aware版との比較で構造の差が観察できる
2. **VAE posterior collapse** - beta warmupまたは小さいbetaで対処。configで制御可能
3. **JIT再コンパイル** - 不完全バッチをdropで回避
4. **TorusLatentの勾配** - cos/sinのゼロ勾配点は実用上問題なし（Dense出力が特定点に集中しない）

---

## Verification

1. **Unit tests**: `python -m pytest tests/` で shape, data, integration テスト通過
2. **Experiment 1 実行**: `python main.py --config=configs/t1_standard.py --workdir=runs/t1_std` -> 学習曲線がlog_every間隔で出力、再構成誤差が十分低い
3. **Experiment 2 実行**: `python main.py --config=configs/t1_torus.py --workdir=runs/t1_torus` -> latent散布図で円周構造が可視化される
4. **可視化確認**: `results/` に以下が生成される
   - `training_curves.png`
   - `reconstructions.png`
   - `latent_scatter.png` (theta色付き)
   - `interpolation.png`
   - `periodicity_check.png`
5. **受け入れ条件**: T^1でlatentに円周的対応が観察でき、T^2で標準AEの限界が確認できること
