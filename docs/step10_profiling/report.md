# Step 10 ボトルネック診断レポート

## 計測条件

- 対象: `run_lattice_step10_experiments.py` の先頭 1 実験 `lattice_factorized_vae_fd_b030_q100_g030_d030_cl010_r030_ld010` を 2 epoch だけ流す。
- ハーネス: [profile_step10.py](../../profile_step10.py)。本体コードには一切手を入れず、モンキーパッチで計時を注入。
- ハード: NVIDIA GeForce RTX 2080 Ti (11 GB)、driver 580.126.09、CUDA 13.0。
- フレームワーク: JAX 0.9.2 / Flax / ml_collections (CUDA device)。
- データセット: lattice / train=5000 / val=1000 / test=1000 / signal_length=100 / batch_size=128。
- モデル: `factorized_vae` / 総パラメータ **135,320** / encoder(256,128,64) / decoder(64,128,256) / latent_dim=6。
- 実行: 2 回
  - `raw/` … `PROFILE_TRACE=1 JAX_LOG_COMPILES=1` で nvidia-smi dmon 併走 (マクロ/per-step/per-metric JSON + JIT ログ + jax profiler トレース)
  - `raw/pyspy_train.svg` … `uvx py-spy record --subprocesses -- uv run python profile_step10.py` で CPU サンプリング (200 Hz、11753 サンプル、エラー 0)

## 結論(先に要点)

1. **GPU 利用率は平均 3.06%、サンプルの 96% が SM<10%。事前仮説「I/O とバッチが悪い」は概ね当たり。** より正確には**I/O ではなく CPU 前処理**(NumPy でのシータ関数再計算)で GPU が常時待たされている。
2. 1 train step の壁時間は **≈53.7 ms** で、そのうち **GPU が走っているのは 2.05 ms (3.8%)、残り 51.6 ms が CPU のバッチ前処理**。
3. 1 実験の壁時間 49.86 s のうち、**訓練部 31.0 s (62%) + `run_full_evaluation` 18.85 s (38%)**。epoch を増やせば訓練の比重は線形に伸び、評価は一定なので長時間ランでは最大支配要因は**訓練ステップの CPU 前処理**になる。
4. 評価側の内訳では `compute_factorized_consistency` が **13.95 s / 74%** を占め、N=5000 の O(N²) NumPy ループ + sklearn `trustworthiness` 相当が原因。
5. JIT コンパイルは **309 回 / 16.57 s**(大半は最初の step で償却、1 step 目が 4227 ms)。300 epoch ラン相当では 0.1% 未満に落ちるため**本質的な問題ではない**。
6. 事前仮説 (A)(B)(C)(D)(E) の寄与見立て:
   - **(A) CPU 前処理 → GPU アイドル: 主犯、訓練側壁時間の ~96%**
   - **(B) モデル小ささによる device underutilization: 有意**。仮に CPU 前処理をゼロ化しても、2 ms/step のまま batch=128 では 2080 Ti は張り付かない
   - **(C) 不要損失の常時評価: 軽微**。warm step 2.05 ms の内訳までは HLO トレース(`raw/jax_trace/`)を Perfetto で見て要確認だが、現状では (A) の裏に隠れている
   - **(D) `run_full_evaluation` が訓練より支配的: 短縮ランでのみ成立(全体の 38%)。本番 300 epoch では比重は小さい**
   - **(E) JIT 再コンパイル / 同期 / チェックポイント: 16.6 s の一回性コスト、本番では無視可**

## 1 step の内訳 (warm steady state)

| 区間 | mean(ms) | median(ms) | first call(ms) | count |
|---|---|---|---|---|
| CPU prep `_make_lattice_partner_batch` | 47.98 | 46.41 | 198.67 | 94 |
| CPU prep `_reduce_tau_batch_to_fd_coords` | 1.69 | 1.65 | 2.06 | 94 |
| CPU prep `_make_j_rank_targets` | 1.46 | 1.36 | 8.94 | 94 |
| CPU prep `_make_teacher_quotient_batch` | 0.54 | 0.27 | 22.93 | 94 |
| **CPU prep 合計 / step** | **≈51.67** | ≈49.7 | ≈232 | 94 |
| JIT step `factorized_lattice_vae` (block_until_ready 同期後) | 2.05 | 1.90 | 4227.36 | 78 |
| 1 epoch の validation パス | 3283.94 | — | 3784.03 | 2 |

- `_make_lattice_partner_batch` が CPU prep の **96%** を占める。内部で [data/generation.py](../../data/generation.py) の `generate_lattice_theta`(NumPy でのシータ関数の再計算)を毎バッチ呼ぶ。
- warm step の GPU 実時間は **2.05 ms**。RTX 2080 Ti の実効ピーク FP32 ≈ 13 TFLOPs に対し、モデル 135K パラメータ × batch 128 は forward/backward 10 MFLOPs 前後 → 0.1 ms もあれば回る計算。実測 2 ms は pairwise L2、argsort、eigh、softmax、logsumexp などの O(n²) op が上乗せされた結果と思われる(Perfetto で要裏取り)。
- train step 1 回の壁時間: **51.67 ms CPU + 2.05 ms GPU ≈ 53.7 ms/step**。GPU busy ratio ≈ 3.8%。

## 評価の内訳 (`run_full_evaluation`)

| 関数 | total(s) | count | mean(ms) |
|---|---|---|---|
| `eval.lattice_evaluation_block` (ラティス評価ブロック全体) | **18.853** | 1 | 18853 |
| ├ `compute_factorized_consistency` | **13.952** | 1 | 13952 |
| ├ `compute_reconstruction_error` | 1.576 | 1 | 1576 |
| ├ `check_modular_invariance` | 1.209 | 1 | 1209 |
| ├ `compute_quotient_chart_quality` | 0.166 | 1 | 166 |
| ├ `compute_j_correlation` | 0.046 | 1 | 46 |
| `encode_dataset` (他の metric 内部で 7 回呼ばれる) | 3.052 | 7 | 436 |

- `compute_factorized_consistency` 単体で評価総時間の **74%**。要因は N=5000 サンプルでの O(N²) NumPy pairwise distance / softmax / logsumexp 系(Explore 事前調査どおり)。
- `encode_dataset` は上記メトリクス内部で重複呼び出しされているため、`eval.lattice_evaluation_block` の合計 18.85 s に含まれている(二重計上ではない)。
- `compute_quotient_chart_quality` (sklearn trustworthiness 含む) は 166 ms と予想より軽い。これは `config.eval.chart_max_samples=2000` のサブサンプリングが効いているため。

## GPU テレメトリ (`nvidia-smi dmon -s pucvmet -d 1`)

| 指標 | 値 |
|---|---|
| サンプル数 | 113 |
| SM% 平均 | **3.06** |
| SM% 最大 | 37 |
| SM% 分布 | [0,10) 108 / [10,20) 3 / [20,30) 1 / [30,40) 1 |
| 消費電力 平均 | 27.6 W |
| 消費電力 最大 | 67 W (TDP 260 W に対し 25%) |

→ GPU はほぼ遊休。SM 高利用率は瞬間的に出現するだけ。

## JIT コンパイル (`JAX_LOG_COMPILES=1`)

| 指標 | 値 |
|---|---|
| `XLA compilation` イベント数 | **309** |
| 総コンパイル時間 | **16.566 s** |
| 主要関数 (上位) | `jit(dot_general)` ×58, `jit(add)` ×46, `jit(broadcast_in_dim)` ×28, `jit(relu)` ×21, `jit(concatenate)` ×15, `jit(dynamic_slice)` ×15, `jit(eval_step_factorized_lattice_vae)` ×4 |

- step 1 の壁時間 4227 ms はほぼ全て JIT 初回コンパイルで、warm 後は 2.05 ms に収束。
- `eval_step_factorized_lattice_vae` が 4 回コンパイルされている点は**本番の複数実験直列実行では再コンパイル分が積み重なる可能性**があるため、Perfetto トレース上で `compile` フレームの出方を今後確認するのが良い(現状の短縮 1 実験ランでは問題は顕在化せず)。

## py-spy CPU ホットスポット (`raw/pyspy_train.svg`、200 Hz、11753 サンプル)

上位関数(self-time ではなく cumulative):

| samples | % | 関数(ファイル:行) |
|---|---|---|
| 982 | 8.36 | `_make_lattice_partner_batch` (train/trainer.py) |
| 949 | 8.07 | **`generate_lattice_theta` (data/generation.py:361)** |
| 980 | 8.34 | `cache_miss` (jax/_src/pjit.py) ← JIT 発火 |
| 906 | 7.71 | `compile` (jax/_src/interpreters/pxla.py:2516) |
| 906 | 7.71 | `_cached_compilation` |
| 905 | 7.70 | `backend_compile_and_load` |
| 903 | 7.68 | `compute_factorized_consistency` (eval/metrics.py:1018) |
| 854 | 7.27 | `train_and_evaluate` (trainer.py:336 = データ生成) |
| 786 | 6.69 | `timed_step` (profile_step10.py) ← harness wrapper |
| 773 | 6.58 | Flax `create_train_state` 初期化 |
| 609 | 5.18 | `compute_factorized_consistency` (metrics.py:979) |
| 544 | 4.63 | `compute_factorized_consistency` (metrics.py:928) |
| 427 | 3.63 | `compute_factorized_consistency` (metrics.py:998) |

- `generate_lattice_theta` が訓練側 CPU 時間の最頂点。
- `compute_factorized_consistency` は複数行に分散して ~20% (cumulative) 。
- `compile` / `_cached_compilation` 系は 2 epoch 短縮ランだから比率が大きい。本番ではこの割合は下がる。

## Perfetto トレース

- 出力先: `docs/step10_profiling/raw/jax_trace/plugins/profile/<timestamp>/scipod.trace.json.gz`
- 読み方: https://ui.perfetto.dev にドラッグ&ドロップ(必要なら `gunzip` 済みのファイルでも開ける)。
- 観察ポイント (このレポートでは未実施、必要なら別タスクで):
  - step ごとの GPU busy window 幅と gap(= host 待ち時間)
  - HLO 内で 2.05 ms のうち pairwise l2 / argsort / eigh / logsumexp に何 ms 使っているか(仮説 (C) の裏取り)
  - `jit_compile` フレームが step 2 以降にも現れるか

## 仮説別の実測評価

| 仮説 | 実測による評価 | 寄与 |
|---|---|---|
| (A) バッチ前処理の CPU 待ち → GPU アイドル | **確認**。CPU prep 51.67 ms vs JIT step 2.05 ms。GPU SM 平均 3% | **主犯 (訓練側の 96%)** |
| (B) モデル小さすぎて device underutilization | 部分的に確認。warm step 2 ms は 2080 Ti のピーク FP32 に対し 0.1% レベルの演算量 | 中 (A を取り除いたあとの次の蓋) |
| (C) 不要損失の常時評価 / O(n²) op 過剰 | 未確定。2.05 ms という数値自体が実小モデルより数倍遅い可能性はあるが、Perfetto 未読解 | 未確定 |
| (D) `run_full_evaluation` が訓練より支配的 | 短縮ランでは 38% を占めるが、本番 300 epoch 相当では ~1% 以下 | 短時間ラン限定 |
| (E) JIT 再コンパイル / 同期 / チェックポイント | 16.6 s、1 回性。本番では 300 epoch に分散 → <0.2% | 無視可 |

## 次のアクション候補(修正は別 plan で)

「なぜ遅いか」は確定したので、**次にどこを直すべきか**の順序は以下を推奨:

1. **`_make_lattice_partner_batch` の JAX 化 / 非同期化 (見込みゲイン: 最大)**
   - `generate_lattice_theta` を JAX に書き換え `jax.jit` & `jax.vmap` すれば、GPU 上で O(K²) op がバッチ並列化される。47 ms → 数 ms 級への短縮が期待できる。
   - 代替案として、全エポックの partner batch を起動時に事前生成してインメモリに貯める手もある(`n_train × 転送数 × signal_length × 4B`)。
2. **バッチ前処理をメイン step とオーバーラップさせる (見込みゲイン: 中)**
   - Python スレッド 1 本 + キューで、次バッチの prep を GPU step と並行に走らせるだけで、warm step 53.7 ms が max(51.67, 2.05) = 51.67 ms に縮む(= GPU は相変わらず遊休だが、step 数/epoch は増やしやすくなる)。
3. **`compute_factorized_consistency` の最適化 or epoch ごと eval のサンプリング削減 (見込みゲイン: 短時間ランでは大)**
   - N=5000 を `config.eval.chart_max_samples` と同様に 2000 にサブサンプリングするだけで大幅短縮。
   - または metric 計算を NumPy → JAX に移植して JIT 化。
4. **weight=0 の損失ブランチを静的に切り落とす (見込みゲイン: 小〜中、要 Perfetto 検証)**
   - step 2.05 ms 内の内訳を Perfetto で確認した上で、`_make_train_step_factorized_lattice_vae` 側で weight==0 の枝を `if`(Python 時)で除去すれば JIT 済みグラフから消える。
5. **batch_size を大きく (128 → 512/1024) して GPU 側の 1 step 充填率を上げる (見込みゲイン: 中、ただし (1) の後で)**
   - CPU prep を先に解決しないと、大バッチ化 = 単にCPU 待ち時間が伸びるだけで無意味。

## 成果物一覧

- `docs/step10_profiling/report.md`(このファイル)
- `docs/step10_profiling/raw/macro_timings.json`
- `docs/step10_profiling/raw/train_step_timings.json`
- `docs/step10_profiling/raw/eval_step_timings.json`
- `docs/step10_profiling/raw/nvidia_smi_dmon.log`
- `docs/step10_profiling/raw/jit_compiles.log`(4449 行 / `grep "^Compiling " | wc` でコンパイル数を確認可)
- `docs/step10_profiling/raw/pyspy_train.svg`(Flamegraph、ブラウザで開くとインタラクティブ)
- `docs/step10_profiling/raw/jax_trace/`(Perfetto 入力)
- `docs/step10_profiling/raw/stdout.log`(ハーネス stdout)
- `docs/step10_profiling/raw/short_report.md`(orchestrator が通常出力する walkthrough 形式の短縮レポート)
- `docs/step10_profiling/raw_pyspy/`(py-spy 親モードラン用の JSON セット、念のため)
