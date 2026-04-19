# Documentation Rulebook

この文書は、このリポジトリで Markdown 文書を追加・更新・整理するときの運用ルールをまとめたものです。
研究内容そのものではなく、**どう文書を構築し、どこに置き、どう更新するか**の基準を扱います。

## 基本方針

- reader-facing の正規入口は `docs/` に置く
- authoring source は `docs_src/` に置く
- 研究の current understanding は `docs/current/` に集約する
- 古い判断・実験メモは `docs/archive/` に保存する
- 前史や補助資料は `docs/background/` に置く

## ディレクトリの意味

### `docs/current/`

今後の研究判断や実装の基準として読むべき canonical documents。

例:

- `ae-latent-study-summary.md`
- `ae-latent-study-roadmap.md`

### `docs/background/`

現在の主線ではないが、背景理解に必要な文書。

例:

- torus 前史
- 初期 requirements

### `docs/archive/`

各時点の判断、途中の roadmap snapshot、実験 walkthrough、会話ベースの整理文書を残す場所。

この階層の文書は原則として「その時点の履歴」を保存するものであり、後から current interpretation に合わせて全面改稿しない。

### `docs_src/`

`mdsplit` 前提の authoring source。

各文書は基本的に次の形で持つ。

- `hierarchy.json`
- `preamble.md`（必要なら）
- `sections/*.md`

## Canonical Output と Source の関係

### 原則

- `docs/current/*.md` は **compose 済みの成果物**
- `docs_src/.../sections/*.md` は **編集対象**

### 編集ルール

- `docs/current/*.md` を直接編集しない
- 更新時は `docs_src/` 側を編集し、`mdsplit compose` で再生成する
- compose 後に差分を確認し、必要なら section 側に戻って修正する

## mdtools の使い方

この repo では `mdtools/` 配下のツールを前提にする。

### `mdsplit`

分割 Markdown の verify / compose に使う。

代表コマンド:

```bash
PYTHONPATH=mdtools mdtools/.venv/bin/python -m mdsplit verify docs_src/ae-latent-study-summary/hierarchy.json
PYTHONPATH=mdtools mdtools/.venv/bin/python -m mdsplit compose docs_src/ae-latent-study-summary/hierarchy.json -o docs/current/ae-latent-study-summary.md
```

roadmap も同様:

```bash
PYTHONPATH=mdtools mdtools/.venv/bin/python -m mdsplit verify docs_src/ae-latent-study-roadmap/hierarchy.json
PYTHONPATH=mdtools mdtools/.venv/bin/python -m mdsplit compose docs_src/ae-latent-study-roadmap/hierarchy.json -o docs/current/ae-latent-study-roadmap.md
```

### `langfilter`

日英混在 source を後から切り出せるようにするための補助ツール。

- v1 では日本語主の単一成果物を優先する
- abstract、用語定義、短いまとめなど、将来英語版が欲しくなりそうな箇所だけ `::: {lang=ja}` / `::: {lang=en}` を使う
- 全文を無理に日英併記しない

### `mdhtml_rewrite`

今回は通常の Markdown 整理には使わない。

- HTML 化前提の大規模 rewrite
- 画像参照や HTML 断片の再構築

などが必要になったときだけ別タスクとして使う。

## 文書作成時の注意

### 役割を先に決める

新しい文書を作る前に、まず次のどれかを決める。

- Current
- Background
- Archive
- 運用文書

役割が曖昧なまま新規 Markdown を増やさない。

### Current 文書の条件

current に置く文書は次を満たす。

- 今後の判断基準として参照される
- 履歴ではなく、現時点の canonical reading を与える
- source of truth の優先順が明確
- 次に何を読むかが分かる

### Archive 文書の扱い

- 当時の wording や判断は基本的に保持する
- 必要なら `docs/current/` から「この文書は履歴」として参照する
- archive の本文を current understanding に合わせて書き換えない

### Root-level Markdown

root に新しい研究文書を増やさない。

例外:

- 互換性のための短い stub
- 明示的に root 配置が必要な project-level 文書

既存 root-level Markdown を移設した場合は、短い `Moved` stub を残して新しい場所へリンクする。

## 数値と記述のルール

研究総括や roadmap で数値を使うときは、次の優先順を守る。

1. `runs/*.json`
2. 最新の walkthrough
3. discuss 文書

walkthrough と JSON が食い違う場合は JSON を優先する。

## リンク方針

- `docs/README.md` から current / background / archive の入口を辿れるようにする
- current 文書から archive を参照するときは、読者目線で分かりやすい相対リンクを張る
- source (`docs_src/`) 側では、最終 compose 先を意識してリンクを書く

## 新しい current 文書を追加するときの手順

1. `docs_src/<doc-name>/` を作る
2. `hierarchy.json` を作る
3. `sections/*.md` と必要なら `preamble.md` を作る
4. `mdsplit verify` を通す
5. `mdsplit compose` で `docs/current/*.md` を生成する
6. `docs/README.md` にリンクを追加する
7. 必要なら旧文書を archive に移し、root には stub を置く

## 更新時の最終チェック

- `docs/README.md` から目的の文書に到達できる
- `mdsplit verify` が通る
- compose 後の `docs/current/*.md` が生成されている
- current / background / archive の区分が崩れていない
- root-level に不要な新規 Markdown が増えていない

## このルールブック自体の扱い

- この文書は `docs/` 配下の運用文書として直接管理する
- 研究の current summary や roadmap とは別物
- 運用フローが変わったら随時更新する
