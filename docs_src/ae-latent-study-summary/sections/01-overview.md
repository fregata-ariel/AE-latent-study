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
